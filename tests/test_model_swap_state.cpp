// Audit C-2 and C-3 at the server call sites, not only in their helpers (C-14).
// C-2: a swap or suspend while a handler holds a lease answers 503 and frees nothing.
// C-3: the channel marker ids of one load never survive into the next.

#include "handlers.h"
#include "handlers_internal.h"

#include "model/tokenizer.h"
#include "runtime/config.h"

#include <gtest/gtest.h>
#include <httplib.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;

// Never dereferenced while the gate holds: model_loaded() only compares it with nullptr.
alignas(64) char g_fake_ctx_storage[64];
ImpContext fake_ctx() { return reinterpret_cast<ImpContext>(g_fake_ctx_storage); }

// A loaded model "current.gguf" and a models dir holding a second one, "next.gguf".
struct LoadedServer {
    std::unique_ptr<ServerState> state = std::make_unique<ServerState>();
    fs::path dir;

    explicit LoadedServer(const char* tag) {
        dir = fs::temp_directory_path() /
              ("imp_swap_state_" + std::string(tag) + "_" + std::to_string(::getpid()));
        fs::create_directories(dir);
        std::ofstream(dir / "next.gguf") << "";
        state->models_dir = dir.string();
        state->model_name = "current.gguf";
        state->ctx = fake_ctx();
        state->runtime_config.server.model_swap = true;
        state->runtime_config.server.model_swap_drain_ms = 50;
    }
    ~LoadedServer() {
        state->ctx = nullptr;  // fake: the destructor must not see it
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

imp::Tokenizer vocab_tokenizer(const std::vector<std::string>& tokens) {
    imp::Tokenizer tok;
    EXPECT_TRUE(tok.load_vocab(tokens, std::vector<float>(tokens.size(), 0.0f), 0, 1));
    return tok;
}

}  // namespace

TEST(ModelSwapState, SwapWhileAHandlerHoldsALeaseAnswers503AndFreesNothing) {
    LoadedServer srv("swap");
    ServerState& st = *srv.state;
    auto lease = acquire_model_lease(st.lifetime);  // handler A: tokenizer snapshotted, not submitted
    ASSERT_NE(lease, nullptr);
    const uint64_t epoch = st.lifetime.epoch();

    httplib::Response res;
    bool swapped = true;
    {
        std::lock_guard<std::timed_mutex> lock(st.mtx);
        swapped = ensure_model_loaded(st, "next.gguf", res);
    }

    EXPECT_FALSE(swapped);
    EXPECT_EQ(res.status, 503) << res.body;
    EXPECT_EQ(res.get_header_value("Retry-After"), "1");
    EXPECT_NE(res.body.find("not started"), std::string::npos) << res.body;
    EXPECT_EQ(st.ctx, fake_ctx()) << "the swap tore the model down under a leased handler";
    EXPECT_EQ(st.model_name, "current.gguf");
    EXPECT_EQ(st.lifetime.epoch(), epoch) << "a load epoch started while a lease was alive";
    EXPECT_TRUE(lease->current());
    EXPECT_FALSE(st.lifetime.exclusive_pending());
    EXPECT_FALSE(st.swapping.load());
    // The gate is usable again: the next handler gets a lease.
    lease.reset();
    EXPECT_NE(acquire_model_lease(st.lifetime), nullptr);
}

TEST(ModelSwapState, SuspendWhileAHandlerHoldsALeaseAnswers503AndFreesNothing) {
    LoadedServer srv("suspend");
    ServerState& st = *srv.state;
    auto lease = acquire_model_lease(st.lifetime);
    ASSERT_NE(lease, nullptr);
    const uint64_t epoch = st.lifetime.epoch();

    json body;
    int status = 0;
    {
        std::lock_guard<std::timed_mutex> lock(st.mtx);
        status = suspend_locked(st, body);
    }

    EXPECT_EQ(status, 503) << body.dump();
    EXPECT_NE(body.dump().find("suspend not started"), std::string::npos) << body.dump();
    EXPECT_EQ(st.ctx, fake_ctx());
    EXPECT_FALSE(st.suspended.load());
    EXPECT_EQ(st.lifetime.epoch(), epoch);
    EXPECT_FALSE(st.lifetime.exclusive_pending());
}

TEST(ModelSwapState, ChannelMarkersOfTheLastLoadDoNotSurviveASwap) {
    auto owned = std::make_unique<ServerState>();
    ServerState& st = *owned;
    const imp::Tokenizer gemma = vocab_tokenizer({"<pad>", "<eos>", "\n", "a", "<|channel>", "<channel|>"});
    detect_channel_markers(st, gemma);
    EXPECT_EQ(st.channel_open_id, 4);
    EXPECT_EQ(st.channel_close_id, 5);
    EXPECT_EQ(st.channel_newline_id, 2);

    // Swap to a model without the pair: ids 4 and 5 are ordinary tokens there.
    const imp::Tokenizer plain = vocab_tokenizer({"<pad>", "<eos>", "\n", "a", "the", "cat"});
    detect_channel_markers(st, plain);
    EXPECT_EQ(st.channel_open_id, -1);
    EXPECT_EQ(st.channel_close_id, -1);
    EXPECT_EQ(st.channel_newline_id, -1);
}

TEST(ModelSwapState, OneChannelTokenAloneIsNoChannelModel) {
    auto owned = std::make_unique<ServerState>();
    ServerState& st = *owned;
    detect_channel_markers(st, vocab_tokenizer({"<pad>", "<eos>", "\n", "<|channel>"}));
    EXPECT_EQ(st.channel_open_id, -1);
    EXPECT_EQ(st.channel_close_id, -1);
    EXPECT_EQ(st.channel_newline_id, -1);
}
