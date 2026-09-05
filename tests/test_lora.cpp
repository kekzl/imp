// =============================================================================
// LoRA hot-swap E2E (issue #522) — C-API level, real model.
//
// Strategy: synthetic PEFT adapters crafted in-test (full control, no
// network):
//   1. zero-B adapter      → delta is exactly 0 → greedy output must be
//      BIT-IDENTICAL to base. This catches every wiring bug (wrong x, wrong
//      buffer, wrong layer) because any misapplied delta breaks identity.
//   2. nonzero-B adapter   → greedy output must DIFFER (the delta reaches
//      the logits) while the engine stays healthy (no abort, non-empty).
//   3. swap back to base   → bit-identical to the original base output
//      (graph re-capture path; a stale capture would reproduce the adapter).
//
// Default model: Llama-3.2-3B (q_proj/v_proj adapters, r=8) — the classic
// PEFT target set. Skips when the model file is absent.
// =============================================================================

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "imp/imp.h"
#include "api/imp_internal.h"
#include "batching_engine.h"
#include "model/model.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"
#include "test_models.h"
#include <chrono>
#include <memory>

namespace {

const char* model_path() {
    if (const char* p = getenv(imp_test::kEnvModelLlama))
        return p;
    return "/models/Llama-3.2-3B-Instruct-Q8_0.gguf";
}

// Minimal safetensors writer: header JSON + F32 tensor blobs.
struct StWriter {
    std::string header_entries;
    std::vector<float> data;

    void add(const std::string& name, int64_t d0, int64_t d1, bool ramp) {
        size_t start = data.size() * sizeof(float);
        for (int64_t i = 0; i < d0 * d1; i++) {
            float v = ramp ? 0.02f * static_cast<float>(static_cast<int>((i * 7 + 3) % 11) - 5) : 0.0f;
            data.push_back(v);
        }
        size_t end = data.size() * sizeof(float);
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%lld,%lld],\"data_offsets\":[%zu,%zu]}",
                 header_entries.empty() ? "" : ",", name.c_str(), (long long)d0, (long long)d1, start,
                 end);
        header_entries += buf;
    }

    void write(const std::string& path) {
        std::string hdr = "{" + header_entries + "}";
        while (hdr.size() % 8 != 0)  // safetensors pads headers with spaces
            hdr += ' ';
        uint64_t hlen = hdr.size();
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&hlen), 8);
        f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * sizeof(float)));
    }
};

// Build a PEFT adapter dir targeting q_proj/v_proj of layers 0..2.
// zero_B=true → mathematically a no-op adapter.
std::string make_adapter(int d_model, int kv_rows, bool zero_B, const char* dirname) {
    const int r = 8;
    std::string dir = std::string(testing::TempDir()) + dirname;
    std::filesystem::create_directories(dir);

    std::ofstream cfg(dir + "/adapter_config.json");
    cfg << "{\"r\": 8, \"lora_alpha\": 16, \"use_rslora\": false, "
           "\"target_modules\": [\"q_proj\", \"v_proj\"]}";
    cfg.close();

    StWriter w;
    for (int layer = 0; layer < 3; layer++) {
        char key[160];
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.q_proj.lora_A.weight",
                 layer);
        w.add(key, r, d_model, /*ramp=*/true);
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.q_proj.lora_B.weight",
                 layer);
        w.add(key, d_model, r, /*ramp=*/!zero_B);
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.v_proj.lora_A.weight",
                 layer);
        w.add(key, r, d_model, /*ramp=*/true);
        snprintf(key, sizeof(key), "base_model.model.model.layers.%d.self_attn.v_proj.lora_B.weight",
                 layer);
        w.add(key, kv_rows, r, /*ramp=*/!zero_B);
    }
    w.write(dir + "/adapter_model.safetensors");
    return dir;
}

// Teacher-forced PPL is the project's bit-stable A/B instrument under
// deterministic mode (#542) — greedy TEXT comparison is invalid here because
// dense greedy logit ties flip across runs (docs/determinism.md), which a
// pre-LoRA control run confirmed on this exact model/prompt.
double ppl(ImpModel model, ImpContext ctx, const char* text) {
    std::vector<int32_t> toks(512);
    int n = 0;
    EXPECT_EQ(imp_tokenize(model, text, toks.data(), &n, 512), IMP_SUCCESS);
    double out = 0.0;
    EXPECT_EQ(imp_perplexity(ctx, toks.data(), n, &out), IMP_SUCCESS);
    return out;
}

// Greedy generation as a HEALTH probe only (non-empty, no abort) — never
// compared for equality.
std::string greedy(ImpContext ctx, const char* prompt) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 24;
    params.temperature = 0.0f;
    params.seed = 42;
    params.apply_chat_template = 0;
    char out[4096];
    size_t out_len = 0;
    if (imp_generate(ctx, prompt, &params, out, sizeof(out), &out_len) != IMP_SUCCESS)
        return "";
    return std::string(out, out_len);
}

TEST(LoraHotSwap, ZeroAdapterIdentity_EffectAdapterDiffers_SwapBack) {
    if (!std::filesystem::exists(model_path()))
        GTEST_SKIP() << "model not present: " << model_path();

    // The identity assertions need bit-stable greedy across runs — dense
    // greedy logit ties otherwise flip mid-sequence (documented determinism
    // boundary, docs/determinism.md). Same env-seed mechanism as
    // DetEvalE2ETest: must be set BEFORE model/engine creation.
    setenv("IMP_DETERMINISTIC", "1", 1);

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(model_path(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig config = imp_config_default();
    config.max_seq_len = 512;
    config.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &config, &ctx), IMP_SUCCESS);

    const int d_model = imp_model_d_model(model);
    const int kv_rows = 1024;  // Llama-3.2-3B: 8 KV heads × 128

    const char* corpus =
        "The three primary colors are red, blue, and yellow. Mixing two primary colors "
        "produces a secondary color: red and blue make purple, blue and yellow make green, "
        "and red and yellow make orange.";

    const double base = ppl(model, ctx, corpus);
    ASSERT_GT(base, 1.0);
    // Control: teacher-forced NLL is bit-stable in deterministic mode — the
    // identity assertions below ride on this.
    const double base2 = ppl(model, ctx, corpus);
    ASSERT_EQ(base, base2) << "baseline PPL not bit-stable — deterministic mode not active?";

    // --- 1. zero-B adapter: delta is exactly 0 → bit-identical PPL ---
    std::string zero_dir = make_adapter(d_model, kv_rows, /*zero_B=*/true, "imp_lora_zero");
    int32_t zero_id = 0;
    ASSERT_EQ(imp_lora_load(ctx, zero_dir.c_str(), &zero_id), IMP_SUCCESS);
    ASSERT_GE(zero_id, 1);
    ASSERT_EQ(imp_lora_set(ctx, zero_id), IMP_SUCCESS);
    const double with_zero = ppl(model, ctx, corpus);
    EXPECT_EQ(base, with_zero) << "zero-B adapter must be a mathematical no-op (wiring bug otherwise)";

    // --- 2. effect adapter: PPL must move materially, generation healthy ---
    std::string eff_dir = make_adapter(d_model, kv_rows, /*zero_B=*/false, "imp_lora_eff");
    int32_t eff_id = 0;
    ASSERT_EQ(imp_lora_load(ctx, eff_dir.c_str(), &eff_id), IMP_SUCCESS);
    ASSERT_EQ(imp_lora_set(ctx, eff_id), IMP_SUCCESS);
    const double with_eff = ppl(model, ctx, corpus);
    EXPECT_GT(std::abs(with_eff - base) / base, 0.01)
        << "a strong nonzero adapter must move teacher-forced PPL (base=" << base
        << ", eff=" << with_eff << ")";
    std::string gen = greedy(ctx, "The three primary colors are");
    EXPECT_GE(gen.size(), 1u) << "generation with active adapter must not abort";

    // --- 3. back to base: bit-identical to the original PPL ---
    ASSERT_EQ(imp_lora_set(ctx, 0), IMP_SUCCESS);
    const double back = ppl(model, ctx, corpus);
    EXPECT_EQ(base, back) << "deactivating must restore the base model exactly (stale graph capture?)";

    imp_context_free(ctx);
    imp_model_free(model);
}

// ---------------------------------------------------------------------------
// AUDIT_arch_2026 dispatch #6 (E-1, F1-6): the adapter is checked against the
// model at load, is a prefix-cache key, and is an admission barrier.
// ---------------------------------------------------------------------------

// q_proj/v_proj adapter with caller-chosen widths (a wrong one must be refused).
std::string make_adapter_dims(int k_in, int q_out, int kv_out, const char* dirname) {
    const int r = 8;
    std::string dir = std::string(testing::TempDir()) + dirname;
    std::filesystem::create_directories(dir);
    StWriter w;
    w.add("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", r, k_in, true);
    w.add("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", q_out, r, false);
    w.add("base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight", r, k_in, true);
    w.add("base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight", kv_out, r, false);
    w.write(dir + "/adapter_model.safetensors");
    return dir;
}

// A header that claims a 2^40-element tensor over a 64-byte payload: the
// loader must refuse before it sizes anything from the shape.
std::string write_lying_adapter(const char* dirname) {
    std::string dir = std::string(testing::TempDir()) + dirname;
    std::filesystem::create_directories(dir);
    std::string hdr =
        "{\"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1048576,1048576],\"data_offsets\":[0,64]},"
        "\"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight\":{\"dtype\":\"F32\","
        "\"shape\":[8,8],\"data_offsets\":[64,320]}}";
    while (hdr.size() % 8 != 0)
        hdr += ' ';
    const uint64_t hlen = hdr.size();
    std::ofstream f(dir + "/adapter_model.safetensors", std::ios::binary);
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
    std::vector<char> payload(320, 0);
    f.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    return dir;
}

TEST(LoraHotSwap, AdapterShapesAreHeldAgainstTheModel) {
    if (!std::filesystem::exists(model_path()))
        GTEST_SKIP() << "model not present: " << model_path();
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(model_path(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig config = imp_config_default();
    config.max_seq_len = 256;
    config.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &config, &ctx), IMP_SUCCESS);
    const int d_model = imp_model_d_model(model);
    const int kv_rows = 1024;
    int32_t id = 0;
    // K wider than d_model: the A kernel would read past the activation row.
    EXPECT_EQ(imp_lora_load(ctx, make_adapter_dims(d_model + 64, d_model, kv_rows, "imp_lora_bad_k").c_str(),
                            &id),
              IMP_ERROR_INVALID_MODEL);
    // N wider than the projection output: the B kernel would write past it.
    EXPECT_EQ(imp_lora_load(ctx, make_adapter_dims(d_model, d_model, kv_rows + 128, "imp_lora_bad_n").c_str(),
                            &id),
              IMP_ERROR_INVALID_MODEL);
    // Payload does not match the declared shape: refused as a bad file, not
    // as a failed 2 TiB host allocation (IMP_ERROR_INTERNAL before the fix).
    EXPECT_EQ(imp_lora_load(ctx, write_lying_adapter("imp_lora_lying").c_str(), &id),
              IMP_ERROR_INVALID_MODEL);
    // Control: the matching adapter still loads.
    EXPECT_EQ(imp_lora_load(ctx, make_adapter_dims(d_model, d_model, kv_rows, "imp_lora_good").c_str(), &id),
              IMP_SUCCESS);
    EXPECT_GE(id, 1);
    imp_context_free(ctx);
    imp_model_free(model);
}

struct Served {
    std::shared_ptr<ServerRequest> sr;
    std::string finish;
    int tokens = 0;
};

std::shared_ptr<ServerRequest> make_server_request(ImpContext ctx, const std::string& prompt, int max_tokens,
                                                   int lora_id) {
    auto sr = std::make_shared<ServerRequest>();
    sr->request = std::make_shared<imp::Request>();
    sr->request->input_tokens = ctx->engine->model()->tokenizer()->encode(prompt);
    sr->request->max_tokens = max_tokens;
    sr->request->ignore_eos = true;
    sr->request->lora_id = lora_id;
    return sr;
}

// Pops every event of one request; false when the deadline passes first.
bool drain(Served& r, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    TokenEvent ev{};
    for (;;) {
        if (r.sr->pop_token(ev, 500)) {
            if (ev.token_id >= 0)
                r.tokens++;
            if (ev.is_last) {
                r.finish = ev.finish_reason ? ev.finish_reason : "";
                return true;
            }
        }
        if (std::chrono::steady_clock::now() > deadline)
            return false;
    }
}

TEST(LoraHotSwap, AdapterIsAPrefixCacheKeyAndAnAdmissionBarrier) {
    if (!std::filesystem::exists(model_path()))
        GTEST_SKIP() << "model not present: " << model_path();
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(model_path(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig config = imp_config_default();
    config.max_seq_len = 1024;
    config.max_batch_size = 2;
    config.use_prefix_caching = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &config, &ctx), IMP_SUCCESS);
    const int d_model = imp_model_d_model(model);
    const int kv_rows = 1024;
    int32_t a = 0, b = 0;
    ASSERT_EQ(imp_lora_load(ctx, make_adapter(d_model, kv_rows, /*zero_B=*/true, "imp_lora_key_a").c_str(),
                            &a),
              IMP_SUCCESS);
    ASSERT_EQ(imp_lora_load(ctx, make_adapter(d_model, kv_rows, /*zero_B=*/true, "imp_lora_key_b").c_str(),
                            &b),
              IMP_SUCCESS);
    ASSERT_NE(a, b);
    imp::Engine* engine = ctx->engine.get();

    BatchingEngine be;
    be.start(ctx);
    std::string prompt;
    for (int i = 0; i < 4; i++)
        prompt +=
            "The three primary colors are red, blue, and yellow. Mixing two primary colors "
            "produces a secondary color: red and blue make purple, blue and yellow make green, "
            "and red and yellow make orange. ";

    // 1. adapter A, cold: nothing to reuse.
    Served r1{make_server_request(ctx, prompt, 4, a)};
    be.submit(r1.sr);
    ASSERT_TRUE(drain(r1, 180000)) << "first request did not finish";
    EXPECT_EQ(r1.sr->request->cached_tokens, 0);
    EXPECT_EQ(engine->active_lora(), a);
    // 2. adapter A, warm: the prefix is served from the cache.
    Served r2{make_server_request(ctx, prompt, 4, a)};
    be.submit(r2.sr);
    ASSERT_TRUE(drain(r2, 60000));
    EXPECT_GT(r2.sr->request->cached_tokens, 0) << "prefix cache did not engage; the key test below is void";
    // 3. adapter B, same tokens: A's KV must not be offered (E-1).
    Served r3{make_server_request(ctx, prompt, 4, b)};
    be.submit(r3.sr);
    ASSERT_TRUE(drain(r3, 60000));
    EXPECT_EQ(r3.sr->request->cached_tokens, 0) << "adapter B reused KV computed under adapter A";
    EXPECT_EQ(engine->active_lora(), b);
    // 4. B warm: B's own entries are reusable.
    Served r4{make_server_request(ctx, prompt, 4, b)};
    be.submit(r4.sr);
    ASSERT_TRUE(drain(r4, 60000));
    EXPECT_GT(r4.sr->request->cached_tokens, 0);

    // 5. Barrier: an A request and a B request submitted back to back are
    //    served one after the other, never in one batch (max_batch_size is 2,
    //    so without the barrier the worker steps both rows together and
    //    decode_batch_max reads 2), and the switch happens between them.
    EXPECT_EQ(be.decode_batch_max.load(), 1);
    Served r5{make_server_request(ctx, prompt, 48, a)};
    Served r6{make_server_request(ctx, prompt, 4, b)};
    be.submit(r5.sr);
    be.submit(r6.sr);
    ASSERT_TRUE(drain(r5, 120000));
    ASSERT_TRUE(drain(r6, 60000));
    EXPECT_EQ(r5.tokens, 48) << r5.finish;
    EXPECT_EQ(r6.tokens, 4) << r6.finish;
    EXPECT_EQ(be.decode_batch_max.load(), 1) << "adapter B was stepped in one batch with adapter A";
    EXPECT_EQ(engine->active_lora(), b);

    be.stop();
    imp_context_free(ctx);
    imp_model_free(model);
}

}  // namespace
