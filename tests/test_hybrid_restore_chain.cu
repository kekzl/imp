// Chained hybrid restores must not drift (recurrent prefix cache): a chat session restores
// turn N-1's snapshot and prefills only the new tail every turn. This measures 30 chained
// restores against a cold prefill of the same final prompt.
// Tokens are the wrong instrument (two runs differing in state agree until an argmax flips,
// then disagree completely), so the comparison is the STATE itself (per-layer recurrent slab,
// max-abs and relative L2). Assertions are the noise floor and a bound calibrated against a
// cold-prefill control, not a constant.
// Finding: a chained restore costs no more than a cold prefill split at the same boundaries,
// independent of state dtype; the recurrent state is CHAOTIC in prefill chunk shape (the
// absolute drift is large and does not grow with chunk count) but two cold runs of the same
// prompt are bit-identical, which is what makes the comparison readable. Full campaign:
// docs/audit/SETTLED.md (2026-09-09). Needs IMP_TEST_MODEL_GDN; runs from make test-e2e.

#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "api/imp_internal.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "imp/imp.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/ssm_state.h"
#include "model/model.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "runtime/snapshot_boundary.h"
#include "test_models.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace imp {
namespace {

// A turn must be long enough to cross a KV block and short enough to stay under the
// chunk-parallel scan's 128-row minimum (~90+-20 tokens/turn, matching a measured 27B
// session). Distinct per turn: a repeated tail would collapse block hashes and restore the
// wrong turn.
constexpr const char* kSentences[] = {
    "A paged block cache stores keys and values once and hands the same physical page to every "
    "sequence whose prefix matches it.",
    "Recurrent state is cumulative, so a prefix can only be skipped where the exact state at that "
    "boundary was written down beforehand.",
    "Chunked prefill re-reads every weight once per chunk, which is why the chunk size decides the "
    "weight traffic and not the arithmetic.",
    "A snapshot keyed by a chained block hash identifies a byte-exact token prefix and nothing "
    "weaker than that would be safe to restore.",
    "The delta rule scan keeps its state in registers for the whole launch and writes it to the "
    "pool exactly once, at the last committed row.",
    "Storage precision is a property of the boundary, not of the arithmetic: the scan computes in "
    "single precision either way.",
    "An engine that plans its capacity can print the ceiling at startup, and an engine that "
    "discovers it can only argue about what was free.",
    "Two runs that differ in the state agree on tokens until the difference crosses an argmax, and "
    "after that they share nothing.",
    "Admission decides how many sequences may be resident and the block pool decides how long each "
    "of them is allowed to become.",
    "A test whose input cannot reach the defect is worthless even when every line of it is "
    "correct, which is why the input is the design.",
    "Bandwidth separates a resident pool from a spilled one far more reliably than any allocator "
    "return code on this platform.",
    "The measurement contract asks for the number, the path it came from, and the arm it is being "
    "compared against.",
};
constexpr int kNumSentences = static_cast<int>(sizeof(kSentences) / sizeof(kSentences[0]));

std::string turn_text(int i) {
    std::string s = "Turn " + std::to_string(i + 1) + ". ";
    for (int k = 0; k < 4; ++k) {
        s += kSentences[(i * 5 + k * 3) % kNumSentences];
        s += ' ';
    }
    return s;
}

constexpr const char* kOpening =
    "The following is a technical conversation about inference engines, memory planning and the "
    "numerics of recurrent state. Each turn adds a short paragraph and the assistant continues the "
    "text. The history is replayed in full on every turn, which is what makes the prefix cache and "
    "the recurrent snapshot store carry the whole session instead of a single request. Nothing in "
    "the exchange depends on outside knowledge; the point is the state that the engine carries "
    "from one turn to the next and whether that state still matches a run that saw the same tokens "
    "in one piece. ";

// ── the drift instrument ─────────────────────────────────────────────────
struct LayerDrift {
    double conv_max = 0.0;  // max |a - b| over the conv window
    double conv_rel = 0.0;  // ||a - b||_2 / ||b||_2 over the conv window
    double h_max = 0.0;
    double h_rel = 0.0;
};

double elem(const void* base, size_t i, QType dtype) {
    if (dtype == QType::BF16)
        return static_cast<double>(__bfloat162float(static_cast<const __nv_bfloat16*>(base)[i]));
    if (dtype == QType::F16)
        return static_cast<double>(__half2float(static_cast<const __half*>(base)[i]));
    return static_cast<double>(static_cast<const float*>(base)[i]);
}

void diff_span(const void* a, const void* b, size_t n, QType dtype, double* out_max, double* out_rel) {
    double max_abs = 0.0, num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = elem(a, i, dtype), y = elem(b, i, dtype);
        const double d = std::fabs(x - y);
        if (d > max_abs)
            max_abs = d;
        num += d * d;
        den += y * y;
    }
    *out_max = max_abs;
    *out_rel = (den > 0.0) ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
}

// ── the fixture ──────────────────────────────────────────────────────────
struct ChainArm {
    bool state_bf16;
    bool chunkpar;
};

std::string arm_name(const ::testing::TestParamInfo<ChainArm>& info) {
    return std::string(info.param.state_bf16 ? "bf16" : "fp32") +
           (info.param.chunkpar ? "_chunkpar" : "_nochunkpar");
}

class HybridRestoreChainTest : public ::testing::TestWithParam<ChainArm> {
protected:
    void SetUp() override {
        const std::string path = imp_test::env_path(imp_test::kEnvModelGdn);
        if (path.empty())
            GTEST_SKIP() << "Set " << imp_test::kEnvModelGdn << " to run the hybrid restore chain";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path.c_str(), imp_test::kEnvModelGdn));
        path_ = path;
        const bool gguf = path_.size() >= 5 && path_.substr(path_.size() - 5) == ".gguf";
        ASSERT_EQ(imp_model_load(path_.c_str(), gguf ? IMP_FORMAT_GGUF : IMP_FORMAT_SAFETENSORS, &model_),
                  IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    std::string path_;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

constexpr int kTurns = 30;   // 30 chained restores; the 27B session broke at 37
constexpr int kGreedy = 32;  // continuation compared between the arms
constexpr int kMaxSeqLen = 4096;

}  // namespace

TEST_P(HybridRestoreChainTest, HybridRestoreChainStateStaysClose) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT: " << imp_test::kEnvModelGdn
                     << " points at a model with ssm_inner_size == 0, and this test covers the "
                        "recurrent snapshot chain. A dense run leaves it UNCOVERED.";

    const ChainArm arm = GetParam();
    RuntimeConfig rc;
    rc.gdn.state_bf16 = arm.state_bf16;
    rc.gdn.chunkpar_scan = arm.chunkpar;
    rc.runtime.deterministic = true;
    set_pending_runtime_config(rc);

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = kMaxSeqLen;
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = 1;
    cfg.use_prefix_caching = 1;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);

    Engine* engine = ctx_->engine.get();
    ASSERT_NE(engine, nullptr);
    SSMState* ssm = engine->ssm_state();
    ASSERT_NE(ssm, nullptr) << "hybrid checkpoint without a recurrent state pool";
    ASSERT_NE(engine->kv_cache(), nullptr);
    const int kv_bs = engine->kv_cache()->block_size();
    const int min_snap = engine->runtime_config().server.snapshot_min_prompt_tokens;

    // The engine RESOLVES the requested dtype (init_resolve_ssm_dtype_), so the
    // arm asserts against what it got, not against what it asked for.
    const QType h_dtype = ssm->h_dtype();
    const int n_layers = ssm->n_ssm_layers();
    ASSERT_GT(n_layers, 0);
    const size_t per_seq = ssm->per_seq_bytes();
    const size_t per_layer = per_seq / static_cast<size_t>(n_layers);
    const size_t h_bytes = ssm->h_bytes();
    ASSERT_GT(per_layer, h_bytes) << "the slab has no conv window";
    const size_t conv_bytes = per_layer - h_bytes;

    // ── tokens ───────────────────────────────────────────────────────────
    auto tokenize = [&](const std::string& text) {
        std::vector<int32_t> out(8192);
        int n = 0;
        EXPECT_EQ(imp_tokenize(model_, text.c_str(), out.data(), &n, static_cast<int>(out.size())),
                  IMP_SUCCESS);
        out.resize(n > 0 ? n : 0);
        return out;
    };
    std::vector<int32_t> prompt = tokenize(kOpening);
    ASSERT_GE(static_cast<int>(prompt.size()), kv_bs * 4) << "opening too short to hold a snapshot";
    std::vector<std::vector<int32_t>> turns(kTurns);
    for (int t = 0; t < kTurns; ++t) {
        turns[t] = tokenize(turn_text(t));
        ASSERT_GE(static_cast<int>(turns[t].size()), kv_bs)
            << "turn " << t << " is shorter than one KV block, the chain would not advance";
    }

    // finish_request registers block hashes, so every step must FINISH or the next step has no
    // cached prefix to restore from. State is copied out the moment the prompt is through (first
    // sampled token present); after that, decode steps advance it further.
    auto run = [&](const std::vector<int32_t>& tokens, int max_tokens, std::vector<uint8_t>* out_state,
                   int* out_cached) {
        auto req = std::make_shared<Request>();
        req->input_tokens = tokens;
        req->max_tokens = max_tokens;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = true;  // fixed length: the arms must be comparable
        req->status = RequestStatus::PENDING;
        engine->add_request(req);
        bool captured = false;
        for (int i = 0; i < 8192; ++i) {
            if (req->status == RequestStatus::FINISHED || req->status == RequestStatus::CANCELLED)
                break;
            (void)engine->step();
            if (!captured && out_state && !req->output_tokens.empty()) {
                const int slot = engine->recurrent_slot(req->id);
                if (slot >= 0) {
                    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());
                    IMP_CUDA_CHECK_LOG(
                        cudaMemcpy(out_state->data(), ssm->seq_base(slot), per_seq, cudaMemcpyDeviceToHost));
                    captured = true;
                }
            }
        }
        EXPECT_EQ(req->status, RequestStatus::FINISHED)
            << "request did not finish (status " << static_cast<int>(req->status) << ")";
        if (out_state)
            EXPECT_TRUE(captured) << "recurrent slab was never captured: the run proves nothing";
        if (out_cached)
            *out_cached = req->cached_tokens;
        return req->output_tokens;
    };

    // Drops every cached block and snapshot directly: imp_context_reset() only does this when a
    // C-API request is live, and this test drives the engine directly, so calling it would be a
    // no-op and the "cold" arm would silently restore the chain's own snapshot.
    auto go_cold = [&]() {
        while (engine->kv_manager()->evict_cached_block()) {}
        engine->clear_recurrent_snapshots();
    };

    // Per-layer drift of slab `a` against reference slab `b`.
    auto measure = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                       std::vector<LayerDrift>& rows) {
        rows.assign(n_layers, LayerDrift{});
        for (int l = 0; l < n_layers; ++l) {
            const void* ca = ssm->conv_state_in(const_cast<uint8_t*>(a.data()), l);
            const void* cb = ssm->conv_state_in(const_cast<uint8_t*>(b.data()), l);
            const void* ha = ssm->h_state_in(const_cast<uint8_t*>(a.data()), l);
            const void* hb = ssm->h_state_in(const_cast<uint8_t*>(b.data()), l);
            diff_span(ca, cb, conv_bytes / sizeof(float), QType::F32, &rows[l].conv_max, &rows[l].conv_rel);
            diff_span(ha, hb, h_bytes / dtype_size(h_dtype), h_dtype, &rows[l].h_max, &rows[l].h_rel);
        }
    };
    auto worst = [&](const std::vector<LayerDrift>& rows, double LayerDrift::*field) {
        int idx = 0;
        double best = 0.0;
        for (int l = 0; l < n_layers; ++l)
            if (rows[l].*field > best) {
                best = rows[l].*field;
                idx = l;
            }
        return std::pair<int, double>(idx, best);
    };

    // Every turn appends fixed text, never the model's own output, so a turn can generate its
    // comparison tokens without changing what the next turn prefills. Checkpoints keep the warm
    // state and its prompt; cold references run afterward, once the cache is dropped.
    struct Checkpoint {
        int turns = 0;
        std::vector<int32_t> prompt;
        std::vector<uint8_t> warm_state;
        std::vector<int32_t> warm_tok;
        int warm_cached = -1;
    };
    std::vector<Checkpoint> cps;
    int restores = 0;
    for (int t = 0; t < kTurns; ++t) {
        prompt.insert(prompt.end(), turns[t].begin(), turns[t].end());
        const bool is_cp = (t + 1 == 5) || (t + 1 == 10) || (t + 1 == 20) || (t + 1 == kTurns);
        int cached = -1;
        if (!is_cp) {
            (void)run(prompt, 1, nullptr, &cached);
        } else {
            Checkpoint cp;
            cp.turns = t + 1;
            cp.prompt = prompt;
            cp.warm_state.resize(per_seq);
            cp.warm_tok = run(prompt, kGreedy, &cp.warm_state, &cached);
            cp.warm_cached = cached;
            cps.push_back(std::move(cp));
        }
        if (cached > 0)
            ++restores;
        if (t == 0)
            EXPECT_EQ(cached, 0) << "the first turn cannot hit a cache it is populating";
    }
    ASSERT_FALSE(cps.empty());
    EXPECT_GE(restores, kTurns - 4)
        << "only " << restores << " of " << kTurns
        << " turns restored a snapshot: the chain never formed, so this measures nothing";
    for (const auto& cp : cps) {
        EXPECT_GT(cp.warm_cached, 0) << "checkpoint at turn " << cp.turns << " restored nothing";
        EXPECT_LE(cp.warm_cached, snapshot_boundary(static_cast<int>(cp.prompt.size()), kv_bs, min_snap))
            << "reuse ran past the snapshot boundary at turn " << cp.turns;
    }

    // ── the references ───────────────────────────────────────────────────
    const Checkpoint& last = cps.back();
    std::vector<std::vector<uint8_t>> cold_states(cps.size());
    std::vector<std::vector<int32_t>> cold_toks(cps.size());
    for (size_t i = 0; i < cps.size(); ++i) {
        go_cold();
        cold_states[i].resize(per_seq);
        int cached = -1;
        cold_toks[i] = run(cps[i].prompt, kGreedy, &cold_states[i], &cached);
        EXPECT_EQ(cached, 0) << "the cold reference at turn " << cps[i].turns << " is not cold";
    }

    // Control 1, the noise floor. The same prompt, cold, a second time. Under
    // runtime.deterministic this must be bit-identical; anything else and every
    // number below is noise and the comparison decides nothing.
    go_cold();
    std::vector<uint8_t> cold2_state(per_seq);
    const std::vector<int32_t> cold2_tok = run(last.prompt, kGreedy, &cold2_state, nullptr);

    // Control 2 (boundary count): cold again but chunked at the chain's turn length - same token
    // path as the cold reference, same number of state round trips as the chain. Separates "the
    // restore is wrong" from "one more state store per turn costs the precision".
    go_cold();
    const int saved_chunk = engine->mutable_runtime_config().runtime.prefill_chunk_size;
    engine->mutable_runtime_config().runtime.prefill_chunk_size =
        std::max(kv_bs, (static_cast<int>(turns[0].size()) / kv_bs) * kv_bs);
    std::vector<uint8_t> chunked_state(per_seq);
    const std::vector<int32_t> chunked_tok = run(last.prompt, kGreedy, &chunked_state, nullptr);
    engine->mutable_runtime_config().runtime.prefill_chunk_size = saved_chunk;

    // ── the table ────────────────────────────────────────────────────────
    std::vector<LayerDrift> rows;
    std::printf(
        "\n[drift] model=%s requested=%s chunkpar=%d resolved_h_dtype=%s turns=%d restores=%d "
        "final_prompt=%d tok final_cached=%d\n",
        path_.c_str(), arm.state_bf16 ? "bf16" : "fp32", arm.chunkpar ? 1 : 0, qtype_name(h_dtype), kTurns,
        restores, static_cast<int>(last.prompt.size()), last.warm_cached);
    std::printf("[drift] %-28s %6s %11s %11s %11s %11s %6s\n", "arm", "tokens", "conv_maxabs", "conv_relL2",
                "h_maxabs", "h_relL2", "greedy");
    auto row = [&](const char* name, int n_tok, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                   bool greedy_ok) {
        measure(a, b, rows);
        const auto cmx = worst(rows, &LayerDrift::conv_max);
        const auto crl = worst(rows, &LayerDrift::conv_rel);
        const auto hmx = worst(rows, &LayerDrift::h_max);
        const auto hrl = worst(rows, &LayerDrift::h_rel);
        std::printf("[drift] %-28s %6d %11.3e %11.3e %11.3e %11.3e %6d\n", name, n_tok, cmx.second,
                    crl.second, hmx.second, hrl.second, greedy_ok ? 1 : 0);
        return hrl.second;
    };
    const double noise = row("control cold-vs-cold", static_cast<int>(last.prompt.size()), cold2_state,
                             cold_states.back(), cold2_tok == cold_toks.back());
    const double chunked = row("control chunkedcold-vs-cold", static_cast<int>(last.prompt.size()),
                               chunked_state, cold_states.back(), chunked_tok == cold_toks.back());
    std::vector<double> chain_rel(cps.size());
    for (size_t i = 0; i < cps.size(); ++i) {
        const std::string name = "chain@turn" + std::to_string(cps[i].turns) + "-vs-cold";
        chain_rel[i] = row(name.c_str(), static_cast<int>(cps[i].prompt.size()), cps[i].warm_state,
                           cold_states[i], cps[i].warm_tok == cold_toks[i]);
    }
    // Per-layer detail for the deepest checkpoint: which layers carry the drift.
    measure(last.warm_state, cold_states.back(), rows);
    std::printf("[drift] per-layer, chain@turn%d vs cold\n", last.turns);
    std::printf("[drift] %-5s %12s %12s %12s %12s\n", "layer", "conv_maxabs", "conv_relL2", "h_maxabs",
                "h_relL2");
    for (int l = 0; l < n_layers; ++l)
        std::printf("[drift] %-5d %12.3e %12.3e %12.3e %12.3e\n", l, rows[l].conv_max, rows[l].conv_rel,
                    rows[l].h_max, rows[l].h_rel);
    std::fflush(stdout);

    // The instrument: two cold runs of the same prompt under runtime.deterministic must be the
    // same run, or the drift numbers above measure only the engine's own jitter.
    EXPECT_EQ(cold2_tok, cold_toks.back())
        << "two cold runs of the same prompt disagreed: every number in this table is noise";
    EXPECT_LT(noise, 1e-9) << "cold-vs-cold state drift " << noise
                           << " - the reference is not reproducible, so nothing is being measured";

    // The property: a chained restore is a prefill split at snapshot boundaries, so it must cost
    // no more than a cold prefill split at the same places. Bound is the control plus its own
    // margin, not a constant - the ABSOLUTE drift is a property of the checkpoint (the recurrent
    // state is chaotic in chunk shape, measured on Qwen3.8-27B-NVFP4: 2048->1024 alone reads 0.34
    // relative L2 and does not grow with chunk count), so only the RATIO is a property of the restore.
    EXPECT_LE(chain_rel.back(), 2.0 * chunked + 1e-9)
        << "the chained restore drifted further than a cold prefill chunked at the same boundaries: "
        << chain_rel.back() << " vs " << chunked << " relative L2 (h dtype " << qtype_name(h_dtype)
        << "). The restore itself is wrong, not the boundary count.";
}

INSTANTIATE_TEST_SUITE_P(Arms, HybridRestoreChainTest,
                         ::testing::Values(ChainArm{true, true}, ChainArm{true, false}, ChainArm{false, true},
                                           ChainArm{false, false}),
                         arm_name);

}  // namespace imp
