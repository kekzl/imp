// E2E proof for [runtime] deterministic — issue #522 item 3.
//
// The deterministic kernel variants (MoE permute/scatter, top-k softmax sum,
// cuBLASLt algo pinning) shipped with the audit-B9 work, gated on
// process_diag_deterministic_gemm(), but nothing ever ASSERTED the resulting
// guarantee. This test is that proof, against the model family where the
// nondeterminism is documented (MoE/hybrid — Qwen3.6-35B is the known
// temp=0 flipper; see memory/audit B-9):
//
//   deterministic=true  ⇒  greedy output and teacher-forced perplexity are
//   byte-/bit-identical for repeated requests on one context (server eval
//   steady state) and across fresh processes (CLI eval runs; verified
//   manually, 2× identical on Qwen3.6-35B).
//
// KNOWN LIMIT (the DISABLED_ tests below are the gate): across FRESH
// CONTEXTS in ONE process, output is only *usually* identical, for both
// greedy decode and perplexity. PPL deltas run to a few percent, i.e.
// state-sized, not rounding-sized. The varying input is per-context engine
// state (VRAM layout, CUDA-graph capture, slot assignment); not yet pinned
// down. Eval workloads don't hit this (server = one context; CLI = one
// process).
//
// WHICH model carries it: docs/determinism.md §4, measured 2026-08-10, is
// the source. Do not re-derive it here. This comment used to say GDN-hybrid,
// which that measurement refutes: the MoE row fails 2/2 across fresh
// contexts while the dense row passes. Historical note: SAME-context perplexity
// drift had a separate root cause — imp_perplexity nulled active_request
// before imp_context_reset, leaking the KV sequence + SSM/GDN slot per call
// (fixed in imp_api.cpp alongside this test).
//
// Runs against EVERY model env var below that names something on disk:
// IMP_TEST_MOE_MODEL (the MoE/hybrid case this suite was written for) and
// IMP_TEST_MODEL (dense). The header used to claim "dense models pass
// trivially since they skip the routed-expert kernels" — #1299 disproved that:
// the dense half failed on its own root cause (a prefix-cache hit reaching a
// test that had asked for none, #1337) and a single-model gate could not see
// it. Parameterising is what makes the second row exist.
//
// The deterministic flag reaches the engine via the legacy IMP_DETERMINISTIC
// env seed: library users without a tool-main RuntimeConfig::load() get
// env-seeded defaults from take_pending_runtime_config() at engine init.

#include <gtest/gtest.h>

#include "imp/imp.h"
#include "test_models.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool is_safetensors_dir(const std::string& p) {
    return p.size() < 5 || p.substr(p.size() - 5) != ".gguf";
}

// One row per model env var. A row whose variable is unset, or set to a path
// that is not there, SKIPS — tests/README.md:5 promises they "never fail for a
// missing prerequisite", and the old predicate only honoured that for an unset
// variable: a wrong path produced hard failures.
struct DetModel {
    const char* env;
    const char* label;
};

std::vector<DetModel> det_models() {
    return {{imp_test::kEnvMoeModel, "moe"}, {imp_test::kEnvModel, "dense"}};
}

class DetEvalE2ETest : public ::testing::TestWithParam<DetModel> {
protected:
    void SetUp() override {
        const char* path = std::getenv(GetParam().env);
        if (path == nullptr)
            GTEST_SKIP() << "Set " << GetParam().env << " to run deterministic-mode E2E";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path, GetParam().env));
        // Must be set BEFORE imp_context_create: engine init pulls the
        // env-seeded RuntimeConfig via take_pending_runtime_config().
        setenv("IMP_DETERMINISTIC", "1", 1);
        path_ = path;
        fmt_ = is_safetensors_dir(path_) ? IMP_FORMAT_SAFETENSORS : IMP_FORMAT_GGUF;
        ASSERT_EQ(imp_model_load(path, fmt_, &model_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
        unsetenv("IMP_DETERMINISTIC");
    }

    void MakeContext(bool cuda_graphs = true) {
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 1024;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = cuda_graphs ? 1 : 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void FreeContext() {
        imp_context_free(ctx_);
        ctx_ = nullptr;
    }

    std::string gen(const char* prompt, int max_tokens) {
        ImpGenerateParams p = imp_generate_params_default();
        p.temperature = 0.0f;
        p.top_k = 1;
        p.top_p = 1.0f;
        p.seed = 42;
        p.max_tokens = max_tokens;
        p.ignore_eos = 0;

        std::string out(16384, '\0');
        size_t out_len = 0;
        EXPECT_EQ(imp_generate(ctx_, prompt, &p, out.data(), out.size(), &out_len), IMP_SUCCESS);
        out.resize(out_len);
        return out;
    }

    std::string path_;
    ImpModelFormat fmt_ = IMP_FORMAT_GGUF;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

// Prompts chosen to push through enough routed-expert decode steps that an
// atomics-order flip would have many chances to surface (the documented
// Qwen3.6-35B flips appeared within ~100 tokens).
constexpr const char* kPrompts[] = {
    "Explain why the sky is blue, then explain why sunsets are red.",
    "List the first ten prime numbers and briefly say why each is prime.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
};

constexpr int kGen = 96;

// 0. DIAGNOSTIC SPLIT: same-context repeats with CUDA graphs OFF. The first
//    request on a context runs partially eager while the conditional graph
//    captures; later requests replay the graph — a different kernel mix for
//    the same step. Graphs-off isolates the underlying kernels from that mix.
TEST_P(DetEvalE2ETest, GreedyReproducibleSameContextGraphsOff) {
    MakeContext(/*cuda_graphs=*/false);
    for (const char* prompt : kPrompts) {
        std::string first = gen(prompt, kGen);
        std::string second = gen(prompt, kGen);

        ASSERT_FALSE(first.empty()) << "model produced no output for: " << prompt;
        EXPECT_EQ(first, second) << "deterministic=1 greedy diverged back-to-back with graphs "
                                    "OFF for prompt:\n  "
                                 << prompt << "\n  run1: " << first << "\n  run2: " << second;
    }
    FreeContext();
}

// 1. Greedy output must be byte-identical for repeated requests on ONE
//    context — the serving/eval steady state (server process, many requests).
TEST_P(DetEvalE2ETest, GreedyReproducibleSameContext) {
    MakeContext();
    for (const char* prompt : kPrompts) {
        std::string first = gen(prompt, kGen);
        std::string second = gen(prompt, kGen);

        ASSERT_FALSE(first.empty()) << "model produced no output for: " << prompt;
        EXPECT_EQ(first, second) << "deterministic=1 greedy output diverged back-to-back on "
                                    "one context for prompt:\n  "
                                 << prompt << "\n  run1: " << first << "\n  run2: " << second;
    }
    FreeContext();
}

// 2. KNOWN LIMIT (see header): cross-context greedy in one process is flaky
//    — DISABLED_ is the gate; enable when the layout-sensitive source is
//    pinned and fixed.
TEST_P(DetEvalE2ETest, DISABLED_GreedyReproducibleAcrossFreshContexts) {
    for (const char* prompt : kPrompts) {
        MakeContext();
        std::string first = gen(prompt, kGen);
        FreeContext();

        MakeContext();
        std::string second = gen(prompt, kGen);
        FreeContext();

        ASSERT_FALSE(first.empty()) << "model produced no output for: " << prompt;
        EXPECT_EQ(first, second) << "deterministic=1 greedy output diverged across fresh "
                                    "contexts for prompt:\n  "
                                 << prompt << "\n  run1: " << first << "\n  run2: " << second;
    }
}

// 3. Teacher-forced perplexity must be bit-identical for repeated scoring on
//    one context — the eval number itself is the artifact agent evals
//    compare. (Guards the per-position NLL reduction: the old cross-block
//    double atomicAdd accumulated in scheduling-dependent order.)
TEST_P(DetEvalE2ETest, PerplexityBitIdenticalSameContext) {
    // Token IDs are model-agnostic small IDs; content doesn't matter for
    // reproducibility, only that the same sequence is scored twice.
    std::vector<int32_t> tokens;
    for (int i = 0; i < 256; ++i)
        tokens.push_back(1000 + (i * 37) % 4000);

    MakeContext();
    double ppl1 = 0.0;
    ASSERT_EQ(imp_perplexity(ctx_, tokens.data(), static_cast<int>(tokens.size()), &ppl1), IMP_SUCCESS);
    double ppl2 = 0.0;
    ASSERT_EQ(imp_perplexity(ctx_, tokens.data(), static_cast<int>(tokens.size()), &ppl2), IMP_SUCCESS);
    FreeContext();

    EXPECT_GT(ppl1, 0.0);
    // Bit-identical, not approximately equal — that is the deliverable.
    EXPECT_EQ(ppl1, ppl2) << "perplexity differs back-to-back under deterministic=1: " << ppl1
                          << " vs " << ppl2;
}

// 4. KNOWN LIMIT companion (see header): cross-context perplexity in one
//    process is flaky, like the greedy variant. For which model, see
//    docs/determinism.md §4, not this comment.
TEST_P(DetEvalE2ETest, DISABLED_PerplexityBitIdenticalAcrossFreshContexts) {
    std::vector<int32_t> tokens;
    for (int i = 0; i < 256; ++i)
        tokens.push_back(1000 + (i * 37) % 4000);

    MakeContext();
    double ppl1 = 0.0;
    ASSERT_EQ(imp_perplexity(ctx_, tokens.data(), static_cast<int>(tokens.size()), &ppl1), IMP_SUCCESS);
    FreeContext();

    MakeContext();
    double ppl2 = 0.0;
    ASSERT_EQ(imp_perplexity(ctx_, tokens.data(), static_cast<int>(tokens.size()), &ppl2), IMP_SUCCESS);
    FreeContext();

    EXPECT_EQ(ppl1, ppl2) << "perplexity differs across fresh contexts under deterministic=1: "
                          << ppl1 << " vs " << ppl2;
}

INSTANTIATE_TEST_SUITE_P(Models, DetEvalE2ETest, ::testing::ValuesIn(det_models()),
                         [](const ::testing::TestParamInfo<DetModel>& info) {
                             return std::string(info.param.label);
                         });

}  // namespace
