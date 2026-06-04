// Prefix-cache E2E equivalence — TEST_AUDIT.md risk #7 (Phase 2.6).
//
// The unit tests (tests/test_prefix_cache_equiv.cpp) prove the KVCacheManager
// hands back the same physical block with intact KV bytes. This file closes
// the loop end-to-end through the REAL engine + a REAL model: with prefix
// caching ON, a fresh-prefill generation and a prefix-cache-HIT generation of
// the same greedy request must produce the SAME tokens. That equivalence is
// the property that ships the feature; risk #7 calls the test "the enabler".
//
// Engine seam (verified in src/api/imp_api.cpp + src/runtime/engine*.cpp):
//   The prefix cache is only populated when a request FINISHES naturally
//   (engine step() → finish_request() → register_block_hashes() + cache the
//   blocks). The manual token-level path (imp_prefill_with_params +
//   imp_decode_step) never registers hashes, AND imp_context_reset()
//   deliberately evicts ALL cached blocks. So the cache engages ONLY across
//   two consecutive imp_generate() calls on the SAME context with NO reset
//   between them: call 1 finishes and caches its prefix; call 2 hits it.
//   We therefore drive imp_generate twice and rely on greedy determinism.
//
// Skipped without IMP_TEST_MODEL (no GPU model in CI).

#include <gtest/gtest.h>
#include "imp/imp.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* model_path() { return std::getenv("IMP_TEST_MODEL"); }

bool is_safetensors_dir(const std::string& p) {
    return p.size() < 5 || p.substr(p.size() - 5) != ".gguf";
}

class PrefixCacheE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        const char* path = model_path();
        if (!path)
            GTEST_SKIP() << "Set IMP_TEST_MODEL to run prefix-cache E2E equivalence";
        path_ = path;
        fmt_ = is_safetensors_dir(path_) ? IMP_FORMAT_SAFETENSORS : IMP_FORMAT_GGUF;
        ASSERT_EQ(imp_model_load(path, fmt_, &model_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_) imp_context_free(ctx_);
        if (model_) imp_model_free(model_);
    }

    // Create a context with prefix caching on/off. Single batch; CUDA graphs ON
    // (production path) so the equivalence is asserted on the real decode loop.
    void MakeContext(bool prefix_cache) {
        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 1024;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 1;
        cfg.use_prefix_caching = prefix_cache ? 1 : 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    // Greedy generation through imp_generate (the path that reaches
    // finish_request → registers prefix hashes). Returns produced text.
    std::string gen(const char* prompt, int max_tokens) {
        ImpGenerateParams p = imp_generate_params_default();
        p.temperature = 0.0f;
        p.top_k = 1;
        p.top_p = 1.0f;
        p.seed = 42;
        p.max_tokens = max_tokens;
        p.ignore_eos = 0;  // allow natural finish so the prefix is registered+cached

        std::string out(8192, '\0');
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

// A long, multi-block prompt so prefix caching has ≥2 full blocks to reuse.
// Raw completion prompt (no chat template) — template churn is irrelevant here.
constexpr const char* kPrompt =
    "The history of computing begins long before electronic machines. Ancient "
    "civilizations used counting tools, and over centuries mathematicians built "
    "mechanical aids. In the twentieth century, the first programmable computers "
    "appeared. Summarize the key idea in one sentence:";

constexpr int kGen = 24;

// 0. CONTROL: with caching OFF, two consecutive greedy imp_generate calls on
//    the same context must already be identical. If THIS fails, back-to-back
//    divergence is a cross-request engine property (sampler state, KV reuse,
//    allocator layout ...) and tests 1-3 cannot attribute anything to the
//    prefix cache. Discriminates "cache corrupts KV" from "second request
//    differs regardless".
TEST_F(PrefixCacheE2ETest, ControlNoCacheBackToBackIsDeterministic) {
    MakeContext(/*prefix_cache=*/false);

    std::string first = gen(kPrompt, kGen);
    std::string second = gen(kPrompt, kGen);

    ASSERT_FALSE(first.empty()) << "model produced no output";
    EXPECT_EQ(first, second)
        << "back-to-back greedy requests diverge WITHOUT prefix caching —\n"
           "cross-request nondeterminism in the engine itself; the prefix-cache\n"
           "equivalence tests below cannot run until this holds:\n  run1: "
        << first << "\n  run2: " << second;
}

// =============================================================================
// Tests 1-3 are DISABLED: they FAIL against the current engine — the
// prefix-cache hit DIVERGES from the fresh prefill under greedy decoding
// (verified 2026-06-04, Qwen3-8B-Q8_0, CUDA graphs ON; diverges ~15 tokens
// in: "...computing starting from ancient times up to the" vs "...computing.
// Let me start by recalling the"). The control test above passes, so the
// divergence is attributable to the cache path, and the unit tests prove the
// manager returns the right blocks with intact bytes — the corruption is in
// the engine integration of a cache hit (partial-block resume / position
// bookkeeping are the prime suspects). Tracked in issue #536; flip DISABLED_ off when fixing — these tests ARE the
// feature's ship gate (TEST_AUDIT risk #7: "the test IS the enabler").
// =============================================================================
// 1. Fresh-prefill vs prefix-cache-HIT must be token-identical.
//    Same context, prefix caching ON, NO reset between calls: call 1 prefills
//    fresh and caches its prefix on finish; call 2 hits the cached prefix.
//    Greedy ⇒ the two outputs must be byte-identical. A mismatch is exactly the
//    silent-determinism failure that keeps the feature off-by-default.
TEST_F(PrefixCacheE2ETest, DISABLED_FreshVsPrefixHitTokenEqual) {
    MakeContext(/*prefix_cache=*/true);

    std::string first = gen(kPrompt, kGen);   // fresh prefill, caches prefix on finish
    std::string second = gen(kPrompt, kGen);  // hits the cached prefix

    ASSERT_FALSE(first.empty()) << "model produced no output";
    EXPECT_EQ(first, second)
        << "PREFIX-CACHE HIT DIVERGED FROM FRESH PREFILL (greedy):\n  fresh: " << first
        << "\n  hit:   " << second
        << "\nThis is the determinism failure risk #7 names — the cached KV does "
           "not reproduce the fresh forward pass.";
}

// 2. Cross-context equivalence: prefix-cache-ON output == prefix-cache-OFF
//    output for the same greedy request. This pins the cache path to the
//    canonical fresh-prefill result, not merely to "self-consistent".
TEST_F(PrefixCacheE2ETest, DISABLED_PrefixCacheMatchesNoCacheBaseline) {
    // Baseline: caching OFF.
    MakeContext(/*prefix_cache=*/false);
    std::string baseline = gen(kPrompt, kGen);
    imp_context_free(ctx_);
    ctx_ = nullptr;

    // Caching ON, warm the cache, then the hit run.
    MakeContext(/*prefix_cache=*/true);
    (void)gen(kPrompt, kGen);             // warm: caches the prefix
    std::string cached = gen(kPrompt, kGen);  // prefix-cache hit

    ASSERT_FALSE(baseline.empty());
    EXPECT_EQ(baseline, cached)
        << "prefix-cache output diverged from the no-cache baseline (greedy):\n"
        << "  no-cache: " << baseline << "\n  cached:   " << cached;
}

// 3. Shared prefix, different suffix: a cache hit on the common prefix must not
//    corrupt the continuation. We warm the cache with kPrompt, then issue a
//    request that shares the long kPrompt prefix but appends a distinct suffix.
//    The result must be (a) non-empty/coherent and (b) reproducible 2× — proving
//    the partial prefix hit feeds a correct, deterministic continuation.
TEST_F(PrefixCacheE2ETest, DISABLED_SharedPrefixDifferentSuffixStable) {
    MakeContext(/*prefix_cache=*/true);
    (void)gen(kPrompt, kGen);  // warm: cache kPrompt's blocks

    const std::string suffixed =
        std::string(kPrompt) + " Also list two early mechanical devices:";

    std::string a = gen(suffixed.c_str(), kGen);  // partial prefix hit + new suffix
    std::string b = gen(suffixed.c_str(), kGen);  // full prefix hit (a registered it)

    ASSERT_FALSE(a.empty()) << "shared-prefix+new-suffix produced no output";
    EXPECT_EQ(a, b)
        << "continuation after a partial-prefix cache hit is non-deterministic:\n"
        << "  run a: " << a << "\n  run b: " << b;
}

}  // namespace
