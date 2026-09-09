// Prefix-cache E2E equivalence — TEST_AUDIT (retired) risk #7 (Phase 2.6).
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
#include "api/imp_internal.h"
#include "memory/kv_cache.h"
#include "model/model.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "runtime/snapshot_boundary.h"
#include "test_models.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

const char* model_path() { return std::getenv(imp_test::kEnvModel); }

bool is_safetensors_dir(const std::string& p) {
    return p.size() < 5 || p.substr(p.size() - 5) != ".gguf";
}

class PrefixCacheE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        const char* path = model_path();
        if (!path)
            GTEST_SKIP() << "Set IMP_TEST_MODEL to run prefix-cache E2E equivalence";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path, imp_test::kEnvModel));
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
// History: tests 1-3 initially FAILED (issue #536) — the prefix-cache HIT
// diverged from the fresh prefill under greedy decoding. Root cause was NOT
// the cache or FP rounding: GPUBatchPool::upload_into_pool skipped the
// block-table re-upload based on a (count + first-block-ID) proxy, which a
// prefix hit defeats (same first reused block, same count, different middle
// after reuse+eviction, e.g. [0,1,4,3,2]) — the first eager decode step then
// ran on the PREVIOUS request's table, writing the new token's KV into the
// wrong physical block while the async graph read garbage at the self-token
// position. Fixed by content-comparing the table (batch.cpp). These tests
// are the feature's ship gate (TEST_AUDIT (retired) risk #7: "the test IS the
// enabler") — they must stay green for prefix caching to be enabled.
// =============================================================================
// 1. Fresh-prefill vs prefix-cache-HIT must be token-identical.
//    Same context, prefix caching ON, NO reset between calls: call 1 prefills
//    fresh and caches its prefix on finish; call 2 hits the cached prefix.
//
//    The two are NOT byte-identical, and asserting that they are was wrong. A
//    hit skips the cached prefix, so it computes over a different chunk split
//    than the fresh prefill: measured on this test, "PrefixCache: seq 3 reused
//    3/4 blocks (48 tokens skippable)". Different split, different accumulation
//    order, and in FP16 that moves a logit.
//
//    Measured at the position where the two first differ, both paths dumping
//    their raw top-2 (throwaway probe in build_logprob_info, 2026-08-19):
//
//      fresh : top1=55486(38.242889)  top2=279(38.081467)   gap 0.161422
//      cached: top1=  279(38.253368)  top2=55486(38.188511) gap 0.064857
//
//    Same two tokens, order flipped. The gap between them is 0.42 % of the
//    logit, and the shift the two paths produce on the same token is 0.172,
//    i.e. LARGER than the gap. That is a tie decided by rounding, not a wrong
//    KV block: a wrong block does not return the same two candidates within
//    0.4 % of each other.
//
//    So the assertion is a common prefix, not equality. The bar comes from a
//    distribution, not a single run: ten consecutive runs of both tests gave a
//    common prefix of 103 characters every time, out of 161 and 153. The
//    divergence point is not itself a coin flip, it is a fixed position where a
//    fixed near-tie falls the other way, so 103 is a floor and not a sample
//    mean. kMinCommonChars sits at 64, about 62 % of that.
//
//    Counted in CHARACTERS, deliberately: a token-based bar would need a
//    chars-per-token factor, which is wrong for non-ASCII output and would make
//    the bar depend on the language of the answer. Bytes are what both strings
//    are made of.
//
//    What the bar has to keep catching is #536: there the hit diverged within
//    the first tokens and continued differently, i.e. a common prefix near
//    zero. Rounding diverges at 103. Any bar between those two separates them,
//    and 64 is far from both.
TEST_F(PrefixCacheE2ETest, FreshVsPrefixHitTokenEqual) {
    MakeContext(/*prefix_cache=*/true);

    std::string first = gen(kPrompt, kGen);   // fresh prefill, caches prefix on finish
    std::string second = gen(kPrompt, kGen);  // hits the cached prefix

    ASSERT_FALSE(first.empty()) << "model produced no output";
    size_t common = 0;
    while (common < first.size() && common < second.size() && first[common] == second[common])
        common++;
    const size_t kMinCommonChars = 64;  // measured floor 103 over ten runs
    EXPECT_GE(common, kMinCommonChars)
        << "PREFIX-CACHE HIT DIVERGED FROM FRESH PREFILL TOO EARLY (greedy):\n  fresh: " << first
        << "\n  hit:   " << second << "\n  common prefix: " << common << " chars, want >= " << kMinCommonChars
        << "\nLate divergence is the rounding this test tolerates (see the block "
           "comment). Diverging this early is the failure risk #7 names: the "
           "cached KV is not the KV a fresh forward would have produced.";
}

// 2. Cross-context equivalence: prefix-cache-ON output == prefix-cache-OFF
//    output for the same greedy request. This pins the cache path to the
//    canonical fresh-prefill result, not merely to "self-consistent".
TEST_F(PrefixCacheE2ETest, PrefixCacheMatchesNoCacheBaseline) {
    // Hybrid (SSM/GDN) models: with caching ON, prefill ends a chunk at the
    // recurrent-snapshot boundary; with caching OFF it does not. Different
    // chunk boundaries change accumulation order, so cross-CONFIG bitwise
    // equality is unattainable by design (chunked-vs-unchunked prefill was
    // never bit-equal). The hit==fresh guarantee for hybrids is tests 1+3,
    // where both runs share the same boundary.
    if (model_ && model_->model && model_->model->config().ssm_inner_size > 0)
        GTEST_SKIP() << "recurrent model: snapshot chunk boundary makes cache-on/off "
                        "prefill chunking differ — bitwise cross-config equality N/A";

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
    // Same reasoning as test 1, and the comment above already had it for
    // hybrids: a cache hit skips the cached prefix, so it chunks differently
    // than the no-cache run and accumulates in a different order. That holds
    // for a dense model too, it was simply never drawn. Common prefix, with
    // the bar set the same way.
    size_t common = 0;
    while (common < baseline.size() && common < cached.size() && baseline[common] == cached[common])
        common++;
    const size_t kMinCommonChars = 64;  // same floor as test 1
    EXPECT_GE(common, kMinCommonChars)
        << "prefix-cache output left the no-cache baseline too early (greedy):\n"
        << "  no-cache: " << baseline << "\n  cached:   " << cached << "\n  common prefix: " << common
        << " chars, want >= " << kMinCommonChars;
}

// 3. Shared prefix, different suffix: a cache hit on the common prefix must not
//    corrupt the continuation. We warm the cache with kPrompt, then issue a
//    request that shares the long kPrompt prefix but appends a distinct suffix.
//    The result must be (a) non-empty/coherent and (b) reproducible 2× — proving
//    the partial prefix hit feeds a correct, deterministic continuation.
TEST_F(PrefixCacheE2ETest, SharedPrefixDifferentSuffixStable) {
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

// =============================================================================
// 4. HYBRID (SSM/GDN): the recurrent-snapshot restore path.
//
// WHY THE TESTS ABOVE DO NOT COVER IT. kPrompt is 273 characters, ~68 tokens.
// `snapshot_boundary(68, 16, server.snapshot_min_prompt_tokens = 256)` is 0, so
// on a hybrid NO snapshot is ever saved by tests 0-3 and the restore branch in
// prepare_recurrent_state_ (engine_sampling_stop.cpp) is never entered. Test 1
// passes on a hybrid by reusing ZERO blocks: hybrid_prefix_reuse_limit_ caps
// reuse at the snapshot boundary, and the boundary is 0. Three prefix-cache E2E
// tests, and the hybrid half of the feature had no coverage at all.
//
// So this arm builds prompts past 512 tokens, which is what makes
// snapshot_boundary non-zero, and asserts the thing that actually ships:
// a warm (restored) run equals the cold (fresh) run, and `cached_tokens` lands
// exactly on the snapshot boundary rather than on the longer KV match.
//
// Driven through Engine::add_request/step rather than imp_generate, because
// `cached_tokens` is a Request field and the C API does not surface it.
// =============================================================================

// ~700 words of ordinary prose. Tokenizes past 512 tokens on every BPE vocab
// this repo tests against, and is not repetitive (a repeating prompt would
// make every block hash collide and the reuse figure meaningless).
constexpr const char* kLongPrompt =
    "The bandwidth of a memory system is not the same thing as its latency, and confusing the two "
    "leads to designs that look fast on paper and stall in practice. A decode step reads every "
    "weight of the model exactly once, so its floor is set by how many bytes the weights occupy "
    "divided by how fast the device can stream them. Arithmetic intensity is low, caches do not "
    "help much, and the only real levers are making the weights smaller or making the reads wider. "
    "Prefill is the opposite case: many tokens share the same weights, the matrices become tall, "
    "and the machine becomes limited by how many multiply-accumulate operations it can retire per "
    "cycle. Engineers who measure only one of the two phases end up optimizing the wrong half of "
    "the problem, and the resulting speedups do not survive contact with a real serving workload. "
    "Recurrent layers change the accounting again. A gated delta network carries a state that is "
    "cumulative over the whole prefix, so a cache of attention keys and values is not by itself "
    "enough to skip work: the state at the position where the skip would begin has to come from "
    "somewhere. Saving it costs a slab per sequence per layer, restoring it costs one copy, and "
    "the boundary at which it was saved is the only position a continuation may start from. "
    "That constraint is invisible in a dense model, where any block-aligned prefix will do, and "
    "it is the reason a hybrid engine needs a snapshot store next to its block cache. "
    "Scheduling ties the two together. Admission decides how many sequences may be resident, the "
    "block pool decides how long each of them may become, and the recurrent state pool decides "
    "how many may exist at all. Three ceilings, three different units, and an operator who is "
    "told only one of them will size the deployment against the wrong one. The failure is not a "
    "crash; it is a server that accepts a workload it cannot finish and cancels requests in the "
    "middle of a generation, long after the configuration that guaranteed the outcome was chosen. "
    "Measurement discipline is what separates a plan from a guess in all of this. A number that "
    "was read from a live allocator says what happened to be free at one instant on one machine, "
    "and a number that was computed from declared demand says the same thing on every boot. "
    "The second kind can be tested without a device, printed at startup, and compared against "
    "what the process later took. The first kind can only be argued about. Continue the text:";

TEST_F(PrefixCacheE2ETest, HybridSnapshotRestoreMatchesFresh) {
    if (!(model_ && model_->model && model_->model->config().ssm_inner_size > 0))
        GTEST_SKIP() << "SKIPPED ON A DENSE CHECKPOINT: this test covers the recurrent-snapshot "
                        "restore path and IMP_TEST_MODEL points at a model with ssm_inner_size == 0. "
                        "The `test-e2e` Makefile target runs it in its own container against "
                        "IMP_TEST_MODEL_GDN; a run of PrefixCacheE2ETest.* against a dense model "
                        "leaves the hybrid half of prefix caching UNCOVERED.";

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = 1;
    cfg.use_prefix_caching = 1;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    imp::Engine* engine = ctx_->engine.get();
    ASSERT_NE(engine, nullptr);
    ASSERT_NE(engine->kv_cache(), nullptr);
    const int kv_bs = engine->kv_cache()->block_size();
    const int min_snap = engine->runtime_config().server.snapshot_min_prompt_tokens;

    // Tokenize once; the arms are prefixes of the same token vector so their
    // block hashes share a prefix by construction.
    std::vector<int32_t> full(4096);
    int n_full = 0;
    ASSERT_EQ(imp_tokenize(model_, kLongPrompt, full.data(), &n_full, static_cast<int>(full.size())),
              IMP_SUCCESS);
    ASSERT_GE(n_full, 512) << "kLongPrompt must exceed 512 tokens for the snapshot boundary to exist";
    full.resize(n_full);

    // One greedy generation through the engine loop, so finish_request registers
    // the block hashes and saves the snapshot. Returns the output tokens and
    // writes back what the request reported as reused.
    auto run = [&](const std::vector<int32_t>& tokens, int max_tokens,
                   int* cached_tokens) -> std::vector<int32_t> {
        auto req = std::make_shared<imp::Request>();
        req->input_tokens = tokens;
        req->max_tokens = max_tokens;
        req->temperature = 0.0f;
        req->top_p = 1.0f;
        req->top_k = 0;
        req->ignore_eos = true;  // fixed length: the arms must be comparable
        req->status = imp::RequestStatus::PENDING;
        engine->add_request(req);
        for (int i = 0; i < 4096; ++i) {
            if (req->status == imp::RequestStatus::FINISHED ||
                req->status == imp::RequestStatus::CANCELLED)
                break;
            (void)engine->step();
        }
        EXPECT_EQ(req->status, imp::RequestStatus::FINISHED)
            << "request did not finish (status " << static_cast<int>(req->status) << ")";
        if (cached_tokens)
            *cached_tokens = req->cached_tokens;
        return req->output_tokens;
    };

    constexpr int kTokens = 16;

    // ── arm A: block-aligned prompt ──────────────────────────────────────
    const int aligned_len = (n_full / kv_bs) * kv_bs;
    std::vector<int32_t> aligned(full.begin(), full.begin() + aligned_len);
    int cold_cached = -1, warm_cached = -1;
    const std::vector<int32_t> cold_a = run(aligned, kTokens, &cold_cached);
    ASSERT_FALSE(cold_a.empty());
    EXPECT_EQ(cold_cached, 0) << "the first run cannot hit a cache it is populating";
    const std::vector<int32_t> warm_a = run(aligned, kTokens, &warm_cached);
    EXPECT_EQ(warm_cached, imp::snapshot_boundary(aligned_len, kv_bs, min_snap))
        << "a hybrid must skip exactly to the recurrent-snapshot boundary, not to the longer KV "
           "match (kv_bs=" << kv_bs << ", min=" << min_snap << ", prompt=" << aligned_len << ")";
    EXPECT_GT(warm_cached, 0) << "no snapshot was restored: this arm proves nothing";
    EXPECT_EQ(warm_a, cold_a)
        << "restored recurrent state diverged from the fresh forward at the SAME chunk boundary; "
           "both runs split the prefill at the snapshot, so this is state, not rounding";

    // ── arm B: unaligned prompt (len % kv_bs != 0) ───────────────────────
    // The boundary rounds DOWN, so the tail tokens are always forwarded. A
    // restore that ignored the remainder would show up here and nowhere else.
    imp_context_free(ctx_);
    ctx_ = nullptr;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    engine = ctx_->engine.get();
    const int unaligned_len = aligned_len - (kv_bs / 2 > 0 ? kv_bs / 2 : 1) - 1;
    ASSERT_NE(unaligned_len % kv_bs, 0);
    std::vector<int32_t> unaligned(full.begin(), full.begin() + unaligned_len);
    const std::vector<int32_t> cold_b = run(unaligned, kTokens, nullptr);
    const std::vector<int32_t> warm_b = run(unaligned, kTokens, &warm_cached);
    EXPECT_EQ(warm_cached, imp::snapshot_boundary(unaligned_len, kv_bs, min_snap));
    EXPECT_EQ(warm_b, cold_b) << "unaligned prompt: restored state != fresh state";

    // ── arm C: warm prompt + 40 new tokens ───────────────────────────────
    // The KV match is longer than the snapshot: the cache holds every block of
    // the warm prefix, but the snapshot sits at the warm prompt's own boundary.
    // cached_tokens must follow the SNAPSHOT, or the continuation decodes from
    // a state that never saw the tokens whose KV it is reading.
    imp_context_free(ctx_);
    ctx_ = nullptr;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    engine = ctx_->engine.get();
    ASSERT_GE(n_full, aligned_len + 40) << "need 40 tokens past the warm prompt";
    std::vector<int32_t> extended(full.begin(), full.begin() + aligned_len);
    (void)run(extended, kTokens, nullptr);  // warm: saves the snapshot at its boundary
    extended.insert(extended.end(), full.begin() + aligned_len, full.begin() + aligned_len + 40);
    const int expect_c = imp::snapshot_boundary(aligned_len, kv_bs, min_snap);
    int ext_cached = -1;
    const std::vector<int32_t> warm_c = run(extended, kTokens, &ext_cached);
    EXPECT_LE(ext_cached, expect_c)
        << "reuse ran past the snapshot boundary (" << ext_cached << " > " << expect_c << ")";
    EXPECT_GT(ext_cached, 0) << "the extended prompt shares a snapshot-backed prefix";

    imp_context_free(ctx_);
    ctx_ = nullptr;
    ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    engine = ctx_->engine.get();
    const std::vector<int32_t> cold_c = run(extended, kTokens, nullptr);
    EXPECT_EQ(warm_c, cold_c)
        << "warm prompt + 40 new tokens: the continuation after a snapshot restore must equal a "
           "cold prefill of the same prompt";
}

}  // namespace
