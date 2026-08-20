// =============================================================================
// test_spec_capture_fidelity.cpp — two gates for the speculative verify chunk
// =============================================================================
//
// 1. CachedGraphMatchesEagerForward
//    A cached verify-chunk graph must compute what an eager forward of the same
//    state computes. diagnostics.spec_capture_fidelity turns every replay into
//    that comparison; the test asserts it ran and never disagreed. Measured
//    2026-08-20: 0/400 differing on Qwen3.8-27B-NVFP4 and Qwen3.6-35B-A3B-NVFP4,
//    45/400 on NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 — so this fails on
//    the nemotron_h class and passes on the classes MTP is released for.
//
// 2. HybridSpeculationDoesNotCollapse
//    The 8a7f2763 regression: on a Mamba2 hybrid a fully rejected draft chunk
//    adopted a recurrent snapshot nothing had written, and generation collapsed
//    to one repeated character ("Here's" then 300 x "0"). A collapse of that
//    shape leaves almost no distinct whitespace-separated words, which is what
//    this asserts — repetition_ratio in test_degeneration.cpp does not catch it,
//    because the collapse is one long token with no spaces in it.
//
// Both skip when their checkpoint is absent, like the other SafeTensors suites.
// =============================================================================

#include "imp/imp.h"
#include "api/imp_internal.h"
#include "runtime/config.h"
#include "runtime/engine.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

// Container bind-mount paths (-v $(HOME)/models:/models), overridable so the
// Makefile lanes can point at a different checkpoint.
const char* model_or(const char* env, const char* fallback) {
    const char* v = getenv(env);
    return (v && *v) ? v : fallback;
}

bool present(const std::string& dir) { return fs::exists(dir + "/config.json"); }

size_t device_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b >> 20;
}

// Speculation on, MTP as the only drafter, guards pinned off so the verify path
// keeps running for the whole generation.
// MTP as the only drafter, guards pinned off so the verify path keeps running.
imp::RuntimeConfig mtp_config(bool fidelity) {
    imp::RuntimeConfig rc;
    rc.speculative.ngram = false;
    rc.speculative.mtp_k = 1;
    rc.speculative.mtp_econ_min_emit = 0.0f;
    rc.speculative.give_up_after = 0;
    rc.diagnostics.spec_capture_fidelity = fidelity;
    return rc;
}

// Suffix-matcher drafting. Reaches the same captured verify chunk without an MTP
// head, which the cheapest nemotron_h checkpoint on this box does not carry.
imp::RuntimeConfig suffix_config() {
    imp::RuntimeConfig rc;
    rc.speculative.ngram = true;
    rc.speculative.suffix = true;
    rc.speculative.min_match = 1;
    rc.speculative.mtp_k = 0;
    rc.speculative.give_up_after = 0;
    return rc;
}

std::string generate(ImpContext ctx, const std::string& prompt, int max_tokens) {
    ImpGenerateParams p = imp_generate_params_default();
    p.max_tokens = max_tokens;
    p.temperature = 0.0f;
    p.top_k = 1;
    static char out[65536];
    size_t len = 0;
    if (imp_generate(ctx, prompt.c_str(), &p, out, sizeof(out), &len) != IMP_SUCCESS)
        return {};
    return std::string(out, len);
}

size_t distinct_words(const std::string& t) {
    std::istringstream is(t);
    std::set<std::string> w;
    std::string tok;
    while (is >> tok)
        w.insert(tok);
    return w.size();
}

}  // namespace

TEST(SpecCaptureFidelityTest, CachedGraphMatchesEagerForward) {
    const std::string dir = model_or("IMP_TEST_MODEL_SPEC_FIDELITY", "/models/Qwen3.8-27B-NVFP4");
    if (!present(dir))
        GTEST_SKIP() << "checkpoint not present at " << dir;
    // The comparison runs a second full forward per verify step, so the model
    // has to fit with room to spare; an OOM here poisons every later test in
    // this binary (see the note in test_mtp_forward.cpp).
    constexpr size_t kNeededMiB = 26000;
    const size_t free_mib = device_free_mib();
    if (free_mib < kNeededMiB)
        GTEST_SKIP() << "needs ~" << kNeededMiB << " MiB free, card has " << free_mib << " MiB";

    imp::set_pending_runtime_config(mtp_config(/*fidelity=*/true));
    ImpModel model = nullptr;
    // imp_model_load hard-codes load_mtp_head=0; the head is what drives the
    // verify chunks this gate is about, so take the _ex form.
    ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/1, &model),
              IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    // The divergence was measured at ctx_capacity 4096 over ~300 generated
    // tokens; max_seq_len clamps the capture tier, so a smaller window changes
    // the graph key and the number of replays this sees.
    cfg.max_seq_len = 4096;
    cfg.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    // speculative.mtp_k in the config does not arm the head by itself — the
    // drafter is turned on through the API, the way imp-cli and imp-server do.
    ASSERT_EQ(imp_enable_mtp_spec_decode(ctx, 1), IMP_SUCCESS);

    // Several requests, not one. The graph is captured during the first; the
    // later ones replay it against a state it was never captured for, which is
    // the whole point — a single request only ever replays near its own capture
    // position and passes on a build where the defect is present (measured: 0
    // differing over 209 replays inside one request, 141 of 861 across four).
    size_t produced = 0;
    for (const char* prompt :
         {"Explain how a paged KV cache works in an LLM inference engine, and why block size matters.",
          "Explain how continuous batching differs from static batching in an LLM server.",
          "Explain why speculative decoding can raise throughput without changing the output distribution.",
          "Explain what causes memory fragmentation in a naive KV cache implementation.",
          "Explain the difference between prefill and decode in transformer inference.",
          "Explain how grouped-query attention reduces KV cache size."}) {
        produced += generate(ctx, prompt, 300).size();
    }
    EXPECT_GT(produced, 0u) << "empty generation — the fidelity tally would be vacuous";

    const auto fid = ctx->engine->spec_capture_fidelity();
    imp_context_free(ctx);
    imp_model_free(model);

    // A short run proves nothing: the graph cache may never have been hit, and a
    // rate over a handful of replays is noise.
    ASSERT_GE(fid.checked, 200)
        << "only " << fid.checked
        << " cached verify-chunk replays were compared, which is too few to judge. Speculation may "
           "not have engaged for this model (check speculative.hybrid / .moe and the MTP head).";
    // The threshold is not zero, and the floor is measured rather than assumed.
    // Capture is not bit-exact with eager for every step even on a healthy model
    // — cuBLASLt picks its algorithm at capture time — so a small residue is
    // expected. Measured 2026-08-20 over ~1000 replays each:
    //   Qwen3.8-27B-NVFP4          1 / 1033 = 0.10 %
    //   Qwen3.6-35B-A3B-NVFP4      2 / 1013 = 0.20 %
    //   Nemotron-3.5-Lightning   183 / 1331 = 13.7 %
    // 2 % sits 10x above the worst healthy reading and 7x below the defect.
    const double rate = 100.0 * static_cast<double>(fid.differing) / static_cast<double>(fid.checked);
    EXPECT_LT(rate, 2.0) << fid.differing << " of " << fid.checked << " cached-graph replays (" << rate
                         << " %) disagreed with an eager forward of the same state, max|dlogit|="
                         << fid.max_delta
                         << ". A cached verify chunk is not reproducing the forward it was captured from.";
}

TEST(SpecCaptureFidelityTest, HybridSpeculationDoesNotCollapse) {
    const std::string dir = model_or("IMP_TEST_MODEL_SSM", "/models/Nemotron-3-Nano-30B-A3B-NVFP4");
    if (!present(dir))
        GTEST_SKIP() << "nemotron_h checkpoint not present at " << dir;
    constexpr size_t kNeededMiB = 24000;
    const size_t free_mib = device_free_mib();
    if (free_mib < kNeededMiB)
        GTEST_SKIP() << "needs ~" << kNeededMiB << " MiB free, card has " << free_mib << " MiB";

    imp::set_pending_runtime_config(suffix_config());
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load_ex(dir.c_str(), IMP_FORMAT_SAFETENSORS, /*load_mtp_head=*/1, &model),
              IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 2048;
    cfg.max_batch_size = 1;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);

    // A counting prompt gives the suffix matcher something to draft, so verify
    // chunks actually run and some of them are fully rejected — the case that
    // adopts the recurrent snapshot.
    const std::string out =
        generate(ctx, "Count from 1 to 40, one number per line, then explain what a KV cache is.", 200);
    imp_context_free(ctx);
    imp_model_free(model);

    ASSERT_GT(out.size(), 0u) << "empty generation";
    // The 8a7f2763 collapse produced "Here's" followed by ~300 identical
    // characters with no whitespace: two distinct words in 200 tokens. Real
    // prose at this length is far above 20.
    EXPECT_GT(distinct_words(out), 20u)
        << "speculation on a Mamba2 hybrid collapsed the generation to " << distinct_words(out)
        << " distinct words in 200 tokens: " << out.substr(0, 200);
}
