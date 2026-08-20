// =============================================================================
// test_nvfp4_prefill_cache_knob.cpp — a decode-only knob must not touch prefill
// =============================================================================
//
// diagnostics.no_nvfp4_decode_cache is declared a decode-side bisection tool
// (core/dispatch_policy.h): "decode runs on the source-precision paths". It
// used to return early from pre_dequant_phase3_nvfp4_decode_ before
// nvfp4_decode_convert_cutlass_ ran, and that function populates
// wcache_->cutlass_nvfp4 — the PREFILL cache that infer_tier_from_wcache
// (exec/pre_dequant_internal.h) reads to set prefill_tier. So the knob silently
// moved every native-NVFP4 dense weight's prefill from W4A4 (the CUTLASS kernel
// quantizes the activation too) to W4A16, and moved a teacher-forced perplexity
// on NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 from 9.165 to 9.051.
//
// The invariant this pins needs only a native-NVFP4 checkpoint that has a
// CUTLASS conversion at all — not the Mamba2/recurrent case — so it runs on the
// smallest such model on the box rather than a 19 GB hybrid.
//
// GTEST_SKIPs when the model is absent, like the other SafeTensors suites.
// =============================================================================

#include "model/model.h"
#include "model/safetensors_loader.h"
#include "runtime/config.h"
#include "runtime/engine.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace {

// Container bind-mount path (-v $(HOME)/models:/models), as in the sibling suites.
constexpr const char kNvfp4ModelDir[] = "/models/Qwen3-8B-NVFP4-cortecs";

// The weight upload is ~6 GiB; leave room for the caches this test is about.
constexpr size_t kNeededMiB = 12000;

size_t device_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b >> 20;
}

imp::EngineConfig knob_engine_config() {
    imp::EngineConfig cfg;
    cfg.max_batch_size = 1;
    cfg.max_seq_len = 256;
    cfg.use_cuda_graphs = false;
    cfg.use_pdl = false;
    cfg.use_fp8_prefill = false;
    // Mode 2 ("NVFP4 only") is what the resolver picks for a native-NVFP4
    // model; pin it so the phase under test actually runs.
    cfg.use_nvfp4_decode = 2;
    cfg.kv_cache_dtype = imp::QType::F16;
    cfg.compute_dtype = imp::QType::F16;
    cfg.kv_block_size = 16;
    cfg.use_green_contexts = false;
    cfg.gpu_layers = -1;
    cfg.use_prefix_caching = false;
    cfg.use_mxfp4_prefill = false;
    cfg.dual_path_quant = false;
    return cfg;
}

}  // namespace

TEST(Nvfp4PrefillCacheKnobTest, DecodeKnobKeepsTheCutlassPrefillCache) {
    if (!fs::exists(std::string(kNvfp4ModelDir) + "/config.json"))
        GTEST_SKIP() << "native-NVFP4 checkpoint not present at " << kNvfp4ModelDir;
    const size_t free_mib = device_free_mib();
    if (free_mib < kNeededMiB)
        GTEST_SKIP() << "needs ~" << kNeededMiB << " MiB free, card has " << free_mib << " MiB";

    imp::RuntimeConfig rc;
    rc.diagnostics.no_nvfp4_decode_cache = true;
    imp::set_pending_runtime_config(rc);

    std::shared_ptr<imp::Model> model = imp::load_safetensors(kNvfp4ModelDir, /*load_mtp_head=*/false);
    ASSERT_NE(model, nullptr);
    ASSERT_TRUE(model->upload_weights_gpu(imp::QType::F16, nullptr, 1ULL << 30));

    // The pre-dequant pipeline runs inside Engine::init. INFO goes to stdout.
    testing::internal::CaptureStdout();
    imp::Engine engine;
    const bool ok = engine.init(model, knob_engine_config());
    const std::string log = testing::internal::GetCapturedStdout();
    ASSERT_TRUE(ok);

    EXPECT_NE(log.find("CUTLASS sm_120 NVFP4 weight cache:"), std::string::npos)
        << "diagnostics.no_nvfp4_decode_cache skipped the CUTLASS *prefill* conversion. "
           "It is declared decode-only, and infer_tier_from_wcache reads cutlass_nvfp4 to "
           "set prefill_tier, so this silently changes prefill numerics. Captured log:\n"
        << log;

    // The knob must still do its own job: no decode-cache build.
    EXPECT_NE(log.find("NVFP4 decode cache DISABLED"), std::string::npos)
        << "the knob stopped disabling the decode cache — this test would then pass "
           "for the wrong reason. Captured log:\n"
        << log;
}
