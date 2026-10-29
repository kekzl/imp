// Every weight form a load allocates is named in the VRAM audit: an unnamed one reads as library
// reserve, which the reserve cache charges again on the next start. Qwen3.5-4B MXFP4 GGUF: the
// MXFP4 FP16 decode bulk (8020 MiB) and the GDN NVFP4 prefill copy were unnamed, reserve 6612 MiB.
// GTEST_SKIPs without the model. Native MXFP4 (the default) holds neither copy: second test.

#include "imp/imp.h"
#include "runtime/config.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <regex>
#include <string>

namespace {

constexpr const char kModel[] = "/models/Qwen3.5-4B-mxfp4.gguf";

double pool_mib(const std::string& log, const std::string& pool) {
    std::smatch m;
    if (!std::regex_search(log, m, std::regex(pool + R"( +([0-9.]+) +[0-9.]+ +[0-9]+)")))
        return -1.0;
    return std::stod(m[1].str());
}

TEST(VramLedgerE2E, Mxfp4GdnWeightFormsAreNamed) {
    if (!std::filesystem::exists(kModel))
        GTEST_SKIP() << "model not present at " << kModel;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || (free_b >> 20) < 16000)
        GTEST_SKIP() << "needs ~16000 MiB free";
    imp::RuntimeConfig rc;
    rc.diagnostics.vram_audit = true;
    rc.attention.mxfp4_fp16_fallback =
        true;  // the FP16-cache path; native is the default since the W4A16 prefill
    imp::set_pending_runtime_config(rc);
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_model_load(kModel, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 512;
    cfg.max_batch_size = 1;
    testing::internal::CaptureStdout();
    const ImpError rc_ctx = imp_context_create(model, &cfg, &ctx);
    const std::string log = testing::internal::GetCapturedStdout();
    ASSERT_EQ(rc_ctx, IMP_SUCCESS) << log;
    EXPECT_GT(pool_mib(log, "WEIGHT_CACHE_MXFP4_FP16"), 0.0) << "the MXFP4 FP16 bulk is unnamed";
    EXPECT_GT(pool_mib(log, "WEIGHT_CACHE_GDN_PREFILL"), 0.0) << "the GDN NVFP4 prefill copy is unnamed";
    std::smatch m;
    ASSERT_TRUE(std::regex_search(log, m, std::regex(R"(named: library reserve +([0-9.]+))"))) << log;
    EXPECT_LT(std::stod(m[1].str()), 2048.0) << "unnamed weight bytes still read as library reserve";
    imp_context_free(ctx);
    imp_model_free(model);
}

// Default native MXFP4: no FP16 decode bulk, the W4A16 GEMM scratch is named instead (one weight as FP16).
TEST(VramLedgerE2E, Mxfp4GdnNativeHoldsNoFp16Cache) {
    if (!std::filesystem::exists(kModel))
        GTEST_SKIP() << "model not present at " << kModel;
    imp::RuntimeConfig rc;
    rc.diagnostics.vram_audit = true;
    imp::set_pending_runtime_config(rc);
    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_model_load(kModel, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 512;
    cfg.max_batch_size = 1;
    testing::internal::CaptureStdout();
    const ImpError rc_ctx = imp_context_create(model, &cfg, &ctx);
    const std::string log = testing::internal::GetCapturedStdout();
    ASSERT_EQ(rc_ctx, IMP_SUCCESS) << log;
    // <= 0: the account is process-wide, the fallback test above leaves the pool listed at 0 MiB.
    EXPECT_LE(pool_mib(log, "WEIGHT_CACHE_MXFP4_FP16"), 0.0) << "native MXFP4 built the FP16 decode bulk";
    const double wdeq = pool_mib(log, "mxfp4_wdeq");
    EXPECT_GT(wdeq, 0.0) << "the W4A16 dequant scratch is unnamed";
    EXPECT_LT(wdeq, 256.0) << "the W4A16 scratch holds more than one weight";
    imp_context_free(ctx);
    imp_model_free(model);
}

}  // namespace
