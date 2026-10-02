// #2446: a failed cudaMallocAsync in the NVFP4 MoE smallM prefill (host-args path, device args off)
// must not launch a kernel on a null buffer. Injection: test_alloc_inject.h. Death test: a fault
// poisons only the child.
#include <gtest/gtest.h>
#include "imp/imp.h"
#include "api/imp_internal.h"
#include "compute/dispatch_record.h"
#include "core/cuda_errors.h"
#include "runtime/config.h"
#include "test_alloc_inject.h"
#include "test_models.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <vector>

namespace {

const char* model_path() {
    return imp_test::env_cstr_or(imp_test::kEnvModelNvfp4Moe, "/models/Qwen3-30B-A3B-NVFP4-Modelopt");
}

bool model_exists() {
    struct stat st{};
    return ::stat(model_path(), &st) == 0;
}

constexpr int kChildOk = 42;

[[noreturn]] void die(int code, const char* what) {
    std::fprintf(stderr, "child: %s\n", what);
    std::_Exit(code);
}

// Greedy prefill; on success *tok is the sampled first token.
ImpError greedy_prefill(ImpContext ctx, const std::vector<int32_t>& tokens, int n, int32_t* tok) {
    ImpGenerateParams gp = imp_generate_params_default();
    gp.temperature = 0.0f;
    const ImpError e = imp_prefill_with_params(ctx, tokens.data(), n, &gp);
    if (e == IMP_SUCCESS && imp_prefill_token(ctx, tok) != IMP_SUCCESS)
        return IMP_ERROR_INTERNAL;
    return e;
}

[[noreturn]] void run_smallm_prefill_under_alloc_failure() {
    imp::set_pending_runtime_config(
        imp::RuntimeConfig::load("", {"moe.nvfp4_smallM=true", "moe.nvfp4_device_args=false"}));
    ImpModel model = nullptr;
    if (imp_model_load(model_path(), IMP_FORMAT_SAFETENSORS, &model) != IMP_SUCCESS)
        die(3, "model load failed");
    ImpConfig config = imp_config_default();
    config.max_seq_len = 1024;
    config.max_batch_size = 1;
    ImpContext ctx = nullptr;
    if (imp_context_create(model, &config, &ctx) != IMP_SUCCESS)
        die(4, "context create failed");

    std::vector<int32_t> tokens(256);
    int n = 0;
    if (imp_tokenize(model, "The quick brown fox jumps over the lazy dog.", tokens.data(), &n, 256) !=
            IMP_SUCCESS ||
        n <= 1)
        die(5, "tokenize failed");

    // Control: without injection this prompt takes the smallM branch; its greedy token is the reference.
    imp::dispatch_record::reset();
    int32_t ref_tok = -1;
    if (greedy_prefill(ctx, tokens, n, &ref_tok) != IMP_SUCCESS)
        die(6, "uninjected prefill failed");
    if (imp::dispatch_record::current().moe_prefill_tier != imp::MoePrefillPath::SMALL_M)
        die(7, "control: the uninjected prefill did not take the smallM branch");
    if (imp_context_reset(ctx) != IMP_SUCCESS)
        die(11, "context reset failed");

    // Injected: either a clean error or the reference token, never a fault or a silent wrong token.
    cudaMemPool_t old_pool = imp_test::exhaust_async_pool();
    if (!old_pool)
        die(12, "injection inactive: a cudaMallocAsync succeeded on the exhausted pool");
    int32_t tok = -1;
    const ImpError err = greedy_prefill(ctx, tokens, n, &tok);
    const cudaError_t sync = cudaDeviceSynchronize();
    std::fprintf(stderr, "child: injected prefill -> ImpError %d, token %d (ref %d), sync %s\n",
                 static_cast<int>(err), tok, ref_tok, cudaGetErrorName(sync));
    if (imp::cuda_error_is_unrecoverable(sync))
        die(8, "injected prefill faulted the context");
    if (err == IMP_SUCCESS && tok != ref_tok)
        die(13, "injected prefill reported success with a different greedy token");
    (void)cudaGetLastError();

    // The engine stays usable once allocations succeed again.
    int dev = 0;
    (void)cudaGetDevice(&dev);
    if (cudaDeviceSetMemPool(dev, old_pool) != cudaSuccess)
        die(9, "pool restore failed");
    if (imp_context_reset(ctx) != IMP_SUCCESS)
        die(11, "context reset failed");
    tok = -1;
    if (greedy_prefill(ctx, tokens, n, &tok) != IMP_SUCCESS || cudaDeviceSynchronize() != cudaSuccess ||
        tok != ref_tok)
        die(10, "prefill after the injection failed or changed the greedy token");

    std::fprintf(stderr, "alloc-failure contract held\n");
    std::_Exit(kChildOk);
}

TEST(MoeAllocFailureTest, SmallMPrefillSurvivesFailedAsyncAlloc) {
    ASSERT_NO_FATAL_FAILURE(imp_test::require_readable_if_set(imp_test::kEnvModelNvfp4Moe));
    if (!model_exists())
        GTEST_SKIP() << "Model not found: " << model_path();
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(run_smallm_prefill_under_alloc_failure(), ::testing::ExitedWithCode(kChildOk),
                "alloc-failure contract held");
}

}  // namespace
