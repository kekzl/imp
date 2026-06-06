// Regression tests for engine relaunch — the imp-server model auto-swap path.
//
// Two production bugs (server [auto-swap] Qwen3.6-35B → Gemma-4-31B, 2026-06):
//
//  1. SIGSEGV on engine re-init after inference: the process-global
//     attention-cuBLAS handle kept the destroyed engine's stream bound
//     (cublasSetStream in the batched-attention path). The next engine's
//     attention_cublas_prewarm() issued its dummy GemmBatchedEx on that
//     dangling stream → cuBLAS algo heuristics → cuStreamGetGreenCtx →
//     segfault inside libcuda. The server died with no error output
//     (exit 139); docker restart-policy masked it as connection drops.
//
//  2. VRAM starvation on swap: weights are allocated via cudaMallocAsync from
//     the device default mempool, whose release threshold Engine init raises
//     to UINT64_MAX. Freeing model+context parked ~weights-sized memory in
//     the pool instead of returning it to the driver — the next model load
//     (plain-cudaMalloc paths, cudaMemGetInfo-based sizing and the upload
//     oversubscription gate) saw ~1.5 GB free on a 32 GB card and failed
//     ("Failed to upload token embedding").
//
// Requires a real model on disk: IMP_TEST_MODEL or the default
// /models/Qwen3-8B-Q8_0.gguf, matching test_api_generate.cpp.

#include <gtest/gtest.h>
#include "imp/imp.h"
#include "test_models.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

static const char* get_model_path() {
    const char* p = std::getenv(imp_test::kEnvModel);
    return p ? p : "/models/Qwen3-8B-Q8_0.gguf";
}

static bool model_exists() {
    FILE* f = fopen(get_model_path(), "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

#define SKIP_IF_NO_MODEL()                                           \
    do {                                                             \
        if (!model_exists())                                         \
            GTEST_SKIP() << "Model not found: " << get_model_path(); \
    } while (0)

static size_t device_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b >> 20;
}

// One full lifecycle: load → create context → generate (binds the global
// attention-cuBLAS handle to this engine's stream) → free everything.
static void run_one_cycle(const char* path) {
    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(path, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    ImpConfig config = imp_config_default();
    config.max_seq_len = 1024;
    config.max_batch_size = 1;

    ImpContext ctx = nullptr;
    ImpError err = imp_context_create(model, &config, &ctx);
    if (err != IMP_SUCCESS) {
        imp_model_free(model);
        FAIL() << "Context creation failed: " << imp_error_string(err);
    }

    ImpGenerateParams params = imp_generate_params_default();
    params.seed = 42;
    params.max_tokens = 8;
    params.temperature = 0.7f;
    params.apply_chat_template = 1;

    char buf[1024] = {};
    size_t n = 0;
    EXPECT_EQ(imp_generate(ctx, "Say hi.", &params, buf, sizeof(buf), &n), IMP_SUCCESS);
    EXPECT_GT(n, 0u);

    imp_context_free(ctx);
    imp_model_free(model);
}

}  // namespace

TEST(EngineRelaunchTest, ReloadAfterInferenceReleasesVramAndDoesNotCrash) {
    SKIP_IF_NO_MODEL();

    size_t free_before = device_free_mib();
    ASSERT_GT(free_before, 0u);

    run_one_cycle(get_model_path());
    if (::testing::Test::HasFatalFailure())
        return;

    // Teardown must hand the weights back to the driver, not park them in the
    // default mempool: the next load sizes itself via cudaMemGetInfo and
    // uploads through plain-cudaMalloc-visible memory. Allow ~2 GiB slack for
    // persistent CUDA/cuBLAS context overhead from the first cycle.
    size_t free_between = device_free_mib();
    EXPECT_GT(free_between + 2048, free_before)
        << "teardown retained " << (free_before - free_between)
        << " MiB — default mempool not trimmed back to the driver";

    // Re-init after inference: before the prewarm stream-rebind fix this
    // segfaulted (dangling stream on the global attention-cuBLAS handle).
    run_one_cycle(get_model_path());
}
