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
#include "api/imp_internal.h"
#include "test_models.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

static const char* get_model_path() {
    return imp_test::env_cstr_or(imp_test::kEnvModel, "/models/Qwen3-8B-Q8_0.gguf");
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

    // Pool "used" is a PROCESS-wide counter, so an absolute bound on it is only
    // true when this test runs first. It did not: in the full test-e2e run
    // earlier tests leave blocks in the default pool and the figure read 1792
    // MiB against a 1024 MiB bound, while the same test passed in isolation.
    // That is the same category error the comment below rejects for the DEVICE
    // figure, one level down — so take a baseline and assert on the delta,
    // which is the part this cycle actually owns.
    unsigned long long pool_used_before = 0;
    {
        cudaMemPool_t p0 = nullptr;
        int d0 = 0;
        cudaGetDevice(&d0);
        if (cudaDeviceGetDefaultMemPool(&p0, d0) == cudaSuccess)
            (void)cudaMemPoolGetAttribute(p0, cudaMemPoolAttrUsedMemCurrent, &pool_used_before);
    }

    run_one_cycle(get_model_path());
    if (::testing::Test::HasFatalFailure())
        return;

    // Teardown must hand the weights back to the default mempool rather than
    // parking them there — that is #507's actual regression, and #834's fix
    // (cudaFreeAsync, not cudaFree) is what makes the trim able to reclaim
    // them. Assert it at POOL level, which is the level at which it is true.
    //
    // This used to assert at device level instead, by cudaMalloc'ing the
    // apparently-missing amount and treating success as proof that the memory
    // was merely under-reported. That check is unsound on WSL2/WDDM: the
    // driver oversubscribes into host memory and returns cudaSuccess, so the
    // probe passes whether or not the memory is really there (AUDIT G18).
    // Measured: a 28 GiB allocation succeeds on a 32 GB card with 22.6 GiB
    // reported free, and runs at 237 GB/s against 1531 GB/s resident.
    //
    // The device-level figure genuinely does not return here — WSL2/WDDM keeps
    // a process's peak commitment for the process lifetime, no matter what the
    // pool does (AUDIT B36) — so asserting on it would encode a platform
    // property as a leak. What still guards #507 is the second full cycle
    // below: the next load must succeed.
    size_t free_between = device_free_mib();
    cudaMemPool_t pool = nullptr;
    int dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
        // UsedMemCurrent, not ReservedMemCurrent. Reserved drops to 0 on the
        // trim even when the blocks were never returned — verified by
        // reverting #834 (cudaFree instead of cudaFreeAsync): reserved still
        // reads 0 while used stays at the full weight footprint and climbs to
        // 16600 MiB on the second cycle, which is the exact signature #834
        // recorded. Asserting on reserved would have passed straight through
        // that regression.
        unsigned long long used = 0;
        ASSERT_EQ(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used), cudaSuccess);
        const unsigned long long retained = used > pool_used_before ? used - pool_used_before : 0;
        EXPECT_LT(retained >> 20, 1024u)
            << "this cycle left " << (retained >> 20) << " MiB in the default mempool as USED ("
            << (pool_used_before >> 20) << " -> " << (used >> 20)
            << " MiB) — the weights were freed with an API that does not return stream-ordered "
            << "blocks to the pool (#507/#834), so the trim can reclaim nothing. Device-reported "
            << "free went " << free_before << " -> " << free_between << " MiB, which is expected "
            << "on this platform (AUDIT B36) and is NOT what this assertion is about.";
    }

    // Re-init after inference: before the prewarm stream-rebind fix this
    // segfaulted (dangling stream on the global attention-cuBLAS handle).
    run_one_cycle(get_model_path());
}

// #830: a SECOND engine on the SAME model handle (load once, create/free/create
// context) must not CUDA-IMA. Some models free their source weight tensors for
// VRAM during the first engine's pre-dequant (Phase-4b), leaving dangling
// pointers a second build would read. The engine now rejects that up front
// (clean error), while models that never drop sources (dense Q8_0) still support
// create/free/create. Either outcome is acceptable here — the invariant is "no
// illegal access / no crash", and the process stays usable afterward.
TEST(EngineRelaunchTest, SecondEngineOnSameModelHandleNeverIMAs) {
    SKIP_IF_NO_MODEL();

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(get_model_path(), IMP_FORMAT_GGUF, &model), IMP_SUCCESS);

    // Lifecycle diagnostic (#830): dump the model's key tensor pointers/qtypes
    // at each phase so a mutation by engine #1 (which engine #2 then reads as
    // dangling) shows up as a diff in the test log.
    auto dump_state = [&](const char* tag) {
        imp::Model* m = model->model.get();
        fprintf(stderr,
                "[TENSORDUMP %s] tok_emb{d=%p q=%d sc=%p dev=%d} out_proj{d=%p q=%d sc=%p} "
                "allocs=%zu consumed=%d\n",
                tag, m->tok_emb_.data, (int)m->tok_emb_.qtype, m->tok_emb_.scales,
                (int)m->tok_emb_.on_device, m->out_proj_.data, (int)m->out_proj_.qtype,
                m->out_proj_.scales, m->gpu_allocations_.size(), (int)m->sources_consumed());
        for (int i = 0; i < 2 && i < m->n_layers(); ++i) {
            const auto& L = m->layer(i);
            fprintf(stderr,
                    "[TENSORDUMP %s] L%d wq{d=%p q=%d sc=%p} w_gate{d=%p q=%d sc=%p} "
                    "w_up{d=%p q=%d} ssm_in{d=%p q=%d sc=%p} ssm_out{d=%p q=%d}\n",
                    tag, i, L.wq.data, (int)L.wq.qtype, L.wq.scales, L.w_gate.data,
                    (int)L.w_gate.qtype, L.w_gate.scales, L.w_up.data, (int)L.w_up.qtype,
                    L.ssm_in.data, (int)L.ssm_in.qtype, L.ssm_in.scales, L.ssm_out.data,
                    (int)L.ssm_out.qtype);
        }
    };
    dump_state("post-load");

    ImpConfig config = imp_config_default();
    config.max_seq_len = 1024;
    config.max_batch_size = 1;

    ImpGenerateParams params = imp_generate_params_default();
    params.seed = 42;
    params.max_tokens = 8;
    params.temperature = 0.7f;
    params.apply_chat_template = 1;

    // First engine on the handle: create, generate, free.
    ImpContext ctx1 = nullptr;
    ASSERT_EQ(imp_context_create(model, &config, &ctx1), IMP_SUCCESS);
    dump_state("post-ctx1-init");
    char buf1[1024] = {};
    size_t n1 = 0;
    EXPECT_EQ(imp_generate(ctx1, "Say hi.", &params, buf1, sizeof(buf1), &n1), IMP_SUCCESS);
    EXPECT_GT(n1, 0u);
    imp_context_free(ctx1);
    dump_state("post-ctx1-free");

    // Second engine on the SAME handle. Must return cleanly — success or a
    // plain error — never an illegal memory access.
    ImpContext ctx2 = nullptr;
    ImpError err = imp_context_create(model, &config, &ctx2);
    if (err == IMP_SUCCESS) {
        // Accepted (sources not dropped) → it must actually work.
        ASSERT_NE(ctx2, nullptr);
        char buf2[1024] = {};
        size_t n2 = 0;
        EXPECT_EQ(imp_generate(ctx2, "Say hi.", &params, buf2, sizeof(buf2), &n2), IMP_SUCCESS);
        EXPECT_GT(n2, 0u);
        imp_context_free(ctx2);
    } else {
        // Rejected (sources consumed) → clean error, no context handed out.
        EXPECT_EQ(ctx2, nullptr);
    }

    // The process must still be usable: a freshly LOADED model works regardless
    // of which branch above ran (proves the CUDA context was not poisoned).
    imp_model_free(model);
    run_one_cycle(get_model_path());
}
