// #2396: a cold cuBLASLt shape first called inside a stream capture must not run the algo probe
// there. M=47 K=5120 N=48 FP16 is the Qwen3.8-27B GDN alpha/beta shape at batch 47: M < 64 skips
// the capture-safe WMMA kernel, so the call reaches cuBLASLt with the bench scratch present.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "compute/gemm.h"
#include "compute/gemm_internal.cuh"
#include "core/tensor.h"
#include "memory/backend.h"
#include "memory/engine_arena.h"
#include "test_cuda_skip.h"

namespace imp {
void gemm_reset_static_cuda_state();  // gemm.cpp, the pre-cudaDeviceReset hook
}  // namespace imp

using namespace imp;

namespace {

// A binary-wide 64 MiB arena may be up (another file registers it): swap in one that holds the
// 64 MiB workspace + 32 MiB bench scratch, put the standing one back after (test_rowwise_topm).
struct ProbeArena {
    bool had = engine_arena().is_open();
    size_t had_bytes = had ? engine_arena().capacity() : 0;
    bool ok = false;
    ProbeArena() {
        gemm_reset_static_cuda_state();  // statics may point into the standing arena
        if (had)
            engine_arena_close();
        ok = engine_arena_open(cuda_malloc_backend(), 160ull << 20) == MemError::Ok;
    }
    ~ProbeArena() {
        gemm_reset_static_cuda_state();  // statics point into the arena closing here
        if (ok)
            engine_arena_close();
        if (had)
            (void)engine_arena_open(cuda_malloc_backend(), had_bytes);
    }
};

}  // namespace

// #2611: the test listener rewinds the arena between tests; gemm_init() kept the old workspace,
// the next tenant got the same bytes (in test-quant: the bench scratch, cuBLASLt hung).
TEST(GemmCaptureProbe, WorkspaceRetakenAfterArenaReset) {
    SKIP_IF_NO_CUDA();
    ProbeArena arena;
    ASSERT_TRUE(arena.ok);
    gemm_init();
    ASSERT_NE(gemm_internal_workspace(), nullptr);
    engine_arena().reset();
    gemm_init();
    const auto* ws = static_cast<const std::byte*>(gemm_internal_workspace());
    const size_t ws_bytes = gemm_internal_workspace_size();
    ASSERT_NE(ws, nullptr);
    auto next = engine_arena().take_bytes(1ull << 20);
    ASSERT_FALSE(next.empty());
    const bool overlaps = next.data() < ws + ws_bytes && ws < next.data() + next.size();
    EXPECT_FALSE(overlaps) << "gemm workspace still points into a slice the reset arena handed out again";
}

TEST(GemmCaptureProbe, ColdShapeInsideCaptureKeepsCaptureValid) {
    SKIP_IF_NO_CUDA();
    constexpr int64_t M = 47, K = 5120, N = 48;
    {
        ProbeArena arena;
        ASSERT_TRUE(arena.ok);
        gemm_init();

        std::vector<__half> hA(M * K), hB(N * K);
        for (size_t i = 0; i < hA.size(); i++)
            hA[i] = __float2half(static_cast<float>(i % 7) * 0.01f);
        for (size_t i = 0; i < hB.size(); i++)
            hB[i] = __float2half(static_cast<float>(i % 5) * 0.01f);
        __half *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
        ASSERT_EQ(cudaMalloc(&dA, hA.size() * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dB, hB.size() * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dC, M * N * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&dRef, M * N * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemset(dC, 0, M * N * sizeof(__half)), cudaSuccess);

        const int64_t sa[2] = {M, K}, sb[2] = {N, K}, sc[2] = {M, N};
        Tensor A(dA, QType::F16, 2, sa, true), B(dB, QType::F16, 2, sb, true);
        Tensor C(dC, QType::F16, 2, sc, true), Ref(dRef, QType::F16, 2, sc, true);

        cudaStream_t s = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), cudaSuccess);
        gemm(A, B, C, 1.0f, 0.0f, s);  // first call at this shape: entry created under capture
        cudaGraph_t g = nullptr;
        ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess) << "the probe invalidated the capture";
        ASSERT_NE(g, nullptr);
        cudaGraphExec_t ge = nullptr;
        ASSERT_EQ(cudaGraphInstantiate(&ge, g, 0), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(ge, s), cudaSuccess);
        gemm(A, B, Ref, 1.0f, 0.0f, s);  // eager, same cache entry: the same algo
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);

        std::vector<__half> hC(M * N), hRef(M * N);
        ASSERT_EQ(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(hRef.data(), dRef, hRef.size() * sizeof(__half), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(std::memcmp(hC.data(), hRef.data(), hC.size() * sizeof(__half)), 0)
            << "captured and eager GEMM disagree: different algos behind one cache entry";
        float c0 = __half2float(hRef[0]);
        EXPECT_NE(c0, 0.0f);

        (void)cudaGraphExecDestroy(ge);
        (void)cudaGraphDestroy(g);
        (void)cudaStreamDestroy(s);
        (void)cudaFree(dA);
        (void)cudaFree(dB);
        (void)cudaFree(dC);
        (void)cudaFree(dRef);
    }
}
