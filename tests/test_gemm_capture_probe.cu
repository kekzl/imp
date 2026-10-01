// #2396: a cold cuBLASLt shape first called inside a stream capture must not run the algo probe
// there. M=47 K=5120 N=48 FP16 is the Qwen3.8-27B GDN alpha/beta shape at batch 47: M < 64 skips
// the capture-safe WMMA kernel, so the call reaches cuBLASLt with the bench scratch present.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "compute/gemm.h"
#include "core/tensor.h"
#include "scoped_engine_arena.h"
#include "test_cuda_skip.h"

namespace imp {
void gemm_reset_static_cuda_state();  // gemm.cpp, the pre-cudaDeviceReset hook
}  // namespace imp

using namespace imp;

TEST(GemmCaptureProbe, ColdShapeInsideCaptureKeepsCaptureValid) {
    SKIP_IF_NO_CUDA();
    constexpr int64_t M = 47, K = 5120, N = 48;
    gemm_reset_static_cuda_state();  // drop statics a previous test set without an arena
    {
        ScopedEngineArena arena(160ull << 20);  // 64 MiB workspace + 32 MiB bench scratch
        ASSERT_TRUE(arena.opened());
        // Statics point into the arena: drop them before it closes, also on an ASSERT return.
        struct ResetGemmStatics {
            ~ResetGemmStatics() { gemm_reset_static_cuda_state(); }
        } reset_gemm_statics;
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
