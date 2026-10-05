// Dense prefill SwiGLU + CUTLASS NVFP4 quantize (#2470): FP16 output, packed FP4 and SfAtom
// scales byte-identical to swiglu() followed by quantize_fp16_to_nvfp4_cutlass().

#include <gtest/gtest.h>
#include "compute/activation.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace imp {
namespace {

struct Buf {
    void* p = nullptr;
    size_t n = 0;
    explicit Buf(size_t bytes) : n(bytes) {
        EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemset(p, 0, bytes), cudaSuccess);
    }
    ~Buf() { cudaFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> h(n);
        EXPECT_EQ(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    }
};

// M crosses a 128-row SfAtom tile and is not a multiple of it; K = Qwen3-14B d_ff.
void run_case(int M, int K, float range) {
    const size_t n = static_cast<size_t>(M) * K;
    std::vector<half> g(n), u(n);
    uint64_t s = 0x2470u + static_cast<uint64_t>(M);
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        g[i] = __float2half(range * (static_cast<float>(s >> 40) / 16777216.0f * 2.0f - 1.0f));
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        u[i] = __float2half(range * (static_cast<float>(s >> 40) / 16777216.0f * 2.0f - 1.0f));
    }
    Buf dg(n * 2), du(n * 2), out_ref(n * 2), out_fused(n * 2);
    ASSERT_EQ(cudaMemcpy(dg.p, g.data(), n * 2, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(du.p, u.data(), n * 2, cudaMemcpyHostToDevice), cudaSuccess);
    const size_t sf = cutlass_nvfp4_sf_size(M, K);
    Buf pk_ref(n / 2), sf_ref(sf), pk_fused(n / 2), sf_fused(sf);

    int64_t shape[2] = {M, K};
    Tensor tg(dg.p, QType::F16, 2, shape, true), tu(du.p, QType::F16, 2, shape, true),
        to(out_ref.p, QType::F16, 2, shape, true);
    swiglu(tg, tu, to, nullptr);
    quantize_fp16_to_nvfp4_cutlass(out_ref.p, pk_ref.p, sf_ref.p, M, K, nullptr);
    swiglu_quantize_fp16_to_nvfp4_cutlass(dg.p, du.p, out_fused.p, pk_fused.p, sf_fused.p, M, K, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(out_ref.bytes(), out_fused.bytes()) << "M=" << M;
    EXPECT_EQ(pk_ref.bytes(), pk_fused.bytes()) << "M=" << M;
    EXPECT_EQ(sf_ref.bytes(), sf_fused.bytes()) << "M=" << M;
}

TEST(SwigluQuantizeCutlass, ByteIdenticalToSwigluThenQuantize) {
    run_case(33, 17408, 8.0f);
    run_case(200, 17408, 8.0f);
    run_case(129, 1024, 40.0f);
}

}  // namespace
}  // namespace imp
