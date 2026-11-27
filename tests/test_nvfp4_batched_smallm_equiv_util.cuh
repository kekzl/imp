#pragma once
// Shared by tests/test_nvfp4_batched_smallm_equiv.cu and
// tests/bench/test_nvfp4_batched_smallm_equiv_bench.cu.
#include <gtest/gtest.h>

#include "core/tensor.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool has_sm120() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess)
        return false;
    return p.major == 12;
}

// Shapes from Qwen3.8-27B's GDN layers, which are the ones the verify chunk
// routes: d_model 5120 into the in/out projections. K stays a multiple of 16.
constexpr int kK = 5120;
constexpr int kN = 512;  // one N tile; the kernel walks N in blocks anyway

struct Fixture {
    imp::NvFP4Owned w{};
    std::vector<half> w_fp16;  // dequantised reference weight, [kN, kK]
};

// Quantise a random weight and keep its dequantised form as the reference.
void build(Fixture& f) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    std::vector<half> host(static_cast<size_t>(kN) * kK);
    for (auto& h : host)
        h = __float2half(dist(rng));

    void* d_w = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w, host.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    int64_t shape[2] = {kN, kK};
    imp::Tensor wt(d_w, imp::QType::F16, 2, shape, /*on_device=*/true);
    imp::quantize_fp16_to_nvfp4(wt, f.w, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // The reference multiplies the weight the kernel actually holds, not the
    // pre-quantisation one: this test gates the GEMV, not the quantiser.
    void* d_deq = nullptr;
    ASSERT_EQ(cudaMalloc(&d_deq, host.size() * sizeof(half)), cudaSuccess);
    imp::dequantize_nvfp4_to_fp16(f.w, d_deq, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    f.w_fp16.resize(host.size());
    ASSERT_EQ(cudaMemcpy(f.w_fp16.data(), d_deq, host.size() * sizeof(half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_deq);
    cudaFree(d_w);
}

class BatchedSmallM : public ::testing::Test {
protected:
    void SetUp() override {
        if (!has_sm120())
            GTEST_SKIP() << "sm_120 required";
        build(f_);
    }
    Fixture f_;
};

}  // namespace
