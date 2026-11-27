#pragma once
// Shared by tests/test_nvfp4_smallm.cu and tests/bench/test_nvfp4_smallm_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

class NvFP4SmallMTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "no CUDA device";
    }
};

// Quantize an FP16 host matrix on device into an NvFP4Owned (released with the buffer).
struct QuantBuf {
    imp::NvFP4Owned q{};
    void quantize(const std::vector<__half>& src, int rows, int K) {
        void* d = nullptr;
        ASSERT_EQ(cudaMalloc(&d, src.size() * sizeof(__half)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d, src.data(), src.size() * sizeof(__half), cudaMemcpyHostToDevice),
                  cudaSuccess);
        int64_t shp[2] = {rows, K};
        imp::Tensor t(d, imp::QType::F16, 2, shp, /*on_device=*/true);
        imp::quantize_fp16_to_nvfp4(t, q, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_NE(q.packed_data, nullptr);
        cudaFree(d);
    }
};

}  // namespace
