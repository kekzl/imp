#pragma once
// Shared by tests/test_nvfp4_smallm_v2.cu and tests/bench/test_nvfp4_smallm_v2_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_owned.h"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

namespace {

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct DeviceQuant {
    imp::NvFP4Owned q{};
    std::vector<uint8_t> packed_h, scales_h;
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
        packed_h.resize((size_t)rows * K / 2);
        scales_h.resize((size_t)rows * K / 16);
        ASSERT_EQ(cudaMemcpy(packed_h.data(), q.packed_data, packed_h.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(scales_h.data(), q.micro_scales, scales_h.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
    }
};

class NvFP4SmallMV2Test : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_available())
            GTEST_SKIP() << "no CUDA device";
    }
};

}  // namespace
