#pragma once
// Shared by tests/test_gemm_capture_fp16_sm120.cu and tests/bench/test_gemm_capture_fp16_sm120_bench.cu.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "compute/gemm_capture_fp16_sm120.h"
#include "compute/gemm.h"
#include "core/tensor.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace imp {
namespace {

struct DeviceFp16 {
    __half* data = nullptr;
    int64_t numel = 0;
    explicit DeviceFp16(int64_t n) : numel(n) { cudaMalloc(&data, n * sizeof(__half)); }
    ~DeviceFp16() {
        if (data)
            cudaFree(data);
    }
    DeviceFp16(const DeviceFp16&) = delete;
    DeviceFp16& operator=(const DeviceFp16&) = delete;
};

void upload_fp16(__half* dst, const std::vector<float>& src) {
    std::vector<__half> h(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        h[i] = __float2half(src[i]);
    cudaMemcpy(dst, h.data(), src.size() * sizeof(__half), cudaMemcpyHostToDevice);
}

class GemmCaptureSm120 : public ::testing::Test {
protected:
    void SetUp() override {
        if (!capture_gemm_fp16_sm120_available())
            GTEST_SKIP() << "Not sm_120+";
    }
};

}  // namespace
}  // namespace imp
