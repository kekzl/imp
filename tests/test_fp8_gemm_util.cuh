#pragma once
// Shared by tests/test_fp8_gemm.cu and tests/bench/test_fp8_gemm_bench.cu.
#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include "quant/dequant_gpu.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace imp {
namespace {

class FP8GemmTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

}  // namespace
}  // namespace imp
