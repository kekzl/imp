#pragma once
// Shared by tests/test_fp8_proj_stream.cu and tests/bench/test_fp8_proj_stream_bench.cu.
#include <gtest/gtest.h>
#include "compute/gemm.h"
#include "quant/fp8_quant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace imp {
namespace {

class FP8ProjStreamTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

}  // namespace
}  // namespace imp
