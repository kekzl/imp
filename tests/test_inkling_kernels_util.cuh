#pragma once
// Shared by tests/test_inkling_kernels.cu and tests/bench/test_inkling_kernels_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "compute/gemm.h"
#include "compute/gemm_grouped_nvfp4_f16_tile.h"
#include "compute/gemv_i8b32.h"
#include "core/tensor.h"
#include "compute/inkling_ops.h"
#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"

namespace {
int32_t* group_scratch() {
    static int32_t* p = nullptr;
    if (!p)
        cudaMalloc(&p, inkling::kMoeSmallGroupInts * sizeof(int32_t));
    return p;
}
}  // namespace
