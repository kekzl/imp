#pragma once
// Shared by tests/test_moe_router_fused.cu and tests/bench/test_moe_router_fused_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/moe_router_fused.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

}  // namespace
