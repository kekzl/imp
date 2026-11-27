#pragma once
// Shared by tests/test_mmq_q8_imma_tail.cu and tests/bench/test_mmq_q8_imma_tail_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "compute/mmq_q8_imma.h"
#include "scoped_engine_arena.h"

namespace imp {
namespace {

constexpr int kBlk = 32;
constexpr int kBlockBytes = 34;  // half d + 32 s8

void gen_q8_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    ASSERT_EQ(K % kBlk, 0);
    const int bpr = K / kBlk;
    W.resize(static_cast<size_t>(N) * bpr * kBlockBytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    for (size_t b = 0; b < W.size() / kBlockBytes; ++b) {
        uint8_t* bp = W.data() + b * kBlockBytes;
        __half d = __float2half(dd(rng));
        std::memcpy(bp, &d, 2);
        for (int i = 0; i < 32; ++i)
            bp[2 + i] = static_cast<uint8_t>(static_cast<int8_t>(qd(rng)));
    }
}

}  // namespace
}  // namespace imp
