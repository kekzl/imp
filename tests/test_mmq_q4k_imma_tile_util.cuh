#pragma once
// Shared by tests/test_mmq_q4k_imma_tile_bench.cu and tests/bench/test_mmq_q4k_imma_tile_sweep_bench.cu.
#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/mmq_q4k_imma_tile.h"

namespace imp {
namespace {

constexpr int kSub = 32;

// -------- Synthetic input generators --------

void gen_random_w_sym_s8(std::vector<int8_t>& W, std::vector<__half>& alpha, std::vector<__half>& beta, int N,
                         int K, unsigned seed) {
    const int subs = K / kSub;
    W.resize(static_cast<size_t>(N) * K);
    alpha.resize(static_cast<size_t>(N) * subs);
    beta.resize(static_cast<size_t>(N) * subs);
    std::srand(seed);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            int q = std::rand() % 16;                                        // ∈ [0, 15] unsigned nibble
            W[static_cast<size_t>(n) * K + k] = static_cast<int8_t>(q - 8);  // symmetric
        }
        for (int s = 0; s < subs; ++s) {
            float d_per_n = 0.005f + 0.002f * (std::rand() % 100);  // d ~ U[0.005, 0.205]
            float dmin_per_n = 0.001f * (std::rand() % 100);
            int sc = std::rand() % 64;
            int m = std::rand() % 64;
            float alpha_f = d_per_n * static_cast<float>(sc);
            float beta_f = 8.0f * d_per_n * static_cast<float>(sc) - dmin_per_n * static_cast<float>(m);
            alpha[static_cast<size_t>(n) * subs + s] = __float2half(alpha_f);
            beta[static_cast<size_t>(n) * subs + s] = __float2half(beta_f);
        }
    }
}

void gen_random_x_s8(std::vector<int8_t>& X, std::vector<__half>& x_scale, std::vector<float>& x_rowsum,
                     int M, int K, unsigned seed) {
    const int subs = K / kSub;
    X.resize(static_cast<size_t>(M) * K);
    x_scale.resize(static_cast<size_t>(M) * subs);
    x_rowsum.resize(static_cast<size_t>(M) * subs);
    std::srand(seed);
    for (int m = 0; m < M; ++m) {
        for (int s = 0; s < subs; ++s) {
            int row_sum = 0;
            for (int k = 0; k < kSub; ++k) {
                int q = (std::rand() % 255) - 127;  // ∈ [-127, 127]
                X[static_cast<size_t>(m) * K + s * kSub + k] = static_cast<int8_t>(q);
                row_sum += q;
            }
            float scale_f = 0.001f + 0.0005f * (std::rand() % 50);
            x_scale[static_cast<size_t>(m) * subs + s] = __float2half(scale_f);
            x_rowsum[static_cast<size_t>(m) * subs + s] = static_cast<float>(row_sum);
        }
    }
}

}  // namespace
}  // namespace imp
