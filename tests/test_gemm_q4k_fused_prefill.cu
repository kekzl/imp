// Correctness test for gemm_q4k_fused_moe_prefill.
// Compares against CPU reference (full Q4_K dequant + FP32 GEMM).

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/gemm.h"
#include "compute/gemm_q4k.h"

namespace imp {
namespace {

constexpr int kQ4kBlockBytes = 144;
constexpr int kQ4kSuperBlock = 256;

inline void get_scale_min_k4_cpu(int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
    if (j < 4) {
        d_out = q[j] & 63u;
        m_out = q[j + 4] & 63u;
    } else {
        d_out = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void gen_q4k_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    const int blocks_per_row = K / kQ4kSuperBlock;
    W.resize(static_cast<size_t>(N) * blocks_per_row * kQ4kBlockBytes);
    std::srand(seed);
    for (size_t b = 0; b < W.size() / kQ4kBlockBytes; ++b) {
        uint8_t* bp = W.data() + b * kQ4kBlockBytes;
        float d = 0.001f + 0.0005f * (std::rand() % 100);
        float dmin = 0.0005f + 0.0001f * (std::rand() % 100);
        __half dh = __float2half(d), dminh = __float2half(dmin);
        uint16_t db = __half_as_ushort(dh), dminb = __half_as_ushort(dminh);
        bp[0] = db & 0xFF;
        bp[1] = (db >> 8) & 0xFF;
        bp[2] = dminb & 0xFF;
        bp[3] = (dminb >> 8) & 0xFF;
        for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
        for (int i = 0; i < 128; ++i) bp[16 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
    }
}

void dequant_q4k_row_cpu(const uint8_t* W_base, int n, int K, std::vector<float>& w_row) {
    const int blocks_per_row = K / kQ4kSuperBlock;
    w_row.assign(K, 0.0f);
    for (int s = 0; s < blocks_per_row; ++s) {
        const uint8_t* bp = W_base + (static_cast<size_t>(n) * blocks_per_row + s) * kQ4kBlockBytes;
        uint16_t db = static_cast<uint16_t>(bp[0]) | (static_cast<uint16_t>(bp[1]) << 8);
        uint16_t dminb = static_cast<uint16_t>(bp[2]) | (static_cast<uint16_t>(bp[3]) << 8);
        float d = __half2float(__ushort_as_half(db));
        float dmin = __half2float(__ushort_as_half(dminb));
        const uint8_t* scales = bp + 4;
        const uint8_t* qs = bp + 16;
        for (int e = 0; e < kQ4kSuperBlock; ++e) {
            const int group = e >> 6;
            const int in_grp = e & 63;
            const int is_high = (in_grp >> 5);
            const int byte_in_qs = group * 32 + (in_grp & 31);
            const int sub_block = group * 2 + is_high;
            uint8_t sc_u, m_u;
            get_scale_min_k4_cpu(sub_block, scales, sc_u, m_u);
            int nibble = is_high ? (qs[byte_in_qs] >> 4) : (qs[byte_in_qs] & 0xF);
            w_row[s * kQ4kSuperBlock + e] =
                d * static_cast<float>(sc_u) * static_cast<float>(nibble) -
                dmin * static_cast<float>(m_u);
        }
    }
}

struct TestCase {
    int n_experts;
    int N;
    int K;
    int total_tokens;
    unsigned seed;
    float max_rel_avg;
};

void run_moe_fused_test(const TestCase& tc) {
    SCOPED_TRACE(testing::Message() << "experts=" << tc.n_experts << " N=" << tc.N << " K=" << tc.K
                                    << " tokens=" << tc.total_tokens << " seed=" << tc.seed);

    const int blocks_per_row = tc.K / kQ4kSuperBlock;
    const size_t expert_weight_bytes = static_cast<size_t>(tc.N) * blocks_per_row * kQ4kBlockBytes;

    // Generate per-expert Q4_K weights
    std::vector<std::vector<uint8_t>> expert_weights(tc.n_experts);
    for (int e = 0; e < tc.n_experts; ++e)
        gen_q4k_weight(expert_weights[e], tc.N, tc.K, tc.seed + e * 137);

    // Pack all experts contiguously
    std::vector<uint8_t> packed(tc.n_experts * expert_weight_bytes);
    for (int e = 0; e < tc.n_experts; ++e)
        memcpy(packed.data() + e * expert_weight_bytes, expert_weights[e].data(), expert_weight_bytes);

    // Generate expert offsets (distribute tokens roughly evenly)
    std::vector<int32_t> offsets(tc.n_experts + 1);
    offsets[0] = 0;
    int remaining = tc.total_tokens;
    for (int e = 0; e < tc.n_experts; ++e) {
        int count = remaining / (tc.n_experts - e);
        offsets[e + 1] = offsets[e] + count;
        remaining -= count;
    }
    ASSERT_EQ(offsets[tc.n_experts], tc.total_tokens);

    // Generate FP16 activations
    std::srand(tc.seed + 999);
    std::vector<float> X_fp32(static_cast<size_t>(tc.total_tokens) * tc.K);
    std::vector<__half> X_fp16(X_fp32.size());
    for (size_t i = 0; i < X_fp32.size(); ++i) {
        float v = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 0.5f;
        X_fp16[i] = __float2half(v);
        X_fp32[i] = __half2float(X_fp16[i]);
    }

    // CPU reference: per-expert GEMM
    std::vector<float> Y_ref(static_cast<size_t>(tc.total_tokens) * tc.N, 0.0f);
    std::vector<float> w_row;
    for (int e = 0; e < tc.n_experts; ++e) {
        int m_start = offsets[e];
        int m_count = offsets[e + 1] - offsets[e];
        for (int n = 0; n < tc.N; ++n) {
            dequant_q4k_row_cpu(expert_weights[e].data(), n, tc.K, w_row);
            for (int m = 0; m < m_count; ++m) {
                float acc = 0.0f;
                for (int k = 0; k < tc.K; ++k)
                    acc += X_fp32[static_cast<size_t>(m_start + m) * tc.K + k] * w_row[k];
                Y_ref[static_cast<size_t>(m_start + m) * tc.N + n] = acc;
            }
        }
    }

    // Upload to device
    void *dW = nullptr, *dX = nullptr, *dY = nullptr;
    int32_t* dOffsets = nullptr;
    cudaMalloc(&dW, packed.size());
    cudaMalloc(&dX, X_fp16.size() * sizeof(__half));
    cudaMalloc(&dY, static_cast<size_t>(tc.total_tokens) * tc.N * sizeof(__half));
    cudaMalloc(&dOffsets, offsets.size() * sizeof(int32_t));
    cudaMemcpy(dW, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X_fp16.data(), X_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dOffsets, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(dY, 0, static_cast<size_t>(tc.total_tokens) * tc.N * sizeof(__half));

    // Run kernel
    gemm_q4k_fused_moe_prefill(dW, dX, dY, dOffsets, tc.N, tc.K, expert_weight_bytes, tc.n_experts);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Pull output
    std::vector<__half> Y_got(static_cast<size_t>(tc.total_tokens) * tc.N);
    cudaMemcpy(Y_got.data(), dY, Y_got.size() * sizeof(__half), cudaMemcpyDeviceToHost);

    // Compare
    double err_sum = 0.0, ref_sum = 0.0;
    float max_abs = 0.0f;
    int worst_m = 0, worst_n = 0;
    for (int m = 0; m < tc.total_tokens; ++m) {
        for (int n = 0; n < tc.N; ++n) {
            float got = __half2float(Y_got[static_cast<size_t>(m) * tc.N + n]);
            float ref = Y_ref[static_cast<size_t>(m) * tc.N + n];
            float diff = std::fabs(got - ref);
            err_sum += diff;
            ref_sum += std::fabs(ref);
            if (diff > max_abs) {
                max_abs = diff;
                worst_m = m;
                worst_n = n;
            }
        }
    }
    double rel_avg = err_sum / std::max(ref_sum, 1e-9);
    std::fprintf(stderr,
                 "[q4k-fused-moe experts=%d N=%d K=%d tokens=%d] max_abs=%.4f rel_avg=%.6f (worst @%d,%d)\n",
                 tc.n_experts, tc.N, tc.K, tc.total_tokens, max_abs, rel_avg, worst_m, worst_n);
    EXPECT_LT(rel_avg, tc.max_rel_avg)
        << "Mean relative error " << rel_avg << " exceeds tolerance " << tc.max_rel_avg;

    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dOffsets);
}

TEST(GemmQ4kFusedMoePrefill, SmallestConfig) {
    run_moe_fused_test({.n_experts = 4, .N = 16, .K = 256, .total_tokens = 8, .seed = 1, .max_rel_avg = 0.01f});
}

TEST(GemmQ4kFusedMoePrefill, MultiBlock) {
    run_moe_fused_test({.n_experts = 8, .N = 64, .K = 512, .total_tokens = 32, .seed = 7, .max_rel_avg = 0.01f});
}

TEST(GemmQ4kFusedMoePrefill, RealisticMoE) {
    run_moe_fused_test(
        {.n_experts = 64, .N = 128, .K = 1024, .total_tokens = 256, .seed = 42, .max_rel_avg = 0.01f});
}

TEST(GemmQ4kFusedMoePrefill, UnevenDistribution) {
    // Uneven: some experts get 0 tokens, others get many
    TestCase tc = {.n_experts = 16, .N = 32, .K = 512, .total_tokens = 20, .seed = 13, .max_rel_avg = 0.01f};
    run_moe_fused_test(tc);
}

TEST(GemmQ4kFusedMoePrefill, SingleToken) {
    run_moe_fused_test({.n_experts = 4, .N = 32, .K = 256, .total_tokens = 1, .seed = 99, .max_rel_avg = 0.01f});
}

// Test the dp4a path: FP16 activations → Q8_1 quantization → dp4a GEMM
void run_dp4a_test(const TestCase& tc) {
    SCOPED_TRACE(testing::Message() << "dp4a experts=" << tc.n_experts << " N=" << tc.N << " K=" << tc.K
                                    << " tokens=" << tc.total_tokens);

    const int blocks_per_row = tc.K / kQ4kSuperBlock;
    const size_t expert_weight_bytes = static_cast<size_t>(tc.N) * blocks_per_row * kQ4kBlockBytes;

    std::vector<std::vector<uint8_t>> expert_weights(tc.n_experts);
    for (int e = 0; e < tc.n_experts; ++e)
        gen_q4k_weight(expert_weights[e], tc.N, tc.K, tc.seed + e * 137);

    std::vector<uint8_t> packed(tc.n_experts * expert_weight_bytes);
    for (int e = 0; e < tc.n_experts; ++e)
        memcpy(packed.data() + e * expert_weight_bytes, expert_weights[e].data(), expert_weight_bytes);

    std::vector<int32_t> offsets(tc.n_experts + 1);
    offsets[0] = 0;
    int remaining = tc.total_tokens;
    for (int e = 0; e < tc.n_experts; ++e) {
        int count = remaining / (tc.n_experts - e);
        offsets[e + 1] = offsets[e] + count;
        remaining -= count;
    }

    std::srand(tc.seed + 999);
    std::vector<float> X_fp32(static_cast<size_t>(tc.total_tokens) * tc.K);
    std::vector<__half> X_fp16(X_fp32.size());
    for (size_t i = 0; i < X_fp32.size(); ++i) {
        float v = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 0.5f;
        X_fp16[i] = __float2half(v);
        X_fp32[i] = __half2float(X_fp16[i]);
    }

    // CPU reference
    std::vector<float> Y_ref(static_cast<size_t>(tc.total_tokens) * tc.N, 0.0f);
    std::vector<float> w_row;
    for (int e = 0; e < tc.n_experts; ++e) {
        int m_start = offsets[e];
        int m_count = offsets[e + 1] - offsets[e];
        for (int n = 0; n < tc.N; ++n) {
            dequant_q4k_row_cpu(expert_weights[e].data(), n, tc.K, w_row);
            for (int m = 0; m < m_count; ++m) {
                float acc = 0.0f;
                for (int k = 0; k < tc.K; ++k)
                    acc += X_fp32[static_cast<size_t>(m_start + m) * tc.K + k] * w_row[k];
                Y_ref[static_cast<size_t>(m_start + m) * tc.N + n] = acc;
            }
        }
    }

    // Upload weights, offsets
    void* dW = nullptr;
    int32_t* dOffsets = nullptr;
    __half *dX = nullptr, *dY = nullptr;
    block_q8_1* dQ8 = nullptr;
    float* dD8 = nullptr;

    cudaMalloc(&dW, packed.size());
    cudaMalloc(&dX, X_fp16.size() * sizeof(__half));
    cudaMalloc(&dOffsets, offsets.size() * sizeof(int32_t));
    int total_elems = tc.total_tokens * tc.K;
    int q8_blocks = total_elems / 32;
    cudaMalloc(&dQ8, q8_blocks * sizeof(block_q8_1));
    cudaMalloc(&dD8, q8_blocks * sizeof(float));
    cudaMalloc(&dY, static_cast<size_t>(tc.total_tokens) * tc.N * sizeof(__half));

    cudaMemcpy(dW, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X_fp16.data(), X_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dOffsets, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(dY, 0, static_cast<size_t>(tc.total_tokens) * tc.N * sizeof(__half));

    // Quantize activations to Q8_1
    quantize_fp16_to_q8_1(dX, dQ8, dD8, total_elems);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Run dp4a kernel
    gemm_q4k_dp4a_moe_fused(dW, dQ8, dD8, dY, dOffsets, tc.K, tc.N, tc.n_experts,
                            expert_weight_bytes);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Compare
    std::vector<__half> Y_got(static_cast<size_t>(tc.total_tokens) * tc.N);
    cudaMemcpy(Y_got.data(), dY, Y_got.size() * sizeof(__half), cudaMemcpyDeviceToHost);

    double err_sum = 0.0, ref_sum = 0.0;
    float max_abs = 0.0f;
    for (int m = 0; m < tc.total_tokens; ++m) {
        for (int n = 0; n < tc.N; ++n) {
            float got = __half2float(Y_got[static_cast<size_t>(m) * tc.N + n]);
            float ref = Y_ref[static_cast<size_t>(m) * tc.N + n];
            float diff = std::fabs(got - ref);
            err_sum += diff;
            ref_sum += std::fabs(ref);
            if (diff > max_abs) max_abs = diff;
        }
    }
    double rel_avg = err_sum / std::max(ref_sum, 1e-9);
    std::fprintf(stderr, "[q4k-dp4a experts=%d N=%d K=%d tokens=%d] max_abs=%.4f rel_avg=%.6f\n",
                 tc.n_experts, tc.N, tc.K, tc.total_tokens, max_abs, rel_avg);
    // dp4a has Q8_1 quant noise on top of FP16 → slightly looser tolerance
    EXPECT_LT(rel_avg, tc.max_rel_avg)
        << "Mean relative error " << rel_avg << " exceeds tolerance " << tc.max_rel_avg;

    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(dOffsets);
    cudaFree(dQ8); cudaFree(dD8);
}

TEST(GemmQ4kDp4aMoePrefill, SmallestConfig) {
    run_dp4a_test({.n_experts = 4, .N = 16, .K = 256, .total_tokens = 8, .seed = 1, .max_rel_avg = 0.03f});
}

TEST(GemmQ4kDp4aMoePrefill, MultiBlock) {
    run_dp4a_test({.n_experts = 8, .N = 64, .K = 512, .total_tokens = 32, .seed = 7, .max_rel_avg = 0.03f});
}

TEST(GemmQ4kDp4aMoePrefill, RealisticMoE) {
    run_dp4a_test({.n_experts = 64, .N = 128, .K = 1024, .total_tokens = 256, .seed = 42, .max_rel_avg = 0.03f});
}

}  // namespace
}  // namespace imp
