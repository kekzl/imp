// Phase 2A correctness test for the Q4_K → symmetric-s8 reorder kernel.
// Verifies that for every element of every row of every super-block,
//   α[r, k/32] * w_sym_s8[r, k] + β[r, k/32]   ==   d * sc[j] * q - dmin * m[j]
// — i.e. the IMMA epilogue identity α · q_sym + β reconstructs the same FP16
// value as the ggml dequant.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/mmq_q4k_imma_layout.h"

namespace imp {
namespace {

constexpr int kBlockBytes = 144;

// 6-bit get_scale_min_k4, CPU version. Identical to the device-side helper.
inline void get_scale_min_k4_cpu(int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
    if (j < 4) {
        d_out = q[j] & 63u;
        m_out = q[j + 4] & 63u;
    } else {
        d_out = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Build a random plausible Q4_K block: d/dmin sampled, scales[12] random, qs[128] random.
void make_q4k_block(std::vector<uint8_t>& out, int blocks_per_row, int n_rows, unsigned seed) {
    out.resize(static_cast<size_t>(blocks_per_row) * n_rows * kBlockBytes);
    std::srand(seed);
    for (size_t b = 0; b < out.size() / kBlockBytes; ++b) {
        uint8_t* bp = out.data() + b * kBlockBytes;
        // d ∈ [0.001, 0.05] is a reasonable Q4_K range for FP16 weights.
        float d = 0.001f + 0.0005f * (std::rand() % 100);
        float dmin = 0.0005f + 0.0001f * (std::rand() % 100);
        __half d_h = __float2half(d), dmin_h = __float2half(dmin);
        uint16_t d_bits = __half_as_ushort(d_h);
        uint16_t dmin_bits = __half_as_ushort(dmin_h);
        bp[0] = d_bits & 0xFF;
        bp[1] = (d_bits >> 8) & 0xFF;
        bp[2] = dmin_bits & 0xFF;
        bp[3] = (dmin_bits >> 8) & 0xFF;
        // scales[12]: random 6-bit values packed via the ggml format.
        // Simplest: random bytes — the 6-bit-extraction helper masks them.
        for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
        for (int i = 0; i < 128; ++i) bp[16 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
    }
}

// CPU oracle: full Q4_K dequant. Returns one FP32 value per (row, k).
void dequant_q4k_cpu(const std::vector<uint8_t>& blocks, int N, int K, std::vector<float>& out) {
    const int blocks_per_row = K / kQ4kSuperBlockSize;
    out.resize(static_cast<size_t>(N) * K);
    for (int r = 0; r < N; ++r) {
        for (int s = 0; s < blocks_per_row; ++s) {
            const uint8_t* bp =
                blocks.data() + static_cast<size_t>(r * blocks_per_row + s) * kBlockBytes;
            uint16_t d_bits = static_cast<uint16_t>(bp[0]) | (static_cast<uint16_t>(bp[1]) << 8);
            uint16_t dmin_bits = static_cast<uint16_t>(bp[2]) | (static_cast<uint16_t>(bp[3]) << 8);
            float d = __half2float(__ushort_as_half(d_bits));
            float dmin = __half2float(__ushort_as_half(dmin_bits));
            const uint8_t* scales = bp + 4;
            const uint8_t* qs = bp + 4 + 12;
            for (int e = 0; e < kQ4kSuperBlockSize; ++e) {
                const int group = e >> 6;
                const int in_grp = e & 63;
                const int is_high = (in_grp >> 5);
                const int byte_in_group = in_grp & 31;
                const int byte_in_qs = group * 32 + byte_in_group;
                const int sub_block = group * 2 + is_high;
                uint8_t sc_u, m_u;
                get_scale_min_k4_cpu(sub_block, scales, sc_u, m_u);
                int nibble = is_high ? (qs[byte_in_qs] >> 4) : (qs[byte_in_qs] & 0xF);
                float val = d * static_cast<float>(sc_u) * static_cast<float>(nibble) -
                            dmin * static_cast<float>(m_u);
                out[static_cast<size_t>(r) * K + static_cast<size_t>(s * kQ4kSuperBlockSize + e)] =
                    val;
            }
        }
    }
}

void check_reorder(int N, int K, unsigned seed) {
    SCOPED_TRACE(testing::Message() << "N=" << N << " K=" << K << " seed=" << seed);
    ASSERT_EQ(K % kQ4kSuperBlockSize, 0);

    const int blocks_per_row = K / kQ4kSuperBlockSize;
    const int subs_per_row = blocks_per_row * kQ4kSubBlocksPerSuper;
    const int total_subs = N * subs_per_row;

    // Build input blocks.
    std::vector<uint8_t> host_blocks;
    make_q4k_block(host_blocks, blocks_per_row, N, seed);

    // CPU oracle.
    std::vector<float> ref_fp32;
    dequant_q4k_cpu(host_blocks, N, K, ref_fp32);

    // Upload + run kernel.
    void* dev_blocks = nullptr;
    cudaMalloc(&dev_blocks, host_blocks.size());
    cudaMemcpy(dev_blocks, host_blocks.data(), host_blocks.size(), cudaMemcpyHostToDevice);

    int8_t* dev_w = nullptr;
    __half* dev_alpha = nullptr;
    __half* dev_beta = nullptr;
    cudaMalloc(&dev_w, static_cast<size_t>(N) * K);
    cudaMalloc(&dev_alpha, static_cast<size_t>(total_subs) * sizeof(__half));
    cudaMalloc(&dev_beta, static_cast<size_t>(total_subs) * sizeof(__half));

    mmq_q4k_imma_reorder(dev_blocks, N, K, dev_w, dev_alpha, dev_beta, nullptr);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // Download outputs.
    std::vector<int8_t> host_w(static_cast<size_t>(N) * K);
    std::vector<__half> host_alpha(total_subs);
    std::vector<__half> host_beta(total_subs);
    cudaMemcpy(host_w.data(), dev_w, host_w.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_alpha.data(), dev_alpha, host_alpha.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(host_beta.data(), dev_beta, host_beta.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // Verify: for every (r, k), α[r, k/32] * w_sym[r, k] + β[r, k/32] ≈ ref[r, k].
    //
    // Reconstruction goes through FP16 α and β. Per-element error budget is
    // bounded by:
    //   |recon - ref|  ≤  ulp(α)·|w| + ulp(β)
    // Worst case (d=0.05, sc=63, w=±8, dmin=0.05, m=63): |α|≈3.15, |β|≈25,
    // |α·w| ≤ 25, ulp at magnitude 25 ≈ 25·2^-10 ≈ 0.025 per term, summed
    // worst case ~0.05. Cap at 0.1 to be safe (relative gates blow up near
    // ref≈0 which is fine — we check abs there).
    float max_abs_err = 0.0f;
    int worst_r = 0, worst_k = 0;
    for (int r = 0; r < N; ++r) {
        for (int k = 0; k < K; ++k) {
            const int sub_global_idx =
                r * subs_per_row + (k / kQ4kSubBlockSize);
            const float a = __half2float(host_alpha[sub_global_idx]);
            const float b = __half2float(host_beta[sub_global_idx]);
            const int w = static_cast<int>(host_w[static_cast<size_t>(r) * K + k]);
            const float recon = a * static_cast<float>(w) + b;
            const float ref = ref_fp32[static_cast<size_t>(r) * K + k];
            const float ae = std::fabs(recon - ref);
            if (ae > max_abs_err) {
                max_abs_err = ae;
                worst_r = r;
                worst_k = k;
            }
            // s8 quant of nibble q-8 must round-trip exactly through int8_t.
            ASSERT_GE(w, -8) << "w_sym out of range at (" << r << ", " << k << ")";
            ASSERT_LE(w, 7) << "w_sym out of range at (" << r << ", " << k << ")";
        }
    }
    EXPECT_LT(max_abs_err, 0.1f)
        << "Reconstruction abs_err > 0.1 at (" << worst_r << ", " << worst_k << ") — "
        << "likely a layout / dequant mismatch (FP16 ulp budget is ~0.05)";
    std::fprintf(stderr, "[q4k-imma-reorder N=%d K=%d] max_abs_err=%.5f at (%d, %d)\n", N, K,
                 max_abs_err, worst_r, worst_k);

    cudaFree(dev_blocks);
    cudaFree(dev_w);
    cudaFree(dev_alpha);
    cudaFree(dev_beta);
}

TEST(MmqQ4kImmaLayout, ReorderRoundTripSmall) {
    check_reorder(/*N=*/4, /*K=*/256, /*seed=*/11);
}

TEST(MmqQ4kImmaLayout, ReorderRoundTripMultiSuper) {
    check_reorder(/*N=*/4, /*K=*/1024, /*seed=*/41);  // 4 super-blocks per row
}

TEST(MmqQ4kImmaLayout, ReorderRoundTripFFNShape) {
    // Qwen3-32B FFN dimensions: d_ff × d_model = 27648 × 5120.
    // Use a slice to keep allocation small but exercise both dims.
    check_reorder(/*N=*/128, /*K=*/2560, /*seed=*/97);  // 10 super-blocks per row
}

}  // namespace
}  // namespace imp
