// End-to-end test for `mmq_q4k_imma_gemm` — the high-level Phase 2C entry
// that owns reorder + activation quant + tile dispatch internally. Verifies
// that the IMMA path produces results algebraically equivalent to a full
// Q4_K dequant + FP32 reference GEMM (within INT8/FP16 quantisation noise).

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

constexpr int kQ4kBlockBytes = 144;
constexpr int kQ4kSuperBlock = 256;
constexpr int kSub = 32;

// CPU helper (matches device get_scale_min_k4 in mmq_q4k_imma_layout.cu).
inline void get_scale_min_k4_cpu(int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
    if (j < 4) {
        d_out = q[j] & 63u;
        m_out = q[j + 4] & 63u;
    } else {
        d_out = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Build a random Q4_K weight blob (N rows × K cols, each row = K/256 super-blocks).
void gen_q4k_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    ASSERT_EQ(K % kQ4kSuperBlock, 0);
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

// Full CPU Q4_K dequant of one row into FP32 (single super-block at a time).
void dequant_q4k_row_cpu(const std::vector<uint8_t>& W, int n, int K, std::vector<float>& w_row) {
    const int blocks_per_row = K / kQ4kSuperBlock;
    w_row.assign(K, 0.0f);
    for (int s = 0; s < blocks_per_row; ++s) {
        const uint8_t* bp = W.data() + (static_cast<size_t>(n * blocks_per_row + s)) * kQ4kBlockBytes;
        uint16_t db = static_cast<uint16_t>(bp[0]) | (static_cast<uint16_t>(bp[1]) << 8);
        uint16_t dminb = static_cast<uint16_t>(bp[2]) | (static_cast<uint16_t>(bp[3]) << 8);
        float d = __half2float(__ushort_as_half(db));
        float dmin = __half2float(__ushort_as_half(dminb));
        const uint8_t* scales = bp + 4;
        const uint8_t* qs = bp + 4 + 12;
        for (int e = 0; e < kQ4kSuperBlock; ++e) {
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
            w_row[s * kQ4kSuperBlock + e] = val;
        }
    }
}

// CPU reference: Y[M, N] = X[M, K] @ W_dequant[N, K]^T in FP32.
void cpu_reference_gemm(const std::vector<float>& X_fp32, const std::vector<uint8_t>& W,
                        std::vector<float>& Y, int M, int N, int K) {
    Y.assign(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> w_row;
    for (int n = 0; n < N; ++n) {
        dequant_q4k_row_cpu(W, n, K, w_row);
        for (int m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += X_fp32[static_cast<size_t>(m) * K + k] * w_row[k];
            }
            Y[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

void run_e2e(int M, int N, int K, unsigned seed, float max_rel_avg) {
    SCOPED_TRACE(testing::Message() << "M=" << M << " N=" << N << " K=" << K << " seed=" << seed);
    ASSERT_EQ(M % 64, 0);
    ASSERT_EQ(N % 32, 0);
    ASSERT_EQ(K % kQ4kSuperBlock, 0);

    // Generate Q4_K weight and FP16 activations.
    std::vector<uint8_t> W;
    gen_q4k_weight(W, N, K, seed);
    std::srand(seed + 1);
    std::vector<float> X_fp32(static_cast<size_t>(M) * K);
    std::vector<__half> X_fp16(static_cast<size_t>(M) * K);
    for (int i = 0; i < M * K; ++i) {
        float v = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 0.5f;
        X_fp32[i] = v;
        X_fp16[i] = __float2half(v);
        // FP16 round-trip in reference, so both paths see the same activation.
        X_fp32[i] = __half2float(X_fp16[i]);
    }

    // Upload to device.
    void* dW = nullptr;
    __half *dX = nullptr, *dY = nullptr;
    cudaMalloc(&dW, W.size());
    cudaMalloc(&dX, X_fp16.size() * sizeof(__half));
    cudaMalloc(&dY, static_cast<size_t>(M) * N * sizeof(__half));
    cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X_fp16.data(), X_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice);

    // Run high-level IMMA path.
    ASSERT_TRUE(mmq_q4k_imma_gemm(dW, dX, dY, M, N, K, nullptr));
    cudaDeviceSynchronize();

    // CPU reference.
    std::vector<float> Y_ref;
    cpu_reference_gemm(X_fp32, W, Y_ref, M, N, K);

    // Pull output.
    std::vector<__half> Y_got(static_cast<size_t>(M) * N);
    cudaMemcpy(Y_got.data(), dY, Y_got.size() * sizeof(__half), cudaMemcpyDeviceToHost);

    // Compare. INT8 activation quant + FP16 epilogue → expect ~1-2 % avg rel error.
    double err_sum = 0.0, ref_sum = 0.0;
    float max_abs = 0.0f;
    int worst_m = 0, worst_n = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float got = __half2float(Y_got[static_cast<size_t>(m) * N + n]);
            float ref = Y_ref[static_cast<size_t>(m) * N + n];
            float diff = got - ref;
            err_sum += std::fabs(diff);
            ref_sum += std::fabs(ref);
            if (std::fabs(diff) > max_abs) {
                max_abs = std::fabs(diff);
                worst_m = m;
                worst_n = n;
            }
        }
    }
    double rel_avg = err_sum / std::max(ref_sum, 1e-9);
    std::fprintf(stderr, "[q4k-imma-gemm e2e M=%d N=%d K=%d] max_abs=%.4f rel_avg=%.4f (worst @%d,%d)\n",
                 M, N, K, max_abs, rel_avg, worst_m, worst_n);
    EXPECT_LT(rel_avg, max_rel_avg)
        << "Mean relative error " << rel_avg << " exceeds tolerance " << max_rel_avg;
}

// One-block-per-dim: smallest configuration that exercises the full path.
TEST(MmqQ4kImmaGemm, E2ESmallest) {
    run_e2e(/*M=*/64, /*N=*/32, /*K=*/256, /*seed=*/3, /*rel_avg=*/0.03f);
}

// Multi-CTA: shapes are multiples of (BLOCK_M, BLOCK_N) and span several K
// super-blocks (256 elements each) so the per-sub-block scale chain sees real
// variation.
TEST(MmqQ4kImmaGemm, E2EMultiTile) {
    run_e2e(/*M=*/128, /*N=*/64, /*K=*/512, /*seed=*/17, /*rel_avg=*/0.03f);
}

// Realistic FFN-projection shape (Qwen3-32B Q4_K_M FFN K=5120 dim, but reduced
// for test speed; same arithmetic regime).
TEST(MmqQ4kImmaGemm, E2EFFNLike) {
    run_e2e(/*M=*/256, /*N=*/128, /*K=*/1024, /*seed=*/41, /*rel_avg=*/0.03f);
}

}  // namespace
}  // namespace imp
