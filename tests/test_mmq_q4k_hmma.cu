// Correctness test for mmq_q4k_hmma_gemm — the Q4_K x FP16 tiled GEMM via
// HMMA m16n8k16. Compares the custom kernel output against a CPU reference
// (full Q4_K dequant + FP32 GEMM) to validate that the in-SMEM decode +
// WMMA path produces bit-equivalent results within FP16 quantisation noise.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/mmq_q4k_hmma.h"
#include "scoped_engine_arena.h"

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);  // T2 arena for the migrated scratches (A7 step 8)
namespace {

constexpr int kQ4kBlockBytes = 144;
constexpr int kQ4kSuperBlock = 256;

// CPU helper: unpack 6-bit scale and min from Q4_K scales[12] array.
// Matches the device get_scale_min_k4 in mmq_q4k_hmma.cu and ggml.
inline void get_scale_min_k4_cpu(int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
    if (j < 4) {
        d_out = q[j] & 63u;
        m_out = q[j + 4] & 63u;
    } else {
        d_out = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Generate a random Q4_K weight blob: N rows x K cols, each row = K/256 super-blocks.
// Uses the same generation pattern as test_mmq_q4k_imma_gemm.cu.
void gen_q4k_weight(std::vector<uint8_t>& W, int N, int K, unsigned seed) {
    ASSERT_EQ(K % kQ4kSuperBlock, 0);
    const int blocks_per_row = K / kQ4kSuperBlock;
    W.resize(static_cast<size_t>(N) * blocks_per_row * kQ4kBlockBytes);
    std::srand(seed);
    for (size_t b = 0; b < W.size() / kQ4kBlockBytes; ++b) {
        uint8_t* bp = W.data() + b * kQ4kBlockBytes;
        // Random FP16 d and dmin in a reasonable range.
        float d = 0.001f + 0.0005f * (std::rand() % 100);
        float dmin = 0.0005f + 0.0001f * (std::rand() % 100);
        __half dh = __float2half(d), dminh = __float2half(dmin);
        uint16_t db = __half_as_ushort(dh), dminb = __half_as_ushort(dminh);
        bp[0] = db & 0xFF;
        bp[1] = (db >> 8) & 0xFF;
        bp[2] = dminb & 0xFF;
        bp[3] = (dminb >> 8) & 0xFF;
        // Random scales (12 bytes) and quant values (128 bytes).
        for (int i = 0; i < 12; ++i) bp[4 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
        for (int i = 0; i < 128; ++i) bp[16 + i] = static_cast<uint8_t>(std::rand() & 0xFF);
    }
}

// Full CPU Q4_K dequant of one row into FP32.
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

void run_hmma_test(int M, int N, int K, unsigned seed, float max_rel_avg) {
    SCOPED_TRACE(testing::Message() << "M=" << M << " N=" << N << " K=" << K << " seed=" << seed);
    ASSERT_EQ(K % kQ4kSuperBlock, 0);

    // Generate Q4_K weights and FP16 activations.
    std::vector<uint8_t> W;
    gen_q4k_weight(W, N, K, seed);

    std::srand(seed + 1);
    std::vector<float> X_fp32(static_cast<size_t>(M) * K);
    std::vector<__half> X_fp16(static_cast<size_t>(M) * K);
    for (size_t i = 0; i < X_fp32.size(); ++i) {
        float v = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 0.5f;
        X_fp16[i] = __float2half(v);
        // FP16 round-trip so both paths see the same activation.
        X_fp32[i] = __half2float(X_fp16[i]);
    }

    // Upload to device.
    void* dW = nullptr;
    __half *dX = nullptr, *dY = nullptr;
    ASSERT_EQ(cudaMalloc(&dW, W.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dX, X_fp16.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dY, static_cast<size_t>(M) * N * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dX, X_fp16.data(), X_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice),
              cudaSuccess);
    // Zero output to catch partial writes.
    ASSERT_EQ(cudaMemset(dY, 0, static_cast<size_t>(M) * N * sizeof(__half)), cudaSuccess);

    // Run HMMA kernel.
    ASSERT_TRUE(mmq_q4k_hmma_gemm(dX, dW, dY, M, N, K, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Check for CUDA errors.
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

    // CPU reference.
    std::vector<float> Y_ref;
    cpu_reference_gemm(X_fp32, W, Y_ref, M, N, K);

    // Pull output.
    std::vector<__half> Y_got(static_cast<size_t>(M) * N);
    ASSERT_EQ(cudaMemcpy(Y_got.data(), dY, Y_got.size() * sizeof(__half), cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Compare: FP16 dequant + HMMA accumulation noise -> expect < 1% average relative error.
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
    std::fprintf(stderr,
        "[q4k-hmma M=%d N=%d K=%d] max_abs=%.4f rel_avg=%.6f (worst @%d,%d)\n",
        M, N, K, max_abs, rel_avg, worst_m, worst_n);

    EXPECT_LT(rel_avg, max_rel_avg)
        << "Mean relative error " << rel_avg << " exceeds tolerance " << max_rel_avg;

    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dY);
}

// ---- Test cases ----

// Minimal: single tile, single K-block.
TEST(MmqQ4kHmma, CorrectnessSmallest) {
    run_hmma_test(/*M=*/64, /*N=*/64, /*K=*/256, /*seed=*/42, /*max_rel_avg=*/0.01f);
}

// Multi-tile in M and N.
TEST(MmqQ4kHmma, CorrectnessMultiTile) {
    run_hmma_test(/*M=*/128, /*N=*/128, /*K=*/512, /*seed=*/17, /*max_rel_avg=*/0.01f);
}

// Non-tile-aligned M and N (kernel must handle bounds).
TEST(MmqQ4kHmma, CorrectnessNonAligned) {
    run_hmma_test(/*M=*/48, /*N=*/96, /*K=*/256, /*seed=*/7, /*max_rel_avg=*/0.01f);
}

// Realistic FFN-like shape.
TEST(MmqQ4kHmma, CorrectnessFFNLike) {
    run_hmma_test(/*M=*/256, /*N=*/512, /*K=*/1024, /*seed=*/31, /*max_rel_avg=*/0.01f);
}

// Reject unsupported shapes.
TEST(MmqQ4kHmma, RejectsInvalidK) {
    // K not multiple of 256.
    EXPECT_FALSE(mmq_q4k_hmma_gemm(nullptr, nullptr, nullptr, 64, 64, 200, nullptr));
}

TEST(MmqQ4kHmma, RejectsTooSmall) {
    EXPECT_FALSE(mmq_q4k_hmma_gemm(nullptr, nullptr, nullptr, 8, 8, 256, nullptr));
}

}  // namespace
}  // namespace imp
