#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/ffn_sparsity_mask.h"
#include "compute/ffn_sparsity_probe.h"
#include "compute/gemm.h"
#include "runtime/config.h"

namespace imp {
namespace {

// CPU reference: silu(g) * u
inline float ref_silu_up(float g, float u) {
    return (g / (1.0f + std::exp(-g))) * u;
}

half* upload_fp16(const std::vector<float>& host) {
    std::vector<half> h(host.size());
    for (size_t i = 0; i < host.size(); ++i) h[i] = __float2half(host[i]);
    half* dev = nullptr;
    cudaMalloc(&dev, h.size() * sizeof(half));
    cudaMemcpy(dev, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice);
    return dev;
}

// Build Q8_0 weight blob for one row of length K. Returns device buffer.
// Mirrors the layout used by gemv_q8_0_q8_1_residual: per-block [half d, int8[32]].
void* make_q8_0_row(int K, unsigned seed) {
    int blocks = K / 32;
    size_t bytes = static_cast<size_t>(blocks) * 34;
    std::vector<uint8_t> host(bytes);
    std::srand(seed);
    for (int b = 0; b < blocks; ++b) {
        uint8_t* bp = host.data() + static_cast<size_t>(b) * 34;
        int8_t qs[32];
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float v = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 4.0f;
            qs[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(v * 32.0f))));
            amax = std::max(amax, std::abs(static_cast<float>(qs[i])));
        }
        float d = amax / 127.0f;
        half dh = __float2half(d);
        std::memcpy(bp, &dh, 2);
        std::memcpy(bp + 2, qs, 32);
    }
    void* dev = nullptr;
    cudaMalloc(&dev, bytes);
    cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice);
    return dev;
}

// -----------------------------------------------------------------------------
// Mask builder: bit-correct against CPU oracle.
// -----------------------------------------------------------------------------
TEST(FFNSparsity, MaskMatchesCPUOracle) {
    const int K = 1024;  // 32 Q8 blocks → 1 mask word
    std::vector<float> g_host(K), u_host(K);
    std::srand(7);
    for (int i = 0; i < K; ++i) {
        g_host[i] = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 4.0f;
        u_host[i] = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 4.0f;
    }
    half* g_dev = upload_fp16(g_host);
    half* u_dev = upload_fp16(u_host);

    const int n_blocks = K / 32;
    const int n_words = (n_blocks + 31) / 32;
    uint32_t* mask_dev = nullptr;
    cudaMalloc(&mask_dev, static_cast<size_t>(n_words) * sizeof(uint32_t));

    const float threshold = 0.05f;
    build_swiglu_block_mask(g_dev, u_dev, mask_dev, K, threshold, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint32_t> mask_host(n_words, 0u);
    cudaMemcpy(mask_host.data(), mask_dev, static_cast<size_t>(n_words) * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    // CPU oracle
    for (int b = 0; b < n_blocks; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int i = b * 32 + j;
            // FP16 round-trip — same precision the kernel sees.
            float gf = __half2float(__float2half(g_host[i]));
            float uf = __half2float(__float2half(u_host[i]));
            amax = std::max(amax, std::abs(ref_silu_up(gf, uf)));
        }
        const bool exp_bit = amax >= threshold;
        const bool got_bit = (mask_host[b >> 5] >> (b & 31)) & 1u;
        EXPECT_EQ(got_bit, exp_bit) << "block " << b << " amax=" << amax;
    }

    cudaFree(g_dev);
    cudaFree(u_dev);
    cudaFree(mask_dev);
}

// -----------------------------------------------------------------------------
// Masked GEMV at threshold=0 ⇒ all mask bits set ⇒ output bit-identical to the
// reference unmasked Q8_0 residual GEMV.
// -----------------------------------------------------------------------------
TEST(FFNSparsity, MaskedGEMVBitIdenticalWhenAllBitsSet) {
    // M must be large enough that the reference dispatcher picks the kpar
    // layout (Q8_0 has kPreferKpar=false; needs M ≳ #SMs for kpar to dominate
    // the rpar block-count baseline). At M=512 the reference's
    // launch_gemv_dp4a routes through gemv_dp4a_kpar_kernel<Q8_0, true>,
    // which is exactly what gemv_q8_0_q8_1_residual_masked mirrors.
    const int K = 1024;
    const int M = 512;

    // Build Q8_0 weights for M rows.
    int blocks_per_row = K / 32;
    size_t row_bytes = static_cast<size_t>(blocks_per_row) * 34;
    void* W_dev = nullptr;
    cudaMalloc(&W_dev, static_cast<size_t>(M) * row_bytes);
    for (int r = 0; r < M; ++r) {
        void* row = make_q8_0_row(K, 123u + r);
        cudaMemcpy(static_cast<uint8_t*>(W_dev) + static_cast<size_t>(r) * row_bytes, row, row_bytes,
                   cudaMemcpyDeviceToDevice);
        cudaFree(row);
    }

    // Random gate / up for input x = silu(g)*u; quantize to Q8_1.
    std::vector<float> g_host(K), u_host(K);
    std::srand(11);
    for (int i = 0; i < K; ++i) {
        g_host[i] = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 4.0f;
        u_host[i] = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 4.0f;
    }
    half* g_dev = upload_fp16(g_host);
    half* u_dev = upload_fp16(u_host);

    // Synthesise x = silu(g)*u on host, upload as FP16.
    std::vector<float> x_host(K);
    for (int i = 0; i < K; ++i) {
        float gf = __half2float(__float2half(g_host[i]));
        float uf = __half2float(__float2half(u_host[i]));
        x_host[i] = ref_silu_up(gf, uf);
    }
    half* x_dev = upload_fp16(x_host);

    int q8_blocks = K / 32;
    int padded = ((K + 255) / 256) * 8;
    block_q8_1* q8_buf = nullptr;
    float* d8_buf = nullptr;
    cudaMalloc(&q8_buf, padded * sizeof(block_q8_1));
    cudaMalloc(&d8_buf, padded * sizeof(float));
    cudaMemset(q8_buf, 0, padded * sizeof(block_q8_1));
    cudaMemset(d8_buf, 0, padded * sizeof(float));
    quantize_fp16_to_q8_1(x_dev, q8_buf, d8_buf, K, nullptr);

    // Residual input
    std::vector<float> r_host(M, 0.0f);
    std::srand(31);
    for (int i = 0; i < M; ++i)
        r_host[i] = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 2.0f;
    half* r_dev = upload_fp16(r_host);

    half *y_ref = nullptr, *y_masked = nullptr;
    cudaMalloc(&y_ref, M * sizeof(half));
    cudaMalloc(&y_masked, M * sizeof(half));

    // Reference
    gemv_q8_0_q8_1_residual(W_dev, q8_buf, d8_buf, y_ref, r_dev, M, K, nullptr);

    // Build mask at threshold=0 → all bits set
    int n_words = (q8_blocks + 31) / 32;
    uint32_t* mask_dev = nullptr;
    cudaMalloc(&mask_dev, static_cast<size_t>(n_words) * sizeof(uint32_t));
    build_swiglu_block_mask(g_dev, u_dev, mask_dev, K, 0.0f, nullptr);

    // Sanity: mask must be all-1
    std::vector<uint32_t> mask_host(n_words, 0u);
    cudaMemcpy(mask_host.data(), mask_dev, static_cast<size_t>(n_words) * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    for (int b = 0; b < q8_blocks; ++b) {
        EXPECT_EQ((mask_host[b >> 5] >> (b & 31)) & 1u, 1u) << "block " << b;
    }

    gemv_q8_0_q8_1_residual_masked(W_dev, q8_buf, d8_buf, mask_dev, y_masked, r_dev, M, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hy_ref(M), hy_masked(M);
    cudaMemcpy(hy_ref.data(), y_ref, M * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy_masked.data(), y_masked, M * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        // Bit-identical: reduction order and arithmetic match.
        EXPECT_EQ(__half_as_ushort(hy_ref[i]), __half_as_ushort(hy_masked[i]))
            << "row " << i << " ref=" << __half2float(hy_ref[i])
            << " masked=" << __half2float(hy_masked[i]);
    }

    cudaFree(W_dev);
    cudaFree(g_dev);
    cudaFree(u_dev);
    cudaFree(x_dev);
    cudaFree(q8_buf);
    cudaFree(d8_buf);
    cudaFree(r_dev);
    cudaFree(y_ref);
    cudaFree(y_masked);
    cudaFree(mask_dev);
}

// -----------------------------------------------------------------------------
// Masked GEMV with a manually-zeroed mask: skipped blocks contribute nothing.
// -----------------------------------------------------------------------------
TEST(FFNSparsity, MaskedGEMVSkipsBlocks) {
    const int K = 1024;
    const int M = 4;
    int blocks_per_row = K / 32;
    size_t row_bytes = static_cast<size_t>(blocks_per_row) * 34;
    void* W_dev = nullptr;
    cudaMalloc(&W_dev, static_cast<size_t>(M) * row_bytes);
    for (int r = 0; r < M; ++r) {
        void* row = make_q8_0_row(K, 99u + r);
        cudaMemcpy(static_cast<uint8_t*>(W_dev) + static_cast<size_t>(r) * row_bytes, row, row_bytes,
                   cudaMemcpyDeviceToDevice);
        cudaFree(row);
    }

    // Input that has high amax in every block (so threshold=0 with all-1 mask is the
    // natural baseline). We override the mask manually instead.
    std::vector<float> x_host(K);
    std::srand(73);
    for (int i = 0; i < K; ++i)
        x_host[i] = (std::rand() / static_cast<float>(RAND_MAX) - 0.5f) * 4.0f;
    half* x_dev = upload_fp16(x_host);

    int q8_blocks = K / 32;
    int padded = ((K + 255) / 256) * 8;
    block_q8_1* q8_buf = nullptr;
    float* d8_buf = nullptr;
    cudaMalloc(&q8_buf, padded * sizeof(block_q8_1));
    cudaMalloc(&d8_buf, padded * sizeof(float));
    cudaMemset(q8_buf, 0, padded * sizeof(block_q8_1));
    cudaMemset(d8_buf, 0, padded * sizeof(float));
    quantize_fp16_to_q8_1(x_dev, q8_buf, d8_buf, K, nullptr);

    half* r_dev = nullptr;
    cudaMalloc(&r_dev, M * sizeof(half));
    cudaMemset(r_dev, 0, M * sizeof(half));

    int n_words = (q8_blocks + 31) / 32;
    uint32_t* mask_dev = nullptr;
    cudaMalloc(&mask_dev, static_cast<size_t>(n_words) * sizeof(uint32_t));

    // First run: all-1 mask
    std::vector<uint32_t> all_ones(n_words, ~0u);
    cudaMemcpy(mask_dev, all_ones.data(), static_cast<size_t>(n_words) * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    half* y_full = nullptr;
    cudaMalloc(&y_full, M * sizeof(half));
    gemv_q8_0_q8_1_residual_masked(W_dev, q8_buf, d8_buf, mask_dev, y_full, r_dev, M, K, nullptr);

    // Second run: zero the second half of blocks
    std::vector<uint32_t> half_mask(n_words, 0u);
    for (int b = 0; b < q8_blocks / 2; ++b) {
        half_mask[b >> 5] |= (1u << (b & 31));
    }
    cudaMemcpy(mask_dev, half_mask.data(), static_cast<size_t>(n_words) * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    half* y_half = nullptr;
    cudaMalloc(&y_half, M * sizeof(half));
    gemv_q8_0_q8_1_residual_masked(W_dev, q8_buf, d8_buf, mask_dev, y_half, r_dev, M, K, nullptr);
    cudaDeviceSynchronize();

    std::vector<half> hy_full(M), hy_half(M);
    cudaMemcpy(hy_full.data(), y_full, M * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy_half.data(), y_half, M * sizeof(half), cudaMemcpyDeviceToHost);

    // Outputs must differ (skipped blocks dropped) — sanity check that the mask is consumed.
    bool any_diff = false;
    for (int i = 0; i < M; ++i) {
        if (__half_as_ushort(hy_full[i]) != __half_as_ushort(hy_half[i])) any_diff = true;
    }
    EXPECT_TRUE(any_diff) << "masked GEMV ignored the mask";

    cudaFree(W_dev);
    cudaFree(x_dev);
    cudaFree(q8_buf);
    cudaFree(d8_buf);
    cudaFree(r_dev);
    cudaFree(mask_dev);
    cudaFree(y_full);
    cudaFree(y_half);
}

// -----------------------------------------------------------------------------
// Probe smoke test: probe entry-points are safe to call from a test fixture
// regardless of whether the probe is enabled (default off). flush also no-ops.
// -----------------------------------------------------------------------------
TEST(FFNSparsity, ProbeOffNoOps) {
    // Default config has sparsity_probe = false.
    const int K = 256;
    std::vector<float> g(K, 0.5f), u(K, 0.5f);
    half* g_dev = upload_fp16(g);
    half* u_dev = upload_fp16(u);
    // Should not crash even when probe is off (no cudaMalloc, no kernel launch).
    probe_ffn_silu_sparsity(0, g_dev, u_dev, K, nullptr);
    flush_ffn_sparsity_probe_log();
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    cudaFree(g_dev);
    cudaFree(u_dev);
}

}  // namespace
}  // namespace imp
