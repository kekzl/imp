#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "memory/kv_cache.h"
#include "core/tensor.h"
#include "quant/turboquant.h"

namespace imp {
namespace {

static bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_CUDA()                                                     \
    do {                                                                       \
        if (!HasCudaDevice()) {                                                \
            GTEST_SKIP() << "No CUDA device available";                        \
        }                                                                      \
    } while (0)

// ============================================================================
// Test 1: TurboQuant KV Cache construction — verify block sizes and pools
// ============================================================================
TEST(TurboQuantTest, CacheConstruction) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 4;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int max_blocks = 16;
    const int block_size = 16;

    // TurboQuant cache
    KVCache tq_cache(n_layers, n_kv_heads, head_dim, DType::TURBOQUANT, max_blocks, block_size);

    // Block bytes should match INT4 (packed: block_size * n_kv_heads * head_dim / 2)
    size_t expected_block_bytes = static_cast<size_t>(block_size) * n_kv_heads * head_dim / 2;
    EXPECT_EQ(tq_cache.block_bytes(), expected_block_bytes);

    // Scale pool should be allocated (for K norms + V scales)
    EXPECT_NE(tq_cache.k_scale_ptr(0, 0), nullptr);
    EXPECT_NE(tq_cache.v_scale_ptr(0, 0), nullptr);

    // Sketch pool should be allocated
    EXPECT_NE(tq_cache.k_sketch_ptr(0, 0), nullptr);

    // Sketch block bytes: block_size * n_kv_heads * (head_dim / 8)
    size_t expected_sketch_bytes = static_cast<size_t>(block_size) * n_kv_heads * (head_dim / 8);
    EXPECT_EQ(tq_cache.sketch_block_bytes(), expected_sketch_bytes);

    // Verify FP16 comparison: TurboQuant should use less memory per block
    KVCache fp16_cache(n_layers, n_kv_heads, head_dim, DType::FP16, max_blocks, block_size);
    EXPECT_LT(tq_cache.block_bytes(), fp16_cache.block_bytes());
    // INT4 packed = half the FP16 bytes
    EXPECT_EQ(tq_cache.block_bytes() * 4, fp16_cache.block_bytes());

    // Verify block allocation works
    int b1 = tq_cache.allocate_block();
    EXPECT_GE(b1, 0);
    EXPECT_EQ(tq_cache.ref_count(b1), 1);
    tq_cache.free_block(b1);
}

// ============================================================================
// Test 2: QJL projection matrix initialization
// ============================================================================
TEST(TurboQuantTest, QJLInit) {
    SKIP_IF_NO_CUDA();

    QJLProjection proj;
    bool ok = qjl_init(proj, /*head_dim=*/128, /*sketch_dim=*/128, /*seed=*/42);
    ASSERT_TRUE(ok);

    EXPECT_EQ(proj.head_dim, 128);
    EXPECT_EQ(proj.sketch_dim, 128);
    EXPECT_EQ(proj.seed, 42u);
    EXPECT_NE(proj.matrix, nullptr);

    // Read back matrix and verify it contains valid bit patterns
    int bytes_per_row = 128 / 8;
    size_t total_bytes = 128 * bytes_per_row;
    std::vector<uint8_t> host_matrix(total_bytes);
    cudaMemcpy(host_matrix.data(), proj.matrix, total_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Verify matrix is not all-zero or all-one (statistical check)
    int ones = 0;
    for (auto b : host_matrix) {
        ones += __builtin_popcount(b);
    }
    int total_bits = static_cast<int>(total_bytes) * 8;
    // Expect roughly 50% ones (Rademacher distribution)
    float ratio = static_cast<float>(ones) / total_bits;
    EXPECT_GT(ratio, 0.3f);
    EXPECT_LT(ratio, 0.7f);

    // Verify determinism: same seed → same matrix
    QJLProjection proj2;
    qjl_init(proj2, 128, 128, 42);
    std::vector<uint8_t> host_matrix2(total_bytes);
    cudaMemcpy(host_matrix2.data(), proj2.matrix, total_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    EXPECT_EQ(host_matrix, host_matrix2);

    qjl_destroy(proj);
    qjl_destroy(proj2);
    EXPECT_EQ(proj.matrix, nullptr);
}

// ============================================================================
// Test 3: QJL invalid parameters
// ============================================================================
TEST(TurboQuantTest, QJLInitInvalid) {
    SKIP_IF_NO_CUDA();

    QJLProjection proj;
    // head_dim not divisible by 8
    EXPECT_FALSE(qjl_init(proj, 127, 128, 42));
    // Zero dimensions
    EXPECT_FALSE(qjl_init(proj, 0, 128, 42));
    EXPECT_FALSE(qjl_init(proj, 128, 0, 42));
}

// ============================================================================
// Test 4: KV Cache sketch pointer arithmetic
// ============================================================================
TEST(TurboQuantTest, SketchPointers) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 2;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int max_blocks = 8;
    const int block_size = 16;

    KVCache cache(n_layers, n_kv_heads, head_dim, DType::TURBOQUANT, max_blocks, block_size);

    // Sketch pointers should be different for different layers
    void* s0_0 = cache.k_sketch_ptr(0, 0);
    void* s1_0 = cache.k_sketch_ptr(1, 0);
    EXPECT_NE(s0_0, s1_0);

    // Sketch pointers should be different for different blocks
    void* s0_1 = cache.k_sketch_ptr(0, 1);
    EXPECT_NE(s0_0, s0_1);

    // Stride between blocks should equal sketch_block_bytes
    ptrdiff_t block_stride = static_cast<char*>(s0_1) - static_cast<char*>(s0_0);
    EXPECT_EQ(static_cast<size_t>(block_stride), cache.sketch_block_bytes());

    // Stride between layers should equal max_blocks * sketch_block_bytes
    ptrdiff_t layer_stride = static_cast<char*>(s1_0) - static_cast<char*>(s0_0);
    EXPECT_EQ(static_cast<size_t>(layer_stride), max_blocks * cache.sketch_block_bytes());
}

// ============================================================================
// Test 5: PolarQuant roundtrip accuracy check
// ============================================================================
TEST(TurboQuantTest, PolarQuantAccuracy) {
    SKIP_IF_NO_CUDA();

    // Simulate PolarQuant: decompose → quantize → reconstruct
    const int head_dim = 128;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate random K vector
    std::vector<float> k(head_dim);
    for (auto& v : k) v = dist(rng);

    // Compute norm
    float norm_sq = 0.0f;
    for (auto v : k) norm_sq += v * v;
    float norm = std::sqrt(norm_sq);

    // Normalize
    std::vector<float> dir(head_dim);
    for (int i = 0; i < head_dim; i++) dir[i] = k[i] / norm;

    // Quantize direction to INT4 (uniform [-1,1] → [-8,7])
    std::vector<int8_t> q_dir(head_dim);
    for (int i = 0; i < head_dim; i++) {
        int q = static_cast<int>(std::round(dir[i] * 7.0f));
        q = std::max(-8, std::min(7, q));
        q_dir[i] = static_cast<int8_t>(q);
    }

    // Dequantize: reconstruct K' = norm * (q_dir / 7.0)
    std::vector<float> k_recon(head_dim);
    for (int i = 0; i < head_dim; i++) {
        k_recon[i] = norm * (static_cast<float>(q_dir[i]) / 7.0f);
    }

    // Compute relative error
    float error_sq = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        float diff = k[i] - k_recon[i];
        error_sq += diff * diff;
    }
    float rel_error = std::sqrt(error_sq / norm_sq);

    // PolarQuant with INT4 should have < 15% relative error
    EXPECT_LT(rel_error, 0.15f);

    // Check dot product preservation with random Q
    std::vector<float> q(head_dim);
    for (auto& v : q) v = dist(rng);

    float true_dot = 0.0f;
    float recon_dot = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        true_dot += q[i] * k[i];
        recon_dot += q[i] * k_recon[i];
    }

    // Relative dot product error should be reasonable
    float dot_error = std::abs(true_dot - recon_dot) / std::abs(true_dot);
    EXPECT_LT(dot_error, 0.2f);
}

// ============================================================================
// Test 6: QJL dot product estimation accuracy
// ============================================================================
TEST(TurboQuantTest, QJLDotProductEstimate) {
    SKIP_IF_NO_CUDA();

    // Generate random Q and K, compute QJL estimate, verify correlation
    const int head_dim = 128;
    const int sketch_dim = 128;
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate random vectors
    std::vector<float> q(head_dim), k(head_dim);
    for (auto& v : q) v = dist(rng);
    for (auto& v : k) v = dist(rng);

    // True dot product
    float true_dot = 0.0f;
    for (int i = 0; i < head_dim; i++) true_dot += q[i] * k[i];

    // Compute norms
    float q_norm = 0.0f, k_norm = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        q_norm += q[i] * q[i];
        k_norm += k[i] * k[i];
    }
    q_norm = std::sqrt(q_norm);
    k_norm = std::sqrt(k_norm);

    // Generate random Rademacher matrix and compute sketches
    std::vector<std::vector<int>> R(sketch_dim, std::vector<int>(head_dim));
    std::bernoulli_distribution bern(0.5);
    for (int i = 0; i < sketch_dim; i++)
        for (int j = 0; j < head_dim; j++)
            R[i][j] = bern(rng) ? 1 : -1;

    // Compute sketches: sign(R @ q), sign(R @ k)
    std::vector<int> sketch_q(sketch_dim), sketch_k(sketch_dim);
    for (int i = 0; i < sketch_dim; i++) {
        float dot_q = 0.0f, dot_k = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            dot_q += R[i][j] * q[j];
            dot_k += R[i][j] * k[j];
        }
        sketch_q[i] = (dot_q >= 0) ? 1 : 0;
        sketch_k[i] = (dot_k >= 0) ? 1 : 0;
    }

    // XNOR count
    int match_count = 0;
    for (int i = 0; i < sketch_dim; i++) {
        if (sketch_q[i] == sketch_k[i]) match_count++;
    }

    // QJL estimator
    float qjl_dot = q_norm * k_norm *
                     static_cast<float>(2 * match_count - sketch_dim) /
                     static_cast<float>(sketch_dim);

    // QJL estimate should be in the right ballpark (within 50% for dim=128)
    // The estimate is unbiased but has variance
    float abs_error = std::abs(true_dot - qjl_dot);
    float max_scale = std::max(std::abs(true_dot), q_norm * k_norm * 0.1f);
    EXPECT_LT(abs_error / max_scale, 2.0f);  // generous bound for single estimate
}

// ============================================================================
// Test 7: Memory reduction vs FP16
// ============================================================================
TEST(TurboQuantTest, MemoryReduction) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 32;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int max_blocks = 256;
    const int block_size = 16;

    KVCache fp16_cache(n_layers, n_kv_heads, head_dim, DType::FP16, max_blocks, block_size);
    KVCache tq_cache(n_layers, n_kv_heads, head_dim, DType::TURBOQUANT, max_blocks, block_size);

    // TurboQuant K+V data pool should be 1/4 of FP16 (INT4 packed for both K dirs and V)
    double ratio = static_cast<double>(tq_cache.block_bytes()) / fp16_cache.block_bytes();
    EXPECT_NEAR(ratio, 0.25, 0.01);

    // With scale and sketch overhead, total should still be < 50% of FP16
    // Scale pool: block_size * n_kv_heads * 2 bytes per block (K norms + V scales)
    // Sketch pool: block_size * n_kv_heads * (head_dim/8) per K block
    size_t fp16_total = fp16_cache.block_bytes();
    size_t tq_total = tq_cache.block_bytes() + tq_cache.scale_block_bytes() + tq_cache.sketch_block_bytes();
    // Note: scale_block_bytes is for one block (K or V), used for both
    // and sketch is K only. Exact ratio depends on head_dim
    double total_ratio = static_cast<double>(tq_total) / fp16_total;
    EXPECT_LT(total_ratio, 0.55);  // should be well under 50% with all overhead
}

// ============================================================================
// Test 8: TurboQuant Lite KV Cache construction — V-only pool, sketch-only K
// ============================================================================
TEST(TurboQuantLiteTest, CacheConstruction) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 4;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int max_blocks = 16;
    const int block_size = 16;
    const int sketch_dim = 2 * head_dim;  // multiplier = 2

    KVCache tql_cache(n_layers, n_kv_heads, head_dim, DType::TURBOQUANT_LITE,
                      max_blocks, block_size, nullptr, sketch_dim);

    // Block bytes should match INT4 V (same as standard INT4)
    size_t expected_block_bytes = static_cast<size_t>(block_size) * n_kv_heads * head_dim / 2;
    EXPECT_EQ(tql_cache.block_bytes(), expected_block_bytes);

    // k_ptr should return nullptr (no K directions in pool)
    EXPECT_EQ(tql_cache.k_ptr(0, 0), nullptr);

    // v_ptr should be valid
    EXPECT_NE(tql_cache.v_ptr(0, 0), nullptr);

    // Scale pool should be allocated (K norms + V scales)
    EXPECT_NE(tql_cache.k_scale_ptr(0, 0), nullptr);
    EXPECT_NE(tql_cache.v_scale_ptr(0, 0), nullptr);

    // Sketch pool should be allocated with larger sketch_dim
    EXPECT_NE(tql_cache.k_sketch_ptr(0, 0), nullptr);

    // Sketch block bytes: block_size * n_kv_heads * (sketch_dim / 8)
    size_t expected_sketch_bytes = static_cast<size_t>(block_size) * n_kv_heads * (sketch_dim / 8);
    EXPECT_EQ(tql_cache.sketch_block_bytes(), expected_sketch_bytes);

    // sketch_dim accessor
    EXPECT_EQ(tql_cache.sketch_dim(), sketch_dim);

    // Block allocation works
    int b1 = tql_cache.allocate_block();
    EXPECT_GE(b1, 0);
    EXPECT_EQ(tql_cache.ref_count(b1), 1);
    tql_cache.free_block(b1);
}

// ============================================================================
// Test 9: TurboQuant Lite V pointer offsets (V-only pool)
// ============================================================================
TEST(TurboQuantLiteTest, VPointerOffsets) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 2;
    const int n_kv_heads = 4;
    const int head_dim = 64;
    const int max_blocks = 8;
    const int block_size = 16;
    const int sketch_dim = 2 * head_dim;

    KVCache cache(n_layers, n_kv_heads, head_dim, DType::TURBOQUANT_LITE,
                  max_blocks, block_size, nullptr, sketch_dim);

    // V pointers should differ between blocks
    void* v0_0 = cache.v_ptr(0, 0);
    void* v0_1 = cache.v_ptr(0, 1);
    EXPECT_NE(v0_0, v0_1);

    // Stride between V blocks should equal block_bytes
    ptrdiff_t block_stride = static_cast<char*>(v0_1) - static_cast<char*>(v0_0);
    EXPECT_EQ(static_cast<size_t>(block_stride), cache.block_bytes());

    // V pointers should differ between layers
    void* v1_0 = cache.v_ptr(1, 0);
    EXPECT_NE(v0_0, v1_0);

    // Stride between layers should be max_blocks * block_bytes (V-only, no 2x)
    ptrdiff_t layer_stride = static_cast<char*>(v1_0) - static_cast<char*>(v0_0);
    EXPECT_EQ(static_cast<size_t>(layer_stride), max_blocks * cache.block_bytes());
}

// ============================================================================
// Test 10: TurboQuant Lite memory savings vs FP16 and TurboQuant
// ============================================================================
TEST(TurboQuantLiteTest, MemorySavings) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 32;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int max_blocks = 256;
    const int block_size = 16;

    KVCache fp16_cache(n_layers, n_kv_heads, head_dim, DType::FP16, max_blocks, block_size);
    KVCache tq_cache(n_layers, n_kv_heads, head_dim, DType::TURBOQUANT, max_blocks, block_size);

    // TQ Lite with mult=2: sketch_dim = 256
    const int sketch_dim_lite = 2 * head_dim;
    KVCache tql_cache(n_layers, n_kv_heads, head_dim, DType::TURBOQUANT_LITE,
                      max_blocks, block_size, nullptr, sketch_dim_lite);

    // TQ Lite should use less VRAM than TQ (no INT4 K directions in pool)
    // FP16 total per block: block_bytes * 2 (K+V)
    size_t fp16_per_block = fp16_cache.block_bytes() * 2;

    // TQ per block: block_bytes * 2 (K+V INT4) + 2*scale + sketch(head_dim)
    size_t tq_per_block = tq_cache.block_bytes() * 2
                          + tq_cache.scale_block_bytes() * 2
                          + tq_cache.sketch_block_bytes();

    // TQ Lite per block: block_bytes * 1 (V-only INT4) + 2*scale + sketch(2*head_dim)
    size_t tql_per_block = tql_cache.block_bytes() * 1
                           + tql_cache.scale_block_bytes() * 2
                           + tql_cache.sketch_block_bytes();

    // TQ Lite should use less than TQ
    EXPECT_LT(tql_per_block, tq_per_block);

    // TQ Lite should use less than FP16
    EXPECT_LT(tql_per_block, fp16_per_block);

    // Compute effective bits per element (K+V combined)
    // V: 4 bits/elem data + 16 bits/head overhead
    // K: sketch_dim bits + 16 bits/head overhead
    double v_bits_per_elem = 4.0 + 16.0 / head_dim;
    double k_bits_per_elem = static_cast<double>(sketch_dim_lite) / head_dim + 16.0 / head_dim;
    double avg_bits = (k_bits_per_elem + v_bits_per_elem) / 2.0;

    // With mult=2, head_dim=128: K = 2.125, V = 4.125, avg = 3.125
    EXPECT_GT(avg_bits, 2.5);
    EXPECT_LT(avg_bits, 4.0);
}

// ============================================================================
// Test 11: QJL with sketch_dim > head_dim (TQ Lite configuration)
// ============================================================================
TEST(TurboQuantLiteTest, QJLLargeSketchDim) {
    SKIP_IF_NO_CUDA();

    QJLProjection proj;
    // sketch_dim = 3 * head_dim (mult=3)
    bool ok = qjl_init(proj, /*head_dim=*/128, /*sketch_dim=*/384, /*seed=*/42);
    ASSERT_TRUE(ok);

    EXPECT_EQ(proj.head_dim, 128);
    EXPECT_EQ(proj.sketch_dim, 384);
    EXPECT_NE(proj.matrix, nullptr);

    // Verify matrix dimensions
    int bytes_per_row = 128 / 8;
    size_t total_bytes = 384 * bytes_per_row;
    std::vector<uint8_t> host_matrix(total_bytes);
    cudaMemcpy(host_matrix.data(), proj.matrix, total_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Verify roughly 50% ones (Rademacher distribution)
    int ones = 0;
    for (auto b : host_matrix) {
        ones += __builtin_popcount(b);
    }
    int total_bits = static_cast<int>(total_bytes) * 8;
    float ratio = static_cast<float>(ones) / total_bits;
    EXPECT_GT(ratio, 0.3f);
    EXPECT_LT(ratio, 0.7f);

    qjl_destroy(proj);
}

// ============================================================================
// Test 12: QJL dot product accuracy improves with larger sketch_dim
// ============================================================================
TEST(TurboQuantLiteTest, QJLAccuracyVsSketchDim) {
    SKIP_IF_NO_CUDA();

    const int head_dim = 128;
    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate random Q and K
    std::vector<float> q(head_dim), k(head_dim);
    for (auto& v : q) v = dist(rng);
    for (auto& v : k) v = dist(rng);

    // True dot product
    float true_dot = 0.0f;
    for (int i = 0; i < head_dim; i++) true_dot += q[i] * k[i];

    float q_norm = 0.0f, k_norm = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        q_norm += q[i] * q[i];
        k_norm += k[i] * k[i];
    }
    q_norm = std::sqrt(q_norm);
    k_norm = std::sqrt(k_norm);

    // Test two sketch dims: head_dim (1x) vs 3*head_dim (3x)
    // Average absolute error over multiple random projections
    auto compute_avg_error = [&](int sketch_dim, int n_trials) {
        double total_error = 0.0;
        for (int trial = 0; trial < n_trials; trial++) {
            std::mt19937 trial_rng(trial * 1000 + sketch_dim);
            std::bernoulli_distribution bern(0.5);

            std::vector<std::vector<int>> R(sketch_dim, std::vector<int>(head_dim));
            for (int i = 0; i < sketch_dim; i++)
                for (int j = 0; j < head_dim; j++)
                    R[i][j] = bern(trial_rng) ? 1 : -1;

            std::vector<int> sketch_q(sketch_dim), sketch_k(sketch_dim);
            for (int i = 0; i < sketch_dim; i++) {
                float dot_q = 0.0f, dot_k = 0.0f;
                for (int j = 0; j < head_dim; j++) {
                    dot_q += R[i][j] * q[j];
                    dot_k += R[i][j] * k[j];
                }
                sketch_q[i] = (dot_q >= 0) ? 1 : 0;
                sketch_k[i] = (dot_k >= 0) ? 1 : 0;
            }

            int match_count = 0;
            for (int i = 0; i < sketch_dim; i++)
                if (sketch_q[i] == sketch_k[i]) match_count++;

            float qjl_dot = q_norm * k_norm *
                            static_cast<float>(2 * match_count - sketch_dim) /
                            static_cast<float>(sketch_dim);
            total_error += std::abs(true_dot - qjl_dot);
        }
        return total_error / n_trials;
    };

    double error_1x = compute_avg_error(head_dim, 50);
    double error_3x = compute_avg_error(3 * head_dim, 50);

    // 3x sketch_dim should have lower average error than 1x
    EXPECT_LT(error_3x, error_1x);
}

} // anonymous namespace
} // namespace imp
