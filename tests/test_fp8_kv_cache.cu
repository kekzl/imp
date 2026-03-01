#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "memory/kv_cache.h"
#include "core/tensor.h"
#include "quant/fp8_quant.h"
#include "compute/attention_paged.h"

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
// Test 1: FP8 KV Cache construction — verify half-sized blocks
// ============================================================================
TEST(FP8KVCache, Construction) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 4;
    const int n_kv_heads = 8;
    const int head_dim = 64;
    const int max_blocks = 16;

    // FP16 cache
    KVCache fp16_cache(n_layers, n_kv_heads, head_dim, DType::FP16, max_blocks);
    size_t fp16_block_bytes = fp16_cache.block_bytes();

    // FP8 cache
    KVCache fp8_cache(n_layers, n_kv_heads, head_dim, DType::FP8_E4M3, max_blocks);
    size_t fp8_block_bytes = fp8_cache.block_bytes();

    // FP8 blocks should be exactly half the size of FP16
    EXPECT_EQ(fp8_block_bytes * 2, fp16_block_bytes);

    // Verify accessor consistency
    EXPECT_EQ(fp8_cache.n_layers(), n_layers);
    EXPECT_EQ(fp8_cache.n_kv_heads(), n_kv_heads);
    EXPECT_EQ(fp8_cache.head_dim(), head_dim);
    EXPECT_EQ(fp8_cache.dtype(), DType::FP8_E4M3);
    EXPECT_EQ(fp8_cache.total_blocks(), max_blocks);
    EXPECT_EQ(fp8_cache.num_free_blocks(), max_blocks);

    // Allocate a block and verify pointers are valid (non-null, distinct K/V)
    int blk = fp8_cache.allocate_block();
    ASSERT_GE(blk, 0);
    void* k = fp8_cache.k_ptr(0, blk);
    void* v = fp8_cache.v_ptr(0, blk);
    EXPECT_NE(k, nullptr);
    EXPECT_NE(v, nullptr);
    EXPECT_NE(k, v);
    fp8_cache.free_block(blk);
}

// ============================================================================
// Test 2: FP8 scale calibration — verify calibrate_fp8_scale()
// ============================================================================
TEST(FP8KVCache, ScaleCalibration) {
    SKIP_IF_NO_CUDA();

    const int n = 256;
    const int k = 128;
    const float expected_max = 2.5f;

    // Create FP16 tensor with known max absolute value
    std::vector<half> h_data(n * k);
    for (int i = 0; i < n * k; i++) {
        // Values in [-expected_max, expected_max]
        float val = expected_max * (2.0f * (i % 100) / 99.0f - 1.0f);
        h_data[i] = __float2half(val);
    }

    // Upload to device
    void* d_data = nullptr;
    cudaMalloc(&d_data, n * k * sizeof(half));
    cudaMemcpy(d_data, h_data.data(), n * k * sizeof(half), cudaMemcpyHostToDevice);

    int64_t shape[2] = {n, k};
    Tensor t(d_data, DType::FP16, 2, shape, true);

    float scale = calibrate_fp8_scale(t, nullptr);
    cudaDeviceSynchronize();

    // Scale should be absmax / 448.0
    float expected_scale = expected_max / 448.0f;

    // Allow small tolerance for FP16 rounding
    EXPECT_NEAR(scale, expected_scale, expected_scale * 0.02f);

    cudaFree(d_data);
}

// ============================================================================
// Test 3: FP8 scale calibration — all zeros should return safe scale
// ============================================================================
TEST(FP8KVCache, ScaleCalibrationZeros) {
    SKIP_IF_NO_CUDA();

    const int n = 64;
    const int k = 64;

    // Zero-initialized tensor
    void* d_data = nullptr;
    cudaMalloc(&d_data, n * k * sizeof(half));
    cudaMemset(d_data, 0, n * k * sizeof(half));

    int64_t shape[2] = {n, k};
    Tensor t(d_data, DType::FP16, 2, shape, true);

    float scale = calibrate_fp8_scale(t, nullptr);
    cudaDeviceSynchronize();

    // calibrate_fp8_scale returns 1.0 as a safety fallback for all-zero tensors,
    // preventing division by zero in the quantization path.
    EXPECT_FLOAT_EQ(scale, 1.0f);

    cudaFree(d_data);
}

// ============================================================================
// Test 4: FP8 KV cache block_bytes matches dtype_size
// ============================================================================
TEST(FP8KVCache, BlockBytesMatchesDTypeSize) {
    SKIP_IF_NO_CUDA();

    const int n_kv_heads = 4;
    const int head_dim = 128;
    const int max_blocks = 8;

    KVCache cache(1, n_kv_heads, head_dim, DType::FP8_E4M3, max_blocks);

    // block_bytes = kKVBlockSize * n_kv_heads * head_dim * dtype_size(FP8_E4M3)
    size_t expected = static_cast<size_t>(kKVBlockSize) * n_kv_heads * head_dim *
                      dtype_size(DType::FP8_E4M3);
    EXPECT_EQ(cache.block_bytes(), expected);
    EXPECT_EQ(dtype_size(DType::FP8_E4M3), 1u);  // FP8 = 1 byte per element
}

// ============================================================================
// Test 5: FP8 Split-K decode consistency vs FP16 reference
// ============================================================================
//
// Creates a small paged attention scenario (batch=1, heads=8, hd=128, ctx=256),
// runs FP16 decode for reference, quantizes KV to FP8, runs FP8 decode with
// Split-K scratch enabled, and checks relative error < 2%.
// ============================================================================

TEST(FP8KVCache, SplitKConsistency) {
    SKIP_IF_NO_CUDA();

    // ---- Config ----
    const int batch_size = 1;
    const int n_heads = 8;
    const int n_kv_heads = 8;  // MHA for simplicity
    const int head_dim = 128;
    const int ctx_len = 256;
    const int block_size = kKVBlockSize;  // 16
    const int num_blocks = (ctx_len + block_size - 1) / block_size;  // 16
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // ---- Allocate Q [batch, 1, n_heads, head_dim] on device ----
    // Use values in [-1, 1] range for meaningful FP8 precision
    const int q_elems = batch_size * 1 * n_heads * head_dim;
    std::vector<half> h_q(q_elems);
    srand(42);
    for (int i = 0; i < q_elems; i++) {
        h_q[i] = __float2half(((rand() % 200) - 100) / 100.0f);
    }
    void* d_q = nullptr;
    cudaMalloc(&d_q, q_elems * sizeof(half));
    cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice);

    // ---- Allocate FP16 KV cache [num_blocks, block_size, n_kv_heads, head_dim] ----
    const int kv_elems_per_block = block_size * n_kv_heads * head_dim;
    const int total_kv_elems = num_blocks * kv_elems_per_block;
    std::vector<half> h_kv(total_kv_elems);
    for (int i = 0; i < total_kv_elems; i++) {
        h_kv[i] = __float2half(0.5f * ((rand() % 200) - 100) / 100.0f);
    }

    void* d_k_fp16 = nullptr;
    void* d_v_fp16 = nullptr;
    cudaMalloc(&d_k_fp16, total_kv_elems * sizeof(half));
    cudaMalloc(&d_v_fp16, total_kv_elems * sizeof(half));
    cudaMemcpy(d_k_fp16, h_kv.data(), total_kv_elems * sizeof(half), cudaMemcpyHostToDevice);
    // Use different random data for V
    for (int i = 0; i < total_kv_elems; i++) {
        h_kv[i] = __float2half(0.5f * ((rand() % 200) - 100) / 100.0f);
    }
    cudaMemcpy(d_v_fp16, h_kv.data(), total_kv_elems * sizeof(half), cudaMemcpyHostToDevice);

    // ---- Block tables: identity mapping [0, 1, 2, ..., num_blocks-1] ----
    std::vector<int> h_bt(num_blocks);
    std::iota(h_bt.begin(), h_bt.end(), 0);
    int* d_bt = nullptr;
    cudaMalloc(&d_bt, num_blocks * sizeof(int));
    cudaMemcpy(d_bt, h_bt.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    // ---- Context lens: [ctx_len] ----
    int* d_ctx = nullptr;
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_ctx, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);

    // ---- Quantize KV to FP8 ----
    int64_t q_shape[4] = {batch_size, 1, n_heads, head_dim};
    int64_t kv_shape[4] = {num_blocks, block_size, n_kv_heads, head_dim};
    Tensor t_k16(d_k_fp16, DType::FP16, 4, kv_shape, true);
    Tensor t_v16(d_v_fp16, DType::FP16, 4, kv_shape, true);

    float k_scale = calibrate_fp8_scale(t_k16, nullptr);
    cudaDeviceSynchronize();
    float v_scale = calibrate_fp8_scale(t_v16, nullptr);
    cudaDeviceSynchronize();
    float kv_scale = fmaxf(k_scale, v_scale);
    if (kv_scale <= 0.0f) kv_scale = 1.0f;

    void* d_k_fp8 = nullptr;
    void* d_v_fp8 = nullptr;
    cudaMalloc(&d_k_fp8, total_kv_elems);  // 1 byte per element
    cudaMalloc(&d_v_fp8, total_kv_elems);
    quantize_fp16_to_fp8_e4m3_scaled(d_k_fp16, d_k_fp8, total_kv_elems, kv_scale, nullptr);
    quantize_fp16_to_fp8_e4m3_scaled(d_v_fp16, d_v_fp8, total_kv_elems, kv_scale, nullptr);
    cudaDeviceSynchronize();

    Tensor t_q(d_q, DType::FP16, 4, q_shape, true);
    Tensor t_k8(d_k_fp8, DType::FP8_E4M3, 4, kv_shape, true);
    Tensor t_v8(d_v_fp8, DType::FP8_E4M3, 4, kv_shape, true);

    // ---- Output buffers ----
    void* d_o_nosplit = nullptr;
    void* d_o_splitk = nullptr;
    cudaMalloc(&d_o_nosplit, q_elems * sizeof(half));
    cudaMalloc(&d_o_splitk, q_elems * sizeof(half));
    cudaMemset(d_o_nosplit, 0, q_elems * sizeof(half));
    cudaMemset(d_o_splitk, 0, q_elems * sizeof(half));

    // ---- Run FP8 decode WITHOUT Split-K (reference) ----
    Tensor t_o_nosplit(d_o_nosplit, DType::FP16, 4, q_shape, true);
    paged_attention_set_splitk_scratch(nullptr, 0);  // force non-split-K
    paged_attention_decode_fp8(t_q, t_k8, t_v8, t_o_nosplit,
                               d_bt, d_ctx, block_size, scale, kv_scale,
                               ctx_len, 0, 0.0f, nullptr);
    cudaDeviceSynchronize();

    // ---- Run FP8 decode WITH Split-K ----
    const int max_splits = 32;
    size_t scratch_size = (size_t)batch_size * n_heads * max_splits
                          * (2 + head_dim) * sizeof(float);
    void* d_scratch = nullptr;
    cudaMalloc(&d_scratch, scratch_size);

    Tensor t_o_splitk(d_o_splitk, DType::FP16, 4, q_shape, true);
    paged_attention_set_splitk_scratch(d_scratch, scratch_size);
    paged_attention_decode_fp8(t_q, t_k8, t_v8, t_o_splitk,
                               d_bt, d_ctx, block_size, scale, kv_scale,
                               ctx_len, 0, 0.0f, nullptr);
    cudaDeviceSynchronize();

    // ---- Compare Split-K vs non-Split-K (same FP8 data, should match closely) ----
    std::vector<half> h_o_nosplit(q_elems);
    std::vector<half> h_o_splitk(q_elems);
    cudaMemcpy(h_o_nosplit.data(), d_o_nosplit, q_elems * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_o_splitk.data(), d_o_splitk, q_elems * sizeof(half), cudaMemcpyDeviceToHost);

    double sum_sq_err = 0.0;
    float max_abs_ref = 0.0f;
    float max_abs_err = 0.0f;
    for (int i = 0; i < q_elems; i++) {
        float ref = __half2float(h_o_nosplit[i]);
        float sk  = __half2float(h_o_splitk[i]);
        float err = ref - sk;
        sum_sq_err += (double)err * err;
        max_abs_ref = fmaxf(max_abs_ref, fabsf(ref));
        max_abs_err = fmaxf(max_abs_err, fabsf(err));
    }
    float rmse = sqrtf((float)(sum_sq_err / q_elems));

    // Split-K vs non-Split-K should agree within 2% (only FP32 accumulation order differs)
    float rel_rmse = (max_abs_ref > 1e-6f) ? (rmse / max_abs_ref) : rmse;
    EXPECT_LT(rel_rmse, 0.02f)
        << "FP8 Split-K vs non-Split-K: relative RMSE = "
        << (rel_rmse * 100.0f) << "% (rmse=" << rmse
        << ", max_abs_err=" << max_abs_err
        << ", max_abs_ref=" << max_abs_ref << ")";

    // ---- Cleanup ----
    paged_attention_set_splitk_scratch(nullptr, 0);
    cudaFree(d_q);
    cudaFree(d_k_fp16);
    cudaFree(d_v_fp16);
    cudaFree(d_k_fp8);
    cudaFree(d_v_fp8);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    cudaFree(d_o_nosplit);
    cudaFree(d_o_splitk);
    cudaFree(d_scratch);
}

// ============================================================================
// Test 6: INT8 KV Cache construction — verify scale pool allocation
// ============================================================================
TEST(INT8KVCache, Construction) {
    SKIP_IF_NO_CUDA();

    const int n_layers = 4;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int max_blocks = 16;

    // FP16 cache for size comparison
    KVCache fp16_cache(n_layers, n_kv_heads, head_dim, DType::FP16, max_blocks);
    size_t fp16_block_bytes = fp16_cache.block_bytes();

    // INT8 cache
    KVCache int8_cache(n_layers, n_kv_heads, head_dim, DType::INT8, max_blocks);
    size_t int8_block_bytes = int8_cache.block_bytes();

    // INT8 blocks should be exactly half the size of FP16
    EXPECT_EQ(int8_block_bytes * 2, fp16_block_bytes);

    // Verify accessors
    EXPECT_EQ(int8_cache.dtype(), DType::INT8);
    EXPECT_EQ(int8_cache.n_kv_heads(), n_kv_heads);
    EXPECT_EQ(int8_cache.head_dim(), head_dim);

    // INT8 scale pool should be allocated
    int blk = int8_cache.allocate_block();
    ASSERT_GE(blk, 0);
    EXPECT_NE(int8_cache.k_scale_ptr(0, blk), nullptr);
    EXPECT_NE(int8_cache.v_scale_ptr(0, blk), nullptr);
    EXPECT_NE(int8_cache.k_scale_ptr(0, blk), int8_cache.v_scale_ptr(0, blk));

    // Scale block bytes = kKVBlockSize * n_kv_heads * sizeof(half)
    size_t expected_scale = static_cast<size_t>(kKVBlockSize) * n_kv_heads * sizeof(half);
    EXPECT_EQ(int8_cache.scale_block_bytes(), expected_scale);

    // FP16 cache should NOT have scale pool
    int blk2 = fp16_cache.allocate_block();
    EXPECT_EQ(fp16_cache.k_scale_ptr(0, blk2), nullptr);

    int8_cache.free_block(blk);
    fp16_cache.free_block(blk2);
}

// ============================================================================
// Test 7: INT8 Split-K decode consistency
// ============================================================================
//
// Creates a small paged attention scenario (batch=1, heads=8, hd=128, ctx=256),
// manually quantizes FP16 KV data to INT8 with per-head scales on the host,
// uploads to device, runs INT8 decode with and without Split-K,
// and checks relative error < 2%.
// ============================================================================

// Host helper: quantize FP16 KV data to INT8 with per-head scales.
// KV layout: [num_blocks, block_size, n_kv_heads, head_dim]
// Scale layout: [num_blocks, block_size, n_kv_heads]
static void quantize_kv_to_int8_host(
    const std::vector<half>& fp16_data,
    std::vector<int8_t>& int8_data,
    std::vector<half>& scales,
    int num_blocks, int block_size, int n_kv_heads, int head_dim)
{
    const int total_tokens = num_blocks * block_size;
    int8_data.resize(fp16_data.size());
    scales.resize(total_tokens * n_kv_heads);

    for (int tok = 0; tok < total_tokens; tok++) {
        for (int h = 0; h < n_kv_heads; h++) {
            int base = tok * n_kv_heads * head_dim + h * head_dim;

            // Find absmax for this head
            float amax = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float val = __half2float(fp16_data[base + d]);
                amax = std::max(amax, std::abs(val));
            }

            float sc = amax / 127.0f;
            float inv_sc = (amax > 1e-8f) ? (127.0f / amax) : 0.0f;

            // Quantize
            for (int d = 0; d < head_dim; d++) {
                float val = __half2float(fp16_data[base + d]);
                int rounded = static_cast<int>(std::round(val * inv_sc));
                rounded = std::max(-127, std::min(127, rounded));
                int8_data[base + d] = static_cast<int8_t>(rounded);
            }

            // Store scale
            scales[tok * n_kv_heads + h] = __float2half(sc);
        }
    }
}

TEST(INT8KVCache, SplitKConsistency) {
    SKIP_IF_NO_CUDA();

    // ---- Config ----
    const int batch_size = 1;
    const int n_heads = 8;
    const int n_kv_heads = 8;
    const int head_dim = 128;
    const int ctx_len = 256;
    const int block_size = kKVBlockSize;
    const int num_blocks = (ctx_len + block_size - 1) / block_size;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // ---- Allocate Q ----
    const int q_elems = batch_size * n_heads * head_dim;
    std::vector<half> h_q(q_elems);
    srand(42);
    for (int i = 0; i < q_elems; i++) {
        h_q[i] = __float2half(((rand() % 200) - 100) / 100.0f);
    }
    void* d_q = nullptr;
    cudaMalloc(&d_q, q_elems * sizeof(half));
    cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice);

    // ---- Generate random FP16 KV data ----
    const int kv_elems = num_blocks * block_size * n_kv_heads * head_dim;
    std::vector<half> h_k_fp16(kv_elems);
    std::vector<half> h_v_fp16(kv_elems);
    for (int i = 0; i < kv_elems; i++) {
        h_k_fp16[i] = __float2half(0.5f * ((rand() % 200) - 100) / 100.0f);
        h_v_fp16[i] = __float2half(0.5f * ((rand() % 200) - 100) / 100.0f);
    }

    // ---- Quantize to INT8 on host ----
    std::vector<int8_t> h_k_int8, h_v_int8;
    std::vector<half> h_k_scales, h_v_scales;
    quantize_kv_to_int8_host(h_k_fp16, h_k_int8, h_k_scales,
                              num_blocks, block_size, n_kv_heads, head_dim);
    quantize_kv_to_int8_host(h_v_fp16, h_v_int8, h_v_scales,
                              num_blocks, block_size, n_kv_heads, head_dim);

    // ---- Upload to device ----
    void* d_k_int8 = nullptr;
    void* d_v_int8 = nullptr;
    void* d_k_scales = nullptr;
    void* d_v_scales = nullptr;
    cudaMalloc(&d_k_int8, kv_elems * sizeof(int8_t));
    cudaMalloc(&d_v_int8, kv_elems * sizeof(int8_t));
    cudaMalloc(&d_k_scales, h_k_scales.size() * sizeof(half));
    cudaMalloc(&d_v_scales, h_v_scales.size() * sizeof(half));
    cudaMemcpy(d_k_int8, h_k_int8.data(), kv_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_int8, h_v_int8.data(), kv_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_scales, h_k_scales.data(), h_k_scales.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_scales, h_v_scales.data(), h_v_scales.size() * sizeof(half), cudaMemcpyHostToDevice);

    // ---- Block tables: identity mapping ----
    std::vector<int> h_bt(num_blocks);
    std::iota(h_bt.begin(), h_bt.end(), 0);
    int* d_bt = nullptr;
    cudaMalloc(&d_bt, num_blocks * sizeof(int));
    cudaMemcpy(d_bt, h_bt.data(), num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    int* d_ctx = nullptr;
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_ctx, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);

    // ---- Create tensors ----
    int64_t q_shape[4] = {batch_size, 1, n_heads, head_dim};
    int64_t kv_shape[4] = {num_blocks, block_size, n_kv_heads, head_dim};
    Tensor t_q(d_q, DType::FP16, 4, q_shape, true);
    Tensor t_k_i8(d_k_int8, DType::INT8, 4, kv_shape, true);
    Tensor t_v_i8(d_v_int8, DType::INT8, 4, kv_shape, true);

    // ---- Output buffers ----
    void* d_o_nosplit = nullptr;
    void* d_o_splitk = nullptr;
    cudaMalloc(&d_o_nosplit, q_elems * sizeof(half));
    cudaMalloc(&d_o_splitk, q_elems * sizeof(half));
    cudaMemset(d_o_nosplit, 0, q_elems * sizeof(half));
    cudaMemset(d_o_splitk, 0, q_elems * sizeof(half));

    // ---- Run INT8 decode WITHOUT Split-K ----
    Tensor t_o_nosplit(d_o_nosplit, DType::FP16, 4, q_shape, true);
    paged_attention_set_splitk_scratch(nullptr, 0);
    paged_attention_decode_int8(t_q, t_k_i8, t_v_i8, t_o_nosplit,
                                static_cast<const half*>(d_k_scales),
                                static_cast<const half*>(d_v_scales),
                                d_bt, d_ctx, block_size, scale,
                                ctx_len, 0, 0.0f, nullptr);
    cudaDeviceSynchronize();

    // ---- Run INT8 decode WITH Split-K ----
    const int max_splits = 32;
    size_t scratch_size = (size_t)batch_size * n_heads * max_splits
                          * (2 + head_dim) * sizeof(float);
    void* d_scratch = nullptr;
    cudaMalloc(&d_scratch, scratch_size);

    Tensor t_o_splitk(d_o_splitk, DType::FP16, 4, q_shape, true);
    paged_attention_set_splitk_scratch(d_scratch, scratch_size);
    paged_attention_decode_int8(t_q, t_k_i8, t_v_i8, t_o_splitk,
                                static_cast<const half*>(d_k_scales),
                                static_cast<const half*>(d_v_scales),
                                d_bt, d_ctx, block_size, scale,
                                ctx_len, 0, 0.0f, nullptr);
    cudaDeviceSynchronize();

    // ---- Compare Split-K vs non-Split-K ----
    std::vector<half> h_o_nosplit(q_elems);
    std::vector<half> h_o_splitk(q_elems);
    cudaMemcpy(h_o_nosplit.data(), d_o_nosplit, q_elems * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_o_splitk.data(), d_o_splitk, q_elems * sizeof(half), cudaMemcpyDeviceToHost);

    double sum_sq_err = 0.0;
    float max_abs_ref = 0.0f;
    float max_abs_err = 0.0f;
    for (int i = 0; i < q_elems; i++) {
        float ref = __half2float(h_o_nosplit[i]);
        float sk  = __half2float(h_o_splitk[i]);
        float err = ref - sk;
        sum_sq_err += (double)err * err;
        max_abs_ref = fmaxf(max_abs_ref, fabsf(ref));
        max_abs_err = fmaxf(max_abs_err, fabsf(err));
    }
    float rmse = sqrtf((float)(sum_sq_err / q_elems));

    float rel_rmse = (max_abs_ref > 1e-6f) ? (rmse / max_abs_ref) : rmse;
    EXPECT_LT(rel_rmse, 0.02f)
        << "INT8 Split-K vs non-Split-K: relative RMSE = "
        << (rel_rmse * 100.0f) << "% (rmse=" << rmse
        << ", max_abs_err=" << max_abs_err
        << ", max_abs_ref=" << max_abs_ref << ")";

    // ---- Cleanup ----
    paged_attention_set_splitk_scratch(nullptr, 0);
    cudaFree(d_q);
    cudaFree(d_k_int8);
    cudaFree(d_v_int8);
    cudaFree(d_k_scales);
    cudaFree(d_v_scales);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    cudaFree(d_o_nosplit);
    cudaFree(d_o_splitk);
    cudaFree(d_scratch);
}

} // namespace
} // namespace imp
