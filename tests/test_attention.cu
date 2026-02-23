#include <gtest/gtest.h>
#include "compute/attention.h"
#include "compute/attention_paged.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

namespace imp {
namespace {

// ---------------------------------------------------------------------------
// Helpers: GPU tensor allocation, host<->device transfer, FP16 conversion
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while (0)

/// Allocate a contiguous FP16 GPU tensor with the given shape (up to 4D).
static Tensor make_gpu_tensor_fp16(int ndim, const int64_t* shape) {
    Tensor t;
    t.dtype = DType::FP16;
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) t.shape[i] = shape[i];
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

/// Allocate a contiguous INT32 GPU tensor.
static Tensor make_gpu_tensor_int32(int ndim, const int64_t* shape) {
    Tensor t;
    t.dtype = DType::INT32;
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) t.shape[i] = shape[i];
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

/// Free a GPU tensor.
static void free_gpu_tensor(Tensor& t) {
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

/// Upload float host array -> FP16 device tensor (element-by-element conversion).
static void upload_fp32_to_fp16(Tensor& dst, const float* src) {
    int64_t n = dst.numel();
    std::vector<half> h_buf(n);
    for (int64_t i = 0; i < n; i++) {
        h_buf[i] = __float2half(src[i]);
    }
    cudaMemcpy(dst.data, h_buf.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

/// Download FP16 device tensor -> float host array.
static void download_fp16_to_fp32(const Tensor& src, float* dst) {
    int64_t n = src.numel();
    std::vector<half> h_buf(n);
    cudaMemcpy(h_buf.data(), src.data, n * sizeof(half), cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < n; i++) {
        dst[i] = __half2float(h_buf[i]);
    }
}

/// Upload int32 host array -> INT32 device memory.
static void upload_int32(void* dst, const int32_t* src, int64_t count) {
    cudaMemcpy(dst, src, count * sizeof(int32_t), cudaMemcpyHostToDevice);
}

/// Fill a float array with deterministic pseudo-random values in [-0.5, 0.5].
static void fill_random(float* data, int64_t n, unsigned seed = 42) {
    std::srand(seed);
    for (int64_t i = 0; i < n; i++) {
        data[i] = (static_cast<float>(std::rand()) / RAND_MAX) - 0.5f;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: naive multi-head attention (supports GQA)
// ---------------------------------------------------------------------------
//
// Layout:
//   Q: [batch, seq_q,  n_heads,    head_dim]  row-major
//   K: [batch, seq_kv, n_kv_heads, head_dim]  row-major
//   V: [batch, seq_kv, n_kv_heads, head_dim]  row-major
//   O: [batch, seq_q,  n_heads,    head_dim]  row-major
//
// For GQA, each group of (n_heads / n_kv_heads) query heads shares one KV head.
//
static void cpu_attention(const float* Q, const float* K, const float* V,
                          float* O, int batch, int seq_q, int seq_kv,
                          int n_heads, int n_kv_heads, int head_dim,
                          float scale, bool causal) {
    int heads_per_kv = n_heads / n_kv_heads;
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / heads_per_kv;
            for (int sq = 0; sq < seq_q; sq++) {
                // Q vector for this position
                const float* q = Q + ((b * seq_q + sq) * n_heads + h) * head_dim;

                // Compute scores: dot(Q, K) * scale, optionally with causal mask
                std::vector<float> scores(seq_kv);
                for (int sk = 0; sk < seq_kv; sk++) {
                    const float* k = K + ((b * seq_kv + sk) * n_kv_heads + kv_h) * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) dot += q[d] * k[d];
                    scores[sk] = dot * scale;
                    if (causal && sk > sq) scores[sk] = -1e9f;
                }

                // Softmax
                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (auto& s : scores) {
                    s = expf(s - max_s);
                    sum_exp += s;
                }
                for (auto& s : scores) s /= sum_exp;

                // Weighted sum of V
                float* o = O + ((b * seq_q + sq) * n_heads + h) * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    o[d] = 0.0f;
                    for (int sk = 0; sk < seq_kv; sk++) {
                        const float* v = V + ((b * seq_kv + sk) * n_kv_heads + kv_h) * head_dim;
                        o[d] += scores[sk] * v[d];
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tolerance constant for FP16 comparisons.
// Flash attention accumulates many multiply-adds in FP16/FP32 mixed precision;
// 5e-2 relative tolerance is appropriate for small test sizes.
// ---------------------------------------------------------------------------
static constexpr float kFP16Tol = 5e-2f;

// ===================================================================
// Test 1: FlashAttentionSmall
//   batch=1, seq=4, heads=2, head_dim=8, non-causal
// ===================================================================
TEST(AttentionTest, FlashAttentionSmall) {
    const int batch    = 1;
    const int seq      = 4;
    const int n_heads  = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const float scale  = 1.0f / sqrtf(static_cast<float>(head_dim));

    const int64_t total_qkv = batch * seq * n_heads * head_dim;

    // Host buffers (FP32)
    std::vector<float> h_Q(total_qkv), h_K(total_qkv), h_V(total_qkv);
    fill_random(h_Q.data(), total_qkv, 1);
    fill_random(h_K.data(), total_qkv, 2);
    fill_random(h_V.data(), total_qkv, 3);

    // CPU reference
    std::vector<float> h_O_ref(total_qkv, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
                  batch, seq, seq, n_heads, n_kv_heads, head_dim, scale,
                  /*causal=*/false);

    // GPU tensors
    int64_t shape[] = {batch, seq, n_heads, head_dim};
    Tensor d_Q  = make_gpu_tensor_fp16(4, shape);
    Tensor d_K  = make_gpu_tensor_fp16(4, shape);
    Tensor d_V  = make_gpu_tensor_fp16(4, shape);
    Tensor d_O  = make_gpu_tensor_fp16(4, shape);

    upload_fp32_to_fp16(d_Q, h_Q.data());
    upload_fp32_to_fp16(d_K, h_K.data());
    upload_fp32_to_fp16(d_V, h_V.data());

    // Run flash attention prefill (non-causal)
    flash_attention_prefill(d_Q, d_K, d_V, d_O, scale, /*causal=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back
    std::vector<float> h_O_gpu(total_qkv);
    download_fp16_to_fp32(d_O, h_O_gpu.data());

    // Compare
    for (int64_t i = 0; i < total_qkv; i++) {
        EXPECT_NEAR(h_O_gpu[i], h_O_ref[i], kFP16Tol)
            << "Mismatch at element " << i;
    }

    free_gpu_tensor(d_Q);
    free_gpu_tensor(d_K);
    free_gpu_tensor(d_V);
    free_gpu_tensor(d_O);
}

// ===================================================================
// Test 2: FlashAttentionCausal
//   batch=1, seq=4, heads=2, head_dim=8, causal=true
//   CPU reference applies causal mask (score = -inf where q_pos < k_pos).
// ===================================================================
TEST(AttentionTest, FlashAttentionCausal) {
    const int batch    = 1;
    const int seq      = 4;
    const int n_heads  = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;
    const float scale  = 1.0f / sqrtf(static_cast<float>(head_dim));

    const int64_t total = batch * seq * n_heads * head_dim;

    std::vector<float> h_Q(total), h_K(total), h_V(total);
    fill_random(h_Q.data(), total, 10);
    fill_random(h_K.data(), total, 20);
    fill_random(h_V.data(), total, 30);

    // CPU reference with causal mask
    std::vector<float> h_O_ref(total, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
                  batch, seq, seq, n_heads, n_kv_heads, head_dim, scale,
                  /*causal=*/true);

    // GPU
    int64_t shape[] = {batch, seq, n_heads, head_dim};
    Tensor d_Q = make_gpu_tensor_fp16(4, shape);
    Tensor d_K = make_gpu_tensor_fp16(4, shape);
    Tensor d_V = make_gpu_tensor_fp16(4, shape);
    Tensor d_O = make_gpu_tensor_fp16(4, shape);

    upload_fp32_to_fp16(d_Q, h_Q.data());
    upload_fp32_to_fp16(d_K, h_K.data());
    upload_fp32_to_fp16(d_V, h_V.data());

    flash_attention_prefill(d_Q, d_K, d_V, d_O, scale, /*causal=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_O_gpu(total);
    download_fp16_to_fp32(d_O, h_O_gpu.data());

    for (int64_t i = 0; i < total; i++) {
        EXPECT_NEAR(h_O_gpu[i], h_O_ref[i], kFP16Tol)
            << "Mismatch at element " << i;
    }

    // Additional structural check: for the first query position (sq=0), the
    // output should be exactly V[0] for each head (only one valid KV token).
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            int idx_o = (0 * n_heads + h) * head_dim + d;  // O[0,0,h,d]
            int idx_v = (0 * n_kv_heads + h) * head_dim + d;  // V[0,0,h,d]
            EXPECT_NEAR(h_O_gpu[idx_o], h_V[idx_v], kFP16Tol)
                << "Causal first position: O[0,h=" << h << ",d=" << d
                << "] should equal V[0,h=" << h << ",d=" << d << "]";
        }
    }

    free_gpu_tensor(d_Q);
    free_gpu_tensor(d_K);
    free_gpu_tensor(d_V);
    free_gpu_tensor(d_O);
}

// ===================================================================
// Test 3: FlashAttentionGQA
//   batch=1, seq=4, n_heads=4, n_kv_heads=2, head_dim=8
//   Tests grouped query attention: 2 Q heads share 1 KV head.
// ===================================================================
TEST(AttentionTest, FlashAttentionGQA) {
    const int batch      = 1;
    const int seq        = 4;
    const int n_heads    = 4;
    const int n_kv_heads = 2;
    const int head_dim   = 8;
    const float scale    = 1.0f / sqrtf(static_cast<float>(head_dim));

    const int64_t total_q = batch * seq * n_heads * head_dim;
    const int64_t total_kv = batch * seq * n_kv_heads * head_dim;

    std::vector<float> h_Q(total_q), h_K(total_kv), h_V(total_kv);
    fill_random(h_Q.data(), total_q, 100);
    fill_random(h_K.data(), total_kv, 200);
    fill_random(h_V.data(), total_kv, 300);

    // CPU reference (GQA-aware)
    std::vector<float> h_O_ref(total_q, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(),
                  batch, seq, seq, n_heads, n_kv_heads, head_dim, scale,
                  /*causal=*/false);

    // GPU tensors -- Q and O have n_heads, K and V have n_kv_heads
    int64_t q_shape[]  = {batch, seq, n_heads,    head_dim};
    int64_t kv_shape[] = {batch, seq, n_kv_heads, head_dim};

    Tensor d_Q = make_gpu_tensor_fp16(4, q_shape);
    Tensor d_K = make_gpu_tensor_fp16(4, kv_shape);
    Tensor d_V = make_gpu_tensor_fp16(4, kv_shape);
    Tensor d_O = make_gpu_tensor_fp16(4, q_shape);

    upload_fp32_to_fp16(d_Q, h_Q.data());
    upload_fp32_to_fp16(d_K, h_K.data());
    upload_fp32_to_fp16(d_V, h_V.data());

    flash_attention_prefill(d_Q, d_K, d_V, d_O, scale, /*causal=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_O_gpu(total_q);
    download_fp16_to_fp32(d_O, h_O_gpu.data());

    for (int64_t i = 0; i < total_q; i++) {
        EXPECT_NEAR(h_O_gpu[i], h_O_ref[i], kFP16Tol)
            << "GQA mismatch at element " << i;
    }

    // Structural check: heads 0 and 1 share kv_head 0, heads 2 and 3 share
    // kv_head 1.  If Q[h=0] == Q[h=1], the outputs must be identical.
    // We set Q[h=0] = Q[h=1] explicitly and re-run to verify.
    // Overwrite h_Q so that head 1 = head 0 for every (batch, seq) position.
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            const float* src = h_Q.data() + ((b * seq + s) * n_heads + 0) * head_dim;
            float* dst       = h_Q.data() + ((b * seq + s) * n_heads + 1) * head_dim;
            std::memcpy(dst, src, head_dim * sizeof(float));
        }
    }
    upload_fp32_to_fp16(d_Q, h_Q.data());
    flash_attention_prefill(d_Q, d_K, d_V, d_O, scale, /*causal=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());
    download_fp16_to_fp32(d_O, h_O_gpu.data());

    // O[h=0] and O[h=1] must now match (same Q, same KV head).
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int d = 0; d < head_dim; d++) {
                int idx0 = ((b * seq + s) * n_heads + 0) * head_dim + d;
                int idx1 = ((b * seq + s) * n_heads + 1) * head_dim + d;
                EXPECT_NEAR(h_O_gpu[idx0], h_O_gpu[idx1], kFP16Tol)
                    << "GQA structural: O[h=0] != O[h=1] at b=" << b
                    << " s=" << s << " d=" << d;
            }
        }
    }

    free_gpu_tensor(d_Q);
    free_gpu_tensor(d_K);
    free_gpu_tensor(d_V);
    free_gpu_tensor(d_O);
}

// ===================================================================
// Test 4: FlashAttentionScaling
//   Verify that different scale values produce different outputs.
// ===================================================================
TEST(AttentionTest, FlashAttentionScaling) {
    const int batch    = 1;
    const int seq      = 4;
    const int n_heads  = 2;
    const int head_dim = 8;
    const int64_t total = batch * seq * n_heads * head_dim;

    std::vector<float> h_Q(total), h_K(total), h_V(total);
    fill_random(h_Q.data(), total, 55);
    fill_random(h_K.data(), total, 66);
    fill_random(h_V.data(), total, 77);

    int64_t shape[] = {batch, seq, n_heads, head_dim};
    Tensor d_Q = make_gpu_tensor_fp16(4, shape);
    Tensor d_K = make_gpu_tensor_fp16(4, shape);
    Tensor d_V = make_gpu_tensor_fp16(4, shape);
    Tensor d_O = make_gpu_tensor_fp16(4, shape);

    upload_fp32_to_fp16(d_Q, h_Q.data());
    upload_fp32_to_fp16(d_K, h_K.data());
    upload_fp32_to_fp16(d_V, h_V.data());

    // Run with two different scale values
    float scale_a = 0.1f;
    float scale_b = 1.0f;

    flash_attention_prefill(d_Q, d_K, d_V, d_O, scale_a, /*causal=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out_a(total);
    download_fp16_to_fp32(d_O, out_a.data());

    flash_attention_prefill(d_Q, d_K, d_V, d_O, scale_b, /*causal=*/false);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out_b(total);
    download_fp16_to_fp32(d_O, out_b.data());

    // The outputs must differ in at least some positions.
    bool any_different = false;
    for (int64_t i = 0; i < total; i++) {
        if (std::fabs(out_a[i] - out_b[i]) > 1e-4f) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different)
        << "Different scale values should produce different attention outputs";

    // Additionally verify each matches the CPU reference for its own scale.
    std::vector<float> ref_a(total, 0.0f), ref_b(total, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), ref_a.data(),
                  batch, seq, seq, n_heads, n_heads, head_dim, scale_a, false);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), ref_b.data(),
                  batch, seq, seq, n_heads, n_heads, head_dim, scale_b, false);

    for (int64_t i = 0; i < total; i++) {
        EXPECT_NEAR(out_a[i], ref_a[i], kFP16Tol)
            << "Scale-A mismatch at " << i;
        EXPECT_NEAR(out_b[i], ref_b[i], kFP16Tol)
            << "Scale-B mismatch at " << i;
    }

    free_gpu_tensor(d_Q);
    free_gpu_tensor(d_K);
    free_gpu_tensor(d_V);
    free_gpu_tensor(d_O);
}

// ===================================================================
// Test 5: PagedAttentionBasic
//   batch=1, n_heads=2, n_kv_heads=2, head_dim=8,
//   context_len=8, block_size=4 (2 logical blocks).
//   Sets up a simple block table, fills K_cache and V_cache with known
//   values, runs paged decode, and compares to CPU naive single-query
//   attention.
// ===================================================================
TEST(AttentionTest, PagedAttentionBasic) {
    const int batch       = 1;
    const int n_heads     = 2;
    const int n_kv_heads  = 2;
    const int head_dim    = 8;
    const int context_len = 8;
    const int block_size  = 4;

    // Derived constants
    const int num_logical_blocks  = (context_len + block_size - 1) / block_size; // 2
    const int num_physical_blocks = 4;   // allocate a few extra physical blocks
    const int max_context_len     = context_len;
    const int max_num_blocks      = num_logical_blocks;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // ---------- Q: [batch, 1, n_heads, head_dim] ----------
    const int64_t total_q = batch * 1 * n_heads * head_dim;
    std::vector<float> h_Q(total_q);
    fill_random(h_Q.data(), total_q, 500);

    // ---------- Flat KV for the CPU reference ----------
    // We will construct a flat K and V with shape [batch, context_len, n_kv_heads, head_dim]
    // from which we also fill the paged K_cache/V_cache.
    const int64_t total_kv_flat = batch * context_len * n_kv_heads * head_dim;
    std::vector<float> h_K_flat(total_kv_flat), h_V_flat(total_kv_flat);
    fill_random(h_K_flat.data(), total_kv_flat, 600);
    fill_random(h_V_flat.data(), total_kv_flat, 700);

    // ---------- CPU reference (single-query attention) ----------
    // Treat this as: Q=[batch,1,n_heads,hd], K=[batch,ctx_len,n_kv_heads,hd], etc.
    std::vector<float> h_O_ref(total_q, 0.0f);
    cpu_attention(h_Q.data(), h_K_flat.data(), h_V_flat.data(), h_O_ref.data(),
                  batch, /*seq_q=*/1, /*seq_kv=*/context_len,
                  n_heads, n_kv_heads, head_dim, scale, /*causal=*/false);

    // ---------- Build paged K_cache / V_cache on CPU ----------
    // K_cache shape: [num_physical_blocks, n_kv_heads, block_size, head_dim]
    const int64_t cache_block_elems = n_kv_heads * block_size * head_dim;
    const int64_t total_cache = num_physical_blocks * cache_block_elems;
    std::vector<float> h_K_cache(total_cache, 0.0f);
    std::vector<float> h_V_cache(total_cache, 0.0f);

    // Block table: map logical block index -> physical block index.
    // Use a non-trivial mapping: logical 0 -> physical 2, logical 1 -> physical 0.
    std::vector<int32_t> h_block_tables(batch * max_num_blocks);
    h_block_tables[0] = 2;  // logical block 0 lives in physical block 2
    h_block_tables[1] = 0;  // logical block 1 lives in physical block 0

    // Fill K_cache and V_cache according to the block table.
    // For batch b=0, logical block blk, token-within-block t:
    //   global token index = blk * block_size + t
    //   physical block     = h_block_tables[blk]
    //   cache index = phys_block * (n_kv_heads * block_size * head_dim)
    //               + kv_h * (block_size * head_dim)
    //               + t * head_dim + d
    //   flat KV index = ((b * ctx_len + global_tok) * n_kv_heads + kv_h) * head_dim + d
    for (int blk = 0; blk < num_logical_blocks; blk++) {
        int phys = h_block_tables[blk];
        for (int t = 0; t < block_size; t++) {
            int global_tok = blk * block_size + t;
            if (global_tok >= context_len) break;
            for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                for (int d = 0; d < head_dim; d++) {
                    int64_t flat_idx = ((int64_t)(0 * context_len + global_tok) * n_kv_heads + kv_h) * head_dim + d;
                    int64_t cache_idx = ((int64_t)phys * n_kv_heads + kv_h) * block_size * head_dim
                                      + (int64_t)t * head_dim + d;
                    h_K_cache[cache_idx] = h_K_flat[flat_idx];
                    h_V_cache[cache_idx] = h_V_flat[flat_idx];
                }
            }
        }
    }

    // ---------- GPU tensors ----------
    // Q: [batch, 1, n_heads, head_dim]
    int64_t q_shape[] = {batch, 1, n_heads, head_dim};
    Tensor d_Q = make_gpu_tensor_fp16(4, q_shape);
    upload_fp32_to_fp16(d_Q, h_Q.data());

    // O: [batch, 1, n_heads, head_dim]
    Tensor d_O = make_gpu_tensor_fp16(4, q_shape);

    // K_cache, V_cache: [num_physical_blocks, n_kv_heads, block_size, head_dim]
    int64_t cache_shape[] = {num_physical_blocks, n_kv_heads, block_size, head_dim};
    Tensor d_K_cache = make_gpu_tensor_fp16(4, cache_shape);
    Tensor d_V_cache = make_gpu_tensor_fp16(4, cache_shape);
    upload_fp32_to_fp16(d_K_cache, h_K_cache.data());
    upload_fp32_to_fp16(d_V_cache, h_V_cache.data());

    // block_tables: [batch, max_num_blocks] -- INT32 on device
    int32_t* d_block_tables = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_tables, batch * max_num_blocks * sizeof(int32_t)));
    upload_int32(d_block_tables, h_block_tables.data(), batch * max_num_blocks);

    // context_lens: [batch] -- INT32 on device
    int32_t* d_context_lens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_context_lens, batch * sizeof(int32_t)));
    int32_t h_ctx_lens[] = {context_len};
    upload_int32(d_context_lens, h_ctx_lens, batch);

    // ---------- Run paged attention decode ----------
    paged_attention_decode(d_Q, d_K_cache, d_V_cache, d_O,
                           d_block_tables, d_context_lens,
                           block_size, scale, max_context_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- Read back and compare ----------
    std::vector<float> h_O_gpu(total_q);
    download_fp16_to_fp32(d_O, h_O_gpu.data());

    for (int64_t i = 0; i < total_q; i++) {
        EXPECT_NEAR(h_O_gpu[i], h_O_ref[i], kFP16Tol)
            << "PagedAttention mismatch at element " << i
            << " (gpu=" << h_O_gpu[i] << " ref=" << h_O_ref[i] << ")";
    }

    // ---------- Cleanup ----------
    free_gpu_tensor(d_Q);
    free_gpu_tensor(d_O);
    free_gpu_tensor(d_K_cache);
    free_gpu_tensor(d_V_cache);
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);
}

} // namespace
} // namespace imp
