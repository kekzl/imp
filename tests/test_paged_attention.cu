#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <cfloat>
#include <numeric>

namespace imp {
namespace {

// KV cache layout: [num_blocks, block_size, n_kv_heads, head_dim]
// block_size = 16 (kKVBlockSize)
static constexpr int BLOCK_SIZE = 16;

// ---- GPU helpers ----

Tensor make_gpu_tensor_fp16(const float* host_data,
                            std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = DType::FP16;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    std::vector<half> h(t.numel());
    for (int64_t j = 0; j < t.numel(); j++)
        h[j] = __float2half(host_data[j]);
    cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

Tensor alloc_gpu_tensor_fp16(std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.dtype = DType::FP16;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list) t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    cudaMemset(t.data, 0, t.nbytes());
    return t;
}

std::vector<float> read_gpu_fp16(const Tensor& t) {
    std::vector<half> h(t.numel());
    cudaMemcpy(h.data(), t.data, t.nbytes(), cudaMemcpyDeviceToHost);
    std::vector<float> result(t.numel());
    for (int64_t j = 0; j < t.numel(); j++)
        result[j] = __half2float(h[j]);
    return result;
}

void free_gpu(Tensor& t) {
    if (t.data) { cudaFree(t.data); t.data = nullptr; }
}

// ---- CPU reference for single-head attention ----
// Q: [head_dim], K: [seq_len, head_dim], V: [seq_len, head_dim]
// Returns O: [head_dim] using softmax(Q.K^T / scale) @ V
void cpu_attention(const float* Q, const float* K, const float* V,
                   float* O, int seq_len, int head_dim, float scale) {
    // Compute scores
    std::vector<float> scores(seq_len);
    for (int s = 0; s < seq_len; s++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += Q[d] * K[s * head_dim + d];
        scores[s] = dot * scale;
    }

    // Softmax
    float max_s = -FLT_MAX;
    for (int s = 0; s < seq_len; s++)
        max_s = std::max(max_s, scores[s]);
    float sum_exp = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        scores[s] = expf(scores[s] - max_s);
        sum_exp += scores[s];
    }
    for (int s = 0; s < seq_len; s++)
        scores[s] /= sum_exp;

    // Weighted sum of V
    for (int d = 0; d < head_dim; d++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++)
            sum += scores[s] * V[s * head_dim + d];
        O[d] = sum;
    }
}

// ---- Helper: fill KV cache blocks from flat K/V arrays ----
// Writes K/V data into the paged cache layout [num_blocks, block_size, n_kv_heads, head_dim]
void fill_kv_cache(std::vector<float>& kv_cache_flat,
                   const float* kv_data, // [seq_len, head_dim] for one kv_head
                   int kv_head, int n_kv_heads, int head_dim,
                   int seq_len, int num_blocks,
                   const std::vector<int>& block_table) {
    for (int s = 0; s < seq_len; s++) {
        int block_idx = s / BLOCK_SIZE;
        int slot = s % BLOCK_SIZE;
        int phys_block = block_table[block_idx];
        int base = phys_block * BLOCK_SIZE * n_kv_heads * head_dim
                 + slot * n_kv_heads * head_dim
                 + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            kv_cache_flat[base + d] = kv_data[s * head_dim + d];
        }
    }
}

// =========================================================================
// Single head, single sequence, short context
// =========================================================================

TEST(PagedAttentionTest, SingleHeadShortContext) {
    constexpr int batch = 1, n_heads = 1, n_kv_heads = 1, head_dim = 64;
    constexpr int seq_len = 5;
    constexpr int num_blocks = 1;  // 5 tokens fit in 1 block of 16
    constexpr int max_blocks = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Random Q, K, V
    std::vector<float> h_Q(head_dim), h_K(seq_len * head_dim), h_V(seq_len * head_dim);
    for (int i = 0; i < head_dim; i++) h_Q[i] = sinf(static_cast<float>(i) * 0.1f);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.05f);
        h_V[i] = sinf(static_cast<float>(i) * 0.03f + 1.0f);
    }

    // CPU reference
    std::vector<float> h_O(head_dim, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O.data(),
                  seq_len, head_dim, scale);

    // Build KV cache (1 block, identity block table)
    int total_cache_elems = num_blocks * BLOCK_SIZE * n_kv_heads * head_dim;
    std::vector<float> h_K_cache(total_cache_elems, 0.0f);
    std::vector<float> h_V_cache(total_cache_elems, 0.0f);
    std::vector<int> block_table = {0};

    fill_kv_cache(h_K_cache, h_K.data(), 0, n_kv_heads, head_dim, seq_len, num_blocks, block_table);
    fill_kv_cache(h_V_cache, h_V.data(), 0, n_kv_heads, head_dim, seq_len, num_blocks, block_table);

    // Upload to GPU
    // Q: [batch, 1, n_heads, head_dim]
    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    // Block table and context lens on device
    int* d_bt = nullptr; int* d_ctx = nullptr;
    cudaMalloc(&d_bt, batch * max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, batch * sizeof(int));
    cudaMemcpy(d_bt, block_table.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx,
                           BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int d = 0; d < head_dim; d++) {
        EXPECT_NEAR(result[d], h_O[d], 0.05f)
            << "Single head short ctx mismatch at dim " << d;
    }

    free_gpu(d_Q); free_gpu(d_K); free_gpu(d_V); free_gpu(d_O);
    cudaFree(d_bt); cudaFree(d_ctx);
}

// =========================================================================
// Multi-block context (spans 2+ KV cache blocks)
// =========================================================================

TEST(PagedAttentionTest, MultiBlock) {
    constexpr int batch = 1, n_heads = 1, n_kv_heads = 1, head_dim = 64;
    constexpr int seq_len = 20;  // 16 + 4 = 2 blocks
    constexpr int max_blocks = 2;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::vector<float> h_Q(head_dim), h_K(seq_len * head_dim), h_V(seq_len * head_dim);
    for (int i = 0; i < head_dim; i++) h_Q[i] = 0.1f * static_cast<float>(i % 8);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.02f);
        h_V[i] = sinf(static_cast<float>(i) * 0.04f);
    }

    std::vector<float> h_O(head_dim, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O.data(),
                  seq_len, head_dim, scale);

    // Block table: non-sequential to test remapping
    // Physical blocks [1, 0] instead of [0, 1]
    int total_phys = 2;
    int total_cache_elems = total_phys * BLOCK_SIZE * n_kv_heads * head_dim;
    std::vector<float> h_K_cache(total_cache_elems, 0.0f);
    std::vector<float> h_V_cache(total_cache_elems, 0.0f);
    std::vector<int> block_table = {1, 0};  // shuffled

    fill_kv_cache(h_K_cache, h_K.data(), 0, n_kv_heads, head_dim, seq_len, total_phys, block_table);
    fill_kv_cache(h_V_cache, h_V.data(), 0, n_kv_heads, head_dim, seq_len, total_phys, block_table);

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(),
                    {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(),
                    {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr; int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, block_table.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx,
                           BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int d = 0; d < head_dim; d++) {
        EXPECT_NEAR(result[d], h_O[d], 0.05f)
            << "Multi-block mismatch at dim " << d;
    }

    free_gpu(d_Q); free_gpu(d_K); free_gpu(d_V); free_gpu(d_O);
    cudaFree(d_bt); cudaFree(d_ctx);
}

// =========================================================================
// Multi-head attention (MHA, not GQA)
// =========================================================================

TEST(PagedAttentionTest, MultiHead) {
    constexpr int batch = 1, n_heads = 4, n_kv_heads = 4, head_dim = 64;
    constexpr int seq_len = 8;
    constexpr int num_blocks = 1;
    constexpr int max_blocks = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Q: [batch, 1, n_heads, head_dim] = 4 separate query heads
    std::vector<float> h_Q(n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.05f);

    // K, V per head: [seq_len, head_dim] each, packed as [seq_len, n_kv_heads, head_dim]
    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.02f);
        h_V[i] = sinf(static_cast<float>(i) * 0.03f + 0.5f);
    }

    // CPU reference per head
    std::vector<float> h_O(n_heads * head_dim, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        // Extract per-head K/V from [seq_len, n_kv_heads, head_dim]
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + h * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + h * head_dim + d];
            }
        }
        cpu_attention(h_Q.data() + h * head_dim, K_head.data(), V_head.data(),
                      h_O.data() + h * head_dim, seq_len, head_dim, scale);
    }

    // Build KV cache: [num_blocks, BLOCK_SIZE, n_kv_heads, head_dim]
    int total_cache_elems = num_blocks * BLOCK_SIZE * n_kv_heads * head_dim;
    std::vector<float> h_K_cache(total_cache_elems, 0.0f);
    std::vector<float> h_V_cache(total_cache_elems, 0.0f);

    // Fill each head
    for (int h = 0; h < n_kv_heads; h++) {
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + h * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + h * head_dim + d];
            }
        }
        std::vector<int> bt = {0};
        fill_kv_cache(h_K_cache, K_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
        fill_kv_cache(h_V_cache, V_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr; int* d_ctx = nullptr;
    std::vector<int> bt = {0};
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx,
                           BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = h * head_dim + d;
            EXPECT_NEAR(result[idx], h_O[idx], 0.05f)
                << "Multi-head mismatch at head " << h << " dim " << d;
        }
    }

    free_gpu(d_Q); free_gpu(d_K); free_gpu(d_V); free_gpu(d_O);
    cudaFree(d_bt); cudaFree(d_ctx);
}

// =========================================================================
// GQA: n_heads=4, n_kv_heads=2 (ratio 2:1)
// =========================================================================

TEST(PagedAttentionTest, GQA) {
    constexpr int batch = 1, n_heads = 4, n_kv_heads = 2, head_dim = 64;
    constexpr int seq_len = 10;
    constexpr int num_blocks = 1;
    constexpr int max_blocks = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::vector<float> h_Q(n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.07f);

    // KV only has n_kv_heads=2 heads
    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.04f);
        h_V[i] = sinf(static_cast<float>(i) * 0.06f + 0.3f);
    }

    // CPU reference: Q heads [0,1] share KV head 0, Q heads [2,3] share KV head 1
    std::vector<float> h_O(n_heads * head_dim, 0.0f);
    for (int qh = 0; qh < n_heads; qh++) {
        int kvh = qh / (n_heads / n_kv_heads);  // GQA mapping
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + kvh * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + kvh * head_dim + d];
            }
        }
        cpu_attention(h_Q.data() + qh * head_dim, K_head.data(), V_head.data(),
                      h_O.data() + qh * head_dim, seq_len, head_dim, scale);
    }

    // Build KV cache
    int total_cache_elems = num_blocks * BLOCK_SIZE * n_kv_heads * head_dim;
    std::vector<float> h_K_cache(total_cache_elems, 0.0f);
    std::vector<float> h_V_cache(total_cache_elems, 0.0f);

    for (int h = 0; h < n_kv_heads; h++) {
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + h * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + h * head_dim + d];
            }
        }
        std::vector<int> bt = {0};
        fill_kv_cache(h_K_cache, K_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
        fill_kv_cache(h_V_cache, V_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr; int* d_ctx = nullptr;
    std::vector<int> bt = {0};
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx,
                           BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int qh = 0; qh < n_heads; qh++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = qh * head_dim + d;
            EXPECT_NEAR(result[idx], h_O[idx], 0.05f)
                << "GQA mismatch at Q-head " << qh << " dim " << d;
        }
    }

    free_gpu(d_Q); free_gpu(d_K); free_gpu(d_V); free_gpu(d_O);
    cudaFree(d_bt); cudaFree(d_ctx);
}

// =========================================================================
// Batch of 2 sequences with different context lengths
// =========================================================================

TEST(PagedAttentionTest, BatchDifferentLengths) {
    constexpr int batch = 2, n_heads = 2, n_kv_heads = 2, head_dim = 64;
    constexpr int seq0 = 5, seq1 = 12;
    constexpr int max_ctx = 12;
    constexpr int max_blocks = (max_ctx + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 1
    constexpr int total_phys = 2;  // 2 physical blocks
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Q: [batch, 1, n_heads, head_dim]
    std::vector<float> h_Q(batch * n_heads * head_dim);
    for (int i = 0; i < batch * n_heads * head_dim; i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.05f);

    // KV data per sequence per head
    std::vector<float> h_K0(seq0 * head_dim), h_V0(seq0 * head_dim);
    std::vector<float> h_K1(seq1 * head_dim), h_V1(seq1 * head_dim);
    for (int i = 0; i < seq0 * head_dim; i++) {
        h_K0[i] = cosf(static_cast<float>(i) * 0.03f);
        h_V0[i] = sinf(static_cast<float>(i) * 0.04f);
    }
    for (int i = 0; i < seq1 * head_dim; i++) {
        h_K1[i] = cosf(static_cast<float>(i) * 0.05f + 1.0f);
        h_V1[i] = sinf(static_cast<float>(i) * 0.06f + 2.0f);
    }

    // CPU reference: only head 0 for simplicity (head 1 will be same pattern)
    // Seq 0: Q[0,h] attends to K0/V0
    // Seq 1: Q[1,h] attends to K1/V1
    std::vector<float> h_O(batch * n_heads * head_dim, 0.0f);
    for (int b = 0; b < batch; b++) {
        int slen = (b == 0) ? seq0 : seq1;
        const float* K = (b == 0) ? h_K0.data() : h_K1.data();
        const float* V = (b == 0) ? h_V0.data() : h_V1.data();
        for (int h = 0; h < n_heads; h++) {
            // For MHA, kv_head == q_head, so each head has its own KV.
            // But for this test we only have per-sequence KV (same for all heads).
            cpu_attention(h_Q.data() + (b * n_heads + h) * head_dim,
                          K, V,
                          h_O.data() + (b * n_heads + h) * head_dim,
                          slen, head_dim, scale);
        }
    }

    // Build KV cache: head 0 only (MHA: n_kv_heads = n_heads but same KV data for simplicity)
    int total_cache_elems = total_phys * BLOCK_SIZE * n_kv_heads * head_dim;
    std::vector<float> h_K_cache(total_cache_elems, 0.0f);
    std::vector<float> h_V_cache(total_cache_elems, 0.0f);

    // Block table: seq 0 uses phys block 0, seq 1 uses phys block 1
    // [batch, max_blocks]
    std::vector<int> block_table = {0, 1};  // seq0 -> block 0, seq1 -> block 1

    for (int h = 0; h < n_kv_heads; h++) {
        // Seq 0 into block 0
        fill_kv_cache(h_K_cache, h_K0.data(), h, n_kv_heads, head_dim, seq0, total_phys, {0});
        fill_kv_cache(h_V_cache, h_V0.data(), h, n_kv_heads, head_dim, seq0, total_phys, {0});
        // Seq 1 into block 1
        fill_kv_cache(h_K_cache, h_K1.data(), h, n_kv_heads, head_dim, seq1, total_phys, {1});
        fill_kv_cache(h_V_cache, h_V1.data(), h, n_kv_heads, head_dim, seq1, total_phys, {1});
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(),
                    {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(),
                    {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr; int* d_ctx = nullptr;
    cudaMalloc(&d_bt, batch * max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, batch * sizeof(int));
    cudaMemcpy(d_bt, block_table.data(), batch * max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx_lens[2] = {seq0, seq1};
    cudaMemcpy(d_ctx, ctx_lens, batch * sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx,
                           BLOCK_SIZE, scale, max_ctx);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int idx = (b * n_heads + h) * head_dim + d;
                EXPECT_NEAR(result[idx], h_O[idx], 0.05f)
                    << "Batch mismatch at seq " << b << " head " << h << " dim " << d;
            }
        }
    }

    free_gpu(d_Q); free_gpu(d_K); free_gpu(d_V); free_gpu(d_O);
    cudaFree(d_bt); cudaFree(d_ctx);
}

// =========================================================================
// Single token context (edge case: seq_len = 1)
// =========================================================================

TEST(PagedAttentionTest, SingleTokenContext) {
    constexpr int batch = 1, n_heads = 1, n_kv_heads = 1, head_dim = 64;
    constexpr int seq_len = 1;
    constexpr int num_blocks = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::vector<float> h_Q(head_dim), h_K(head_dim), h_V(head_dim);
    for (int i = 0; i < head_dim; i++) {
        h_Q[i] = static_cast<float>(i) * 0.1f;
        h_K[i] = static_cast<float>(head_dim - i) * 0.1f;
        h_V[i] = static_cast<float>(i + 1) * 0.01f;
    }

    // With seq_len=1, softmax is trivially 1.0, so O = V[0]
    std::vector<float> h_O = h_V;

    int total_cache_elems = num_blocks * BLOCK_SIZE * n_kv_heads * head_dim;
    std::vector<float> h_K_cache(total_cache_elems, 0.0f);
    std::vector<float> h_V_cache(total_cache_elems, 0.0f);
    std::vector<int> bt = {0};
    fill_kv_cache(h_K_cache, h_K.data(), 0, n_kv_heads, head_dim, seq_len, num_blocks, bt);
    fill_kv_cache(h_V_cache, h_V.data(), 0, n_kv_heads, head_dim, seq_len, num_blocks, bt);

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(),
                    {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr; int* d_ctx = nullptr;
    cudaMalloc(&d_bt, sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    int bt_val = 0, ctx = 1;
    cudaMemcpy(d_bt, &bt_val, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx,
                           BLOCK_SIZE, scale, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int d = 0; d < head_dim; d++) {
        EXPECT_NEAR(result[d], h_O[d], 0.02f)
            << "Single token mismatch at dim " << d;
    }

    free_gpu(d_Q); free_gpu(d_K); free_gpu(d_V); free_gpu(d_O);
    cudaFree(d_bt); cudaFree(d_ctx);
}

} // namespace
} // namespace imp
