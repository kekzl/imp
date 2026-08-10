#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/attention_paged.h"
#include "core/tensor.h"

#include <vector>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <algorithm>

namespace imp {
namespace {

// KV cache layout: [num_blocks, block_size, n_kv_heads, head_dim]
// block_size = 16 (kKVBlockSize)
static constexpr int BLOCK_SIZE = 16;

// ---- GPU helpers ----

Tensor make_gpu_tensor_fp16(const float* host_data, std::initializer_list<int64_t> shape_list) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
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
    t.qtype = QType::F16;
    t.ndim = static_cast<int>(shape_list.size());
    int i = 0;
    for (auto s : shape_list)
        t.shape[i++] = s;
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
    if (t.data) {
        cudaFree(t.data);
        t.data = nullptr;
    }
}

// ---- CPU reference for single-head attention ----
// Q: [head_dim], K: [seq_len, head_dim], V: [seq_len, head_dim]
// Returns O: [head_dim] using softmax(Q.K^T / scale) @ V
void cpu_attention(const float* Q, const float* K, const float* V, float* O, int seq_len, int head_dim,
                   float scale) {
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
                   const float* kv_data,  // [seq_len, head_dim] for one kv_head
                   int kv_head, int n_kv_heads, int head_dim, int seq_len, int num_blocks,
                   const std::vector<int>& block_table) {
    for (int s = 0; s < seq_len; s++) {
        int block_idx = s / BLOCK_SIZE;
        int slot = s % BLOCK_SIZE;
        int phys_block = block_table[block_idx];
        int base = phys_block * BLOCK_SIZE * n_kv_heads * head_dim + slot * n_kv_heads * head_dim +
                   kv_head * head_dim;
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
    for (int i = 0; i < head_dim; i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.1f);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.05f);
        h_V[i] = sinf(static_cast<float>(i) * 0.03f + 1.0f);
    }

    // CPU reference
    std::vector<float> h_O(head_dim, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O.data(), seq_len, head_dim, scale);

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
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    // Block table and context lens on device
    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, batch * max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, batch * sizeof(int));
    cudaMemcpy(d_bt, block_table.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int d = 0; d < head_dim; d++) {
        EXPECT_NEAR(result[d], h_O[d], 0.05f) << "Single head short ctx mismatch at dim " << d;
    }

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
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
    for (int i = 0; i < head_dim; i++)
        h_Q[i] = 0.1f * static_cast<float>(i % 8);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.02f);
        h_V[i] = sinf(static_cast<float>(i) * 0.04f);
    }

    std::vector<float> h_O(head_dim, 0.0f);
    cpu_attention(h_Q.data(), h_K.data(), h_V.data(), h_O.data(), seq_len, head_dim, scale);

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
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, block_table.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int d = 0; d < head_dim; d++) {
        EXPECT_NEAR(result[d], h_O[d], 0.05f) << "Multi-block mismatch at dim " << d;
    }

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
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
        cpu_attention(h_Q.data() + h * head_dim, K_head.data(), V_head.data(), h_O.data() + h * head_dim,
                      seq_len, head_dim, scale);
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
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    std::vector<int> bt = {0};
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = h * head_dim + d;
            EXPECT_NEAR(result[idx], h_O[idx], 0.05f) << "Multi-head mismatch at head " << h << " dim " << d;
        }
    }

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
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
        cpu_attention(h_Q.data() + qh * head_dim, K_head.data(), V_head.data(), h_O.data() + qh * head_dim,
                      seq_len, head_dim, scale);
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
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    std::vector<int> bt = {0};
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int qh = 0; qh < n_heads; qh++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = qh * head_dim + d;
            EXPECT_NEAR(result[idx], h_O[idx], 0.05f) << "GQA mismatch at Q-head " << qh << " dim " << d;
        }
    }

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

// =========================================================================
// Batch of 2 sequences with different context lengths
// =========================================================================

TEST(PagedAttentionTest, BatchDifferentLengths) {
    constexpr int batch = 2, n_heads = 2, n_kv_heads = 2, head_dim = 64;
    constexpr int seq0 = 5, seq1 = 12;
    constexpr int max_ctx = 12;
    constexpr int max_blocks = (max_ctx + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 1
    constexpr int total_phys = 2;                                        // 2 physical blocks
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
            cpu_attention(h_Q.data() + (b * n_heads + h) * head_dim, K, V,
                          h_O.data() + (b * n_heads + h) * head_dim, slen, head_dim, scale);
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
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {total_phys, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, batch * max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, batch * sizeof(int));
    cudaMemcpy(d_bt, block_table.data(), batch * max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx_lens[2] = {seq0, seq1};
    cudaMemcpy(d_ctx, ctx_lens, batch * sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, max_ctx);
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

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
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
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    int bt_val = 0, ctx = 1;
    cudaMemcpy(d_bt, &bt_val, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, 1);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int d = 0; d < head_dim; d++) {
        EXPECT_NEAR(result[d], h_O[d], 0.02f) << "Single token mismatch at dim " << d;
    }

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

// =========================================================================
// GQA with long context: exercises cluster kernel path (n_q_per_kv=4, 256 tokens)
// =========================================================================

TEST(PagedAttentionTest, GQALongContext) {
    constexpr int batch = 1, n_heads = 8, n_kv_heads = 2, head_dim = 64;
    constexpr int seq_len = 256;
    constexpr int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 16
    constexpr int max_blocks = num_blocks;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::vector<float> h_Q(n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.03f);

    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.02f);
        h_V[i] = sinf(static_cast<float>(i) * 0.04f + 0.5f);
    }

    // CPU reference
    std::vector<float> h_O(n_heads * head_dim, 0.0f);
    for (int qh = 0; qh < n_heads; qh++) {
        int kvh = qh / (n_heads / n_kv_heads);
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + kvh * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + kvh * head_dim + d];
            }
        }
        cpu_attention(h_Q.data() + qh * head_dim, K_head.data(), V_head.data(), h_O.data() + qh * head_dim,
                      seq_len, head_dim, scale);
    }

    // Block table: identity mapping
    std::vector<int> bt(num_blocks);
    for (int i = 0; i < num_blocks; i++)
        bt[i] = i;

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
        fill_kv_cache(h_K_cache, K_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
        fill_kv_cache(h_V_cache, V_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    auto result = read_gpu_fp16(d_O);
    for (int qh = 0; qh < n_heads; qh++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = qh * head_dim + d;
            EXPECT_NEAR(result[idx], h_O[idx], 0.05f)
                << "GQA long-context mismatch at Q-head " << qh << " dim " << d;
        }
    }

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

// =========================================================================
// gpt-oss decode shape (#547): split-K + hd=64 + GQA 8:1 (+ learned sinks).
// The split-K branch only activates when scratch is set — none of the older
// tests set it, so the hd=64 split-K path was never covered (gpt-oss is the
// first hd=64 model; the rest of the zoo is hd>=128).
// =========================================================================

void run_gptoss_shape_splitk_case(bool with_sinks, int seq_len = 256, bool with_scratch = true) {
    constexpr int batch = 1, n_heads = 64, n_kv_heads = 8, head_dim = 64;
    // seq_len 256 = 16 blocks >= 4 → split-K heuristics fire (compute_splitk_splits).
    // seq_len  48 =  3 blocks  < 4 → they do not, which is the ONLY way to reach
    // crosswarp_reduce_and_write's sink term; the split-K reduction handles sinks
    // in a different function entirely.
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int max_blocks = num_blocks;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::vector<float> h_Q(n_heads * head_dim);
    for (size_t i = 0; i < h_Q.size(); i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.013f);
    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (size_t i = 0; i < h_K.size(); i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.007f);
        h_V[i] = sinf(static_cast<float>(i) * 0.011f + 0.5f);
    }
    std::vector<float> h_sinks(n_heads);
    for (int h = 0; h < n_heads; h++) {
        // The sink logit has to compete with the SUM of seq_len score
        // exponentials, so a value near the score range contributes almost
        // nothing. At the original -1.0 + 0.05h the sink carried 0-13% of the
        // mass and deleting it moved the output by at most 0.0017 — under the
        // tolerance this test compared with, which is why mutant M31 (#1303)
        // survived a test written to catch it. At 4.0 + 0.05h it carries
        // 0.5-96% and deleting it moves the output by up to 0.057.
        h_sinks[h] = 4.0f + 0.05f * static_cast<float>(h);
    }

    // CPU reference: softmax over [scores, sink]; sink column dropped.
    std::vector<float> h_O(n_heads * head_dim, 0.0f);
    for (int qh = 0; qh < n_heads; qh++) {
        int kvh = qh / (n_heads / n_kv_heads);
        std::vector<float> scores(seq_len);
        float m = with_sinks ? h_sinks[qh] : -FLT_MAX;
        for (int s = 0; s < seq_len; s++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++)
                dot += h_Q[qh * head_dim + d] * h_K[(s * n_kv_heads + kvh) * head_dim + d];
            scores[s] = dot * scale;
            m = std::max(m, scores[s]);
        }
        float denom = with_sinks ? expf(h_sinks[qh] - m) : 0.0f;
        for (int s = 0; s < seq_len; s++)
            denom += expf(scores[s] - m);
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int s = 0; s < seq_len; s++)
                acc += expf(scores[s] - m) / denom * h_V[(s * n_kv_heads + kvh) * head_dim + d];
            h_O[qh * head_dim + d] = acc;
        }
    }

    std::vector<int> bt(num_blocks);
    std::iota(bt.begin(), bt.end(), 0);
    int total_cache_elems = num_blocks * BLOCK_SIZE * n_kv_heads * head_dim;
    std::vector<float> h_K_cache(total_cache_elems, 0.0f);
    std::vector<float> h_V_cache(total_cache_elems, 0.0f);
    // Direct layout copy: cache[blk][slot][kvh][d] = K[s][kvh][d]
    for (int s = 0; s < seq_len; s++) {
        int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
        for (int h = 0; h < n_kv_heads; h++)
            for (int d = 0; d < head_dim; d++) {
                int dst = ((blk * BLOCK_SIZE + slot) * n_kv_heads + h) * head_dim + d;
                int src = (s * n_kv_heads + h) * head_dim + d;
                h_K_cache[dst] = h_K[src];
                h_V_cache[dst] = h_V[src];
            }
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    // Split-K needs scratch: with it and >=4 blocks the split path takes over,
    // without it num_splits stays 1 and the dispatch falls through to the GQA
    // branch — which is how the cluster kernel becomes reachable (see the test
    // list at the bottom of this block).
    constexpr int kMaxSplits = 64;
    size_t scratch_size = static_cast<size_t>(batch) * n_heads * kMaxSplits * (2 + head_dim) * sizeof(float);
    void* d_scratch = nullptr;
    if (with_scratch) {
        cudaMalloc(&d_scratch, scratch_size);
        paged_attention_set_splitk_scratch(d_scratch, scratch_size);
    } else {
        paged_attention_set_splitk_scratch(nullptr, 0);
    }

    half* d_sinks = nullptr;
    if (with_sinks) {
        std::vector<half> hs(n_heads);
        for (int h = 0; h < n_heads; h++)
            hs[h] = __float2half(h_sinks[h]);
        cudaMalloc(&d_sinks, n_heads * sizeof(half));
        cudaMemcpy(d_sinks, hs.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
    }

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len,
                           /*sliding_window=*/0, /*softcap=*/0.0f, /*stream=*/nullptr,
                           /*max_blocks_per_seq=*/max_blocks, /*n_sinks=*/0, d_sinks);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    auto result = read_gpu_fp16(d_O);
    double max_err = 0.0;
    for (int qh = 0; qh < n_heads; qh++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = qh * head_dim + d;
            max_err = std::max(max_err, static_cast<double>(std::abs(result[idx] - h_O[idx])));
            // 2e-3, not the 0.05 this used to allow: measured max error against
            // the fp32 host reference is 1.7e-4 (seq_len 48) / 7.1e-5 (256), so
            // 0.05 was ~300x looser than the kernel needs and swallowed the
            // whole sink term. 2e-3 keeps a 12x margin over the kernel and still
            // catches a dropped sink by 4.7x (seq_len 256) to 28x (seq_len 48).
            EXPECT_NEAR(result[idx], h_O[idx], 2e-3f)
                << "hd64 decode mismatch at Q-head " << qh << " dim " << d << " (with_sinks=" << with_sinks
                << ", seq_len=" << seq_len << ", splitk_scratch=" << with_scratch << ", max_err=" << max_err
                << ")";
        }
    }

    paged_attention_set_splitk_scratch(nullptr, 0);
    if (d_scratch)
        cudaFree(d_scratch);
    if (d_sinks)
        cudaFree(d_sinks);
    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

TEST(PagedAttentionTest, GQA_SplitK_HD64) { run_gptoss_shape_splitk_case(/*with_sinks=*/false); }
TEST(PagedAttentionTest, GQA_SplitK_HD64_Sinks) { run_gptoss_shape_splitk_case(/*with_sinks=*/true); }

// Same shape, short enough that split-K does NOT fire, so the decode goes
// through crosswarp_reduce_and_write. Without this the sink term in that
// function can be deleted with the whole suite still green (mutant M31, #1303):
// both cases above are sized so the split-K path takes over.
TEST(PagedAttentionTest, GQA_NoSplitK_HD64_Sinks) {
    run_gptoss_shape_splitk_case(/*with_sinks=*/true, /*seq_len=*/48);
}
TEST(PagedAttentionTest, GQA_NoSplitK_HD64) {
    run_gptoss_shape_splitk_case(/*with_sinks=*/false, /*seq_len=*/48);
}

// Third sink implementation, third test. `crosswarp_reduce_and_write` in
// attention_paged_common.cuh is shared by the cluster kernel and every
// quantised-KV decode (int4/int8/nvfp4/nvfp4_tc/fp8) — but only the cluster
// kernel passes it a sink pointer; the others leave the argument at its nullptr
// default. So the sink branch in that helper is reachable through exactly one
// launch configuration: no split-K scratch (num_splits stays 1), n_q_per_kv in
// {2,4,8}, >=8 context blocks and head_dim in {64,96,128,256}.
//
// That is what mutant M31 (#1303) deletes. The two cases above cannot reach it:
// 48 tokens is 3 blocks (< 8, so the plain GQA kernel runs, which handles sinks
// inline) and 256 tokens with scratch goes split-K (its own reduction, also
// inline). Both stayed green with the term removed.
TEST(PagedAttentionTest, GQA_Cluster_HD64_Sinks) {
    run_gptoss_shape_splitk_case(/*with_sinks=*/true, /*seq_len=*/256, /*with_scratch=*/false);
}
TEST(PagedAttentionTest, GQA_Cluster_HD64) {
    run_gptoss_shape_splitk_case(/*with_sinks=*/false, /*seq_len=*/256, /*with_scratch=*/false);
}

// =========================================================================
// Learned attention sinks on the INT8 KV decode path (#1345)
// =========================================================================
//
// gpt-oss is the only architecture shipping learned sinks, and its KV cache is
// the one under the most long-context pressure — so "use fp16 KV for this
// model" was an expensive guard. INT8 has two decode implementations and the
// sink term enters each in a different place: the split-K path applies it in
// the shared reduce kernel, the non-split-K fallback in
// `crosswarp_reduce_and_write`. Both are covered below; either one left unwired
// serves a softmax denominator short one column.
//
// The reference is built from the DEQUANTIZED K/V, not the original FP16. That
// is deliberate and is the opposite of what tests/test_attention_paged_oracle.cu
// wants: the property under test here is the sink term, not the quantizer, and
// referencing the quant grid removes INT8 noise from the comparison so a 2e-3
// tolerance can see a sink term that INT8 error would otherwise swallow. The
// quantizer itself is held to account in the oracle test.
//
// Tolerance alone is not trusted: each case also asserts the result is far
// CLOSER to the sink-aware reference than to the sink-free one. A dropped sink
// then fails on the ratio even if someone later loosens the tolerance.
enum class QuantKV { INT8, INT4 };

void run_quant_sink_case(QuantKV kind, bool with_sinks, int seq_len, bool with_scratch) {
    const bool is_int4 = (kind == QuantKV::INT4);
    const int qmax = is_int4 ? 7 : 127;
    constexpr int batch = 1, n_heads = 8, n_kv_heads = 2, head_dim = 64;  // gpt-oss GQA shape
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int max_blocks = num_blocks;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::vector<float> h_Q(n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++)
        h_Q[i] = 0.6f * sinf(static_cast<float>(i) * 0.021f);
    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (size_t i = 0; i < h_K.size(); i++) {
        h_K[i] = 0.5f * cosf(static_cast<float>(i) * 0.013f);
        // V carries a DC component on purpose. With a zero-mean V the softmax
        // average over 256 tokens cancels to ~0.002, and the sink term — which
        // scales the whole output by the mass it takes — then moves it by 4e-4,
        // below any tolerance a quantised path can hold. The vacuity guard
        // below catches that; the offset is what makes the case non-vacuous.
        h_V[i] = 0.5f + 0.4f * sinf(static_cast<float>(i) * 0.017f + 0.5f);
    }

    // Sink magnitude matches the FP16 sink tests: at 4.0 + 0.05h the sink term
    // carries real mass, so deleting it moves the output far outside 2e-3.
    std::vector<float> h_sinks(n_heads);
    for (int h = 0; h < n_heads; h++)
        h_sinks[h] = 4.0f + 0.05f * static_cast<float>(h);

    // Host quantize into the INT8 cache layout the kernel reads:
    //   data   [num_blocks, BLOCK_SIZE, n_kv_heads, head_dim] int8
    //   scales [num_blocks, BLOCK_SIZE, n_kv_heads]           half
    // Mirrors write_kv_cache_int8_kernel (per-head absmax / 127, round-to-nearest).
    const size_t cache_elems = (size_t)num_blocks * BLOCK_SIZE * n_kv_heads *
                               (is_int4 ? head_dim / 2 : head_dim);
    const size_t row_elems = is_int4 ? head_dim / 2 : head_dim;
    const size_t scale_elems = (size_t)num_blocks * BLOCK_SIZE * n_kv_heads;
    std::vector<int8_t> k_q(cache_elems, 0), v_q(cache_elems, 0);
    std::vector<half> k_sc(scale_elems, __float2half(0.0f)), v_sc(scale_elems, __float2half(0.0f));
    // Dequantized copies — the reference is computed from these.
    std::vector<float> k_deq(h_K.size(), 0.0f), v_deq(h_V.size(), 0.0f);

    auto quantize = [&](const std::vector<float>& src, std::vector<int8_t>& dst, std::vector<half>& sc,
                        std::vector<float>& deq) {
        for (int s = 0; s < seq_len; s++) {
            int blk = s / BLOCK_SIZE, slot = s % BLOCK_SIZE;
            for (int kvh = 0; kvh < n_kv_heads; kvh++) {
                size_t src_base = ((size_t)s * n_kv_heads + kvh) * head_dim;
                float amax = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    amax = std::max(amax, std::fabs(src[src_base + d]));
                float step = (amax > 1e-8f) ? amax / (float)qmax : 0.0f;
                float inv = (amax > 1e-8f) ? (float)qmax / amax : 0.0f;
                size_t dst_base = ((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads * row_elems +
                                  (size_t)kvh * row_elems;
                sc[((size_t)blk * BLOCK_SIZE + slot) * n_kv_heads + kvh] = __float2half(step);
                float step_h = __half2float(__float2half(step));  // the scale the kernel reads back
                for (int d = 0; d < head_dim; d++) {
                    int q = static_cast<int>(std::lround(src[src_base + d] * inv));
                    q = std::max(-qmax, std::min(qmax, q));
                    if (is_int4) {
                        // low nibble = even d, high nibble = odd d (write_kv_cache_int4_kernel)
                        uint8_t& byte = reinterpret_cast<uint8_t&>(dst[dst_base + d / 2]);
                        if (d % 2 == 0)
                            byte = static_cast<uint8_t>((byte & 0xF0) | (q & 0x0F));
                        else
                            byte = static_cast<uint8_t>((byte & 0x0F) | ((q & 0x0F) << 4));
                    } else {
                        dst[dst_base + d] = static_cast<int8_t>(q);
                    }
                    deq[src_base + d] = static_cast<float>(q) * step_h;
                }
            }
        }
    };
    quantize(h_K, k_q, k_sc, k_deq);
    quantize(h_V, v_q, v_sc, v_deq);

    // Two fp32 references over the dequantized grid: with and without the sink
    // column in the denominator.
    auto reference = [&](bool sinks_on) {
        std::vector<float> O(n_heads * head_dim, 0.0f);
        for (int qh = 0; qh < n_heads; qh++) {
            int kvh = qh / (n_heads / n_kv_heads);
            std::vector<float> scores(seq_len);
            float m = sinks_on ? h_sinks[qh] : -FLT_MAX;
            for (int s = 0; s < seq_len; s++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += h_Q[qh * head_dim + d] * k_deq[((size_t)s * n_kv_heads + kvh) * head_dim + d];
                scores[s] = dot * scale;
                m = std::max(m, scores[s]);
            }
            float denom = sinks_on ? expf(h_sinks[qh] - m) : 0.0f;
            for (int s = 0; s < seq_len; s++)
                denom += expf(scores[s] - m);
            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int s = 0; s < seq_len; s++)
                    acc += expf(scores[s] - m) / denom * v_deq[((size_t)s * n_kv_heads + kvh) * head_dim + d];
                O[qh * head_dim + d] = acc;
            }
        }
        return O;
    };
    const std::vector<float> ref_sinks = reference(true);
    const std::vector<float> ref_nosinks = reference(false);
    const std::vector<float>& ref_want = with_sinks ? ref_sinks : ref_nosinks;
    const std::vector<float>& ref_other = with_sinks ? ref_nosinks : ref_sinks;

    std::vector<int> bt(max_blocks);
    std::iota(bt.begin(), bt.end(), 0);

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K{}, d_V{};
    for (Tensor* t : {&d_K, &d_V}) {
        t->qtype = QType::INT8;
        t->ndim = 4;
        t->shape[0] = num_blocks;
        t->shape[1] = BLOCK_SIZE;
        t->shape[2] = n_kv_heads;
        t->shape[3] = (int64_t)row_elems;
        t->compute_strides();
        t->on_device = true;
        cudaMalloc(&t->data, cache_elems);
    }
    cudaMemcpy(d_K.data, k_q.data(), cache_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V.data, v_q.data(), cache_elems, cudaMemcpyHostToDevice);
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    half *d_ks = nullptr, *d_vs = nullptr;
    cudaMalloc(&d_ks, scale_elems * sizeof(half));
    cudaMalloc(&d_vs, scale_elems * sizeof(half));
    cudaMemcpy(d_ks, k_sc.data(), scale_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, v_sc.data(), scale_elems * sizeof(half), cudaMemcpyHostToDevice);

    int *d_bt = nullptr, *d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    // Scratch present → split-K takes over (sink applied in the reduce kernel);
    // absent → num_splits stays 1 and the fallback kernel runs (sink applied in
    // crosswarp_reduce_and_write).
    constexpr int kMaxSplits = 64;
    size_t scratch_size = static_cast<size_t>(batch) * n_heads * kMaxSplits * (2 + head_dim) * sizeof(float);
    void* d_scratch = nullptr;
    if (with_scratch) {
        cudaMalloc(&d_scratch, scratch_size);
        paged_attention_set_splitk_scratch(d_scratch, scratch_size);
    } else {
        paged_attention_set_splitk_scratch(nullptr, 0);
    }

    half* d_sinks = nullptr;
    if (with_sinks) {
        std::vector<half> hs(n_heads);
        for (int h = 0; h < n_heads; h++)
            hs[h] = __float2half(h_sinks[h]);
        cudaMalloc(&d_sinks, n_heads * sizeof(half));
        cudaMemcpy(d_sinks, hs.data(), n_heads * sizeof(half), cudaMemcpyHostToDevice);
    }

    if (is_int4) {
        paged_attention_decode_int4(d_Q, d_K, d_V, d_O, d_ks, d_vs, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, /*stream=*/nullptr,
                                    /*max_blocks_per_seq=*/max_blocks, /*n_sinks=*/0, d_sinks);
    } else {
        paged_attention_decode_int8(d_Q, d_K, d_V, d_O, d_ks, d_vs, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len,
                                    /*sliding_window=*/0, /*softcap=*/0.0f, /*stream=*/nullptr,
                                    /*max_blocks_per_seq=*/max_blocks, /*n_sinks=*/0, d_sinks);
    }
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    auto result = read_gpu_fp16(d_O);
    double err_want = 0.0, err_other = 0.0;
    for (int i = 0; i < n_heads * head_dim; i++) {
        err_want = std::max(err_want, static_cast<double>(std::abs(result[i] - ref_want[i])));
        err_other = std::max(err_other, static_cast<double>(std::abs(result[i] - ref_other[i])));
        EXPECT_NEAR(result[i], ref_want[i], is_int4 ? 6e-3f : 2e-3f)
            << (is_int4 ? "INT4" : "INT8") << " sink decode mismatch at " << i
            << " (with_sinks=" << with_sinks << ", seq_len=" << seq_len << ", splitk_scratch=" << with_scratch
            << ")";
    }
    // Discriminator: the two references must be far apart, and the kernel must
    // land on the right one. Without this a loosened tolerance would let a
    // dropped sink pass silently — how M31 survived on the FP16 side (#1303).
    double ref_gap = 0.0;
    for (int i = 0; i < n_heads * head_dim; i++)
        ref_gap = std::max(ref_gap, static_cast<double>(std::abs(ref_sinks[i] - ref_nosinks[i])));
    EXPECT_GT(ref_gap, 1e-2) << "sink term too weak to discriminate — test would be vacuous";
    EXPECT_LT(err_want * 4.0, err_other)
        << (is_int4 ? "INT4" : "INT8") << " decode is not closer to its own reference than to the other one "
        << "(err_want=" << err_want << ", err_other=" << err_other << ", with_sinks=" << with_sinks << ")";

    paged_attention_set_splitk_scratch(nullptr, 0);
    if (d_scratch)
        cudaFree(d_scratch);
    if (d_sinks)
        cudaFree(d_sinks);
    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_ks);
    cudaFree(d_vs);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

TEST(PagedAttentionTest, INT8_SplitK_HD64_Sinks) {
    run_quant_sink_case(QuantKV::INT8, /*with_sinks=*/true, /*seq_len=*/256, /*with_scratch=*/true);
}
TEST(PagedAttentionTest, INT8_SplitK_HD64) {
    run_quant_sink_case(QuantKV::INT8, /*with_sinks=*/false, /*seq_len=*/256, /*with_scratch=*/true);
}
TEST(PagedAttentionTest, INT8_NoSplitK_HD64_Sinks) {
    run_quant_sink_case(QuantKV::INT8, /*with_sinks=*/true, /*seq_len=*/256, /*with_scratch=*/false);
}
TEST(PagedAttentionTest, INT8_NoSplitK_HD64) {
    run_quant_sink_case(QuantKV::INT8, /*with_sinks=*/false, /*seq_len=*/256, /*with_scratch=*/false);
}

// INT4 runs the same oracle. gpt-oss answers EMPTY on INT4 KV end to end, which
// is exactly the signature of a dropped sink — these cases are what separate
// "the sink term is missing" from "4 bits per value on a 64-wide head is too
// coarse for this model". They pass, so it is the latter.
TEST(PagedAttentionTest, INT4_SplitK_HD64_Sinks) {
    run_quant_sink_case(QuantKV::INT4, /*with_sinks=*/true, /*seq_len=*/256, /*with_scratch=*/true);
}
TEST(PagedAttentionTest, INT4_SplitK_HD64) {
    run_quant_sink_case(QuantKV::INT4, /*with_sinks=*/false, /*seq_len=*/256, /*with_scratch=*/true);
}
TEST(PagedAttentionTest, INT4_NoSplitK_HD64_Sinks) {
    run_quant_sink_case(QuantKV::INT4, /*with_sinks=*/true, /*seq_len=*/256, /*with_scratch=*/false);
}
TEST(PagedAttentionTest, INT4_NoSplitK_HD64) {
    run_quant_sink_case(QuantKV::INT4, /*with_sinks=*/false, /*seq_len=*/256, /*with_scratch=*/false);
}

// =========================================================================
// GQA with HD=256: exercises split-K and cluster paths for Gemma-3 config
// =========================================================================

TEST(PagedAttentionTest, GQA_HD256) {
    constexpr int batch = 1, n_heads = 16, n_kv_heads = 8, head_dim = 256;
    constexpr int seq_len = 128;
    constexpr int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int max_blocks = num_blocks;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    std::vector<float> h_Q(n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.01f);

    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.005f);
        h_V[i] = sinf(static_cast<float>(i) * 0.007f + 0.3f);
    }

    // CPU reference
    std::vector<float> h_O(n_heads * head_dim, 0.0f);
    for (int qh = 0; qh < n_heads; qh++) {
        int kvh = qh / (n_heads / n_kv_heads);
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + kvh * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + kvh * head_dim + d];
            }
        }
        cpu_attention(h_Q.data() + qh * head_dim, K_head.data(), V_head.data(), h_O.data() + qh * head_dim,
                      seq_len, head_dim, scale);
    }

    // Block table: identity mapping
    std::vector<int> bt(num_blocks);
    for (int i = 0; i < num_blocks; i++)
        bt[i] = i;

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
        fill_kv_cache(h_K_cache, K_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
        fill_kv_cache(h_V_cache, V_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

    auto result = read_gpu_fp16(d_O);
    int mismatches = 0;
    for (int qh = 0; qh < n_heads; qh++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = qh * head_dim + d;
            if (std::abs(result[idx] - h_O[idx]) > 0.05f)
                mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << "HD=256 GQA: " << mismatches << " mismatches out of " << n_heads * head_dim;

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

// =========================================================================
// Gemma-4 Global layer geometry: GQA with hd=512, nkv=2, nh=16, short ctx
// (reproduces decode config at the 7th token of "The capital of France is").
// =========================================================================
TEST(PagedAttentionTest, GQA_HD512_Gemma4Global) {
    constexpr int batch = 1, n_heads = 16, n_kv_heads = 2, head_dim = 512;
    constexpr int seq_len = 7;  // context after prefill(6) + decode token
    constexpr int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int max_blocks = num_blocks;
    const float scale = 1.0f;  // Gemma-4 uses f_attention_scale=1.0

    std::vector<float> h_Q(n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++)
        h_Q[i] = 0.1f * sinf(static_cast<float>(i) * 0.013f);

    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; i++) {
        h_K[i] = 0.1f * cosf(static_cast<float>(i) * 0.005f);
        h_V[i] = 0.1f * sinf(static_cast<float>(i) * 0.007f + 0.3f);
    }

    // CPU reference
    std::vector<float> h_O(n_heads * head_dim, 0.0f);
    for (int qh = 0; qh < n_heads; qh++) {
        int kvh = qh / (n_heads / n_kv_heads);
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + kvh * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + kvh * head_dim + d];
            }
        }
        cpu_attention(h_Q.data() + qh * head_dim, K_head.data(), V_head.data(), h_O.data() + qh * head_dim,
                      seq_len, head_dim, scale);
    }

    std::vector<int> bt(num_blocks);
    for (int i = 0; i < num_blocks; i++)
        bt[i] = i;

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
        fill_kv_cache(h_K_cache, K_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
        fill_kv_cache(h_V_cache, V_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);

    auto result = read_gpu_fp16(d_O);
    int mismatches = 0;
    float max_err = 0.0f;
    for (int qh = 0; qh < n_heads; qh++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = qh * head_dim + d;
            float err_val = std::abs(result[idx] - h_O[idx]);
            max_err = std::max(max_err, err_val);
            if (err_val > 0.01f)
                mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << "HD=512 GQA Gemma-4 Global: " << mismatches << " mismatches out of "
                             << n_heads * head_dim << " (max_err=" << max_err << ")";

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

// =========================================================================
// Gemma-4 SWA layer geometry: GQA with hd=256, nkv=8, nh=16, short ctx
// =========================================================================
TEST(PagedAttentionTest, GQA_HD256_Gemma4SWA) {
    constexpr int batch = 1, n_heads = 16, n_kv_heads = 8, head_dim = 256;
    constexpr int seq_len = 7;
    constexpr int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int max_blocks = num_blocks;
    const float scale = 1.0f;

    std::vector<float> h_Q(n_heads * head_dim);
    for (int i = 0; i < n_heads * head_dim; i++)
        h_Q[i] = 0.1f * sinf(static_cast<float>(i) * 0.013f);

    std::vector<float> h_K(seq_len * n_kv_heads * head_dim);
    std::vector<float> h_V(seq_len * n_kv_heads * head_dim);
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; i++) {
        h_K[i] = 0.1f * cosf(static_cast<float>(i) * 0.005f);
        h_V[i] = 0.1f * sinf(static_cast<float>(i) * 0.007f + 0.3f);
    }

    std::vector<float> h_O(n_heads * head_dim, 0.0f);
    for (int qh = 0; qh < n_heads; qh++) {
        int kvh = qh / (n_heads / n_kv_heads);
        std::vector<float> K_head(seq_len * head_dim), V_head(seq_len * head_dim);
        for (int s = 0; s < seq_len; s++) {
            for (int d = 0; d < head_dim; d++) {
                K_head[s * head_dim + d] = h_K[s * n_kv_heads * head_dim + kvh * head_dim + d];
                V_head[s * head_dim + d] = h_V[s * n_kv_heads * head_dim + kvh * head_dim + d];
            }
        }
        cpu_attention(h_Q.data() + qh * head_dim, K_head.data(), V_head.data(), h_O.data() + qh * head_dim,
                      seq_len, head_dim, scale);
    }

    std::vector<int> bt(num_blocks);
    for (int i = 0; i < num_blocks; i++)
        bt[i] = i;

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
        fill_kv_cache(h_K_cache, K_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
        fill_kv_cache(h_V_cache, V_head.data(), h, n_kv_heads, head_dim, seq_len, num_blocks, bt);
    }

    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});
    Tensor d_K = make_gpu_tensor_fp16(h_K_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_V = make_gpu_tensor_fp16(h_V_cache.data(), {num_blocks, BLOCK_SIZE, n_kv_heads, head_dim});
    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    cudaMemcpy(d_bt, bt.data(), max_blocks * sizeof(int), cudaMemcpyHostToDevice);
    int ctx = seq_len;
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode(d_Q, d_K, d_V, d_O, d_bt, d_ctx, BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    auto result = read_gpu_fp16(d_O);
    int mismatches = 0;
    float max_err = 0.0f;
    for (int qh = 0; qh < n_heads; qh++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = qh * head_dim + d;
            float err_val = std::abs(result[idx] - h_O[idx]);
            max_err = std::max(max_err, err_val);
            if (err_val > 0.01f)
                mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << "HD=256 GQA Gemma-4 SWA: " << mismatches << " mismatches / "
                             << n_heads * head_dim << " (max_err=" << max_err << ")";

    free_gpu(d_Q);
    free_gpu(d_K);
    free_gpu(d_V);
    free_gpu(d_O);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

// =========================================================================
// INT4 pack/unpack roundtrip
// =========================================================================

TEST(INT4QuantTest, PackUnpackRoundtrip) {
    // INT4 signed range: [-8, 7]. Two values packed per byte: lo nibble + hi nibble.
    auto pack_int4 = [](int lo, int hi) -> uint8_t {
        return static_cast<uint8_t>((lo & 0xF) | ((hi & 0xF) << 4));
    };
    auto unpack_lo = [](uint8_t b) -> int {
        int v = b & 0xF;
        return (v >= 8) ? (v - 16) : v;
    };
    auto unpack_hi = [](uint8_t b) -> int {
        int v = (b >> 4) & 0xF;
        return (v >= 8) ? (v - 16) : v;
    };

    // Test all valid INT4 pairs
    for (int lo = -8; lo <= 7; lo++) {
        for (int hi = -8; hi <= 7; hi++) {
            uint8_t packed = pack_int4(lo, hi);
            EXPECT_EQ(unpack_lo(packed), lo) << "lo=" << lo << " hi=" << hi;
            EXPECT_EQ(unpack_hi(packed), hi) << "lo=" << lo << " hi=" << hi;
        }
    }
}

// =========================================================================
// INT4 paged attention decode: single head, short context
// =========================================================================

TEST(PagedAttentionINT4Test, DecodeSingleHead) {
    constexpr int batch = 1, n_heads = 1, n_kv_heads = 1, head_dim = 64;
    constexpr int seq_len = 8;
    constexpr int num_blocks = 1;
    constexpr int max_blocks = 1;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    srand(42);
    // Generate FP16 Q and FP32 K/V reference data
    std::vector<float> h_Q(head_dim), h_K(seq_len * head_dim), h_V(seq_len * head_dim);
    for (int i = 0; i < head_dim; i++)
        h_Q[i] = sinf(static_cast<float>(i) * 0.1f);
    for (int i = 0; i < seq_len * head_dim; i++) {
        h_K[i] = cosf(static_cast<float>(i) * 0.05f) * 0.5f;
        h_V[i] = sinf(static_cast<float>(i) * 0.03f + 1.0f) * 0.5f;
    }

    // Quantize K/V to INT4: per-head-per-slot scale, packed uint8 [head_dim/2]
    // INT4 KV cache layout: [num_blocks, block_size, n_kv_heads, head_dim/2] packed uint8
    // Scales: [num_blocks, block_size, n_kv_heads] FP16
    int half_hd = head_dim / 2;
    int cache_bytes = num_blocks * BLOCK_SIZE * n_kv_heads * half_hd;
    int scale_elems = num_blocks * BLOCK_SIZE * n_kv_heads;
    std::vector<uint8_t> k_int4(cache_bytes, 0), v_int4(cache_bytes, 0);
    std::vector<half> k_scales(scale_elems, __float2half(0.0f));
    std::vector<half> v_scales(scale_elems, __float2half(0.0f));

    // K/V dequantized via INT4 (for CPU reference)
    std::vector<float> k_deq(seq_len * head_dim, 0.0f), v_deq(seq_len * head_dim, 0.0f);

    auto quant_and_fill = [&](const std::vector<float>& src, std::vector<uint8_t>& dst_int4,
                              std::vector<half>& dst_scales, std::vector<float>& deq_out) {
        for (int s = 0; s < seq_len; s++) {
            int slot = s % BLOCK_SIZE;
            int block = s / BLOCK_SIZE;
            const float* row = src.data() + s * head_dim;
            // Find max abs for scale
            float amax = 0;
            for (int d = 0; d < head_dim; d++)
                amax = fmaxf(amax, fabsf(row[d]));
            float sc = amax / 7.0f;  // INT4 signed max = 7
            dst_scales[block * BLOCK_SIZE * n_kv_heads + slot * n_kv_heads + 0] = __float2half(sc);
            float inv_sc = (sc > 0) ? 1.0f / sc : 0.0f;
            int base = block * BLOCK_SIZE * n_kv_heads * half_hd + slot * n_kv_heads * half_hd;
            for (int d = 0; d < head_dim; d += 2) {
                int q0 = static_cast<int>(roundf(row[d] * inv_sc));
                int q1 = static_cast<int>(roundf(row[d + 1] * inv_sc));
                q0 = std::max(-8, std::min(7, q0));
                q1 = std::max(-8, std::min(7, q1));
                dst_int4[base + d / 2] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
                // Dequant for CPU reference
                deq_out[s * head_dim + d] = q0 * sc;
                deq_out[s * head_dim + d + 1] = q1 * sc;
            }
        }
    };
    quant_and_fill(h_K, k_int4, k_scales, k_deq);
    quant_and_fill(h_V, v_int4, v_scales, v_deq);

    // CPU reference attention using dequantized INT4 K/V
    std::vector<float> h_O(head_dim, 0.0f);
    cpu_attention(h_Q.data(), k_deq.data(), v_deq.data(), h_O.data(), seq_len, head_dim, scale);

    // Upload to GPU
    Tensor d_Q = make_gpu_tensor_fp16(h_Q.data(), {batch, 1, n_heads, head_dim});

    // INT4 cache: [num_blocks, block_size, n_kv_heads, head_dim/2]
    Tensor d_K_cache, d_V_cache;
    d_K_cache.qtype = QType::INT8;  // raw bytes
    d_K_cache.ndim = 4;
    d_K_cache.shape[0] = num_blocks;
    d_K_cache.shape[1] = BLOCK_SIZE;
    d_K_cache.shape[2] = n_kv_heads;
    d_K_cache.shape[3] = half_hd;
    d_K_cache.compute_strides();
    d_K_cache.on_device = true;
    cudaMalloc(&d_K_cache.data, cache_bytes);
    cudaMemcpy(d_K_cache.data, k_int4.data(), cache_bytes, cudaMemcpyHostToDevice);

    d_V_cache.qtype = QType::INT8;
    d_V_cache.ndim = 4;
    d_V_cache.shape[0] = num_blocks;
    d_V_cache.shape[1] = BLOCK_SIZE;
    d_V_cache.shape[2] = n_kv_heads;
    d_V_cache.shape[3] = half_hd;
    d_V_cache.compute_strides();
    d_V_cache.on_device = true;
    cudaMalloc(&d_V_cache.data, cache_bytes);
    cudaMemcpy(d_V_cache.data, v_int4.data(), cache_bytes, cudaMemcpyHostToDevice);

    Tensor d_O = alloc_gpu_tensor_fp16({batch, 1, n_heads, head_dim});

    half *d_k_scales, *d_v_scales;
    cudaMalloc(&d_k_scales, scale_elems * sizeof(half));
    cudaMalloc(&d_v_scales, scale_elems * sizeof(half));
    cudaMemcpy(d_k_scales, k_scales.data(), scale_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_scales, v_scales.data(), scale_elems * sizeof(half), cudaMemcpyHostToDevice);

    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    cudaMalloc(&d_bt, max_blocks * sizeof(int));
    cudaMalloc(&d_ctx, sizeof(int));
    int bt_val = 0, ctx = seq_len;
    cudaMemcpy(d_bt, &bt_val, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice);

    paged_attention_decode_int4(d_Q, d_K_cache, d_V_cache, d_O, d_k_scales, d_v_scales, d_bt, d_ctx,
                                BLOCK_SIZE, scale, seq_len);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "INT4 paged attention kernel failed";

    auto result = read_gpu_fp16(d_O);
    for (int d = 0; d < head_dim; d++) {
        EXPECT_NEAR(result[d], h_O[d], 0.15f) << "INT4 paged attention mismatch at dim " << d;
    }

    free_gpu(d_Q);
    free_gpu(d_O);
    cudaFree(d_K_cache.data);
    cudaFree(d_V_cache.data);
    cudaFree(d_k_scales);
    cudaFree(d_v_scales);
    cudaFree(d_bt);
    cudaFree(d_ctx);
}

}  // namespace
}  // namespace imp
