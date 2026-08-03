#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "exec/executor_kernels.h"

#include <vector>
#include <cmath>
#include <numeric>

namespace imp {
namespace {

// ── Constants matching typical model configs ────────────────────────────

static constexpr int kBlockSize = 16;  // kKVBlockSize

// ── Helpers ─────────────────────────────────────────────────────────────

struct MockPagedCache {
    int n_kv_heads;
    int head_dim;
    int n_blocks;
    int block_size;

    int row_elems;     // n_kv_heads * head_dim
    int block_stride;  // block_size * row_elems

    half* d_cache = nullptr;

    MockPagedCache(int nkv, int hd, int nblocks, int bs = kBlockSize)
        : n_kv_heads(nkv),
          head_dim(hd),
          n_blocks(nblocks),
          block_size(bs),
          row_elems(nkv * hd),
          block_stride(bs * nkv * hd) {
        size_t total = static_cast<size_t>(n_blocks) * block_stride * sizeof(half);
        cudaMalloc(&d_cache, total);
        cudaMemset(d_cache, 0, total);
    }

    ~MockPagedCache() {
        if (d_cache)
            cudaFree(d_cache);
    }

    // Read one slot from the cache (block_id, slot_in_block) → host vector
    std::vector<float> read_slot(int block_id, int slot) {
        std::vector<half> h(row_elems);
        half* src = d_cache + static_cast<int64_t>(block_id) * block_stride +
                    static_cast<int64_t>(slot) * row_elems;
        cudaMemcpy(h.data(), src, row_elems * sizeof(half), cudaMemcpyDeviceToHost);
        std::vector<float> result(row_elems);
        for (int i = 0; i < row_elems; i++)
            result[i] = __half2float(h[i]);
        return result;
    }
};

// =========================================================================
// Paged slot resolution — multiple tokens, flat block table, crossing a
// block boundary. Exercised through write_kv_cache_fused_kernel: the
// non-fused write_kv_cache_kernel was removed once it had no production
// caller, so the shared kv_resolve_slot() logic is asserted on the kernel
// the engine actually launches.
// =========================================================================

TEST(KVCacheWriteTest, BasicPagedWrite) {
    // Setup: 2 tokens at positions 3 and 19 (block 0 slot 3, block 1 slot 3)
    const int n_kv_heads = 2, head_dim = 64;
    const int n_tokens = 2;
    const int row_elems = n_kv_heads * head_dim;

    // Input data: token t → K = t+1, V = 10*(t+1), so a K/V mix-up is visible.
    std::vector<half> h_k(n_tokens * row_elems), h_v(n_tokens * row_elems);
    for (int t = 0; t < n_tokens; t++)
        for (int i = 0; i < row_elems; i++) {
            h_k[t * row_elems + i] = __float2half(static_cast<float>(t + 1));
            h_v[t * row_elems + i] = __float2half(10.0f * static_cast<float>(t + 1));
        }

    half *d_k, *d_v;
    cudaMalloc(&d_k, h_k.size() * sizeof(half));
    cudaMalloc(&d_v, h_v.size() * sizeof(half));
    cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(half), cudaMemcpyHostToDevice);

    // Positions: token 0 → pos 3, token 1 → pos 19
    int h_positions[2] = {3, 19};
    int* d_positions;
    cudaMalloc(&d_positions, 2 * sizeof(int));
    cudaMemcpy(d_positions, h_positions, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Block table: flat [block0_id=5, block1_id=2]
    // pos 3 → block_idx=0 (3/16=0), slot=3 → physical block 5
    // pos 19 → block_idx=1 (19/16=1), slot=3 → physical block 2
    int h_bt[2] = {5, 2};
    int* d_bt;
    cudaMalloc(&d_bt, 2 * sizeof(int));
    cudaMemcpy(d_bt, h_bt, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate caches with 8 physical blocks
    MockPagedCache k_cache(n_kv_heads, head_dim, 8);
    MockPagedCache v_cache(n_kv_heads, head_dim, 8);

    dim3 grid(n_tokens, 2);  // blockIdx.y: 0=K, 1=V
    write_kv_cache_fused_kernel<<<grid, 256, 0, nullptr>>>(d_k, d_v, d_positions, d_bt, k_cache.d_cache,
                                                           v_cache.d_cache, k_cache.block_stride, row_elems,
                                                           kBlockSize, n_tokens,
                                                           0 /* max_blocks_per_seq=0 → flat */,
                                                           1 /* n_sequences */);
    cudaDeviceSynchronize();

    // Verify: block 5, slot 3 holds token 0 (K=1.0, V=10.0)
    auto k_slot0 = k_cache.read_slot(5, 3);
    auto v_slot0 = v_cache.read_slot(5, 3);
    for (int i = 0; i < row_elems; i++) {
        EXPECT_NEAR(k_slot0[i], 1.0f, 0.01f) << "K block 5, slot 3, index " << i;
        EXPECT_NEAR(v_slot0[i], 10.0f, 0.01f) << "V block 5, slot 3, index " << i;
    }

    // Verify: block 2, slot 3 holds token 1 (K=2.0, V=20.0) — block_idx 1, i.e.
    // the position crossed a block boundary.
    auto k_slot1 = k_cache.read_slot(2, 3);
    auto v_slot1 = v_cache.read_slot(2, 3);
    for (int i = 0; i < row_elems; i++) {
        EXPECT_NEAR(k_slot1[i], 2.0f, 0.01f) << "K block 2, slot 3, index " << i;
        EXPECT_NEAR(v_slot1[i], 20.0f, 0.01f) << "V block 2, slot 3, index " << i;
    }

    // Verify: block 0, slot 0 should still be zero (untouched)
    auto untouched = k_cache.read_slot(0, 0);
    for (int i = 0; i < row_elems; i++) {
        EXPECT_FLOAT_EQ(untouched[i], 0.0f) << "Block 0, slot 0 should be untouched";
    }

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_positions);
    cudaFree(d_bt);
}

// =========================================================================
// write_kv_cache_fused_kernel: K+V in one launch
// =========================================================================

TEST(KVCacheWriteTest, FusedKVWrite) {
    const int n_kv_heads = 4, head_dim = 32;
    const int n_tokens = 1;
    const int row_elems = n_kv_heads * head_dim;

    // K = all 3.0, V = all 7.0
    std::vector<half> h_k(row_elems), h_v(row_elems);
    for (int i = 0; i < row_elems; i++) {
        h_k[i] = __float2half(3.0f);
        h_v[i] = __float2half(7.0f);
    }

    half *d_k, *d_v;
    cudaMalloc(&d_k, row_elems * sizeof(half));
    cudaMalloc(&d_v, row_elems * sizeof(half));
    cudaMemcpy(d_k, h_k.data(), row_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), row_elems * sizeof(half), cudaMemcpyHostToDevice);

    int h_pos[1] = {5};  // pos 5 → block 0, slot 5
    int* d_pos;
    cudaMalloc(&d_pos, sizeof(int));
    cudaMemcpy(d_pos, h_pos, sizeof(int), cudaMemcpyHostToDevice);

    int h_bt[1] = {0};  // block 0
    int* d_bt;
    cudaMalloc(&d_bt, sizeof(int));
    cudaMemcpy(d_bt, h_bt, sizeof(int), cudaMemcpyHostToDevice);

    MockPagedCache k_cache(n_kv_heads, head_dim, 4);
    MockPagedCache v_cache(n_kv_heads, head_dim, 4);

    dim3 grid(n_tokens, 2);  // blockIdx.y: 0=K, 1=V
    write_kv_cache_fused_kernel<<<grid, 256, 0, nullptr>>>(d_k, d_v, d_pos, d_bt, k_cache.d_cache,
                                                           v_cache.d_cache, k_cache.block_stride, row_elems,
                                                           kBlockSize, n_tokens, 0, 1);
    cudaDeviceSynchronize();

    // K cache: block 0, slot 5 = 3.0
    auto k_slot = k_cache.read_slot(0, 5);
    for (int i = 0; i < row_elems; i++) {
        EXPECT_NEAR(k_slot[i], 3.0f, 0.01f) << "K mismatch at " << i;
    }

    // V cache: block 0, slot 5 = 7.0
    auto v_slot = v_cache.read_slot(0, 5);
    for (int i = 0; i < row_elems; i++) {
        EXPECT_NEAR(v_slot[i], 7.0f, 0.01f) << "V mismatch at " << i;
    }

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_pos);
    cudaFree(d_bt);
}

// =========================================================================
// Batched block table (n_sequences > 1) — the 2-D indexing
// bt[seq * max_blocks_per_seq + block_idx] that every paged write kernel
// shares. This is the only direct assertion on it in the write path, so it
// rides the fused kernel now that the non-fused one is gone.
// =========================================================================

TEST(KVCacheWriteTest, BatchedBlockTable) {
    const int n_kv_heads = 2, head_dim = 64;
    const int row_elems = n_kv_heads * head_dim;

    // 2 sequences, each decode (1 token each), so n_tokens=2
    const int n_tokens = 2;
    const int max_blocks_per_seq = 3;

    // Input: token 0 → K 10.0 / V 100.0, token 1 → K 20.0 / V 200.0
    std::vector<half> h_k(n_tokens * row_elems), h_v(n_tokens * row_elems);
    for (int i = 0; i < row_elems; i++) {
        h_k[i] = __float2half(10.0f);
        h_k[row_elems + i] = __float2half(20.0f);
        h_v[i] = __float2half(100.0f);
        h_v[row_elems + i] = __float2half(200.0f);
    }

    half *d_k, *d_v;
    cudaMalloc(&d_k, h_k.size() * sizeof(half));
    cudaMalloc(&d_v, h_v.size() * sizeof(half));
    cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(half), cudaMemcpyHostToDevice);

    // Positions: seq0 at pos 8, seq1 at pos 33
    // pos 8 → block_idx 0, slot 8
    // pos 33 → block_idx 2, slot 1
    int h_pos[2] = {8, 33};
    int* d_pos;
    cudaMalloc(&d_pos, 2 * sizeof(int));
    cudaMemcpy(d_pos, h_pos, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // 2D block table: [seq0: 1, 4, 6], [seq1: 3, 7, 0]
    // seq0, block_idx 0 → physical block 1
    // seq1, block_idx 2 → physical block 0
    int h_bt[6] = {1, 4, 6, 3, 7, 0};
    int* d_bt;
    cudaMalloc(&d_bt, 6 * sizeof(int));
    cudaMemcpy(d_bt, h_bt, 6 * sizeof(int), cudaMemcpyHostToDevice);

    MockPagedCache k_cache(n_kv_heads, head_dim, 10);
    MockPagedCache v_cache(n_kv_heads, head_dim, 10);

    dim3 grid(n_tokens, 2);  // blockIdx.y: 0=K, 1=V
    write_kv_cache_fused_kernel<<<grid, 256, 0, nullptr>>>(d_k, d_v, d_pos, d_bt, k_cache.d_cache,
                                                           v_cache.d_cache, k_cache.block_stride, row_elems,
                                                           kBlockSize, n_tokens, max_blocks_per_seq,
                                                           2 /* n_sequences */);
    cudaDeviceSynchronize();

    // seq0: pos 8 → block_idx 0, slot 8. bt[0*3+0]=1 → physical block 1
    auto k0 = k_cache.read_slot(1, 8);
    auto v0 = v_cache.read_slot(1, 8);
    for (int i = 0; i < row_elems; i++) {
        EXPECT_NEAR(k0[i], 10.0f, 0.01f) << "Seq0 K mismatch at " << i;
        EXPECT_NEAR(v0[i], 100.0f, 0.01f) << "Seq0 V mismatch at " << i;
    }

    // seq1: pos 33 → block_idx 2, slot 1. bt[1*3+2]=0 → physical block 0
    auto k1 = k_cache.read_slot(0, 1);
    auto v1 = v_cache.read_slot(0, 1);
    for (int i = 0; i < row_elems; i++) {
        EXPECT_NEAR(k1[i], 20.0f, 0.01f) << "Seq1 K mismatch at " << i;
        EXPECT_NEAR(v1[i], 200.0f, 0.01f) << "Seq1 V mismatch at " << i;
    }

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_pos);
    cudaFree(d_bt);
}

// =========================================================================
// write_kv_cache_int8_kernel: verify per-head scale and quantized data
// =========================================================================

TEST(KVCacheWriteTest, Int8PerHeadScale) {
    const int n_kv_heads = 2, head_dim = 64;
    const int row_elems = n_kv_heads * head_dim;
    const int n_tokens = 1;

    // Head 0: values in [-1, 1], Head 1: values in [-10, 10]
    std::vector<half> h_data(row_elems);
    for (int d = 0; d < head_dim; d++) {
        float v0 = static_cast<float>(d) / head_dim * 2.0f - 1.0f;    // [-1, 1]
        float v1 = static_cast<float>(d) / head_dim * 20.0f - 10.0f;  // [-10, 10]
        h_data[0 * head_dim + d] = __float2half(v0);
        h_data[1 * head_dim + d] = __float2half(v1);
    }

    half* d_data;
    cudaMalloc(&d_data, row_elems * sizeof(half));
    cudaMemcpy(d_data, h_data.data(), row_elems * sizeof(half), cudaMemcpyHostToDevice);

    int h_pos[1] = {0};
    int* d_pos;
    cudaMalloc(&d_pos, sizeof(int));
    cudaMemcpy(d_pos, h_pos, sizeof(int), cudaMemcpyHostToDevice);

    int h_bt[1] = {0};
    int* d_bt;
    cudaMalloc(&d_bt, sizeof(int));
    cudaMemcpy(d_bt, h_bt, sizeof(int), cudaMemcpyHostToDevice);

    // INT8 cache: block_stride = block_size * n_kv_heads * head_dim (int8 elems)
    int block_stride = kBlockSize * row_elems;
    int scale_block_stride = kBlockSize * n_kv_heads;

    int8_t* d_k_cache;
    int8_t* d_v_cache;
    half* d_k_scale;
    half* d_v_scale;
    cudaMalloc(&d_k_cache, block_stride * sizeof(int8_t));
    cudaMalloc(&d_v_cache, block_stride * sizeof(int8_t));
    cudaMalloc(&d_k_scale, scale_block_stride * sizeof(half));
    cudaMalloc(&d_v_scale, scale_block_stride * sizeof(half));
    cudaMemset(d_k_cache, 0, block_stride * sizeof(int8_t));
    cudaMemset(d_v_cache, 0, block_stride * sizeof(int8_t));
    cudaMemset(d_k_scale, 0, scale_block_stride * sizeof(half));
    cudaMemset(d_v_scale, 0, scale_block_stride * sizeof(half));

    // Launch: blockIdx.y=0 writes K
    dim3 grid(n_tokens, 2);
    write_kv_cache_int8_kernel<<<grid, 256, 0, nullptr>>>(d_data, d_data,  // K and V same data for simplicity
                                                          d_pos, d_bt, d_k_cache, d_v_cache, d_k_scale,
                                                          d_v_scale, block_stride, scale_block_stride,
                                                          n_kv_heads, head_dim, kBlockSize, n_tokens, 0, 1);
    cudaDeviceSynchronize();

    // Read K scales for slot 0
    std::vector<half> h_scales(n_kv_heads);
    cudaMemcpy(h_scales.data(), d_k_scale, n_kv_heads * sizeof(half), cudaMemcpyDeviceToHost);

    float scale0 = __half2float(h_scales[0]);
    float scale1 = __half2float(h_scales[1]);

    // Head 0 max ~1.0 → scale ~1/127 ≈ 0.0079
    // Head 1 max ~10.0 → scale ~10/127 ≈ 0.0787
    EXPECT_GT(scale0, 0.005f);
    EXPECT_LT(scale0, 0.015f);
    EXPECT_GT(scale1, 0.05f);
    EXPECT_LT(scale1, 0.15f);
    // Head 1 scale should be ~10x head 0 scale
    EXPECT_NEAR(scale1 / scale0, 10.0f, 2.0f);

    // Read INT8 data and verify dequantized values are close
    std::vector<int8_t> h_int8(row_elems);
    cudaMemcpy(h_int8.data(), d_k_cache, row_elems * sizeof(int8_t), cudaMemcpyDeviceToHost);

    for (int d = 0; d < head_dim; d++) {
        float original = __half2float(h_data[d]);
        float dequant = static_cast<float>(h_int8[d]) * scale0;
        EXPECT_NEAR(dequant, original, 0.02f) << "Head 0, dim " << d;
    }

    cudaFree(d_data);
    cudaFree(d_pos);
    cudaFree(d_bt);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_k_scale);
    cudaFree(d_v_scale);
}

}  // anonymous namespace
}  // namespace imp
