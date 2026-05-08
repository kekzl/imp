#include "compute/kv_gather.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <vector>
#include <random>

namespace imp {

// Build a synthetic paged FP16 cache with deterministic values:
//   src[block, slot, kv_head, d] = float(block * 1000 + slot * 10 + kv_head + d * 0.01)
// Then verify gather to flat layout reads back via block_table.
TEST(KVGatherTest, FP16_PagedToFlat_RoundTrip) {
    const int num_blocks = 8;
    const int block_size = 16;
    const int nkv = 4;
    const int hd = 64;
    const int n_past = 100;  // 100 tokens → 7 full blocks + partial

    // Permuted block_table: maps logical block_idx → physical block.
    std::vector<int> h_bt = {3, 1, 0, 5, 2, 7, 4, 6};

    size_t total_elems = (size_t)num_blocks * block_size * nkv * hd;
    std::vector<half> h_src(total_elems);
    for (int b = 0; b < num_blocks; b++) {
        for (int s = 0; s < block_size; s++) {
            for (int k = 0; k < nkv; k++) {
                for (int d = 0; d < hd; d++) {
                    float v = (float)b + 0.001f * (float)s + 0.0001f * (float)k + 0.00001f * (float)d;
                    size_t idx = ((size_t)b * block_size + s) * nkv * hd + (size_t)k * hd + d;
                    h_src[idx] = __float2half(v);
                }
            }
        }
    }

    half* d_src;
    int* d_bt;
    half* d_dst;
    cudaMalloc(&d_src, total_elems * sizeof(half));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_dst, (size_t)n_past * nkv * hd * sizeof(half));
    cudaMemcpy(d_src, h_src.data(), total_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    paged_kv_gather_fp16(d_dst, d_src, d_bt, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * nkv * hd);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Verify: dst[pos, kv_head, d] should equal src[block_table[pos/bs], pos%bs, kv_head, d].
    for (int pos = 0; pos < n_past; pos++) {
        int phys_block = h_bt[pos / block_size];
        int slot = pos % block_size;
        for (int k = 0; k < nkv; k++) {
            for (int d = 0; d < hd; d++) {
                size_t src_idx = ((size_t)phys_block * block_size + slot) * nkv * hd + (size_t)k * hd + d;
                size_t dst_idx = (size_t)pos * nkv * hd + (size_t)k * hd + d;
                ASSERT_EQ(__half_as_ushort(h_dst[dst_idx]), __half_as_ushort(h_src[src_idx]))
                    << "pos=" << pos << " k=" << k << " d=" << d;
            }
        }
    }

    cudaFree(d_src);
    cudaFree(d_bt);
    cudaFree(d_dst);
}

TEST(KVGatherTest, FP16_PartialLastBlock) {
    // n_past = block_size + 1 → last block has exactly 1 valid slot.
    const int num_blocks = 4;
    const int block_size = 16;
    const int nkv = 2;
    const int hd = 32;
    const int n_past = 17;

    std::vector<int> h_bt = {2, 0};  // need 2 blocks for 17 tokens
    size_t total_elems = (size_t)num_blocks * block_size * nkv * hd;
    std::vector<half> h_src(total_elems, __float2half(0.f));
    // Mark slot 0 of physical block 0 with a sentinel
    for (int k = 0; k < nkv; k++)
        for (int d = 0; d < hd; d++)
            h_src[((size_t)0 * block_size + 0) * nkv * hd + (size_t)k * hd + d] = __float2half(42.0f);

    half* d_src; int* d_bt; half* d_dst;
    cudaMalloc(&d_src, total_elems * sizeof(half));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_dst, (size_t)n_past * nkv * hd * sizeof(half));
    cudaMemcpy(d_src, h_src.data(), total_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    paged_kv_gather_fp16(d_dst, d_src, d_bt, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * nkv * hd);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Token 16 (the partial-last-block token) should be the sentinel 42.0.
    for (int k = 0; k < nkv; k++) {
        for (int d = 0; d < hd; d++) {
            size_t dst_idx = (size_t)16 * nkv * hd + (size_t)k * hd + d;
            EXPECT_NEAR(__half2float(h_dst[dst_idx]), 42.0f, 0.001f);
        }
    }

    cudaFree(d_src); cudaFree(d_bt); cudaFree(d_dst);
}

TEST(KVGatherTest, FP8_PagedToFlat_DequantMatchesReference) {
    const int num_blocks = 4;
    const int block_size = 16;
    const int nkv = 2;
    const int hd = 32;
    const int n_past = 32;
    const float kv_scale = 0.25f;

    std::vector<int> h_bt = {1, 3};
    size_t total_elems = (size_t)num_blocks * block_size * nkv * hd;

    // Synthesize FP8 values via float→fp8 conversion.
    std::vector<__nv_fp8_e4m3> h_src(total_elems);
    for (size_t i = 0; i < total_elems; i++) {
        float v = (float)((i % 17) - 8);  // small range, representable in FP8
        h_src[i] = __nv_fp8_e4m3(v);
    }

    __nv_fp8_e4m3* d_src; int* d_bt; half* d_dst;
    cudaMalloc(&d_src, total_elems * sizeof(__nv_fp8_e4m3));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_dst, (size_t)n_past * nkv * hd * sizeof(half));
    cudaMemcpy(d_src, h_src.data(), total_elems * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    paged_kv_gather_fp8_to_fp16(d_dst, d_src, d_bt, kv_scale, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * nkv * hd);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    for (int pos = 0; pos < n_past; pos++) {
        int phys_block = h_bt[pos / block_size];
        int slot = pos % block_size;
        for (int k = 0; k < nkv; k++) {
            for (int d = 0; d < hd; d++) {
                size_t src_idx = ((size_t)phys_block * block_size + slot) * nkv * hd + (size_t)k * hd + d;
                size_t dst_idx = (size_t)pos * nkv * hd + (size_t)k * hd + d;
                float expected = static_cast<float>(h_src[src_idx]) * kv_scale;
                EXPECT_NEAR(__half2float(h_dst[dst_idx]), expected, 0.005f);  // FP16 round-off
            }
        }
    }

    cudaFree(d_src); cudaFree(d_bt); cudaFree(d_dst);
}

}  // namespace imp
