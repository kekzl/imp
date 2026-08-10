#include "compute/kv_gather.h"
#include "exec/executor_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
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

// INT8 gather, tested against the writer that produces the layout it reads.
//
// A standalone gather test would have to restate the INT8 cache layout, which
// is exactly the thing that can drift; #1348 was the chunked-prefill path
// aborting on INT8 KV for want of this kernel. So the oracle here is a real
// round trip: quantize FP16 through `write_kv_cache_int8_kernel`, gather it
// back, and require the result within one quantization step of the input.
TEST(KVGatherTest, INT8_WriteThenGather_RoundTrip) {
    const int num_blocks = 4;
    const int block_size = 16;
    const int nkv = 2;
    const int hd = 64;
    const int n_past = 40;  // 2 full blocks + 8 tokens
    const int row_elems = nkv * hd;

    // Permuted logical → physical mapping, so a gather that ignores the block
    // table reads the wrong rows.
    std::vector<int> h_bt = {2, 0, 3};

    // Per-head amplitude differs by 10x so a per-head scale is required to
    // reproduce both: a single shared scale would blow the tolerance on head 0.
    std::vector<half> h_in((size_t)n_past * row_elems);
    for (int t = 0; t < n_past; t++) {
        for (int k = 0; k < nkv; k++) {
            for (int d = 0; d < hd; d++) {
                float amp = (k == 0) ? 1.0f : 10.0f;
                float v = amp * std::sin(0.1f * (float)(t * hd + d) + (float)k);
                h_in[(size_t)t * row_elems + (size_t)k * hd + d] = __float2half(v);
            }
        }
    }
    std::vector<int> h_pos(n_past);
    for (int t = 0; t < n_past; t++)
        h_pos[t] = t;

    const int block_stride = block_size * row_elems;  // int8 elems
    const int scale_block_stride = block_size * nkv;  // half elems

    half* d_in;
    int* d_pos;
    int* d_bt;
    int8_t* d_k_cache;
    int8_t* d_v_cache;
    half* d_k_scale;
    half* d_v_scale;
    half* d_dst;
    cudaMalloc(&d_in, h_in.size() * sizeof(half));
    cudaMalloc(&d_pos, n_past * sizeof(int));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_k_cache, (size_t)num_blocks * block_stride * sizeof(int8_t));
    cudaMalloc(&d_v_cache, (size_t)num_blocks * block_stride * sizeof(int8_t));
    cudaMalloc(&d_k_scale, (size_t)num_blocks * scale_block_stride * sizeof(half));
    cudaMalloc(&d_v_scale, (size_t)num_blocks * scale_block_stride * sizeof(half));
    cudaMalloc(&d_dst, (size_t)n_past * row_elems * sizeof(half));
    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, h_pos.data(), n_past * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(n_past, 2);
    write_kv_cache_int8_kernel<<<grid, 256, 0, nullptr>>>(d_in, d_in, d_pos, d_bt, d_k_cache, d_v_cache,
                                                          d_k_scale, d_v_scale, block_stride,
                                                          scale_block_stride, nkv, hd, block_size, n_past, 0,
                                                          1);
    cudaDeviceSynchronize();

    paged_kv_gather_int8_to_fp16(d_dst, d_k_cache, d_k_scale, d_bt, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * row_elems);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    for (int t = 0; t < n_past; t++) {
        for (int k = 0; k < nkv; k++) {
            // Tolerance is one quantization step of this head's own scale.
            float amax = 0.0f;
            for (int d = 0; d < hd; d++)
                amax = std::max(amax,
                                std::fabs(__half2float(h_in[(size_t)t * row_elems + (size_t)k * hd + d])));
            float tol = amax / 127.0f * 0.55f + 1e-3f;
            for (int d = 0; d < hd; d++) {
                size_t idx = (size_t)t * row_elems + (size_t)k * hd + d;
                EXPECT_NEAR(__half2float(h_dst[idx]), __half2float(h_in[idx]), tol)
                    << "t=" << t << " k=" << k << " d=" << d;
            }
        }
    }

    cudaFree(d_in);
    cudaFree(d_pos);
    cudaFree(d_bt);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_k_scale);
    cudaFree(d_v_scale);
    cudaFree(d_dst);
}

// A -1 block-table entry is the SWA/StreamingLLM hole sentinel; the gather must
// zero-fill it rather than index the pool at -1.
TEST(KVGatherTest, INT8_NegativeBlockSentinelZeroFills) {
    const int num_blocks = 2;
    const int block_size = 16;
    const int nkv = 2;
    const int hd = 32;
    const int n_past = 32;  // logical block 0 = hole, block 1 = real
    const int row_elems = nkv * hd;

    std::vector<int> h_bt = {-1, 1};

    size_t cache_elems = (size_t)num_blocks * block_size * row_elems;
    std::vector<int8_t> h_cache(cache_elems);
    for (size_t i = 0; i < cache_elems; i++)
        h_cache[i] = static_cast<int8_t>((i % 255) - 127);
    std::vector<half> h_scales((size_t)num_blocks * block_size * nkv, __float2half(0.5f));

    int8_t* d_cache;
    half* d_scales;
    int* d_bt;
    half* d_dst;
    cudaMalloc(&d_cache, cache_elems);
    cudaMalloc(&d_scales, h_scales.size() * sizeof(half));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_dst, (size_t)n_past * row_elems * sizeof(half));
    cudaMemcpy(d_cache, h_cache.data(), cache_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice);

    paged_kv_gather_int8_to_fp16(d_dst, d_cache, d_scales, d_bt, n_past, block_size, nkv, hd, 0);
    cudaDeviceSynchronize();

    std::vector<half> h_dst((size_t)n_past * row_elems);
    cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(half), cudaMemcpyDeviceToHost);

    for (int pos = 0; pos < block_size; pos++)
        for (int i = 0; i < row_elems; i++)
            EXPECT_EQ(__half2float(h_dst[(size_t)pos * row_elems + i]), 0.0f) << "pos=" << pos;

    // The real block behind the hole still dequantizes.
    for (int pos = block_size; pos < n_past; pos++) {
        int slot = pos % block_size;
        for (int k = 0; k < nkv; k++) {
            for (int d = 0; d < hd; d++) {
                size_t src_idx = ((size_t)1 * block_size + slot) * row_elems + (size_t)k * hd + d;
                size_t dst_idx = (size_t)pos * row_elems + (size_t)k * hd + d;
                EXPECT_NEAR(__half2float(h_dst[dst_idx]), (float)h_cache[src_idx] * 0.5f, 0.05f);
            }
        }
    }

    cudaFree(d_cache);
    cudaFree(d_scales);
    cudaFree(d_bt);
    cudaFree(d_dst);
}

}  // namespace imp
