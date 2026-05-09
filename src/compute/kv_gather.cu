#include "compute/kv_gather.h"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace imp {

// Each thread handles one (token, kv_head, head_dim_elem) triple.
// Grid: (ceil(n_past / TOKENS_PER_BLOCK), nkv).
// Block: TOKENS_PER_BLOCK * (hd / VEC) threads (tunable; here 256 threads).
//
// We use a flat 1D thread index inside the block over (token_in_block, hd_elem)
// to keep the kernel simple and let the compiler vectorize the half loads.
//
// NOTE: __ldcs is a streaming hint — KV bytes don't pollute L2, matching
// paged_attention_decode_fp8 / decode_int8 / decode behavior.

static constexpr int TOKENS_PER_BLOCK = 8;

__global__ void paged_kv_gather_fp16_kernel(half* __restrict__ dst, const half* __restrict__ src,
                                            const int* __restrict__ block_table, int n_past,
                                            int block_size, int nkv, int hd) {
    const int block_group = blockIdx.x;     // group of TOKENS_PER_BLOCK tokens
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int threads_per_token = blockDim.x / TOKENS_PER_BLOCK;
    const int token_in_block = tid / threads_per_token;
    const int d_lane = tid % threads_per_token;

    const int pos = block_group * TOKENS_PER_BLOCK + token_in_block;
    if (pos >= n_past)
        return;

    const int blk_idx = pos / block_size;
    const int slot = pos % block_size;
    const int phys_block = block_table[blk_idx];

    const int kv_block_stride = block_size * nkv * hd;
    const int kv_slot_stride = nkv * hd;

    const half* src_row = src + (size_t)phys_block * kv_block_stride
                              + (size_t)slot * kv_slot_stride
                              + (size_t)kv_head * hd;
    half* dst_row = dst + (size_t)pos * nkv * hd + (size_t)kv_head * hd;

    for (int d = d_lane; d < hd; d += threads_per_token) {
        // Streaming load (skip L1, evict-first from L2) so KV bytes don't pollute
        // L2 for the FFN GEMM that follows. Same hint as paged_attention_decode.
        unsigned short raw = __ldcs(reinterpret_cast<const unsigned short*>(src_row + d));
        dst_row[d] = __ushort_as_half(raw);
    }
}

__global__ void paged_kv_gather_fp8_to_fp16_kernel(half* __restrict__ dst,
                                                    const __nv_fp8_e4m3* __restrict__ src,
                                                    const int* __restrict__ block_table,
                                                    float kv_scale, int n_past, int block_size,
                                                    int nkv, int hd) {
    const int block_group = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int threads_per_token = blockDim.x / TOKENS_PER_BLOCK;
    const int token_in_block = tid / threads_per_token;
    const int d_lane = tid % threads_per_token;

    const int pos = block_group * TOKENS_PER_BLOCK + token_in_block;
    if (pos >= n_past)
        return;

    const int blk_idx = pos / block_size;
    const int slot = pos % block_size;
    const int phys_block = block_table[blk_idx];

    const int kv_block_stride = block_size * nkv * hd;
    const int kv_slot_stride = nkv * hd;

    const __nv_fp8_e4m3* src_row = src + (size_t)phys_block * kv_block_stride
                                       + (size_t)slot * kv_slot_stride
                                       + (size_t)kv_head * hd;
    half* dst_row = dst + (size_t)pos * nkv * hd + (size_t)kv_head * hd;

    for (int d = d_lane; d < hd; d += threads_per_token) {
        // Streaming load via __ldcs on uint8 (FP8 is 1 byte).
        unsigned char raw = __ldcs(reinterpret_cast<const unsigned char*>(src_row + d));
        __nv_fp8_e4m3 fp8;
        memcpy(&fp8, &raw, 1);
        float f = static_cast<float>(fp8);
        dst_row[d] = __float2half(f * kv_scale);
    }
}

void paged_kv_gather_fp16(half* dst, const half* src, const int* block_table, int n_past,
                          int block_size, int nkv, int hd, cudaStream_t stream) {
    if (n_past <= 0 || nkv <= 0 || hd <= 0)
        return;
    int n_block_groups = (n_past + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    dim3 grid(n_block_groups, nkv);
    int threads = 256;  // 8 tokens × 32 lanes; works for hd up to 256 with stride-32, OK for hd=512 too
    paged_kv_gather_fp16_kernel<<<grid, threads, 0, stream>>>(dst, src, block_table, n_past,
                                                              block_size, nkv, hd);
}

void paged_kv_gather_fp8_to_fp16(half* dst, const __nv_fp8_e4m3* src, const int* block_table,
                                 float kv_scale, int n_past, int block_size, int nkv, int hd,
                                 cudaStream_t stream) {
    if (n_past <= 0 || nkv <= 0 || hd <= 0)
        return;
    int n_block_groups = (n_past + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    dim3 grid(n_block_groups, nkv);
    int threads = 256;
    paged_kv_gather_fp8_to_fp16_kernel<<<grid, threads, 0, stream>>>(dst, src, block_table, kv_scale,
                                                                      n_past, block_size, nkv, hd);
}

// NVFP4 → FP16 dequant gather. Same TOKENS_PER_BLOCK / threads_per_token grid
// as the FP16/FP8 variants. Per (token, hd) elem: read packed FP4 byte → decode
// to half2 via `cvt.rn.f16x2.e2m1x2` PTX → pick the correct nibble's half →
// multiply by per-group-of-16 UE4M3 scale → store as FP16. Matches the write
// path in `write_kv_cache_nvfp4_kernel`.
__global__ void paged_kv_gather_nvfp4_to_fp16_kernel(
    half* __restrict__ dst,
    const uint8_t* __restrict__ src_packed,    // [block, slot, nkv, hd/2]
    const uint8_t* __restrict__ src_scales,    // [block, slot, nkv, hd/16] UE4M3
    const int* __restrict__ block_table,
    int n_past, int block_size, int nkv, int hd) {
    constexpr int kGroup = 16;

    const int block_group = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int threads_per_token = blockDim.x / TOKENS_PER_BLOCK;
    const int token_in_block = tid / threads_per_token;
    const int d_lane = tid % threads_per_token;

    const int pos = block_group * TOKENS_PER_BLOCK + token_in_block;
    if (pos >= n_past)
        return;

    const int blk_idx = pos / block_size;
    const int slot = pos % block_size;
    const int phys_block = block_table[blk_idx];

    const int kv_block_stride_bytes = block_size * nkv * (hd / 2);
    const int kv_slot_stride_bytes = nkv * (hd / 2);
    const int sc_block_stride = block_size * nkv * (hd / kGroup);
    const int sc_slot_stride = nkv * (hd / kGroup);

    const uint8_t* src_row = src_packed + (size_t)phys_block * kv_block_stride_bytes
                                        + (size_t)slot * kv_slot_stride_bytes
                                        + (size_t)kv_head * (hd / 2);
    const uint8_t* sc_row = src_scales + (size_t)phys_block * sc_block_stride
                                       + (size_t)slot * sc_slot_stride
                                       + (size_t)kv_head * (hd / kGroup);
    half* dst_row = dst + (size_t)pos * nkv * hd + (size_t)kv_head * hd;

    for (int d = d_lane; d < hd; d += threads_per_token) {
        // FP4 nibble decode: PTX cvt produces a half2 from two packed nibbles.
        const unsigned char byte =
            __ldcs(reinterpret_cast<const unsigned char*>(src_row + (d / 2)));
        unsigned int fp16x2;
        asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
            : "=r"(fp16x2) : "r"((unsigned int)byte));
        half2 hh = *reinterpret_cast<half2*>(&fp16x2);
        half v = (d & 1) ? hh.y : hh.x;

        // UE4M3 scale (per group of 16 hd elems).
        unsigned char sc_byte =
            __ldcs(reinterpret_cast<const unsigned char*>(sc_row + (d / kGroup)));
        __nv_fp8_e4m3 sc_fp8;
        memcpy(&sc_fp8, &sc_byte, 1);
        float scale = static_cast<float>(sc_fp8);

        dst_row[d] = __float2half(__half2float(v) * scale);
    }
}

void paged_kv_gather_nvfp4_to_fp16(half* dst, const uint8_t* src_packed,
                                   const uint8_t* src_scales, const int* block_table,
                                   int n_past, int block_size, int nkv, int hd,
                                   cudaStream_t stream) {
    if (n_past <= 0 || nkv <= 0 || hd <= 0)
        return;
    int n_block_groups = (n_past + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    dim3 grid(n_block_groups, nkv);
    int threads = 256;
    paged_kv_gather_nvfp4_to_fp16_kernel<<<grid, threads, 0, stream>>>(
        dst, src_packed, src_scales, block_table, n_past, block_size, nkv, hd);
}

}  // namespace imp
