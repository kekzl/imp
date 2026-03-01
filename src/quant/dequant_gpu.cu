#include "quant/dequant_gpu.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif
#include <cstdio>

namespace imp {

// ---------------------------------------------------------------------------
// Raw byte count per row for GGML quantized formats
// ---------------------------------------------------------------------------

size_t ggml_quant_row_bytes(GGMLQuantType qtype, int64_t cols) {
    switch (qtype) {
        case GGMLQuantType::Q6_K:  return static_cast<size_t>(cols / 256) * 210;
        case GGMLQuantType::Q8_0:  return static_cast<size_t>(cols / 32) * 34;
        case GGMLQuantType::Q4_0:  return static_cast<size_t>(cols / 32) * 18;
        case GGMLQuantType::Q8_1:  return static_cast<size_t>(cols / 32) * 36;
        case GGMLQuantType::Q4_1:  return static_cast<size_t>(cols / 32) * 20;
        case GGMLQuantType::Q5_0:  return static_cast<size_t>(cols / 32) * 22;
        case GGMLQuantType::Q5_1:  return static_cast<size_t>(cols / 32) * 24;
        case GGMLQuantType::Q2_K:  return static_cast<size_t>(cols / 256) * 84;
        case GGMLQuantType::Q3_K:  return static_cast<size_t>(cols / 256) * 110;
        case GGMLQuantType::Q4_K:  return static_cast<size_t>(cols / 256) * 144;
        case GGMLQuantType::Q5_K:  return static_cast<size_t>(cols / 256) * 176;
        case GGMLQuantType::Q8_K:  return static_cast<size_t>(cols / 256) * 292;
        case GGMLQuantType::F16:
        case GGMLQuantType::BF16:  return static_cast<size_t>(cols) * 2;
        case GGMLQuantType::F32:   return static_cast<size_t>(cols) * 4;  // NONE == F32 == 0
        default:                   return static_cast<size_t>(cols) * 2;
    }
}

bool dequant_gpu_supported(GGMLQuantType qtype) {
    switch (qtype) {
        case GGMLQuantType::Q6_K:
        case GGMLQuantType::Q8_0:
        case GGMLQuantType::Q4_0:
        case GGMLQuantType::Q5_0:
        case GGMLQuantType::Q5_1:
        case GGMLQuantType::Q4_K:
        case GGMLQuantType::Q5_K:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Q6_K GPU dequantization kernel
//
// Block format (210 bytes per 256 elements):
//   ql[128]   : lower 4 bits, GGML interleaved layout
//   qh[64]    : upper 2 bits, GGML interleaved layout
//   scales[16]: int8 sub-block scales (one per 16 elements)
//   d[2]      : fp16 super-block scale
//
// GGML Q6_K packing: 256 values split into 2 groups of 128.
// Each group has 4 sub-groups of 32 (q1, q2, q3, q4).
//   q1 (vals 0..31):   ql[l] low nibble,    qh[l] bits 0..1
//   q2 (vals 32..63):  ql[l+32] low nibble,  qh[l] bits 2..3
//   q3 (vals 64..95):  ql[l] high nibble,   qh[l] bits 4..5
//   q4 (vals 96..127): ql[l+32] high nibble, qh[l] bits 6..7
// Second group (vals 128..255) offsets: ql+=64, qh+=32, sc+=8.
// ---------------------------------------------------------------------------

// Original scalar kernel (fallback)
__global__ void dequant_q6k_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i   = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + static_cast<int64_t>(row * blocks_per_row + blk) * 210;
    const uint8_t* ql    = block_ptr;           // 128 bytes
    const uint8_t* qh    = block_ptr + 128;     // 64 bytes
    const int8_t*  scales = reinterpret_cast<const int8_t*>(block_ptr + 192); // 16 bytes
    half d_val = *reinterpret_cast<const half*>(block_ptr + 208);

    // GGML interleaved layout
    int group  = i >> 7;           // i / 128 → 0 or 1
    int within = i & 127;          // i % 128 → 0..127
    int quad   = within >> 5;      // within / 32 → 0..3
    int l      = within & 31;      // within % 32 → 0..31

    int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    int qh_idx = (group << 5) + l;

    uint8_t ql_byte = ql[ql_idx];
    uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
    uint8_t high2 = (qh[qh_idx] >> (quad * 2)) & 0x3u;
    int q6 = static_cast<int>((high2 << 4) | low4) - 32;

    float val = __half2float(d_val) * static_cast<float>(scales[i >> 4]) * static_cast<float>(q6);
    dst[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Optimized Q6_K dequant kernel — block-centric indexing
//
// One CUDA thread block per Q6_K super-block (256 elements).
// 128 threads, each processing 2 consecutive elements with half2 writes.
//
// Eliminates expensive integer division (row/col from flat index) by
// mapping blockIdx.x directly to a Q6_K block.  Consecutive Q6_K blocks
// in memory map to consecutive blockIdx values, so:
//   src_ptr  = src + blockIdx.x * 210
//   dst_ptr  = dst + blockIdx.x * 256
// No row/col computation needed.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int dequant_q6k_element(
    const uint8_t* __restrict__ bp, int i)
{
    int group  = i >> 7;
    int within = i & 127;
    int quad   = within >> 5;
    int l      = within & 31;

    int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    int qh_idx = (group << 5) + l;

    uint8_t ql_byte = bp[ql_idx];
    uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
    uint8_t high2 = (bp[128 + qh_idx] >> (quad * 2)) & 0x3u;
    return static_cast<int>((high2 << 4) | low4) - 32;
}

__global__ void dequant_q6k_v2_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int total_blocks)
{
    int blk_id = blockIdx.x;
    if (blk_id >= total_blocks) return;

    const uint8_t* bp = src + static_cast<int64_t>(blk_id) * 210;
    half2* out = reinterpret_cast<half2*>(dst + static_cast<int64_t>(blk_id) * 256);

    // Super-block scale — same for all 256 elements.
    // All threads in the block read the same address; hits L1 cache.
    float d_w = __half2float(*reinterpret_cast<const half*>(bp + 208));
    const int8_t* scales = reinterpret_cast<const int8_t*>(bp + 192);

    // Each thread handles 2 consecutive elements → half2 vectorized write
    int i0 = threadIdx.x * 2;
    int i1 = i0 + 1;

    int q0 = dequant_q6k_element(bp, i0);
    int q1 = dequant_q6k_element(bp, i1);

    float sc0 = static_cast<float>(scales[i0 >> 4]);
    float sc1 = static_cast<float>(scales[i1 >> 4]);

    half2 result;
    result.x = __float2half(d_w * sc0 * static_cast<float>(q0));
    result.y = __float2half(d_w * sc1 * static_cast<float>(q1));

    out[threadIdx.x] = result;
}

// ---------------------------------------------------------------------------
// High-throughput Q6_K dequant kernel — grid-stride, no shared memory
//
// v2 launches one CTA per Q6_K super-block (256 elements), creating ~786K
// CTAs for a typical MoE projection. CTA scheduling overhead can limit
// bandwidth utilization on large GPUs (192 SMs).
//
// v3 uses a grid-stride loop: each CTA processes many Q6_K blocks,
// reducing the CTA count. No shared memory or sync barriers needed —
// each thread independently dequants its 2 elements from L1-cached
// global memory reads (the 210-byte Q6K block fits in 2 cache lines,
// broadcast to all threads in the warp).
// ---------------------------------------------------------------------------

__global__ void dequant_q6k_v3_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int total_blocks)
{
    const int tid = threadIdx.x;  // 0..127
    // Pre-compute element indices (constant across loop iterations)
    const int i0 = tid * 2;
    const int i1 = i0 + 1;

    // Pre-compute Q6K decode indices for both elements
    const int group0  = i0 >> 7;
    const int within0 = i0 & 127;
    const int quad0   = within0 >> 5;
    const int l0      = within0 & 31;
    const int ql_idx0 = (group0 << 6) + ((quad0 & 1) << 5) + l0;
    const int qh_idx0 = (group0 << 5) + l0;

    const int group1  = i1 >> 7;
    const int within1 = i1 & 127;
    const int quad1   = within1 >> 5;
    const int l1      = within1 & 31;
    const int ql_idx1 = (group1 << 6) + ((quad1 & 1) << 5) + l1;
    const int qh_idx1 = (group1 << 5) + l1;

    for (int blk_id = blockIdx.x; blk_id < total_blocks; blk_id += gridDim.x) {
        const uint8_t* bp = src + static_cast<int64_t>(blk_id) * 210;

        // Super-block scale — broadcast via L1 cache to all 128 threads
        float d_w = __half2float(*reinterpret_cast<const half*>(bp + 208));
        const int8_t* scales = reinterpret_cast<const int8_t*>(bp + 192);

        // Dequant element 0
        uint8_t ql_byte0 = bp[ql_idx0];
        uint8_t low4_0 = (quad0 >= 2) ? ((ql_byte0 >> 4) & 0xFu) : (ql_byte0 & 0xFu);
        uint8_t high2_0 = (bp[128 + qh_idx0] >> (quad0 * 2)) & 0x3u;
        int q0 = static_cast<int>((high2_0 << 4) | low4_0) - 32;

        // Dequant element 1
        uint8_t ql_byte1 = bp[ql_idx1];
        uint8_t low4_1 = (quad1 >= 2) ? ((ql_byte1 >> 4) & 0xFu) : (ql_byte1 & 0xFu);
        uint8_t high2_1 = (bp[128 + qh_idx1] >> (quad1 * 2)) & 0x3u;
        int q1 = static_cast<int>((high2_1 << 4) | low4_1) - 32;

        float sc0 = static_cast<float>(scales[i0 >> 4]);
        float sc1 = static_cast<float>(scales[i1 >> 4]);

        half2* out = reinterpret_cast<half2*>(dst + static_cast<int64_t>(blk_id) * 256);
        half2 result;
        result.x = __float2half(d_w * sc0 * static_cast<float>(q0));
        result.y = __float2half(d_w * sc1 * static_cast<float>(q1));
        out[tid] = result;
    }
}

// ---------------------------------------------------------------------------
// Q8_0 GPU dequantization kernel
//
// Block format (34 bytes per 32 elements):
//   d[2]   : fp16 scale
//   qs[32] : int8 quantized values
// ---------------------------------------------------------------------------

__global__ void dequant_q8_0_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i   = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + static_cast<int64_t>(row * blocks_per_row + blk) * 34;
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    int8_t q = reinterpret_cast<const int8_t*>(block_ptr + 2)[i];

    float val = __half2float(d_val) * static_cast<float>(q);
    dst[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Q4_0 GPU dequantization kernel
//
// Block format (18 bytes per 32 elements):
//   d[2]   : fp16 scale
//   qs[16] : packed nibbles (2 x 4-bit values per byte, low nibble first)
// ---------------------------------------------------------------------------

__global__ void dequant_q4_0_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i   = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + static_cast<int64_t>(row * blocks_per_row + blk) * 18;
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    const uint8_t* qs = block_ptr + 2;

    int byte_idx = i / 2;
    uint8_t packed = qs[byte_idx];
    int nibble = (i % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);

    float val = __half2float(d_val) * static_cast<float>(nibble - 8);
    dst[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Q5_0 GPU dequantization kernel
//
// Block format (22 bytes per 32 elements):
//   d[2]   : fp16 scale
//   qh[4]  : high bits (bit 4 of each element, packed 8 per byte)
//   qs[16] : low 4-bit nibbles (2 per byte)
// ---------------------------------------------------------------------------

__global__ void dequant_q5_0_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i   = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + static_cast<int64_t>(row * blocks_per_row + blk) * 22;
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    const uint8_t* qh = block_ptr + 2;   // 4 bytes high bits
    const uint8_t* qs = block_ptr + 6;   // 16 bytes low nibbles

    int byte_idx = i / 2;
    uint8_t packed = qs[byte_idx];
    int low4 = (i % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    int high1 = (qh[i / 8] >> (i % 8)) & 1;
    int q5 = (high1 << 4) | low4;

    float val = __half2float(d_val) * static_cast<float>(q5 - 16);
    dst[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Q5_1 GPU dequantization kernel
//
// Block format (24 bytes per 32 elements):
//   d[2]   : fp16 scale
//   m[2]   : fp16 min
//   qh[4]  : high bits (bit 4 of each element)
//   qs[16] : low 4-bit nibbles
// ---------------------------------------------------------------------------

__global__ void dequant_q5_1_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i   = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + static_cast<int64_t>(row * blocks_per_row + blk) * 24;
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    half m_val = *reinterpret_cast<const half*>(block_ptr + 2);
    const uint8_t* qh = block_ptr + 4;   // 4 bytes high bits
    const uint8_t* qs = block_ptr + 8;   // 16 bytes low nibbles

    int byte_idx = i / 2;
    uint8_t packed = qs[byte_idx];
    int low4 = (i % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    int high1 = (qh[i / 8] >> (i % 8)) & 1;
    int q5 = (high1 << 4) | low4;

    float val = __half2float(d_val) * static_cast<float>(q5) + __half2float(m_val);
    dst[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Q4_K GPU dequantization kernel
//
// Super-block format (144 bytes per 256 elements):
//   d[2]           : fp16 super-block scale
//   dmin[2]        : fp16 super-block min
//   scales[12]     : packed sub-block scales and mins (6 bits each)
//   qs[128]        : 4-bit quantized values (2 per byte)
//
// 8 sub-blocks of 32 elements each. Each sub-block has a 6-bit scale
// and 6-bit min packed into the 12-byte scales array.
// ---------------------------------------------------------------------------

__global__ void dequant_q4k_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i   = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + static_cast<int64_t>(row * blocks_per_row + blk) * 144;
    float d    = __half2float(*reinterpret_cast<const half*>(block_ptr));
    float dmin = __half2float(*reinterpret_cast<const half*>(block_ptr + 2));
    const uint8_t* sc = block_ptr + 4;    // 12 bytes packed scales
    const uint8_t* qs = block_ptr + 16;   // 128 bytes quants

    int sub = i / 32;   // sub-block index 0..7 (= scale index)

    // Unpack 6-bit scale and min for this sub-block.
    // GGML packing (get_scale_min_k4):
    //   sub < 4: sc_val = scales[sub] & 63,       min_val = scales[sub+4] & 63
    //   sub >= 4: sc_val = (scales[sub+4] low4) | (scales[sub-4] top2 << 4)
    //             min_val = (scales[sub+4] high4) | (scales[sub] top2 << 4)
    uint8_t sc_val, min_val;
    if (sub < 4) {
        sc_val  = sc[sub] & 63;
        min_val = sc[sub + 4] & 63;
    } else {
        sc_val  = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        min_val = (sc[sub + 4] >> 4)   | ((sc[sub]     >> 6) << 4);
    }

    // Extract 4-bit quant value.
    // Q4_K layout: each 64-element chunk uses 32 bytes. First 32 elements
    // in low nibbles, next 32 in high nibbles of the SAME 32 bytes.
    int qs_byte = (i / 64) * 32 + (i % 32);
    int use_high = (i / 32) & 1;
    uint8_t packed = qs[qs_byte];
    int q4 = use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);

    float val = d * static_cast<float>(sc_val) * static_cast<float>(q4)
              - dmin * static_cast<float>(min_val);
    dst[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Q5_K GPU dequantization kernel
//
// Super-block format (176 bytes per 256 elements):
//   d[2]           : fp16 super-block scale
//   dmin[2]        : fp16 super-block min
//   scales[12]     : packed sub-block scales and mins (6 bits each, same as Q4_K)
//   qh[32]         : high bits (5th bit) for 256 elements
//   qs[128]        : 4-bit quantized values (low 4 bits, 2 per byte)
//
// 8 sub-blocks of 32 elements each. Dequant: val = d * sc * q5 - dmin * min
// where q5 is a 5-bit value: q5 = (q4 & 0xF) | ((qh_bit) << 4)
// ---------------------------------------------------------------------------

__global__ void dequant_q5k_kernel(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    int rows, int cols)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total) return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i   = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + static_cast<int64_t>(row * blocks_per_row + blk) * 176;
    float d    = __half2float(*reinterpret_cast<const half*>(block_ptr));
    float dmin = __half2float(*reinterpret_cast<const half*>(block_ptr + 2));
    const uint8_t* sc = block_ptr + 4;    // 12 bytes packed scales
    const uint8_t* qh = block_ptr + 16;   // 32 bytes high bits
    const uint8_t* qs = block_ptr + 48;   // 128 bytes quants

    int sub = i / 32;

    // Unpack 6-bit scale and min (same packing as Q4_K)
    uint8_t sc_val, min_val;
    if (sub < 4) {
        sc_val  = sc[sub] & 63;
        min_val = sc[sub + 4] & 63;
    } else {
        sc_val  = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        min_val = (sc[sub + 4] >> 4)   | ((sc[sub]     >> 6) << 4);
    }

    // Extract low 4-bit quant value (same layout as Q4_K)
    int qs_byte = (i / 64) * 32 + (i % 32);
    int use_high = (i / 32) & 1;
    uint8_t packed = qs[qs_byte];
    int q4 = use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);

    // Extract 5th bit from qh array
    // qh has 256 bits = 32 bytes, bit i corresponds to element i
    int qh_bit = (qh[i / 8] >> (i % 8)) & 1;
    int q5 = q4 | (qh_bit << 4);

    float val = d * static_cast<float>(sc_val) * static_cast<float>(q5)
              - dmin * static_cast<float>(min_val);
    dst[idx] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Q6_K → FP8 E4M3 dequantization kernel
//
// Same Q6_K block decoding as dequant_q6k_v2_kernel, but writes FP8 E4M3
// instead of FP16. Uses the existing dequant_q6k_element() helper.
// Q6_K dequanted values are typically |val| < 10 — well within FP8 E4M3
// range (max 448), so scale=1.0 with saturating conversion is safe.
//
// One CTA per Q6_K super-block (256 elements), 128 threads, 2 elements/thread.
// Writes uint16_t (2 packed FP8 bytes) per thread for coalesced 2-byte stores.
// ---------------------------------------------------------------------------

__global__ void dequant_q6k_to_fp8_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int total_blocks)
{
    int blk_id = blockIdx.x;
    if (blk_id >= total_blocks) return;

#ifdef __CUDA_FP8_TYPES_EXIST__
    const uint8_t* bp = src + static_cast<int64_t>(blk_id) * 210;
    uint16_t* out = reinterpret_cast<uint16_t*>(dst + static_cast<int64_t>(blk_id) * 256);
    float d_w = __half2float(*reinterpret_cast<const half*>(bp + 208));
    const int8_t* scales = reinterpret_cast<const int8_t*>(bp + 192);
    int i0 = threadIdx.x * 2;
    int i1 = i0 + 1;
    int q0 = dequant_q6k_element(bp, i0);
    int q1 = dequant_q6k_element(bp, i1);
    float v0 = d_w * static_cast<float>(scales[i0 >> 4]) * static_cast<float>(q0);
    float v1 = d_w * static_cast<float>(scales[i1 >> 4]) * static_cast<float>(q1);
    __nv_fp8_e4m3 f0 = __nv_fp8_e4m3(v0);
    __nv_fp8_e4m3 f1 = __nv_fp8_e4m3(v1);
    uint8_t b0, b1;
    memcpy(&b0, &f0, 1);
    memcpy(&b1, &f1, 1);
    out[threadIdx.x] = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
#else
    // FP8 types unavailable — zero fill (should not be reached on sm_90+)
    uint16_t* out = reinterpret_cast<uint16_t*>(dst + static_cast<int64_t>(blk_id) * 256);
    out[threadIdx.x] = 0;
#endif
}

// ---------------------------------------------------------------------------
// Dispatch: Q6_K → FP8 E4M3
// ---------------------------------------------------------------------------

void dequant_gpu_fp8(const void* src, void* dst, GGMLQuantType qtype,
                     int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;

    switch (qtype) {
        case GGMLQuantType::Q6_K: {
            int total_blocks = rows * (cols / 256);
            dequant_q6k_to_fp8_kernel<<<total_blocks, 128, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<uint8_t*>(dst),
                total_blocks);
            break;
        }
        default:
            IMP_LOG_ERROR("dequant_gpu_fp8: unsupported qtype %u", static_cast<unsigned>(qtype));
            break;
    }
}

// ---------------------------------------------------------------------------
// Dispatch: FP16
// ---------------------------------------------------------------------------

void dequant_gpu(const void* src, void* dst, GGMLQuantType qtype,
                 int rows, int cols, cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (total == 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    switch (qtype) {
        case GGMLQuantType::Q6_K: {
            int total_q6k_blocks = rows * (cols / 256);
            dequant_q6k_v2_kernel<<<total_q6k_blocks, 128, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                total_q6k_blocks);
            break;
        }

        case GGMLQuantType::Q8_0:
            dequant_q8_0_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                rows, cols);
            break;

        case GGMLQuantType::Q4_0:
            dequant_q4_0_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                rows, cols);
            break;

        case GGMLQuantType::Q5_0:
            dequant_q5_0_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                rows, cols);
            break;

        case GGMLQuantType::Q5_1:
            dequant_q5_1_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                rows, cols);
            break;

        case GGMLQuantType::Q4_K:
            dequant_q4k_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                rows, cols);
            break;

        case GGMLQuantType::Q5_K:
            dequant_q5k_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                rows, cols);
            break;

        default:
            IMP_LOG_ERROR("dequant_gpu: unsupported qtype %u", static_cast<unsigned>(qtype));
            break;
    }
}

} // namespace imp
