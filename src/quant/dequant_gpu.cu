#include "quant/dequant_gpu.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <utility>

namespace imp {

// qtype_row_bytes lives in src/core/qtype.cpp now.

bool dequant_gpu_supported(QType qtype) {
    switch (qtype) {
        case QType::Q6_K:
        case QType::Q8_0:
        case QType::Q4_0:
        case QType::Q4_1:
        case QType::Q5_0:
        case QType::Q5_1:
        case QType::Q2_K:
        case QType::Q3_K:
        case QType::Q4_K:
        case QType::Q5_K:
        case QType::Q8_K:
        case QType::IQ3_S:
        case QType::IQ4_NL:
        case QType::IQ4_XS:
            return true;
        default:
            return false;
    }
}

// Q6_K GPU dequant, 210 bytes per 256-element block: ql[128] lower 4 bits (GGML
// interleaved), qh[64] upper 2 bits (GGML interleaved), scales[16] int8 sub-block scales
// (1/16 elements), d[2] fp16 super-block scale. 256 values = 2 groups of 128, each 4
// sub-groups of 32:
//   q1 (0..31):   ql[l] low nibble,   qh[l] bits 0-1
//   q2 (32..63):  ql[l+32] low nibble, qh[l] bits 2-3
//   q3 (64..95):  ql[l] high nibble,  qh[l] bits 4-5
//   q4 (96..127): ql[l+32] high nibble, qh[l] bits 6-7
// Second group (128..255): ql+=64, qh+=32, sc+=8.

// Block-centric Q6_K dequant: one thread block per 256-element super-block, 128 threads
// each handling 2 consecutive elements with half2 writes. Avoids row/col integer division:
// consecutive Q6_K blocks map to consecutive blockIdx.x, so src=src+blockIdx.x*210,
// dst=dst+blockIdx.x*256 directly.

__device__ __forceinline__ int dequant_q6k_element(const uint8_t* __restrict__ bp, int i) {
    int group = i >> 7;
    int within = i & 127;
    int quad = within >> 5;
    int l = within & 31;

    int ql_idx = (group << 6) + ((quad & 1) << 5) + l;
    int qh_idx = (group << 5) + l;

    uint8_t ql_byte = bp[ql_idx];
    uint8_t low4 = (quad >= 2) ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
    uint8_t high2 = (bp[128 + qh_idx] >> (quad * 2)) & 0x3u;
    return static_cast<int>((high2 << 4) | low4) - 32;
}

__global__ void dequant_q6k_v2_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst,
                                      int total_blocks) {
    int blk_id = blockIdx.x;
    if (blk_id >= total_blocks)
        return;

    const uint8_t* bp = src + (static_cast<int64_t>(blk_id) * 210);
    half2* out = reinterpret_cast<half2*>(dst + (static_cast<int64_t>(blk_id) * 256));

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

// Q8_0 GPU dequant, block format (34 bytes per 32 elements): d[2] fp16 scale,
// qs[32] int8 quantized values.

__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                    int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 34);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    int8_t q = reinterpret_cast<const int8_t*>(block_ptr + 2)[i];

    float val = __half2float(d_val) * static_cast<float>(q);
    dst[idx] = __float2half(val);
}

// Q8_K GPU dequant, block format (292 bytes per 256 elements): d[4] float scale,
// qs[256] int8 quantized values, bsums[32] int16 block sums (unused for dequant, used for
// optimized dot product).

__global__ void dequant_q8k_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                   int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 292);
    float d_val;
    memcpy(&d_val, block_ptr, 4);
    int8_t q = reinterpret_cast<const int8_t*>(block_ptr + 4)[i];

    float val = d_val * static_cast<float>(q);
    dst[idx] = __float2half(val);
}

// Q4_0 GPU dequant, block format (18 bytes per 32 elements): d[2] fp16 scale,
// qs[16] packed nibbles (2x4-bit values per byte, low nibble first).

__global__ void dequant_q4_0_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                    int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 18);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    const uint8_t* qs = block_ptr + 2;

    // ggml layout: element e (0..15) is low nibble of qs[e]; element e+16 is
    // high nibble of qs[e]. NOT a linear (i/2, i%2) interleave.
    int byte_idx = i & 15;
    uint8_t packed = qs[byte_idx];
    int nibble = (i < 16) ? (packed & 0xF) : ((packed >> 4) & 0xF);

    float val = __half2float(d_val) * static_cast<float>(nibble - 8);
    dst[idx] = __float2half(val);
}

// Q4_1 GPU dequant, block format (20 bytes per 32 elements): d[2] fp16 scale, m[2] fp16 min,
// qs[16] 4-bit values (2 per byte). Dequant: val = d*nibble + m (unsigned nibble 0..15).

__global__ void dequant_q4_1_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                    int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 20);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    half m_val = *reinterpret_cast<const half*>(block_ptr + 2);
    const uint8_t* qs = block_ptr + 4;

    int byte_idx = i & 15;
    uint8_t packed = qs[byte_idx];
    int nibble = (i < 16) ? (packed & 0xF) : ((packed >> 4) & 0xF);

    float val = (__half2float(d_val) * static_cast<float>(nibble)) + __half2float(m_val);
    dst[idx] = __float2half(val);
}

// Q5_0 GPU dequant, block format (22 bytes per 32 elements): d[2] fp16 scale,
// qh[4] high bits (bit 4 of each element, packed 8/byte), qs[16] low 4-bit nibbles (2/byte).

__global__ void dequant_q5_0_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                    int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 22);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    const uint8_t* qh = block_ptr + 2;  // 4 bytes high bits
    const uint8_t* qs = block_ptr + 6;  // 16 bytes low nibbles

    int byte_idx = i & 15;
    uint8_t packed = qs[byte_idx];
    int low4 = (i < 16) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    // ggml packs high bits into a u32 where element i (0..31) has its high bit at bit i.
    // Ref dequantize_row_q5_0: xh_0 = ((qh>>j)<<4)&0x10 (bit j -> element j),
    // xh_1 = ((qh>>(j+12)))&0x10 (bit j+16 -> element j+16).
    int high1 = (qh[i / 8] >> (i % 8)) & 1;
    int q5 = (high1 << 4) | low4;

    float val = __half2float(d_val) * static_cast<float>(q5 - 16);
    dst[idx] = __float2half(val);
}

// Q5_1 GPU dequant, block format (24 bytes per 32 elements): d[2] fp16 scale, m[2] fp16 min,
// qh[4] high bits (bit 4 of each element), qs[16] low 4-bit nibbles.

__global__ void dequant_q5_1_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                    int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 24);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    half m_val = *reinterpret_cast<const half*>(block_ptr + 2);
    const uint8_t* qh = block_ptr + 4;  // 4 bytes high bits
    const uint8_t* qs = block_ptr + 8;  // 16 bytes low nibbles

    int byte_idx = i & 15;
    uint8_t packed = qs[byte_idx];
    int low4 = (i < 16) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    // Element i (0..31) uses bit i of the 32-bit qh (ref Q5_1 dequant).
    int high1 = (qh[i / 8] >> (i % 8)) & 1;
    int q5 = (high1 << 4) | low4;

    float val = (__half2float(d_val) * static_cast<float>(q5)) + __half2float(m_val);
    dst[idx] = __float2half(val);
}

// Q4_K GPU dequant, super-block format (144 bytes per 256 elements): d[2] fp16 super-block
// scale, dmin[2] fp16 super-block min, scales[12] packed 6-bit sub-block scales+mins,
// qs[128] 4-bit values (2/byte). 8 sub-blocks of 32 elements, each with its own 6-bit
// scale+min packed into the 12-byte scales array.

__global__ void dequant_q4k_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                   int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 144);
    float d = __half2float(*reinterpret_cast<const half*>(block_ptr));
    float dmin = __half2float(*reinterpret_cast<const half*>(block_ptr + 2));
    const uint8_t* sc = block_ptr + 4;   // 12 bytes packed scales
    const uint8_t* qs = block_ptr + 16;  // 128 bytes quants

    int sub = i / 32;  // sub-block index 0..7 (= scale index)

    // Unpack 6-bit scale/min per sub-block, GGML get_scale_min_k4 packing:
    //   sub<4:  sc=scales[sub]&63,             min=scales[sub+4]&63
    //   sub>=4: sc=(scales[sub+4] low4)|((scales[sub-4] top2)<<4)
    //           min=(scales[sub+4] high4)|((scales[sub] top2)<<4)
    uint8_t sc_val, min_val;
    if (sub < 4) {
        sc_val = sc[sub] & 63;
        min_val = sc[sub + 4] & 63;
    } else {
        sc_val = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        min_val = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }

    // Extract 4-bit quant value.
    // Q4_K layout: each 64-element chunk uses 32 bytes. First 32 elements
    // in low nibbles, next 32 in high nibbles of the SAME 32 bytes.
    int qs_byte = ((i / 64) * 32) + (i % 32);
    int use_high = (i / 32) & 1;
    uint8_t packed = qs[qs_byte];
    int q4 = use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);

    float val = (d * static_cast<float>(sc_val) * static_cast<float>(q4)) -
                (dmin * static_cast<float>(min_val));
    dst[idx] = __float2half(val);
}

// Q5_K GPU dequant, super-block format (176 bytes per 256 elements): d[2]/dmin[2] fp16
// super-block scale/min, scales[12] packed 6-bit (same as Q4_K), qh[32] high (5th) bit for
// 256 elements, qs[128] low 4-bit values (2/byte). 8 sub-blocks of 32.
// Dequant: val = d*sc*q5 - dmin*min, where q5 = (q4&0xF) | (qh_bit<<4).

__global__ void dequant_q5k_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                   int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 176);
    float d = __half2float(*reinterpret_cast<const half*>(block_ptr));
    float dmin = __half2float(*reinterpret_cast<const half*>(block_ptr + 2));
    const uint8_t* sc = block_ptr + 4;   // 12 bytes packed scales
    const uint8_t* qh = block_ptr + 16;  // 32 bytes high bits
    const uint8_t* qs = block_ptr + 48;  // 128 bytes quants

    int sub = i / 32;

    // Unpack 6-bit scale and min (same packing as Q4_K)
    uint8_t sc_val, min_val;
    if (sub < 4) {
        sc_val = sc[sub] & 63;
        min_val = sc[sub + 4] & 63;
    } else {
        sc_val = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
        min_val = (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    }

    // Extract low 4-bit quant value (same layout as Q4_K)
    int qs_byte = ((i / 64) * 32) + (i % 32);
    int use_high = (i / 32) & 1;
    uint8_t packed = qs[qs_byte];
    int q4 = use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);

    // Extracts the 5th bit from qh (ggml dequantize_row_q5_K layout): 32 bytes, byte[l] holds
    // the high bits for element l across all 4 chunks (j=0,64,128,192) at bit positions 0..7.
    // For chunk c (0..3), half h (0..1), l = index in chunk-half: byte_idx=l, bit_pos=c*2+h,
    // element index in block = c*64+h*32+l.
    int c = i >> 6;         // chunk 0..3
    int h2 = (i >> 5) & 1;  // half 0/1
    int l = i & 31;
    int bit_pos = (c << 1) | h2;
    int qh_bit = (qh[l] >> bit_pos) & 1;
    int q5 = q4 | (qh_bit << 4);

    float val = (d * static_cast<float>(sc_val) * static_cast<float>(q5)) -
                (dmin * static_cast<float>(min_val));
    dst[idx] = __float2half(val);
}

// Block-centric Q4_K / Q5_K: one warp per super-block, 8 per 256-thread CTA, lane = 8 consecutive
// elements of one 32-element sub-block (uint2 qs / qh loads, one uint4 store). Same products as the
// element kernels; needs 16-B aligned src and dst (the dispatcher falls back otherwise).
constexpr int kKqBlocksPerCta = 8;

template <bool kQ5>
__global__ void dequant_q45k_v2_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst,
                                       int n_blocks) {
    constexpr int kBytes = kQ5 ? 176 : 144;
    constexpr int kQs = kQ5 ? 48 : 16;
    const int64_t b = (static_cast<int64_t>(blockIdx.x) * kKqBlocksPerCta) + (threadIdx.x >> 5);
    if (b >= n_blocks)
        return;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const uint8_t* bp = src + (b * kBytes);
    const int sub = lane >> 2;
    const int in0 = (lane & 3) * 8;
    const uint8_t* sc = bp + 4;
    int s = 0, m = 0;
    if (sub < 4) {
        s = __ldg(sc + sub) & 63;
        m = __ldg(sc + sub + 4) & 63;
    } else {
        s = (__ldg(sc + sub + 4) & 0xF) | ((__ldg(sc + sub - 4) >> 6) << 4);
        m = (__ldg(sc + sub + 4) >> 4) | ((__ldg(sc + sub) >> 6) << 4);
    }
    const half2 dm = __ldg(reinterpret_cast<const half2*>(bp));
    const float dl = __low2float(dm) * static_cast<float>(s);
    const float ml = __high2float(dm) * static_cast<float>(m);
    const int qs_off = kQs + ((sub >> 1) * 32) + in0;
    const uint2 q = __ldg(reinterpret_cast<const uint2*>(bp + qs_off));
    uint2 h = make_uint2(0u, 0u);
    if constexpr (kQ5)
        h = __ldg(reinterpret_cast<const uint2*>(bp + 16 + in0));
    const int nshift = (sub & 1) * 4;
    alignas(16) half out[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const uint32_t qw = k < 4 ? q.x : q.y;
        int v = static_cast<int>((qw >> ((8 * (k & 3)) + nshift)) & 0xFu);
        if constexpr (kQ5) {
            const uint32_t hw = k < 4 ? h.x : h.y;
            v |= static_cast<int>((hw >> ((8 * (k & 3)) + sub)) & 1u) << 4;
        }
        out[k] = __float2half((dl * static_cast<float>(v)) - ml);
    }
    const int e0 = 8 * lane;
    *reinterpret_cast<uint4*>(dst + (b * 256) + e0) = *reinterpret_cast<const uint4*>(out);
}

// Q2_K GPU dequant, super-block format (84 bytes per 256 elements): scales[16] 4-bit packed
// (low nibble=scale, high nibble=min, 1 byte/16 elements), qs[64] 2-bit packed (4/byte),
// d[2]/dmin[2] fp16 super-block scale/min. 16 sub-blocks of 16 elements; within each
// 128-element half, shift cycles 0,2,4,6 over groups of 32 quant bytes:
// q2_val=(qs[byte]>>shift)&3, val = d*sc_low4*q2_val - dmin*sc_high4.

__global__ void dequant_q2k_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                   int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 84);
    // Q2_K layout: scales[16], qs[64], d(fp16), dmin(fp16)
    const uint8_t* scales = block_ptr;   // 16 bytes
    const uint8_t* qs = block_ptr + 16;  // 64 bytes
    float d = __half2float(*reinterpret_cast<const half*>(block_ptr + 80));
    float dmin = __half2float(*reinterpret_cast<const half*>(block_ptr + 82));

    // Which 128-element half (0 or 1)
    int half_idx = i / 128;
    int within = i % 128;
    // Within 128 elements: 4 groups of 32, shift cycles 0,2,4,6
    int group = within / 32;   // 0..3
    int in_grp = within % 32;  // 0..31
    int shift = group * 2;

    // qs index: 32 bytes per 128-element half, element maps to in_grp + (in_grp>=16 ? 16 : 0)
    // Actually: first 16 elements use qs[l], next 16 use qs[l+16]
    int qs_idx = (half_idx * 32) + (in_grp < 16 ? in_grp : in_grp);
    uint8_t qs_byte = qs[qs_idx];
    int q2 = (qs_byte >> shift) & 3;

    // Scale index: 16 sub-blocks of 16 elements
    int sc_idx = i / 16;  // 0..15
    uint8_t sc_byte = scales[sc_idx];
    float dl = d * static_cast<float>(sc_byte & 0xF);
    float ml = dmin * static_cast<float>(sc_byte >> 4);

    float val = (dl * static_cast<float>(q2)) - ml;
    dst[idx] = __float2half(val);
}

// Q3_K GPU dequant, super-block format (110 bytes per 256 elements): hmask[32] high bit
// (bit 2) per element, qs[64] 2-bit packed (low 2 bits, 4/byte), scales[12] packed 6-bit
// (same packing as Q4_K/Q5_K), d[2] fp16 super-block scale. 16 sub-blocks of 16 elements;
// q3 = qs_2bit + (hmask_bit?0:-4) giving range [-4..3]; val = d*(unpacked_6bit_scale-32)*q3.

__global__ void dequant_q3k_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                   int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 110);
    // Q3_K layout: hmask[32], qs[64], scales[12], d(fp16)
    const uint8_t* hmask = block_ptr;        // 32 bytes
    const uint8_t* qs = block_ptr + 32;      // 64 bytes
    const uint8_t* sc_raw = block_ptr + 96;  // 12 bytes
    float d_all = __half2float(*reinterpret_cast<const half*>(block_ptr + 108));

    // Extract 2-bit quant: same layout as Q2_K
    int half_idx = i / 128;
    int within = i % 128;
    int group = within / 32;   // 0..3
    int in_grp = within % 32;  // 0..31
    int shift = group * 2;

    int qs_idx = (half_idx * 32) + in_grp;
    int q2 = (qs[qs_idx] >> shift) & 3;

    // hmask's 256 high bits are NOT stored in output order: ggml walks one bitplane per
    // 32-element sub-block, so element i reads bit i/32 of byte i%32, not bit i%8 of byte i/8
    // (the two agree only for i<8, why this survived every by-eye check, AUDIT_ARCH_2026 D-5).
    int hm_bit = (hmask[i % 32] >> (i / 32)) & 1;
    int q3 = q2 - (hm_bit ? 0 : 4);

    // Q3_K 6-bit scale unpack (16 sub-blocks of 16 elements), same 12-byte packing as the
    // 16-scale format: scales[0..3]/[4..7] hold low 4 bits, scales[8..11] hold interleaved
    // upper 2 bits.
    int sub = i / 16;  // 0..15
    int8_t sc_val;
    {
        // Unpack the 12-byte packed scale array following GGML convention
        // First 8 scales use aux[0] and aux[1] low nibbles + tmp bits 0-3
        // Last 8 scales use aux[0] and aux[1] high nibbles + tmp bits 4-7
        uint32_t aux0, aux1, aux2;
        memcpy(&aux0, sc_raw, 4);
        memcpy(&aux1, sc_raw + 4, 4);
        memcpy(&aux2, sc_raw + 8, 4);

        constexpr uint32_t kmask2 = 0x0f0f0f0fu;
        constexpr uint32_t kmask1 = 0x03030303u;

        uint32_t s[4];
        s[0] = (aux0 & kmask2) | (((aux2 >> 0) & kmask1) << 4);
        s[1] = (aux1 & kmask2) | (((aux2 >> 2) & kmask1) << 4);
        s[2] = ((aux0 >> 4) & kmask2) | (((aux2 >> 4) & kmask1) << 4);
        s[3] = ((aux1 >> 4) & kmask2) | (((aux2 >> 6) & kmask1) << 4);

        const int8_t* unpacked = reinterpret_cast<const int8_t*>(s);
        sc_val = unpacked[sub];
    }

    float val = d_all * static_cast<float>(sc_val - 32) * static_cast<float>(q3);
    dst[idx] = __float2half(val);
}

// Block-centric Q3_K: 8 super-blocks per 256-thread CTA staged to smem with u16 loads, one warp per
// super-block, 8 consecutive elements per lane (one sub-block scale, one hmask bit, one uint4 store).
// Same products as dequant_q3k_kernel; needs a 16-B aligned dst (the dispatcher falls back otherwise).
__global__ void dequant_q3k_v2_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int n_blocks) {
    __shared__ uint16_t stage[kKqBlocksPerCta * 55];
    const int64_t blk0 = static_cast<int64_t>(blockIdx.x) * kKqBlocksPerCta;
    const int n_cta = min(kKqBlocksPerCta, static_cast<int>(n_blocks - blk0));
    const auto* s16 = reinterpret_cast<const uint16_t*>(src + (blk0 * 110));
    for (int i = static_cast<int>(threadIdx.x); i < n_cta * 55; i += static_cast<int>(blockDim.x))
        stage[i] = __ldg(s16 + i);
    __syncthreads();

    const int w = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (w >= n_cta)
        return;
    const int stage_off = w * 110;
    const auto* bp = reinterpret_cast<const uint8_t*>(stage) + stage_off;
    const int in_grp0 = (lane & 3) * 8;  // element e0 = 8 * lane: half lane>>4, group (lane>>2)&3
    const int shift = 2 * ((lane >> 2) & 3);
    const int hbit = lane >> 2;  // hmask bit plane e0 / 32
    const int sub = lane >> 1;   // 16-element sub-block
    const int j = sub & 3;
    const int lo = (bp[96 + (((sub >> 2) & 1) * 4) + j] >> (4 * (sub >> 3))) & 0xF;
    const int hi = (bp[104 + j] >> (2 * (sub >> 2))) & 3;
    const float dl = __half2float(*reinterpret_cast<const half*>(bp + 108)) *
                     static_cast<float>((lo | (hi << 4)) - 32);
    const int qs_off = 32 + ((lane >> 4) * 32) + in_grp0;
    const uint8_t* qs = bp + qs_off;
    const uint8_t* hm = bp + in_grp0;
    alignas(16) half out[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const int q3 = ((qs[k] >> shift) & 3) - (((hm[k] >> hbit) & 1) ? 0 : 4);
        out[k] = __float2half(dl * static_cast<float>(q3));
    }
    const int e0 = 8 * lane;
    *reinterpret_cast<uint4*>(dst + ((blk0 + w) * 256) + e0) = *reinterpret_cast<const uint4*>(out);
}

// IQ4_NL GPU dequant, block format (18 bytes per 32 elements): d[2] fp16 scale, qs[16]
// packed 4-bit codebook indices (2/byte, into kvalues_iq4nl). Non-linear 4-bit: index
// selects from a fixed signed codebook. Layout matches Q4_0 (element e = low nibble of
// qs[e], e+16 = high nibble). val = d * codebook[nibble].

__device__ __constant__ int8_t kvalues_iq4nl_gpu[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                                        1,    13,   25,  38,  53,  69,  89,  113};

__global__ void dequant_iq4_nl_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                      int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 32;
    int i = col % 32;
    int blocks_per_row = cols / 32;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 18);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    const uint8_t* qs = block_ptr + 2;

    int byte_idx = i & 15;
    uint8_t packed = qs[byte_idx];
    int nibble = (i < 16) ? (packed & 0xF) : ((packed >> 4) & 0xF);

    float val = __half2float(d_val) * static_cast<float>(kvalues_iq4nl_gpu[nibble]);
    dst[idx] = __float2half(val);
}

// IQ4_XS GPU dequant, super-block format (136 bytes per 256 elements): d[2] fp16
// super-block scale, scales_h[2]/scales_l[4] packed 6-bit sub-block scales (bias 32),
// qs[128] packed 4-bit codebook indices (2/byte, same codebook as IQ4_NL). 8 sub-blocks
// of 32, nibble layout matches IQ4_NL. ls = (scales_l nibble)|((scales_h>>2*sub)&3)<<4;
// val = d*(ls-32)*codebook[nibble].

__global__ void dequant_iq4_xs_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                      int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 136);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    uint16_t scales_h;
    memcpy(&scales_h, block_ptr + 2, 2);
    const uint8_t* scales_l = block_ptr + 4;  // 4 bytes, 8 nibbles
    const uint8_t* qs = block_ptr + 8;        // 128 bytes

    int sub = i / 32;     // sub-block 0..7
    int within = i & 31;  // 0..31 within the sub-block

    int ls_low = (scales_l[sub / 2] >> (4 * (sub & 1))) & 0xF;
    int ls_high = (scales_h >> (2 * sub)) & 3;
    int ls = ls_low | (ls_high << 4);

    int byte_idx = (sub * 16) + (within & 15);
    uint8_t packed = qs[byte_idx];
    int nibble = (within < 16) ? (packed & 0xF) : ((packed >> 4) & 0xF);

    float val = __half2float(d_val) * static_cast<float>(ls - 32) *
                static_cast<float>(kvalues_iq4nl_gpu[nibble]);
    dst[idx] = __float2half(val);
}

// IQ3_S GPU dequant, 110 bytes per 256 elements: d[2] fp16, qs[64] grid index low bits,
// qh[8] grid index bit 8, signs[32] one bit per element, scales[4] two 4-bit scales per byte.
// Element i: ib = i/32, l = (i%32)/8, h = (i%8)/4; grid = kIq3sGrid[qs[8ib+2l+h] | qh bit << 8];
// val = d*(1+2*scale nibble ib)*grid byte (i%4)*(sign bit (i%8) of signs[4ib+l] ? -1 : 1).

__device__ __constant__ uint32_t kIq3sGridGpu[512] = {
#include "quant/iq3s_grid.inc"
};

__global__ void dequant_iq3_s_kernel(const uint8_t* __restrict__ src, half* __restrict__ dst, int rows,
                                     int cols) {
    int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (idx >= total)
        return;

    int row = static_cast<int>(idx / cols);
    int col = static_cast<int>(idx % cols);
    int blk = col / 256;
    int i = col % 256;
    int blocks_per_row = cols / 256;

    const uint8_t* block_ptr = src + (static_cast<int64_t>((row * blocks_per_row) + blk) * 110);
    half d_val = *reinterpret_cast<const half*>(block_ptr);
    const uint8_t* qs = block_ptr + 2;
    const uint8_t* qh = block_ptr + 66;
    const uint8_t* signs = block_ptr + 74;
    const uint8_t* scales = block_ptr + 106;

    int ib = i / 32;
    int l = (i & 31) / 8;
    int h = (i & 7) / 4;
    int grid_idx = qs[(ib * 8) + (2 * l) + h] | (((qh[ib] >> ((2 * l) + h)) & 1) << 8);
    int mag = (kIq3sGridGpu[grid_idx] >> (8 * (i & 3))) & 0xFF;
    int sc = (scales[ib / 2] >> (4 * (ib & 1))) & 0xF;
    float sign = ((signs[(ib * 4) + l] >> (i & 7)) & 1) ? -1.0f : 1.0f;

    float val = __half2float(d_val) * static_cast<float>(1 + (2 * sc)) * static_cast<float>(mag) * sign;
    dst[idx] = __float2half(val);
}

// Q6_K -> FP8 E4M3 dequant: same block decoding as dequant_q6k_v2_kernel, writes FP8 E4M3
// instead of FP16 via dequant_q6k_element(). Q6_K values are |val|<10, well within E4M3's
// 448 max, so scale=1.0 with saturating conversion is safe. One CTA per 256-element
// super-block, 128 threads, 2 elements/thread, packed uint16_t (2 FP8 bytes) per thread.

__global__ void dequant_q6k_to_fp8_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                          int total_blocks) {
    int blk_id = blockIdx.x;
    if (blk_id >= total_blocks)
        return;

    const uint8_t* bp = src + (static_cast<int64_t>(blk_id) * 210);
    uint16_t* out = reinterpret_cast<uint16_t*>(dst + (static_cast<int64_t>(blk_id) * 256));
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
}

// ---------------------------------------------------------------------------
// Dispatch: Q6_K → FP8 E4M3
// ---------------------------------------------------------------------------

void dequant_gpu_fp8(const void* src, void* dst, QType qtype, int rows, int cols, cudaStream_t stream) {
    if (rows == 0 || cols == 0)
        return;

    switch (qtype) {
        case QType::Q6_K: {
            int total_blocks = rows * (cols / 256);
            dequant_q6k_to_fp8_kernel<<<total_blocks, 128, 0, stream>>>(static_cast<const uint8_t*>(src),
                                                                        static_cast<uint8_t*>(dst),
                                                                        total_blocks);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        }
        default:
            IMP_LOG_ERROR("dequant_gpu_fp8: unsupported qtype %u", std::to_underlying(qtype));
            break;
    }
}

// ---------------------------------------------------------------------------
// Dispatch: FP16
// ---------------------------------------------------------------------------

// Macro for scalar dequant kernels that share (src_u8, dst_fp16, rows, cols) signature.
// KERNEL names a __global__ in <<<>>>, where a parenthesized name does not parse (#2210)
// NOLINTBEGIN(bugprone-macro-parentheses)
#define DEQUANT_CASE(QTYPE, KERNEL)                                                                       \
    case QType::QTYPE:                                                                                    \
        KERNEL<<<blocks, threads, 0, stream>>>(static_cast<const uint8_t*>(src), static_cast<half*>(dst), \
                                               rows, cols);                                               \
        IMP_CUDA_CHECK_LAUNCH();                                                                          \
        break;
// NOLINTEND(bugprone-macro-parentheses)

void dequant_gpu(const void* src, void* dst, QType qtype, int rows, int cols, cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (total == 0)
        return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    switch (qtype) {
        case QType::Q6_K: {
            // Q6_K uses block-centric v2 kernel with different grid/thread config.
            int total_q6k_blocks = rows * (cols / 256);
            dequant_q6k_v2_kernel<<<total_q6k_blocks, 128, 0, stream>>>(static_cast<const uint8_t*>(src),
                                                                        static_cast<half*>(dst),
                                                                        total_q6k_blocks);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        }
        case QType::Q3_K:
        case QType::Q4_K:
        case QType::Q5_K: {
            // Warp-per-super-block kernels: Q3_K needs a 16-B dst, Q4_K / Q5_K also a 16-B src.
            const auto* s8 = static_cast<const uint8_t*>(src);
            auto* d16 = static_cast<half*>(dst);
            const int n_blocks = rows * (cols / 256);
            const int grid = (n_blocks + kKqBlocksPerCta - 1) / kKqBlocksPerCta;
            const uintptr_t mis = reinterpret_cast<uintptr_t>(dst) |
                                  (qtype == QType::Q3_K ? 0u : reinterpret_cast<uintptr_t>(src));
            if ((mis & 15u) != 0 && qtype == QType::Q3_K) {
                dequant_q3k_kernel<<<blocks, threads, 0, stream>>>(s8, d16, rows, cols);
                IMP_CUDA_CHECK_LAUNCH();
            } else if ((mis & 15u) != 0 && qtype == QType::Q4_K) {
                dequant_q4k_kernel<<<blocks, threads, 0, stream>>>(s8, d16, rows, cols);
                IMP_CUDA_CHECK_LAUNCH();
            } else if ((mis & 15u) != 0) {
                dequant_q5k_kernel<<<blocks, threads, 0, stream>>>(s8, d16, rows, cols);
                IMP_CUDA_CHECK_LAUNCH();
            } else if (qtype == QType::Q3_K) {
                dequant_q3k_v2_kernel<<<grid, 256, 0, stream>>>(s8, d16, n_blocks);
                IMP_CUDA_CHECK_LAUNCH();
            } else if (qtype == QType::Q4_K) {
                dequant_q45k_v2_kernel<false><<<grid, 256, 0, stream>>>(s8, d16, n_blocks);
                IMP_CUDA_CHECK_LAUNCH();
            } else {
                dequant_q45k_v2_kernel<true><<<grid, 256, 0, stream>>>(s8, d16, n_blocks);
                IMP_CUDA_CHECK_LAUNCH();
            }
            break;
        }

            DEQUANT_CASE(Q8_0, dequant_q8_0_kernel)
            DEQUANT_CASE(Q4_0, dequant_q4_0_kernel)
            DEQUANT_CASE(Q4_1, dequant_q4_1_kernel)
            DEQUANT_CASE(Q5_0, dequant_q5_0_kernel)
            DEQUANT_CASE(Q5_1, dequant_q5_1_kernel)
            DEQUANT_CASE(Q2_K, dequant_q2k_kernel)
            DEQUANT_CASE(Q8_K, dequant_q8k_kernel)
            DEQUANT_CASE(IQ3_S, dequant_iq3_s_kernel)
            DEQUANT_CASE(IQ4_NL, dequant_iq4_nl_kernel)
            DEQUANT_CASE(IQ4_XS, dequant_iq4_xs_kernel)

        default:
            IMP_LOG_ERROR("dequant_gpu: unsupported qtype %u", std::to_underlying(qtype));
            break;
    }
}

#undef DEQUANT_CASE

}  // namespace imp
