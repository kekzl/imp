#include "quant/dequant_gpu.h"
#include "core/logging.h"
#include <cuda_fp16.h>
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
// Dispatch
// ---------------------------------------------------------------------------

void dequant_gpu(const void* src, void* dst, GGMLQuantType qtype,
                 int rows, int cols, cudaStream_t stream)
{
    int64_t total = static_cast<int64_t>(rows) * cols;
    if (total == 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    switch (qtype) {
        case GGMLQuantType::Q6_K:
            dequant_q6k_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const uint8_t*>(src),
                static_cast<half*>(dst),
                rows, cols);
            break;

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

        default:
            IMP_LOG_ERROR("dequant_gpu: unsupported qtype %u", static_cast<unsigned>(qtype));
            break;
    }
}

} // namespace imp
