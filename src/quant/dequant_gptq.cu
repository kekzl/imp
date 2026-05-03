#include "quant/dequant_gptq.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// ---------------------------------------------------------------------------
// GPTQ 4-bit dequantization kernel
//
// GPTQ packs 8 x 4-bit values per INT32 in qweight and qzeros.
// Layout:
//   qweight[K/8, N]  — K/8 rows, N columns (column = output neuron)
//   qzeros[num_groups/8, N] — packed zero points per group
//   scales[num_groups, N]   — FP16 per-group scales
//   g_idx[K] (optional)     — maps input dim k to its group index (desc_act)
//
// For each output element [row, col] in the [N, K] result:
//   group = g_idx ? g_idx[col] : col / group_size
//   qval  = extract 4 bits from qweight[col/8, row] at position (col%8)*4
//   zero  = extract 4 bits from qzeros[group/8, row] at position (group%8)*4
//   scale = scales[group, row]
//   weight = scale * (qval - zero)
// ---------------------------------------------------------------------------

__global__ void dequant_gptq4_kernel(half* __restrict__ out,               // [N, K]
                                     const int32_t* __restrict__ qweight,  // [K/8, N]
                                     const int32_t* __restrict__ qzeros,   // [num_groups/8, N]
                                     const half* __restrict__ scales,      // [num_groups, N]
                                     const int32_t* __restrict__ g_idx,    // [K] or nullptr
                                     int N, int K, int group_size) {
    // Each thread handles one element in the [N, K] output
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension
    int col = blockIdx.y * blockDim.y + threadIdx.y;  // K dimension

    if (row >= N || col >= K)
        return;

    // Determine group for this column
    int group = g_idx ? g_idx[col] : col / group_size;

    // Extract 4-bit quantized weight value from packed INT32
    // qweight layout: [K/8, N] row-major, so qweight[pack_row * N + row]
    int pack_row = col / 8;
    int bit_offset = (col % 8) * 4;
    int32_t packed_w = qweight[pack_row * N + row];
    int qval = (packed_w >> bit_offset) & 0xF;

    // Extract 4-bit zero point from packed INT32
    // qzeros layout: [num_groups/8, N], groups packed 8 per INT32
    int zero_pack_row = group / 8;
    int zero_bit_offset = (group % 8) * 4;
    int32_t packed_z = qzeros[zero_pack_row * N + row];
    int zp = (packed_z >> zero_bit_offset) & 0xF;

    // Scale and dequantize
    float s = __half2float(scales[group * N + row]);
    float w = s * (static_cast<float>(qval) - static_cast<float>(zp));

    out[row * K + col] = __float2half(w);
}

void dequant_gptq4(half* out, const int32_t* qweight, const int32_t* qzeros, const half* scales,
                   const int32_t* g_idx, int N, int K, int group_size, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (K + block.y - 1) / block.y);
    dequant_gptq4_kernel<<<grid, block, 0, stream>>>(out, qweight, qzeros, scales, g_idx, N, K, group_size);
}

}  // namespace imp
