#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// =========================================================================
// NVFP4 quantization — HW-consumption layout
// =========================================================================
//
// Adapted from thu-ml/SageAttention3 (Apache-2.0), specifically
// scaled_fp4_quant_kernel in sageattention3_blackwell/.../fp4_quantization_4d.cu.
// This is the layout required by
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
// for the `sfa`/`sfb` scale operand. Unlike nvfp4_quant_ref (linear), here
// the scale bytes are stored in a specific interleaved pattern that
// matches the hardware MMA fetch.
//
// Input shape:  [batch, n_heads, n_tokens, head_dim] FP16 or BF16
// Output NVFP4: [batch, n_heads, n_tokens, head_dim/2] packed uint8
// Output SF:    FP8 UE4M3 scale per 16-element group, in hardware layout.
//               Allocated size = (n_tokens_rounded_64) * 128 bytes per
//               (batch, head) where n_tokens_rounded_64 = ceil(n_tokens/64)*64.
//
// Constraints:
//   - head_dim must be 64 or 128
//   - head_dim must be divisible by 16 (always true for 64, 128)
// =========================================================================

// Host-callable entry: quantize a 4D FP16/BF16 tensor to NVFP4 + HW-layout
// FP8 UE4M3 scales. Returns false on invalid parameters. Lays out the
// scale buffer in the format consumed by the block-scaled MMA.
bool nvfp4_quant_hw_fp16(
    const half* d_input,
    uint8_t* d_nvfp4,
    uint8_t* d_sf,
    int batch_size,
    int n_heads,
    int n_tokens,
    int head_dim,
    int stride_bz_input,     // elements
    int stride_h_input,
    int stride_seq_input,
    int stride_bz_output,    // bytes (nvfp4 is 1/2 byte per elem)
    int stride_h_output,
    int stride_seq_output,
    int stride_bz_output_sf,
    int stride_h_output_sf,
    int stride_seq_output_sf,
    cudaStream_t stream);

// Dequantize back to FP16 using the same HW layout — inverse of the
// quant kernel. Used for round-trip validation. Production attention
// kernels do not need this because the MMA dequants implicitly.
bool nvfp4_dequant_hw_fp16(
    const uint8_t* d_nvfp4,
    const uint8_t* d_sf,
    half* d_output,
    int batch_size,
    int n_heads,
    int n_tokens,
    int head_dim,
    int stride_bz_input,
    int stride_h_input,
    int stride_seq_input,
    int stride_bz_output,
    int stride_h_output,
    int stride_seq_output,
    int stride_bz_input_sf,
    int stride_h_input_sf,
    int stride_seq_input_sf,
    cudaStream_t stream);

} // namespace imp
