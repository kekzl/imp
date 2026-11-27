#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// NVFP4 quantization, HW-consumption layout. Adapted from thu-ml/SageAttention
// (Apache-2.0), scaled_fp4_quant_kernel in
// sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu
// (repo SageAttention; method/subtree SageAttention3; see THIRD_PARTY_LICENSES.md).
// Layout required by mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64's
// sfa/sfb scale operand (interleaved, unlike nvfp4_quant_ref's linear layout).
// Input: [batch,n_heads,n_tokens,head_dim] FP16/BF16. Output NVFP4:
// [batch,n_heads,n_tokens,head_dim/2] packed uint8. Output SF: FP8 UE4M3 per 16-elem group,
// HW layout, size = ceil(n_tokens/64)*64 * 128 bytes per (batch,head).
// Constraints: head_dim in {64,128} (always divisible by 16).

// Host-callable entry: quantize a 4D FP16/BF16 tensor to NVFP4 + HW-layout
// FP8 UE4M3 scales. Returns false on invalid parameters. Lays out the
// scale buffer in the format consumed by the block-scaled MMA.
// Arguments in the old parameter order (audit D-11); all required. Input strides in elements,
// output strides in bytes (nvfp4 is 1/2 byte per elem).
struct Nvfp4QuantHwArgs {
    const half* d_input = nullptr;
    uint8_t* d_nvfp4 = nullptr;
    uint8_t* d_sf = nullptr;
    int batch_size = 0;
    int n_heads = 0;
    int n_tokens = 0;
    int head_dim = 0;
    int64_t stride_bz_input = 0;
    int64_t stride_h_input = 0;
    int64_t stride_seq_input = 0;
    int64_t stride_bz_output = 0;
    int64_t stride_h_output = 0;
    int64_t stride_seq_output = 0;
    int64_t stride_bz_output_sf = 0;
    int64_t stride_h_output_sf = 0;
    int64_t stride_seq_output_sf = 0;
    cudaStream_t stream = nullptr;
};
[[nodiscard]] bool nvfp4_quant_hw_fp16(const Nvfp4QuantHwArgs& a);

// Dequantize back to FP16 using the same HW layout — inverse of the
// quant kernel. Used for round-trip validation. Production attention
// kernels do not need this because the MMA dequants implicitly.
// Arguments in the old parameter order (audit D-11); all required.
struct Nvfp4DequantHwArgs {
    const uint8_t* d_nvfp4 = nullptr;
    const uint8_t* d_sf = nullptr;
    half* d_output = nullptr;
    int batch_size = 0;
    int n_heads = 0;
    int n_tokens = 0;
    int head_dim = 0;
    int64_t stride_bz_input = 0;
    int64_t stride_h_input = 0;
    int64_t stride_seq_input = 0;
    int64_t stride_bz_output = 0;
    int64_t stride_h_output = 0;
    int64_t stride_seq_output = 0;
    int64_t stride_bz_input_sf = 0;
    int64_t stride_h_input_sf = 0;
    int64_t stride_seq_input_sf = 0;
    cudaStream_t stream = nullptr;
};
[[nodiscard]] bool nvfp4_dequant_hw_fp16(const Nvfp4DequantHwArgs& a);

}  // namespace imp
