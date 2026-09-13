#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// Reference NVFP4 quantization, linear layout (no HW-MMA scale interleaving). Each thread
// processes 16 consecutive FP16 elements: 1 FP8 UE4M3 scale + 16 NVFP4 nibbles (8 bytes).
// nvfp4[i/2]: low nibble=elem i (even), high nibble=elem i+1 (odd).
// sf[i/16]: FP8 UE4M3 scale for group [16*g..16*g+15].
// NOT for MMA consumption (needs HW scale interleaving, see sageattention3_blackwell);
// round-trip correctness validation only.
void nvfp4_quant_linear_fp16(const half* d_input, uint8_t* d_nvfp4, uint8_t* d_sf, int n_elements,
                             cudaStream_t stream);

// Inverse: NVFP4 + FP8 UE4M3 scale → FP16. Linear layout reverse.
void nvfp4_dequant_linear_fp16(const uint8_t* d_nvfp4, const uint8_t* d_sf, half* d_output, int n_elements,
                               cudaStream_t stream);

}  // namespace imp
