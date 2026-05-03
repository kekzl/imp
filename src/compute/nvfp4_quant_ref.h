#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// Reference-style NVFP4 quantization, linear layout (no hardware-MMA
// scale interleaving). Each thread processes 16 consecutive FP16
// elements, computes one FP8 UE4M3 scale factor, and writes 16 NVFP4
// nibbles (8 bytes) + 1 scale byte.
//
// Storage layout:
//   nvfp4[i / 2]  — low nibble = element i (even), high nibble = element i+1 (odd)
//   sf[i / 16]    — FP8 UE4M3 scale for the 16-element group [16*g .. 16*g+15]
//
// Input shape: [n_elements] FP16/BF16, contiguous
// Output nvfp4: [n_elements / 2] uint8 (packed nibbles)
// Output sf:    [n_elements / 16] uint8 (FP8 UE4M3)
//
// NOT used for MMA consumption — the hardware MMA requires a specific
// scale interleaving (see sageattention3_blackwell). This is only for
// round-trip correctness validation of the quant math.
void nvfp4_quant_linear_fp16(const half* d_input, uint8_t* d_nvfp4, uint8_t* d_sf, int n_elements,
                             cudaStream_t stream);

// Inverse: NVFP4 + FP8 UE4M3 scale → FP16. Linear layout reverse.
void nvfp4_dequant_linear_fp16(const uint8_t* d_nvfp4, const uint8_t* d_sf, half* d_output, int n_elements,
                               cudaStream_t stream);

}  // namespace imp
