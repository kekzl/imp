#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// End-to-end Q.K^T harness for mxf4nvf4.block_scale MMA. Q[16,64] FP16, K[8,64] FP16
// row-major device ptrs; D[16,8] FP32 device output.
// Quantizes Q/K to E2M1 on the fly, issues one
// mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 with scale=1.0
// (FP8 UE4M3 byte 0x38), writes 16x8 output per CUTLASS D-fragment layout.
// Agreement with an FP32 reference matmul validates the CUTLASS (T32,V32)->(M16,K64)
// operand layout against HW expectations.
bool qkt_mxf4nvf4_validate(const half* d_Q, const half* d_K, float* d_D, cudaStream_t stream);

}  // namespace imp
