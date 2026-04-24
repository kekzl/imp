#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// End-to-end Q·K^T correctness harness for the mxf4nvf4.block_scale MMA.
// Q: [16, 64] FP16, K: [8, 64] FP16 — both row-major device pointers.
// D: [16, 8] FP32 device output.
//
// Internally quantizes Q and K to E2M1 on-the-fly, issues one
// mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 with
// uniform scale = 1.0 (FP8 UE4M3 byte 0x38), writes the 16×8 output
// back per CUTLASS D-fragment layout.
//
// Use to compare against an FP32 reference matmul of the same Q, K
// inputs. Agreement validates that our CUTLASS (T32,V32)→(M16,K64)
// operand layout interpretation matches HW expectations.
bool qkt_mxf4nvf4_validate(const half* d_Q, const half* d_K, float* d_D,
                           cudaStream_t stream);

} // namespace imp
