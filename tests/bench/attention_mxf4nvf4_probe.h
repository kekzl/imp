#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Compile + launch gate for the SageAttention3-style MMA instruction
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...
// on sm_120f + CUDA 13.2.
//
// Returns true if the kernel launches and synchronizes without CUDA
// errors. Does NOT validate numerical correctness — this is only a gate
// for whether a full MXFP4 FMHA upgrade project is feasible.
bool probe_mxf4nvf4_blockscale(cudaStream_t stream);

// Correctness check: runs the same MMA with all-zero A operands. With
// E2M1 zero-encoded (0x00) in A, any B content, and any scale factors,
// the fused accumulator `d += (dequant(A) * dequant(B))` must evaluate
// to exactly 0 across all output lanes.
//
// Outputs the first thread's 4 accumulator values via `out_d[4]`. A
// non-zero return from any element signals the hardware / operand
// layout assumption is wrong and Project B needs re-investigation.
bool probe_mxf4nvf4_allzero_a(cudaStream_t stream, float out_d[4]);

}  // namespace imp
