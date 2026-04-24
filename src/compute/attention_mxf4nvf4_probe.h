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

} // namespace imp
