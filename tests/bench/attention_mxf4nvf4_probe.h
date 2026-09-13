#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Compile + launch gate for mma.sync mxf4nvf4.block_scale (SageAttention3-style) on sm_120f.
// Returns true if the kernel launches/syncs without CUDA errors; does NOT validate numerics.
// Gate for whether a full MXFP4 FMHA upgrade is feasible.
bool probe_mxf4nvf4_blockscale(cudaStream_t stream);

// Zero-A correctness check: E2M1 zero-encoded A, any B/scales, must yield accumulator == 0.
// Non-zero output signals a wrong hardware/operand-layout assumption; needs re-investigation.
bool probe_mxf4nvf4_allzero_a(cudaStream_t stream, float out_d[4]);

}  // namespace imp
