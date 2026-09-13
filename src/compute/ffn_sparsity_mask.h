#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Builds a packed bit mask over Q8_0-aligned K-blocks (32 elements each):
// mask[b] = 1 iff max_{i in block b} |silu(gate[i])*up[i]| >= threshold.
// Layout: packed uint32, bit b of word w = q8-block index (w*32+b); mask needs >= ceil((K/32)/32)
// words. K must be a multiple of 32.
void build_swiglu_block_mask(const __half* gate, const __half* up, uint32_t* mask, int K,
                             float threshold, cudaStream_t stream);

// Mask-aware variant of gemv_q8_0_q8_1_residual (kpar layout): skips entire Q8_0 weight-blocks
// whose mask bit is 0 (no HBM load, no dp4a, no scale fetch). Output bit-identical to the
// unmasked kernel when every mask bit is 1. Layout: 4 warps/output row, mirrors
// gemv_dp4a_traits.cuh:580.
void gemv_q8_0_q8_1_residual_masked(const void* W, const struct block_q8_1* q8_1, const float* d8,
                                    const uint32_t* mask, __half* y, const __half* residual,
                                    int M, int K, cudaStream_t stream);

}  // namespace imp
