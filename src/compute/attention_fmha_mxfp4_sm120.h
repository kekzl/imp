#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// MXFP4 Flash Attention for sm_120 (Blackwell): tiled FP4 E2M1 Q·K^T with
// online softmax, P·V in FP16 WMMA.  O(n) memory — no S matrix materialization.
//
// Uses bare FP4 MMA (mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32)
// with per-row scale correction:  S_true[i,j] = q_scale[i] * k_scale[j] * S_mma[i,j].
// Q and K are quantized to FP4 E2M1 per-tile in shared memory with per-row absmax.
//
// Compared to the CUTLASS-based attention_mxfp4_prefill.cu, this kernel:
//   - Uses tiled flash attention (O(n) memory, not O(seq²))
//   - Is a single fused kernel (no separate quant + GEMM + softmax + P·V launches)
//   - Supports sliding window and softcap
//
// Requirements:
//   - sm_120+ (__CUDA_ARCH__ >= 1200, f8f6f4 MMA instructions)
//   - head_dim % 32 == 0 (FP4 MMA k-dim = 32)
//   - Supported head_dim: 64, 96, 128, 256
//
// Returns false if config unsupported (caller falls back to FP8/FP16 FMHA).
bool fmha_sm120_mxfp4_prefill(
    const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
    float scale, bool causal, int sliding_window, float softcap,
    cudaStream_t stream);

} // namespace imp
