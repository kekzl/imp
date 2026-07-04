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
// When use_blockscale=true, swaps the Phase 1 MMA from
//   mma.sync.kind::f8f6f4.m16n8k32   (legacy, 2× K-chunks per issue)
// to
//   mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64  (half MMA count)
// with REAL per-16-element UE4M3 scales (per-(row, k_group) absmax) applied by
// the hardware — finer quantization granularity than the legacy per-row path.
// head_dim must be a multiple of 64 (legacy path requires only multiple of
// 32); unsupported head_dim 96 falls through to the legacy path.
//
// #846 SageAttention3-recipe knobs (read from process_diag, blockscale only):
//   mxfp4_ksmooth: K per-channel mean smoothing — a pre-pass computes the
//     per-(batch, kv_head, channel) mean of K over seq_kv and the kernel
//     subtracts it before quantization. The dropped Q·mean^T score term is
//     constant per query row, so softmax is invariant. Auto-disabled when
//     softcap > 0 (tanh breaks the shift invariance).
//   mxfp4_pv_fp4: P·V in NVFP4 too — P quantized per-row two-level (rescaled
//     to the full E4M3 scale range before 1x16 microscaling to prevent
//     scale-factor collapse on the post-softmax long tail), V per-16-block
//     along the KV dim, PV via the same block-scaled MMA.
//   mxfp4_promote_budget: ThriftAttention-style outlier promotion (arXiv
//     2605.23081). A pre-pass scores every (q_tile, kv_tile) pair by the
//     block-mean dot Q̄·K̄^T and promotes the top budget-fraction of visible
//     KV tiles (sink + diagonal always included) to exact compute: FP32
//     scores from global FP16, FP16 WMMA P·V. Requires blockscale; head_dim
//     64/128 only. INVARIANT: with ksmooth active, the promoted score path
//     subtracts the same K channel mean — mixing shifted (FP4) and unshifted
//     (promoted) columns inside one softmax row would corrupt the result.
//
// q_offset: global position of Q row 0 (chunked-prefill continuation) —
// causal/sliding-window masks use q_offset + local row.
bool fmha_sm120_mxfp4_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                              bool causal, int sliding_window, float softcap, cudaStream_t stream,
                              bool use_blockscale = false, int q_offset = 0);

}  // namespace imp
