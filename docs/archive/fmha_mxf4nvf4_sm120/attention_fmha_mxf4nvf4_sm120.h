#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// MXFP4 Flash Attention using hardware block-scale MMA (sm_120):
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.
//
// Upgrade path from the legacy kind::f8f6f4.m16n8k32 kernel
// (attention_fmha_mxfp4_sm120.cu). Raw MMA speedup measured at 2.60× on
// RTX 5090 via mxf4nvf4_mma_bench. Real end-to-end attention gain is
// expected to be 1.5-2.5× after accounting for memory + softmax + P·V.
//
// Key differences vs legacy:
//   - K-dim 32 → 64 per MMA (half as many MMAs for the same tile)
//   - Scale stored as FP8 UE4M3 per-16-elem instead of FP32 per-row
//   - Hardware applies scale inside the MMA — no manual scale-apply
//   - Scale layout matches SageAttention3 HW-consumption formula (see
//     nvfp4_quant_hw.cu for the layout details)
//
// Default path for MXFP4 attention prefill since 2026-04-25 (+1.8% vs
// legacy at HD=128). Set IMP_FMHA_BLOCKSCALE=0 to force the legacy path
// (A/B debug). Returns false if config is unsupported (caller falls
// through to next fallback in the dispatcher).
//
// Requirements: sm_120+, head_dim ∈ {64, 128, 256} (HD=96 falls through
// internally to legacy kind::f8f6f4 because 96 % 64 != 0).
bool fmha_sm120_mxf4nvf4_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                 bool causal, int sliding_window, float softcap, cudaStream_t stream);

// Reports whether the blockscale path is active (default ON; false only
// when IMP_FMHA_BLOCKSCALE=0 explicitly disables it).
bool mxf4nvf4_blockscale_enabled();
bool mxf4nvf4_blockscale_disabled();

}  // namespace imp
