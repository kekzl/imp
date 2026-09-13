#pragma once

#include "core/tensor.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

// MXFP4 flash attention for sm_120: tiled FP4 E2M1 Q.K^T with online softmax, FP16 WMMA P.V.
// O(n) memory, no S materialization. mma.sync.aligned.kind::f8f6f4.m16n8k32.e2m1.e2m1.f32 with
// per-row scale correction: S_true[i,j] = q_scale[i]*k_scale[j]*S_mma[i,j].
// Requires sm_120+ (__CUDA_ARCH__>=1200), head_dim%32==0. Supported head_dim: 64,96,128,256.
// Returns false if unsupported (caller falls back to FP8/FP16 FMHA).
// use_blockscale swaps Phase 1 to kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 (half the
// MMA count, real per-16-elem UE4M3 scales via HW); requires head_dim % 64 == 0, else falls
// through to the legacy (%32) path.
// #846 knobs (process_diag, blockscale only): mxfp4_ksmooth subtracts per-(batch,kv_head,
// channel) K mean before quant (dropped Q.mean^T term cancels under softmax; auto-disabled when
// softcap>0). mxfp4_pv_fp4 runs P.V in NVFP4 too (P two-level per-row quant, V per-16-block).
// mxfp4_promote_budget (ThriftAttention, arXiv 2605.23081): promotes the top budget-fraction of
// visible KV tiles (sink+diagonal always included) to exact FP32 scores + FP16 WMMA P.V.
// INVARIANT: with ksmooth active, promoted tiles must subtract the same K mean, or mixed
// shifted/unshifted columns corrupt the softmax row.
// q_offset: global position of Q row 0 (chunked-prefill continuation); masks use q_offset+row.
bool fmha_sm120_mxfp4_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                              bool causal, int sliding_window, float softcap, cudaStream_t stream,
                              bool use_blockscale = false, int q_offset = 0);

// #846 KV-append-quant: chunked-prefill attention reading K/V directly from the paged NVFP4 KV
// cache (current chunk must already be appended before calling). promote_budget>0 promotes PAST
// tiles to exact FP32 dots over dequantized cache K; the CURRENT chunk is always exact FP16
// (quantizing the recency window is where quality damage lives). Requires batch=1, hd=128, Q FP16.
bool fmha_sm120_mxfp4_prefill_paged(const Tensor& Q, Tensor& O, const half* k_fresh,
                                    const half* v_fresh, const uint8_t* k_data,
                                    const uint8_t* k_scales, const uint8_t* v_data,
                                    const uint8_t* v_scales, const int* block_table, int block_size,
                                    int seq_kv, int n_kv_heads, float scale, bool causal,
                                    int sliding_window, float softcap, cudaStream_t stream, int q_offset,
                                    float promote_budget);

}  // namespace imp
