#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Native sm_120 FMHA for prefill: WMMA HMMA fragments (mma.sync.m16n8k16.f16) - NOT
// wgmma/TMEM/tcgen05 (sm_90a, sm_100+ only, unavailable on sm_120a).
// Supports FP16, causal, softcap, sliding window, GQA. Head dims 64/96/128/256; falls back
// otherwise. Q:[batch,seq_q,n_heads,hd] K,V:[batch,seq_kv,n_kv_heads,hd] O: same as Q.
// sinks (gpt-oss #547): optional [n_heads] FP16, virtual extra logit column with no V
// contribution, folded into online-softmax init (m=sink, l=1). nullptr = off.
// fixed_kv_order: forward KV scan for every Q tile; a row's bits then do not depend on its tile.
[[nodiscard]] bool fmha_sm120_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                        bool causal, int sliding_window, float softcap, cudaStream_t stream,
                        int q_offset = 0, const half* sinks = nullptr, bool fixed_kv_order = false);

// FA2 ("echtes FA"): true register-resident FlashAttention-2. QK^T and PV via hand-written
// mma.m16n8k16 f16, softmax+P kept in registers (no S/P/O smem round-trip) - exploits the layout
// identity between the QK accumulator and the PV A-operand (no transpose). Q in registers, only
// K+V (f16) staged in smem. Each warp: 16 rows, independent online softmax. Head dims 64/128/256.
// d_kv_len: optional device int overriding KV length (K.shape[1] is only buffer capacity);
// kernel derives q_offset=seq_kv-seq_q. Grid depends only on seq_q, so a captured graph replays
// correctly as context grows (#847).
[[nodiscard]] bool fmha_sm120_fa2_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                                          float scale, bool causal, int sliding_window, float softcap,
                                          cudaStream_t stream, int q_offset = 0,
                                          const int* d_kv_len = nullptr, const half* sinks = nullptr);

}  // namespace imp
