#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Native sm_120 FMHA for prefill: WMMA HMMA fragments (mma.sync.m16n8k16.f16,
// mma.sync.m16n8k32.e4m3) - NOT wgmma/TMEM/tcgen05 (sm_90+/sm_100+ only, unavailable on sm_120a).
// Supports FP16, causal, softcap, sliding window, GQA. Head dims 64/96/128/256; falls back
// otherwise. Q:[batch,seq_q,n_heads,hd] K,V:[batch,seq_kv,n_kv_heads,hd] O: same as Q.
// sinks (gpt-oss #547): optional [n_heads] FP16, virtual extra logit column with no V
// contribution, folded into online-softmax init (m=sink, l=1). nullptr = off.
bool fmha_sm120_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                        bool causal, int sliding_window, float softcap, cudaStream_t stream,
                        int q_offset = 0, const half* sinks = nullptr);

// FP8 variant: QK^T computed in FP8 E4M3 (m16n8k32) for 2x score throughput.
// Q,K converted to FP8 on-the-fly in shared memory. PV stays FP16.
// Requires SM120+ with CUTE_ARCH_F8F6F4_MMA_ENABLED.
bool fmha_sm120_fp8_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream,
                            int q_offset = 0);

// FA2 ("echtes FA"): true register-resident FlashAttention-2. QK^T in FP8 E4M3 (m16n8k32),
// softmax+P kept in registers (no S/P/O smem round-trip), PV via hand-written mma.m16n8k16 f16 -
// exploits the layout identity between the QK accumulator and the PV A-operand (no transpose).
// Only K(fp8)+V(f16) staged in smem, one syncthreads/KV tile. Each warp: 16 rows, independent
// online softmax. Head dims: 128 (first). fp16_qk=true: QK via mma.m16n8k16.f16 (Q/K in smem as
// f16) - no e4m3 noise, safe below fmha_prefill_threshold (#511/#512). Bq=64 only.
// d_kv_len: optional device int overriding KV length (K.shape[1] is only buffer capacity);
// kernel derives q_offset=seq_kv-seq_q. Grid depends only on seq_q, so a captured graph replays
// correctly as context grows (#847). fp16_qk path only.
bool fmha_sm120_fa2_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream,
                            int q_offset = 0, bool fp16_qk = false, const int* d_kv_len = nullptr);

}  // namespace imp
