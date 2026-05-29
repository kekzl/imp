#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// Native sm_120 FMHA for prefill attention.
//
// Uses WMMA HMMA fragments (mma.sync.m16n8k16.f16, mma.sync.m16n8k32.e4m3) —
// NOT wgmma: wgmma.mma_async / TMEM / tcgen05 are Hopper-and-later (sm_90+/
// SM100+) and unavailable on Consumer Blackwell (sm_120a). FA4 is therefore
// permanently incompatible with this target. See review/phase2_perf.md §3.
//
// Supports: FP16, causal masking, softcap, sliding window, GQA.
// Head dims: 64, 96, 128, 256. Falls back for unsupported configs.
//
// Q: [batch, seq_q, n_heads, head_dim]
// K,V: [batch, seq_kv, n_kv_heads, head_dim]
// O: [batch, seq_q, n_heads, head_dim]
//
// Returns true on success, false if config unsupported (caller falls back).
bool fmha_sm120_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                        bool causal, int sliding_window, float softcap, cudaStream_t stream,
                        int q_offset = 0);

// FP8 variant: QK^T computed in FP8 E4M3 (m16n8k32) for 2x score throughput.
// Q,K converted to FP8 on-the-fly in shared memory. PV stays FP16.
// Requires SM120+ with CUTE_ARCH_F8F6F4_MMA_ENABLED.
bool fmha_sm120_fp8_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream,
                            int q_offset = 0);

// FA2 variant ("echtes FA"): true register-resident FlashAttention-2.
// QK^T in FP8 E4M3 (m16n8k32), softmax + P kept in REGISTERS (no S/P/O smem
// round-trip), PV via hand-written mma.sync.m16n8k16 (f16) — exploiting the
// layout identity between the m16n8 accumulator output and the m16n8k16 A
// operand, so P feeds PV with no transpose. Only K (fp8) + V (f16) are staged
// in smem → one __syncthreads per KV tile. Each warp owns 16 query rows and
// runs its online softmax independently (no cross-warp reduction).
// Target: long-context prefill where the smem-materializing fp8 kernel is
// barrier-bound (ncu: 14.5% compute, 75.7% L1/TEX). Head dims: 128 (first).
bool fmha_sm120_fa2_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream,
                            int q_offset = 0);

}  // namespace imp
