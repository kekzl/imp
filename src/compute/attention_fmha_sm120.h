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
                        bool causal, int sliding_window, float softcap, cudaStream_t stream);

// FP8 variant: QK^T computed in FP8 E4M3 (m16n8k32) for 2x score throughput.
// Q,K converted to FP8 on-the-fly in shared memory. PV stays FP16.
// Requires SM120+ with CUTE_ARCH_F8F6F4_MMA_ENABLED.
bool fmha_sm120_fp8_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                            bool causal, int sliding_window, float softcap, cudaStream_t stream);

// Cluster variant: blocks sharing a KV head form a cluster of size n_q_per_kv;
// block-rank 0 loads K and V into its shared memory, sibling Q-head blocks
// DSMEM-read the tiles via cluster.map_shared_rank(). Saves
// n_q_per_kv× global KV bandwidth on GQA configs.
//
// Returns false (no kernel launched, fall back to fmha_sm120_prefill) when:
//   - RuntimeConfig.attention.no_fmha_cluster is set
//   - n_q_per_kv ∉ {2, 4, 8} (cluster dim must be power-of-2 ≤ 8 on GB202)
//   - head_dim ∉ {64, 96, 128, 256}
//   - seq_kv < CL_Bkv * 8 (short prompts — cluster.sync barriers dominate)
//   - device smem budget can't host the largest selected Bq
//
// Sibling kernel of fmha_sm120_prefill — see attention_fmha_sm120_cluster.cu.
bool try_fmha_sm120_cluster_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                                    bool causal, int sliding_window, float softcap, cudaStream_t stream);

}  // namespace imp
