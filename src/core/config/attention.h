#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Attention {
    std::string fp8_prefill = "auto";
    // fp8-QK FMHA (e4m3 RAW Q/K, no per-tile scaling): ~10% relative score error
    // that compounds across layers (#511). Opt-in ("on") for experiments only;
    // anything else = off.
    std::string fp8_fmha = "never";
    int fmha_prefill_threshold = -1;  // -1 = auto (derived from S-matrix capacity)
    std::string fmha_sm120 = "auto";
    // Register-resident FA2 prefill kernel. "on" (default) serves F16 hd=128 in
    // the tiled prefill chain; QK^T is f16 by default, fp8-QK only when
    // fa2_fp16qk=never AND fp8_fmha=on. Declines to FP16 WMMA FMHA otherwise.
    std::string fmha_fa2 = "on";
    // FP16-QK FA2 for short prefill (seq < fmha_prefill_threshold, hd=128): f16
    // QK^T, f32 accumulate, avoids the e4m3 quality cliff (#511/#512). Declined
    // configs fall back to cuBLAS; "never" restores the materialized path always.
    std::string fa2_fp16qk = "on";
    // f16-accumulate QK^T (#597): sm_120 runs f16-src/f32-acc HMMA at 1/4 rate
    // (#606); f16 accumulate reaches full rate. Only affects the fa2_fp16qk
    // path, not fp8-QK (keeps f32 accumulate). Default on.
    bool fa2_f16acc = true;
    // f16-accumulate the PV MMA too (the last 1/4-rate HMMA in FA2); packing O
    // as half2 halves the O-fragment register footprint. Safe: O rows are convex
    // combinations of V. Requires fa2_f16acc. Default on.
    bool fa2_pv_f16acc = true;
    // HD=256 FA2 port (Qwen3.6 hybrids / gemma-class): routes head_dim=256
    // prefill through the register-resident FA2 kernel (fp16-qk, Bq=64/TWOSLOT)
    // instead of SMEM-tiled WMMA/cuBLAS. Also gates the FP8-KV cuBLAS skip at hd=256.
    bool fa2_hd256 = true;
    // KV tile rows for the HD=256 FA2 instance: 64 or 32. At Bkv=64 the TWOSLOT
    // tile is 67.6 KB of the 100 KB SM budget (1 CTA/SM); Bkv=32 halves it for
    // 2 CTAs/SM. Register/spill data: tools/kernel_resource_baseline.txt.
    int fa2_hd256_bkv = 64;
    // Dense (hd=128, Bq=128) FA2 at 2 CTAs/SM: TWOSLOT tile 35 KB, plus
    // __launch_bounds__(256, 2) pins 128 registers (137 unconstrained, 40 B
    // local frame per the ptxas baseline). Default on.
    bool fa2_dense_2cta = true;
    // FP8 paged decode (HD=128): tokens per warp iteration. 4 = multitok
    // kernels (Q heads of a KV head grouped per CTA, 16-lanes-per-row layout),
    // 1 = the plain per-head kernel serves shapes multitok does not.
    int paged_fp8_multitok = 4;
    // NVFP4 paged decode (HD=128/256): tokens per warp iteration. 4 = multitok
    // kernels grouping a KV head's Q heads per CTA (each row converted once);
    // 1 = the scalar kernels.
    int paged_nvfp4_multitok = 4;
    // F16 paged decode (HD=128/256, GQA 1..8, non-split-K path): tokens per
    // warp iteration on the multitok kernel, sharing each KV row across a
    // CTA's Q heads. 1 = the cooperative and per-head split-K kernels.
    int paged_f16_multitok = 4;
    // Causal FA2 CTA order: heaviest q-tiles first. A causal q-tile attends
    // more KV tiles than an early one, so ascending order left the wave tail
    // idling most SMs. Reversing the tile index per head is output bit-identical.
    bool fa2_heavy_first = true;
    // amax-scaled e4m3 conversion for the fp8-QK FA2 path (#680): scales Q/K to
    // the full e4m3 range (the FlashInfer numerics class) instead of raw
    // conversion's #511 quality cliff. Only affects the fp8-QK path. Experimental.
    bool fp8_qk_scaled = false;
    std::string mxfp4 = "auto";
    // #846 NVFP4-attention spike (SageAttention3 recipe); all three require the
    // MXFP4 FMHA to serve prefill (mxfp4=always).
    // mxfp4_blockscale: per-16-element UE4M3 block scales (mxf4nvf4.block_scale
    // MMA) instead of legacy per-row software scales.
    // mxfp4_ksmooth: subtract the per-(batch,kv_head,channel) K mean before
    // quant; the dropped Q.mean^T term is row-constant and cancels under
    // softmax. Auto-disabled when softcap>0. Requires mxfp4_blockscale.
    // mxfp4_pv_fp4: P.V in NVFP4 too (P rescaled to the full E4M3 range,
    // per-row two-level; V per-16-block along KV). Requires mxfp4_blockscale.
    // mxfp4_promote_budget: ThriftAttention outlier promotion (arXiv 2605.23081):
    // top-scoring fraction of causal 64-token KV tiles computed exact in
    // FP32/FP16 (sink+diagonal force-included). 0=off, 1=all. Requires
    // mxfp4_blockscale; head_dim 64/128 only.
    bool mxfp4_blockscale = false;
    bool mxfp4_ksmooth = false;
    bool mxfp4_pv_fp4 = false;
    float mxfp4_promote_budget = 0.0f;
    // mxfp4_paged_kv: continuation prefill chunks read K/V directly from the
    // paged NVFP4 KV cache (quantized once at append, no gather->FP16 pass).
    // Requires kv_cache.dtype=nvfp4, head_dim 128, single sequence.
    bool mxfp4_paged_kv = false;
    bool mxfp4_fp16_fallback = false;
    // MXFP4->FP16 cache pruning: "legacy" caches FP16 for every MXFP4 tensor.
    // "pruned" skips MoE expert_*_packed (bypassed by the batch-dequant path)
    // and LM head (routed through generic-dequant); needed for Qwen3.5-27B MXFP4 on 32 GiB.
    std::string mxfp4_fp16_cache_policy = "legacy";
    bool force_cublas_decode = false;
    // MLA absorbed-decode latent KV (DeepSeek-V2/V3). Off = materialized Stage A
    // (full per-head K/V reconstructed at projection, standard paged attention).
    // On: stores only the compressed latent + decoupled RoPE key, single-sequence only.
    bool mla_absorb = false;
    bool no_qknorm_fused = false;
    bool splitk_pipe = true;
    // Token-tiled FP8 split-K decode attention (hd=128/bs=16 only). Off =
    // per-token pipeline kernel; A/B + rollback knob.
    bool fp8_tile = true;
    // GQA-batched tile variant: one block computes all Q heads of a KV head
    // from a shared smem tile (L2 KV traffic /G). Off = per-head tile
    // kernel; A/B + rollback knob.
    bool fp8_tile_gqa = true;
    bool gate_concat = false;
    // Max VRAM (MiB) for the materialized cuBLAS-attention S-matrix; caps
    // prefill context on the fast path before falling back to FMHA (auto
    // fmha_prefill_threshold = cap+1). Auto-shrinks if the alloc fails. Default 384.
    int attn_scores_mib = 384;
    // Sparse decode attention (Quest-class top-k page selection). 0=off; >0
    // reads only the top-scoring KV blocks per step, bit-identical to dense at
    // or below budget. v1 gates: F16/FP8 KV, uniform geometry, non-growable, non-MLA.
    int sparse_topk_tokens = 0;
    // Below this context length decode stays dense even when the budget is
    // exceeded: the selection's win only outgrows its overhead past ~12k on
    // the measured dense model (8k measured -7%, 16k +6%, 32k +25%).
    int sparse_min_ctx = 12288;
    // Blocks covering the first sparse_sink_tokens positions are always kept.
    int sparse_sink_tokens = 16;
    // Blocks covering the last sparse_recent_tokens positions are always kept
    // (includes the partially filled tail block).
    int sparse_recent_tokens = 256;
    // Page score: true = mean of the page's keys with their stddev as offset
    // (arXiv 2605.27740); false = the Quest min/max corner bound. Same formula
    // sum_d(q*center+|q|*offset); corner ranks worse at small budgets (driven by one extreme token).
    bool sparse_score_meanstd = true;
    // Offset weight for sparse_score_meanstd. Larger keeps more of the
    // spread, 0 ranks on the mean alone. Ignored by the corner bound.
    float sparse_score_std_coef = 1.0f;
};
}  // namespace imp::cfg
