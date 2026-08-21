#pragma once

// Attention configuration, one of the nine sections split out of
// core/dispatch_policy.h on 2026-08-21.
//
// WHY. dispatch_policy.h aggregates all nine and is included by 23 translation
// units, of which 21 touch two sections or fewer. Adding one field to it costs
// 137.1 s of incremental rebuild, against 9.1 s for a small .cpp and 14.6 s for
// the largest .cu the file-size gate polices. A TU that needs only this section
// can include only this header and stop rebuilding when the others change.
//
// This is F-10 one level down, and dispatch_policy.h's own preamble records the
// original: config.h was included by 22 files, 85 TUs transitively, and changed
// 130 times in six months - "the highest build cost in the repo". Lifting nine
// sections into an aggregate fixed that, and gave the aggregate the same
// property for the same reason.
//
// Pure move: the contents below are byte-identical to their previous form, and
// dispatch_policy.h includes every one of these, so no existing include breaks.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Attention {
    std::string fp8_prefill = "auto";
    // fp8-QK FMHA family (smem-materializing fp8 kernel + FA2 in fp8-QK
    // mode): converts Q/K to e4m3 RAW (no per-tile scaling) — ~10% relative
    // score error on real activations that compounds across layers (#511).
    // Teacher-forced PPL when this kernel actually serves prefill:
    // gemma-3-12b 16.6 -> 549 (production chunked long-ctx), Qwen3-8B
    // 40.5 -> 4506 (forced). The #511 "no measurable loss above threshold"
    // needle check never exercised this kernel (fa2_fp16qk served those
    // chunks). Opt-in ("on") for experiments; anything else = off.
    std::string fp8_fmha = "never";
    int fmha_prefill_threshold = -1;  // -1 = auto (derived from S-matrix capacity)
    std::string fmha_sm120 = "auto";
    // Register-resident FA2 prefill kernel (fmha_sm120_fa2_kernel). When "on"
    // (default) it serves supported configs (F16, head_dim=128) in the tiled
    // prefill chain — keeps S/P/O in registers, 1 __syncthreads/KV tile.
    // QK^T mode follows fa2_fp16qk: f16-QK by default (no e4m3 score noise,
    // #511); fp8-QK only when fa2_fp16qk=never AND fp8_fmha=on. Declines
    // (-> FP16 WMMA FMHA) for hd!=128 (Gemma), non-F16, or insufficient
    // smem, so it's safe by default. Legacy env: IMP_FMHA_FA2.
    std::string fmha_fa2 = "on";
    // FP16-QK FA2 for SHORT prefill (seq < fmha_prefill_threshold, hd=128):
    // replaces the materialized cuBLAS+softmax path with the register-
    // resident FA2 kernel running QK^T in f16 (mma.m16n8k16) instead of
    // e4m3 — same numerical class as the cuBLAS reference (f16 inputs,
    // f32 accumulate), so the short-seq e4m3 quality cliff (#511/#512)
    // does not apply. O(n) memory: no S-matrix alloc. Declined configs
    // (hd!=128, dual-head-dim Gemma-4) fall back to cuBLAS, never to the
    // fp8 FMHA family. "never" restores the materialized cuBLAS path.
    std::string fa2_fp16qk = "on";
    // f16-accumulate QK^T in the FP16-QK FA2 kernel (#597). GeForce sm_120
    // runs f16-src/f32-acc HMMA at 1/4 rate (#606); accumulating the score
    // MMA in f16 lifts it to the full-rate class. Measured +4.7-5.0%
    // pp4096 NVFP4 prefill (Qwen3-14B / 30B-A3B, 2026-06-11, chunk-2048
    // era), decode neutral. Quality gate on a 5.8k teacher-forced corpus:
    // 14B-NVFP4 PPL identical, 30B-A3B +0.10%, Q8_0 GGUF +0.013% —
    // scores are softmaxed immediately, so the reduced accumulate
    // precision stays in the noise. Default ON since 2026-06-11; set
    // false to restore f32 accumulate. Only affects the fa2_fp16qk path,
    // the fp8-QK path keeps f32 accumulate.
    bool fa2_f16acc = true;
    // f16-accumulate the PV MMA as well. Post-#673 the PV accumulate was
    // the last 1/4-rate HMMA in the FA2 kernel, dominating its tensor-
    // pipe time ~4:1; packing O as half2 also halves the O-fragment
    // register footprint of the Bq=128 band. Measured (2026-06-11, nsys
    // kernel sums): FA2 kernel −18% pp4096, e2e +9.7% 30B-A3B-NVFP4 /
    // +3.7% 14B-NVFP4. Quality gate on a 14.8k teacher-forced corpus:
    // 14B −0.06%, 30B-A3B −0.30%, Q8_0 +0.002% — all noise (O rows are
    // convex combinations of V, so range is safe; the per-tile rescale
    // rounding stays below the f16 output precision). Default ON since
    // 2026-06-11; set false to restore f32 PV accumulate. Requires
    // fa2_f16acc.
    bool fa2_pv_f16acc = true;
    // HD=256 FA2 port (Qwen3.6 hybrids / gemma-class): route head_dim=256
    // prefill through the register-resident FA2 kernel (fp16-qk,
    // Bq=64/TWOSLOT) instead of the SMEM-tiled WMMA FMHA / cuBLAS.
    // Default ON since stage 3 (#930 measured: kernel 4.3x vs WMMA,
    // e2e +10.6% pp4096 / +24.8% pp8192, PPL 10.44 vs 10.58 on
    // Qwen3.6-35B; split-D stage 2 was refuted — the 4-warp instance is
    // the keeper). Also gates the FP8-KV deterministic-cuBLAS skip for
    // hd=256 models (engine_init_resolver fa2_serves_attention).
    bool fa2_hd256 = true;
    // amax-scaled e4m3 conversion for the fp8-QK FA2 path (#680). The
    // raw conversion is the #511 quality cliff; scaling Q and K to the
    // full e4m3 range is the numerics class FlashInfer runs. Only
    // takes effect on the fp8-QK path (fa2_fp16qk=never or declined).
    // Experimental quality probe.
    bool fp8_qk_scaled = false;
    std::string mxfp4 = "auto";
    // #846 NVFP4-attention spike (SageAttention3 recipe). All three only
    // take effect when the MXFP4 FMHA serves prefill (mxfp4 = "always").
    // mxfp4_blockscale: per-16-element UE4M3 block scales applied by the
    //   mxf4nvf4.block_scale MMA (vs legacy per-row software scales).
    // mxfp4_ksmooth: subtract the per-(batch,kv_head,channel) K mean before
    //   quantization — the dropped Q·mean^T term is per-row-constant and
    //   cancels under softmax. Auto-disabled when softcap > 0 (tanh breaks
    //   the shift invariance). Requires mxfp4_blockscale.
    // mxfp4_pv_fp4: P·V in NVFP4 too — P quantized per-row two-level
    //   (rescaled to the full E4M3 scale range before 1x16 microscaling),
    //   V per-16-block along KV. Requires mxfp4_blockscale.
    // mxfp4_promote_budget: ThriftAttention-style outlier promotion
    //   (arXiv 2605.23081) — per q-tile, the top-scoring fraction of
    //   causally visible 64-token KV tiles (block-mean importance score
    //   Q̄·K̄^T; sink + diagonal tiles force-included) is computed exactly
    //   in FP32/FP16 instead of FP4. 0 = off, 1 = promote everything.
    //   Requires mxfp4_blockscale; head_dim 64/128 only.
    bool mxfp4_blockscale = false;
    bool mxfp4_ksmooth = false;
    bool mxfp4_pv_fp4 = false;
    float mxfp4_promote_budget = 0.0f;
    // mxfp4_paged_kv: KV-append-quant chunked prefill (#846 follow-up) —
    //   continuation chunks read K/V DIRECTLY from the paged NVFP4 KV
    //   cache (quantization paid once at append; no gather→FP16 pass, no
    //   in-kernel quant). Combines with mxfp4_promote_budget for outlier
    //   promotion. Requires kv_cache.dtype=nvfp4, head_dim 128, single
    //   sequence; engages independently of `mxfp4` mode.
    bool mxfp4_paged_kv = false;
    bool mxfp4_fp16_fallback = false;
    // MXFP4 → FP16 cache pruning policy. "legacy" (default) caches FP16
    // for every MXFP4 tensor. "pruned" skips MoE expert_*_packed and
    // LM head (out_proj_) — those slots are either not read on the
    // dispatch hot path (MoE expert FP16 cache is only consumed by
    // executor_forward_moe.cu's pre-cached FP16 fallback, which is
    // bypassed by the more efficient batch-dequant path for MXFP4)
    // or routed through generic-dequant (LM head). Pruning is the
    // Phase A1+A2 path — it
    // unlocks Qwen3.5-27B MXFP4 load on 32 GiB VRAM by shrinking the
    // ~48 GiB FP16 fallback to ~8-12 GiB.
    std::string mxfp4_fp16_cache_policy = "legacy";
    bool force_cublas_decode = false;
    // MLA absorbed-decode latent KV cache (DeepSeek-V2/V3, Phase 3). When
    // off (default) the materialized Stage A path runs (full per-head K/V
    // reconstructed at projection time + standard paged attention). When on,
    // decode stores only the compressed latent + decoupled RoPE key and runs
    // the mathematically-equivalent absorbed attention (~9x smaller per-token
    // KV footprint). Prefill stays materialized; the latent cache is
    // populated during prefill/decode. Single-sequence only (falls back to
    // materialized otherwise). Env: none.
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
    // Max VRAM (MiB) for the materialized cuBLAS-attention S-matrix. Caps the
    // prefill context length that uses the fast cuBLAS attention path before
    // falling back to FMHA (auto fmha_prefill_threshold = S-matrix cap + 1).
    // 256 MiB caps ~32-head models at seq 2048 but high-head-count models
    // (e.g. Qwen3-14B, 40 heads → ~1824) drop to the slower FMHA at 2048.
    // Larger = longer prefill on the fast path, at the cost of KV headroom.
    // Auto-shrinks if the alloc fails.
    // 384 keeps the fast cuBLAS attention path up to seq 2048 for up to
    // 48-head models (e.g. Qwen3-14B, 40 heads: +21% pp2048 vs the old 256
    // cap which dropped it to FMHA at ~1824). Only allocates what the
    // model's max_tokens×heads needs (capped here); +128 MiB vs 256 at most.
    int attn_scores_mib = 384;
};
}  // namespace imp::cfg
