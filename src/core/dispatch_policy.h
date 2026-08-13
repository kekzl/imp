#pragma once

// The nine RuntimeConfig sections that src/exec reads, lifted out of
// runtime/config.h so the hot layer does not depend on the top one.
//
// Audit finding F-10: config.h was included by 22 files in src/exec/ (85
// translation units transitively) and changed 130 times in six months — the
// highest build cost in the repo. About half that churn lands in sections
// exec/ never reads (Runtime, Vram, Server, Rope); this is what decouples it.
// It is also the durable form of F-1: ProcessDiag exists precisely because
// leaf utilities could not carry a RuntimeConfig.
//
// They live in namespace imp::cfg because un-nesting collides otherwise:
// imp::KVCache is the paged cache in memory/kv_cache.h, and imp-bench has its
// own AttentionConfig. Member names are unchanged, so every existing
// cfg.attention.x still reads exactly the same.
//
// DispatchPolicy is a SNAPSHOT, filled after the init resolvers (engine.cpp:
// resolvers ~851-856, handover ~972). One writer exists after that point —
// executor_workspace_buffers.cu const_casts the derived
// attention.fmha_prefill_threshold — and every reader of that field is inside
// src/exec/, so it writes and reads the same snapshot. Checked, not assumed.
// A future mutation whose readers live OUTSIDE exec would silently diverge.

#include <cstdint>
#include <string>
#include <vector>

namespace imp {
enum class SwaSizingMode { Off, On, Auto };

namespace cfg {

struct KVCache {
    // "auto" (default) keeps FP16 but upgrades to FP8 E4M3 for models whose
    // author declares kv_cache_quant_algo=FP8 AND whose arch family is
    // verified safe for long-context FP8 KV (see kv_fp8_hint_default_safe).
    // "fp16" forces FP16 (opt out of the hint). fp8|int8|int4|nvfp4|mxfp4
    // force that dtype regardless of the hint.
    std::string dtype = "auto";
    bool allow_nondeterministic_fp8 = false;
    // Legacy unconditional FP8 auto-upgrade: force FP8 E4M3 whenever the
    // dtype resolves to FP16 (pre-hint behavior). imp.conf key only — the
    // old IMP_KV_FP8_AUTO env var is no longer read.
    bool fp8_auto_legacy = false;
    // BitDecoding Phase 3: residual FP16 cache for newest N tokens.
    // 0 = disabled (keeps Phase 1+2 behavior). Typical: 4..32.
    // Only meaningful with kv_cache.dtype = "nvfp4" + kv_cache.bitdecoding_qk.
    int bitdecoding_residual_tokens = 0;
    // BitDecoding TC path for NVFP4 paged attention QK.
    bool bitdecoding_qk = false;
    // SWA-aware KV sizing: sliding-window layers (gpt-oss window=128 on
    // every other layer, gemma-3 5:1 pattern) allocate only the trailing
    // window in a small dedicated block group instead of full-length KV
    // (~2x more KV tokens on gpt-oss, ~5-6x on gemma-3). Auto-disabled
    // (logged) for models without SWA layers, INT8/INT4 KV, hybrids,
    // MLA, StreamingLLM, green contexts, and deterministic mode.
    // Numerically exact: PPL bit-parity vs full-length KV on gemma-3-12b
    // and gpt-oss-20b (deterministic_gemm A/B, 2026-07-24).
    // Tri-state: "auto" enables the savings only when prefix caching is
    // off (one-shot imp-cli runs), so serving keeps warm-prefix TTFT;
    // "on" forces sizing and disables prefix caching (freed window
    // blocks cannot back prefix reuse — snapshot-based reuse is a
    // follow-up); "off" disables. Legacy bools map to on/off.
    std::string swa_sizing = "auto";

    // SWA window snapshots: device budget (MiB) for packed windowed-layer
    // KV snapshots — what makes prefix caching valid under SWA sizing
    // (freed window blocks cannot back reuse; the snapshot restores the
    // trailing window at the reuse boundary, like the recurrent-state
    // snapshots do for hybrids). One snapshot per prefill end, LRU.
    // 0 = off: swa_sizing=auto then yields to prefix caching, and
    // swa_sizing=on force-disables it.
    int swa_snapshot_mb = 0;

    // Pin the KV pool to exactly this many blocks; 0 = size it from the
    // VRAM budget as usual. An operator sharing a card wants the pool to
    // be a declared quantity rather than "whatever was left", and it is
    // what makes the admission guardrail (I6) reachable from a config
    // rather than only through the C API.
    int max_blocks = 0;

    SwaSizingMode swa_sizing_mode() const {
        if (swa_sizing == "auto")
            return SwaSizingMode::Auto;
        if (swa_sizing == "on" || swa_sizing == "true" || swa_sizing == "True" || swa_sizing == "1" ||
            swa_sizing == "yes")
            return SwaSizingMode::On;
        return SwaSizingMode::Off;
    }
};

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

struct MoE {
    int expert_overhead_pct = 10;
    int force_host_experts = 0;  // last N layers forced to host (0 = none)
    bool skip = false;
    bool force_fp16_sync = false;
    bool no_expert_cache = false;
    // Share of free VRAM the expert LRU cache may claim, in percent. The pool
    // depth this yields is what decides how many tokens of routing history the
    // cache can hold — 73 slots/layer on a 30B-A3B is ~3 tokens, which catches
    // the ~45% next-token reuse but not the ~80%-within-8 band. Exposed so that
    // trade is measurable rather than hardcoded; 15 is the long-standing value.
    int expert_cache_budget_pct = 15;
    // Copy host-resident NVFP4 experts into pinned host memory at load, so the
    // per-expert H2D transfers become real DMAs instead of driver-staged
    // copies. On WSL2 an mmap cannot be page-locked in place, which is why this
    // is a copy rather than a registration (the GGUF packed path does the same
    // thing at weight_upload.cu's Path A1).
    //
    // It is a TRADE, not a win, which is why it is off by default. Measured on
    // Qwen3-30B-A3B-NVFP4 with all 48 MoE layers host-resident, SIX alternating
    // paired rounds (the switch exists partly so the arms can alternate without
    // a rebuild):
    //   prefill pp512  276.6 -> 790.8 tok/s   2.9x
    //   decode  tg256  no effect              3 pairs up, 3 down
    //   model load     5.1 s -> 22.6 s        4.4x, for ~14 GiB of pinned copies
    //
    // The prefill figure is 2.9x rather than the +14.8 % first measured because
    // this flag also gates whole-layer staging: the per-projection slabs it
    // builds are what make one memcpy per projection possible, and a pageable
    // source is driver-staged whatever its size. With this off, layer staging
    // measures 252-286 tok/s, i.e. nothing, so its 324 MiB is not allocated.
    // Decode is unaffected because its cache hits 96-98 % and barely transfers;
    // prefill touches every expert and is what the transfers cost.
    //
    // Three paired rounds read decode as -33 % and that was noise: this path's
    // own decode spread is wider than the effect (the off arm alone measured
    // 34.7 to 66.1 tok/s across six runs). Do not re-derive this from fewer
    // than six pairs.
    bool pin_host_experts = false;
    // Dispatch a STAGED host-resident layer through the CUTLASS grouped NVFP4
    // prefill instead of the per-expert dequant fallback. Requires
    // pin_host_experts (which is what makes staging possible at all).
    //
    // Measured on Qwen3-30B-A3B-NVFP4, all 48 MoE layers host-resident, six
    // alternating paired rounds:
    //   prefill pp512  663.2 -> 1563.9 tok/s   +136 %, 6/6 pairs, spread <1 %
    //   decode  tg256   59.4 ->   37.7 tok/s   -36 %,  6/6 pairs
    //
    // Opt-in because that decode figure is real but NOT understood, and it
    // reverses with context: at pp8 instead of pp512 the same arms measure
    // 25.5 -> 30.6 tok/s, i.e. the staged path is FASTER. This code only runs
    // at n > 1, so it cannot slow the decode kernels directly; what differs is
    // the expert cache's state when decode inherits it (hit rate 91 % vs
    // 92.9-98.4 % after a long prefill, 84.8 % vs 80.6 % after a short one).
    // Until that is explained, a 2.4x prefill win does not get to impose an
    // unexplained decode cost by default.
    bool staged_cutlass_prefill = false;
    // Phase 2 (MoE host-offload Graphs design): assert device-side mirror
    // == host-side LRU state after every cache mutation. Off by default;
    // turn on via `moe.expert_cache_debug_parity = true` in imp.conf for
    // CI / regression diagnosis. Has a meaningful cost (D2H readback of
    // ~120 KiB per cache update) — never enable in perf runs.
    bool expert_cache_debug_parity = false;
    // Phase 4 (async prefetch): at the start of layer L, issue async
    // H2D for up to this many of layer L+1's most-recent (proj, expert)
    // pairs that aren't currently cached. 0 disables the prefetcher
    // (default — safety first, Phase 4 perf gains depend on workload
    // and need per-model measurement). Sensible values: 3..16.
    int prefetch_top_k = 0;
    // Drop the "experts on host → graphs off" guard. Kept as an escape
    // hatch, but measured 2026-08-11 it currently buys NOTHING: every MoE
    // path serving host-resident experts reads routing on the host, so
    // moe_host_args_capture_guard throws under capture and the runner
    // aborts to per-step decode on every attempt. The older note here —
    // "correct only when prefetch coverage matches router selection" —
    // oversold it; capture never reaches the point where that would be the
    // question. Making this real needs routing AND expert residency
    // resolved device-side, and residency needs a host-issued H2D on a
    // miss. See docs/roadmap.md.
    bool allow_graphs_under_offload = false;
    bool zero_workspace = false;
    bool no_shared_mlp = false;
    bool no_shexp_gate = false;
    bool no_cutlass3x = false;
    // Per-process MoE workspace reserve override (MiB). 0 = use computed
    // default.
    int reserve_mib = 0;
    // CUTLASS 3.x device-args full path for NVFP4 MoE prefill. Default ON
    // since 2026-05-14 (+11-39% pp512 on 4-model A/B).
    bool nvfp4_device_args = true;
    // Opt-in smallM kernel branch for NVFP4 MoE prefill.
    bool nvfp4_smallM = false;
    // Threshold M for smallM kernel (clamped to [0,128]).
    int nvfp4_smallM_threshold = 64;
    // Rows-per-block (NR) for multi-row NVFP4 MoE decode kernels
    // (gemv_nvfp4_moe_{gate_up,decode}_mr<NR>). One warp computes one
    // row, so threads-per-block = NR * 32. Higher NR amortizes block
    // launch overhead at the cost of fewer concurrent CTAs. Valid
    // values: 4, 8 (default), 16, 32. Other values fall back to 8.
    int mr_nr = 8;
};

struct GDN {
    bool fp32_scan = false;
    bool fp32_out = false;
    float norm_eps_override = 0.0f;  // 0 = use model default
    bool ref_kernel = false;
    bool vhead_reorder = false;
    // GDN chunkwise SSD scan refactor — Phase 1b.1 structural prototype.
    // When true,
    // the executor dispatches GDN scan through
    // `gdn_scan_chunkwise_{f32,fp32out}` (chunk-cached K/Q in shared
    // memory) instead of the per-token-loop `gdn_scan_fused_{f32,fp32out}`.
    // Bit-near-equivalent output (FP16 1e-3 / FP32 1e-5 tolerances per
    // Phase 1a); microbench shows +16.7 % on the GDN scan kernel alone
    // at n_tok=4096 (1.567 → 1.343 µs/tok on RTX 5090). Phase 4
    // cold-median A/B on Qwen3.6-35B-A3B Q4_K_M showed the end-to-end
    // wall delta is within the cuBLAS variance band (±0.5 % across
    // pp512 / pp2048 / tg128), so flipping the default on is wall-neutral
    // for the hero MoE model and unlocks the kernel-level win for
    // workloads where the GDN scan is a larger share of wall (longer
    // contexts, pure-GDN models like Qwen3.5-4B-GDN / Qwen3.5-9B-GDN
    // when bench data becomes available). Opt out via
    // `--set gdn.chunkwise_scan=false` if a model regresses.
    // After the Phase 2 ladder (2a / 2b / 2c, all shipped) was
    // exhaustively benched, Phase 1b.1 remains the fastest chunkwise
    // path on sm_120 — the WY-rep + TC-MMA variants all stay behind it.
    bool chunkwise_scan = true;
    // Override gated-DeltaNet weight layout.
    std::string layout_override;
};

struct GEMM {
    bool no_dp4a_gemv = false;
    bool no_dp4a_lm = false;
    bool no_mmvq = false;
    bool no_mmvq_q8_0 = false;
    // Q4_K x FP16 HMMA GEMM: in-SMEM nibble decode + FP16 tensor core
    // m16n8k16 tile kernel. Phase 0 scaffold (default off). When enabled,
    // prefill (M >= 32) Q4_K weights bypass dequant-to-FP16 + cuBLAS.
    bool q4k_hmma_enabled = false;
    // Q8_0 INT8 IMMA prefill GEMM (mmq_q8_imma.cu): fused dequant on the
    // int8 tensor cores (s8.s8.s32 measured 968 TOPS — full rate, unlike
    // the quartered f32-accumulate paths). Replaces the dequant-to-FP16 →
    // cuBLAS round-trip for Q8_0 prefill (M ≥ 64). Redesigned against the
    // Q4_K-IMMA phase-2B ceiling diagnosis (SMEM-staged scales, 128x128x64
    // tiles, symmetric epilogue). Default on.
    bool q8_imma_enabled = true;
    // Q4_K dense prefill via the (new-stack) IMMA kernel: uses
    // mmq_q4k_imma_reorder's symmetric-s8 + α/β form with the unified
    // β·rowsum epilogue. Experimental: default off. (Distinct from the
    // retired 2026-05 64x32 q4k_imma kernel that plateaued at 40 TOPS.)
    bool q4k_imma_prefill = false;
    // MoE batch prefill via the grouped IMMA kernel (one launch over all
    // experts, gridDim.z = expert, BM=32 small-M tile for the typical
    // ~32-rows-per-expert routing at pp512). Covers Q8_0/Q4_K expert
    // tensors; others (Q6_K down_proj) stay on dequant→cuBLAS. This is
    // lever #1 for the 2.4-2.6x GGUF-MoE prefill gap
    // (docs/archive/prefill_gap_2026_06_07.md §4.2). Default on.
    bool moe_imma_prefill = true;
    // Extend NVFP4 decode cache to ALL quantized types (Q4_K, Q3_K, etc.),
    // not just the default Q8_0/Q6_K/Q5_K set. Trades VRAM for decode
    // throughput on sub-8-bit models (e.g. Gemma-3-12B Q4_K_M: dp4a GEMV
    // at 130 tok/s → NVFP4 kpar GEMV target ~165 tok/s).
    bool nvfp4_decode_all = false;
    // Quantize a native-precision (FP16/BF16) LM head to an NVFP4 decode
    // cache. Native-NVFP4 checkpoints (llm-compressor/Modelopt) store
    // lm_head in BF16, so decode pays a cuBLAS FP16 GEMV over the
    // vocab×d_model matrix (~0.78 ms/token, ~19% of decode on Qwen3-8B).
    // The GGUF path already NVFP4-caches a Q*_K/Q8_0 output_proj; this
    // extends the same win to native-NVFP4 dense models. Excluded for
    // GDN/SSM-hybrid models (LM-head NVFP4 degrades recurrent-state
    // quality — see memory lm_head_only_nvfp4_qwen3_6_refuted).
    // "auto" (#982 net rule) | "on" | "off" (legacy true/false accepted).
    // auto = ON for native BF16/F16 heads (4x byte win, +8-16% decode,
    // +2.2% PPL, GOAL-listed trade) and for small dense GGUF heads
    // (d_model <= 4096: 4B/8B measured net-positive); OFF for larger or
    // MoE GGUF heads where the 2026-07-12 parity sweep measured the PPL
    // cost above the decode win (14B +1.9%/+2.1%, 30B-A3B +3.7%/+5.0%).
    std::string nvfp4_lm_head = "auto";
    // FP16-accumulate cuBLAS prefill GEMMs (CUBLAS_COMPUTE_16F instead of
    // 32F). GeForce sm_120 runs FP16 tensor cores with FP32 accumulate at
    // 1/4 rate (measured 2026-06-07: 253 vs 1956 TFLOPS saturated
    // mma.sync); the cuBLAS 32F prefill GEMMs sit at ~225 TFLOPS — ~89% of
    // that quarter-rate ceiling, so the kernel is fine, the compute type
    // is the cap. 16F measured +24.9% q8 pp512 model-level (2026-06-07,
    // paired same-day restarts), decode neutral. "auto" (default) enables
    // it per-arch at engine init: ON except GEMMA3/GEMMA4 (measured +0.7%
    // PPL on gemma-3-12b) and GPT_OSS (known FP16-residual-overflow
    // sensitivity — f16 accumulators are the same hazard class). "on"
    // forces it everywhere, "off" restores 32F accumulate. Legacy bool
    // values (true/false/1/0) parse as on/off. Applies only to
    // F16xF16→F16 with M>1 (prefill); decode GEMV and mixed-precision
    // paths are untouched. Standalone tools that skip engine init treat
    // "auto" as off.
    std::string cublas_fp16_acc = "auto";
    // Allow NVFP4 LM head even on GDN/SSM-hybrid models (normally excluded —
    // an older NVFP4 method degraded recurrent-state coherence, memory
    // lm_head_only_nvfp4_qwen3_6_refuted). Quantified 2026-05-29 with the
    // current quantize-FP16→NVFP4 path + the new `imp-cli --perplexity` tool
    // on Qwen3.6-35B: decode +11.4% (219.6→244.7 tok/s; the 248k-vocab
    // lm_head is ~14% of decode) at a small but REAL quality cost —
    // perplexity 15.90 (FP16) → 16.25 (NVFP4), +2.2% (FP16 PPL is stable
    // run-to-run, so it's signal not noise). Default ON: the +11.4% decode
    // gain serves the primary mission metric (best batch=1 tok/s on the
    // 5090) and the +2.2% PPL cost is small; set false to keep the FP16
    // lm_head for maximum coherence. gemm.nvfp4_lm_head=false still kills
    // the NVFP4 lm_head entirely (dense + GDN).
    bool nvfp4_lm_head_gdn = true;
    // Batched-decode (n>1) LM head via a single CUTLASS NVFP4 tensor-core
    // GEMM instead of the FP16-activation batched-M GEMV. Reads the LM-head
    // weight once for the whole batch (vs ceil(n/4)x for the GEMV), but the
    // FP4×FP4 MMA forces NVFP4 activations on the final logits (the GEMV kept
    // FP16 activations) — a quality/speed trade. Costs ~vocab*d_model/16 B of
    // SfAtom scales (FP4 data borrowed from the decode cache; only allocated
    // when max_batch_size > 1). Default ON since the 2026-07-12 PPL sweep
    // (teacher-forced, 13.5k-token corpus, imp-cli --perplexity): MoE/hybrid
    // +1.9-2.1% PPL (Coder-30B 10.19→10.38, Modelopt-30B 11.65→11.88,
    // Qwen3.6-35B 13.39→13.66), dense +0.2-0.5% (within the ±0.3-0.5%
    // run-to-run spread), for +8% aggregate concurrent decode @16 — the
    // 1173-tok/s Coder-30B headline. Single-stream (n==1) output is
    // bit-identical either way: the n==1 decode GEMV and the spec-verify
    // LM head (for_each_lm_head_batch_ allow_cutlass=false) never take
    // this path. Set false for maximum batched-serving coherence.
    bool nvfp4_lm_head_cutlass = true;
    // (gemm.nvfp4_ssm_proj — the 2026-05-30 opt-in that forced GGUF-hybrid
    // GDN projections into the NVFP4 decode cache — was REMOVED 2026-07-11:
    // it had bit-rotted in the tier refactors (measured 71 tok/s vs its
    // original 248 on Qwen3.6-35B Q4_K_M) and is superseded by the GGUF
    // branch of fp8_ssm_proj below, which is faster than the flag ever was
    // and quality-safer than 4-bit into the recurrent scan. The recurrent
    // in_proj/out_proj stay OUT of the NVFP4 decode cache unconditionally:
    // they feed the GDN/SSM scan, which accumulates quantization error in
    // the state H across tokens.)
    // Native-NVFP4 hybrid models store SOME projections BF16 because the
    // Modelopt/llm-compressor recipe excluded them from NVFP4. At decode these
    // run as FP16 GEMVs (gemv_fp16_kernel). This opt-in quantizes the
    // recipe-excluded BF16 **attention q/k/v/o** to an NVFP4 decode-cache
    // entry at init (direct quantize_fp16_to_nvfp4_async, mirroring
    // nvfp4_lm_head_gdn). q/k/v/o are stateless within a step → low quality
    // risk. MEASURED (2026-05-30): Nemotron-3-Nano-30B **+3.8% decode**,
    // perplexity-neutral (within noise). No-op on models whose attention is
    // already NVFP4 (e.g. Qwen3.6-35B). Default false.
    //
    // NOTE: the analogous lever for the BF16 GDN/Mamba in_proj/out_proj was
    // built and MEASURED to REGRESS decode −9% (Nemotron) to −20% (Qwen3.6) —
    // the tuned FP16 GEMV (70-81% HBM) beats the NVFP4 GEMV for the wide
    // GDN-output shapes, so the bandwidth saving never materializes. Keeping
    // those projections FP16 is correct for SPEED (not just quality); no flag
    // is provided for them. (See docs/MTP/SafeTensors profiling notes.)
    bool nvfp4_attn_proj = false;
    // FP8 (E4M3, per-tensor amax scale) DECODE sidecar for the native-
    // precision GDN/Mamba in_proj/out_proj that the NVFP4 lever above
    // regresses on: FP8 halves the FP16 GEMV bytes with byte-aligned dense
    // loads (none of the 4-bit packing overhead that made NVFP4 lose to the
    // tuned FP16 GEMV on the wide GDN shapes), and its quality risk into
    // the recurrent scan is far smaller than 4-bit. Prefill and the M>1
    // verify chunks keep the resident FP16 source (quality); only the M=1
    // decode GEMV dispatches the FP8 copy (gemv_fp8_rowscale, per-row
    // scales — one per-tensor scale over the heterogeneous GDN input pack
    // cost +4% PPL; per-row is PPL-flat). Costs +0.5 byte/elem VRAM for
    // the sidecar copies.
    // GGUF hybrids: also covers Q8_0-source ssm_in/gdn_gate/ssm_out
    // (UD-Q4_K_M keeps exactly these at Q8_0). Those handles were in no
    // decode cache at all (phase-3 quality lock) → every decode token
    // paid a full dequant→cuBLAS round-trip; the FP8 copy is byte-neutral
    // vs Q8_0 but runs the tuned rowscale GEMV. Sub-8-bit sources are
    // excluded (FP8 would increase decode bytes and stack rounding on a
    // coarse lattice). No-op on 27B-class checkpoints whose SSM
    // projections are already native NVFP4.
    // MEASURED (2026-07-10): Qwen3.6-35B-A3B-NVFP4 decode +19% (268.6→320.3
    // tok/s spec-off, 261→308 with default spec), PPL flat (8.021→8.012);
    // Nemotron-3-Nano PPL flat (4.184→4.117).
    // MEASURED (2026-07-11, GGUF branch): Qwen3.6-35B UD-Q4_K_M decode +21%
    // (224.4→272.0 defaults, 219.2→265.9 spec-off), PPL 4.215→4.289 (+1.8%
    // on the 201-token corpus — E4M3 stacked on the Q8_0 lattice; an
    // accepted trade like nvfp4_lm_head_gdn, set false to revert),
    // degen_suite 33/33. Default ON.
    bool fp8_ssm_proj = true;
    // FP8 decode sidecar for FULL-PRECISION attention projections
    // (wq/wk/wv/wo), same per-row-scale mechanism as fp8_ssm_proj:
    // decode-only (M=1 GEMV), prefill keeps the full-precision source.
    // "auto" = ON only for gpt-oss (#984): its BF16 SafeTensors dense
    // weights get no NVFP4 decode cache (nvfp4_beneficial is GGUF-only),
    // so q/k/v/o decode as 2 B/elem FP16 GEMVs = 33.5% of the decode
    // window at ~1.1 TB/s — halving the bytes is ~halving that time.
    // Other arches stay off pending a PPL gate (the phase-2 comment's
    // attn-weight precision concern was measured on PREFILL compounding;
    // decode-only per-row FP8 is the #949 recipe).
    std::string fp8_attn_proj = "auto";
    // Route native-NVFP4 (Modelopt/llm-compressor) MoE expert DECODE (M=1)
    // through the fast per-expert gemv_nvfp4_moe kernels by borrowing the
    // already-resident contiguous expert data + scales, instead of the
    // CUTLASS grouped-GEMM (which under-utilizes the GPU at M=1). +54-80%
    // MoE decode on Qwen3-30B-A3B / Coder-30B / Gemma-4-26B. Prefill stays
    // on CUTLASS.
    bool nvfp4_moe_decode = true;
};

struct Generation {
    bool no_logit_softcap = false;
    bool lm_dequant_fp16 = false;
    bool force_bos = false;
    // Disable banned-token list (debug).
    bool no_ban = false;
    // Disable RoPE inside the MTP draft head (diagnostic).
    bool mtp_no_rope = false;
};

struct Speculative {
    bool ngram = true;  // prompt-lookup speculation, default-on (batch-1, greedy, dense)
    // Context cap for DENSE chunk-verify drafting (#964). With the
    // verify_decode_attn route below (2026-07-12), dense n-gram WINS up
    // to at least 12k context on the captured-verify (server) path:
    // Qwen3-8B Q8_0 route+capture vs spec-off +44% @512, +43% @2k,
    // +25% @8k, +29% @12288 — but still −23% at 15872 (the near-msl
    // band: the capture-ready ceiling ends there and the remaining
    // verify cost is the weight-bound small-M GEMMs + per-step host
    // work, not attention). Drafting is gated once a request's context
    // crosses the cap (checked per step). MoE-NVFP4 and GDN-hybrid
    // requests are exempt (deep drafts: Coder-30B code-edit 15.9
    // tok/verify, MTP chains — the verify pays for itself there).
    // History: pre-route the FA2-tile verify was net-negative past ~2k
    // (−62% @16k, 2026-07-11) and the cap defaulted 2048; the route
    // (verify_decode_attn) then raised it to 12288. Superseded 2026-07-12
    // by the DEPTH-aware gate below (shallow_draft_ctx): the residual
    // long-context loss was never a context cost — it was draft depth
    // (a verify step costs ~1.4x a decode step @512 rising to ~2.6x
    // @16k, so 1-token drafts stop paying past ~14k while 3-token
    // drafts win +24% at 13k). Hard cap now defaults OFF. 0 = no cap.
    int draft_ctx_cap = 0;
    // #964 stage 2: past this request context, 1-token n-gram/suffix
    // drafts are discarded instead of verified (the miss-burst path
    // serves the step at plain decode speed); drafts of depth >= 2 are
    // always verified. Bench evidence (Qwen3-8B Q8_0, route+capture):
    // depth-3 drafts +24% @13312, depth-1 drafts −23% @15872 — the
    // break-even for depth 1 sits near 14k where verify/decode cost
    // reaches ~2x. MoE-NVFP4 and hybrid requests are exempt (deep
    // drafts). 0 = never gate.
    int shallow_draft_ctx = 12288;
    // #964 structural fix: run the dense verify chunk's ATTENTION through
    // the batched-decode split-K paged kernels (rows become same-KV
    // "sequences" with per-row context lens p0+1+i — causality holds by
    // construction; pad rows attend 1 token) instead of the small-M
    // prefill FA2 tile + full-context FP16 KV gather. nsys @16k (Qwen3-8B
    // Q8_0): 557 us/layer FA2 vs ~65 us/layer split-K — ~20 ms of the
    // ~30 ms verify step, and the reason FP8 KV never moved the verify
    // cost (the old path gathered to FP16 first). Composes with the
    // captured verify; requires the 3/5 capture buckets (the split count
    // bakes from the PADDED row count). Measured end-to-end (route +
    // capture vs spec-off, 3 trials): +44% @512, +43% @2k, +25% @8k,
    // +29% @12288. Dense non-MLA/non-SWA/non-MoE/non-hybrid only;
    // MoE/hybrid keep the FA2 chunk path. Kill switch for A/B.
    bool verify_decode_attn = true;
    // #998: verify-chunk GEMMs (M <= largest capture bucket) read the
    // NVFP4 decode overlay in one weight pass per MR tile instead of the
    // M>1 prefill dequant path. On GGUF K-quants without a direct
    // small-M kernel (Q6_K) the per-chunk source dequant cost ~7x a
    // decode step (dequant = 52% of the tg window at ctx 2048 on
    // Qwen3-14B Q6_K, tg -39% vs spec-off). Also aligns verify argmax
    // with the decode path (same weights). Real prefills are never
    // affected. Kill switch for A/B.
    bool verify_nvfp4_gemm = true;
    // Speculation on MoE models with NATIVE-NVFP4 experts (the gate
    // additionally requires profile().moe_experts_nvfp4). Measured on
    // Qwen3-Coder-30B-FP4 (2026-07-02): code-edit +49-81% (93% accept,
    // 15.9 tok/verify), draft-poor code-gen -3-7% (miss_burst hybrid
    // bounds the downside). GGUF-MoE verify re-dequants every activated
    // expert per step (-22% measured) and never engages regardless of
    // this flag. imp-cli --bench pins this false so the canonical
    // perf-baseline decode signal stays raw (verify inherits grouped-GEMM
    // restart variance).
    bool moe = true;
    // #1003 stage 1: at decode batch > 1, ONE request per step may run
    // its spec verify (round-robin, cyclic id order) while the remaining
    // rows decode batched. Three guards make it pay: a draft-depth floor
    // (min_draft = 2x batch — the whole batch stalls for the verify, so
    // shallow drafts measured -10% unguarded), a pipeline yield (the
    // chained batched decode otherwise never gives a turn), and an
    // ADAPTIVE yield cadence (8 -> 64 steps exponential backoff on empty
    // turns; fruitless chain breaks alone cost -1.5..-2.7% before it).
    // Default-ON matrix (Coder-30B NVFP4, 2026-07-15, 2 trials/cell,
    // stream-free host): code-edit +7.9/+11.0% at batch 8 (twice,
    // non-overlapping trials), +2.6/+4.4% at 16; diverse chat neutral
    // (+2.2/+3.5% at 4/8, remaining negatives inside the +-6% intra-arm
    // trial spread). Batch-1 behavior unchanged. Kill switch for A/B.
    bool batch_rr = true;
    int k = 16;  // draft tokens per verify step (verify cost is ~flat in k)
    // Token-Recycling adjacency drafting (ACL 2025, arXiv 2408.08696;
    // plan docs/plans/2026-07-22-token-recycling-spec-tree.md):
    // engine-scoped cross-request `token -> top-M successors` table fed
    // from emitted bigrams and the model's own verify-chunk top-K
    // logits. Fires on unigram context, so it drafts on fresh
    // reasoning/agentic prose where suffix/n-gram matching finds
    // nothing (measured 0 drafts on reasoning, 2026-07-22/23). Runs as
    // a fallback AFTER suffix/n-gram and MTP; lossless via the greedy
    // argmax verify. Default off until the reasoning A/B proves it.
    bool token_recycling = false;
    int recycle_slots = 8;  // successors kept per token (MRU/rank order)
    // Linear draft length. Default 3 -> chunk 4 -> capture bucket 4 =
    // exactly one batched-GEMV weight sweep (M=4, #1055); deeper chains
    // pad into bucket 5+ and pay a second sweep.
    int recycle_depth = 3;
    // Precision gate (#1055): only draft hops whose front slot was
    // re-confirmed at least this many times (bigram repeated / model
    // top-1 stable). A verify step costs ~1.4x a decode step while a
    // miss just rides the async-loop burst — precision beats recall.
    // 0 = draft on any known successor.
    int recycle_min_streak = 1;
    // Multi-candidate verify (route (a) of the spec-tree plan): when the
    // decode-attn route is available (dense, non-MLA/SWA/MoE/hybrid, no
    // penalties/MTP), verify `recycle_width` adjacency candidates in one
    // chunk — each candidate gets its own t0 row and private copies of
    // the KV blocks its rows write (per-row block tables), so no token
    // mask is needed and the argmax accept stays lossless. Rows are
    // capped at the 17 bucket (width * (1+depth) <= 17), i.e. width 4
    // -> depth 3. Default 1 (linear): at bucket 17 the chunk GEMMs run
    // CUTLASS at ~51% effective bandwidth (measured 2026-07-23) — the
    // multi-candidate accept lift (2.0 vs 1.44 emitted/verify) does not
    // pay for the 2x-costlier chunk; linear bucket-4 chunks ride the
    // single-sweep batched GEMV instead.
    int recycle_width = 1;
    // SuffixDecoding-style indexed drafting (arXiv 2411.04975):
    // hash-indexed suffix matching (O(1) amortized vs the legacy O(n)
    // backward scan per verify step) with frequency-voted continuations
    // across all occurrences, and adaptive draft length — a draft backed
    // by multiple agreeing occurrences or a maximal-length (max_match)
    // context match extends past `k` up to `suffix_k_max`. false =
    // legacy single-most-recent scan.
    bool suffix = true;
    int suffix_k_max = 64;
    // Longer suffix matches trade draft frequency for precision — and
    // precision wins decisively: min_match 6 vs 3 measured +16% on
    // code-edit (50% acceptance) while cutting the structured-content
    // worst case from -13% to -2% (false 3-gram matches in number
    // tables produce drafts that never verify).
    int min_match = 6;   // shortest accepted suffix n-gram match
    int max_match = 12;  // longest suffix extension searched
    // After this many consecutive draft misses the request gives up on
    // speculation and re-enters the async conditional graph loop (the
    // eager per-token path costs ~2x vs the loop — a draft-poor context
    // must not pay that for the whole generation). 0 = never give up.
    int give_up_after = 64;
    // Burst-hybrid: while given up, the async loop runs in bursts of
    // this many tokens; after each burst the request re-probes drafts
    // for a couple of steps (think models produce their draft-rich
    // region only after the reasoning prose). 0 = give-up is final.
    int burst = 128;
    // On a draft miss the request falls back to the async loop for this
    // many tokens (cheap rearm, no graph recapture) instead of paying
    // the ~2x eager per-token tax until the next draft shows up.
    // 0 = stay eager between drafts (legacy behavior).
    int miss_burst = 8;
    // Reuse the parked captured graph across bursts (rearm instead of
    // recapture, ~10-20 ms saved per burst). The #683 wrong-token
    // artifact was NOT the rearm itself but the fresh-captured loop
    // initializing position/context one too high (fixed in
    // CudaGraphConditionalRunner::setup) — rearm and fresh capture now
    // share the same first-forward semantics.
    bool burst_rearm = true;
    // Speculation on hybrid (GDN/SSM) models. The verify chunk advances
    // recurrent state through rejected draft positions, so the committed
    // per-sequence state slab is snapshotted before the chunk; a fully
    // accepted draft pays only that copy (~60 MiB D2D on Qwen3.6-35B),
    // a partial acceptance restores the slab and re-forwards the
    // accepted prefix (one extra chunk forward, amortized over the
    // accepted tokens). imp-cli --bench pins this false (same
    // baseline-semantics rule as the moe/suffix pins).
    bool hybrid = true;
    // MTP-head chain-draft length for the verify loop (models shipping a
    // trained MTP head, e.g. Qwen3.6 model_mtp.safetensors). 0 = off.
    // When >0 the server loads the head (~1.6 GiB VRAM) and enables MTP;
    // drafts fill verify steps where the suffix/ngram matcher has no
    // match (draft-poor prose — 78-94% depth-1 accept on Qwen3.6-35B-A3B,
    // PR #804). imp-cli equivalent: --mtp-spec-decode <k>.
    int mtp_k = 0;
    // Serve the MTP chain's full-vocab logits GEMV from the NVFP4 LM-head
    // decode cache when one exists (#847 lever 3). The chain re-reads the
    // LM head once per drafted token (~2.5 GB FP16 on Qwen3.6-27B's 248k
    // vocab — more traffic than the main forward at k=4); NVFP4 reads ~4x
    // less. Draft-only precision, verification stays lossless. Off = keep
    // the FP16 chain GEMV (A/B kill switch).
    bool mtp_nvfp4_head = true;
    // MTP economics guard: after an 8-verify sample, average emitted
    // tokens per MTP-filled verify below this dooms MTP drafting for the
    // request (the verify chunk + chain cost can't amortize). 0 disables
    // the guard (raw-economics measurement). Default from the #852
    // break-even measurement; re-derive when chain/verify costs change —
    // note a chain of k can emit at most k+1 per verify, so values >= k+1
    // doom that k unconditionally.
    float mtp_econ_min_emit = 4.0f;
    // Graph-captured verify chunk (#847): cache one CUDA graph per
    // draft-length bucket and replay it each verify step — the chunk
    // metadata and KV lengths are read from device buffers, so the graph
    // survives context growth. Removes the eager launch-pacing tax
    // (~1800 launches/verify cycle). Drafts are padded up to the bucket
    // length (extra rows are causally invisible to the real rows and
    // their KV is rolled back with the rejected drafts). Engages only
    // where the FP16-QK FA2 kernel serves the chunk (uniform hd=128, no
    // sinks/MLA/LongRoPE) on non-hybrid models; anything else stays
    // eager. Any capture failure falls back to the eager verify and
    // disables capture for the process after repeated failures.
    bool capture = true;
    // Context capacity the captured gather grids and the persistent K/V
    // scratch are sized for (2 x ctx_cap x nkv x hd x 2B VRAM, e.g. 2x
    // 32 MiB at 32k for nkv=4/hd=128). Verify steps whose context
    // exceeds this run eager. Clamped to the model's max_seq_len.
    int capture_ctx_cap = 32768;
};

struct FFN {
    // SwiGLU/GeGLU sparsity probe (instrumentation-only — no skipping).
    // When enabled, every dense-FFN decode step runs a reduce kernel
    // that counts, for each of 5 hard-coded thresholds {0.005, 0.01,
    // 0.02, 0.05, 0.1}, the number of intermediate-dim rows i with
    // |silu(gate[i]) * up[i]| < t. Per-layer counters accumulate
    // across all generations of the process and are flushed via
    // imp::flush_ffn_sparsity_probe_log() (engine destruction or
    // explicit call). Purpose: measure the upside of contextual FFN
    // sparsity on this model class before writing a single gather
    // kernel. ~1 µs overhead per layer per token when on; zero when
    // off. Default off.
    bool sparsity_probe = false;

    // Phase 2 — actual FFN row-skipping in down_proj via per-block mask.
    // For each Q8_0-block of K (=32 elements) compute amax of
    // |silu(gate)*up|; if amax < threshold the whole 34-byte Q8_0
    // weight block is skipped (no HBM load) in the down_proj GEMV.
    // 0.0 = disabled = bit-identical to baseline. Recommended range
    // 0.005..0.05; per-layer sparsity see ffn_sparsity_probe data.
    // Only active for Q8_0 down_proj decode (n=1) today; other dtypes
    // fall through to the unmasked dispatch automatically.
    float sparsity_threshold = 0.0f;
};

struct Diagnostics {
    // Process log level: "debug" | "info" | "warn" | "error" | "fatal".
    // Applied by process_diag_install(), which runs from both tool mains
    // AND Engine::init, so a C-API consumer reaches it too.
    // Until 2026-08-03 there was no way to set this at all: log_set_level()
    // was the only writer of g_log_level and nothing called it, so the level
    // was pinned at INFO and all 76 IMP_LOG_DEBUG sites were unreachable —
    // a debug facility that could not be switched on.
    std::string log_level = "info";
    bool debug_forward = false;
    bool debug_template = false;
    std::string dump_hidden_dir;
    std::string dump_logits_dir;   // path or empty
    std::string dump_routing_dir;  // path or empty
    // Path for the per-layer MoE expert-activation histogram (JSON), written at
    // executor teardown. Empty = off. Unlike dump_routing_dir, which logs one
    // token's top-k as a DEBUG line, this counts EVERY routing decision of the
    // run — the dataset the resident/host expert-split question needs.
    std::string moe_expert_hist;
    // Path for a per-token MoE expert TRACE (JSON), written at executor
    // teardown. Empty = off. The histogram above is an aggregate and cannot see
    // temporal locality, which is the whole question for a cache: an LRU pays
    // only if an expert selected now is selected again soon. Decode only (n==1),
    // so each record is one (token, layer).
    std::string moe_expert_trace;
    bool dump_tokens = false;
    // Teacher-forced perplexity: restrict the NLL sum to logit rows
    // i in [ppl_first, ppl_last] (row i predicts token i+1); ppl_last=-1
    // means "through the end" (n-2). Matches llama-perplexity's window
    // (`first = n_ctx/2`, rows [first, n_ctx-2]) when llama.cpp runs the
    // same corpus with `-c C --chunks 1` (llama-perplexity refuses
    // single-chunk-over-everything: it wants >= 2*n_ctx total tokens):
    // set ppl_first = C/2, ppl_last = C-2 (+ token-offset if the streams
    // are BOS-shifted). The cross-engine PPL-parity bar (GOAL release
    // bar 1) is measured this way.
    int ppl_first = 0;
    int ppl_last = -1;
    int exit_layer = -1;
    bool profile = false;
    bool graph_diag = false;
    // Trace knobs that used to be raw getenv() reads in the hot path
    // (#1207). CLAUDE.md's rule is that IMP_DETERMINISTIC and IMP_FMHA_FA2
    // are the only seeded env vars; IMP_SPEC_TRACE / IMP_JUMP_TRACE /
    // IMP_PPL_DUMP had crept back in. The env names still work — they are
    // debug aids people have in their shell history — but they are seeded
    // into these keys at load, so `--set` and imp.conf reach them too and
    // `imp-cli --help`/imp.conf.example can document them.
    bool spec_trace = false;  // per-step speculative draft/verify trace
    bool jump_trace = false;  // conditional-graph jump trace
    std::string ppl_dump;     // path: dump per-token NLL from --perplexity
    std::string graph_dump_dir;
    // Force NVFP4 dispatch through dequant->FP16 GEMV (M=1 bisection
    // tool — see Mistral-Small-3.2-NVFP4 long-form repetition loops).
    bool nvfp4_force_dequant = false;
    // Skip building the NVFP4 decode cache entirely (bisection/eval
    // tool): decode runs on the source-precision paths (dp4a GEMV for
    // GGUF quants, FP16 GEMV otherwise) — the pre-cache decode
    // semantics. Distinct from nvfp4_force_dequant, which dequantizes
    // the already-NVFP4-quantized cache and so keeps NVFP4 values.
    bool no_nvfp4_decode_cache = false;
    // #847 graph-captured-verify feasibility probe: stream-capture every
    // spec verify chunk forward, instantiate + launch the graph (falls
    // back to the eager forward on any failure). Logs per-attempt
    // outcomes — a capturability census, not a perf path.
    bool spec_capture_probe = false;
    // Log shape + per-candidate algoId/tileId + chosen algo for every
    // benchmark_and_select_algo call.
    bool log_gemm_algo = false;
    // MTP pattern logging (predicted, actual, match per step).
    bool mtp_pattern_log = false;
    // MTP: pass main model's post-RMSNorm hidden to draft head (vLLM
    // variant).
    bool mtp_prenorm_h = false;
    // Audit NVFP4 weight scales at load time.
    bool audit_nvfp4_scales = false;
    // Per-component VRAM accounting harness (MemAccount): lifecycle
    // checkpoints + per-pool notes + device-used peak sampler. Default off
    // (zero overhead). See src/memory/mem_account.h.
    bool vram_audit = false;
    // Optional append-only file the VRAM audit table is mirrored into.
    std::string vram_audit_dump;
    // [RETIRED] tq_skip_qjl removed in Phase 5 (TurboQuant retired 2026-05-17).
};

}  // namespace cfg

// The sections exec/ needs, together — a distinct type rather than a handle
// on RuntimeConfig, so core/ never depends on runtime/.
struct DispatchPolicy {
    cfg::KVCache kv_cache;
    cfg::Attention attention;
    cfg::MoE moe;
    cfg::GDN gdn;
    cfg::GEMM gemm;
    cfg::Generation generation;
    cfg::Speculative speculative;
    cfg::FFN ffn;
    cfg::Diagnostics diagnostics;
};

}  // namespace imp
