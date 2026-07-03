#pragma once

// imp.conf — central runtime configuration.
//
// Replaces ~50 ad-hoc IMP_*-prefixed environment variables that were scattered
// over ~80 getenv() call sites in src/runtime/ and src/exec/. The same values
// now flow through a single RuntimeConfig struct loaded once at startup,
// optionally from a TOML file, with CLI-flag overrides on top.
//
// Loading precedence (first non-empty wins):
//   1. --config <path>              CLI flag (passed via load_with_path)
//   2. $IMP_CONFIG                  environment variable
//   3. ./imp.conf                   working-dir relative
//   4. ~/.config/imp/imp.conf       user config directory
//   5. embedded defaults            (no file)
//
// Per-run overrides come on top via apply_overrides({"section.field=value"}).

#include <string>
#include <vector>

namespace imp {

struct RuntimeConfig {
    struct Runtime {
        bool deterministic_gemm = false;
        // Opt-in full reproducibility mode for temperature=0 agent evals.
        // When true, run-to-run non-determinism in MoE token routing
        // (atomic expert-bucket scatter ordering) and top-k sampling
        // (atomicMax/atomicAdd softmax-stat races) is eliminated by
        // selecting deterministic kernel variants. It ALSO implies
        // deterministic_gemm (timing-based cuBLAS algo selection is itself
        // a non-determinism source), so a single switch covers GEMM +
        // routing + sampling. Costs a little throughput (serial / ordered
        // reductions), so it is strictly OFF by default — the default code
        // path runs the exact same kernels as before with zero overhead.
        // Legacy env: IMP_DETERMINISTIC=1. See audit B-9
        // (docs/audit/performance_agent_readiness_2026_05_31.md).
        bool deterministic = false;
        std::string cuda_graphs = "auto";  // "auto" | "always" | "never"
        bool warmup = false;               // opt-in for prod rollout; off in dev/CI
        int max_seq_len = 0;               // 0 = use model default
        // Hard VRAM budget for THIS process (MiB, 0 = uncapped). Every sizing
        // decision (weight caches, KV clamp, expert offload, workspaces,
        // upload gates) sees a virtual GPU of this size, so multiple
        // imp-server processes can share one card without overcommitting it.
        // Best-effort: leave ~1 GiB real headroom between the sum of budgets
        // and the card (small fixed buffers + cuBLAS internals sit outside).
        // CLI flag --vram-budget / C-API ImpConfig.vram_budget_mb override.
        int vram_budget_mb = 0;
        bool no_pdl = false;
        bool debug_raw = false;        // raw stream debug
        bool no_vision_graph = false;  // disable SigLIP graph capture
        // cudaStreamCaptureMode passed to begin_capture / conditional bodies:
        // "global" | "relaxed" (default) | "thread_local". "relaxed" drops the
        // cross-thread sync constraint that CUTLASS 3.x grouped-GEMM
        // collective scheduler is suspected to deadlock on under prefill
        // capture (Blocker B in prefill_graph_blockers_2026_05_14). Default
        // flipped to "relaxed" 2026-05-16 as the M3-probe for prefill_graph
        // unblock — `cudaStreamCaptureModeRelaxed` is a strict superset of
        // capturable behaviors so the decode fast path that previously
        // worked under "global" continues to work, while the prefill path
        // gets a real chance of capturing without hanging.
        //
        // Set graph_capture_mode = "global" via imp.conf to opt back into
        // the legacy strict mode (any decode regression should also be
        // investigated under "thread_local" before assuming relaxed is the
        // cause). Legacy env: IMP_GRAPH_CAPTURE_MODE.
        std::string graph_capture_mode = "relaxed";
        // Capture prefill into a CUDA graph (in addition to decode). Default
        // flipped 2026-05-17 after the M3 Phase 4 A/B sweep across
        // Gemma-4-26B-NVFP4 / Qwen3.6-35B-NVFP4 / Qwen3-Coder-30B-FP4
        // (3 trials × 4 capture modes each, harness at
        // /tmp/imp-bench-results/run_bench.sh): no hang on any
        // (model, capture_mode) combination — Blocker B
        // (`prefill_graph_blockers_2026_05_14.md`) is gone now that
        // `graph_capture_mode = "relaxed"` is the default. Decode tg
        // is flat ±1-2% across all four capture-mode configs for every
        // model; prefill pp is variance-dominated (cuBLAS algo-selection
        // noise documented in CLAUDE.md) but the candidate (relaxed)
        // never regressed below baseline. Opt out via
        // `--set runtime.prefill_graph=false` or imp.conf if a model
        // regresses. Legacy env: IMP_PREFILL_GRAPH.
        bool prefill_graph = true;
        // 0 = auto: the engine sizes the decode batch from the model's weight
        // footprint (a >20 GiB MoE auto-picks 1). A positive value forces it.
        // (Was 4, which both contradicted the documented "0 = auto" semantics
        // and never reached engine sizing — it only acted as the decode cap.)
        int max_batch_size = 0;
        // Bound for the autonomous decode graph loop on a NON-streaming request
        // with speculation off (which would otherwise run UNBOUNDED to
        // max_tokens on-device, so a client disconnect/timeout — polled only
        // between bursts — couldn't interrupt it and burned a full generation).
        // The loop runs in bursts of this many tokens and returns to the host
        // to re-poll cancellation; output is identical (same decode, chunked).
        // Larger = less relaunch overhead but higher cancel latency. Streaming
        // and speculation paths are unaffected. <=0 restores the old unbounded
        // behavior. IGNORED when `deterministic` is set: the unbounded loop is
        // the only greedy-bit-reproducible decode path, and evals run to
        // completion (no mid-burst cancel needed), so determinism wins there.
        int decode_burst = 128;
        // Cap the prefill chunk while other sequences are DECODING: prefill
        // and decode share one stream, so every chunk forward inserts its
        // full latency between two of their decode steps. Measured (Qwen3-8B
        // Q8, 7.2k-token ingest against one active decoder): 2048 → decoder
        // p95 inter-token 164 ms; 1024 → 94 ms at +27% ingest TTFT; 512 →
        // 65 ms at +85%. 1024 is the default compromise; set 512 for
        // latency-critical multi-tenant serving, 0 to disable (full chunk).
        // The full chunk returns as soon as nobody is decoding.
        int prefill_chunk_decode_cap = 1024;
        // Hybrid (SSM/GDN) decode fairness: the recurrent scan kernels are
        // single-sequence, so concurrent sessions time-slice the decode.
        // This is the slice length in tokens — after it, the engine rotates
        // to the next DECODING request (round-robin). Rotation re-captures
        // the decode graphs for the new sequence's state slot (~10-20 ms),
        // so smaller values buy latency fairness at capture overhead
        // (128 ≈ 1-2% at typical hybrid decode rates). 0 restores the old
        // head-of-line behavior (first request runs to completion).
        int hybrid_decode_quantum = 128;
    } runtime;

    struct KVCache {
        // "auto" (default) keeps FP16 but upgrades to FP8 E4M3 for models whose
        // author declares kv_cache_quant_algo=FP8 AND whose arch family is
        // verified safe for long-context FP8 KV (see kv_fp8_hint_default_safe).
        // "fp16" forces FP16 (opt out of the hint). fp8|int8|int4|nvfp4|mxfp4
        // force that dtype regardless of the hint.
        std::string dtype = "auto";
        bool allow_nondeterministic_fp8 = false;
        bool fp8_auto_legacy = false;  // legacy IMP_KV_FP8_AUTO compat
        // BitDecoding Phase 3: residual FP16 cache for newest N tokens.
        // 0 = disabled (keeps Phase 1+2 behavior). Typical: 4..32.
        // Only meaningful with kv_cache.dtype = "nvfp4" + kv_cache.bitdecoding_qk.
        int bitdecoding_residual_tokens = 0;
        // BitDecoding TC path for NVFP4 paged attention QK. Legacy env:
        // IMP_USE_BITDECODING_QK.
        bool bitdecoding_qk = false;
    } kv_cache;

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
        // the fp8-QK path keeps f32 accumulate. Env: IMP_FA2_F16ACC.
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
        // fa2_f16acc. Env: IMP_FA2_PV_F16ACC.
        bool fa2_pv_f16acc = true;
        // amax-scaled e4m3 conversion for the fp8-QK FA2 path (#680). The
        // raw conversion is the #511 quality cliff; scaling Q and K to the
        // full e4m3 range is the numerics class FlashInfer runs. Only
        // takes effect on the fp8-QK path (fa2_fp16qk=never or declined).
        // Experimental quality probe. Env: IMP_FP8_QK_SCALED.
        bool fp8_qk_scaled = false;
        std::string mxfp4 = "auto";
        bool mxfp4_fp16_fallback = false;
        // MXFP4 → FP16 cache pruning policy. "legacy" (default) caches FP16
        // for every MXFP4 tensor. "pruned" skips MoE expert_*_packed and
        // LM head (out_proj_) — those slots are either not read on the
        // dispatch hot path (MoE expert FP16 cache is only consumed by
        // executor_forward_moe.cu's pre-cached FP16 fallback, which is
        // bypassed by the more efficient batch-dequant path for MXFP4)
        // or routed through generic-dequant (LM head). Pruning is the
        // Phase A1+A2 path from
        // docs/plans/qwen35_27b_mxfp4_host_dequant_design_2026_05_17.md —
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
        bool gate_concat = false;
        // Max VRAM (MiB) for the materialized cuBLAS-attention S-matrix. Caps the
        // prefill context length that uses the fast cuBLAS attention path before
        // falling back to FMHA (auto fmha_prefill_threshold = S-matrix cap + 1).
        // 256 MiB caps ~32-head models at seq 2048 but high-head-count models
        // (e.g. Qwen3-14B, 40 heads → ~1824) drop to the slower FMHA at 2048.
        // Larger = longer prefill on the fast path, at the cost of KV headroom.
        // Legacy env: IMP_ATTN_SCORES_MIB. Auto-shrinks if the alloc fails.
        // 384 keeps the fast cuBLAS attention path up to seq 2048 for up to
        // 48-head models (e.g. Qwen3-14B, 40 heads: +21% pp2048 vs the old 256
        // cap which dropped it to FMHA at ~1824). Only allocates what the
        // model's max_tokens×heads needs (capped here); +128 MiB vs 256 at most.
        int attn_scores_mib = 384;
    } attention;

    struct MoE {
        int expert_overhead_pct = 10;
        int force_host_experts = 0;  // last N layers forced to host (0 = none)
        bool skip = false;
        bool force_fp16_sync = false;
        bool no_expert_cache = false;
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
        // Phase 5 (CUDA Graphs under host-offload): drop the "experts on
        // host → graphs off" guard at engine.cpp:1158. Default false because
        // it is opt-in experimental — the dispatch path's host-side
        // get_or_load() captures cudaMemcpyAsync nodes with fixed (src host
        // ptr, dst slot) pairs that don't adapt to per-token routing changes.
        // Output is correct only when prefetch coverage matches router
        // selection 1:1 (workloads with extremely stable expert patterns).
        // Phase 5.1+ will refactor the dispatch kernels to read the device
        // mirror at runtime so captured graphs adapt correctly.
        bool allow_graphs_under_offload = false;
        bool zero_workspace = false;
        bool no_shared_mlp = false;
        bool no_shexp_gate = false;
        bool no_cutlass3x = false;
        // Per-process MoE workspace reserve override (MiB). 0 = use computed
        // default. Legacy env: IMP_MOE_RESERVE_MIB.
        int reserve_mib = 0;
        // CUTLASS 3.x device-args full path for NVFP4 MoE prefill. Default ON
        // since 2026-05-14 (+11-39% pp512 on 4-model A/B). Legacy env:
        // IMP_NVFP4_DEVICE_ARGS (0 disables).
        bool nvfp4_device_args = true;
        // Opt-in smallM kernel branch for NVFP4 MoE prefill. Legacy env:
        // IMP_NVFP4_SMALLM.
        bool nvfp4_smallM = false;
        // Threshold M for smallM kernel (clamped to [0,128]). Legacy env:
        // IMP_NVFP4_SMALLM_THRESHOLD.
        int nvfp4_smallM_threshold = 64;
        // Rows-per-block (NR) for multi-row NVFP4 MoE decode kernels
        // (gemv_nvfp4_moe_{gate_up,decode}_mr<NR>). One warp computes one
        // row, so threads-per-block = NR * 32. Higher NR amortizes block
        // launch overhead at the cost of fewer concurrent CTAs. Valid
        // values: 4, 8 (default), 16, 32. Other values fall back to 8.
        // Env: IMP_MOE_MR_NR.
        int mr_nr = 8;
    } moe;

    struct GDN {
        bool fp32_scan = false;
        bool fp32_out = false;
        float norm_eps_override = 0.0f;  // 0 = use model default
        bool ref_kernel = false;
        bool vhead_reorder = false;
        // GDN chunkwise SSD scan refactor — Phase 1b.1 structural prototype
        // (docs/plans/gdn_chunkwise_scan_design_2026_05_23.md). When true,
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
        // Override gated-DeltaNet weight layout. Legacy env: IMP_GDN_LAYOUT.
        std::string layout_override;
    } gdn;

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
        // tiles, symmetric epilogue). Experimental: default off.
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
        // (docs/audit/prefill_gap_2026_06_07.md §4.2). Default off.
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
        // quality — see memory lm_head_only_nvfp4_qwen3_6_refuted). Legacy
        // env: IMP_NO_NVFP4_LM_HEAD=1 to disable.
        bool nvfp4_lm_head = true;
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
        // lm_head for maximum coherence. Env IMP_NO_NVFP4_LM_HEAD=1 still kills
        // the NVFP4 lm_head entirely (dense + GDN) via gemm.nvfp4_lm_head.
        bool nvfp4_lm_head_gdn = true;
        // Batched-decode (n>1) LM head via a single CUTLASS NVFP4 tensor-core
        // GEMM instead of the FP16-activation batched-M GEMV. Reads the LM-head
        // weight once for the whole batch (vs ceil(n/4)x for the GEMV), but the
        // FP4×FP4 MMA forces NVFP4 activations on the final logits (the GEMV kept
        // FP16 activations) — a quality/speed trade. Costs ~vocab*d_model/16 B of
        // SfAtom scales (FP4 data borrowed from the decode cache). Opt-in until
        // the PPL trade is measured per family. Single-stream (n==1) is unaffected.
        bool nvfp4_lm_head_cutlass = false;
        // Hybrid GDN/SSM models (Nemotron-3-Nano-30B, Qwen3.6-35B-A3B) keep the
        // recurrent in_proj/out_proj (ssm_in/ssm_out) OUT of the NVFP4 decode
        // cache by default: they feed the GDN/SSM recurrent scan, which
        // accumulates quantization error in the state H across tokens, so 4-bit
        // was thought to degrade quality on 9B+ models. At decode they therefore
        // GGUF hybrid models (e.g. Qwen3.6-35B-A3B Q4_K_M): the GDN/SSM
        // in_proj/out_proj are excluded from the NVFP4 decode cache (Q4_K source
        // is not nvfp4_beneficial), so decode runs them via Q4_K→FP16→cuBLAS — a
        // memory-bound tax. This opt-in forces them into the NVFP4 decode cache.
        // MEASURED (2026-05-30): Qwen3.6-35B Q4_K_M **+53% decode** (161→248
        // tok/s), perplexity flat (−0.01%), coherent — reverses the documented
        // −31% GGUF-hybrid-decode loss vs llama.cpp. No-op on native-NVFP4 models
        // (their SSM projections are already NVFP4-cached). Default false.
        bool nvfp4_ssm_proj = false;
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
        // Route native-NVFP4 (Modelopt/llm-compressor) MoE expert DECODE (M=1)
        // through the fast per-expert gemv_nvfp4_moe kernels by borrowing the
        // already-resident contiguous expert data + scales, instead of the
        // CUTLASS grouped-GEMM (which under-utilizes the GPU at M=1). +54-80%
        // MoE decode on Qwen3-30B-A3B / Coder-30B / Gemma-4-26B. Prefill stays
        // on CUTLASS. Legacy env: IMP_NO_NVFP4_MOE_DECODE=1 to disable.
        bool nvfp4_moe_decode = true;
    } gemm;

    // (RuntimeConfig::Gemma4 lived here through Phase 4 of the architecture
    // refactor. Phase 5 Track A moved it to ModelConfig::Overrides::Gemma4 —
    // see src/model/model_config.h. Model-specific knobs do not belong on a
    // global runtime singleton.)

    struct Generation {
        bool no_logit_softcap = false;
        bool lm_dequant_fp16 = false;
        bool force_bos = false;
        // Disable banned-token list (debug). Legacy env: IMP_NO_BAN.
        bool no_ban = false;
        // Disable RoPE inside the MTP draft head (diagnostic). Legacy env:
        // IMP_MTP_NO_ROPE.
        bool mtp_no_rope = false;
    } generation;

    struct Server {
        // Prefix caching: reuse KV blocks for shared prompt prefixes. Default
        // ON for the server/CLI — this is the documented behaviour (README,
        // imp.conf.example) and what delivers the advertised warm-prompt TTFT
        // win + cache_read_input_tokens reporting (#758: shipping OFF meant the
        // prebuilt image never cached unless an imp.conf opted in). Library /
        // C-API embedders are unaffected — they drive EngineConfig directly
        // (off-by-default there). The engine ORs this into
        // EngineConfig.use_prefix_caching at init. PrefixCacheE2ETest is the
        // ship gate. For hybrid (SSM/GDN) models it additionally requires the
        // recurrent snapshot store below.
        bool prefix_cache = true;
        // Cap on cache_control/cache_prompt-pinned blocks, % of the KV pool.
        int prefix_pin_budget_pct = 25;
        // Device budget (MiB) for recurrent-state snapshots — what makes
        // prefix caching work on hybrid (SSM/GDN) models: KV blocks alone
        // cannot skip prefill there, the recurrent state at the skip boundary
        // must be restored too. One snapshot = one per-sequence state slab
        // (~64 MiB for Qwen3.6-35B), saved per prefill, LRU-evicted. Buffers
        // are pre-allocated at engine init (free VRAM is ~0 at serving time
        // by design) and accounted in the expert-offload reserve. imp-cli
        // --bench pins this to 0 (baseline semantics unchanged).
        // 0 disables snapshots AND hybrid prefix caching (dense unaffected).
        int recurrent_snapshot_mb = 256;
        // Green Contexts / prefill-decode overlap streams in the server engine.
        // OFF by default (suspected memSyncDomain race on sm_120 fallback
        // streams — gemma-3-12b IMA); opt in via [server] green_contexts = true.
        bool green_contexts = false;
    } server;

    struct Bench {
        bool generate = false;
    } bench;

    struct Paths {
        std::string mmproj;
    } paths;

    // n-gram (prompt-lookup) speculative decoding. Drafts come from suffix
    // matches against the request's own prompt+output tokens — no draft
    // model, no MTP head. Greedy-only Phase 1: the verify step replays the
    // draft as a teacher-forced continuation chunk and accepts the longest
    // argmax-matching prefix, so output is token-identical to plain greedy
    // decode. The verify loop runs eager (no async conditional graph loop);
    // burst_rearm + miss_burst keep draft-miss fragmentation ~free, so the old
    // tg128 -15% draft-poor downside no longer reproduces (-0.2%/-0.9% on
    // dense Q8/NVFP4, 2026-06-16) — hence default-ON. spec_ngram_gates_ok_
    // confines engagement to batch-1 / greedy / no-penalty-window / no-json /
    // no-logprobs / non-recurrent requests (MoE additionally requires
    // native-NVFP4 experts, see `moe` below); everything else falls back
    // cleanly, so default-on is a no-op for sampled chat, tool/JSON calls,
    // concurrent batches, and GGUF-MoE (which the async loop carries).
    struct Speculative {
        bool ngram = true;  // prompt-lookup speculation, default-on (batch-1, greedy, dense)
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
        int k = 16;          // draft tokens per verify step (verify cost is ~flat in k)
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
    } speculative;

    // Constrained decoding (json_mode / json_schema).
    struct Constrained {
        // Jump-ahead over schema-forced spans (#844): when the schema FSM
        // forces the next CHARACTERS (skeleton keys/punctuation — the text
        // is forced even though its tokenization is not), one speculative
        // chunk forward drafts the canonical tokenization and materializes
        // per-position logits rows; subsequent tokens are then sampled from
        // those rows without running forwards. Exact for greedy AND
        // sampling — each row is the true logits given the accepted prefix;
        // a token that diverges from the draft re-enters normal pipelining
        // (one wasted chunk forward, nothing else). OPT-IN: measured net
        // -3-5% on Qwen3-8B (Q8 + NVFP4, 2026-07-03) — the model picks
        // context-dependent tokenization splits the canonical draft misses,
        // so wasted chunks outweigh consumed rows. Also note the chunk path
        // is not bit-identical to per-token decode (prefill vs decode
        // kernels), so free-text AFTER a consumed span can diverge from a
        // jump-off run (same cross-path property as spec-ngram verify).
        bool jump_ahead = false;
        // Minimum draft length (tokens) worth the speculative chunk;
        // shorter forced spans stay on the per-token pipeline.
        int jump_min_run = 4;
    } constrained;

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
    } ffn;

    struct Diagnostics {
        bool debug_forward = false;
        bool debug_template = false;
        std::string dump_hidden_dir;
        std::string dump_logits_dir;   // path or empty
        std::string dump_routing_dir;  // path or empty
        bool dump_tokens = false;
        int exit_layer = -1;
        bool profile = false;
        bool graph_diag = false;
        std::string graph_dump_dir;
        // Force NVFP4 dispatch through dequant->FP16 GEMV (M=1 bisection
        // tool — see Mistral-Small-3.2-NVFP4 long-form repetition loops).
        // Legacy env: IMP_NVFP4_FORCE_DEQUANT.
        bool nvfp4_force_dequant = false;
        // Log shape + per-candidate algoId/tileId + chosen algo for every
        // benchmark_and_select_algo call. Legacy env: IMP_LOG_GEMM_ALGO.
        bool log_gemm_algo = false;
        // MTP pattern logging (predicted, actual, match per step). Legacy
        // env: IMP_MTP_PATTERN_LOG.
        bool mtp_pattern_log = false;
        // MTP: pass main model's post-RMSNorm hidden to draft head (vLLM
        // variant). Legacy env: IMP_MTP_PRENORM_H.
        bool mtp_prenorm_h = false;
        // Audit NVFP4 weight scales at load time. Legacy env:
        // IMP_AUDIT_NVFP4_SCALES.
        bool audit_nvfp4_scales = false;
        // Per-component VRAM accounting harness (MemAccount): lifecycle
        // checkpoints + per-pool notes + device-used peak sampler. Default off
        // (zero overhead). See src/memory/mem_account.h.
        bool vram_audit = false;
        // Optional append-only file the VRAM audit table is mirrored into.
        std::string vram_audit_dump;
        // [RETIRED] tq_skip_qjl removed in Phase 5 (TurboQuant retired 2026-05-17).
    } diagnostics;

    // ----- Loading -----

    // Find a config file in the search-path order documented above.
    // Returns empty string if no file is found.
    static std::string find_default_path();

    // Load from disk; returns true on success. On parse error, the struct
    // is left at its default state and an error is logged.
    bool load_from_file(const std::string& path);

    // Apply key=value strings (e.g. "kv_cache.dtype=fp8"). Each entry is
    // parsed via dotted-section lookup. Unknown keys log a warning but
    // don't stop loading.
    void apply_overrides(const std::vector<std::string>& kvs);

    // Convenience: locate + load + apply overrides + log a one-line summary.
    // Pass empty path to use the search-path default.
    static RuntimeConfig load(const std::string& explicit_path, const std::vector<std::string>& overrides);
};

// ---- Pending-config handoff (tool-main → Engine) -----------------------
//
// The C API constructs Engine inside src/api/imp_api.cpp. Tool mains
// (imp-cli, imp-server) load a RuntimeConfig from imp.conf + CLI
// overrides at startup and need to hand that to Engine::init without
// passing it through the ABI-stable ImpConfig C struct.
//
// Workflow: tool main calls set_pending_runtime_config(loaded_cfg) once,
// then later imp_context_create() pulls it via take_pending_runtime_config()
// and passes to Engine::init. This replaces the former
// RuntimeConfig::install() process-wide singleton (Phase 5 Track D
// follow-up, 2026-05-20) — the lifetime is now bounded to a single
// Engine construction; there is no per-call accessor.
void set_pending_runtime_config(RuntimeConfig cfg);
RuntimeConfig take_pending_runtime_config();

}  // namespace imp
