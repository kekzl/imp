#pragma once

// imp.conf — central runtime configuration.
//
// Replaces ~50 ad-hoc IMP_*-prefixed environment variables that were scattered
// over ~80 getenv() call sites in src/runtime/ and src/graph/. The same values
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
        std::string cuda_graphs = "auto";  // "auto" | "always" | "never"
        bool warmup = false;               // opt-in for prod rollout; off in dev/CI
        int max_seq_len = 0;               // 0 = use model default
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
    } runtime;

    struct KVCache {
        std::string dtype = "fp16";  // fp16 | fp8 | int8 | int4 | nvfp4
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
        std::string fp8_fmha = "auto";
        std::string fmha_sm120 = "auto";
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
        std::string fmha_blockscale = "auto";
        bool naive = false;
        bool no_cublas = false;
        bool force_cublas_decode = false;
        bool no_qknorm_fused = false;
        bool no_naive_swa = false;
        bool splitk_pipe = true;
        bool gate_concat = false;
        // M5 Slice 2: opt-out of the cluster FMHA kernel
        // (attention_fmha_sm120_cluster.cu). **Default true** (cluster
        // DISABLED) — the 2026-05-17 A/B sweep across the four production
        // NVFP4 MoE models (Qwen3.6-35B, Gemma-4-26B, Qwen3-Coder-30B,
        // Qwen3-30B-Modelopt) found cluster wins +6-11 % on HD=128 GQA=8
        // at pp=512 but loses up to -22 % on HD=256 (Qwen3.6 pp=2048,
        // Gemma-4 pp=512). Net user-facing impact is negative on the
        // dominant Qwen3.6 model, so cluster path stays opt-in until the
        // HD=256 regression is root-caused. See
        // m5_slice2_cluster_refuted_2026_05_17.md memo. Set false via
        // imp.conf to re-enable for opt-in HD=128 GQA=8 workloads.
        bool no_fmha_cluster = true;
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
    } moe;

    struct GDN {
        bool fp32_scan = false;
        bool fp32_out = false;
        float norm_eps_override = 0.0f;  // 0 = use model default
        bool ref_kernel = false;
        bool vhead_reorder = false;
        // Override gated-DeltaNet weight layout. Legacy env: IMP_GDN_LAYOUT.
        std::string layout_override;
    } gdn;

    struct GEMM {
        bool no_dp4a = false;
        bool no_dp4a_gemv = false;
        bool no_dp4a_lm = false;
        bool no_mmvq = false;
        bool no_mmvq_q8_0 = false;
    } gemm;

    struct Gemma4 {
        bool fp32_gemm_out = false;
        bool no_graphs = false;
        bool force_mmvq = false;
        bool fp32_expert_down = false;
        bool no_decode_fast = false;
        bool no_post_ffw_1 = false;
        bool ggml_prefill = false;
    } gemma4;

    struct Generation {
        bool no_logit_softcap = false;
        bool lm_dequant_fp16 = false;
        int think_budget = 0;
        bool force_bos = false;
        // Disable banned-token list (debug). Legacy env: IMP_NO_BAN.
        bool no_ban = false;
        // Disable RoPE inside the MTP draft head (diagnostic). Legacy env:
        // IMP_MTP_NO_ROPE.
        bool mtp_no_rope = false;
    } generation;

    struct Server {
        bool prefix_cache = false;
    } server;

    struct Bench {
        bool generate = false;
    } bench;

    struct Paths {
        std::string mmproj;
    } paths;

    struct Diagnostics {
        bool debug_forward = false;
        bool debug_gemm_dispatch = false;
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

    // ---- Process-wide singleton -----------------------------------------
    //
    // Engine init calls install() once with the loaded RuntimeConfig.
    // All ~80 former getenv("IMP_*") call sites read via current(), which
    // returns a const reference to the installed config (or a default-
    // constructed one if install() was never called).
    //
    // This is intentionally a process-wide singleton because the runtime
    // configuration is global state — there is no expectation of two
    // engines with different runtime configs in the same process. If
    // that ever changes, threading a const RuntimeConfig& through every
    // executor call site is a mechanical refactor; the read-side API
    // (current().runtime.no_pdl etc.) does not need to change.
    static const RuntimeConfig& current();
    static void install(const RuntimeConfig& cfg);
};

}  // namespace imp
