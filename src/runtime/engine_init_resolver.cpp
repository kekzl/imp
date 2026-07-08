// Engine init phase: resolve quant/KV/SSM dtype policies + compute max
// sequence length from VRAM budget. Pure orchestration of RuntimeConfig
// + Model metadata — no kernel launches, no allocations.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. Methods remain Engine::* with declarations in engine.h.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "model/model_arch.h"
#include "runtime/process_diag.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "memory/vram_query.h"

#include <algorithm>
#include <cstdlib>

namespace imp {

// IMP_DEBUG_RAW meta-flag: forces the engine into a "naked" FP16 forward pass
// for reproducible byte-level comparison against a reference implementation
// (e.g. llama.cpp). Forces downstream paths off (FP8/NVFP4/warmup/graphs) and
// cuBLAS to deterministic. Triggered via [runtime] debug_raw = true.
void Engine::init_apply_debug_raw_overrides_() {
    const bool debug_raw_ = runtime_config_.runtime.debug_raw;
    if (!debug_raw_)
        return;
    IMP_LOG_INFO(
        "[runtime] debug_raw=true: naked FP16 path (FP8/NVFP4/graphs/warmup/FP8-KV off; deterministic "
        "cuBLAS)");
    // Weight storage: keep FP16 (skip the lossy cache paths)
    config_.use_fp8_prefill = 0;
    config_.use_nvfp4_decode = 0;
    config_.dual_path_quant = false;
    // CUDA graphs off (graph capture can mask state bugs)
    config_.use_cuda_graphs = 0;
    setenv("IMP_NO_CUDA_GRAPH", "1", 0);
    // No warmup (warmup can leak state into first request)
    setenv("IMP_NO_WARMUP", "1", 0);
    // Deterministic cuBLAS (bit-exact across runs, no algo jitter)
    setenv("IMP_DETERMINISTIC_GEMM", "1", 0);
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
    // MoE: no expert LRU cache (state-carrying)
    setenv("IMP_NO_EXPERT_CACHE", "1", 0);
    // GDN: use reference unfused scan (no register-state reordering)
    setenv("IMP_GDN_REF", "1", 0);
    // NOTE: intentionally NOT forcing IMP_FORCE_CUBLAS_DECODE / IMP_NO_FMHA_SM120 /
    // IMP_NO_MMVQ — those trigger incompatible kernel paths that produce IMAs on
    // some combinations. The RAW flag is about disabling *caches and approximations*,
    // not about swapping kernel variants.
}

// Runtime RoPE-scaling override (imp.conf [rope]): inject YaRN/linear
// scaling into a model that doesn't declare it (or replace what it does
// declare). Mirrors the HF-loader semantics (hf_config_loader.cpp,
// rope_scaling type=yarn/linear): rope_freq_scale stores the FACTOR, the
// kernel applies 1/factor itself and self-computes the paper mscale
// 1 + 0.1*ln(factor) (rope_yarn.cuh). Must run before
// init_compute_max_seq_len_() so the extended window reaches the KV sizing
// math, and before executor setup so yarn_corr_dims_ + the MTP draft head
// (shared ModelConfig + rope_yarn.cuh) see the final fields.
bool apply_rope_override(ModelConfig& mcfg, const RuntimeConfig::Rope& rope) {
    if (rope.scaling.empty())
        return false;
    if (rope.scaling != "yarn" && rope.scaling != "linear") {
        IMP_LOG_WARN("[rope] scaling=\"%s\" unknown (yarn|linear) — override ignored",
                     rope.scaling.c_str());
        return false;
    }
    if (rope.factor <= 1.0f) {
        IMP_LOG_WARN("[rope] factor=%.3f must be > 1.0 — override ignored", rope.factor);
        return false;
    }
    // Refuse model classes where a scalar factor is silently wrong:
    // per-dimension frequency tables (LongRoPE / llama3 precomputed pairs),
    // MLA (YaRN mscale is entangled with the softmax scale ratio, #880),
    // and NoPE (no rotary embedding to scale).
    if (!mcfg.rope_short_factor.empty() || !mcfg.rope_long_factor.empty()) {
        IMP_LOG_ERROR("[rope] model uses per-dimension RoPE tables (LongRoPE/llama3) — "
                      "override refused");
        return false;
    }
    if (mcfg.is_mla()) {
        IMP_LOG_ERROR("[rope] MLA models entangle YaRN mscale with the softmax scale — "
                      "override refused");
        return false;
    }
    if (mcfg.rope_attn_disabled) {
        IMP_LOG_ERROR("[rope] model uses NoPE attention (no rotary embedding) — override refused");
        return false;
    }

    // Resolve the native window the factor applies to BEFORE touching
    // max_seq_len: explicit rope.orig_ctx > model-declared rope_n_ctx_orig >
    // the model's declared context.
    int orig_ctx = rope.orig_ctx > 0 ? rope.orig_ctx
                   : mcfg.rope_n_ctx_orig > 0 ? mcfg.rope_n_ctx_orig
                                              : mcfg.max_seq_len;
    if (orig_ctx <= 0) {
        IMP_LOG_WARN("[rope] cannot resolve original context (orig_ctx=0, model ctx=%d) — "
                     "override ignored",
                     mcfg.max_seq_len);
        return false;
    }

    if (mcfg.rope_freq_scale != 1.0f || mcfg.yarn_ext_factor > 0.0f) {
        IMP_LOG_WARN("[rope] model already declares scaling (freq_scale=%.3f, yarn_ext=%.2f) — "
                     "replacing with %s factor=%.2f (set rope.orig_ctx if the declared context "
                     "is already extended)",
                     mcfg.rope_freq_scale, mcfg.yarn_ext_factor, rope.scaling.c_str(), rope.factor);
    }

    mcfg.rope_freq_scale = rope.factor;
    if (rope.scaling == "yarn") {
        mcfg.yarn_ext_factor = 1.0f;
        mcfg.yarn_attn_factor = rope.attn_factor;
        mcfg.yarn_beta_fast = rope.beta_fast;
        mcfg.yarn_beta_slow = rope.beta_slow;
        mcfg.rope_n_ctx_orig = orig_ctx;
    } else {  // linear: pure interpolation, no YaRN blending
        mcfg.yarn_ext_factor = 0.0f;
        mcfg.rope_n_ctx_orig = orig_ctx;
    }

    int extended = static_cast<int>(rope.factor * static_cast<float>(orig_ctx));
    if (extended > mcfg.max_seq_len)
        mcfg.max_seq_len = extended;

    if (mcfg.sliding_window_pattern > 0 || !mcfg.swa_layers.empty()) {
        IMP_LOG_INFO("[rope] note: sliding-window layers keep freq_scale=1.0 by design — the "
                     "override affects global-attention layers only");
    }
    IMP_LOG_INFO("[rope] override applied: %s factor=%.2f orig_ctx=%d → model ctx %d "
                 "(attn_factor=%.2f, beta=%.1f/%.1f)",
                 rope.scaling.c_str(), rope.factor, orig_ctx, mcfg.max_seq_len, rope.attn_factor,
                 rope.beta_fast, rope.beta_slow);
    return true;
}

void Engine::init_apply_rope_override_() {
    apply_rope_override(model_->config_, runtime_config_.rope);
}

// KV cache dtype policy + FP8 KV NaN-bug deterministic-cuBLAS workaround +
// max_batch_size auto-sizing. Default: FP16 (safe). FP8 / NVFP4 / MXFP4-KV
// are opt-in. See the inline rationale for the 2026-04-24 root-cause memo.
void Engine::init_resolve_kv_dtype_policy_() {
    const auto& mcfg = model_->config();
    const bool debug_raw_ = runtime_config_.runtime.debug_raw;
    const bool force_kv_fp16 = (runtime_config_.kv_cache.dtype == "fp16");
    const bool fp8_auto_legacy = runtime_config_.kv_cache.fp8_auto_legacy;

    // Resolve the config-file KV dtype string into the engine enum. The CLI flags
    // (--kv-fp8 / --kv-int4 / …) set config_.kv_cache_dtype directly and win — we
    // only resolve here when the CLI left it at the F16 default. The default "auto"
    // honors the model author's kv_cache_quant_algo=FP8 hint, but ONLY for arch
    // families verified safe for long-context FP8 KV (kv_fp8_hint_default_safe — the
    // measured PPL/coherence gate). Explicit "fp16" opts out; explicit
    // fp8/int8/int4/nvfp4/mxfp4 force that dtype.
    if (config_.kv_cache_dtype == QType::F16) {
        const std::string& kv_str = runtime_config_.kv_cache.dtype;
        if (kv_str == "fp8") {
            config_.kv_cache_dtype = QType::FP8_E4M3;
        } else if (kv_str == "int8") {
            config_.kv_cache_dtype = QType::INT8;
        } else if (kv_str == "int4") {
            config_.kv_cache_dtype = QType::INT4;
        } else if (kv_str == "nvfp4") {
            config_.kv_cache_dtype = QType::NVFP4;
        } else if (kv_str == "mxfp4") {
            config_.kv_cache_dtype = QType::MXFP4_KV;
        } else if (kv_str == "auto" && mcfg.kv_cache_quant_hint == "FP8" &&
                   kv_fp8_hint_default_safe(mcfg.arch)) {
            config_.kv_cache_dtype = QType::FP8_E4M3;
            IMP_LOG_INFO("KV cache dtype: FP8_E4M3 (auto — honoring model author's "
                         "kv_cache_quant_algo=FP8; %s verified safe for long-context FP8 KV; "
                         "set kv_cache.dtype=fp16 to opt out)",
                         model_arch_name(mcfg.arch));
        }
    }

    if (fp8_auto_legacy && config_.kv_cache_dtype == QType::F16 && !debug_raw_ && !force_kv_fp16) {
        config_.kv_cache_dtype = QType::FP8_E4M3;
        IMP_LOG_INFO("KV cache dtype: kv_cache.fp8_auto_legacy → FP8_E4M3 (legacy auto-upgrade)");
    } else if (config_.kv_cache_dtype == QType::F16) {
        IMP_LOG_INFO("KV cache dtype: FP16 (default — pass --kv-fp8 for FP8 E4M3 memory savings)");
    } else if (config_.kv_cache_dtype == QType::NVFP4) {
        IMP_LOG_INFO("KV cache dtype: NVFP4 (FP4 E2M1 + UE4M3 per-16-elem scales, ~3.6× compression)");
        if (config_.use_fp8_prefill) {
            IMP_LOG_INFO("NVFP4 KV: disabling FP8 prefill cache (avoid stacked low-precision drift)");
            config_.use_fp8_prefill = 0;
        }
    } else if (config_.kv_cache_dtype == QType::MXFP4_KV) {
        IMP_LOG_INFO("KV cache dtype: MXFP4_KV (FP4 E2M1 + UE8M0 per-16-elem scales, ~3.6× compression)");
        if (config_.use_fp8_prefill) {
            IMP_LOG_INFO("MXFP4_KV: disabling FP8 prefill cache (avoid stacked low-precision drift)");
            config_.use_fp8_prefill = 0;
        }
    }

    // Scope (2026-06-12, #680 diagnosis): the deterministic forcing below is
    // a PR-#52-era (April) workaround for NaNs in the cuBLAS-attention + FP8
    // round-trip — it predates the FA2 default-on stack (#478/#548). When the
    // f16-QK FA2 chain serves every attention call (hd=128, fa2 enabled),
    // cuBLAS attention never touches the FP8 KV and the forcing only drags in
    // the single-block deterministic MoE permute (measured: the entire −35%
    // pp4096 regression of --kv-fp8 on Qwen3-30B-A3B was that one kernel;
    // without it kv-fp8 is perf-neutral-to-positive at clean PPL on
    // Qwen3-30B/14B/8B). Non-FA2 configs (gemma-class hd!=128, attn sinks
    // via head_dim, fa2 disabled) keep the workaround.
    const bool fa2_serves_attention = mcfg.head_dim == 128 &&
                                      runtime_config_.attention.fmha_fa2 != "never" &&
                                      runtime_config_.attention.fa2_fp16qk != "never";
    if (config_.kv_cache_dtype == QType::FP8_E4M3 && fa2_serves_attention &&
        !runtime_config_.runtime.deterministic_gemm) {
        IMP_LOG_INFO("FP8 KV cache: FA2 serves all attention (hd=128) — skipping the legacy "
                     "deterministic-cuBLAS forcing (set kv_cache.allow_nondeterministic_fp8=false "
                     "semantics unchanged for non-FA2 configs)");
    }
    if (config_.kv_cache_dtype == QType::FP8_E4M3 && !fa2_serves_attention &&
        !runtime_config_.kv_cache.allow_nondeterministic_fp8 &&
        !runtime_config_.runtime.deterministic_gemm) {
        // Phase 5 Track D: mutate the per-Engine RuntimeConfig in place
        // (formerly an install() call into the global singleton). Free-
        // function readers (gemm.cu's algo-selection skip-benchmark branch)
        // now read from the process_diag cache; update it here too so the
        // promotion is visible to them.
        runtime_config_.runtime.deterministic_gemm = true;
        process_diag_set_deterministic_gemm(true);
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
        IMP_LOG_INFO(
            "FP8 KV cache: forcing runtime.deterministic_gemm=true "
            "(non-deterministic cuBLAS + FP8 round-trip → NaN). "
            "Set kv_cache.allow_nondeterministic_fp8=true to opt out.");
    }

    if (config_.max_batch_size <= 0) {
        size_t approx_weight_bytes = static_cast<size_t>(mcfg.d_model) * mcfg.d_model * mcfg.n_layers * 12;
        if (mcfg.n_experts > 0) {
            approx_weight_bytes += static_cast<size_t>(mcfg.n_experts) * mcfg.expert_d_ff * mcfg.d_model *
                                   mcfg.n_layers * 2;
        }
        // Weight-footprint tier — kept as a FLOOR so this never regresses below
        // the previous default for any model.
        int tier;
        if (approx_weight_bytes > 20ULL * 1024 * 1024 * 1024)
            tier = 1;
        else if (approx_weight_bytes > 10ULL * 1024 * 1024 * 1024)
            tier = 4;
        else if (approx_weight_bytes > 5ULL * 1024 * 1024 * 1024)
            tier = 8;
        else
            tier = 16;

        // VRAM-aware concurrency cap. The KV cache is a shared paged pool clamped
        // to the free-VRAM headroom downstream (vram_budget), so a larger cap can
        // NOT OOM — it only lets more sequences share the pool (continuous
        // batching). The old weight-footprint-only heuristic capped a 30B MoE at
        // batch=1 even with ~10 GB headroom, serving concurrent requests one at a
        // time (measured: batch=16 ≈ 2.4× aggregate server throughput). Size the
        // cap so each concurrent slot retains at least a serving context floor
        // (kRefCtxTokens) of KV within a fraction of the headroom. max_seq_len is
        // not resolved yet here, so use the reference length (not the model max)
        // — this avoids the batch↔seq_len circular dependency and the downstream
        // KV clamp keeps it safe regardless. FP16-conservative + all-layers
        // over-estimate KV per token (safe: over-estimate → smaller batch).
        int auto_batch = tier;
        size_t free_vram = 0, total_vram = 0;
        size_t headroom = 0;
        if (vram_budget_mem_get_info(&free_vram, &total_vram) && free_vram > 0) {
            constexpr int kRefCtxTokens = 4096;  // per-slot serving context floor
            constexpr int kMaxAutoBatch = 32;
            constexpr double kKvHeadroomFrac = 0.6;  // rest: workspaces + long-ctx + safety
            // Weights are NOT in VRAM yet at resolver time (still host/mmap; the
            // GPU upload runs later), so cudaMemGetInfo reports the near-empty
            // card. Subtract the weight footprint that WILL be uploaded to get the
            // real post-load headroom. approx_weight_bytes under-counts the NVFP4
            // decode + CUTLASS-SF caches, but kKvHeadroomFrac (0.6) reserves for
            // those + workspaces, and the downstream KV clamp is the hard backstop.
            headroom = (free_vram > approx_weight_bytes) ? (free_vram - approx_weight_bytes) : 0;
            int nkv = mcfg.n_kv_heads > 0 ? mcfg.n_kv_heads : 1;
            int hd = mcfg.head_dim > 0 ? mcfg.head_dim
                                       : (mcfg.n_heads > 0 ? mcfg.d_model / mcfg.n_heads : 128);
            // K+V, 2 B/elem (fp16-conservative; FP8 KV is half), all layers.
            size_t per_tok_kv = static_cast<size_t>(nkv) * hd * 2 * 2 * mcfg.n_layers;
            size_t per_slot_kv = per_tok_kv * static_cast<size_t>(kRefCtxTokens);
            if (per_slot_kv > 0) {
                int fit = static_cast<int>((static_cast<double>(headroom) * kKvHeadroomFrac) /
                                           static_cast<double>(per_slot_kv));
                auto_batch = std::clamp(std::max(tier, fit), 1, kMaxAutoBatch);
            }
        }
        config_.max_batch_size = auto_batch;
        IMP_LOG_INFO("max_batch_size: auto → %d (approx_weights=%.1f GB, post-load headroom=%.1f GB, "
                     "tier-floor=%d, VRAM-aware)",
                     config_.max_batch_size, approx_weight_bytes / (1024.0 * 1024.0 * 1024.0),
                     headroom / (1024.0 * 1024.0 * 1024.0), tier);
    } else {
        IMP_LOG_INFO("max_batch_size: %d (configured)", config_.max_batch_size);
    }
}

// Auto-detect SSM state dtype for hybrid models. Nemotron-H and similar
// Mamba models: use FP16 (~50% VRAM savings). GDN models (Qwen3.5/3.6)
// MUST keep FP32: the delta-rule scan kernel writes FP32 (float) into
// h_state and assumes 4 bytes/element. FP16 allocation would be half the
// size and the next layer's state region would overflow — shipped bug
// that corrupted L1+ GDN state on every Qwen 3.6 forward, producing 37%
// scan-output divergence vs llama.cpp.
void Engine::init_resolve_ssm_dtype_() {
    const auto& mcfg = model_->config();
    const bool has_gdn_for_dtype = (mcfg.ssm_state_size > 0) && model_->profile().is_gdn;
    if (config_.ssm_state_dtype == QType::F32 && mcfg.ssm_state_size > 0 && !has_gdn_for_dtype) {
        config_.ssm_state_dtype = QType::F16;
        IMP_LOG_INFO("SSM state dtype: auto → FP16 (hybrid SSM model, state_size=%d)", mcfg.ssm_state_size);
    }
}

// Auto-detect FP8 prefill. Under runtime.debug_raw or
// [attention] fp8_prefill = "never", keep disabled. The "never" escape
// hatch is for models (e.g. DeepSeek-R1-Distill-Qwen-14B Q6_K) that
// produce garbage decode with FP8 weight cache active — accumulated
// dequant error through deep narrow-GQA stacks.
void Engine::init_resolve_fp8_prefill_() {
    const bool no_fp8_prefill = (runtime_config_.attention.fp8_prefill == "never");
    const bool is_nvfp4_native = model_->config().is_nvfp4_prequant;
    if (is_nvfp4_native && !config_.use_fp8_prefill) {
        IMP_LOG_INFO("FP8 prefill: disabled for native NVFP4 (CUTLASS NVFP4 GEMM used instead)");
    } else if (!config_.use_fp8_prefill && !runtime_config_.runtime.debug_raw && !no_fp8_prefill) {
        int sm_major = 0;
        cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, 0);
        int sm_minor = 0;
        cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, 0);
        int sm = sm_major * 10 + sm_minor;
        if (sm >= 120 && runtime_config_.attention.fp8_prefill != "always") {
            IMP_LOG_INFO(
                "FP8 prefill: auto → DISABLED on sm_%d (cuBLAS 13.4 FP8 returns "
                "NOT_SUPPORTED at non-aligned M on consumer Blackwell; "
                "use --set attention.fp8_prefill=always to force)",
                sm);
        } else {
            config_.use_fp8_prefill = true;
            IMP_LOG_INFO("FP8 prefill: auto → enabled");
        }
    } else if (no_fp8_prefill) {
        IMP_LOG_INFO("FP8 prefill: disabled (attention.fp8_prefill=never)");
    }
}

// Resolve NVFP4 decode mode (additive/only/none) + dual-path quant
// validation + Gemma-4 model-specific carve-outs (force FP16 paths
// until proper kernels land, except CUDA Graphs which Gemma-4 keeps
// because the MoE decode fast path is fully captured). The biggest
// init helper — central place where the quant-stack profile is fixed.
void Engine::init_resolve_quant_flags_() {
    const auto& mcfg = model_->config();
    // --- Resolve auto-detection flags ---

    // gemm.cublas_fp16_acc=auto → per-arch default. GeForce sm_120 quarters
    // FP32-accumulate FP16 tensor-core rate (PR #606 calibration); 16F
    // accumulate restores full rate (+24.9% q8 pp512 measured 2026-06-07,
    // decode neutral, PPL flat on Qwen3-8B). Denied per measurement/risk:
    // Gemma-3/4 (+0.7% PPL on gemma-3-12b) and gpt-oss (documented FP16
    // residual-overflow sensitivity — f16 accumulators are the same hazard
    // class). "on"/"off" bypass this and were applied at install time.
    if (runtime_config_.gemm.cublas_fp16_acc == "auto") {
        const auto& prof = model_->profile();
        const bool deny = (prof.is_gemma3 || prof.is_gemma4 || prof.is_gpt_oss);
        process_diag_set_cublas_fp16_acc(!deny);
        IMP_LOG_INFO("cuBLAS FP16-accumulate prefill: auto → %s (arch=%s)", deny ? "OFF" : "ON",
                     model_arch_name(mcfg.arch));
    }

    // NVFP4 decode mode
    config_.nvfp4_decode_all = runtime_config_.gemm.nvfp4_decode_all;

    if (config_.use_nvfp4_decode < 0) {
        const auto wq_qtype = model_->layer(0).wq.qtype;
        // IQ4_NL/IQ4_XS count as beneficial unconditionally: i-quants have no
        // dp4a/MMVQ decode kernels, so the NVFP4 decode cache is their only
        // fast (and graph-capturable) decode path.
        const bool is_iq4 = (wq_qtype == QType::IQ4_NL || wq_qtype == QType::IQ4_XS);
        const bool nvfp4_beneficial_qtype = (wq_qtype == QType::Q8_0 || wq_qtype == QType::Q8_K ||
                                              wq_qtype == QType::Q6_K || wq_qtype == QType::Q5_K ||
                                              is_iq4 ||
                                              (config_.nvfp4_decode_all &&
                                               (wq_qtype == QType::Q4_K || wq_qtype == QType::Q3_K ||
                                                wq_qtype == QType::Q2_K)));
        const bool is_moe = model_->profile().is_moe;
        const bool is_gdn = model_->profile().is_gdn;

        const bool sub8bit_qtype = (wq_qtype == QType::Q4_K || wq_qtype == QType::Q3_K ||
                                     wq_qtype == QType::Q2_K || is_iq4);
        if (nvfp4_beneficial_qtype && !is_moe && !is_gdn && !sub8bit_qtype) {
            // Dense Q*_K (6-8 bit GGUF) on sm_120: mode 1 (additive — high-
            // precision prefill cache (FP8 1 B/elem, or FP16 2 B/elem when FP8 is
            // unavailable) PLUS an NVFP4 decode cache (0.5 B/elem)). Prefill stays
            // high-precision (prefill-on-NVFP4 corrupts the prompt context and
            // degenerates output for 8-bit GGUF — that is why mode 2 is reserved
            // for sub-8-bit weights below). Decode uses the NVFP4 cache, which is
            // both fast and coherent. Measures +4% decode over mode 2 on Qwen3-14B
            // Q6_K @ ctx=2048 (151 vs 145.6 tok/s, PR #364). GOAL.md ranks decode
            // #1 for the north-star, so dense Q*_K defaults to mode 1.
            config_.use_nvfp4_decode = 1;
            IMP_LOG_INFO("NVFP4 decode: auto → mode 1 (dense Q*_K — decode-first)");
        } else if (nvfp4_beneficial_qtype && !is_moe && !is_gdn && sub8bit_qtype) {
            // Sub-8-bit with nvfp4_decode_all: mode 2 (NVFP4 only). Mode 1
            // wastes budget on FP16 cache that starves the NVFP4 decode cache.
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO("NVFP4 decode: auto → mode 2 (sub-8-bit Q*_K + decode_all)");
        } else if (is_gdn) {
            // GDN models with large d_model: enable NVFP4 for attention + FFN weights,
            // but SSM/GDN projections (ssm_in/ssm_out) will be excluded in
            // pre_dequant_weights to preserve recurrent state precision.
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO(
                "NVFP4 decode: auto → mode 2 (GDN model — "
                "ssm_in/ssm_out excluded for precision)");
        } else {
            const char* why = is_moe ? "MoE" : "non-Q*_K-6-8bit";
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO("NVFP4 decode: auto → mode 2 (%s)", why);
        }
    }

    // FP8 prefill auto-disable for sub-8-bit models: Q4_K→FP8 loses ~1 bit
    // per weight element; with 48 attention layers this compounds into
    // degenerate output (verified on Qwen3-30B Q4_K_M). The dequant fallback
    // (PR #431) handles these models by dequanting Q4_K→FP16 on each forward.
    if (config_.use_fp8_prefill) {
        auto qtype = model_->layer(0).wq.qtype;
        bool sub_8bit = (qtype == QType::Q4_0 || qtype == QType::Q4_K || qtype == QType::Q5_0 ||
                         qtype == QType::Q5_K || qtype == QType::Q3_K || qtype == QType::Q2_K ||
                         qtype == QType::Q4_1 || qtype == QType::Q5_1);
        if (sub_8bit) {
            config_.use_fp8_prefill = 0;
            IMP_LOG_INFO("FP8 prefill cache: auto-disabled (sub-8-bit weights → dequant fallback)");
        }
    }

    // Dual-path quant validation: requires NVFP4 decode + FP8 prefill.
    // If either is missing, auto-enable or warn.
    if (config_.dual_path_quant) {
        if (config_.use_nvfp4_decode <= 0) {
            IMP_LOG_WARN("Dual-path quant requires NVFP4 decode — enabling mode 2 (NVFP4 only)");
            config_.use_nvfp4_decode = 2;
        }
        if (!config_.use_fp8_prefill) {
            IMP_LOG_INFO("Dual-path quant: auto-enabling FP8 prefill for attention weight quality");
            config_.use_fp8_prefill = true;
        }
    }

    // Gemma 4: FP8 prefill, NVFP4 prefill, CUTLASS paths, and CUDA graphs all have
    // incompatibilities with the per-layer head_dim + split MoE tensor layout.
    // Force plain FP16 paths for Gemma 4 until proper kernels are added.
    // GDN models can't use FP8 prefill: recurrent state accumulates precision
    // error per token, FP8 E4M3 (3-bit mantissa) amplifies it through the delta
    // rule scan and degenerates output after ~50 multi-turn special tokens.
    // Decide this BEFORE executor_->init() so the fp8_activation scratch
    // buffer + d_act_scale / d_fp8_block_maxes / d_fp8_absmax aren't allocated
    // and then never used (was happening when the disable lived inside
    // init_kv_cache, ~3 MiB pure waste). Dual-path quant keeps the FP8 path
    // for FFN even on GDN — only attention drops to FP16.
    if (config_.use_fp8_prefill && !config_.dual_path_quant && model_->profile().is_gdn) {
        IMP_LOG_INFO("GDN model: disabling FP8 prefill (recurrent state needs FP16 precision)");
        config_.use_fp8_prefill = 0;
    }
    // DENSE Gemma (3 and 4) from GGUF: the mode-2 NVFP4 conversion used to be
    // broken here (#514 server-only IMA truncation on gemma-3-12b, #516 NaN
    // logits from step 2 on gemma-4-31B) and was capped to mode 1 as a
    // mitigation. Re-validated 2026-06-06 (#552): both manifestations are
    // gone on current main (fixed by the intervening decode-path work, most
    // likely the #539 in-place compaction race fix) — gemma-3-12b mode 2 is
    // degen-suite clean CLI+server, gemma-4-31B mode 2 answers coherently
    // (55 tok/s vs 21.7 at mode 1) with multi-turn green. The cap is removed;
    // dense Gemma follows the same sub-8-bit mode-2 auto-pick as every other
    // arch (still gated behind gemm.nvfp4_decode_all for Q4_K-class sources).
    if (model_->profile().is_gemma4) {
        // CUDA graphs: enabled for Gemma-4 decode. The MoE decode fast path is fully
        // device-side (dp4a GEMV, no D2H memcpy), so graph capture works.
        // Only the MoE prefill path uses D2H sync, but prefill is never graph-captured.
        // FP8 prefill carve-out removed 2026-05-15. The 2026-05-09 measurement
        // showed -5..-19% prefill on Gemma-4 vs FP16; since then (PRs #177, #181)
        // the gap has closed. Re-measured 2026-05-15 on Q4_K_M:
        //   pp128:  +1.0%  pp512:  -0.9%  pp833:  -4.2%  pp2048: +7.3%
        // Net effect is neutral with a long-context advantage. FP8 also halves
        // the activation cache, which helps VRAM at long context. Users wanting
        // max prefill at medium pp can opt out via [attention] fp8_prefill = "never".
        if (config_.use_nvfp4_decode) {
            // Prequant SafeTensors NVFP4 weights are already in NVFP4 layout on
            // disk. Phase 3a (Q*_K → NVFP4 conversion) and Phase 3b
            // (NVFP4 → CUTLASS sm_120) iterate `wcache_.nvfp4` which stays
            // empty for prequant, so they are no-ops. Phase 3-MoE (the
            // cache_moe_native_nvfp4 lambda in executor_pre_dequant.cu) IS
            // load-bearing — it builds the contiguous per-layer expert buffer
            // that lights up the M=1 decode fast path (gemv_nvfp4_*) and lets
            // CUDA Graphs capture decode without D2H expert_offsets sync.
            //
            // For Q*_K source weights the per-tensor convert→quantize loop in
            // executor_pre_dequant.cu builds wcache_.nvfp4 per tensor; the
            // per-layer head_dim (256 SWA / 512 global) is uniformly handled
            // since each entry carries its own (N, K) shape. Verified 2026-05-15
            // on Q4_K_M + UD-Q4_K_M: tg256 184 → 204 tok/s (+11%), pp512
            // 1795 → 2347 tok/s (+30%). Coherent on chat prompts; the
            // pre-existing Q4_K_M code-gen drift (see roadmap) is orthogonal.
            IMP_LOG_INFO("Gemma 4: NVFP4 decode cache enabled (use_nvfp4_decode=%d, prequant=%d)",
                         config_.use_nvfp4_decode,
                         (int)model_->config().is_nvfp4_prequant);
        }
        if (config_.dual_path_quant) {
            IMP_LOG_INFO("Gemma 4: disabling dual_path_quant");
            config_.dual_path_quant = false;
        }
        // (Gemma-4 force-FP16 KV carve-out removed 2026-05-01.) The original
        // bug was the FP8 KV calibration reading garbage beyond the per-layer
        // live K/V region — Gemma-4 has dual head_dim (256 SWA / 512 global)
        // and the workspace is allocated for max_head_dim, leaving a
        // tail-region of uninitialized memory on SWA layers. The fix in
        // src/exec/executor_kv_write.cu narrows the calibration view to
        // `nkv * hd` per layer; FP8 KV is now safe to opt into on Gemma-4.
        // Gemma 4 output_norm has extreme outliers (max=588). Small numeric jitter
        // from cuBLAS algo autotuning / split-K atomics amplifies into wildly
        // different top-1 picks (coherent " Paris" vs garbage "\n"). Force
        // deterministic GEMM paths so generation is stable run-to-run.
        if (!runtime_config_.runtime.deterministic_gemm) {
            // Phase 5 Track D: mutate the per-Engine RuntimeConfig in place
            // (formerly an install() call into the global singleton). Also
            // update the process_diag cache so free-function gemm.cu reader
            // observes the promotion (see FP8-KV block above).
            runtime_config_.runtime.deterministic_gemm = true;
            process_diag_set_deterministic_gemm(true);
            IMP_LOG_INFO(
                "Gemma 4: enabling runtime.deterministic_gemm (output_norm outliers amplify algo jitter)");
        }
        if (!getenv("CUBLAS_WORKSPACE_CONFIG")) {
            setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
            IMP_LOG_INFO("Gemma 4: setting CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic grouped GEMM");
        }
        // Gemma 4: CUDA graphs are fully enabled by default. The user can opt
        // out via [gemma4] no_graphs = true for bisecting regressions.
        if (model_->config().overrides.gemma4.no_graphs) {
            IMP_LOG_INFO("Gemma 4: disabling all CUDA graphs (gemma4.no_graphs=true)");
            config_.use_cuda_graphs = false;
        }
        // Enable MMVQ for all weight GEMMs — quantized matmul matching llama.cpp's
        // accumulation behavior, critical for 128-expert MoE precision.
        if (!model_->config().overrides.gemma4.force_mmvq) {
            model_->config_.overrides.gemma4.force_mmvq = true;
            IMP_LOG_INFO("Gemma 4: enabling MMVQ for all weight GEMMs (numerical parity with llama.cpp)");
        }
    }
}

// Auto-detect max_seq_len. Runs AFTER model-specific overrides (Gemma-4
// forces FP16 KV etc.) so the per-token cost reflects the actual dtype
// that will be allocated. Conservative cap at 16K; the user can override
// via runtime.max_seq_len.
void Engine::init_compute_max_seq_len_() {
    const auto& mcfg = model_->config();
    if (int v = runtime_config_.runtime.max_seq_len; v > 0) {
        config_.max_seq_len = v;
        IMP_LOG_INFO("max_seq_len: runtime.max_seq_len=%d", v);
    }
    if (config_.max_seq_len <= 0) {
        int model_ctx = mcfg.max_seq_len;  // from GGUF metadata
        size_t free_vram = 0, total_vram = 0;
        vram_budget_mem_get_info(&free_vram, &total_vram);
        int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
        // Hybrid models (Qwen3.5/3.6 GDN, Nemotron-H Mamba2) populate
        // n_kv_heads_per_layer with zeros for non-attention layers — those don't
        // contribute to the KV cache. Counting only nonzero entries avoids a
        // 4-9× per-token-bytes overestimate that clamped max_seq_len far below
        // VRAM-feasible (e.g. Qwen3.5-4B GDN: 32 total / 8 attention = 4×).
        int kv_layer_count = mcfg.n_layers;
        if (!mcfg.n_kv_heads_per_layer.empty()) {
            int populated = 0;
            for (int v : mcfg.n_kv_heads_per_layer)
                if (v > 0)
                    ++populated;
            if (populated > 0)
                kv_layer_count = populated;
        }
        // SWA-aware sizing (kv_cache.swa_sizing): sliding-window layers hold
        // only a fixed trailing window, so they don't scale with context —
        // count only global layers for the per-token cost. (The final gate is
        // resolved in init_kv_cache; if it declines there the budget clamps
        // conservatively, never OOMs.)
        if (runtime_config_.kv_cache.swa_sizing) {
            int swa_layers = 0;
            for (int l = 0; l < mcfg.n_layers; l++)
                if (layer_swa_window(mcfg, model_->profile(), l) > 0)
                    ++swa_layers;
            if (swa_layers > 0 && swa_layers < kv_layer_count) {
                IMP_LOG_INFO("max_seq_len auto: SWA sizing — %d/%d windowed layers excluded "
                             "from per-token KV cost",
                             swa_layers, kv_layer_count);
                kv_layer_count -= swa_layers;
            }
        }
        auto kv = config_.kv_cache_dtype;
        bool packed_int4 = (kv == QType::INT4);
        size_t per_tok_elems = static_cast<size_t>(mcfg.n_kv_heads) * head_dim * kv_layer_count *
                               2;  // K+V, per KV head, attention layers only
        size_t kv_bytes_per_token = packed_int4 ? (per_tok_elems / 2) : (per_tok_elems * dtype_size(kv));
        // The budget planner downstream targets kv_fraction (default 0.8) of
        // free VRAM for KV. Cap the auto-detect at 0.75 × that (0.6 at the
        // default) so it doesn't undershoot what the planner can afford and
        // keeps tracking a tuned vram.kv_fraction. (Was a flat 30%, calibrated
        // when weight caches competed at FP16.)
        float kv_fraction = std::clamp(config_.kv_fraction, 0.05f, 0.95f);
        int max_by_vram = (kv_bytes_per_token > 0)
                              ? static_cast<int>(free_vram * (0.75 * kv_fraction) / kv_bytes_per_token)
                              : 131072;
        // Agentic workloads (tool loops, accumulating history, large file
        // context) routinely exceed 16K and run a single long sequence rather
        // than many short ones, so a high per-request ceiling is the right
        // default. Cap lifted 16K → 64K; max_by_vram still bounds it to what
        // VRAM honestly affords and model_ctx to what the model supports, so
        // this only raises the ceiling on models that declare (and can hold)
        // more than 16K.
        constexpr int kAutoMaxSeqLenCap = 65536;
        config_.max_seq_len = std::min({model_ctx, std::max(max_by_vram, 4096), kAutoMaxSeqLenCap});
        IMP_LOG_INFO(
            "max_seq_len: auto → %d (model=%d, vram_cap=%d, auto_cap=%d, kv=%zu B/tok, attn_layers=%d/%d)",
            config_.max_seq_len, model_ctx, max_by_vram, kAutoMaxSeqLenCap, kv_bytes_per_token,
            kv_layer_count, mcfg.n_layers);
    }
}

}  // namespace imp
