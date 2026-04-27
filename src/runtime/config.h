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
        bool        deterministic_gemm = false;
        std::string cuda_graphs        = "auto";   // "auto" | "always" | "never"
        bool        warmup             = true;
        int         max_seq_len        = 0;        // 0 = use model default
        bool        no_pdl             = false;
    } runtime;

    struct KVCache {
        std::string dtype                       = "fp16";  // fp16 | fp8 | int8 | int4 | nvfp4
        bool        allow_nondeterministic_fp8  = false;
    } kv_cache;

    struct Attention {
        std::string fp8_prefill          = "auto";
        std::string fp8_fmha             = "auto";
        std::string fmha_sm120           = "auto";
        std::string mxfp4                = "auto";
        bool        mxfp4_fp16_fallback  = false;
        std::string fmha_blockscale      = "auto";
        bool        naive                = false;
        bool        no_cublas            = false;
        bool        force_cublas_decode  = false;
        bool        no_qknorm_fused      = false;
        bool        no_naive_swa         = false;
        bool        splitk_pipe          = true;
        bool        gate_concat          = false;
    } attention;

    struct MoE {
        int  expert_overhead_pct = 10;
        bool force_host_experts  = false;
        bool skip                = false;
        bool force_fp16_sync     = false;
        bool no_expert_cache     = false;
    } moe;

    struct GDN {
        bool  fp32_scan          = false;
        bool  fp32_out           = false;
        float norm_eps_override  = 0.0f;   // 0 = use model default
        bool  ref_kernel         = false;
        bool  vhead_reorder      = false;
    } gdn;

    struct GEMM {
        bool no_dp4a      = false;
        bool no_dp4a_gemv = false;
        bool no_dp4a_lm   = false;
        bool no_mmvq      = false;
        bool no_mmvq_q8_0 = false;
    } gemm;

    struct Gemma4 {
        bool fp32_gemm_out = false;
        bool no_graphs     = false;
        bool force_mmvq    = false;
    } gemma4;

    struct Generation {
        bool no_logit_softcap = false;
        bool lm_dequant_fp16  = false;
        int  think_budget     = 0;
    } generation;

    struct Paths {
        std::string mmproj;
    } paths;

    struct Diagnostics {
        bool        debug_forward       = false;
        bool        debug_raw           = false;
        bool        debug_gemm_dispatch = false;
        std::string dump_hidden_dir;
        bool        dump_logits         = false;
        bool        dump_routing        = false;
        bool        dump_tokens         = false;
        int         exit_layer          = -1;
        bool        profile             = false;
        bool        graph_diag          = false;
        std::string graph_dump_dir;
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
    static RuntimeConfig load(const std::string& explicit_path,
                              const std::vector<std::string>& overrides);
};

} // namespace imp
