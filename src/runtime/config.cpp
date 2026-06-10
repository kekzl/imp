#include "runtime/config.h"
#include "core/logging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <pwd.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace imp {

namespace {

// ----- Tiny INI/TOML-subset parser ---------------------------------------
//
// Supports:
//   [section]            section header
//   key = value          plain
//   key = "value"        quoted string
//   key = true | false   booleans
//   key = 42             integers
//   key = 3.14           floats
//   # comment            line comment
//
// This is a minimal subset of TOML — enough for imp.conf which is flat
// (no nested tables, no arrays). When the project picks up tomlplusplus
// the parser body here can be replaced without touching the call sites.

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r'))
        ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r'))
        --e;
    return s.substr(b, e - b);
}

std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

bool parse_bool(const std::string& v, bool fallback) {
    if (v == "true" || v == "True" || v == "1" || v == "yes" || v == "on")
        return true;
    if (v == "false" || v == "False" || v == "0" || v == "no" || v == "off")
        return false;
    return fallback;
}

int parse_int(const std::string& v, int fallback) {
    if (v.empty())
        return fallback;
    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

float parse_float(const std::string& v, float fallback) {
    if (v.empty())
        return fallback;
    try {
        return std::stof(v);
    } catch (...) {
        return fallback;
    }
}

// Apply a single dotted key (e.g. "kv_cache.dtype") with raw value string.
// Logs a warning for unknown keys but keeps going.
void apply_one(RuntimeConfig& cfg, const std::string& dotted_key, const std::string& raw) {
    std::string val = strip_quotes(trim(raw));

    // Typed key binders. Each binds one dotted key to its destination field;
    // the field's type selects the parser, and the compiler rejects a key bound
    // to a wrong-typed field. First match wins (the `matched` guard mirrors the
    // old else-if short-circuit); unknown keys fall through to the warning.
    bool matched = false;
    auto B = [&](const char* k, bool& f) {
        if (!matched && dotted_key == k) { f = parse_bool(val, f); matched = true; }
    };
    auto I = [&](const char* k, int& f) {
        if (!matched && dotted_key == k) { f = parse_int(val, f); matched = true; }
    };
    auto F = [&](const char* k, float& f) {
        if (!matched && dotted_key == k) { f = parse_float(val, f); matched = true; }
    };
    auto S = [&](const char* k, std::string& f) {
        if (!matched && dotted_key == k) { f = val; matched = true; }
    };

    // [runtime]
    B("runtime.deterministic_gemm", cfg.runtime.deterministic_gemm);
    if (!matched && dotted_key == "runtime.deterministic") {
        cfg.runtime.deterministic = parse_bool(val, cfg.runtime.deterministic);
        // Full determinism implies deterministic GEMM algo selection — the
        // compute kernels gate routing/sampling determinism on the same
        // process_diag_deterministic_gemm() snapshot, so this one switch
        // covers GEMM + MoE routing + top-k sampling.
        if (cfg.runtime.deterministic)
            cfg.runtime.deterministic_gemm = true;
        matched = true;
    }
    S("runtime.cuda_graphs", cfg.runtime.cuda_graphs);
    B("runtime.warmup", cfg.runtime.warmup);
    I("runtime.max_seq_len", cfg.runtime.max_seq_len);
    B("runtime.no_pdl", cfg.runtime.no_pdl);
    B("runtime.debug_raw", cfg.runtime.debug_raw);
    B("runtime.no_vision_graph", cfg.runtime.no_vision_graph);
    S("runtime.graph_capture_mode", cfg.runtime.graph_capture_mode);
    B("runtime.prefill_graph", cfg.runtime.prefill_graph);
    I("runtime.max_batch_size", cfg.runtime.max_batch_size);

    // [kv_cache]
    S("kv_cache.dtype", cfg.kv_cache.dtype);
    B("kv_cache.allow_nondeterministic_fp8", cfg.kv_cache.allow_nondeterministic_fp8);
    B("kv_cache.fp8_auto_legacy", cfg.kv_cache.fp8_auto_legacy);
    I("kv_cache.bitdecoding_residual_tokens", cfg.kv_cache.bitdecoding_residual_tokens);
    B("kv_cache.bitdecoding_qk", cfg.kv_cache.bitdecoding_qk);

    // [attention]
    S("attention.fp8_prefill", cfg.attention.fp8_prefill);
    S("attention.fp8_fmha", cfg.attention.fp8_fmha);
    S("attention.fmha_sm120", cfg.attention.fmha_sm120);
    S("attention.fmha_fa2", cfg.attention.fmha_fa2);
    S("attention.fa2_fp16qk", cfg.attention.fa2_fp16qk);
    B("attention.fa2_f16acc", cfg.attention.fa2_f16acc);
    I("attention.fmha_prefill_threshold", cfg.attention.fmha_prefill_threshold);
    I("attention.attn_scores_mib", cfg.attention.attn_scores_mib);
    S("attention.mxfp4", cfg.attention.mxfp4);
    B("attention.mxfp4_fp16_fallback", cfg.attention.mxfp4_fp16_fallback);
    S("attention.mxfp4_fp16_cache_policy", cfg.attention.mxfp4_fp16_cache_policy);
    B("attention.force_cublas_decode", cfg.attention.force_cublas_decode);
    B("attention.no_qknorm_fused", cfg.attention.no_qknorm_fused);
    B("attention.splitk_pipe", cfg.attention.splitk_pipe);
    B("attention.gate_concat", cfg.attention.gate_concat);

    // [moe]
    I("moe.expert_overhead_pct", cfg.moe.expert_overhead_pct);
    I("moe.force_host_experts", cfg.moe.force_host_experts);
    B("moe.skip", cfg.moe.skip);
    B("moe.force_fp16_sync", cfg.moe.force_fp16_sync);
    B("moe.no_expert_cache", cfg.moe.no_expert_cache);
    B("moe.expert_cache_debug_parity", cfg.moe.expert_cache_debug_parity);
    I("moe.prefetch_top_k", cfg.moe.prefetch_top_k);
    B("moe.allow_graphs_under_offload", cfg.moe.allow_graphs_under_offload);
    B("moe.zero_workspace", cfg.moe.zero_workspace);
    B("moe.no_shared_mlp", cfg.moe.no_shared_mlp);
    B("moe.no_shexp_gate", cfg.moe.no_shexp_gate);
    B("moe.no_cutlass3x", cfg.moe.no_cutlass3x);
    I("moe.reserve_mib", cfg.moe.reserve_mib);
    B("moe.nvfp4_device_args", cfg.moe.nvfp4_device_args);
    B("moe.nvfp4_smallM", cfg.moe.nvfp4_smallM);
    I("moe.nvfp4_smallM_threshold", cfg.moe.nvfp4_smallM_threshold);
    I("moe.mr_nr", cfg.moe.mr_nr);

    // [gdn]
    B("gdn.fp32_scan", cfg.gdn.fp32_scan);
    B("gdn.fp32_out", cfg.gdn.fp32_out);
    F("gdn.norm_eps_override", cfg.gdn.norm_eps_override);
    S("gdn.layout_override", cfg.gdn.layout_override);
    B("gdn.ref_kernel", cfg.gdn.ref_kernel);
    B("gdn.vhead_reorder", cfg.gdn.vhead_reorder);
    B("gdn.chunkwise_scan", cfg.gdn.chunkwise_scan);

    // [gemm]
    B("gemm.no_dp4a_gemv", cfg.gemm.no_dp4a_gemv);
    B("gemm.no_dp4a_lm", cfg.gemm.no_dp4a_lm);
    B("gemm.no_mmvq", cfg.gemm.no_mmvq);
    B("gemm.no_mmvq_q8_0", cfg.gemm.no_mmvq_q8_0);
    B("gemm.q4k_hmma_enabled", cfg.gemm.q4k_hmma_enabled);
    B("gemm.q8_imma_enabled", cfg.gemm.q8_imma_enabled);
    B("gemm.q4k_imma_prefill", cfg.gemm.q4k_imma_prefill);
    B("gemm.moe_imma_prefill", cfg.gemm.moe_imma_prefill);
    B("gemm.nvfp4_decode_all", cfg.gemm.nvfp4_decode_all);
    B("gemm.nvfp4_lm_head", cfg.gemm.nvfp4_lm_head);
    if (!matched && dotted_key == "gemm.cublas_fp16_acc") {
        // tri-state auto|on|off; legacy bool spellings stay valid
        if (val == "auto" || val == "on" || val == "off")
            cfg.gemm.cublas_fp16_acc = val;
        else
            cfg.gemm.cublas_fp16_acc = parse_bool(val, false) ? "on" : "off";
        matched = true;
    }
    B("gemm.nvfp4_lm_head_gdn", cfg.gemm.nvfp4_lm_head_gdn);
    B("gemm.nvfp4_ssm_proj", cfg.gemm.nvfp4_ssm_proj);
    B("gemm.nvfp4_attn_proj", cfg.gemm.nvfp4_attn_proj);
    B("gemm.nvfp4_moe_decode", cfg.gemm.nvfp4_moe_decode);

    // [gemma4] section moved to ModelConfig::Overrides::Gemma4 in Phase 5
    // Track A of the architecture refactor. Per-model knobs no longer live
    // on the global RuntimeConfig singleton — they are now populated by the
    // GGUF / SafeTensors loader or the engine init resolver onto the model.

    // [generation]
    B("generation.no_logit_softcap", cfg.generation.no_logit_softcap);
    B("generation.lm_dequant_fp16", cfg.generation.lm_dequant_fp16);
    I("generation.think_budget", cfg.generation.think_budget);
    B("generation.force_bos", cfg.generation.force_bos);
    B("generation.no_ban", cfg.generation.no_ban);
    B("generation.mtp_no_rope", cfg.generation.mtp_no_rope);

    // [server]
    B("server.prefix_cache", cfg.server.prefix_cache);
    I("server.prefix_pin_budget_pct", cfg.server.prefix_pin_budget_pct);
    B("server.green_contexts", cfg.server.green_contexts);

    // [bench]
    B("bench.generate", cfg.bench.generate);

    // [paths]
    S("paths.mmproj", cfg.paths.mmproj);

    // [diagnostics]
    B("diagnostics.debug_forward", cfg.diagnostics.debug_forward);
    B("diagnostics.debug_template", cfg.diagnostics.debug_template);
    S("diagnostics.dump_hidden_dir", cfg.diagnostics.dump_hidden_dir);
    S("diagnostics.dump_logits_dir", cfg.diagnostics.dump_logits_dir);
    S("diagnostics.dump_routing_dir", cfg.diagnostics.dump_routing_dir);
    B("diagnostics.dump_tokens", cfg.diagnostics.dump_tokens);
    I("diagnostics.exit_layer", cfg.diagnostics.exit_layer);
    B("diagnostics.profile", cfg.diagnostics.profile);
    B("diagnostics.graph_diag", cfg.diagnostics.graph_diag);
    S("diagnostics.graph_dump_dir", cfg.diagnostics.graph_dump_dir);
    B("diagnostics.nvfp4_force_dequant", cfg.diagnostics.nvfp4_force_dequant);
    B("diagnostics.log_gemm_algo", cfg.diagnostics.log_gemm_algo);
    B("diagnostics.mtp_pattern_log", cfg.diagnostics.mtp_pattern_log);
    B("diagnostics.mtp_prenorm_h", cfg.diagnostics.mtp_prenorm_h);
    B("diagnostics.audit_nvfp4_scales", cfg.diagnostics.audit_nvfp4_scales);

    // [ffn]
    B("ffn.sparsity_probe", cfg.ffn.sparsity_probe);
    F("ffn.sparsity_threshold", cfg.ffn.sparsity_threshold);

    // [speculative]
    B("speculative.ngram", cfg.speculative.ngram);
    I("speculative.k", cfg.speculative.k);
    I("speculative.min_match", cfg.speculative.min_match);
    I("speculative.max_match", cfg.speculative.max_match);
    I("speculative.give_up_after", cfg.speculative.give_up_after);
    I("speculative.burst", cfg.speculative.burst);

    if (!matched)
        IMP_LOG_WARN("imp.conf: unknown key '%s' (value '%s') — ignoring", dotted_key.c_str(), val.c_str());
}

bool file_exists(const std::string& path) {
    if (path.empty())
        return false;
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string home_dir() {
    if (const char* h = std::getenv("HOME"))
        return h;
    if (struct passwd* pw = getpwuid(getuid()))
        return pw->pw_dir;
    return {};
}

// Backward-compat: legacy IMP_* env vars seed the matching RuntimeConfig
// fields before [imp.conf] file overrides. Semantics preserved exactly
// per original call-site checks (see review/phase3_maint.md §9.1).
void seed_from_env(RuntimeConfig& cfg) {
    // runtime.deterministic — IMP_DETERMINISTIC: '1'/'true' enables full
    // reproducibility (also implies deterministic_gemm).
    if (const char* e = std::getenv("IMP_DETERMINISTIC")) {
        cfg.runtime.deterministic = parse_bool(e, cfg.runtime.deterministic);
        if (cfg.runtime.deterministic)
            cfg.runtime.deterministic_gemm = true;
    }

    // moe.reserve_mib — IMP_MOE_RESERVE_MIB: integer MiB.
    if (const char* e = std::getenv("IMP_MOE_RESERVE_MIB"))
        cfg.moe.reserve_mib = parse_int(e, cfg.moe.reserve_mib);

    // kv_cache.bitdecoding_qk — IMP_USE_BITDECODING_QK: '1' enables.
    if (const char* e = std::getenv("IMP_USE_BITDECODING_QK"))
        cfg.kv_cache.bitdecoding_qk = (e[0] == '1');

    // moe.nvfp4_device_args — IMP_NVFP4_DEVICE_ARGS: '0' disables, default ON.
    if (const char* e = std::getenv("IMP_NVFP4_DEVICE_ARGS"))
        cfg.moe.nvfp4_device_args = (std::atoi(e) != 0);

    // gemm.nvfp4_lm_head — IMP_NO_NVFP4_LM_HEAD: '1' disables the FP16-LM-head
    // NVFP4 decode cache (default ON).
    if (const char* e = std::getenv("IMP_NO_NVFP4_LM_HEAD"))
        cfg.gemm.nvfp4_lm_head = (std::atoi(e) == 0);

    // gemm.nvfp4_moe_decode — IMP_NO_NVFP4_MOE_DECODE: '1' disables the fast
    // per-expert NVFP4 MoE decode path (default ON).
    if (const char* e = std::getenv("IMP_NO_NVFP4_MOE_DECODE"))
        cfg.gemm.nvfp4_moe_decode = (std::atoi(e) == 0);

    // attention.attn_scores_mib — IMP_ATTN_SCORES_MIB: cuBLAS attention S-matrix cap.
    if (const char* e = std::getenv("IMP_ATTN_SCORES_MIB")) {
        int v = std::atoi(e);
        if (v > 0)
            cfg.attention.attn_scores_mib = v;
    }

    // attention.fmha_fa2 — IMP_FMHA_FA2: '1' enables the register-resident FA2
    // prefill kernel (A/B vs the legacy FP8 FMHA), '0' forces it off.
    if (const char* e = std::getenv("IMP_FMHA_FA2"))
        cfg.attention.fmha_fa2 = (std::atoi(e) != 0) ? "on" : "never";

    // attention.fa2_f16acc — IMP_FA2_F16ACC: '1' enables f16-accumulate QK^T
    // in the fp16-qk FA2 kernel (#597, +3-4% long-ctx prefill / +0.37% PPL).
    if (const char* e = std::getenv("IMP_FA2_F16ACC"))
        cfg.attention.fa2_f16acc = (std::atoi(e) != 0);

    // moe.nvfp4_smallM — IMP_NVFP4_SMALLM: integer != 0 enables.
    if (const char* e = std::getenv("IMP_NVFP4_SMALLM"))
        cfg.moe.nvfp4_smallM = (std::atoi(e) != 0);

    // moe.nvfp4_smallM_threshold — IMP_NVFP4_SMALLM_THRESHOLD: clamped int.
    if (const char* e = std::getenv("IMP_NVFP4_SMALLM_THRESHOLD")) {
        int v = std::atoi(e);
        if (v < 0)
            v = 0;
        if (v > 128)
            v = 128;
        cfg.moe.nvfp4_smallM_threshold = v;
    }

    // moe.mr_nr — IMP_MOE_MR_NR: rows-per-block for NVFP4 MoE decode kernels.
    if (const char* e = std::getenv("IMP_MOE_MR_NR"))
        cfg.moe.mr_nr = std::atoi(e);

    // diagnostics.nvfp4_force_dequant — IMP_NVFP4_FORCE_DEQUANT: '1' only.
    if (const char* e = std::getenv("IMP_NVFP4_FORCE_DEQUANT"))
        cfg.diagnostics.nvfp4_force_dequant = (e[0] == '1');

    // diagnostics.log_gemm_algo — IMP_LOG_GEMM_ALGO: '1' only.
    if (const char* e = std::getenv("IMP_LOG_GEMM_ALGO"))
        cfg.diagnostics.log_gemm_algo = (e[0] == '1');

    // runtime.graph_capture_mode — IMP_GRAPH_CAPTURE_MODE: string.
    if (const char* e = std::getenv("IMP_GRAPH_CAPTURE_MODE"))
        cfg.runtime.graph_capture_mode = e;

    // runtime.prefill_graph — IMP_PREFILL_GRAPH: presence enables.
    if (std::getenv("IMP_PREFILL_GRAPH") != nullptr)
        cfg.runtime.prefill_graph = true;

    // generation.no_ban — IMP_NO_BAN: '1' only.
    if (const char* e = std::getenv("IMP_NO_BAN"))
        cfg.generation.no_ban = (e[0] == '1');

    // generation.mtp_no_rope — IMP_MTP_NO_ROPE: set AND first char != '0'.
    // Note: empty string ('\0' != '0') enables. Preserves original behavior.
    if (const char* e = std::getenv("IMP_MTP_NO_ROPE"))
        cfg.generation.mtp_no_rope = (e[0] != '0');

    // diagnostics.mtp_pattern_log — IMP_MTP_PATTERN_LOG: non-empty AND first char != '0'.
    if (const char* e = std::getenv("IMP_MTP_PATTERN_LOG"))
        cfg.diagnostics.mtp_pattern_log = (std::strlen(e) > 0 && e[0] != '0');

    // diagnostics.mtp_prenorm_h — IMP_MTP_PRENORM_H: same convention.
    if (const char* e = std::getenv("IMP_MTP_PRENORM_H"))
        cfg.diagnostics.mtp_prenorm_h = (std::strlen(e) > 0 && e[0] != '0');

    // diagnostics.audit_nvfp4_scales — IMP_AUDIT_NVFP4_SCALES: presence enables.
    if (std::getenv("IMP_AUDIT_NVFP4_SCALES") != nullptr)
        cfg.diagnostics.audit_nvfp4_scales = true;

    // gdn.layout_override — IMP_GDN_LAYOUT: string value.
    if (const char* e = std::getenv("IMP_GDN_LAYOUT"))
        cfg.gdn.layout_override = e;

    // ffn.sparsity_probe — IMP_FFN_SPARSITY_PROBE: '1' enables.
    if (const char* e = std::getenv("IMP_FFN_SPARSITY_PROBE"))
        cfg.ffn.sparsity_probe = (e[0] == '1');

    // ffn.sparsity_threshold — IMP_FFN_SPARSITY_THRESHOLD: float.
    if (const char* e = std::getenv("IMP_FFN_SPARSITY_THRESHOLD"))
        cfg.ffn.sparsity_threshold = parse_float(e, cfg.ffn.sparsity_threshold);

    // speculative.ngram — IMP_SPEC_NGRAM: '1' enables.
    if (const char* e = std::getenv("IMP_SPEC_NGRAM"))
        cfg.speculative.ngram = (e[0] == '1');

    // speculative.k — IMP_SPEC_K: int.
    if (const char* e = std::getenv("IMP_SPEC_K"))
        cfg.speculative.k = parse_int(e, cfg.speculative.k);

    // attention.fp8_prefill — IMP_NO_FP8_PREFILL: '1' sets fp8_prefill="never".
    if (const char* e = std::getenv("IMP_NO_FP8_PREFILL"))
        if (e[0] == '1')
            cfg.attention.fp8_prefill = "never";
}

}  // anonymous namespace

// -----------------------------------------------------------------------

std::string RuntimeConfig::find_default_path() {
    if (const char* p = std::getenv("IMP_CONFIG")) {
        if (file_exists(p))
            return p;
    }
    if (file_exists("./imp.conf"))
        return "./imp.conf";
    std::string home = home_dir();
    if (!home.empty()) {
        std::string user_path = home + "/.config/imp/imp.conf";
        if (file_exists(user_path))
            return user_path;
    }
    return {};
}

bool RuntimeConfig::load_from_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        IMP_LOG_ERROR("imp.conf: cannot open %s", path.c_str());
        return false;
    }

    std::string line;
    std::string section;
    int line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;
        // Strip comment from '#' onwards (unless inside quotes — we ignore that
        // edge case for the minimal parser; quote a value with " to keep #).
        size_t hash = line.find('#');
        if (hash != std::string::npos) {
            // Don't strip if inside quotes
            size_t q1 = line.find('"');
            size_t q2 = (q1 == std::string::npos) ? std::string::npos : line.find('"', q1 + 1);
            if (!(q1 != std::string::npos && q2 != std::string::npos && q1 < hash && hash < q2)) {
                line = line.substr(0, hash);
            }
        }
        std::string s = trim(line);
        if (s.empty())
            continue;

        if (s.front() == '[' && s.back() == ']') {
            section = trim(s.substr(1, s.size() - 2));
            continue;
        }

        size_t eq = s.find('=');
        if (eq == std::string::npos) {
            IMP_LOG_WARN("imp.conf:%d: ignoring malformed line: %s", line_no, s.c_str());
            continue;
        }
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));

        std::string dotted = section.empty() ? key : (section + "." + key);
        apply_one(*this, dotted, val);
    }
    return true;
}

void RuntimeConfig::apply_overrides(const std::vector<std::string>& kvs) {
    for (const auto& kv : kvs) {
        size_t eq = kv.find('=');
        if (eq == std::string::npos) {
            IMP_LOG_WARN("imp.conf override: ignoring malformed '%s' (expected key=value)", kv.c_str());
            continue;
        }
        std::string key = trim(kv.substr(0, eq));
        std::string val = trim(kv.substr(eq + 1));
        apply_one(*this, key, val);
    }
}

RuntimeConfig RuntimeConfig::load(const std::string& explicit_path,
                                  const std::vector<std::string>& overrides) {
    RuntimeConfig cfg;
    // Seed legacy IMP_* env vars first; file values + CLI overrides win on top.
    seed_from_env(cfg);
    std::string path = explicit_path.empty() ? find_default_path() : explicit_path;
    if (!path.empty()) {
        if (cfg.load_from_file(path)) {
            IMP_LOG_INFO("imp.conf loaded from %s", path.c_str());
        }
    } else {
        IMP_LOG_INFO("imp.conf: no config file found, using built-in defaults");
    }
    cfg.apply_overrides(overrides);
    return cfg;
}

// ---- Pending-config handoff (tool main → Engine::init) ------------------
//
// Replaces the former RuntimeConfig::current()/install() singleton
// (Phase 5 Track D follow-up, 2026-05-20). The static storage now lives
// for at most one Engine construction: tool main stashes the loaded
// config via set_pending_runtime_config(); imp_context_create() takes
// it via take_pending_runtime_config() and hands it to Engine::init.
// If the take() call finds no pending config (library users that never
// called the setter), it returns a freshly loaded RuntimeConfig (with
// the seed_from_env() legacy IMP_* compat path) so behavior matches
// the historical first-touch-of-singleton initialization.

namespace {
RuntimeConfig& pending_slot() {
    static RuntimeConfig slot;
    return slot;
}
bool& pending_set() {
    static bool b = false;
    return b;
}
}  // anonymous namespace

void set_pending_runtime_config(RuntimeConfig cfg) {
    pending_slot() = std::move(cfg);
    pending_set() = true;
}

RuntimeConfig take_pending_runtime_config() {
    if (pending_set()) {
        pending_set() = false;
        return std::move(pending_slot());
    }
    // No tool-main install — fall back to env-seeded defaults so tests
    // and library users that skip RuntimeConfig::load() still observe
    // legacy IMP_* env values.
    RuntimeConfig cfg;
    seed_from_env(cfg);
    return cfg;
}

}  // namespace imp
