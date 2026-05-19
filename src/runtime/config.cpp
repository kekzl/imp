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

    auto eq = [&](const char* k) { return dotted_key == k; };

    // [runtime]
    if (eq("runtime.deterministic_gemm"))
        cfg.runtime.deterministic_gemm = parse_bool(val, cfg.runtime.deterministic_gemm);
    else if (eq("runtime.cuda_graphs"))
        cfg.runtime.cuda_graphs = val;
    else if (eq("runtime.warmup"))
        cfg.runtime.warmup = parse_bool(val, cfg.runtime.warmup);
    else if (eq("runtime.max_seq_len"))
        cfg.runtime.max_seq_len = parse_int(val, cfg.runtime.max_seq_len);
    else if (eq("runtime.no_pdl"))
        cfg.runtime.no_pdl = parse_bool(val, cfg.runtime.no_pdl);
    else if (eq("runtime.debug_raw"))
        cfg.runtime.debug_raw = parse_bool(val, cfg.runtime.debug_raw);
    else if (eq("runtime.no_vision_graph"))
        cfg.runtime.no_vision_graph = parse_bool(val, cfg.runtime.no_vision_graph);
    else if (eq("runtime.graph_capture_mode"))
        cfg.runtime.graph_capture_mode = val;
    else if (eq("runtime.prefill_graph"))
        cfg.runtime.prefill_graph = parse_bool(val, cfg.runtime.prefill_graph);

    // [kv_cache]
    else if (eq("kv_cache.dtype"))
        cfg.kv_cache.dtype = val;
    else if (eq("kv_cache.allow_nondeterministic_fp8"))
        cfg.kv_cache.allow_nondeterministic_fp8 = parse_bool(val, cfg.kv_cache.allow_nondeterministic_fp8);
    else if (eq("kv_cache.fp8_auto_legacy"))
        cfg.kv_cache.fp8_auto_legacy = parse_bool(val, cfg.kv_cache.fp8_auto_legacy);
    else if (eq("kv_cache.bitdecoding_residual_tokens"))
        cfg.kv_cache.bitdecoding_residual_tokens = parse_int(val, cfg.kv_cache.bitdecoding_residual_tokens);
    else if (eq("kv_cache.bitdecoding_qk"))
        cfg.kv_cache.bitdecoding_qk = parse_bool(val, cfg.kv_cache.bitdecoding_qk);

    // [attention]
    else if (eq("attention.fp8_prefill"))
        cfg.attention.fp8_prefill = val;
    else if (eq("attention.fp8_fmha"))
        cfg.attention.fp8_fmha = val;
    else if (eq("attention.fmha_sm120"))
        cfg.attention.fmha_sm120 = val;
    else if (eq("attention.mxfp4"))
        cfg.attention.mxfp4 = val;
    else if (eq("attention.mxfp4_fp16_fallback"))
        cfg.attention.mxfp4_fp16_fallback = parse_bool(val, cfg.attention.mxfp4_fp16_fallback);
    else if (eq("attention.mxfp4_fp16_cache_policy"))
        cfg.attention.mxfp4_fp16_cache_policy = val;
    else if (eq("attention.fmha_blockscale"))
        cfg.attention.fmha_blockscale = val;
    else if (eq("attention.naive"))
        cfg.attention.naive = parse_bool(val, cfg.attention.naive);
    else if (eq("attention.no_cublas"))
        cfg.attention.no_cublas = parse_bool(val, cfg.attention.no_cublas);
    else if (eq("attention.force_cublas_decode"))
        cfg.attention.force_cublas_decode = parse_bool(val, cfg.attention.force_cublas_decode);
    else if (eq("attention.no_qknorm_fused"))
        cfg.attention.no_qknorm_fused = parse_bool(val, cfg.attention.no_qknorm_fused);
    else if (eq("attention.no_naive_swa"))
        cfg.attention.no_naive_swa = parse_bool(val, cfg.attention.no_naive_swa);
    else if (eq("attention.splitk_pipe"))
        cfg.attention.splitk_pipe = parse_bool(val, cfg.attention.splitk_pipe);
    else if (eq("attention.gate_concat"))
        cfg.attention.gate_concat = parse_bool(val, cfg.attention.gate_concat);
    else if (eq("attention.no_fmha_cluster"))
        cfg.attention.no_fmha_cluster = parse_bool(val, cfg.attention.no_fmha_cluster);

    // [moe]
    else if (eq("moe.expert_overhead_pct"))
        cfg.moe.expert_overhead_pct = parse_int(val, cfg.moe.expert_overhead_pct);
    else if (eq("moe.force_host_experts"))
        cfg.moe.force_host_experts = parse_int(val, cfg.moe.force_host_experts);
    else if (eq("moe.skip"))
        cfg.moe.skip = parse_bool(val, cfg.moe.skip);
    else if (eq("moe.force_fp16_sync"))
        cfg.moe.force_fp16_sync = parse_bool(val, cfg.moe.force_fp16_sync);
    else if (eq("moe.no_expert_cache"))
        cfg.moe.no_expert_cache = parse_bool(val, cfg.moe.no_expert_cache);
    else if (eq("moe.expert_cache_debug_parity"))
        cfg.moe.expert_cache_debug_parity = parse_bool(val, cfg.moe.expert_cache_debug_parity);
    else if (eq("moe.prefetch_top_k"))
        cfg.moe.prefetch_top_k = parse_int(val, cfg.moe.prefetch_top_k);
    else if (eq("moe.allow_graphs_under_offload"))
        cfg.moe.allow_graphs_under_offload = parse_bool(val, cfg.moe.allow_graphs_under_offload);
    else if (eq("moe.zero_workspace"))
        cfg.moe.zero_workspace = parse_bool(val, cfg.moe.zero_workspace);
    else if (eq("moe.no_shared_mlp"))
        cfg.moe.no_shared_mlp = parse_bool(val, cfg.moe.no_shared_mlp);
    else if (eq("moe.no_shexp_gate"))
        cfg.moe.no_shexp_gate = parse_bool(val, cfg.moe.no_shexp_gate);
    else if (eq("moe.no_cutlass3x"))
        cfg.moe.no_cutlass3x = parse_bool(val, cfg.moe.no_cutlass3x);
    else if (eq("moe.reserve_mib"))
        cfg.moe.reserve_mib = parse_int(val, cfg.moe.reserve_mib);
    else if (eq("moe.nvfp4_device_args"))
        cfg.moe.nvfp4_device_args = parse_bool(val, cfg.moe.nvfp4_device_args);
    else if (eq("moe.nvfp4_smallM"))
        cfg.moe.nvfp4_smallM = parse_bool(val, cfg.moe.nvfp4_smallM);
    else if (eq("moe.nvfp4_smallM_threshold"))
        cfg.moe.nvfp4_smallM_threshold = parse_int(val, cfg.moe.nvfp4_smallM_threshold);
    else if (eq("moe.mr_nr"))
        cfg.moe.mr_nr = parse_int(val, cfg.moe.mr_nr);

    // [gdn]
    else if (eq("gdn.fp32_scan"))
        cfg.gdn.fp32_scan = parse_bool(val, cfg.gdn.fp32_scan);
    else if (eq("gdn.fp32_out"))
        cfg.gdn.fp32_out = parse_bool(val, cfg.gdn.fp32_out);
    else if (eq("gdn.norm_eps_override"))
        cfg.gdn.norm_eps_override = parse_float(val, cfg.gdn.norm_eps_override);
    else if (eq("gdn.layout_override"))
        cfg.gdn.layout_override = val;
    else if (eq("gdn.ref_kernel"))
        cfg.gdn.ref_kernel = parse_bool(val, cfg.gdn.ref_kernel);
    else if (eq("gdn.vhead_reorder"))
        cfg.gdn.vhead_reorder = parse_bool(val, cfg.gdn.vhead_reorder);

    // [gemm]
    else if (eq("gemm.no_dp4a"))
        cfg.gemm.no_dp4a = parse_bool(val, cfg.gemm.no_dp4a);
    else if (eq("gemm.no_dp4a_gemv"))
        cfg.gemm.no_dp4a_gemv = parse_bool(val, cfg.gemm.no_dp4a_gemv);
    else if (eq("gemm.no_dp4a_lm"))
        cfg.gemm.no_dp4a_lm = parse_bool(val, cfg.gemm.no_dp4a_lm);
    else if (eq("gemm.no_mmvq"))
        cfg.gemm.no_mmvq = parse_bool(val, cfg.gemm.no_mmvq);
    else if (eq("gemm.no_mmvq_q8_0"))
        cfg.gemm.no_mmvq_q8_0 = parse_bool(val, cfg.gemm.no_mmvq_q8_0);
    else if (eq("gemm.q4k_imma_enabled"))
        cfg.gemm.q4k_imma_enabled = parse_bool(val, cfg.gemm.q4k_imma_enabled);

    // [gemma4]
    else if (eq("gemma4.fp32_gemm_out"))
        cfg.gemma4.fp32_gemm_out = parse_bool(val, cfg.gemma4.fp32_gemm_out);
    else if (eq("gemma4.no_graphs"))
        cfg.gemma4.no_graphs = parse_bool(val, cfg.gemma4.no_graphs);
    else if (eq("gemma4.force_mmvq"))
        cfg.gemma4.force_mmvq = parse_bool(val, cfg.gemma4.force_mmvq);
    else if (eq("gemma4.fp32_expert_down"))
        cfg.gemma4.fp32_expert_down = parse_bool(val, cfg.gemma4.fp32_expert_down);
    else if (eq("gemma4.no_decode_fast"))
        cfg.gemma4.no_decode_fast = parse_bool(val, cfg.gemma4.no_decode_fast);
    else if (eq("gemma4.no_post_ffw_1"))
        cfg.gemma4.no_post_ffw_1 = parse_bool(val, cfg.gemma4.no_post_ffw_1);
    else if (eq("gemma4.ggml_prefill"))
        cfg.gemma4.ggml_prefill = parse_bool(val, cfg.gemma4.ggml_prefill);

    // [generation]
    else if (eq("generation.no_logit_softcap"))
        cfg.generation.no_logit_softcap = parse_bool(val, cfg.generation.no_logit_softcap);
    else if (eq("generation.lm_dequant_fp16"))
        cfg.generation.lm_dequant_fp16 = parse_bool(val, cfg.generation.lm_dequant_fp16);
    else if (eq("generation.think_budget"))
        cfg.generation.think_budget = parse_int(val, cfg.generation.think_budget);
    else if (eq("generation.force_bos"))
        cfg.generation.force_bos = parse_bool(val, cfg.generation.force_bos);
    else if (eq("generation.no_ban"))
        cfg.generation.no_ban = parse_bool(val, cfg.generation.no_ban);
    else if (eq("generation.mtp_no_rope"))
        cfg.generation.mtp_no_rope = parse_bool(val, cfg.generation.mtp_no_rope);

    // [server]
    else if (eq("server.prefix_cache"))
        cfg.server.prefix_cache = parse_bool(val, cfg.server.prefix_cache);

    // [bench]
    else if (eq("bench.generate"))
        cfg.bench.generate = parse_bool(val, cfg.bench.generate);

    // [paths]
    else if (eq("paths.mmproj"))
        cfg.paths.mmproj = val;

    // [diagnostics]
    else if (eq("diagnostics.debug_forward"))
        cfg.diagnostics.debug_forward = parse_bool(val, cfg.diagnostics.debug_forward);
    else if (eq("diagnostics.debug_gemm_dispatch"))
        cfg.diagnostics.debug_gemm_dispatch = parse_bool(val, cfg.diagnostics.debug_gemm_dispatch);
    else if (eq("diagnostics.debug_template"))
        cfg.diagnostics.debug_template = parse_bool(val, cfg.diagnostics.debug_template);
    else if (eq("diagnostics.dump_hidden_dir"))
        cfg.diagnostics.dump_hidden_dir = val;
    else if (eq("diagnostics.dump_logits_dir"))
        cfg.diagnostics.dump_logits_dir = val;
    else if (eq("diagnostics.dump_routing_dir"))
        cfg.diagnostics.dump_routing_dir = val;
    else if (eq("diagnostics.dump_tokens"))
        cfg.diagnostics.dump_tokens = parse_bool(val, cfg.diagnostics.dump_tokens);
    else if (eq("diagnostics.exit_layer"))
        cfg.diagnostics.exit_layer = parse_int(val, cfg.diagnostics.exit_layer);
    else if (eq("diagnostics.profile"))
        cfg.diagnostics.profile = parse_bool(val, cfg.diagnostics.profile);
    else if (eq("diagnostics.graph_diag"))
        cfg.diagnostics.graph_diag = parse_bool(val, cfg.diagnostics.graph_diag);
    else if (eq("diagnostics.graph_dump_dir"))
        cfg.diagnostics.graph_dump_dir = val;
    else if (eq("diagnostics.nvfp4_force_dequant"))
        cfg.diagnostics.nvfp4_force_dequant = parse_bool(val, cfg.diagnostics.nvfp4_force_dequant);
    else if (eq("diagnostics.log_gemm_algo"))
        cfg.diagnostics.log_gemm_algo = parse_bool(val, cfg.diagnostics.log_gemm_algo);
    else if (eq("diagnostics.mtp_pattern_log"))
        cfg.diagnostics.mtp_pattern_log = parse_bool(val, cfg.diagnostics.mtp_pattern_log);
    else if (eq("diagnostics.mtp_prenorm_h"))
        cfg.diagnostics.mtp_prenorm_h = parse_bool(val, cfg.diagnostics.mtp_prenorm_h);
    else if (eq("diagnostics.audit_nvfp4_scales"))
        cfg.diagnostics.audit_nvfp4_scales = parse_bool(val, cfg.diagnostics.audit_nvfp4_scales);

    // [ffn]
    else if (eq("ffn.sparsity_probe"))
        cfg.ffn.sparsity_probe = parse_bool(val, cfg.ffn.sparsity_probe);
    else if (eq("ffn.sparsity_threshold"))
        cfg.ffn.sparsity_threshold = parse_float(val, cfg.ffn.sparsity_threshold);

    else {
        IMP_LOG_WARN("imp.conf: unknown key '%s' (value '%s') — ignoring", dotted_key.c_str(), val.c_str());
    }
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
    // moe.reserve_mib — IMP_MOE_RESERVE_MIB: integer MiB.
    if (const char* e = std::getenv("IMP_MOE_RESERVE_MIB"))
        cfg.moe.reserve_mib = parse_int(e, cfg.moe.reserve_mib);

    // kv_cache.bitdecoding_qk — IMP_USE_BITDECODING_QK: '1' enables.
    if (const char* e = std::getenv("IMP_USE_BITDECODING_QK"))
        cfg.kv_cache.bitdecoding_qk = (e[0] == '1');

    // moe.nvfp4_device_args — IMP_NVFP4_DEVICE_ARGS: '0' disables, default ON.
    if (const char* e = std::getenv("IMP_NVFP4_DEVICE_ARGS"))
        cfg.moe.nvfp4_device_args = (std::atoi(e) != 0);

    // moe.nvfp4_smallM — IMP_NVFP4_SMALLM: integer != 0 enables.
    if (const char* e = std::getenv("IMP_NVFP4_SMALLM"))
        cfg.moe.nvfp4_smallM = (std::atoi(e) != 0);

    // moe.nvfp4_smallM_threshold — IMP_NVFP4_SMALLM_THRESHOLD: clamped int.
    if (const char* e = std::getenv("IMP_NVFP4_SMALLM_THRESHOLD")) {
        int v = std::atoi(e);
        if (v < 0) v = 0;
        if (v > 128) v = 128;
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

// ---- Process-wide singleton ---------------------------------------------
//
// First touch of the singleton seeds from legacy IMP_* env vars. This means
// tests / library users that never call RuntimeConfig::load() still observe
// env-based defaults. Subsequent install() calls overwrite the singleton.

namespace {
RuntimeConfig& mutable_current() {
    static RuntimeConfig instance = []() {
        RuntimeConfig cfg;
        seed_from_env(cfg);
        return cfg;
    }();
    return instance;
}
}  // anonymous namespace

const RuntimeConfig& RuntimeConfig::current() { return mutable_current(); }

void RuntimeConfig::install(const RuntimeConfig& cfg) { mutable_current() = cfg; }

}  // namespace imp
