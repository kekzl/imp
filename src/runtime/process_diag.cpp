#include "runtime/process_diag.h"
#include "runtime/config.h"
#include "core/logging.h"

namespace imp {

namespace {

struct ProcessDiag {
    // Diagnostics
    bool debug_forward = false;
    bool debug_template = false;
    bool graph_diag = false;
    bool nvfp4_force_dequant = false;
    bool log_gemm_algo = false;
    bool audit_nvfp4_scales = false;
    std::string dump_hidden_dir;
    std::string dump_hidden_dir_resolved;  // "1"/"all" → "/tmp"
    std::string graph_dump_dir;

    // Runtime
    bool no_pdl = false;
    bool no_vision_graph = false;
    std::string graph_capture_mode = "relaxed";
    bool prefill_graph_enabled = true;

    // GEMM (may be promoted in place by engine_init_resolver)
    bool deterministic_gemm = false;
    bool cublas_fp16_acc = false;

    // Attention
    bool attention_splitk_pipe = true;
    bool attention_fp8_tile = true;
    bool attention_fp8_tile_gqa = true;
    bool attention_fa2_f16acc = true;     // matches the config.h default
    bool attention_fa2_pv_f16acc = true;  // matches the config.h default
    bool attention_fa2_hd256 = true;  // matches the config.h default (on since #932)
    bool attention_fp8_qk_scaled = false;
    bool force_splitk_fallback = false;  // test hook
    std::string attention_mxfp4_mode = "auto";
    bool mxfp4_blockscale = false;
    bool mxfp4_ksmooth = false;
    bool mxfp4_pv_fp4 = false;
    float mxfp4_promote_budget = 0.0f;

    // FFN
    bool ffn_sparsity_probe = false;

    // MoE
    int moe_mr_nr = 8;
    int moe_expert_overhead_pct = 10;
    int moe_force_host_experts = 0;
    bool moe_pin_host_experts = false;

    // GDN
    std::string gdn_layout_override;
};

ProcessDiag& slot() {
    static ProcessDiag d;
    return d;
}

}  // namespace

void process_diag_install(const RuntimeConfig& cfg) {
    auto& d = slot();
    // Log level first, so anything this function or its callers log afterwards
    // already obeys it. An unknown word warns rather than silently picking a
    // level: the whole point of this key is that the level used to be
    // unsettable, and a typo that quietly resolves to INFO would restore that.
    {
        LogLevel lvl;
        if (log_level_from_string(cfg.diagnostics.log_level.c_str(), lvl)) {
            log_set_level(lvl);
        } else {
            IMP_LOG_WARN(
                "diagnostics.log_level: unknown value '%s' — keeping the current level "
                "(expected debug|info|warn|error|fatal)",
                cfg.diagnostics.log_level.c_str());
        }
    }
    d.debug_forward = cfg.diagnostics.debug_forward;
    d.debug_template = cfg.diagnostics.debug_template;
    d.graph_diag = cfg.diagnostics.graph_diag;
    d.nvfp4_force_dequant = cfg.diagnostics.nvfp4_force_dequant;
    d.log_gemm_algo = cfg.diagnostics.log_gemm_algo;
    d.audit_nvfp4_scales = cfg.diagnostics.audit_nvfp4_scales;
    d.dump_hidden_dir = cfg.diagnostics.dump_hidden_dir;
    if (d.dump_hidden_dir == "1" || d.dump_hidden_dir == "all") {
        d.dump_hidden_dir_resolved = "/tmp";
    } else {
        d.dump_hidden_dir_resolved = d.dump_hidden_dir;
    }
    d.graph_dump_dir = cfg.diagnostics.graph_dump_dir;
    d.no_pdl = cfg.runtime.no_pdl;
    d.no_vision_graph = cfg.runtime.no_vision_graph;
    d.graph_capture_mode = cfg.runtime.graph_capture_mode;
    d.prefill_graph_enabled = cfg.runtime.prefill_graph;
    d.deterministic_gemm = cfg.runtime.deterministic_gemm;
    // "auto" resolves per-arch at engine init (init_resolve_quant_flags_ →
    // process_diag_set_cublas_fp16_acc); standalone tools without an engine
    // treat auto as off.
    d.cublas_fp16_acc = (cfg.gemm.cublas_fp16_acc == "on");
    d.attention_splitk_pipe = cfg.attention.splitk_pipe;
    d.attention_fp8_tile = cfg.attention.fp8_tile;
    d.attention_fp8_tile_gqa = cfg.attention.fp8_tile_gqa;
    d.attention_fa2_f16acc = cfg.attention.fa2_f16acc;
    d.attention_fa2_pv_f16acc = cfg.attention.fa2_pv_f16acc;
    d.attention_fa2_hd256 = cfg.attention.fa2_hd256;
    d.attention_fp8_qk_scaled = cfg.attention.fp8_qk_scaled;
    d.attention_mxfp4_mode = cfg.attention.mxfp4;
    d.mxfp4_blockscale = cfg.attention.mxfp4_blockscale;
    d.mxfp4_ksmooth = cfg.attention.mxfp4_ksmooth;
    d.mxfp4_pv_fp4 = cfg.attention.mxfp4_pv_fp4;
    d.mxfp4_promote_budget = cfg.attention.mxfp4_promote_budget;
    d.ffn_sparsity_probe = cfg.ffn.sparsity_probe;
    d.moe_mr_nr = cfg.moe.mr_nr;
    d.moe_expert_overhead_pct = cfg.moe.expert_overhead_pct;
    d.moe_force_host_experts = cfg.moe.force_host_experts;
    d.moe_pin_host_experts = cfg.moe.pin_host_experts;
    d.gdn_layout_override = cfg.gdn.layout_override;
}

bool process_diag_debug_forward() { return slot().debug_forward; }
bool process_diag_debug_template() { return slot().debug_template; }
bool process_diag_graph_diag() { return slot().graph_diag; }
bool process_diag_nvfp4_force_dequant() { return slot().nvfp4_force_dequant; }
bool process_diag_log_gemm_algo() { return slot().log_gemm_algo; }
bool process_diag_audit_nvfp4_scales() { return slot().audit_nvfp4_scales; }
const char* process_diag_dump_hidden_dir() {
    return slot().dump_hidden_dir_resolved.empty() ? nullptr
                                                   : slot().dump_hidden_dir_resolved.c_str();
}
const char* process_diag_graph_dump_dir() {
    return slot().graph_dump_dir.empty() ? nullptr : slot().graph_dump_dir.c_str();
}
bool process_diag_no_pdl() { return slot().no_pdl; }
bool process_diag_no_vision_graph() { return slot().no_vision_graph; }
const std::string& process_diag_graph_capture_mode() { return slot().graph_capture_mode; }
bool process_diag_prefill_graph_enabled() { return slot().prefill_graph_enabled; }
bool process_diag_deterministic_gemm() { return slot().deterministic_gemm; }
bool process_diag_cublas_fp16_acc() { return slot().cublas_fp16_acc; }
void process_diag_set_cublas_fp16_acc(bool v) { slot().cublas_fp16_acc = v; }
void process_diag_set_deterministic_gemm(bool v) { slot().deterministic_gemm = v; }
bool process_diag_attention_splitk_pipe() { return slot().attention_splitk_pipe; }
bool process_diag_attention_fp8_tile() { return slot().attention_fp8_tile; }
bool process_diag_attention_fp8_tile_gqa() { return slot().attention_fp8_tile_gqa; }
bool process_diag_fa2_f16acc() { return slot().attention_fa2_f16acc; }
bool process_diag_fa2_pv_f16acc() { return slot().attention_fa2_pv_f16acc; }
void process_diag_set_fa2_f16acc(bool v) { slot().attention_fa2_f16acc = v; }
void process_diag_set_fa2_pv_f16acc(bool v) { slot().attention_fa2_pv_f16acc = v; }
bool process_diag_fa2_hd256() { return slot().attention_fa2_hd256; }
void process_diag_set_fa2_hd256(bool v) { slot().attention_fa2_hd256 = v; }
bool process_diag_fp8_qk_scaled() { return slot().attention_fp8_qk_scaled; }
void process_diag_set_fp8_qk_scaled(bool v) { slot().attention_fp8_qk_scaled = v; }
bool process_diag_force_splitk_fallback() { return slot().force_splitk_fallback; }
void process_diag_set_force_splitk_fallback(bool v) { slot().force_splitk_fallback = v; }
const std::string& process_diag_attention_mxfp4_mode() { return slot().attention_mxfp4_mode; }
bool process_diag_mxfp4_blockscale() { return slot().mxfp4_blockscale; }
bool process_diag_mxfp4_ksmooth() { return slot().mxfp4_ksmooth; }
bool process_diag_mxfp4_pv_fp4() { return slot().mxfp4_pv_fp4; }
void process_diag_set_mxfp4_ksmooth(bool v) { slot().mxfp4_ksmooth = v; }
void process_diag_set_mxfp4_pv_fp4(bool v) { slot().mxfp4_pv_fp4 = v; }
float process_diag_mxfp4_promote_budget() { return slot().mxfp4_promote_budget; }
void process_diag_set_mxfp4_promote_budget(float v) { slot().mxfp4_promote_budget = v; }
bool process_diag_ffn_sparsity_probe() { return slot().ffn_sparsity_probe; }
int process_diag_moe_mr_nr() { return slot().moe_mr_nr; }
int process_diag_moe_expert_overhead_pct() { return slot().moe_expert_overhead_pct; }
int process_diag_moe_force_host_experts() { return slot().moe_force_host_experts; }
bool process_diag_moe_pin_host_experts() { return slot().moe_pin_host_experts; }
const std::string& process_diag_gdn_layout_override() { return slot().gdn_layout_override; }

}  // namespace imp
