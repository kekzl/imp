#include "runtime/process_diag_install.h"
#include "runtime/config.h"
#include "core/process_diag.h"
#include "core/logging.h"

namespace imp {

void process_diag_install(const RuntimeConfig& cfg) {
    // Copy-modify-set: fields that no RuntimeConfig key feeds (test hooks such
    // as paged_*_hpc, force_splitk_fallback) keep whatever a setter put there.
    ProcessDiag d = process_diag_current();
    // Log level first, so anything this function or its callers log
    // afterwards already obeys it. An unknown word warns rather than
    // silently picking a level: a typo that resolves to INFO would make
    // this key effectively unsettable.
    {
        if (const auto lvl = log_level_from_string(cfg.diagnostics.log_level)) {
            log_set_level(*lvl);
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
    d.prefill_graph_enabled = cfg.runtime.prefill_graph;
    d.deterministic_gemm = cfg.runtime.deterministic_gemm;
    d.deterministic = cfg.runtime.deterministic;
    // "auto" resolves per-arch at engine init (init_resolve_quant_flags_ →
    // process_diag_set_cublas_fp16_acc); standalone tools without an engine
    // treat auto as off.
    d.cublas_fp16_acc = (cfg.gemm.cublas_fp16_acc == "on");
    // runtime.deterministic: no timed cuBLASLt algo pick.
    d.nvfp4_cublaslt_min_m = cfg.runtime.deterministic ? 0 : imp::cfg::kNvfp4CublasltMinM;
    // "auto" resolves per model at engine init (init_resolve_quant_flags_); without an engine: on.
    d.nvfp4_prompt_streamk = cfg.gemm.nvfp4_prompt_streamk != "off";
    d.moe_force_host_experts = cfg.moe.force_host_experts;
    d.moe_pin_host_experts = cfg.moe.pin_host_experts;
    d.moe_host_expert_pool_mib = cfg.moe.host_expert_pool_mib;
    d.gdn_layout_override = cfg.gdn.layout_override;
    process_diag_set(d);
}

}  // namespace imp
