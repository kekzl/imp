#include "runtime/process_diag_install.h"
#include "runtime/config.h"
#include "core/process_diag.h"
#include "core/logging.h"

namespace imp {

void process_diag_install(const RuntimeConfig& cfg) {
    // Copy-modify-set: fields that no RuntimeConfig key feeds (test hooks such
    // as paged_*_hpc, force_splitk_fallback) keep whatever a setter put there.
    ProcessDiag d = process_diag_current();
    // Log level first, so anything this function or its callers log afterwards
    // already obeys it. An unknown word warns rather than silently picking a
    // level: the whole point of this key is that the level used to be
    // unsettable, and a typo that quietly resolves to INFO would restore that.
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
    d.nvfp4_force_dequant = cfg.diagnostics.nvfp4_force_dequant;
    d.prefill_graph_ignore_dequant_cap = cfg.diagnostics.prefill_graph_ignore_dequant_cap;
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
    d.upload_ring_depth = cfg.vram.upload_ring_depth;
    d.upload_ring_chunk_mib = cfg.vram.upload_ring_chunk_mib;
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
    d.attention_fa2_hd256_bkv = cfg.attention.fa2_hd256_bkv;
    d.attention_fa2_dense_2cta = cfg.attention.fa2_dense_2cta;
    d.attention_paged_fp8_multitok = cfg.attention.paged_fp8_multitok;
    d.attention_paged_nvfp4_multitok = cfg.attention.paged_nvfp4_multitok;
    d.attention_paged_f16_multitok = cfg.attention.paged_f16_multitok;
    d.attention_fa2_heavy_first = cfg.attention.fa2_heavy_first;
    d.attention_fp8_qk_scaled = cfg.attention.fp8_qk_scaled;
    d.nvfp4_cutlass_streamk = cfg.gemm.nvfp4_cutlass_streamk;
    d.attention_mxfp4_mode = cfg.attention.mxfp4;
    d.mxfp4_blockscale = cfg.attention.mxfp4_blockscale;
    d.mxfp4_ksmooth = cfg.attention.mxfp4_ksmooth;
    d.mxfp4_pv_fp4 = cfg.attention.mxfp4_pv_fp4;
    d.mxfp4_promote_budget = cfg.attention.mxfp4_promote_budget;
    d.ffn_sparsity_probe = cfg.ffn.sparsity_probe;
    d.moe_mr_nr = cfg.moe.mr_nr;
    d.verify_row_parity = cfg.speculative.verify_row_parity;
    d.moe_expert_overhead_pct = cfg.moe.expert_overhead_pct;
    d.moe_force_host_experts = cfg.moe.force_host_experts;
    d.moe_pin_host_experts = cfg.moe.pin_host_experts;
    d.gdn_layout_override = cfg.gdn.layout_override;
    process_diag_set(d);
}

}  // namespace imp
