#include "core/process_diag.h"

namespace imp {

namespace {

ProcessDiag& slot() {
    static ProcessDiag d;
    return d;
}

}  // namespace

const ProcessDiag& process_diag_current() { return slot(); }
void process_diag_set(const ProcessDiag& d) { slot() = d; }

bool process_diag_debug_forward() { return slot().debug_forward; }
bool process_diag_debug_template() { return slot().debug_template; }
bool process_diag_graph_diag() { return slot().graph_diag; }
bool process_diag_nvfp4_force_dequant() { return slot().nvfp4_force_dequant; }
bool process_diag_prefill_graph_ignore_dequant_cap() { return slot().prefill_graph_ignore_dequant_cap; }
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
int process_diag_upload_ring_depth() { return slot().upload_ring_depth; }
int process_diag_upload_ring_chunk_mib() { return slot().upload_ring_chunk_mib; }
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
bool process_diag_fa2_dense_2cta() { return slot().attention_fa2_dense_2cta; }
int process_diag_paged_fp8_multitok() { return slot().attention_paged_fp8_multitok; }
void process_diag_set_paged_fp8_multitok(int v) { slot().attention_paged_fp8_multitok = v; }
int process_diag_paged_nvfp4_multitok() { return slot().attention_paged_nvfp4_multitok; }
void process_diag_set_paged_nvfp4_multitok(int v) { slot().attention_paged_nvfp4_multitok = v; }
int process_diag_paged_f16_multitok() { return slot().attention_paged_f16_multitok; }
void process_diag_set_paged_f16_multitok(int v) { slot().attention_paged_f16_multitok = v; }
int process_diag_paged_f16_hpc() { return slot().attention_paged_f16_hpc; }
void process_diag_set_paged_f16_hpc(int v) { slot().attention_paged_f16_hpc = v; }
int process_diag_paged_nvfp4_hpc() { return slot().attention_paged_nvfp4_hpc; }
void process_diag_set_paged_nvfp4_hpc(int v) { slot().attention_paged_nvfp4_hpc = v; }
void process_diag_set_fa2_dense_2cta(bool v) { slot().attention_fa2_dense_2cta = v; }
bool process_diag_fa2_heavy_first() { return slot().attention_fa2_heavy_first; }
void process_diag_set_fa2_heavy_first(bool v) { slot().attention_fa2_heavy_first = v; }
int process_diag_fa2_hd256_bkv() { return slot().attention_fa2_hd256_bkv; }
void process_diag_set_fa2_hd256_bkv(int v) { slot().attention_fa2_hd256_bkv = v; }
bool process_diag_fp8_qk_scaled() { return slot().attention_fp8_qk_scaled; }
void process_diag_set_fp8_qk_scaled(bool v) { slot().attention_fp8_qk_scaled = v; }
int process_diag_nvfp4_cutlass_streamk() { return slot().nvfp4_cutlass_streamk; }
void process_diag_set_nvfp4_cutlass_streamk(int v) { slot().nvfp4_cutlass_streamk = v; }
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
bool process_diag_verify_row_parity() { return slot().verify_row_parity; }
void process_diag_set_verify_row_parity(bool v) { slot().verify_row_parity = v; }
int process_diag_moe_expert_overhead_pct() { return slot().moe_expert_overhead_pct; }
int process_diag_moe_force_host_experts() { return slot().moe_force_host_experts; }
bool process_diag_moe_pin_host_experts() { return slot().moe_pin_host_experts; }
const std::string& process_diag_gdn_layout_override() { return slot().gdn_layout_override; }

}  // namespace imp
