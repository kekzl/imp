#pragma once

// Process-wide diagnostic / runtime-mode flags, snapshotted from
// RuntimeConfig once at startup (tool main calls process_diag_install()).
//
// These exist because a handful of leaf utilities — graph_diag inline
// helpers, executor_debug inline helpers, the CUDA-graph capture-mode
// selector, PDL gate, vision-encoder graph gate — are called from
// hundreds of sites that don't otherwise carry a RuntimeConfig and
// can't reasonably take one as a parameter. Snapshotting at startup
// trades the former per-call RuntimeConfig::current() accessor for a
// narrow set of typed POD reads with no global mutable state during
// inference.
//
// Set via process_diag_install(); read via the typed accessors below.
// Default values match the RuntimeConfig defaults so library users
// that never call install() still get sane behaviour.

#include <string>

namespace imp {

struct RuntimeConfig;  // fwd

void process_diag_install(const RuntimeConfig& cfg);

// Diagnostics
bool process_diag_debug_forward();
bool process_diag_debug_template();
bool process_diag_graph_diag();
bool process_diag_nvfp4_force_dequant();
bool process_diag_log_gemm_algo();
bool process_diag_audit_nvfp4_scales();
const char* process_diag_dump_hidden_dir();   // nullptr when unset; "1"/"all" → "/tmp"
const char* process_diag_graph_dump_dir();    // nullptr when unset

// Runtime modes
bool process_diag_no_pdl();
bool process_diag_no_vision_graph();
// "global" | "relaxed" | "thread_local" (default "relaxed")
const std::string& process_diag_graph_capture_mode();
bool process_diag_prefill_graph_enabled();

// GEMM
// Mirrored at engine init from cfg.runtime.deterministic_gemm; some arch
// resolvers (Gemma-4, FP8 KV) promote this flag during init_resolve_*.
// process_diag_set_deterministic_gemm() lets engine_init_resolver update
// the cache in place (replaces the former RuntimeConfig::install dual-write).
bool process_diag_deterministic_gemm();
void process_diag_set_deterministic_gemm(bool v);
// FP16-accumulate cuBLAS prefill GEMMs (gemm.cublas_fp16_acc) — read by the
// free-function gemm() in compute/gemm.cu, which carries no RuntimeConfig.
// "auto" (default) is resolved per-arch by init_resolve_quant_flags_ via the
// setter (ON except Gemma-3/4 and gpt-oss); install() maps auto → off so
// engine-less tools keep 32F accumulate.
bool process_diag_cublas_fp16_acc();
void process_diag_set_cublas_fp16_acc(bool v);

// Attention
bool process_diag_attention_splitk_pipe();
bool process_diag_attention_fp8_tile();
bool process_diag_fa2_f16acc();  // f16-accumulate QK^T in the fp16-qk FA2 kernel (#597)
bool process_diag_fa2_pv_f16acc();  // f16-accumulate the PV MMA too (#667 follow-up)
// test hooks (mirror process_diag_set_cublas_fp16_acc)
void process_diag_set_fa2_f16acc(bool v);
void process_diag_set_fa2_pv_f16acc(bool v);
bool process_diag_fp8_qk_scaled();  // amax-scaled e4m3 fp8-QK (#680)
void process_diag_set_fp8_qk_scaled(bool v);
// test hook: force the paged-decode split-K path onto its single-split GQA/MHA
// fallback even on a clean launch, so the fallback can be verified against the
// split-K result without provoking a real cudaErrorInvalidValue. Default off.
bool process_diag_force_splitk_fallback();
void process_diag_set_force_splitk_fallback(bool v);
// "auto" | "always" | "never" (default "auto"); attention_mxfp4_available()
// only enables MXFP4 attention when mode == "always".
const std::string& process_diag_attention_mxfp4_mode();
// #846 NVFP4-attention spike knobs (only meaningful when mxfp4 == "always").
bool process_diag_mxfp4_blockscale();
bool process_diag_mxfp4_ksmooth();
bool process_diag_mxfp4_pv_fp4();
void process_diag_set_mxfp4_blockscale(bool v);
void process_diag_set_mxfp4_ksmooth(bool v);
void process_diag_set_mxfp4_pv_fp4(bool v);
// ThriftAttention-style outlier promotion budget (0 = off, requires blockscale).
float process_diag_mxfp4_promote_budget();
void process_diag_set_mxfp4_promote_budget(float v);

// FFN
bool process_diag_ffn_sparsity_probe();

// MoE
int process_diag_moe_mr_nr();  // rows-per-block for NVFP4 MoE decode (4/8/16/32)
// Read at model-load time by weight_upload (Pass 2 expert offload budget
// + force-host-experts). No per-Engine context at that point.
int process_diag_moe_expert_overhead_pct();
int process_diag_moe_force_host_experts();

// GDN: layout override read at model-load time by hf_config_loader (no
// per-Engine context at that point in the loader pipeline).
const std::string& process_diag_gdn_layout_override();

}  // namespace imp
