#pragma once

// Process-wide diagnostic / runtime-mode flags, snapshotted from
// RuntimeConfig once at startup by process_diag_install()
// (runtime/process_diag_install.h, called from the tool mains and Engine::init).
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

// The snapshot itself. Filled from RuntimeConfig by process_diag_install()
// (runtime/process_diag_install.h); the defaults below are the library-user
// values when nothing installs.
struct ProcessDiag {
    // Diagnostics
    bool debug_forward = false;
    bool debug_template = false;
    bool graph_diag = false;
    bool nvfp4_force_dequant = false;
    bool prefill_graph_ignore_dequant_cap = false;
    bool log_gemm_algo = false;
    bool audit_nvfp4_scales = false;
    std::string dump_hidden_dir;
    std::string dump_hidden_dir_resolved;  // "1"/"all" → "/tmp"
    std::string graph_dump_dir;

    // Runtime
    bool no_pdl = false;
    // Pinned staging ring for the weight upload (#1653).
    int upload_ring_depth = 4;
    int upload_ring_chunk_mib = 4;
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
    int attention_fa2_hd256_bkv = 64;  // matches the config.h default
    bool attention_fa2_dense_2cta = true;  // matches the config.h default
    int attention_paged_fp8_multitok = 4;  // matches the config.h default
    int attention_paged_nvfp4_multitok = 4;  // matches the config.h default
    int attention_paged_f16_multitok = 4;    // matches the config.h default
    int attention_paged_f16_hpc = 0;         // 0 = auto; tests sweep 1/2/4
    int attention_paged_nvfp4_hpc = 0;       // 0 = auto; tests sweep 1/2/3/4
    bool attention_fa2_heavy_first = true;  // matches the config.h default
    bool attention_fp8_qk_scaled = false;
    int nvfp4_cutlass_streamk = 1;  // matches the config.h default
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
    bool verify_row_parity = false;
    int moe_expert_overhead_pct = 10;
    int moe_force_host_experts = 0;
    bool moe_pin_host_experts = false;

    // GDN
    std::string gdn_layout_override;
};

const ProcessDiag& process_diag_current();
void process_diag_set(const ProcessDiag& d);

// Diagnostics
bool process_diag_debug_forward();
bool process_diag_debug_template();
bool process_diag_graph_diag();
bool process_diag_nvfp4_force_dequant();
bool process_diag_prefill_graph_ignore_dequant_cap();
bool process_diag_log_gemm_algo();
bool process_diag_audit_nvfp4_scales();
const char* process_diag_dump_hidden_dir();   // nullptr when unset; "1"/"all" → "/tmp"
const char* process_diag_graph_dump_dir();    // nullptr when unset

// Runtime modes
bool process_diag_no_pdl();

// Pinned staging ring for the weight upload (#1653). Depth and chunk size in
// MiB; the defaults are 4 and 4. Load-time only - the ring is built once per
// upload pass and destroyed with it.
int process_diag_upload_ring_depth();
int process_diag_upload_ring_chunk_mib();
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
bool process_diag_attention_fp8_tile_gqa();
bool process_diag_fa2_f16acc();  // f16-accumulate QK^T in the fp16-qk FA2 kernel (#597)
bool process_diag_fa2_pv_f16acc();  // f16-accumulate the PV MMA too (#667 follow-up)
// test hooks (mirror process_diag_set_cublas_fp16_acc)
void process_diag_set_fa2_f16acc(bool v);
void process_diag_set_fa2_pv_f16acc(bool v);
bool process_diag_fa2_hd256();  // HD=256 FA2 port (attention.fa2_hd256, default on since #932)
void process_diag_set_fa2_hd256(bool v);
bool process_diag_fa2_dense_2cta();  // dense Bq=128 FA2 at 2 CTAs/SM (attention.fa2_dense_2cta)
int process_diag_paged_fp8_multitok();  // FP8 paged decode tokens per warp iteration
                                        // (attention.paged_fp8_multitok)
void process_diag_set_paged_fp8_multitok(int v);
int process_diag_paged_nvfp4_multitok();  // NVFP4 paged decode tokens per warp iteration
                                          // (attention.paged_nvfp4_multitok)
void process_diag_set_paged_nvfp4_multitok(int v);
int process_diag_paged_f16_multitok();  // F16 paged decode tokens per warp iteration
                                        // (attention.paged_f16_multitok)
void process_diag_set_paged_f16_multitok(int v);
int process_diag_paged_f16_hpc();  // F16 multitok Q heads per CTA, 0 = auto (tests only)
void process_diag_set_paged_f16_hpc(int v);
int process_diag_paged_nvfp4_hpc();  // NVFP4 multitok Q heads per CTA, 0 = auto, 1 = per-head (tests only)
void process_diag_set_paged_nvfp4_hpc(int v);
void process_diag_set_fa2_dense_2cta(bool v);
bool process_diag_fa2_heavy_first();  // causal FA2 CTA order, heavy q-tiles first (attention.fa2_heavy_first)
void process_diag_set_fa2_heavy_first(bool v);
int process_diag_fa2_hd256_bkv();  // KV tile rows of the HD=256 instance (attention.fa2_hd256_bkv)
void process_diag_set_fa2_hd256_bkv(int v);
bool process_diag_fp8_qk_scaled();  // amax-scaled e4m3 fp8-QK (#680)
void process_diag_set_fp8_qk_scaled(bool v);
int process_diag_nvfp4_cutlass_streamk();  // gemm.nvfp4_cutlass_streamk: 0 off, 1 sub-2-wave grids, 2 forced
void process_diag_set_nvfp4_cutlass_streamk(int v);
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
void process_diag_set_mxfp4_ksmooth(bool v);
void process_diag_set_mxfp4_pv_fp4(bool v);
// ThriftAttention-style outlier promotion budget (0 = off, requires blockscale).
float process_diag_mxfp4_promote_budget();
void process_diag_set_mxfp4_promote_budget(float v);

// FFN
bool process_diag_ffn_sparsity_probe();

// MoE
bool process_diag_verify_row_parity();  // verify chunk reduces K like the decode GEMV
void process_diag_set_verify_row_parity(bool);
int process_diag_moe_mr_nr();  // rows-per-block for NVFP4 MoE decode (4/8/16/32)
// Read at model-load time by weight_upload (Pass 2 expert offload budget
// + force-host-experts). No per-Engine context at that point.
int process_diag_moe_expert_overhead_pct();
int process_diag_moe_force_host_experts();
// Copy host-resident NVFP4 experts into pinned host memory at load (a trade:
// +14.7 % prefill for 4.6x model-load time — see dispatch_policy.h).
bool process_diag_moe_pin_host_experts();

// GDN: layout override read at model-load time by hf_config_loader (no
// per-Engine context at that point in the loader pipeline).
const std::string& process_diag_gdn_layout_override();

}  // namespace imp
