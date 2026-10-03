#pragma once

#include "model/model_config.h"  // ModelArch

namespace imp {

class Model;

// Architecture-derived facts decided once at init, replacing scattered inline
// if(arch==...) branches and repeated GDN/SSM detection loops (D1 structural-debt audit,
// #514/#516 bug class). ModelConfig holds static file metadata; ModelProfile holds derived
// classification/dispatch decisions, filled once by derive_model_profile().
struct ModelProfile {
    // --- classification (scanned from the layers) ---
    bool is_moe = false;       // n_experts > 0
    bool is_gdn = false;       // any layer carries a gdn_gate (Gated DeltaNet)
    bool is_ssm = false;       // any layer carries an ssm_in (Mamba2 / GDN)
    bool has_pure_ssm = false; // any layer is SSM WITHOUT a gdn_gate (Mamba2,
                               // e.g. Nemotron-H) — these disable CUDA graphs
    bool is_hybrid = false;    // recurrent (gdn/ssm) AND attention layers coexist
    bool is_dense = true;      // !is_moe
    bool moe_experts_nvfp4 = false;  // MoE + NVFP4-prequant checkpoint: experts
                                     // get the contiguous native NVFP4 cache, so
                                     // batched verify/prefill reads quantized
                                     // weights directly (no per-chunk dequant)

    // Capability traits: executors branch on these, never on the arch enum. A new arch sets
    // the ones it needs in derive_model_profile(); the arch -> trait map lives only there.
    // Gemma-4 today:
    bool sandwich_norms = false;               // post-attn and pre/post-FFN norms per branch
    bool fp32_residual_norms = false;          // norms read the FP32 residual (fp32_accum_buf_)
    bool sanitize_ffn_fp16 = false;            // clear inf/NaN in FFN output before post-norm
    bool scaled_router_norm = false;           // router_in = rmsnorm(h) / sqrt(d) * gate_inp_scale
    bool router_bias_is_expert_scale = false;  // moe_router_bias holds ffn_down_exps.scale: ignore
    bool expert_out_scale = false;             // per-expert down scale folded into routing weights
    bool per_layer_head_shapes = false;        // per-layer n_heads from wq rows, not config
    bool k_as_v_without_wv = false;            // layers without wv alias V from K
    bool v_rmsnorm = false;                    // unweighted per-head RMSNorm on V
    bool rope_full_head_dim = false;           // rotate full hd; freq factors encode partial rotary
    bool unit_softmax_scale = false;           // attention softmax scale 1.0
    bool outlier_sensitive_logits = false;     // deterministic GEMM, MMVQ, no BOS warmup
    // gpt-oss today:
    bool learned_attn_sinks = false;             // sink logits: cuBLAS prefill, no FA2 / chunk capture
    bool moe_expert_bias_glu = false;            // per-expert gate/up/down bias + clamped GLU
    bool moe_router_logit_bias = false;          // linear router bias before softmax/top-k
    bool experts_convert_at_predequant = false;  // MXFP4 experts stay on host until NVFP4 convert
    bool fp8_attn_proj_full = false;             // gemm.fp8_attn_proj=auto caches q/k/v/o
    bool residual_rescale_in_scales = false;     // embed_scale folded into NVFP4 Wo/down scales
    // Gemma-3, Gemma-4, gpt-oss today:
    bool deny_cublas_fp16_acc = false;  // FP16 residual overflow: cublas_fp16_acc=auto -> off

    bool is_gemma3 = false;
    bool is_llama4 = false;
    // Encoder-only embedder (#836, nomic-bert): bidirectional attention,
    // post-LN with bias, no KV cache / LM head / sampling. Served by the
    // dedicated encoder forward, never by the decoder loop.
    bool is_encoder = false;
    bool gated_residual = false;  // Qwen4Exp: hc_attn_norm present, residual is hc_count x d_model wide

    // Attention variant, decided from arch+swa_layers+rope_attn_disabled:
    //   STANDARD   - RoPE, no per-arch SWA (Gemma-3's sliding_window_pattern is separate,
    //                still STANDARD here)
    //   GEMMA4_SWA - Gemma-4 per-layer SWA mask (local rope_theta)
    //   GPTOSS_SWA - gpt-oss even=sliding/odd=full, same YaRN RoPE on both
    //   NOPE       - no positional encoding (Nemotron-H attention)
    //   MLA        - DeepSeek-V2/V3, compressed KV projection, no standard RoPE on the latent path
    enum class AttnVariant { STANDARD, GEMMA4_SWA, GPTOSS_SWA, NOPE, MLA };
    AttnVariant attn_variant = AttnVariant::STANDARD;
};

// Pure: no side effects, no allocation. Reads the model's layers + config once.
ModelProfile derive_model_profile(const Model& model, const ModelConfig& cfg);

// Per-layer SWA size: 0 for global layers, window in tokens for SWA layers. Single source
// of truth for all four SWA variants (Gemma-4/gpt-oss swa_layers[], Gemma-3
// sliding_window_pattern, plain Mistral sliding_window); consumed by both the attention
// mask and KV sizing so they can't drift. `layer` is the global index, not the KV index.
int layer_swa_window(const ModelConfig& cfg, const ModelProfile& prof, int layer);

}  // namespace imp
