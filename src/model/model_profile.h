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

    // Hot path (executor_attention/executor_forward_moe/...) keys many kernel/norm-selection
    // branches off architecture. These booleans are the ONE place mapping the arch enum to
    // those branches; every cfg.arch==ModelArch::X in the executors should read the flag here.
    bool is_gemma3 = false;
    bool is_gemma4 = false;
    bool is_gpt_oss = false;
    bool is_llama4 = false;
    // Encoder-only embedder (#836, nomic-bert): bidirectional attention,
    // post-LN with bias, no KV cache / LM head / sampling. Served by the
    // dedicated encoder forward, never by the decoder loop.
    bool is_encoder = false;

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
