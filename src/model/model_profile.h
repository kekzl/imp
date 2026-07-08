#pragma once

#include "model/model_config.h"  // ModelArch

namespace imp {

class Model;

// Architecture-derived facts, decided ONCE at init and read everywhere — the
// single source of truth that replaces the scattered inline `if (arch==…)`
// branches and the repeated "loop the layers, check gdn_gate/ssm_in" detection
// (D1 in the structural-debt audit, docs/archive/README.md; the #514/#516 bug class).
//
// ModelConfig holds STATIC metadata (loaded from the file). ModelProfile holds
// the DERIVED classification/dispatch decisions computed from that metadata +
// the loaded layers. Filled by derive_model_profile() once, before any forward
// pass or the engine-init resolvers that currently re-derive it inline.
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

    // --- architecture identity (mirrors ModelConfig::arch) ---
    // The hot path (executor_attention / executor_forward_moe / …) keys many
    // kernel- and norm-selection branches off the architecture. These booleans
    // are the ONE place that maps the arch enum to those branches: every
    // `cfg.arch == ModelArch::X` in the executors reads the matching flag here
    // instead of re-comparing the enum inline.
    bool is_gemma3 = false;
    bool is_gemma4 = false;
    bool is_gpt_oss = false;
    bool is_llama4 = false;
    // Encoder-only embedder (#836, nomic-bert): bidirectional attention,
    // post-LN with bias, no KV cache / LM head / sampling. Served by the
    // dedicated encoder forward, never by the decoder loop.
    bool is_encoder = false;

    // --- attention variant (drives the SWA / NoPE dispatch in run_attention) ---
    // The attention path forks on how positions + sliding windows work. Encoding
    // it as one variant (decided here from arch + swa_layers + rope_attn_disabled)
    // replaces the per-call `is_gemma4 && !swa_layers.empty()` /
    // `is_gpt_oss && !swa_layers.empty()` / `!rope_attn_disabled` re-derivation.
    //   STANDARD    — RoPE, no per-arch SWA pattern (Gemma-3's
    //                 sliding_window_pattern path is handled separately and is
    //                 still STANDARD here).
    //   GEMMA4_SWA  — Gemma-4 per-layer SWA mask in swa_layers (local rope_theta).
    //   GPTOSS_SWA  — gpt-oss even=sliding/odd=full, same YaRN RoPE on both.
    //   NOPE        — no positional encoding at all (Nemotron-H attention).
    //   MLA         — Multi-head Latent Attention (DeepSeek-V2/V3): compressed
    //                 KV projection; no standard RoPE on the latent path.
    enum class AttnVariant { STANDARD, GEMMA4_SWA, GPTOSS_SWA, NOPE, MLA };
    AttnVariant attn_variant = AttnVariant::STANDARD;
};

// Pure: no side effects, no allocation. Reads the model's layers + config once.
ModelProfile derive_model_profile(const Model& model, const ModelConfig& cfg);

// Per-layer sliding-window size: 0 for global-attention layers, the window in
// tokens for SWA layers. Single source of truth for the four SWA variants
// (Gemma-4 / gpt-oss swa_layers[], Gemma-3 sliding_window_pattern, plain
// Mistral-style sliding_window) — consumed by the attention dispatch mask AND
// the SWA-aware KV sizing, so the two can't drift. `layer` is the global layer
// index (not the compacted KV-layer index).
int layer_swa_window(const ModelConfig& cfg, const ModelProfile& prof, int layer);

}  // namespace imp
