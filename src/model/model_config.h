#pragma once

#include "core/tensor.h"
#include "model/model_arch.h"
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

// FFN activation function type
// GPT_OSS_GLU (gpt-oss): gate/up clamped to ±limit (gate only from above),
// glu = gate·sigmoid(1.702·gate), out = (up + 1)·glu — plus per-expert biases.
enum class FFNActivation { SWIGLU, GEGLU, RELU_SQR, GPT_OSS_GLU };

// Norm placement relative to residual connection
enum class NormPlacement { PRE_NORM, POST_NORM };

struct ModelConfig {
    ModelArch arch = ModelArch::GENERIC;
    int n_layers = 0, n_heads = 0, n_kv_heads = 0;
    int d_model = 0, d_ff = 0, vocab_size = 0, max_seq_len = 0;
    int head_dim = 0;  // 0 = infer as d_model / n_heads
    float rope_theta = 10000.0f, rms_norm_eps = 1e-5f, rope_freq_scale = 1.0f;
    float embed_scale = 0.0f;         // >0 = multiply embeddings by this (e.g. sqrt(d_model) for Gemma)
    float norm_weight_offset = 0.0f;  // Gemma: 1.0 (norms use w+1 instead of w)
    int n_experts = 0, n_experts_active = 0, expert_d_ff = 0;

    // Per-layer config (empty for standard transformers)
    std::vector<int> n_kv_heads_per_layer;  // 0 = no attention this layer
    std::vector<int> d_ff_per_layer;        // 0 = no dense FFN (SSM or attention-only)
    std::vector<int> head_dim_per_layer;    // Gemma 4: different head_dim per layer type
    std::vector<int> n_heads_per_layer;     // Gemma 4: per-layer Q head count
    std::vector<uint8_t> swa_layers;        // Gemma 4: 1 = SWA layer, 0 = full attention
    float rope_theta_swa = 0.0f;            // Gemma 4: RoPE theta for SWA layers (default: rope_local_theta)

    // Mamba2 SSM config
    int ssm_conv_kernel = 0;  // 4
    int ssm_state_size = 0;   // 128
    int ssm_group_count = 0;  // 8
    int ssm_inner_size = 0;   // 4096
    int ssm_dt_rank = 0;      // 64
    // Conv1d channel count (xBC width: x plus the B/C group states). Derived,
    // not stored — the single source for what used to be hand-derived at
    // every SSM sizing/upload/assign site. 0 on non-SSM models.
    int ssm_conv_channels() const { return ssm_inner_size + 2 * ssm_group_count * ssm_state_size; }
    // Asymmetric-head GDN (n_v_heads > n_k_heads) head storage layout.
    //   false (default): tiled order, h % n_groups gives group_id (GGUF Qwen3.5/3.6).
    //   true: grouped order, h / n_v_per_k gives group_id (HF SafeTensors Qwen3.5/3.6).
    bool gdn_grouped_head_layout = false;
    int rope_dim = 0;       // 0 = full head_dim, 84 = partial
    bool rope_neox = true;  // true = NeoX/split (i, i+d/2), false = interleaved (2i, 2i+1)

    // M-RoPE (Qwen-VL): rotary pairs split across 3 position axes (text/time, image height,
    // width). Half-counts sum to rope_dim/2 (Qwen3-VL: [24,20,20] for rope_dim 128). All zero
    // = no M-RoPE, every pair follows the single text position (also what text-only reduces to).
    int mrope_section[3] = {0, 0, 0};
    // Qwen3-VL interleaves the 3 axes across the frequency spectrum (T,H,W,T,H,W,... then T
    // tail) instead of 3 contiguous blocks. The two layouts rotate different dims by different
    // angles; not cosmetic.
    bool mrope_interleaved = false;
    bool has_mrope() const { return mrope_section[1] > 0 || mrope_section[2] > 0; }

    // MLA (DeepSeek-V2/V3). kv_lora_rank > 0 selects the MLA path.
    int kv_lora_rank = 0;      // 512 (V2-Lite) / 1024 (V3)
    int q_lora_rank = 0;       // 0 = full Q projection (V2-Lite); >0 = Q down/up LoRA (V3)
    int qk_rope_head_dim = 0;  // 64  — decoupled RoPE key dims
    int qk_nope_head_dim = 0;  // 128 — non-RoPE key dims
    int v_head_dim = 0;        // 128 — value head dim
    // YaRN mscale used for the softmax attention-scale adjustment.
    // Stores rope_scaling.mscale_all_dim if present, else rope_scaling.mscale.
    // For DeepSeek-V2-Lite both are 0.707; the distinction matters for V3.
    float mla_mscale = 1.0f;
    // Raw rope_scaling.mscale, the numerator of the RoPE cos/sin scale. HF DeepseekV2 scales
    // cos/sin by the RATIO yarn_get_mscale(factor,mscale)/yarn_get_mscale(factor,mscale_all_dim),
    // which is 1.0 for V2-Lite (both 0.707). Kept separate from mla_mscale (mscale_all_dim) so
    // this ratio isn't double-applied.
    float mla_mscale_num = 1.0f;
    bool is_mla() const { return kv_lora_rank > 0; }
    int first_k_dense_replace = 0;  // layers [0, k) use dense FFN even in a MoE model
    // NoPE attention (Nemotron-H family): attention layers use NO rotary
    // embedding — position is carried by the recurrent (Mamba) layers.
    bool rope_attn_disabled = false;

    // YaRN / Dynamic NTK RoPE scaling
    float yarn_ext_factor = 0.0f;   // 0 = linear/none, 1.0 = YaRN blending
    float yarn_attn_factor = 1.0f;  // mscale: attention factor (pre-compensated)
    float yarn_beta_fast = 32.0f;   // wavelength threshold for fast-rotating dims
    float yarn_beta_slow = 1.0f;    // wavelength threshold for slow-rotating dims
    int rope_n_ctx_orig = 0;        // original training context length (0 = use max_seq_len)

    // LongRoPE per-dimension frequency scaling (Phi-4)
    std::vector<float> rope_short_factor;  // [rope_pairs] short-context factors
    std::vector<float> rope_long_factor;   // [rope_pairs] long-context factors
    int rope_scaling_orig_max_pos = 0;     // threshold: short if seqlen <= this, else long
    int sliding_window = 0;                // 0 = disabled, >0 = window size (Qwen3, Mistral)
    int sliding_window_pattern = 0;        // Gemma-3: 6 = every 6th layer is global (no window)
    float rope_local_theta = 0.0f;         // Gemma-3: RoPE theta for local/sliding layers (10000)
    FFNActivation ffn_activation = FFNActivation::SWIGLU;
    NormPlacement norm_placement = NormPlacement::PRE_NORM;

    // Extended MoE config
    int n_experts_shared = 0;           // 1
    int expert_shared_d_ff = 0;         // 3712
    float expert_weights_scale = 1.0f;  // 2.5
    bool expert_weights_norm = false;
    bool moe_sigmoid_gating = false;   // Nemotron-H uses sigmoid instead of softmax
    float attn_logit_softcap = 0.0f;   // Gemma-2/3: tanh(score/cap)*cap before softmax (0=disabled)
    float final_logit_softcap = 0.0f;  // Gemma-2/3: tanh(logit/cap)*cap on output logits (0=disabled)

    // MXFP4 Hadamard rotation (set by converter via GGUF metadata)
    int mxfp4_hadamard_attn = 0;  // block size for attention weights (0=disabled)
    int mxfp4_hadamard_ffn = 0;   // block size for FFN weights (0=disabled)

    // Checkpoint is a multimodal wrapper: config nests text hyperparameters under text_config,
    // tensors live under model.language_model.* plus a separate vision tower. Flag rather than
    // an arch list, since the TEXT model of such a checkpoint is an ordinary one (Qwen3-VL's
    // is plain Qwen3).
    bool multimodal_wrapper = false;

    // Checkpoint declares an audio encoder; imp has no audio path at all (no encoder, input
    // type, or tokenizer route). Read from audio_config being a JSON OBJECT, not key presence:
    // Gemma-4-26B writes audio_config:null with no audio tensor, Gemma-4-12B writes the object.
    bool has_audio_config = false;

    // Checkpoint declared a rope_scaling.type this loader doesn't implement: no scaling
    // applied, model rotates unscaled while reporting its declared context. Not set for
    // "default"/"none", which mean exactly that.
    bool rope_scaling_unhandled = false;

    bool is_nvfp4_prequant = false;
    int nvfp4_group_size = 16;
    // llm-compressor NVFP4 format: weight_global_scale is a divisor, not a multiplier.
    // Reconstruction: val = fp4 * weight_scale_fp8 / weight_global_scale
    // (Modelopt: val = fp4 * weight_scale_fp8 * weight_scale_2)
    bool is_llm_compressor_nvfp4 = false;
    // quantization_config.ignore: modules the author kept at source precision. With
    // targets:["Linear"] this is a COMPLETE partition of the checkpoint's Linears, letting the
    // loader tell "author kept this in BF16" from "imp lost this module's weight_scale" (#1960).
    std::vector<std::string> nvfp4_exclude_modules;

    // MXFP4 pre-quantized model (e.g. GPT-OSS). Only metadata is recognised on the SafeTensors
    // path; the decode path is GGUF-only (QType::MXFP4). Surfaces the format so it warns
    // instead of silently falling through to FP16.
    bool is_mxfp4_prequant = false;
    int mxfp4_block_size = 32;  // E8M0 scale per 32 elements is standard

    // AWQ pre-quantized model. Detection-only today — imp does not yet
    // have an AWQ dequant kernel; weights load with their wire dtype but
    // inference will likely produce wrong results.
    bool is_awq_prequant = false;
    int awq_group_size = 128;

    // Author-declared KV-cache quant hint from Modelopt's hf_quant_config.json
    // kv_cache_quant_algo. Informational only: the engine does not auto-flip KV dtype from it,
    // since FP8 KV breaks some model families even with author opt-in.
    std::string kv_cache_quant_hint;

    // tri-state config flags (-1 = unset, 0 = false, 1 = true).
    // Cross-checked against actual tensor presence after load.
    int tie_word_embeddings = -1;
    int attention_bias = -1;
    int mlp_bias = -1;

    // True when load_config() couldn't identify the architecture and fell back to
    // ModelArch::GENERIC. Downstream loaders can gate tensor-name heuristics on it; higher
    // layers can warn the user explicitly.
    bool arch_inferred_fallback = false;

    // Model-specific runtime knobs formerly on the global RuntimeConfig singleton (Phase 5
    // Track A). Only writer is engine_init_resolver for GEMMA4; no config surface. Six
    // bring-up bisect flags with no writer since #319 were deleted with their branches
    // (AUDIT_arch_2026 G-7).
    struct Overrides {
        struct Gemma4 {
            bool force_mmvq = false;
        } gemma4;
    } overrides;
};

// YaRN mscale attention-scale multiplier for MLA (DeepSeek-V2/V3):
//   mscale_adj = 0.1*mscale_all_dim*ln(factor)+1.0  (factor = rope_freq_scale)
//   attention_scale = (1/sqrt(head_dim)) * mscale_adj^2
// Returns mscale_adj^2 when YaRN factor>1 on an MLA model, else 1.0.
inline float mla_attention_scale_multiplier(const ModelConfig& cfg) {
    if (!cfg.is_mla() || cfg.rope_freq_scale <= 1.0f)
        return 1.0f;
    const float mscale_adj = 0.1f * cfg.mla_mscale * std::log(cfg.rope_freq_scale) + 1.0f;
    return mscale_adj * mscale_adj;
}

// Forward declaration — full definition in quant/nvfp4_quant.h.
// Used by nvfp4_moe_*_ptr borrowed pointers below.
struct NvFP4MoEQuantResult;

// Pre-quantized NVFP4 weights (Model Optimizer / llm-compressor SafeTensors), live only on
// Model::nvfp4_scratch_ until Phase 0 promotes them onto the weight's .scales/.tensor_scale.
//   weight_scale   [N, K/group_size] FP8 E4M3 micro-scales
//   weight_scale_2 [1] FP32 tensor-scale (may be missing for some Modelopt variants)
//   input_scale    [1] FP32 activation scale (optional)
// valid() requires only weight_scale.
struct NvFP4PreQuantWeight {
    Tensor weight_scale;
    Tensor weight_scale_2;
    Tensor input_scale;
    bool valid() const { return weight_scale.data != nullptr; }
};

struct TransformerLayer {
    Tensor wq, wk, wv, wo, attn_norm;
    // MLA projections (DeepSeek-V2/V3): kv_a_proj_with_mqa packs latent(512)+rope(64)
    // down-proj; kv_a_layernorm is RMSNorm on the 512-dim latent (never quantized); kv_b_proj
    // up-projects to 16*(128+128)=4096.
    Tensor kv_a_proj, kv_a_layernorm, kv_b_proj;
    Tensor q_bias, k_bias, v_bias;  // Attention biases (Qwen2)
    Tensor o_bias;                  // Output-projection bias (gpt-oss)
    Tensor attn_sinks;              // Per-head sink logits [n_heads] (gpt-oss)
    Tensor router_bias;             // Router logits bias [n_experts] (gpt-oss)
    Tensor expert_gate_bias;        // Per-expert gate bias [ne, d_ff] (gpt-oss, de-interleaved)
    Tensor expert_up_bias;          // Per-expert up bias [ne, d_ff] (gpt-oss, de-interleaved)
    Tensor expert_down_bias;        // Per-expert down bias [ne, d_model] (gpt-oss)
    // gpt-oss raw checkpoint slots (HF MXFP4 layout, consumed at upload):
    Tensor expert_gate_up_packed_blocks;   // U8 [ne, 2*d_ff, K/32, 16] e2m1, rows interleaved g0,u0,g1,...
    Tensor expert_gate_up_packed_scales;   // U8 [ne, 2*d_ff, K/32] ue8m0
    Tensor expert_gate_up_bias_fused;      // BF16 [ne, 2*d_ff] interleaved
    Tensor expert_down_packed_blocks;      // U8 [ne, d_model, d_ff/32, 16]
    Tensor expert_down_packed_scales;      // U8 [ne, d_model, d_ff/32]
    Tensor attn_q_norm, attn_k_norm;       // QK-norm (Qwen3-style per-head RMSNorm)
    Tensor post_attn_norm, post_ffn_norm;  // Post-layer norms (Gemma-3)
    // Encoder post-LN biases (#836, nomic-bert): true LayerNorm with bias,
    // applied AFTER the residual add (weights live in post_attn/ffn_norm).
    Tensor post_attn_norm_bias, post_ffn_norm_bias;
    // Gemma 4 extras
    Tensor ffn_pre_norm_2;      // pre-norm for expert branch (operates on attn_out)
    Tensor ffn_post_norm_1;     // post-norm for shared MLP branch
    Tensor ffn_post_norm_2;     // post-norm for expert branch
    Tensor ffn_gate_inp_scale;  // router input scale [d_model]
    Tensor layer_out_scale;     // per-layer output scalar (optional)
    Tensor rope_freqs;          // per-layer RoPE frequency factors (full-attn layers only)
    bool kv_equals_k = false;   // Gemma 4: V=K (wv absent for this layer)
    // Provenance for the fused-projection scale split (src/exec/nvfp4_merged_scale_guard.h).
    // Set only by weight_map.cpp when splitting a fused qkv_proj/gate_up_proj tensor. Required
    // by the loader's scale fix-up: on a separate-tensor checkpoint this arm is
    // indistinguishable from a promotion failure, and firing then misroutes a sibling's scales.
    bool qkv_split_from_fused = false;
    bool gate_up_split_from_fused = false;
    Tensor w_gate, w_up, w_down, ffn_norm;
    Tensor moe_gate;
    std::vector<Tensor> expert_w_gate, expert_w_up, expert_w_down;

    // Packed expert tensors (3D: [n_experts, rows, cols]) loaded from GGUF *_exps
    // These are temporary: weight_upload slices them into the per-expert vectors above.
    Tensor expert_gate_packed, expert_up_packed, expert_down_packed;

    // Shared expert (always-active, e.g. Nemotron/DeepSeek)
    Tensor w_up_shared, w_down_shared, w_gate_shared;

    // Qwen3-Next/Qwen3.6 shared-expert input gate: [d_model] FP32 projection whose sigmoid
    // produces a per-token scalar gating the shared expert output before adding to the MoE
    // output. GGUF: blk.{i}.ffn_gate_inp_shexp.weight. Absent for Qwen2-MoE/Qwen3 MoE.
    Tensor shared_expert_gate_inp;

    // Per-group scales ride on each tensor's .scales field (core/tensor.h); legacy *_scales
    // mirror fields were removed in Stage F, qtype mirrors removed in Stage G.

    // WeightRegistry indices (populated by pre_dequant_weights, Phase 2+).
    // kInvalidTensorID means the corresponding Tensor is absent on this layer
    // (e.g. ssm_in_id is kInvalidTensorID on attention-only layers).
    TensorID wq_id = kInvalidTensorID;
    TensorID wk_id = kInvalidTensorID;
    TensorID wv_id = kInvalidTensorID;
    TensorID wo_id = kInvalidTensorID;
    // MLA WeightRegistry indices
    TensorID kv_a_proj_id = kInvalidTensorID;
    TensorID kv_a_norm_id = kInvalidTensorID;
    TensorID kv_b_proj_id = kInvalidTensorID;
    TensorID w_gate_id = kInvalidTensorID;
    TensorID w_up_id = kInvalidTensorID;
    TensorID w_down_id = kInvalidTensorID;
    // Shared-expert FFN (Nemotron / DeepSeek / Qwen3.5-MoE). kInvalidTensorID
    // when the layer has no shared expert branch.
    TensorID w_gate_shared_id = kInvalidTensorID;
    TensorID w_up_shared_id = kInvalidTensorID;
    TensorID w_down_shared_id = kInvalidTensorID;
    TensorID ssm_in_id = kInvalidTensorID;
    TensorID ssm_out_id = kInvalidTensorID;
    TensorID gdn_gate_id = kInvalidTensorID;
    TensorID gdn_alpha_id = kInvalidTensorID;
    TensorID gdn_beta_id = kInvalidTensorID;
    TensorID gdn_alpha_beta_packed_id = kInvalidTensorID;
    TensorID gdn_input_packed_id = kInvalidTensorID;
    // Fused KV / gate+up handles. Populated when the runtime decides to build
    // a fused weight pair for strided batched prefill GEMM. kInvalidTensorID
    // means no fused weight was built for this layer (per-layer dispatch path).
    TensorID fused_kv_id = kInvalidTensorID;
    TensorID fused_gate_up_id = kInvalidTensorID;

    // Per-expert WeightRegistry indices (populated by pre_dequant_weights, Task 3.4).
    // Parallel to expert_w_gate / expert_w_up / expert_w_down vectors.
    std::vector<TensorID> expert_gate_ids;
    std::vector<TensorID> expert_up_ids;
    std::vector<TensorID> expert_down_ids;
    // Router and shared-expert gate projections
    TensorID moe_gate_id = kInvalidTensorID;
    TensorID shared_expert_gate_id = kInvalidTensorID;

    // Borrowed pointers into wcache_.nvfp4_moe (packed 3D expert NVFP4 cache).
    // Set by pre_dequant_weights when the packed expert tensors are NVFP4-cached.
    // Null means the nvfp4_moe path is unavailable for this layer.
    const NvFP4MoEQuantResult* nvfp4_moe_gate_ptr = nullptr;
    const NvFP4MoEQuantResult* nvfp4_moe_up_ptr = nullptr;
    const NvFP4MoEQuantResult* nvfp4_moe_down_ptr = nullptr;

    // Borrowed pointers into wcache_.fp16 for packed expert tensors.
    // Set by pre_dequant_weights when the entire packed expert tensor has a
    // pre-dequantised FP16 entry (contiguous [n_experts*rows, cols] layout).
    const Tensor* fp16_packed_gate_cache = nullptr;
    const Tensor* fp16_packed_up_cache = nullptr;
    const Tensor* fp16_packed_down_cache = nullptr;

    // Mamba2 SSM weights
    Tensor ssm_in, ssm_out;             // Projections
    Tensor ssm_conv1d_w, ssm_conv1d_b;  // Conv1d weight + bias
    Tensor ssm_dt_b;                    // dt bias
    Tensor ssm_a, ssm_d;                // A (log) and D (skip connection)
    Tensor ssm_norm_w;                  // Group RMSNorm weight

    // Gated DeltaNet (GDN) weights (Qwen3.5 hybrid)
    Tensor gdn_gate;   // [d_model, inner_size] output gating projection
    Tensor gdn_alpha;  // [d_model, n_gdn_heads] delta rule decay
    Tensor gdn_beta;   // [d_model, n_gdn_heads] delta rule learning rate

    // Decode-only fused weight: gdn_alpha/gdn_beta interleaved along N as
    // [d_model, 2*n_gdn_heads], firing the M=1 path as one GEMV instead of two. Built at load
    // time when both are FP16/BF16 + same shape; null on GGUF/MXFP4 (falls back to two-call
    // path). Superseded by gdn_input_packed when the full 4-way fusion fires.
    Tensor gdn_alpha_beta_packed;

    // Full GDN input fusion: stacks ssm_in+gdn_gate+gdn_alpha+gdn_beta along N into one
    // [total_out, d_model] weight (total_out = conv_channels+inner+2*n_heads). One GEMV
    // produces all four outputs; run_gdn slices:
    //   [0, conv_channels)                         -> xBC (conv1d input)
    //   [conv_channels, conv_channels+inner)       -> z / gate_out
    //   [conv_channels+inner, ...+n_heads)         -> alpha
    //   [conv_channels+inner+n_heads, total_out)   -> beta
    // When set, ssm_in/gdn_gate/gdn_alpha/gdn_beta device memory is released (.data=nullptr).
    Tensor gdn_input_packed;
    int gdn_packed_conv_channels = 0;
    int gdn_packed_inner = 0;
    int gdn_packed_n_heads = 0;

    // Router bias (Nemotron MoE)
    Tensor moe_router_bias;

    // Per-expert output scale (Gemma 4): one scalar per expert, applied to each
    // expert's down-projection output BEFORE the routing weighted sum. Loaded
    // from `blk.X.ffn_down_exps.scale` (shape [n_expert]).
    Tensor expert_down_scale;

    // GPTQ quantized weights (temporary — dequantized to FP16 during upload)
    struct GPTQWeight {
        Tensor qweight;  // packed INT32
        Tensor qzeros;   // zero points
        Tensor scales;   // per-group scales (FP16)
        Tensor g_idx;    // group index (optional, for desc_act)
        int bits = 0;
        int group_size = 128;
        bool desc_act = false;  // config-declared activation reordering
    };
    GPTQWeight gptq_q, gptq_k, gptq_v, gptq_o;
    GPTQWeight gptq_gate, gptq_up, gptq_down;
};

}  // namespace imp
