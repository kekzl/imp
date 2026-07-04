#pragma once

#include "core/tensor.h"
#include "model/model_arch.h"
#include <cmath>
#include <cstdint>
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
    // Asymmetric-head GDN (n_v_heads > n_k_heads) head storage layout.
    // false (default): heads in tiled order (h % n_groups gives group_id).
    //                  GGUF Qwen3.5/3.6 converters use this layout.
    // true:            heads in grouped order (h / n_v_per_k gives group_id).
    //                  HF SafeTensors Qwen3.5/3.6 use this. Set by the
    //                  SafeTensors loader; GGUF loader leaves it false.
    bool gdn_grouped_head_layout = false;
    int rope_dim = 0;       // 0 = full head_dim, 84 = partial
    bool rope_neox = true;  // true = NeoX/split (i, i+d/2), false = interleaved (2i, 2i+1)

    // MLA (DeepSeek-V2/V3). kv_lora_rank > 0 selects the MLA path.
    int   kv_lora_rank      = 0;     // 512 (V2-Lite) / 1024 (V3)
    int   q_lora_rank       = 0;     // 0 = full Q projection (V2-Lite); >0 = Q down/up LoRA (V3)
    int   qk_rope_head_dim  = 0;     // 64  — decoupled RoPE key dims
    int   qk_nope_head_dim  = 0;     // 128 — non-RoPE key dims
    int   v_head_dim        = 0;     // 128 — value head dim
    // YaRN mscale used for the softmax attention-scale adjustment.
    // Stores rope_scaling.mscale_all_dim if present, else rope_scaling.mscale.
    // For DeepSeek-V2-Lite both are 0.707; the distinction matters for V3.
    float mla_mscale        = 1.0f;
    bool  is_mla() const { return kv_lora_rank > 0; }
    int   first_k_dense_replace = 0; // layers [0, k) use dense FFN even in a MoE model
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

    // NVFP4 pre-quantized model (from Model Optimizer SafeTensors)
    bool is_nvfp4_prequant = false;
    int nvfp4_group_size = 16;
    // llm-compressor NVFP4 format: weight_global_scale is a divisor, not a multiplier.
    // Reconstruction: val = fp4 * weight_scale_fp8 / weight_global_scale
    // (Modelopt: val = fp4 * weight_scale_fp8 * weight_scale_2)
    bool is_llm_compressor_nvfp4 = false;

    // MXFP4 pre-quantized model (e.g. GPT-OSS exports). Only the metadata
    // is recognised on the SafeTensors path today — the actual decode path
    // is GGUF-only (QType::MXFP4 wire format). Use these flags to surface
    // the format and warn the user instead of falling through to FP16 silently.
    bool is_mxfp4_prequant = false;
    int mxfp4_block_size = 32;  // E8M0 scale per 32 elements is standard

    // AWQ pre-quantized model. Detection-only today — imp does not yet
    // have an AWQ dequant kernel; weights load with their wire dtype but
    // inference will likely produce wrong results.
    bool is_awq_prequant = false;
    int awq_group_size = 128;

    // Author-declared KV-cache quantization hint, sourced from Modelopt's
    // hf_quant_config.json `kv_cache_quant_algo` field (e.g. "FP8"). Empty
    // means the model author did not declare one. Surfaced to the engine as
    // an informational signal — engine does NOT auto-flip the KV cache dtype
    // because FP8 KV is known to break several model families even with
    // model-author opt-in (see kv_dtype_tradeoffs memory).
    std::string kv_cache_quant_hint;

    // tri-state config flags (-1 = unset, 0 = false, 1 = true).
    // Cross-checked against actual tensor presence after load.
    int tie_word_embeddings = -1;
    int attention_bias = -1;
    int mlp_bias = -1;

    // True when load_config() couldn't identify the architecture from
    // config.json and fell back to ModelArch::GENERIC. Downstream loaders
    // can use this to decide whether to invoke tensor-name heuristics, and
    // higher layers can surface an explicit warning to the user.
    bool arch_inferred_fallback = false;

    // Model-specific runtime knobs that used to live on the global
    // RuntimeConfig singleton. Phase 5 Track A of the architecture refactor
    // moved them onto ModelConfig so the "model-name in a runtime
    // singleton" smell goes away. Populated from imp.conf (legacy
    // [gemma4] section seeds compat) or set by engine_init_resolver when
    // the arch is GEMMA4; default-constructed otherwise.
    struct Overrides {
        struct Gemma4 {
            bool fp32_gemm_out = false;
            bool no_graphs = false;
            bool force_mmvq = false;
            bool fp32_expert_down = false;
            bool no_decode_fast = false;
            bool no_post_ffw_1 = false;
            bool ggml_prefill = false;
        } gemma4;
    } overrides;
};

// YaRN mscale attention-scale multiplier for MLA models (Task 2.5).
//
// DeepSeek-V2/V3 uses a YaRN-adjusted softmax scale:
//   mscale_adj = 0.1 * mscale_all_dim * ln(factor) + 1.0   (factor = rope_freq_scale)
//   attention_scale = (1 / sqrt(head_dim)) * mscale_adj^2
//
// This helper returns mscale_adj^2 when the config is an MLA model with
// YaRN factor > 1, and 1.0f otherwise (leaving non-MLA models unaffected).
inline float mla_attention_scale_multiplier(const ModelConfig& cfg) {
    if (!cfg.is_mla() || cfg.rope_freq_scale <= 1.0f) return 1.0f;
    const float mscale_adj =
        0.1f * cfg.mla_mscale * std::log(cfg.rope_freq_scale) + 1.0f;
    return mscale_adj * mscale_adj;
}

// Forward declaration — full definition in quant/nvfp4_quant.h.
// Used by nvfp4_moe_*_ptr borrowed pointers below.
struct NvFP4MoEQuantResult;

// Pre-quantized NVFP4 weights (from Model Optimizer / llm-compressor via
// SafeTensors). Lives only on the Model::nvfp4_scratch_ load-time map —
// once executor_pre_dequant.cu Phase 0 promotes the scale pointers onto
// the corresponding main weight tensor's .scales / .tensor_scale fields,
// the entry is dropped.
//
//   weight_scale     [N, K/group_size]  FP8 E4M3 micro-scales
//   weight_scale_2   [1] or scalar      FP32 tensor-scale
//   input_scale      [1]                FP32 activation scale (optional)
//
// `valid()` only requires weight_scale; weight_scale_2 is permitted to be
// missing for some Modelopt variants.
struct NvFP4PreQuantWeight {
    Tensor weight_scale;
    Tensor weight_scale_2;
    Tensor input_scale;
    bool valid() const { return weight_scale.data != nullptr; }
};

struct TransformerLayer {
    Tensor wq, wk, wv, wo, attn_norm;
    // MLA (Multi-head Latent Attention) projections (DeepSeek-V2/V3).
    // kv_a_proj: kv_a_proj_with_mqa — packed latent(512)+rope(64) down-projection.
    // kv_a_layernorm: RMSNorm weight on the 512-dim latent (never quantized).
    // kv_b_proj: up-projection, output 16*(128+128)=4096.
    Tensor kv_a_proj, kv_a_layernorm, kv_b_proj;
    Tensor q_bias, k_bias, v_bias;         // Attention biases (Qwen2)
    Tensor o_bias;                         // Output-projection bias (gpt-oss)
    Tensor attn_sinks;                     // Per-head sink logits [n_heads] (gpt-oss)
    Tensor router_bias;                    // Router logits bias [n_experts] (gpt-oss)
    Tensor expert_gate_bias;               // Per-expert gate bias [ne, d_ff] (gpt-oss, de-interleaved)
    Tensor expert_up_bias;                 // Per-expert up bias [ne, d_ff] (gpt-oss, de-interleaved)
    Tensor expert_down_bias;               // Per-expert down bias [ne, d_model] (gpt-oss)
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
    Tensor w_gate, w_up, w_down, ffn_norm;
    Tensor moe_gate;
    std::vector<Tensor> expert_w_gate, expert_w_up, expert_w_down;

    // Packed expert tensors (3D: [n_experts, rows, cols]) loaded from GGUF *_exps
    // These are temporary: weight_upload slices them into the per-expert vectors above.
    Tensor expert_gate_packed, expert_up_packed, expert_down_packed;

    // Shared expert (always-active, e.g. Nemotron/DeepSeek)
    Tensor w_up_shared, w_down_shared, w_gate_shared;

    // Qwen3-Next / Qwen3.6 shared-expert input gate: a [d_model] FP32 projection
    // that, after sigmoid, produces a per-token scalar used to gate the shared
    // expert output before it is added to the MoE output. Stored in the GGUF as
    // `blk.{i}.ffn_gate_inp_shexp.weight`. Absent for Qwen2-MoE / Qwen3 MoE.
    Tensor shared_expert_gate_inp;

    // (Per-group scales now ride along on each tensor's .scales field —
    //  see core/tensor.h. The legacy *_scales mirror fields were removed
    //  in Stage F.)
    //
    // (qtype mirrors removed in Stage G — read tensor.qtype directly.)

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

    // Decode-only fused weight: gdn_alpha and gdn_beta interleaved along N
    // as [d_model, 2*n_gdn_heads]. Fires the M=1 path with a single GEMV
    // launch instead of two. Built at load-time when both alpha and beta are
    // FP16/BF16 + same shape; left null on GGUF/MXFP4 paths so the dispatcher
    // falls back to the two-call path. Superseded by `gdn_input_packed` below
    // when the full 4-way fusion fires.
    Tensor gdn_alpha_beta_packed;

    // Full GDN input projection fusion: stacks ssm_in + gdn_gate + gdn_alpha +
    // gdn_beta along the N (output) dim into one [total_out, d_model] weight,
    // where total_out = conv_channels + inner + 2*n_heads. A single GEMV
    // produces all four outputs contiguously, then run_gdn slices at:
    //   [0,                          conv_channels)        → xBC (conv1d input)
    //   [conv_channels,              conv_channels+inner)  → z / gate_out
    //   [conv_channels+inner,        ...+n_heads)          → alpha
    //   [conv_channels+inner+n_heads,total_out)            → beta
    //
    // When set, ssm_in / gdn_gate / gdn_alpha / gdn_beta have their device
    // memory released (their .data goes nullptr, .on_device=false). Null =
    // fall back to the four separate gemm_dispatch calls.
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
