#include "model/hf_config_loader.h"
#include "model/json_util.h"
#include "model/llm_compressor_loader.h"
#include "core/logging.h"
#include "runtime/process_diag.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {

// ---- Architecture mapping ----

ModelArch HFConfigLoader::map_architecture(const std::string& hf_arch) {
    static const std::unordered_map<std::string, ModelArch> arch_map = {
        {"LlamaForCausalLM", ModelArch::LLAMA},
        {"MistralForCausalLM", ModelArch::MISTRAL},
        {"Mistral3ForConditionalGeneration", ModelArch::MISTRAL},
        {"MixtralForCausalLM", ModelArch::MIXTRAL},
        {"Qwen2ForCausalLM", ModelArch::QWEN3},
        {"Qwen2MoeForCausalLM", ModelArch::QWEN3_MOE},
        {"Qwen3ForCausalLM", ModelArch::QWEN3},
        {"Qwen3MoeForCausalLM", ModelArch::QWEN3_MOE},
        {"Qwen3_5ForCausalLM", ModelArch::QWEN35},
        {"Qwen3_5ForConditionalGeneration", ModelArch::QWEN35},
        {"Qwen3_5MoeForCausalLM", ModelArch::QWEN36_MOE},
        {"Qwen3_5MoeForConditionalGeneration", ModelArch::QWEN36_MOE},
        {"NemotronHForCausalLM", ModelArch::NEMOTRON_H_MOE},
        {"Gemma2ForCausalLM", ModelArch::GEMMA3},
        {"GemmaForCausalLM", ModelArch::GEMMA3},
        {"Gemma3ForCausalLM", ModelArch::GEMMA3},
        {"Gemma3ForConditionalGeneration", ModelArch::GEMMA3},
        {"Gemma4ForCausalLM", ModelArch::GEMMA4},
        {"Gemma4ForConditionalGeneration", ModelArch::GEMMA4},
        {"Gemma4UnifiedForConditionalGeneration", ModelArch::GEMMA4},  // multimodal unified (text_config nested)
        {"GptOssForCausalLM", ModelArch::GPT_OSS},
        {"DeepseekV2ForCausalLM", ModelArch::DEEPSEEK},
        {"DeepseekV3ForCausalLM", ModelArch::DEEPSEEK},
        {"Llama4ForCausalLM", ModelArch::LLAMA4},
        {"Llama4ForConditionalGeneration", ModelArch::LLAMA4},
        {"PhiForCausalLM", ModelArch::LLAMA},
        {"Phi3ForCausalLM", ModelArch::LLAMA},
        {"Phi3SmallForCausalLM", ModelArch::LLAMA},
        {"InternLM2ForCausalLM", ModelArch::LLAMA},
        {"Starcoder2ForCausalLM", ModelArch::LLAMA},
        {"CohereForCausalLM", ModelArch::LLAMA},
    };

    auto it = arch_map.find(hf_arch);
    if (it != arch_map.end())
        return it->second;

    IMP_LOG_WARN(
        "unknown HF architecture '%s' — falling back to GENERIC + tensor-name "
        "heuristics. If inference is incoherent, the architecture is likely "
        "unsupported. Add a class mapping in hf_config_loader.cpp.",
        hf_arch.c_str());
    return ModelArch::GENERIC;
}

// ---- load_config ----

bool HFConfigLoader::load_config(const std::string& model_dir, ModelConfig& cfg) {
    std::string path = model_dir + "/config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    IMP_LOG_INFO("loading HF config from %s", path.c_str());

    // Architecture detection: prefer "architectures" array, fall back to "model_type"
    const JValue* archs = jobj_find(root, "architectures");
    if (archs && archs->type == JType::ARRAY && !archs->arr.empty()) {
        // Encoder-only (BERT/RoBERTa/embedding) models have no causal LM head and
        // need pooling — the GGUF loader already rejects them at load; the
        // SafeTensors/HF path must too, or a `NomicBertModel` falls through to
        // the generic decoder and hits a CUDA IMA on the first request (#818).
        if (is_encoder_only_arch(archs->arr[0].str_val)) {
            throw std::runtime_error("encoder-only architecture '" + archs->arr[0].str_val +
                                     "' is not supported (imp runs causal decoder LMs; "
                                     "embedding encoders need pooling support)");
        }
        cfg.arch = map_architecture(archs->arr[0].str_val);
        if (archs->arr.size() > 1) {
            std::string dropped;
            for (size_t i = 1; i < archs->arr.size(); i++) {
                if (i > 1)
                    dropped += ", ";
                dropped += archs->arr[i].str_val;
            }
            IMP_LOG_WARN(
                "config.json `architectures` has %zu entries; using '%s' and ignoring [%s]",
                archs->arr.size(), archs->arr[0].str_val.c_str(), dropped.c_str());
        }
    } else {
        // Fallback: map model_type string to arch
        std::string model_type;
        if (jobj_get_string(root, "model_type", model_type)) {
            // Same encoder-only reject on the model_type path (e.g. a config with
            // only `"model_type": "nomic_bert"` and no `architectures` array).
            if (is_encoder_only_arch(model_type)) {
                throw std::runtime_error("encoder-only model_type '" + model_type +
                                         "' is not supported (imp runs causal decoder LMs; "
                                         "embedding encoders need pooling support)");
            }
            // Common model_type values → architecture class names
            static const std::unordered_map<std::string, std::string> type_to_class = {
                {"llama", "LlamaForCausalLM"},
                {"mistral", "MistralForCausalLM"},
                {"mixtral", "MixtralForCausalLM"},
                {"qwen2", "Qwen2ForCausalLM"},
                {"qwen2_moe", "Qwen2MoeForCausalLM"},
                {"qwen3", "Qwen3ForCausalLM"},
                {"qwen3_moe", "Qwen3MoeForCausalLM"},
                {"qwen3_5", "Qwen3_5ForCausalLM"},
                {"qwen3_5_text", "Qwen3_5ForCausalLM"},
                {"qwen3_5_moe", "Qwen3_5MoeForCausalLM"},
                {"qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"},
                {"nemotron_h", "NemotronHForCausalLM"},
                {"gemma", "GemmaForCausalLM"},
                {"gemma2", "Gemma2ForCausalLM"},
                {"gemma3", "Gemma3ForCausalLM"},
                {"gemma4", "Gemma4ForCausalLM"},
                {"gemma4_unified", "Gemma4ForCausalLM"},
                {"llama4", "Llama4ForCausalLM"},
                {"deepseek_v2", "DeepseekV2ForCausalLM"},
                {"deepseek_v3", "DeepseekV3ForCausalLM"},
                {"phi", "PhiForCausalLM"},
                {"phi3", "Phi3ForCausalLM"},
                {"cohere", "CohereForCausalLM"},
                {"starcoder2", "Starcoder2ForCausalLM"},
            };
            auto it = type_to_class.find(model_type);
            if (it != type_to_class.end()) {
                cfg.arch = map_architecture(it->second);
            } else {
                IMP_LOG_WARN(
                    "unknown model_type '%s' — falling back to GENERIC + "
                    "tensor-name heuristics. If inference is incoherent, the "
                    "architecture is likely unsupported.",
                    model_type.c_str());
                cfg.arch = ModelArch::GENERIC;
            }
        }
    }

    // Multimodal wrappers (Gemma 3/4 ConditionalGeneration) nest the text-model
    // hyperparameters under `text_config`. If present, use that as the effective
    // root for all subsequent reads so we don't have to duplicate every lookup.
    const JValue* text_cfg = jobj_find(root, "text_config");
    const JValue& eff = (text_cfg && text_cfg->type == JType::OBJECT) ? *text_cfg : root;

    // Core dimensions
    jobj_get_int(eff, "hidden_size", cfg.d_model);
    jobj_get_int(eff, "num_attention_heads", cfg.n_heads);
    jobj_get_int(eff, "intermediate_size", cfg.d_ff);
    jobj_get_int(eff, "num_hidden_layers", cfg.n_layers);
    jobj_get_int(eff, "vocab_size", cfg.vocab_size);
    jobj_get_int(eff, "max_position_embeddings", cfg.max_seq_len);
    jobj_get_int(eff, "head_dim", cfg.head_dim);

    // KV heads: default to n_heads (MHA) if not specified
    if (!jobj_get_int(eff, "num_key_value_heads", cfg.n_kv_heads)) {
        cfg.n_kv_heads = cfg.n_heads;
    }

    // Norm epsilon: try rms_norm_eps first, then layer_norm_eps
    if (!jobj_get_float(eff, "rms_norm_eps", cfg.rms_norm_eps)) {
        jobj_get_float(eff, "layer_norm_eps", cfg.rms_norm_eps);
    }

    // RoPE. Newer HF configs (Qwen3.5/3.6, Qwen3-Next) move `rope_theta` and
    // `partial_rotary_factor` under a nested `rope_parameters` object. Read the
    // top-level keys first (older convention) then fall back to the nested
    // `rope_parameters.*` to override. Without this, Qwen3.6 silently runs with
    // theta=10000 instead of 10000000 (1000× wrong base) and full-dim RoPE.
    jobj_get_float(eff, "rope_theta", cfg.rope_theta);
    float partial_factor = 0.0f;
    jobj_get_float(eff, "partial_rotary_factor", partial_factor);
    {
        const JValue* rope_params = jobj_find(eff, "rope_parameters");
        if (rope_params && rope_params->type == JType::OBJECT) {
            jobj_get_float(*rope_params, "rope_theta", cfg.rope_theta);
            jobj_get_float(*rope_params, "partial_rotary_factor", partial_factor);
        }
    }
    if (partial_factor > 0.0f && partial_factor < 1.0f) {
        int hd_for_rope = (cfg.head_dim > 0) ? cfg.head_dim
                                             : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
        if (hd_for_rope > 0) {
            cfg.rope_dim = static_cast<int>(hd_for_rope * partial_factor);
            IMP_LOG_INFO("Partial RoPE: partial_rotary_factor=%.3f head_dim=%d → rope_dim=%d", partial_factor,
                         hd_for_rope, cfg.rope_dim);
        }
    }

    // RoPE scaling (object with type, factor, and optional YaRN/LongRoPE params)
    const JValue* rope_scaling = jobj_find(eff, "rope_scaling");
    if (rope_scaling && rope_scaling->type == JType::OBJECT) {
        std::string rope_type;
        jobj_get_string(*rope_scaling, "type", rope_type);
        // Also check "rope_type" (some HF configs use this instead)
        if (rope_type.empty()) {
            jobj_get_string(*rope_scaling, "rope_type", rope_type);
        }

        float factor = 1.0f;
        jobj_get_float(*rope_scaling, "factor", factor);

        // imp convention (matches the GGUF loader / rope_forward): rope_freq_scale
        // stores the FACTOR (>1, e.g. 32); the kernel applies 1/factor itself.
        // Storing 1/factor here double-inverted YaRN for gpt-oss (#547):
        // interpolated dims rotated factor^2=1024x too fast and the mscale
        // compensation flipped to 0.653 instead of 1.347.
        if (rope_type == "linear") {
            cfg.rope_freq_scale = factor;
        } else if (rope_type == "yarn") {
            cfg.rope_freq_scale = factor;
            jobj_get_float(*rope_scaling, "attn_factor", cfg.yarn_attn_factor);
            jobj_get_float(*rope_scaling, "beta_fast", cfg.yarn_beta_fast);
            jobj_get_float(*rope_scaling, "beta_slow", cfg.yarn_beta_slow);
            jobj_get_int(*rope_scaling, "original_max_position_embeddings", cfg.rope_n_ctx_orig);
            // YaRN uses ext_factor=1.0 by default
            cfg.yarn_ext_factor = 1.0f;
        } else if (rope_type == "longrope" || rope_type == "long_rope") {
            // LongRoPE: per-dimension frequency scaling factors
            jobj_get_int(*rope_scaling, "original_max_position_embeddings", cfg.rope_scaling_orig_max_pos);

            const JValue* short_f = jobj_find(*rope_scaling, "short_factor");
            if (short_f && short_f->type == JType::ARRAY) {
                cfg.rope_short_factor.reserve(short_f->arr.size());
                for (const auto& v : short_f->arr) {
                    cfg.rope_short_factor.push_back(static_cast<float>(v.num_val));
                }
            }
            const JValue* long_f = jobj_find(*rope_scaling, "long_factor");
            if (long_f && long_f->type == JType::ARRAY) {
                cfg.rope_long_factor.reserve(long_f->arr.size());
                for (const auto& v : long_f->arr) {
                    cfg.rope_long_factor.push_back(static_cast<float>(v.num_val));
                }
            }
        } else if (rope_type == "llama3") {
            // Llama-3.x per-frequency RoPE scaling. Reuses LongRoPE infrastructure:
            // we precompute one factor per rope-pair so the kernel sees
            //   freqs[i] = base_freq[i] / factor[i]
            // matching the HF reference algorithm. Since this scaling is
            // independent of sequence length, both short and long arrays carry
            // identical values.
            float low_freq_factor = 1.0f, high_freq_factor = 4.0f;
            int orig_max_pos = 0;
            jobj_get_float(*rope_scaling, "low_freq_factor", low_freq_factor);
            jobj_get_float(*rope_scaling, "high_freq_factor", high_freq_factor);
            jobj_get_int(*rope_scaling, "original_max_position_embeddings", orig_max_pos);
            if (orig_max_pos <= 0) {
                jobj_get_int(eff, "original_max_position_embeddings", orig_max_pos);
            }

            int hd = cfg.head_dim > 0 ? cfg.head_dim
                                      : (cfg.n_heads > 0 ? cfg.d_model / cfg.n_heads : 0);
            int rd = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
            int pairs = rd / 2;
            if (pairs > 0 && orig_max_pos > 0 && factor > 1.0f &&
                high_freq_factor > low_freq_factor) {
                const float low_wavelen = static_cast<float>(orig_max_pos) / low_freq_factor;
                const float high_wavelen = static_cast<float>(orig_max_pos) / high_freq_factor;
                const float two_pi = 6.28318530717958647692f;
                cfg.rope_short_factor.resize(pairs);
                cfg.rope_long_factor.resize(pairs);
                for (int i = 0; i < pairs; i++) {
                    const float base_freq =
                        1.0f / std::pow(cfg.rope_theta, (2.0f * i) / static_cast<float>(rd));
                    const float wavelen = two_pi / base_freq;
                    float pair_factor;
                    if (wavelen < high_wavelen) {
                        pair_factor = 1.0f;
                    } else if (wavelen > low_wavelen) {
                        pair_factor = factor;
                    } else {
                        const float smooth =
                            (static_cast<float>(orig_max_pos) / wavelen - low_freq_factor) /
                            (high_freq_factor - low_freq_factor);
                        pair_factor = factor / (1.0f - smooth + smooth * factor);
                    }
                    cfg.rope_short_factor[i] = pair_factor;
                    cfg.rope_long_factor[i] = pair_factor;
                }
                cfg.rope_scaling_orig_max_pos = orig_max_pos;
                IMP_LOG_INFO(
                    "Llama-3 RoPE: factor=%.1f low=%.1f high=%.1f orig_max_pos=%d → %d freq pairs",
                    factor, low_freq_factor, high_freq_factor, orig_max_pos, pairs);
            } else {
                IMP_LOG_WARN(
                    "Llama-3 RoPE: skipping (pairs=%d orig_max_pos=%d factor=%.2f low=%.2f high=%.2f)",
                    pairs, orig_max_pos, factor, low_freq_factor, high_freq_factor);
            }
        }
        // "dynamic" uses the same factor as linear at runtime
        else if (rope_type == "dynamic") {
            cfg.rope_freq_scale = 1.0f / factor;
        }
    }

    // Sliding window attention
    jobj_get_int(eff, "sliding_window", cfg.sliding_window);

    // Softcapping (Gemma-2/3)
    jobj_get_float(eff, "attn_logit_softcapping", cfg.attn_logit_softcap);
    jobj_get_float(eff, "final_logit_softcapping", cfg.final_logit_softcap);
    // Gemma 4 uses `final_logit_softcapping` same semantics.
    jobj_get_float(root, "final_logit_softcapping", cfg.final_logit_softcap);

    // FFN activation
    std::string hidden_act;
    if (jobj_get_string(eff, "hidden_act", hidden_act) ||
        jobj_get_string(eff, "hidden_activation", hidden_act)) {
        if (hidden_act == "silu" || hidden_act == "swiglu") {
            cfg.ffn_activation = FFNActivation::SWIGLU;
        } else if (hidden_act == "gelu" || hidden_act == "gelu_pytorch_tanh" || hidden_act == "geglu") {
            cfg.ffn_activation = FFNActivation::GEGLU;
        }
    }

    // MoE config
    if (!jobj_get_int(eff, "num_local_experts", cfg.n_experts)) {
        if (!jobj_get_int(eff, "num_experts", cfg.n_experts)) {
            // DeepSeek-V2/V3: routed experts count (distinct from shared experts)
            jobj_get_int(eff, "n_routed_experts", cfg.n_experts);
        }
    }
    if (!jobj_get_int(eff, "num_experts_per_tok", cfg.n_experts_active)) {
        jobj_get_int(eff, "top_k_experts", cfg.n_experts_active);
    }
    if (!jobj_get_int(eff, "moe_intermediate_size", cfg.expert_d_ff)) {
        jobj_get_int(eff, "expert_intermediate_size", cfg.expert_d_ff);
    }
    // Shared (always-active) experts alongside routed experts (DeepSeek-V2/V3,
    // Nemotron-H). Parsed here for all architectures; the arch-specific blocks
    // below may set additional fields.
    {
        int n_shared = 0;
        if (jobj_get_int(eff, "n_shared_experts", n_shared) && n_shared > 0) {
            cfg.n_experts_shared = n_shared;
        }
    }
    // gpt-oss: no separate MoE intermediate key — intermediate_size IS the
    // per-expert FFN width (2880); there is no dense FFN at all.
    if (cfg.expert_d_ff == 0 && cfg.n_experts > 0 && cfg.arch == ModelArch::GPT_OSS) {
        cfg.expert_d_ff = cfg.d_ff;
    }

    // Qwen3.5/3.6 GDN: linear-attention layer config. The HF config exposes
    // these as `linear_*` fields; we map them onto the existing ssm_*
    // ModelConfig slots that the GDN forward path reads (executor_ssm_gdn.cu).
    // Without this plumbing, ssm_inner_size stays 0 → conv_channels=0 →
    // ssm_proj_buf_ allocates 0 bytes → IMA on the first GDN GEMM.
    //
    //   linear_value_head_dim × linear_num_value_heads → ssm_inner_size
    //   linear_key_head_dim                            → ssm_state_size
    //   linear_num_key_heads                           → ssm_group_count
    //   linear_num_value_heads                         → ssm_dt_rank (n_heads)
    //   linear_conv_kernel_dim                         → ssm_conv_kernel
    if (cfg.arch == ModelArch::QWEN36_MOE || cfg.arch == ModelArch::QWEN35_MOE ||
        cfg.arch == ModelArch::QWEN35) {
        // HF SafeTensors stores GDN heads in grouped order (heads 0..n_v_per_k-1
        // belong to group 0, etc). The scan kernel's default `g = h % n_groups`
        // formula assumes the GGUF tiled layout (heads 0..n_groups-1 are replica
        // 0). Set the flag so executor_ssm_gdn passes grouped_layout=1 to the
        // kernel for HF-loaded checkpoints.
        //
        // Cross-converted checkpoints (HF → GGUF → HF, or weights re-packed by
        // a third-party tool) can ship in the opposite layout. Override via
        // gdn.layout_override="tiled"; default
        // for SafeTensors stays grouped.
        cfg.gdn_grouped_head_layout = true;
        {
            const std::string& v = process_diag_gdn_layout_override();
            if (v == "tiled" || v == "TILED") {
                cfg.gdn_grouped_head_layout = false;
                IMP_LOG_INFO("GDN head layout: forced to TILED via gdn.layout_override=tiled");
            } else if (v == "grouped" || v == "GROUPED") {
                cfg.gdn_grouped_head_layout = true;
                IMP_LOG_INFO("GDN head layout: forced to GROUPED via gdn.layout_override=grouped");
            } else if (!v.empty()) {
                IMP_LOG_WARN("gdn.layout_override='%s' not recognized (expected 'tiled' or 'grouped')",
                             v.c_str());
            }
        }

        int lin_v_heads = 0, lin_v_hdim = 0;
        int lin_k_heads = 0, lin_k_hdim = 0;
        int lin_conv = 0;
        jobj_get_int(eff, "linear_num_value_heads", lin_v_heads);
        jobj_get_int(eff, "linear_value_head_dim", lin_v_hdim);
        jobj_get_int(eff, "linear_num_key_heads", lin_k_heads);
        jobj_get_int(eff, "linear_key_head_dim", lin_k_hdim);
        jobj_get_int(eff, "linear_conv_kernel_dim", lin_conv);

        if (lin_v_heads > 0 && lin_v_hdim > 0) {
            cfg.ssm_inner_size = lin_v_heads * lin_v_hdim;
            cfg.ssm_dt_rank = lin_v_heads;
        }
        if (lin_k_hdim > 0)
            cfg.ssm_state_size = lin_k_hdim;
        if (lin_k_heads > 0)
            cfg.ssm_group_count = lin_k_heads;
        if (lin_conv > 0)
            cfg.ssm_conv_kernel = lin_conv;

        // layer_types[] decides per-layer GDN-vs-attention. The GGUF Qwen3.6
        // loader infers this from tensor presence; on the HF side we surface
        // it explicitly so executor_workspace + ssm-state sizing can pick the
        // right layer count.
        const JValue* lt = jobj_find(eff, "layer_types");
        if (lt && lt->type == JType::ARRAY) {
            cfg.n_kv_heads_per_layer.clear();
            cfg.n_kv_heads_per_layer.reserve(lt->arr.size());
            for (const auto& v : lt->arr) {
                // 0 = no attention this layer (GDN), else cfg.n_kv_heads.
                bool is_linear = (v.str_val == "linear_attention");
                cfg.n_kv_heads_per_layer.push_back(is_linear ? 0 : cfg.n_kv_heads);
            }
        }

        IMP_LOG_INFO("  GDN config: inner=%d state=%d groups=%d n_heads=%d conv_kernel=%d",
                     cfg.ssm_inner_size, cfg.ssm_state_size, cfg.ssm_group_count, cfg.ssm_dt_rank,
                     cfg.ssm_conv_kernel);

        // Qwen3.5 / 3.6 MoE shared-expert intermediate size. Used by the MTP
        // forward to size the shared-expert FFN scratch and by diagnostic logs
        // for the main model. Key name differs from DeepSeek-style configs.
        int qwen_shared_d_ff = 0;
        jobj_get_int(eff, "shared_expert_intermediate_size", qwen_shared_d_ff);
        if (qwen_shared_d_ff == 0)
            jobj_get_int(eff, "moe_shared_expert_intermediate_size", qwen_shared_d_ff);
        if (qwen_shared_d_ff > 0)
            cfg.expert_shared_d_ff = qwen_shared_d_ff;
    }

    // Nemotron-H MoE: hybrid Mamba2 + MoE-Expert + Attention. Read the
    // mamba/MoE-specific fields and parse `hybrid_override_pattern` to fill
    // `n_kv_heads_per_layer` so executor_workspace can size buffers per layer.
    // Without this plumbing, ssm_inner_size stays 0 → SSM kernels read past
    // their allocated state → IMA on the first prefill (the symptom we see).
    //
    //   mamba_head_dim × mamba_num_heads → ssm_inner_size
    //   ssm_state_size                    → ssm_state_size
    //   n_groups                          → ssm_group_count
    //   mamba_num_heads                   → ssm_dt_rank (n_heads)
    //   conv_kernel                       → ssm_conv_kernel
    //   hybrid_override_pattern (M/E/*)   → n_kv_heads_per_layer (0 / 0 / n_kv)
    if (cfg.arch == ModelArch::NEMOTRON_H_MOE) {
        int mamba_head_dim = 0, mamba_num_heads = 0, n_groups_v = 0;
        int ssm_state = 0, conv_k = 0;
        jobj_get_int(eff, "mamba_head_dim", mamba_head_dim);
        jobj_get_int(eff, "mamba_num_heads", mamba_num_heads);
        jobj_get_int(eff, "n_groups", n_groups_v);
        jobj_get_int(eff, "ssm_state_size", ssm_state);
        jobj_get_int(eff, "conv_kernel", conv_k);
        if (mamba_head_dim > 0 && mamba_num_heads > 0)
            cfg.ssm_inner_size = mamba_head_dim * mamba_num_heads;
        if (ssm_state > 0)
            cfg.ssm_state_size = ssm_state;
        if (n_groups_v > 0)
            cfg.ssm_group_count = n_groups_v;
        if (mamba_num_heads > 0)
            cfg.ssm_dt_rank = mamba_num_heads;
        if (conv_k > 0)
            cfg.ssm_conv_kernel = conv_k;

        // Extended MoE config (DeepSeek-style routing): n_routed_experts is the
        // total expert count, num_experts_per_tok is top_k (already read above),
        // moe_shared_expert_intermediate_size sizes the always-on shared expert.
        int n_routed = 0, n_shared = 0, shared_d_ff = 0;
        jobj_get_int(eff, "n_routed_experts", n_routed);
        jobj_get_int(eff, "n_shared_experts", n_shared);
        // moe_shared_expert_intermediate_size: DeepSeek / Qwen2-MoE naming.
        // shared_expert_intermediate_size: Qwen3.5 / 3.6 naming (used by their
        // shared-expert variant of GroupQuery MoE).
        jobj_get_int(eff, "moe_shared_expert_intermediate_size", shared_d_ff);
        if (shared_d_ff == 0)
            jobj_get_int(eff, "shared_expert_intermediate_size", shared_d_ff);
        if (n_routed > 0 && cfg.n_experts == 0)
            cfg.n_experts = n_routed;
        if (n_shared > 0)
            cfg.n_experts_shared = n_shared;
        if (shared_d_ff > 0)
            cfg.expert_shared_d_ff = shared_d_ff;
        jobj_get_float(eff, "routed_scaling_factor", cfg.expert_weights_scale);
        // norm_topk_prob arrives as a bool (number 0/1 in our JSON parser)
        const JValue* ntp = jobj_find(eff, "norm_topk_prob");
        if (ntp && ntp->type == JType::NUMBER)
            cfg.expert_weights_norm = (ntp->num_val != 0.0);

        // hybrid_override_pattern is a 52-char string with one char per layer:
        //   M = Mamba2 (no attention)
        //   E = MoE expert FFN (no attention)
        //   * = Attention layer
        std::string pat;
        if (jobj_get_string(eff, "hybrid_override_pattern", pat) && !pat.empty()) {
            cfg.n_kv_heads_per_layer.clear();
            cfg.n_kv_heads_per_layer.reserve(pat.size());
            int n_attn = 0, n_ssm = 0, n_moe = 0;
            for (char c : pat) {
                if (c == '*') {
                    cfg.n_kv_heads_per_layer.push_back(cfg.n_kv_heads);
                    ++n_attn;
                } else {
                    cfg.n_kv_heads_per_layer.push_back(0);
                    if (c == 'M')
                        ++n_ssm;
                    else if (c == 'E')
                        ++n_moe;
                }
            }
            IMP_LOG_INFO("  Nemotron-H hybrid pattern: %d Mamba2 + %d MoE + %d Attention = %d layers",
                         n_ssm, n_moe, n_attn, (int) pat.size());
        }

        IMP_LOG_INFO("  Nemotron-H SSM: inner=%d state=%d groups=%d n_heads=%d conv_kernel=%d",
                     cfg.ssm_inner_size, cfg.ssm_state_size, cfg.ssm_group_count, cfg.ssm_dt_rank,
                     cfg.ssm_conv_kernel);
        IMP_LOG_INFO("  Nemotron-H MoE: n_experts=%d top_k=%d n_shared=%d shared_d_ff=%d "
                     "scale=%.2f norm_topk=%d",
                     cfg.n_experts, cfg.n_experts_active, cfg.n_experts_shared,
                     cfg.expert_shared_d_ff, cfg.expert_weights_scale, (int) cfg.expert_weights_norm);
    }

    // gpt-oss: alternating attention — layer_types[] marks "sliding_attention"
    // (window 128) on even layers, "full_attention" on odd. Uniform head_dim
    // (64); the per-head sink logits + per-expert biases are tensor-level
    // (weight_map.cpp). Router = topk-then-softmax (resolved in the MoE path).
    if (cfg.arch == ModelArch::GPT_OSS) {
        const JValue* lt = jobj_find(eff, "layer_types");
        if (lt && lt->type == JType::ARRAY) {
            cfg.swa_layers.clear();
            cfg.swa_layers.reserve(lt->arr.size());
            for (const auto& v : lt->arr)
                cfg.swa_layers.push_back(v.str_val == "sliding_attention" ? 1 : 0);
        }
        if (cfg.sliding_window <= 0)
            cfg.sliding_window = 128;
    }

    // Gemma 4: per-layer geometry. layer_types[] tells SWA vs global, and
    // head_dim / global_head_dim + num_key_value_heads / num_global_key_value_heads
    // define the dual geometry. Build the per-layer vectors so
    // executor_attention.cu picks up the right shapes/theta per layer.
    if (cfg.arch == ModelArch::GEMMA4) {
        int global_head_dim = 0;
        int num_global_kv = 0;
        jobj_get_int(eff, "global_head_dim", global_head_dim);
        jobj_get_int(eff, "num_global_key_value_heads", num_global_kv);

        // rope params nested under rope_parameters.{full_attention,sliding_attention}
        const JValue* rp = jobj_find(eff, "rope_parameters");
        float theta_full = cfg.rope_theta > 0.0f ? cfg.rope_theta : 1e6f;
        float theta_swa = 1e4f;
        if (rp && rp->type == JType::OBJECT) {
            const JValue* fa = jobj_find(*rp, "full_attention");
            if (fa && fa->type == JType::OBJECT)
                jobj_get_float(*fa, "rope_theta", theta_full);
            const JValue* sa = jobj_find(*rp, "sliding_attention");
            if (sa && sa->type == JType::OBJECT)
                jobj_get_float(*sa, "rope_theta", theta_swa);
        }
        cfg.rope_theta = theta_full;
        cfg.rope_theta_swa = theta_swa;

        const JValue* lt = jobj_find(eff, "layer_types");
        if (lt && lt->type == JType::ARRAY) {
            cfg.swa_layers.clear();
            cfg.head_dim_per_layer.clear();
            cfg.n_kv_heads_per_layer.clear();
            cfg.swa_layers.reserve(lt->arr.size());
            cfg.head_dim_per_layer.reserve(lt->arr.size());
            cfg.n_kv_heads_per_layer.reserve(lt->arr.size());
            for (const auto& v : lt->arr) {
                bool is_swa = (v.str_val == "sliding_attention");
                cfg.swa_layers.push_back(is_swa ? 1 : 0);
                cfg.head_dim_per_layer.push_back(
                    is_swa ? cfg.head_dim : (global_head_dim > 0 ? global_head_dim : cfg.head_dim));
                cfg.n_kv_heads_per_layer.push_back(
                    is_swa ? cfg.n_kv_heads : (num_global_kv > 0 ? num_global_kv : cfg.n_kv_heads));
            }
            // Scalar head_dim = max(per-layer head_dim) so KV-cache buffers and
            // attention workspace are sized for the *largest* head_dim, not the
            // SWA-only value. The GGUF loader does the same (gguf_loader.cpp).
            // Without this, full-attention layers (head_dim=512) write past
            // their allocated stride into adjacent layer slots → corrupted KV
            // cache → garbage attention output. Symptom: L0 attention output
            // diverges from GGUF reference even though Q/K/V projections agree.
            int max_hd = 0;
            for (int v : cfg.head_dim_per_layer)
                max_hd = std::max(max_hd, v);
            if (max_hd > cfg.head_dim) {
                IMP_LOG_INFO("Gemma 4 (HF): scalar head_dim %d → %d (max of per-layer)", cfg.head_dim,
                             max_hd);
                cfg.head_dim = max_hd;
            }
            // Same for n_kv_heads — sizing the cache for the largest layer.
            int max_nkv = 0;
            for (int v : cfg.n_kv_heads_per_layer)
                max_nkv = std::max(max_nkv, v);
            if (max_nkv > cfg.n_kv_heads) {
                IMP_LOG_INFO("Gemma 4 (HF): scalar n_kv_heads %d → %d (max of per-layer)", cfg.n_kv_heads,
                             max_nkv);
                cfg.n_kv_heads = max_nkv;
            }
        }
    }

    // tie_word_embeddings: store as tri-state so the SafeTensors loader can
    // cross-check the flag against actual lm_head.weight presence rather than
    // tying purely on null-detection.
    const JValue* tie = jobj_find(root, "tie_word_embeddings");
    if (tie && tie->type == JType::NUMBER) {
        cfg.tie_word_embeddings = (tie->num_val != 0.0) ? 1 : 0;
        IMP_LOG_INFO("  tie_word_embeddings = %s",
                     cfg.tie_word_embeddings == 1 ? "true" : "false");
    }

    // attention_bias / mlp_bias: tri-state so the loader can warn when the
    // config promises bias but the SafeTensors export omits it.
    const JValue* attn_bias = jobj_find(eff, "attention_bias");
    if (attn_bias && attn_bias->type == JType::NUMBER) {
        cfg.attention_bias = (attn_bias->num_val != 0.0) ? 1 : 0;
    }
    const JValue* mlp_bias = jobj_find(eff, "mlp_bias");
    if (mlp_bias && mlp_bias->type == JType::NUMBER) {
        cfg.mlp_bias = (mlp_bias->num_val != 0.0) ? 1 : 0;
    }

    // DeepSeek V2/V3 Multi-head Latent Attention (MLA) config.
    // kv_lora_rank > 0 is the unambiguous MLA indicator. When present, override
    // head_dim and rope_dim to match the decoupled-head layout.
    if (cfg.arch == ModelArch::DEEPSEEK) {
        jobj_get_int(eff, "kv_lora_rank",        cfg.kv_lora_rank);
        jobj_get_int(eff, "q_lora_rank",         cfg.q_lora_rank);     // absent/null -> stays 0
        jobj_get_int(eff, "qk_rope_head_dim",    cfg.qk_rope_head_dim);
        jobj_get_int(eff, "qk_nope_head_dim",    cfg.qk_nope_head_dim);
        jobj_get_int(eff, "v_head_dim",          cfg.v_head_dim);
        jobj_get_int(eff, "first_k_dense_replace", cfg.first_k_dense_replace);
        if (cfg.is_mla()) {
            // Decoupled-head layout: each attention head has qk_nope_head_dim
            // non-RoPE dims plus qk_rope_head_dim RoPE dims.
            cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
            cfg.rope_dim = cfg.qk_rope_head_dim;
            // YaRN mscale from rope_scaling. HF DeepSeek-V2 uses the two mscale
            // fields DIFFERENTLY and they must NOT be conflated:
            //   - the softmax attention scale gets yarn_get_mscale(factor, mscale_all_dim)^2
            //   - the RoPE cos/sin get the RATIO
            //       yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
            // For DeepSeek-V2-Lite both are 0.707, so the rope ratio is exactly
            // 1.0 (no cos/sin scaling); V3 may differ. Load both separately.
            const JValue* rs = jobj_find(eff, "rope_scaling");
            if (rs && rs->type == JType::OBJECT) {
                float mscale = 1.0f, mscale_all_dim = 1.0f;
                bool has_mscale = jobj_get_float(*rs, "mscale", mscale);
                bool has_all_dim = jobj_get_float(*rs, "mscale_all_dim", mscale_all_dim);
                // Softmax scale prefers mscale_all_dim, falls back to mscale.
                cfg.mla_mscale = has_all_dim ? mscale_all_dim : (has_mscale ? mscale : 1.0f);
                // RoPE ratio numerator is the raw mscale (fallback: mscale_all_dim
                // → ratio 1.0, i.e. no rope scaling, which is HF's default when
                // the two coincide).
                cfg.mla_mscale_num = has_mscale ? mscale : cfg.mla_mscale;
            }
            IMP_LOG_INFO("  MLA: kv_lora_rank=%d q_lora_rank=%d "
                         "qk_rope=%d qk_nope=%d v_head=%d head_dim=%d mla_mscale=%.4f",
                         cfg.kv_lora_rank, cfg.q_lora_rank,
                         cfg.qk_rope_head_dim, cfg.qk_nope_head_dim, cfg.v_head_dim,
                         cfg.head_dim, cfg.mla_mscale);
            // MLA + YaRN rope mscale correction.
            //
            // imp's rope_yarn kernel scales cos/sin by:
            //   mscale_final = yarn_attn_factor * (1 + 0.1 * log(rope_freq_scale))
            //
            // HF DeepseekV2YarnRotaryEmbedding scales cos/sin by the RATIO of the
            // two configured mscales (modeling_deepseek.py):
            //   _mscale = yarn_get_mscale(factor, mscale)
            //             / yarn_get_mscale(factor, mscale_all_dim)
            // where yarn_get_mscale(f, m) = 0.1*m*ln(f) + 1.
            //
            // For V2-Lite (factor=40, mscale == mscale_all_dim == 0.707) this
            // ratio is EXACTLY 1.0 — HF does not scale the rotary at all. The
            // earlier code used yarn_get_mscale(factor, mscale_all_dim)=1.261 as
            // the target, inflating the rope cos/sin by 1.261x; the error
            // compounds with position and cost ~+24% PPL at 512 tokens. Set
            // yarn_attn_factor so mscale_final == the HF ratio.
            //
            // (The softmax attention scale separately receives
            // yarn_get_mscale(factor, mscale_all_dim)^2 via
            // mla_attention_scale_multiplier — that is unchanged.)
            if (cfg.yarn_ext_factor > 0.0f && cfg.rope_freq_scale > 1.0f) {
                const float log_scale = std::log(cfg.rope_freq_scale);
                const float ms_num = 0.1f * cfg.mla_mscale_num * log_scale + 1.0f;
                const float ms_den = 0.1f * cfg.mla_mscale     * log_scale + 1.0f;
                const float hf_rope_mscale = ms_num / ms_den;
                const float imp_mscale = 0.1f * log_scale + 1.0f;
                cfg.yarn_attn_factor = hf_rope_mscale / imp_mscale;
                IMP_LOG_INFO("  MLA YaRN rope-mscale adjust: yarn_attn_factor=%.4f "
                             "(hf_rope_mscale=%.4f, softmax_scale_mult=%.4f)",
                             cfg.yarn_attn_factor, hf_rope_mscale, ms_den * ms_den);
            }
        }
    }

    // Multimodal vision-tower detection. Imp's SafeTensors loader skips
    // vision tensors today; the user only gets the LLM head. Surface this
    // explicitly when `vision_config` is present so chat-with-images use
    // cases don't silently degrade to text-only.
    const JValue* vc = jobj_find(root, "vision_config");
    if (vc && vc->type == JType::OBJECT) {
        std::string model_type;
        jobj_get_string(*vc, "model_type", model_type);
        IMP_LOG_WARN(
            "Multimodal model detected (vision_config present, model_type='%s'). "
            "imp's SafeTensors loader handles only the language head; the "
            "vision tower will be skipped. Multimodal SafeTensors support is "
            "a separate audit item (#18).",
            model_type.empty() ? "?" : model_type.c_str());
    }

    if (cfg.arch == ModelArch::GENERIC) {
        cfg.arch_inferred_fallback = true;
    }

    IMP_LOG_INFO("  arch=%s layers=%d d_model=%d heads=%d kv_heads=%d d_ff=%d vocab=%d",
                 model_arch_name(cfg.arch), cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff,
                 cfg.vocab_size);

    return true;
}

// ---- load_generation_config ----

bool HFConfigLoader::load_generation_config(const std::string& model_dir, GenerationConfig& cfg) {
    std::string path = model_dir + "/generation_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    // EOS IDs (single number or array)
    if (const JValue* eos = jobj_find(root, "eos_token_id")) {
        if (eos->type == JType::NUMBER) {
            cfg.eos_token_ids.push_back(static_cast<int32_t>(eos->num_val));
        } else if (eos->type == JType::ARRAY) {
            for (const auto& v : eos->arr) {
                if (v.type == JType::NUMBER) {
                    cfg.eos_token_ids.push_back(static_cast<int32_t>(v.num_val));
                }
            }
        }
    }

    // Sampling defaults. Each field is only overwritten if the JSON has it;
    // otherwise the struct's sentinel survives.
    jobj_get_float(root, "temperature", cfg.temperature);
    jobj_get_float(root, "top_p", cfg.top_p);
    jobj_get_int(root, "top_k", cfg.top_k);
    jobj_get_float(root, "repetition_penalty", cfg.repetition_penalty);

    // do_sample=false → force greedy. Authors set this when they want
    // deterministic output regardless of temperature.
    if (const JValue* ds = jobj_find(root, "do_sample")) {
        if (ds->type == JType::NUMBER && ds->num_val == 0.0) {
            cfg.temperature = 0.0f;
        }
    }

    IMP_LOG_INFO(
        "generation_config.json: eos=%zu temp=%.3g top_p=%.3g top_k=%d "
        "rep_penalty=%.3g",
        cfg.eos_token_ids.size(), cfg.temperature, cfg.top_p, cfg.top_k, cfg.repetition_penalty);
    return true;
}

// ---- load_chat_template ----

std::string HFConfigLoader::load_chat_template(const std::string& model_dir) {
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return "";

    // Case 1: chat_template is a plain string in tokenizer_config.json
    std::string chat_template;
    if (jobj_get_string(root, "chat_template", chat_template) && !chat_template.empty()) {
        IMP_LOG_INFO("loaded chat_template from tokenizer_config.json (%zu chars)", chat_template.size());
        return chat_template;
    }

    // Case 2: chat_template is an array of {name, template} objects in
    // tokenizer_config.json (HuggingFace format: [{"name":"default","template":"..."}, …]).
    const JValue* ct = jobj_find(root, "chat_template");
    if (ct && ct->type == JType::ARRAY) {
        // Prefer "default" entry
        for (const auto& entry : ct->arr) {
            if (entry.type != JType::OBJECT)
                continue;
            std::string name;
            if (!jobj_get_string(entry, "name", name))
                continue;
            if (name == "default") {
                if (jobj_get_string(entry, "template", chat_template) && !chat_template.empty()) {
                    IMP_LOG_INFO("loaded chat_template (default) from tokenizer_config.json (%zu chars)",
                                 chat_template.size());
                    return chat_template;
                }
            }
        }
        // Fallback: first entry with a valid template string
        for (const auto& entry : ct->arr) {
            if (entry.type != JType::OBJECT)
                continue;
            if (jobj_get_string(entry, "template", chat_template) && !chat_template.empty()) {
                std::string name;
                jobj_get_string(entry, "name", name);
                IMP_LOG_INFO("loaded chat_template (%s) from tokenizer_config.json (%zu chars)",
                             name.empty() ? "unnamed" : name.c_str(), chat_template.size());
                return chat_template;
            }
        }
        IMP_LOG_WARN("chat_template array found but no usable entry in %s", path.c_str());
    }

    // Case 3: standalone chat_template.jinja file alongside tokenizer_config.json.
    // Newer HuggingFace exports (e.g. Gemma-4 llm-compressor NVFP4) ship the
    // template as a separate file with no chat_template field in
    // tokenizer_config.json. Read it raw — the Jinja2 engine consumes the
    // string, so no parsing required here.
    std::string jinja_path = model_dir + "/chat_template.jinja";
    if (std::ifstream in(jinja_path); in.good()) {
        std::stringstream buf;
        buf << in.rdbuf();
        chat_template = buf.str();
        if (!chat_template.empty()) {
            IMP_LOG_INFO("loaded chat_template from chat_template.jinja (%zu chars)", chat_template.size());
            return chat_template;
        }
    }

    return "";
}

// ---- load_added_tokens ----

std::vector<HFConfigLoader::AddedToken> HFConfigLoader::load_added_tokens(const std::string& model_dir) {
    std::vector<AddedToken> result;
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return result;

    const JValue* added = jobj_find(root, "added_tokens_decoder");
    if (!added || added->type != JType::OBJECT)
        return result;

    for (const auto& [id_str, val] : added->obj) {
        if (val.type != JType::OBJECT)
            continue;
        AddedToken tok;
        tok.id = std::atoi(id_str.c_str());
        jobj_get_string(val, "content", tok.content);
        // "special" field — treat as bool via number (true=1.0, false=0.0)
        const JValue* sp = jobj_find(val, "special");
        tok.special = sp && sp->type == JType::NUMBER && sp->num_val != 0.0;
        if (!tok.content.empty()) {
            result.push_back(std::move(tok));
        }
    }

    if (!result.empty()) {
        IMP_LOG_INFO("loaded %zu added tokens from tokenizer_config.json", result.size());
    }
    return result;
}

// ---- load_special_tokens_map ----

namespace {

// Token entry in special_tokens_map.json comes in two shapes:
//   "<unk>"                                    (plain string)
//   {"content": "<s>", "lstrip": false, ...}   (object with metadata)
// Extract the content string from either; returns empty if neither matches.
std::string extract_token_content(const JValue& v) {
    if (v.type == JType::STRING)
        return v.str_val;
    if (v.type == JType::OBJECT) {
        std::string s;
        jobj_get_string(v, "content", s);
        return s;
    }
    return {};
}

}  // namespace

bool HFConfigLoader::load_special_tokens_map(const std::string& model_dir, SpecialTokensMap& out) {
    std::string path = model_dir + "/special_tokens_map.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    if (const JValue* arr = jobj_find(root, "additional_special_tokens"); arr && arr->type == JType::ARRAY) {
        out.additional_special_tokens.reserve(arr->arr.size());
        for (const auto& v : arr->arr) {
            std::string s = extract_token_content(v);
            if (!s.empty())
                out.additional_special_tokens.push_back(std::move(s));
        }
    }

    if (const JValue* bos = jobj_find(root, "bos_token"))
        out.bos_token = extract_token_content(*bos);
    if (const JValue* eos = jobj_find(root, "eos_token"))
        out.eos_token = extract_token_content(*eos);
    if (const JValue* pad = jobj_find(root, "pad_token"))
        out.pad_token = extract_token_content(*pad);
    if (const JValue* unk = jobj_find(root, "unk_token"))
        out.unk_token = extract_token_content(*unk);

    IMP_LOG_INFO(
        "special_tokens_map.json: %zu additional_special_tokens, "
        "bos='%s' eos='%s' pad='%s' unk='%s'",
        out.additional_special_tokens.size(), out.bos_token.c_str(), out.eos_token.c_str(),
        out.pad_token.c_str(), out.unk_token.c_str());
    return true;
}

// ---- load_tokenizer_flags ----

bool HFConfigLoader::load_tokenizer_flags(const std::string& model_dir, TokenizerFlags& out) {
    std::string path = model_dir + "/tokenizer_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    auto read_bool = [&](const char* key, int& field) {
        const JValue* v = jobj_find(root, key);
        if (v && v->type == JType::NUMBER) {
            field = (v->num_val != 0.0) ? 1 : 0;
        }
    };

    read_bool("add_bos_token", out.add_bos_token);
    read_bool("add_eos_token", out.add_eos_token);
    read_bool("add_prefix_space", out.add_prefix_space);
    read_bool("use_default_system_prompt", out.use_default_system_prompt);

    // Read BOS/EOS token content strings — may be plain strings or AddedToken
    // objects ({"__type": "AddedToken", "content": "...", ...}).
    // DeepSeek models use AddedToken objects for their custom BOS/EOS strings.
    if (const JValue* bos = jobj_find(root, "bos_token"))
        out.bos_token = extract_token_content(*bos);
    if (const JValue* eos = jobj_find(root, "eos_token"))
        out.eos_token = extract_token_content(*eos);

    IMP_LOG_INFO(
        "tokenizer_config.json: add_bos=%s add_eos=%s add_prefix_space=%s "
        "use_default_system_prompt=%s bos_token='%s' eos_token='%s'",
        out.add_bos_token < 0 ? "unset" : (out.add_bos_token ? "true" : "false"),
        out.add_eos_token < 0 ? "unset" : (out.add_eos_token ? "true" : "false"),
        out.add_prefix_space < 0 ? "unset" : (out.add_prefix_space ? "true" : "false"),
        out.use_default_system_prompt < 0 ? "unset" : (out.use_default_system_prompt ? "true" : "false"),
        out.bos_token.c_str(), out.eos_token.c_str());
    return true;
}

// ---- load_gptq_config ----

bool HFConfigLoader::load_gptq_config(const std::string& model_dir, GPTQConfig& cfg) {
    std::string path = model_dir + "/quantize_config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    IMP_LOG_INFO("loading GPTQ config from %s", path.c_str());

    jobj_get_int(root, "bits", cfg.bits);
    jobj_get_int(root, "group_size", cfg.group_size);

    const JValue* da = jobj_find(root, "desc_act");
    cfg.desc_act = da && da->type == JType::NUMBER && da->num_val != 0.0;

    IMP_LOG_INFO("  GPTQ: bits=%d group_size=%d desc_act=%s", cfg.bits, cfg.group_size,
                 cfg.desc_act ? "true" : "false");
    return true;
}

// ---- load_nvfp4_config ----

namespace {

bool file_exists_at(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

}  // namespace

bool HFConfigLoader::load_nvfp4_config(const std::string& model_dir, NvFP4Config& cfg) {
    bool has_modelopt = file_exists_at(model_dir + "/hf_quant_config.json");
    bool has_compressor = file_exists_at(model_dir + "/recipe.yaml");

    if (has_modelopt && has_compressor) {
        IMP_LOG_WARN("Both quant config files present in %s — preferring modelopt", model_dir.c_str());
    }

    if (has_modelopt) {
        // Existing modelopt parsing (unchanged from before).
        std::string path = model_dir + "/hf_quant_config.json";
        JValue root;
        if (!parse_json_file(path, root))
            return false;

        const JValue* quant = jobj_find(root, "quantization");
        if (!quant || quant->type != JType::OBJECT)
            return false;

        const JValue* algo = jobj_find(*quant, "quant_algo");
        if (!algo || algo->type != JType::STRING)
            return false;
        if (algo->str_val != "NVFP4" && algo->str_val != "nvfp4")
            return false;

        jobj_get_int(*quant, "group_size", cfg.group_size);

        const JValue* kv_algo = jobj_find(*quant, "kv_cache_quant_algo");
        if (kv_algo && kv_algo->type == JType::STRING)
            cfg.kv_cache_quant_algo = kv_algo->str_val;

        const JValue* exclude = jobj_find(*quant, "exclude_modules");
        if (exclude && exclude->type == JType::ARRAY) {
            for (const auto& v : exclude->arr) {
                if (v.type == JType::STRING)
                    cfg.exclude_modules.push_back(v.str_val);
            }
        }

        cfg.format = NvFP4Format::MODELOPT;
        IMP_LOG_INFO("NVFP4 model (Model Optimizer): group_size=%d, kv_cache=%s, exclude=%zu modules",
                     cfg.group_size, cfg.kv_cache_quant_algo.c_str(), cfg.exclude_modules.size());
        return true;
    }

    if (has_compressor) {
        return imp::llm_compressor::parse_recipe_yaml(model_dir, cfg);
    }

    // Neither file present.
    return false;
}

// ---- load_mxfp4_config ----

bool HFConfigLoader::load_mxfp4_config(const std::string& model_dir, MxFP4Config& cfg) {
    // MXFP4 is declared at config.json:quantization_config (top-level, not
    // a separate file like NVFP4 modelopt). GPT-OSS / TorchAO MXFP4 exports
    // use `quant_method == "mxfp4"` (case-insensitive) plus a `block_size`
    // field that's typically 32 (E8M0 scale per 32 elements).
    std::string path = model_dir + "/config.json";
    JValue root;
    if (!parse_json_file(path, root))
        return false;

    const JValue* qc = jobj_find(root, "quantization_config");
    if (!qc || qc->type != JType::OBJECT)
        return false;

    std::string method;
    if (!jobj_get_string(*qc, "quant_method", method))
        return false;

    // Accept the common spellings.
    bool is_mxfp4 = (method == "mxfp4" || method == "MXFP4" || method == "mx_fp4");
    if (!is_mxfp4)
        return false;

    int bs = 0;
    if (!jobj_get_int(*qc, "block_size", bs)) {
        // Some exports nest the block size under a "weight" sub-object.
        const JValue* w = jobj_find(*qc, "weight");
        if (w && w->type == JType::OBJECT)
            jobj_get_int(*w, "block_size", bs);
    }
    if (bs > 0)
        cfg.block_size = bs;

    IMP_LOG_INFO("MXFP4 config: quant_method=%s block_size=%d", method.c_str(), cfg.block_size);
    return true;
}

// ---- load_awq_config ----

bool HFConfigLoader::load_awq_config(const std::string& model_dir, AWQConfig& cfg) {
    // AWQ ships its config in one of two places:
    //   1) `config.json:quantization_config` (HuggingFace-standard location)
    //   2) a separate `quant_config.json` (older AutoAWQ exports)
    // The fields we care about are `quant_method == "awq"`, `bits`,
    // `group_size`, `zero_point`, `version` ("gemm"/"gemv"/"marlin").
    auto try_parse = [&](const std::string& path, bool nested_under_qc) -> bool {
        JValue root;
        if (!parse_json_file(path, root))
            return false;
        const JValue* base = &root;
        if (nested_under_qc) {
            const JValue* qc = jobj_find(root, "quantization_config");
            if (!qc || qc->type != JType::OBJECT)
                return false;
            base = qc;
        }
        std::string method;
        if (!jobj_get_string(*base, "quant_method", method))
            return false;
        if (method != "awq" && method != "AWQ")
            return false;

        jobj_get_int(*base, "bits", cfg.bits);
        jobj_get_int(*base, "w_bit", cfg.bits);  // older AutoAWQ field name
        jobj_get_int(*base, "group_size", cfg.group_size);
        jobj_get_int(*base, "q_group_size", cfg.group_size);  // older field name

        const JValue* zp = jobj_find(*base, "zero_point");
        if (zp && zp->type == JType::NUMBER)
            cfg.zero_point = (zp->num_val != 0.0);

        jobj_get_string(*base, "version", cfg.version);
        return true;
    };

    if (try_parse(model_dir + "/config.json", /*nested_under_qc=*/true))
        return true;
    if (try_parse(model_dir + "/quant_config.json", /*nested_under_qc=*/false))
        return true;
    return false;
}

}  // namespace imp
