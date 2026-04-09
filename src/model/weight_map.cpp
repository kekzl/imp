#include "model/weight_map.h"
#include "core/logging.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>

namespace imp {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Split a string by delimiter into tokens.
static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == delim) {
            if (i > start) {
                tokens.push_back(s.substr(start, i - start));
            }
            start = i + 1;
        }
    }
    return tokens;
}

// Try to parse a non-negative integer from a string. Returns -1 on failure.
static int parse_int(const std::string& s) {
    if (s.empty()) return -1;
    for (char c : s) {
        if (c < '0' || c > '9') return -1;
    }
    return std::atoi(s.c_str());
}

// Ensure model.layers_ has at least (idx + 1) elements.
static void ensure_layer(Model& model, int idx) {
    if (idx >= static_cast<int>(model.layers_.size())) {
        model.layers_.resize(idx + 1);
    }
}

// Ensure expert vectors within a layer have at least (idx + 1) elements.
static void ensure_expert(TransformerLayer& layer, int idx) {
    int needed = idx + 1;
    if (static_cast<int>(layer.expert_w_gate.size()) < needed)
        layer.expert_w_gate.resize(needed);
    if (static_cast<int>(layer.expert_w_up.size()) < needed)
        layer.expert_w_up.resize(needed);
    if (static_cast<int>(layer.expert_w_down.size()) < needed)
        layer.expert_w_down.resize(needed);
    if (static_cast<int>(layer.expert_nvfp4_gate.size()) < needed)
        layer.expert_nvfp4_gate.resize(needed);
    if (static_cast<int>(layer.expert_nvfp4_up.size()) < needed)
        layer.expert_nvfp4_up.resize(needed);
    if (static_cast<int>(layer.expert_nvfp4_down.size()) < needed)
        layer.expert_nvfp4_down.resize(needed);
}

// ---------------------------------------------------------------------------
// WeightMap
// ---------------------------------------------------------------------------

WeightMap::WeightMap(ModelArch arch) : arch_(arch) {
    // name_map_ is used by map_name() for quick lookups of non-layer weights.
    // Layer weights are handled by pattern matching in apply_weights().

    // All supported architectures share the same top-level embedding names
    // (HuggingFace convention).
    name_map_["model.embed_tokens.weight"] = "tok_emb";
    name_map_["model.norm.weight"]         = "out_norm";
    name_map_["lm_head.weight"]            = "out_proj";
}

std::string WeightMap::map_name(const std::string& name) const {
    // Check static map first.
    auto it = name_map_.find(name);
    if (it != name_map_.end()) {
        return it->second;
    }

    // Pattern-match layer weights to produce a human-readable internal name.
    auto parts = split(name, '.');
    // Expected: model . layers . {i} . <rest>
    if (parts.size() >= 4 && parts[0] == "model" && parts[1] == "layers") {
        int layer = parse_int(parts[2]);
        if (layer < 0) return name;

        std::string prefix = "layer." + parts[2] + ".";

        // Attention weights: self_attn.{q,k,v,o}_proj.weight
        if (parts.size() >= 6 && parts[3] == "self_attn" && parts[5] == "weight") {
            if (parts[4] == "q_proj") return prefix + "wq";
            if (parts[4] == "k_proj") return prefix + "wk";
            if (parts[4] == "v_proj") return prefix + "wv";
            if (parts[4] == "o_proj") return prefix + "wo";
        }

        // Attention norm
        if (parts.size() >= 5 && parts[3] == "input_layernorm" && parts[4] == "weight") {
            return prefix + "attn_norm";
        }

        // FFN norm
        if (parts.size() >= 5 && parts[3] == "post_attention_layernorm" && parts[4] == "weight") {
            return prefix + "ffn_norm";
        }

        // Dense MLP: mlp.{gate_proj,up_proj,down_proj}.weight
        if (parts.size() >= 6 && parts[3] == "mlp" && parts[5] == "weight") {
            if (parts[4] == "gate_proj") return prefix + "w_gate";
            if (parts[4] == "up_proj")   return prefix + "w_up";
            if (parts[4] == "down_proj") return prefix + "w_down";
        }

        // Mixtral MoE: block_sparse_moe.gate.weight
        if (parts.size() >= 6 && parts[3] == "block_sparse_moe" &&
            parts[4] == "gate" && parts[5] == "weight") {
            return prefix + "moe_gate";
        }

        // Mixtral MoE experts: block_sparse_moe.experts.{e}.w{1,2,3}.weight
        if (parts.size() >= 8 && parts[3] == "block_sparse_moe" &&
            parts[4] == "experts" && parts[7] == "weight") {
            int expert = parse_int(parts[5]);
            if (expert >= 0) {
                std::string ep = prefix + "expert." + parts[5] + ".";
                if (parts[6] == "w1") return ep + "w_gate";
                if (parts[6] == "w3") return ep + "w_up";
                if (parts[6] == "w2") return ep + "w_down";
            }
        }

        // DeepSeek MoE router: mlp.gate.weight / mlp.gate.bias
        if (parts.size() >= 6 && parts[3] == "mlp" && parts[4] == "gate") {
            if (parts[5] == "weight") return prefix + "moe_gate";
            if (parts[5] == "bias")   return prefix + "moe_router_bias";
        }

        // DeepSeek MoE experts: mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight
        if (parts.size() >= 8 && parts[3] == "mlp" &&
            parts[4] == "experts" && parts[7] == "weight") {
            int expert = parse_int(parts[5]);
            if (expert >= 0) {
                std::string ep = prefix + "expert." + parts[5] + ".";
                if (parts[6] == "gate_proj") return ep + "w_gate";
                if (parts[6] == "up_proj")   return ep + "w_up";
                if (parts[6] == "down_proj") return ep + "w_down";
            }
        }

        // Shared expert: mlp.shared_expert.{gate,up,down}_proj.weight
        if (parts.size() >= 7 && parts[3] == "mlp" &&
            parts[4] == "shared_expert" && parts[6] == "weight") {
            if (parts[5] == "gate_proj") return prefix + "w_gate_shared";
            if (parts[5] == "up_proj")   return prefix + "w_up_shared";
            if (parts[5] == "down_proj") return prefix + "w_down_shared";
        }

        // Attention biases: self_attn.{q,k,v}_proj.bias
        if (parts.size() >= 6 && parts[3] == "self_attn" && parts[5] == "bias") {
            if (parts[4] == "q_proj") return prefix + "q_bias";
            if (parts[4] == "k_proj") return prefix + "k_bias";
            if (parts[4] == "v_proj") return prefix + "v_bias";
        }

        // QK-Norm: self_attn.{q,k}_norm.weight
        if (parts.size() >= 6 && parts[3] == "self_attn" && parts[5] == "weight") {
            if (parts[4] == "q_norm") return prefix + "attn_q_norm";
            if (parts[4] == "k_norm") return prefix + "attn_k_norm";
        }

        // Post-layer norms (Gemma-3)
        if (parts.size() >= 5 && parts[4] == "weight") {
            if (parts[3] == "post_feedforward_layernorm") return prefix + "post_ffn_norm";
            if (parts[3] == "pre_feedforward_layernorm")  return prefix + "ffn_norm";
        }

        // Mixtral MoE router bias: block_sparse_moe.gate.bias
        if (parts.size() >= 6 && parts[3] == "block_sparse_moe" &&
            parts[4] == "gate" && parts[5] == "bias") {
            return prefix + "moe_router_bias";
        }

        // GPTQ weights: self_attn.{q,k,v,o}_proj.{qweight,qzeros,scales,g_idx}
        if (parts.size() >= 6 && parts[3] == "self_attn") {
            const std::string& field = parts[5];
            if (field == "qweight" || field == "qzeros" || field == "scales" || field == "g_idx") {
                return prefix + parts[4] + "." + field;
            }
        }
        // GPTQ weights: mlp.{gate,up,down}_proj.{qweight,qzeros,scales,g_idx}
        if (parts.size() >= 6 && parts[3] == "mlp") {
            const std::string& field = parts[5];
            if (field == "qweight" || field == "qzeros" || field == "scales" || field == "g_idx") {
                return prefix + parts[4] + "." + field;
            }
        }

        // GDN (Gated DeltaNet / Qwen3.5): temporal_block.{gate_proj,alpha,beta}.weight
        if (parts.size() >= 6 && parts[3] == "temporal_block" && parts[5] == "weight") {
            if (parts[4] == "gate_proj") return prefix + "gdn_gate";
            if (parts[4] == "alpha")     return prefix + "gdn_alpha";
            if (parts[4] == "beta")      return prefix + "gdn_beta";
        }

        // SSM (Mamba2 / Nemotron-H)
        if (parts[3] == "mamba") {
            if (parts.size() >= 6 && parts[5] == "weight") {
                if (parts[4] == "in_proj")  return prefix + "ssm_in";
                if (parts[4] == "out_proj") return prefix + "ssm_out";
                if (parts[4] == "conv1d")   return prefix + "ssm_conv1d_w";
                if (parts[4] == "norm")     return prefix + "ssm_norm_w";
            }
            if (parts.size() >= 6 && parts[4] == "conv1d" && parts[5] == "bias")
                return prefix + "ssm_conv1d_b";
            if (parts.size() >= 5 && parts[4] == "dt_bias") return prefix + "ssm_dt_b";
            if (parts.size() >= 5 && parts[4] == "A_log")   return prefix + "ssm_a";
            if (parts.size() >= 5 && parts[4] == "D")       return prefix + "ssm_d";
        }
    }

    return name;
}

bool WeightMap::apply_weights(
        Model& model,
        const std::unordered_map<std::string, Tensor>& tensors) {

    if (tensors.empty()) {
        IMP_LOG_ERROR("WeightMap: no tensors to apply");
        return false;
    }

    int assigned = 0;
    int skipped  = 0;

    const bool is_gemma4 = (arch_ == ModelArch::GEMMA4);

    for (auto& [orig_name, tensor] : tensors) {
        // Gemma 4 uses Gemma4ForConditionalGeneration wrapper with
        // `model.language_model.` and `model.vision_tower.` prefixes. Strip
        // the language_model prefix so downstream handlers see the same
        // `model.layers.X...` layout as other archs. Skip vision tower for
        // now (text-only loading path).
        std::string name = orig_name;
        if (is_gemma4) {
            const std::string vt_prefix = "model.vision_tower.";
            if (name.compare(0, vt_prefix.size(), vt_prefix) == 0) {
                ++skipped;  // silently skip vision encoder weights
                continue;
            }
            const std::string lm_prefix = "model.language_model.";
            if (name.compare(0, lm_prefix.size(), lm_prefix) == 0) {
                name = "model." + name.substr(lm_prefix.size());
            }
            // Also handle top-level aliases the wrapper introduces.
            if (name == "model.embed_tokens.weight" ||
                name == "language_model.embed_tokens.weight") {
                name = "model.embed_tokens.weight";
            }
        }
        auto parts = split(name, '.');

        // -----------------------------------------------------------------
        // Top-level (non-layer) weights
        // -----------------------------------------------------------------
        if (name == "model.embed_tokens.weight") {
            model.tok_emb_ = tensor;
            IMP_LOG_DEBUG("  assigned: %s -> tok_emb", name.c_str());
            ++assigned;
            continue;
        }
        if (name == "model.norm.weight") {
            model.out_norm_ = tensor;
            IMP_LOG_DEBUG("  assigned: %s -> out_norm", name.c_str());
            ++assigned;
            continue;
        }
        if (name == "lm_head.weight") {
            model.out_proj_ = tensor;
            IMP_LOG_DEBUG("  assigned: %s -> out_proj", name.c_str());
            ++assigned;
            continue;
        }
        // NVFP4 prequant LM head scales (Model Optimizer)
        if (name == "lm_head.weight_scale") {
            model.nvfp4_out_proj_.weight_scale = tensor;
            IMP_LOG_DEBUG("  assigned: %s -> nvfp4_out_proj.weight_scale", name.c_str());
            ++assigned;
            continue;
        }
        if (name == "lm_head.weight_scale_2") {
            model.nvfp4_out_proj_.weight_scale_2 = tensor;
            IMP_LOG_DEBUG("  assigned: %s -> nvfp4_out_proj.weight_scale_2", name.c_str());
            ++assigned;
            continue;
        }
        if (name == "lm_head.input_scale") {
            model.nvfp4_out_proj_.input_scale = tensor;
            IMP_LOG_DEBUG("  assigned: %s -> nvfp4_out_proj.input_scale", name.c_str());
            ++assigned;
            continue;
        }

        // -----------------------------------------------------------------
        // Layer weights: model.layers.{i}.<rest>
        // -----------------------------------------------------------------
        if (parts.size() < 4 || parts[0] != "model" || parts[1] != "layers") {
            IMP_LOG_WARN("WeightMap: unrecognised weight name: %s", name.c_str());
            ++skipped;
            continue;
        }

        int layer_idx = parse_int(parts[2]);
        if (layer_idx < 0) {
            IMP_LOG_WARN("WeightMap: bad layer index in: %s", name.c_str());
            ++skipped;
            continue;
        }
        ensure_layer(model, layer_idx);
        TransformerLayer& layer = model.layers_[layer_idx];

        bool matched = false;

        // -- Attention: self_attn.{q,k,v,o}_proj.weight --
        if (parts.size() >= 6 && parts[3] == "self_attn" && parts[5] == "weight") {
            const std::string& proj = parts[4];
            if (proj == "q_proj") { layer.wq = tensor; matched = true; }
            else if (proj == "k_proj") { layer.wk = tensor; matched = true; }
            else if (proj == "v_proj") { layer.wv = tensor; matched = true; }
            else if (proj == "o_proj") { layer.wo = tensor; matched = true; }
        }

        // -- Attention norm: input_layernorm.weight --
        if (!matched && parts.size() >= 5 &&
            parts[3] == "input_layernorm" && parts[4] == "weight") {
            layer.attn_norm = tensor;
            matched = true;
        }

        // -- FFN norm: post_attention_layernorm.weight --
        //    Llama convention: post_attention_layernorm is actually the pre-FFN norm.
        //    Gemma 3/4 convention: post_attention_layernorm is the sandwich norm
        //    applied AFTER attention output. Routed below in the Gemma-4 block.
        if (!matched && !is_gemma4 && parts.size() >= 5 &&
            parts[3] == "post_attention_layernorm" && parts[4] == "weight") {
            layer.ffn_norm = tensor;
            matched = true;
        }

        // -- Gemma 4: mlp.{gate,up,down}_proj.weight is the SHARED EXPERT,
        //    NOT dense MLP. Route to w_*_shared instead. Must come before
        //    the generic dense-MLP branch below.
        if (!matched && is_gemma4 && parts.size() >= 6 &&
            parts[3] == "mlp" && parts[5] == "weight") {
            const std::string& proj = parts[4];
            if (proj == "gate_proj") { layer.w_gate_shared = tensor; matched = true; }
            else if (proj == "up_proj") { layer.w_up_shared = tensor; matched = true; }
            else if (proj == "down_proj") { layer.w_down_shared = tensor; matched = true; }
        }

        // -- Gemma 4: router + packed MoE experts + per-layer extras --
        //    experts.gate_up_proj  (3D fused [n_exp, 2*moe_ff, d]) -> expert_gate_packed
        //    experts.down_proj     (3D      [n_exp, d, moe_ff])    -> expert_down_packed
        //    router.proj.weight    [n_exp, d]                      -> moe_gate
        //    *_layernorm(_1|_2) variants                            -> ffn_{pre,post}_norm_{1,2}
        //    post_attention_layernorm.weight                        -> post_attn_norm
        if (!matched && is_gemma4) {
            // experts.gate_up_proj / experts.down_proj
            if (parts.size() >= 5 && parts[3] == "experts") {
                if (parts[4] == "gate_up_proj") {
                    layer.expert_gate_packed = tensor;
                    // expert_up_packed left null → weight_upload splits the fused tensor
                    matched = true;
                } else if (parts[4] == "down_proj") {
                    layer.expert_down_packed = tensor;
                    matched = true;
                }
            }
            // router.proj.weight (the gating matrix)
            else if (parts.size() >= 6 && parts[3] == "router" &&
                     parts[4] == "proj" && parts[5] == "weight") {
                layer.moe_gate = tensor;
                matched = true;
            }
            // router.scale  (per-channel router input scale == ffn_gate_inp.scale)
            else if (parts.size() >= 5 && parts[3] == "router" &&
                     parts[4] == "scale") {
                layer.ffn_gate_inp_scale = tensor;
                matched = true;
            }
            // router.per_expert_scale  (per-expert down output scale)
            else if (parts.size() >= 5 && parts[3] == "router" &&
                     parts[4] == "per_expert_scale") {
                layer.expert_down_scale = tensor;
                matched = true;
            }
            // layer_scalar  (per-layer output scalar)
            else if (parts.size() >= 4 && parts[3] == "layer_scalar") {
                layer.layer_out_scale = tensor;
                matched = true;
            }
            // Gemma 4 FFN norm variants (parallel shared-MLP + MoE branches)
            else if (parts.size() >= 5 && parts[4] == "weight") {
                if (parts[3] == "pre_feedforward_layernorm_2") {
                    layer.ffn_pre_norm_2 = tensor; matched = true;
                } else if (parts[3] == "post_feedforward_layernorm_1") {
                    layer.ffn_post_norm_1 = tensor; matched = true;
                } else if (parts[3] == "post_feedforward_layernorm_2") {
                    layer.ffn_post_norm_2 = tensor; matched = true;
                } else if (parts[3] == "post_attention_layernorm") {
                    // Gemma 3/4 sandwich norm — distinct from Llama's FFN norm.
                    layer.post_attn_norm = tensor; matched = true;
                }
            }
        }

        // -- Dense MLP (Llama / Mistral / DeepSeek dense layers) --
        if (!matched && parts.size() >= 6 && parts[3] == "mlp" && parts[5] == "weight") {
            const std::string& proj = parts[4];
            if (proj == "gate_proj") { layer.w_gate = tensor; matched = true; }
            else if (proj == "up_proj") { layer.w_up = tensor; matched = true; }
            else if (proj == "down_proj") { layer.w_down = tensor; matched = true; }
        }

        // -- NVFP4 scale tensors (ModelOpt pre-quantized) --
        // self_attn.{q,k,v,o}_proj.{weight_scale,weight_scale_2,input_scale}
        if (!matched && parts.size() >= 6 && parts[3] == "self_attn" &&
            (parts[5] == "weight_scale" || parts[5] == "weight_scale_2" || parts[5] == "input_scale")) {
            const std::string& proj = parts[4];
            const std::string& kind = parts[5];
            auto assign = [&](TransformerLayer::NvFP4PreQuantWeight& nw) {
                if (kind == "weight_scale")   nw.weight_scale = tensor;
                else if (kind == "weight_scale_2") nw.weight_scale_2 = tensor;
                else if (kind == "input_scale")    nw.input_scale = tensor;
            };
            if (proj == "q_proj") { assign(layer.nvfp4_q); matched = true; }
            else if (proj == "k_proj") { assign(layer.nvfp4_k); matched = true; }
            else if (proj == "v_proj") { assign(layer.nvfp4_v); matched = true; }
            else if (proj == "o_proj") { assign(layer.nvfp4_o); matched = true; }
        }
        // mlp.{gate,up,down}_proj.{weight_scale,weight_scale_2,input_scale}
        if (!matched && parts.size() >= 6 && parts[3] == "mlp" &&
            (parts[5] == "weight_scale" || parts[5] == "weight_scale_2" || parts[5] == "input_scale")) {
            const std::string& proj = parts[4];
            const std::string& kind = parts[5];
            auto assign = [&](TransformerLayer::NvFP4PreQuantWeight& nw) {
                if (kind == "weight_scale")   nw.weight_scale = tensor;
                else if (kind == "weight_scale_2") nw.weight_scale_2 = tensor;
                else if (kind == "input_scale")    nw.input_scale = tensor;
            };
            if (proj == "gate_proj") { assign(layer.nvfp4_gate); matched = true; }
            else if (proj == "up_proj") { assign(layer.nvfp4_up); matched = true; }
            else if (proj == "down_proj") { assign(layer.nvfp4_down); matched = true; }
        }

        // -----------------------------------------------------------------
        // MoE weights -- Mixtral style
        //   block_sparse_moe.gate.weight               -> moe_gate
        //   block_sparse_moe.experts.{e}.w1.weight      -> expert_w_gate[e]
        //   block_sparse_moe.experts.{e}.w3.weight      -> expert_w_up[e]
        //   block_sparse_moe.experts.{e}.w2.weight      -> expert_w_down[e]
        // -----------------------------------------------------------------
        if (!matched && parts[3] == "block_sparse_moe") {
            if (parts.size() >= 6 && parts[4] == "gate" && parts[5] == "weight") {
                layer.moe_gate = tensor;
                matched = true;
            } else if (parts.size() >= 8 && parts[4] == "experts" && parts[7] == "weight") {
                int expert_idx = parse_int(parts[5]);
                if (expert_idx >= 0) {
                    ensure_expert(layer, expert_idx);
                    const std::string& wname = parts[6];
                    if (wname == "w1") { layer.expert_w_gate[expert_idx] = tensor; matched = true; }
                    else if (wname == "w3") { layer.expert_w_up[expert_idx] = tensor; matched = true; }
                    else if (wname == "w2") { layer.expert_w_down[expert_idx] = tensor; matched = true; }
                }
            }
        }

        // -----------------------------------------------------------------
        // MoE weights -- DeepSeek style
        //   mlp.gate.weight                              -> moe_gate
        //   mlp.gate.bias                                -> moe_router_bias
        //   mlp.experts.{e}.gate_proj.weight             -> expert_w_gate[e]
        //   mlp.experts.{e}.up_proj.weight               -> expert_w_up[e]
        //   mlp.experts.{e}.down_proj.weight             -> expert_w_down[e]
        //   mlp.shared_expert.{gate,up,down}_proj.weight -> w_{gate,up,down}_shared
        // -----------------------------------------------------------------
        if (!matched && parts[3] == "mlp") {
            // MoE router: mlp.gate.weight
            // Note: parts[4]=="gate" && parts[5]=="weight" with exactly 6 parts
            // distinguishes from dense mlp.gate_proj.weight (which has
            // parts[4]=="gate_proj").
            if (parts.size() >= 6 && parts[4] == "gate" && parts[5] == "weight") {
                layer.moe_gate = tensor;
                matched = true;
            }
            // MoE router bias: mlp.gate.bias
            else if (parts.size() >= 6 && parts[4] == "gate" && parts[5] == "bias") {
                layer.moe_router_bias = tensor;
                matched = true;
            }
            // MoE experts: mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight
            else if (parts.size() >= 8 && parts[4] == "experts" && parts[7] == "weight") {
                int expert_idx = parse_int(parts[5]);
                if (expert_idx >= 0) {
                    ensure_expert(layer, expert_idx);
                    const std::string& proj = parts[6];
                    if (proj == "gate_proj") { layer.expert_w_gate[expert_idx] = tensor; matched = true; }
                    else if (proj == "up_proj") { layer.expert_w_up[expert_idx] = tensor; matched = true; }
                    else if (proj == "down_proj") { layer.expert_w_down[expert_idx] = tensor; matched = true; }
                }
            }
            // MoE expert NVFP4 scales: mlp.experts.{e}.{proj}.{weight_scale,weight_scale_2,input_scale}
            else if (parts.size() >= 8 && parts[4] == "experts" &&
                     (parts[7] == "weight_scale" || parts[7] == "weight_scale_2" || parts[7] == "input_scale")) {
                int expert_idx = parse_int(parts[5]);
                if (expert_idx >= 0) {
                    ensure_expert(layer, expert_idx);
                    const std::string& proj = parts[6];
                    const std::string& kind = parts[7];
                    auto assign = [&](TransformerLayer::NvFP4PreQuantWeight& nw) {
                        if (kind == "weight_scale")   nw.weight_scale = tensor;
                        else if (kind == "weight_scale_2") nw.weight_scale_2 = tensor;
                        else if (kind == "input_scale")    nw.input_scale = tensor;
                    };
                    if (proj == "gate_proj") { assign(layer.expert_nvfp4_gate[expert_idx]); matched = true; }
                    else if (proj == "up_proj") { assign(layer.expert_nvfp4_up[expert_idx]); matched = true; }
                    else if (proj == "down_proj") { assign(layer.expert_nvfp4_down[expert_idx]); matched = true; }
                }
            }
            // Shared expert: mlp.shared_expert.{gate,up,down}_proj.weight
            else if (parts.size() >= 7 && parts[4] == "shared_expert" && parts[6] == "weight") {
                const std::string& proj = parts[5];
                if (proj == "gate_proj") { layer.w_gate_shared = tensor; matched = true; }
                else if (proj == "up_proj") { layer.w_up_shared = tensor; matched = true; }
                else if (proj == "down_proj") { layer.w_down_shared = tensor; matched = true; }
            }
        }

        // -----------------------------------------------------------------
        // MoE router bias -- Mixtral style: block_sparse_moe.gate.bias
        // -----------------------------------------------------------------
        if (!matched && parts[3] == "block_sparse_moe" &&
            parts.size() >= 6 && parts[4] == "gate" && parts[5] == "bias") {
            layer.moe_router_bias = tensor;
            matched = true;
        }

        // -----------------------------------------------------------------
        // Attention biases -- Qwen2-style: self_attn.{q,k,v}_proj.bias
        // -----------------------------------------------------------------
        if (!matched && parts.size() >= 6 && parts[3] == "self_attn" && parts[5] == "bias") {
            const std::string& proj = parts[4];
            if (proj == "q_proj") { layer.q_bias = tensor; matched = true; }
            else if (proj == "k_proj") { layer.k_bias = tensor; matched = true; }
            else if (proj == "v_proj") { layer.v_bias = tensor; matched = true; }
        }

        // -----------------------------------------------------------------
        // QK-Norm -- Qwen3-style: self_attn.{q,k}_norm.weight
        // -----------------------------------------------------------------
        if (!matched && parts.size() >= 6 && parts[3] == "self_attn" && parts[5] == "weight") {
            const std::string& proj = parts[4];
            if (proj == "q_norm") { layer.attn_q_norm = tensor; matched = true; }
            else if (proj == "k_norm") { layer.attn_k_norm = tensor; matched = true; }
        }

        // -----------------------------------------------------------------
        // Post-layer norms -- Gemma-3 style
        //   post_feedforward_layernorm.weight  -> post_ffn_norm
        //   pre_feedforward_layernorm.weight   -> ffn_norm (Gemma variant)
        //   post_attention_layernorm.weight     (already handled as ffn_norm above)
        // -----------------------------------------------------------------
        if (!matched && parts.size() >= 5 && parts[4] == "weight") {
            if (parts[3] == "post_feedforward_layernorm") {
                layer.post_ffn_norm = tensor;
                matched = true;
            } else if (parts[3] == "pre_feedforward_layernorm") {
                layer.ffn_norm = tensor;
                matched = true;
            }
        }

        // -----------------------------------------------------------------
        // GPTQ weights: self_attn.{q,k,v,o}_proj.{qweight,qzeros,scales,g_idx}
        //               mlp.{gate,up,down}_proj.{qweight,qzeros,scales,g_idx}
        // -----------------------------------------------------------------
        if (!matched && parts.size() >= 6 && parts[3] == "self_attn") {
            const std::string& proj = parts[4];
            const std::string& field = parts[5];
            TransformerLayer::GPTQWeight* gptq = nullptr;
            if (proj == "q_proj") gptq = &layer.gptq_q;
            else if (proj == "k_proj") gptq = &layer.gptq_k;
            else if (proj == "v_proj") gptq = &layer.gptq_v;
            else if (proj == "o_proj") gptq = &layer.gptq_o;

            if (gptq) {
                if (field == "qweight") { gptq->qweight = tensor; matched = true; }
                else if (field == "qzeros") { gptq->qzeros = tensor; matched = true; }
                else if (field == "scales") { gptq->scales = tensor; matched = true; }
                else if (field == "g_idx") { gptq->g_idx = tensor; matched = true; }
            }
        }

        if (!matched && parts.size() >= 6 && parts[3] == "mlp") {
            const std::string& proj = parts[4];
            const std::string& field = parts[5];
            TransformerLayer::GPTQWeight* gptq = nullptr;
            if (proj == "gate_proj") gptq = &layer.gptq_gate;
            else if (proj == "up_proj") gptq = &layer.gptq_up;
            else if (proj == "down_proj") gptq = &layer.gptq_down;

            if (gptq) {
                if (field == "qweight") { gptq->qweight = tensor; matched = true; }
                else if (field == "qzeros") { gptq->qzeros = tensor; matched = true; }
                else if (field == "scales") { gptq->scales = tensor; matched = true; }
                else if (field == "g_idx") { gptq->g_idx = tensor; matched = true; }
            }
        }

        // -----------------------------------------------------------------
        // GDN (Gated DeltaNet / Qwen3.5) weights
        //   temporal_block.gate_proj.weight  -> gdn_gate
        //   temporal_block.alpha.weight      -> gdn_alpha
        //   temporal_block.beta.weight       -> gdn_beta
        // -----------------------------------------------------------------
        if (!matched && parts.size() >= 6 && parts[3] == "temporal_block" && parts[5] == "weight") {
            const std::string& proj = parts[4];
            if (proj == "gate_proj") { layer.gdn_gate = tensor; matched = true; }
            else if (proj == "alpha") { layer.gdn_alpha = tensor; matched = true; }
            else if (proj == "beta") { layer.gdn_beta = tensor; matched = true; }
        }

        // -----------------------------------------------------------------
        // SSM (Mamba2 / Nemotron-H) weights
        //   mamba.in_proj.weight   -> ssm_in
        //   mamba.out_proj.weight  -> ssm_out
        //   mamba.conv1d.weight    -> ssm_conv1d_w
        //   mamba.conv1d.bias      -> ssm_conv1d_b
        //   mamba.dt_bias          -> ssm_dt_b
        //   mamba.A_log            -> ssm_a
        //   mamba.D                -> ssm_d
        //   mamba.norm.weight      -> ssm_norm_w
        // -----------------------------------------------------------------
        if (!matched && parts[3] == "mamba") {
            if (parts.size() >= 6 && parts[5] == "weight") {
                const std::string& proj = parts[4];
                if (proj == "in_proj") { layer.ssm_in = tensor; matched = true; }
                else if (proj == "out_proj") { layer.ssm_out = tensor; matched = true; }
                else if (proj == "conv1d") { layer.ssm_conv1d_w = tensor; matched = true; }
                else if (proj == "norm") { layer.ssm_norm_w = tensor; matched = true; }
            } else if (parts.size() >= 6 && parts[4] == "conv1d" && parts[5] == "bias") {
                layer.ssm_conv1d_b = tensor;
                matched = true;
            } else if (parts.size() >= 5 && parts[4] == "dt_bias") {
                layer.ssm_dt_b = tensor;
                matched = true;
            } else if (parts.size() >= 5 && parts[4] == "A_log") {
                layer.ssm_a = tensor;
                matched = true;
            } else if (parts.size() >= 5 && parts[4] == "D") {
                layer.ssm_d = tensor;
                matched = true;
            }
        }

        if (matched) {
            IMP_LOG_DEBUG("  assigned: %s -> layer %d", name.c_str(), layer_idx);
            ++assigned;
        } else {
            IMP_LOG_WARN("WeightMap: unrecognised layer weight: %s", name.c_str());
            ++skipped;
        }
    }

    // Update n_layers in the config to reflect what we actually loaded.
    model.config_.n_layers = static_cast<int>(model.layers_.size());

    // Update n_experts from the first layer that has experts.
    for (auto& layer : model.layers_) {
        int ne = static_cast<int>(layer.expert_w_gate.size());
        if (ne > 0) {
            model.config_.n_experts = std::max(model.config_.n_experts, ne);
        }
    }

    IMP_LOG_INFO("WeightMap (%s): assigned %d tensors, skipped %d, "
                 "layers=%d, experts=%d",
                 model_arch_name(arch_), assigned, skipped,
                 model.config_.n_layers, model.config_.n_experts);

    if (assigned == 0) {
        IMP_LOG_ERROR("WeightMap: no tensors were assigned -- check weight names");
        return false;
    }

    // Validate that essential weights were populated.
    if (!model.tok_emb_.data) {
        IMP_LOG_WARN("WeightMap: token embedding (tok_emb) was not found");
    }
    if (!model.out_norm_.data) {
        IMP_LOG_WARN("WeightMap: output norm (out_norm) was not found");
    }
    if (!model.out_proj_.data) {
        // Some models tie lm_head to embed_tokens.
        IMP_LOG_WARN("WeightMap: output projection (out_proj) was not found "
                     "(may be tied to embed_tokens)");
    }

    return true;
}

} // namespace imp
