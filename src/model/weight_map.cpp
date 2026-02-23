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

        // DeepSeek MoE router: mlp.gate.weight
        if (parts.size() >= 6 && parts[3] == "mlp" &&
            parts[4] == "gate" && parts[5] == "weight") {
            return prefix + "moe_gate";
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

    for (auto& [name, tensor] : tensors) {
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

        // -----------------------------------------------------------------
        // Layer weights: model.layers.{i}.<rest>
        // -----------------------------------------------------------------
        if (parts.size() < 5 || parts[0] != "model" || parts[1] != "layers") {
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
        if (!matched && parts.size() >= 5 &&
            parts[3] == "post_attention_layernorm" && parts[4] == "weight") {
            layer.ffn_norm = tensor;
            matched = true;
        }

        // -- Dense MLP (Llama / Mistral / DeepSeek dense layers) --
        if (!matched && parts.size() >= 6 && parts[3] == "mlp" && parts[5] == "weight") {
            const std::string& proj = parts[4];
            if (proj == "gate_proj") { layer.w_gate = tensor; matched = true; }
            else if (proj == "up_proj") { layer.w_up = tensor; matched = true; }
            else if (proj == "down_proj") { layer.w_down = tensor; matched = true; }
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
        //   mlp.experts.{e}.gate_proj.weight             -> expert_w_gate[e]
        //   mlp.experts.{e}.up_proj.weight               -> expert_w_up[e]
        //   mlp.experts.{e}.down_proj.weight             -> expert_w_down[e]
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
