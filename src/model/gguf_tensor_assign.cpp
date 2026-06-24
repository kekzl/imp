// ============================================================================
// GGUF tensor → model weight assignment. Maps each GGUF tensor name to the
// correct Model/Layer slot. Split out of gguf_loader.cpp to bound recompile
// blast radius (see tools/check_filesize.py). Top-level load orchestration
// stays in gguf_loader.cpp.
// ============================================================================

#include "model/gguf_loader.h"
#include "model/gguf_loader_internal.h"
#include "model/loader_assign.h"

#include <string>
#include <vector>

namespace imp {

// ---- Split string by delimiter ----

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == delim) {
            parts.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return parts;
}

// ---- Assign a single tensor to the model by GGUF name ----

bool assign_tensor(Model& model, const std::string& name, const Tensor& tensor, GgufWireType gtype) {
    auto qtype = static_cast<QType>(static_cast<uint32_t>(gtype));
    if (name == "token_embd.weight") {
        assign_quant(model.tok_emb_, tensor);
        return true;
    }
    if (name == "output_norm.weight") {
        assign_quant(model.out_norm_, tensor);
        return true;
    }
    if (name == "rope_freqs.weight") {
        model.layers_[0].rope_freqs = tensor;
        return true;
    }
    if (name == "output.weight") {
        assign_quant(model.out_proj_, tensor);
        return true;
    }

    // Layer weights: "blk.{i}.{field}" or "blk.{i}.{field}.{expert}.weight"
    if (name.substr(0, 4) != "blk.")
        return false;

    auto parts = split(name, '.');
    // Minimum: ["blk", "0", "ssm_a"] = 3 parts (some SSM tensors have no suffix)
    if (parts.size() < 3)
        return false;

    int layer_idx = 0;
    try {
        layer_idx = std::stoi(parts[1]);
    } catch (...) {
        return false;
    }

    if (layer_idx < 0 || layer_idx >= model.n_layers())
        return false;
    auto& layer = model.layers_[layer_idx];

    // 3-part: "blk.{i}.{name}" — SSM scalar/vector tensors without .weight/.bias suffix
    if (parts.size() == 3) {
        const auto& field = parts[2];
        if (field == "ssm_a")
            layer.ssm_a = tensor;
        else if (field == "ssm_d")
            layer.ssm_d = tensor;
        else
            return false;
        return true;
    }

    // 4-part: "blk.{i}.{name}.weight" or "blk.{i}.{name}.bias"
    if (parts.size() == 4) {
        const auto& field = parts[2];
        const auto& suffix = parts[3];  // "weight" or "bias"

        // Attention projections: distinguish weight vs bias
        if (field == "attn_q") {
            if (suffix == "bias")
                layer.q_bias = tensor;
            else
                assign_quant(layer.wq, tensor);
        } else if (field == "attn_k") {
            if (suffix == "bias")
                layer.k_bias = tensor;
            else
                assign_quant(layer.wk, tensor);
        } else if (field == "attn_v") {
            if (suffix == "bias")
                layer.v_bias = tensor;
            else
                assign_quant(layer.wv, tensor);
        } else if (field == "attn_output") {
            // gpt-oss adds an output-projection bias.
            if (suffix == "bias")
                layer.o_bias = tensor;
            else
                assign_quant(layer.wo, tensor);
        }
        // gpt-oss learned attention sinks: per-head sink logits [n_heads].
        else if (field == "attn_sinks")
            layer.attn_sinks = tensor;
        else if (field == "attn_norm")
            layer.attn_norm = tensor;
        else if (field == "attn_q_norm")
            layer.attn_q_norm = tensor;
        else if (field == "attn_k_norm")
            layer.attn_k_norm = tensor;
        // Fused QKV: either standard attention (Phi-4) or GDN (Qwen3.5)
        else if (field == "attn_qkv") {
            const auto& cfg = model.config();
            int64_t total_rows = tensor.shape[0];  // outermost dim after reversal
            int64_t d_model = tensor.shape[1];     // inner dim

            // Check if this is a GDN layer (total rows match SSM conv_channels)
            int ssm_conv_channels = cfg.ssm_inner_size + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
            if (cfg.ssm_inner_size > 0 && total_rows == ssm_conv_channels) {
                // GDN layer: treat attn_qkv as ssm_in (fused projection → conv1d input)
                assign_quant(layer.ssm_in, tensor);
            } else {
                // Standard fused QKV: split into separate Q, K, V
                // For Qwen3.5 attention: Q has 2× output (Q + gate interleaved),
                // so q_rows = total_rows - k_rows - v_rows (not just n_heads * head_dim).
                int q_rows = cfg.n_heads * cfg.head_dim;
                int k_rows = cfg.n_kv_heads * cfg.head_dim;
                size_t row_bytes = qtype_row_bytes(qtype, d_model);

                uint8_t* base = static_cast<uint8_t*>(tensor.data);
                int64_t q_shape[4] = {q_rows, d_model, 1, 1};
                int64_t kv_shape[4] = {k_rows, d_model, 1, 1};

                Tensor q_t(base, tensor.qtype, 2, q_shape, tensor.on_device);
                Tensor k_t(base + static_cast<size_t>(q_rows) * row_bytes, tensor.qtype, 2, kv_shape,
                           tensor.on_device);
                Tensor v_t(base + static_cast<size_t>(q_rows + k_rows) * row_bytes, tensor.qtype, 2, kv_shape,
                           tensor.on_device);
                assign_quant(layer.wq, q_t);
                assign_quant(layer.wk, k_t);
                assign_quant(layer.wv, v_t);
            }
        }
        // Post-layer norms (Gemma-3)
        else if (field == "post_attention_norm")
            layer.post_attn_norm = tensor;
        else if (field == "post_ffw_norm")
            layer.post_ffn_norm = tensor;
        // Gemma 4: parallel shared MLP + MoE expert branch norms
        else if (field == "pre_ffw_norm_2")
            layer.ffn_pre_norm_2 = tensor;
        else if (field == "post_ffw_norm_1")
            layer.ffn_post_norm_1 = tensor;
        else if (field == "post_ffw_norm_2")
            layer.ffn_post_norm_2 = tensor;
        else if (field == "layer_output_scale")
            layer.layer_out_scale = tensor;
        else if (field == "rope_freqs")
            layer.rope_freqs = tensor;
        // Gemma 4: fused gate+up experts: [n_experts, n_ff_exp*2, d_model]
        // We keep it packed; the MoE executor handles de-interleaving at dispatch.
        else if (field == "ffn_gate_up_exps") {
            // Reuses gate packed slot with full fused tensor.
            // Mark fused by leaving expert_up_packed null — executor detects this.
            assign_quant(layer.expert_gate_packed, tensor);
        }
        // FFN
        else if (field == "ffn_gate")
            assign_quant(layer.w_gate, tensor);
        else if (field == "ffn_up")
            assign_quant(layer.w_up, tensor);
        else if (field == "ffn_down")
            assign_quant(layer.w_down, tensor);
        else if (field == "ffn_norm")
            layer.ffn_norm = tensor;
        else if (field == "ffn_gate_inp") {
            // Distinguish .weight (the gate matrix) from .scale (per-channel multiplier).
            // Gemma 4 stores `blk.X.ffn_gate_inp.scale` as a 4-part tensor name; without
            // this branch the scale would be silently misassigned to layer.moe_gate.
            if (suffix == "scale")
                layer.ffn_gate_inp_scale = tensor;
            else if (suffix == "bias")
                layer.router_bias = tensor;  // gpt-oss router logits bias [n_experts]
            else
                layer.moe_gate = tensor;
        }
        // Packed expert tensors: 3D [n_experts, rows, cols]. gpt-oss adds a
        // per-expert .bias on each projection — without the suffix split the
        // bias tensor clobbers the packed weight (load order dependent).
        else if (field == "ffn_gate_exps") {
            if (suffix == "bias")
                layer.expert_gate_bias = tensor;
            else
                assign_quant(layer.expert_gate_packed, tensor);
        } else if (field == "ffn_up_exps") {
            if (suffix == "bias")
                layer.expert_up_bias = tensor;
            else
                assign_quant(layer.expert_up_packed, tensor);
        } else if (field == "ffn_down_exps") {
            // Distinguish .weight (the per-expert FFN down weights) from .scale
            // (per-expert output multiplier, shape [n_expert]) and gpt-oss's
            // per-expert .bias. Same 4-part-name bug as ffn_gate_inp.scale: a
            // non-weight tensor would otherwise overwrite expert_down_packed.
            if (suffix == "scale")
                layer.expert_down_scale = tensor;
            else if (suffix == "bias")
                layer.expert_down_bias = tensor;
            else
                assign_quant(layer.expert_down_packed, tensor);
        }
        // Shared expert (always-active, e.g. Nemotron/DeepSeek)
        else if (field == "ffn_gate_shexp")
            assign_quant(layer.w_gate_shared, tensor);
        else if (field == "ffn_up_shexp")
            assign_quant(layer.w_up_shared, tensor);
        else if (field == "ffn_down_shexp")
            assign_quant(layer.w_down_shared, tensor);
        // Qwen3-Next / Qwen3.6 per-token sigmoid gate on the shared expert
        // output. 1D [d_model] FP32 projection; sigmoid(cur @ W) yields [M, 1].
        else if (field == "ffn_gate_inp_shexp")
            layer.shared_expert_gate_inp = tensor;
        // SSM weights (Mamba2)
        else if (field == "ssm_in")
            assign_quant(layer.ssm_in, tensor);
        else if (field == "ssm_out")
            assign_quant(layer.ssm_out, tensor);
        else if (field == "ssm_dt") {
            // Some converters (Qwen3.5-27B-mxfp4) emit A_log under the name
            // "ssm_dt.weight" — a 1D vector of shape [n_heads]. Differentiate
            // bias vs weight: bias → ssm_dt_b (per-head dt bias),
            // weight → ssm_a (per-head A_log). Without this branch the weight
            // silently overwrites the bias and ssm_a stays null, causing the
            // GDN scan kernel to NULL-deref A_log[h] on first launch.
            if (suffix == "bias")
                layer.ssm_dt_b = tensor;
            else if (suffix == "weight")
                layer.ssm_a = tensor;
            else
                return false;
        } else if (field == "ssm_norm")
            layer.ssm_norm_w = tensor;
        // SSM conv1d: "blk.{i}.ssm_conv1d.weight" / "blk.{i}.ssm_conv1d.bias"
        else if (field == "ssm_conv1d") {
            if (suffix == "weight")
                layer.ssm_conv1d_w = tensor;
            else if (suffix == "bias")
                layer.ssm_conv1d_b = tensor;
            else
                return false;
        }
        // Gated DeltaNet (GDN) weights (Qwen3.5)
        else if (field == "attn_gate")
            assign_quant(layer.gdn_gate, tensor);
        else if (field == "ssm_alpha")
            assign_quant(layer.gdn_alpha, tensor);
        else if (field == "ssm_beta")
            assign_quant(layer.gdn_beta, tensor);
        // Router bias (Nemotron MoE)
        else if (field == "exp_probs_b")
            layer.moe_router_bias = tensor;
        else
            return false;
        return true;
    }

    // 5-part: "blk.{i}.ffn_*.{expert_idx}.weight" — MoE per-expert weights
    //    or: "blk.{i}.ffn_gate_inp.scale.weight" / "blk.{i}.ffn_down_exps.scale.weight" (Gemma 4)
    if (parts.size() == 5) {
        const auto& field = parts[2];
        const auto& subfield = parts[3];

        // Gemma 4 scale tensors
        if (subfield == "scale") {
            if (field == "ffn_gate_inp") {
                layer.ffn_gate_inp_scale = tensor;
                return true;
            }
            if (field == "ffn_down_exps") {
                // Per-expert output scale. Not yet consumed by the executor — store as
                // router bias slot so weight_upload at least preserves it.
                layer.moe_router_bias = tensor;
                return true;
            }
            return false;
        }

        // MoE expert weights: "blk.{i}.ffn_*.{expert_idx}.weight"
        int expert_idx = 0;
        try {
            expert_idx = std::stoi(parts[3]);
        } catch (...) {
            return false;
        }

        int n_experts = model.config().n_experts;
        if (expert_idx < 0 || expert_idx >= n_experts)
            return false;

        // Per-expert vectors: assign to slot N. The layer-wide qtype mirror is
        // populated only on the first expert (all experts share the same qtype).
        if (field == "ffn_gate") {
            layer.expert_w_gate[expert_idx] = tensor;
            if (expert_idx == 0)
                layer.expert_gate_packed.qtype = qtype;
        } else if (field == "ffn_up") {
            layer.expert_w_up[expert_idx] = tensor;
            if (expert_idx == 0)
                layer.expert_up_packed.qtype = qtype;
        } else if (field == "ffn_down") {
            layer.expert_w_down[expert_idx] = tensor;
            if (expert_idx == 0)
                layer.expert_down_packed.qtype = qtype;
        } else
            return false;
        return true;
    }

    return false;
}

}  // namespace imp
