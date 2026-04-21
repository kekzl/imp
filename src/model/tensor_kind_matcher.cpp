#include "model/tensor_kind_matcher.h"

namespace imp {

namespace {

bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.substr(s.size() - suffix.size()) == suffix;
}

bool contains(std::string_view s, std::string_view needle) {
    return s.find(needle) != std::string_view::npos;
}

} // namespace

TensorKind match_tensor_kind(std::string_view name) {
    // Top-level embeddings / head / final norm
    if (name == "token_embd.weight" || name == "tok_embeddings.weight" ||
        name == "model.embed_tokens.weight")
        return TensorKind::TOK_EMBED;
    if (name == "output.weight" || name == "lm_head.weight")
        return TensorKind::LM_HEAD;
    if (name == "output_norm.weight" || name == "norm.weight" ||
        name == "model.norm.weight")
        return TensorKind::FFN_NORM;

    // Per-layer tensors: "blk.N." prefix, "layers.N." prefix, or "model.layers.N." prefix
    const bool is_layer = (name.substr(0, 4) == "blk." ||
                           name.substr(0, 7) == "layers." ||
                           name.substr(0, 13) == "model.layers.");
    if (!is_layer) return TensorKind::UNKNOWN;

    // SafeTensors (HuggingFace) naming convention
    // model.layers.N.self_attn.{q,k,v,o}_proj.weight
    if (contains(name, ".self_attn.")) {
        if (contains(name, ".q_proj."))  return TensorKind::WQ;
        if (contains(name, ".k_proj."))  return TensorKind::WK;
        if (contains(name, ".v_proj."))  return TensorKind::WV;
        if (contains(name, ".o_proj."))  return TensorKind::WO;
        if (contains(name, ".q_norm."))  return TensorKind::QK_NORM_Q;
        if (contains(name, ".k_norm."))  return TensorKind::QK_NORM_K;
    }
    // model.layers.N.input_layernorm.weight / post_attention_layernorm.weight
    if (contains(name, ".input_layernorm."))         return TensorKind::ATTN_NORM;
    if (contains(name, ".post_attention_layernorm.")) return TensorKind::POST_ATTN_NORM;
    if (contains(name, ".pre_feedforward_layernorm."))  return TensorKind::FFN_NORM;
    if (contains(name, ".post_feedforward_layernorm.")) return TensorKind::POST_FFN_NORM;
    // model.layers.N.mlp.{gate,up,down}_proj.weight
    if (contains(name, ".mlp.")) {
        if (contains(name, ".gate_proj.")) return TensorKind::W_GATE;
        if (contains(name, ".up_proj."))   return TensorKind::W_UP;
        if (contains(name, ".down_proj.")) return TensorKind::W_DOWN;
    }
    // GDN (Qwen3.5 SafeTensors): temporal_block.gate_proj / alpha / beta
    if (contains(name, ".temporal_block.")) {
        if (contains(name, ".gate_proj.")) return TensorKind::GDN_GATE;
        if (contains(name, ".alpha."))     return TensorKind::ALPHA;
        if (contains(name, ".beta."))      return TensorKind::BETA;
    }
    // Mamba2 (Nemotron-H SafeTensors): mamba.{in_proj,out_proj,conv1d,...}
    if (contains(name, ".mamba.")) {
        if (contains(name, ".in_proj."))  return TensorKind::SSM_IN;
        if (contains(name, ".out_proj.")) return TensorKind::SSM_OUT;
        if (contains(name, ".conv1d.weight")) return TensorKind::CONV1D_W;
        if (contains(name, ".conv1d.bias"))   return TensorKind::CONV1D_B;
        if (contains(name, ".dt_bias"))   return TensorKind::DT_BIAS;
        if (contains(name, ".A_log"))     return TensorKind::A_LOG;
        if (contains(name, ".norm."))     return TensorKind::SSM_GROUP_NORM;
    }

    // Attention projections
    if (contains(name, ".attn_q.") || contains(name, ".wq."))   return TensorKind::WQ;
    if (contains(name, ".attn_k.") || contains(name, ".wk."))   return TensorKind::WK;
    if (contains(name, ".attn_v.") || contains(name, ".wv."))   return TensorKind::WV;
    if (contains(name, ".attn_output.") || contains(name, ".wo.")) return TensorKind::WO;
    if (contains(name, ".attn_qkv."))                           return TensorKind::QKV_FUSED;

    // FFN / MoE — ORDER MATTERS: ffn_gate_inp_shexp must come before ffn_gate_inp,
    // and ffn_gate_inp before ffn_gate_exps before ffn_gate.
    if (contains(name, ".ffn_gate_inp_shexp.")) return TensorKind::SHARED_EXPERT_GATE;
    if (contains(name, ".ffn_gate_inp."))       return TensorKind::ROUTER;
    if (contains(name, ".ffn_gate_exps."))      return TensorKind::EXPERT_GATE;
    if (contains(name, ".ffn_up_exps."))        return TensorKind::EXPERT_UP;
    if (contains(name, ".ffn_down_exps."))      return TensorKind::EXPERT_DOWN;
    if (contains(name, ".ffn_gate."))           return TensorKind::W_GATE;
    if (contains(name, ".ffn_up."))             return TensorKind::W_UP;
    if (contains(name, ".ffn_down."))           return TensorKind::W_DOWN;

    // GDN / Mamba2 — GDN's gdn_gate / attn_gate first to avoid any prefix collision
    if (contains(name, ".gdn_gate.") || contains(name, ".gdn_output_gate.") ||
        contains(name, ".attn_gate."))
        return TensorKind::GDN_GATE;
    if (contains(name, ".ssm_in."))     return TensorKind::SSM_IN;
    if (contains(name, ".ssm_out."))    return TensorKind::SSM_OUT;
    if (contains(name, ".ssm_conv1d."))
        return ends_with(name, ".bias") ? TensorKind::CONV1D_B : TensorKind::CONV1D_W;
    if (contains(name, ".ssm_a"))       return TensorKind::A_LOG;
    if (contains(name, ".ssm_dt_b"))    return TensorKind::DT_BIAS;
    if (contains(name, ".ssm_beta"))    return TensorKind::BETA;
    if (contains(name, ".ssm_alpha"))   return TensorKind::ALPHA;
    if (contains(name, ".ssm_norm."))   return TensorKind::SSM_GROUP_NORM;

    // Norms — ORDER MATTERS: attn_q_norm before attn_norm, post_attn_norm before attn_norm.
    if (contains(name, ".attn_q_norm."))    return TensorKind::QK_NORM_Q;
    if (contains(name, ".attn_k_norm."))    return TensorKind::QK_NORM_K;
    if (contains(name, ".post_attn_norm.")) return TensorKind::POST_ATTN_NORM;
    if (contains(name, ".post_ffn_norm."))  return TensorKind::POST_FFN_NORM;
    if (contains(name, ".attn_norm."))      return TensorKind::ATTN_NORM;
    if (contains(name, ".ffn_norm."))       return TensorKind::FFN_NORM;

    // RoPE
    if (contains(name, ".rope_freqs")) return TensorKind::ROPE_FREQS;

    return TensorKind::UNKNOWN;
}

} // namespace imp
