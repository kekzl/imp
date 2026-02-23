#include "graph/graph.h"
#include <sstream>
#include <cmath>

namespace imp {

int Graph::add_node(OpType type, const std::vector<int>& inputs, int layer) {
    OpNode node;
    node.id = static_cast<int>(nodes_.size());
    node.type = type;
    node.inputs = inputs;
    node.layer = layer;

    // Wire output edges: for each input node, register this node as an output
    for (int inp : inputs) {
        if (inp >= 0 && inp < static_cast<int>(nodes_.size())) {
            nodes_[inp].outputs.push_back(node.id);
        }
    }

    nodes_.push_back(node);
    return node.id;
}

int Graph::build_transformer_layer(int layer_idx, int input_node,
                                   const ModelConfig& config, bool is_prefill) {
    int head_dim = config.d_model / config.n_heads;
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // --- Attention block ---
    // input -> RMSNORM -> QKV_PROJ -> ROPE -> ATTENTION -> O_PROJ -> RESIDUAL_ADD

    int attn_norm = add_node(OpType::RMSNORM, {input_node}, layer_idx);
    {
        auto& n = nodes_[attn_norm];
        n.params.rmsnorm.eps = config.rms_norm_eps;
    }

    int qkv = add_node(OpType::QKV_PROJ, {attn_norm}, layer_idx);
    {
        auto& n = nodes_[qkv];
        n.params.gemm.output_dim = config.d_model + 2 * (config.n_kv_heads * head_dim);
    }

    int rope = add_node(OpType::ROPE, {qkv}, layer_idx);
    {
        auto& n = nodes_[rope];
        n.params.rope.head_dim = head_dim;
        n.params.rope.theta = config.rope_theta;
    }

    int attn;
    if (is_prefill) {
        attn = add_node(OpType::ATTENTION_PREFILL, {rope}, layer_idx);
        auto& n = nodes_[attn];
        n.params.attention.scale = attn_scale;
        n.params.attention.causal = true;
        n.params.attention.n_heads = config.n_heads;
        n.params.attention.n_kv_heads = config.n_kv_heads;
        n.params.attention.head_dim = head_dim;
    } else {
        attn = add_node(OpType::ATTENTION_DECODE, {rope}, layer_idx);
        auto& n = nodes_[attn];
        n.params.paged_attention.scale = attn_scale;
        n.params.paged_attention.n_heads = config.n_heads;
        n.params.paged_attention.n_kv_heads = config.n_kv_heads;
        n.params.paged_attention.head_dim = head_dim;
    }

    int oproj = add_node(OpType::O_PROJ, {attn}, layer_idx);
    {
        auto& n = nodes_[oproj];
        n.params.gemm.output_dim = config.d_model;
    }

    // Residual add: input + o_proj output
    int attn_res = add_node(OpType::RESIDUAL_ADD, {input_node, oproj}, layer_idx);

    // --- FFN block ---
    // residual -> RMSNORM -> GATE_PROJ + UP_PROJ -> SWIGLU -> DOWN_PROJ -> RESIDUAL_ADD

    int ffn_norm = add_node(OpType::RMSNORM, {attn_res}, layer_idx);
    {
        auto& n = nodes_[ffn_norm];
        n.params.rmsnorm.eps = config.rms_norm_eps;
    }

    bool is_moe = (config.n_experts > 0 && config.n_experts_active > 0);

    if (is_moe) {
        int gate = add_node(OpType::MOE_GATE, {ffn_norm}, layer_idx);
        int gather = add_node(OpType::MOE_GATHER, {gate}, layer_idx);
        int gemm1 = add_node(OpType::GROUPED_GEMM, {gather}, layer_idx);
        int swiglu = add_node(OpType::SWIGLU, {gemm1}, layer_idx);
        int gemm2 = add_node(OpType::GROUPED_GEMM, {swiglu}, layer_idx);
        int scatter = add_node(OpType::MOE_SCATTER, {gemm2}, layer_idx);
        int ffn_res = add_node(OpType::RESIDUAL_ADD, {attn_res, scatter}, layer_idx);
        return ffn_res;
    } else {
        int gate_proj = add_node(OpType::GATE_PROJ, {ffn_norm}, layer_idx);
        {
            auto& n = nodes_[gate_proj];
            n.params.gemm.output_dim = config.d_ff;
        }

        int up_proj = add_node(OpType::UP_PROJ, {ffn_norm}, layer_idx);
        {
            auto& n = nodes_[up_proj];
            n.params.gemm.output_dim = config.d_ff;
        }

        int swiglu = add_node(OpType::SWIGLU, {gate_proj, up_proj}, layer_idx);

        int down_proj = add_node(OpType::DOWN_PROJ, {swiglu}, layer_idx);
        {
            auto& n = nodes_[down_proj];
            n.params.gemm.output_dim = config.d_model;
        }

        int ffn_res = add_node(OpType::RESIDUAL_ADD, {attn_res, down_proj}, layer_idx);
        return ffn_res;
    }
}

Graph Graph::build_transformer(const ModelConfig& config, bool is_prefill) {
    Graph g;

    // Embedding
    int emb = g.add_node(OpType::EMBEDDING, {});

    // Transformer layers
    int prev = emb;
    for (int i = 0; i < config.n_layers; ++i) {
        prev = g.build_transformer_layer(i, prev, config, is_prefill);
    }

    // Output norm
    int out_norm = g.add_node(OpType::OUTPUT_NORM, {prev});
    {
        auto& n = g.nodes_[out_norm];
        n.params.rmsnorm.eps = config.rms_norm_eps;
    }

    // LM head
    int lm_head = g.add_node(OpType::LM_HEAD, {out_norm});
    {
        auto& n = g.nodes_[lm_head];
        n.params.gemm.output_dim = config.vocab_size;
    }

    // Sampling
    g.add_node(OpType::SAMPLING, {lm_head});

    return g;
}

// Legacy single-layer builder (kept for backward compatibility)
Graph Graph::build_transformer_layer(bool is_moe) {
    Graph g;

    // Stub: create a minimal linear chain
    int norm = g.add_node(OpType::RMSNORM, {});
    int qkv = g.add_node(OpType::QKV_PROJ, {norm});
    int rope = g.add_node(OpType::ROPE, {qkv});
    int attn = g.add_node(OpType::ATTENTION_PREFILL, {rope});
    int oproj = g.add_node(OpType::O_PROJ, {attn});
    int res1 = g.add_node(OpType::RESIDUAL_ADD, {oproj});

    int ffn_norm = g.add_node(OpType::RMSNORM, {res1});

    if (is_moe) {
        int gate = g.add_node(OpType::MOE_GATE, {ffn_norm});
        int gather = g.add_node(OpType::MOE_GATHER, {gate});
        int gemm = g.add_node(OpType::GROUPED_GEMM, {gather});
        int swiglu = g.add_node(OpType::SWIGLU, {gemm});
        int gemm2 = g.add_node(OpType::GROUPED_GEMM, {swiglu});
        int scatter = g.add_node(OpType::MOE_SCATTER, {gemm2});
        g.add_node(OpType::RESIDUAL_ADD, {scatter});
    } else {
        int gate_proj = g.add_node(OpType::GATE_PROJ, {ffn_norm});
        int up_proj = g.add_node(OpType::UP_PROJ, {ffn_norm});
        int swiglu = g.add_node(OpType::SWIGLU, {gate_proj, up_proj});
        int down_proj = g.add_node(OpType::DOWN_PROJ, {swiglu});
        g.add_node(OpType::RESIDUAL_ADD, {down_proj});
    }

    return g;
}

std::string Graph::to_dot() const {
    std::ostringstream ss;
    ss << "digraph TransformerGraph {\n";
    ss << "  rankdir=TB;\n";
    ss << "  node [shape=box, style=filled, fillcolor=lightblue];\n\n";

    for (const auto& node : nodes_) {
        ss << "  n" << node.id << " [label=\"" << op_type_name(node.type);
        if (node.layer >= 0) {
            ss << "\\nL" << node.layer;
        }
        ss << "\"];\n";
    }

    ss << "\n";

    for (const auto& node : nodes_) {
        for (int inp : node.inputs) {
            ss << "  n" << inp << " -> n" << node.id << ";\n";
        }
    }

    ss << "}\n";
    return ss.str();
}

} // namespace imp
