#pragma once

#include "graph/op.h"
#include "model/model.h"
#include <vector>
#include <string>

namespace imp {

class Graph {
public:
    Graph() = default;

    // Add a node and return its index. Automatically wires output edges.
    int add_node(OpType type, const std::vector<int>& inputs, int layer = -1);

    const OpNode& node(int id) const { return nodes_[id]; }
    OpNode& node(int id) { return nodes_[id]; }
    int num_nodes() const { return static_cast<int>(nodes_.size()); }

    // Build the full transformer graph (all layers + embedding + LM head).
    // This is used for visualization and debugging, NOT for execution dispatch.
    static Graph build_transformer(const ModelConfig& config, bool is_prefill = true);

    // Build a single transformer layer subgraph (returns node IDs for input/output).
    // input_node: the node ID whose output feeds into this layer.
    // Returns the node ID of the final RESIDUAL_ADD in the FFN block.
    int build_transformer_layer(int layer_idx, int input_node,
                                const ModelConfig& config, bool is_prefill);

    // Legacy single-layer builder for testing
    static Graph build_transformer_layer(bool is_moe = false);

    // Dump graph as DOT format for Graphviz visualization
    std::string to_dot() const;

private:
    std::vector<OpNode> nodes_;
};

} // namespace imp
