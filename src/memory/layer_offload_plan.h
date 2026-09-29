#pragma once

#include "model/model.h"
#include <cstddef>
#include <vector>

namespace imp {

// Per-layer weights LayerOffloadManager stages through its slots. Plan and scan share this list.
template <class Layer, class Fn>
void for_each_offloadable_tensor(Layer& ly, Fn&& fn) {
    for (auto* t : {&ly.wq, &ly.wk, &ly.wv, &ly.wo, &ly.attn_norm, &ly.attn_q_norm, &ly.attn_k_norm,
                    &ly.w_gate, &ly.w_up, &ly.w_down, &ly.ffn_norm, &ly.moe_gate, &ly.w_up_shared,
                    &ly.w_down_shared, &ly.w_gate_shared, &ly.moe_router_bias, &ly.expert_gate_packed,
                    &ly.expert_up_packed, &ly.expert_down_packed, &ly.ssm_in, &ly.ssm_out,
                    &ly.ssm_conv1d_w, &ly.ssm_conv1d_b, &ly.ssm_dt_b, &ly.ssm_a, &ly.ssm_d,
                    &ly.ssm_norm_w})
        fn(*t);
}

// Bytes a weight occupies in storage.
size_t tensor_storage_bytes(const Tensor& t);

// Which layers stream from host under --gpu-layers, and what each costs.
struct LayerOffloadPlan {
    bool active = false;              // 0 <= gpu_layers < n_layers
    std::vector<bool> offloaded;      // [n_layers] true = weights stream from host
    std::vector<size_t> layer_bytes;  // [n_layers] all offloadable weights, 256 B aligned
    std::vector<size_t> host_bytes;   // [n_layers] of which host-resident now (slot payload)
    int n_offloaded = 0;
    size_t planned_bytes = 0;   // sum of layer_bytes over offloaded layers
    size_t max_host_bytes = 0;  // staging slot size: max host_bytes over offloaded layers
};

// Keeps gpu_layers layers on GPU by priority attention > SSM > other, stable in layer order.
LayerOffloadPlan plan_layer_offload(const Model& model, int gpu_layers);

}  // namespace imp
