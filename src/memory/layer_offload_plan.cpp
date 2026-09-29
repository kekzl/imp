#include "memory/layer_offload_plan.h"
#include <algorithm>

namespace imp {

size_t tensor_storage_bytes(const Tensor& t) { return t.nbytes(); }

LayerOffloadPlan plan_layer_offload(const Model& model, int gpu_layers) {
    LayerOffloadPlan plan;
    const int n_layers = model.n_layers();
    if (gpu_layers < 0 || gpu_layers >= n_layers)
        return plan;
    plan.active = true;

    struct LayerPriority {
        int layer_idx;
        int priority;  // 0 = attention, 1 = SSM, 2 = MoE-only / other
    };
    std::vector<LayerPriority> priorities(n_layers);
    for (int i = 0; i < n_layers; i++) {
        const auto& ly = model.layer(i);
        int prio = 2;
        if (ly.wq.data != nullptr)
            prio = 0;
        else if (ly.ssm_in.data != nullptr)
            prio = 1;
        priorities[i] = {i, prio};
    }
    std::stable_sort(priorities.begin(), priorities.end(),
                     [](const auto& a, const auto& b) { return a.priority < b.priority; });

    plan.offloaded.assign(n_layers, true);
    for (int i = 0; i < gpu_layers; i++)
        plan.offloaded[priorities[i].layer_idx] = false;

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    plan.layer_bytes.assign(n_layers, 0);
    plan.host_bytes.assign(n_layers, 0);
    for (int i = 0; i < n_layers; i++) {
        size_t all = 0, host = 0;
        for_each_offloadable_tensor(model.layer(i), [&](const Tensor& t) {
            if (!t.data || t.on_device)
                return;
            const size_t nbytes = tensor_storage_bytes(t);
            if (nbytes == 0)
                return;
            all = align256(all + nbytes);
            host = align256(host + nbytes);
        });
        plan.layer_bytes[i] = all;
        plan.host_bytes[i] = host;
        if (!plan.offloaded[i])
            continue;
        plan.n_offloaded++;
        plan.planned_bytes += all;
        plan.max_host_bytes = std::max(plan.max_host_bytes, host);
    }
    return plan;
}

}  // namespace imp
