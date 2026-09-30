#pragma once

#include "core/cuda_raii.h"
#include "model/model.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace imp {

// Per-layer weights LayerOffloadManager stages through its slots. Plan and scan share this list.
template <class Layer, class Fn>
void for_each_offloadable_tensor(Layer& ly, Fn&& fn) {
    for (auto* t : {&ly.wq,
                    &ly.wk,
                    &ly.wv,
                    &ly.wo,
                    &ly.attn_norm,
                    &ly.attn_q_norm,
                    &ly.attn_k_norm,
                    &ly.w_gate,
                    &ly.w_up,
                    &ly.w_down,
                    &ly.ffn_norm,
                    &ly.moe_gate,
                    &ly.w_up_shared,
                    &ly.w_down_shared,
                    &ly.w_gate_shared,
                    &ly.moe_router_bias,
                    &ly.expert_gate_packed,
                    &ly.expert_up_packed,
                    &ly.expert_down_packed,
                    &ly.ssm_in,
                    &ly.ssm_out,
                    &ly.ssm_conv1d_w,
                    &ly.ssm_conv1d_b,
                    &ly.ssm_dt_b,
                    &ly.ssm_a,
                    &ly.ssm_d,
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

// Manages double-buffered GPU staging for layer weight offloading. Keeps N layers on GPU
// (priority: attention > SSM > MoE routing), offloads the rest to host memory. Async
// prefetch hides H2D transfer latency.
class LayerOffloadManager {
public:
    LayerOffloadManager() = default;
    ~LayerOffloadManager();

    // Owns raw per-slot cudaStream_t/cudaEvent_t destroyed in the dtor; a copy
    // would alias them and double-free. Non-copyable.
    LayerOffloadManager(const LayerOffloadManager&) = delete;
    LayerOffloadManager& operator=(const LayerOffloadManager&) = delete;

    // Initialize offloading for the given model. gpu_layers: number of layers to keep on GPU
    // (0 = all offloaded, -1 = all on GPU, disabled). Priority: attention layers first, then
    // SSM, then MoE-only. Returns false when planned layers are device-resident (#2291).
    [[nodiscard]] bool init(Model* model, int gpu_layers);

    // Ensure layer's weights are accessible on GPU. If the layer is resident
    // (not offloaded), this is a no-op. Otherwise, waits for the prefetch
    // to complete and remaps the layer's tensor pointers to the staging slot.
    void ensure_layer(int layer, cudaStream_t compute_stream);

    // Start async prefetch of next_layer into the inactive slot.
    // No-op if next_layer is resident or already loaded in a slot.
    void prefetch_layer(int next_layer);

    // Restore layer tensor pointers to their original host pointers. Records the slot's free
    // event on compute_stream: the next upload into that slot waits for the layer's kernels.
    void release_layer(int layer, cudaStream_t compute_stream);

    bool is_enabled() const { return enabled_; }
    // Layer streams from host (its tensor pointers change in ensure_layer / release_layer).
    bool is_offloaded(int layer) const {
        return enabled_ && layer >= 0 && layer < static_cast<int>(offloaded_.size()) && offloaded_[layer];
    }

private:
    // Per-weight metadata for an offloaded layer
    struct WeightEntry {
        Tensor* tensor;         // pointer into Model's TransformerLayer
        void* host_ptr;         // original host pointer (mmap'd or pinned)
        size_t raw_bytes;       // byte size of the weight
        size_t offset_in_slot;  // byte offset within the GPU staging slot
    };

    struct Slot {
        void* gpu_buf = nullptr;
        size_t buf_size = 0;
        int loaded_layer = -1;
        CudaEvent ready_event;
        CudaEvent free_event;  // compute stream done reading the slot's previous layer
        CudaStream transfer_stream;
    };

    Model* model_ = nullptr;
    bool enabled_ = false;

    Slot slots_[2];
    int active_slot_ = 0;  // slot that ensure_layer will use

    std::vector<bool> offloaded_;                          // [n_layers]
    std::vector<std::vector<WeightEntry>> layer_entries_;  // [n_layers]
    std::vector<size_t> layer_slot_bytes_;                 // [n_layers] total bytes per layer

    // Scan a layer's tensors and build WeightEntry list.
    void scan_layer_weights(int layer);

    // Copy all weight data for a layer from host to GPU slot.
    void upload_layer_to_slot(int layer, int slot_idx);
};

}  // namespace imp
