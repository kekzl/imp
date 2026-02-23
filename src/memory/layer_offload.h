#pragma once

#include "model/model.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

// Manages double-buffered GPU staging for layer weight offloading.
// Keeps N layers on GPU (highest priority: attention > SSM > MoE routing),
// offloads the rest to host memory. Async prefetch hides H2D transfer latency.
//
// Usage:
//   offload_mgr->ensure_layer(i, compute_stream);     // wait if needed
//   offload_mgr->prefetch_layer(i + 1);                // start async H2D
//   // run layer i...
class LayerOffloadManager {
public:
    LayerOffloadManager() = default;
    ~LayerOffloadManager();

    // Initialize offloading for the given model.
    // gpu_layers: number of layers to keep on GPU (0 = all offloaded,
    //             -1 = all on GPU i.e. disabled).
    // Priority: attention layers first, then SSM, then MoE-only.
    bool init(Model* model, int gpu_layers);

    // Ensure layer's weights are accessible on GPU. If the layer is resident
    // (not offloaded), this is a no-op. Otherwise, waits for the prefetch
    // to complete and remaps the layer's tensor pointers to the staging slot.
    void ensure_layer(int layer, cudaStream_t compute_stream);

    // Start async prefetch of next_layer into the inactive slot.
    // No-op if next_layer is resident or already loaded in a slot.
    void prefetch_layer(int next_layer);

    // Restore layer tensor pointers to their original host pointers.
    // Called after processing a layer to avoid dangling GPU pointers.
    void release_layer(int layer);

    bool is_offloaded(int layer) const {
        return layer >= 0 && layer < static_cast<int>(offloaded_.size()) && offloaded_[layer];
    }

    bool is_enabled() const { return enabled_; }

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
        cudaEvent_t ready_event = nullptr;
        cudaStream_t transfer_stream = nullptr;
    };

    Model* model_ = nullptr;
    bool enabled_ = false;

    Slot slots_[2];
    int active_slot_ = 0;  // slot that ensure_layer will use

    std::vector<bool> offloaded_;                         // [n_layers]
    std::vector<std::vector<WeightEntry>> layer_entries_; // [n_layers]
    std::vector<size_t> layer_slot_bytes_;                // [n_layers] total bytes per layer

    // Scan a layer's tensors and build WeightEntry list.
    void scan_layer_weights(int layer);

    // Copy all weight data for a layer from host to GPU slot.
    void upload_layer_to_slot(int layer, int slot_idx);
};

} // namespace imp
