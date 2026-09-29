#include "memory/layer_offload.h"
#include "memory/layer_offload_plan.h"
#include "core/logging.h"
#include <algorithm>
#include <numeric>

namespace imp {

LayerOffloadManager::~LayerOffloadManager() {
    for (auto& slot : slots_) {
        if (slot.gpu_buf) {
            IMP_CUDA_CHECK_LOG(cudaFree(slot.gpu_buf));
            slot.gpu_buf = nullptr;
        }
        // Buffer freed before the slot event/stream (same order as before).
        slot.ready_event.reset();
        slot.transfer_stream.reset();
    }
}

void LayerOffloadManager::scan_layer_weights(int layer) {
    auto& entries = layer_entries_[layer];
    entries.clear();

    size_t offset = 0;
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    for_each_offloadable_tensor(model_->layer(layer), [&](Tensor& t) {
        if (!t.data || t.on_device)
            return;
        const size_t nbytes = tensor_storage_bytes(t);
        if (nbytes == 0)
            return;
        entries.push_back({&t, t.data, nbytes, offset});
        offset = align256(offset + nbytes);
    });
    layer_slot_bytes_[layer] = offset;
}

bool LayerOffloadManager::init(Model* model, int gpu_layers) {
    model_ = model;
    const int n_layers = model->n_layers();
    const LayerOffloadPlan plan = plan_layer_offload(*model, gpu_layers);
    if (!plan.active) {
        enabled_ = false;
        return true;  // all on GPU, nothing to offload
    }

    offloaded_ = plan.offloaded;
    layer_entries_.resize(n_layers);
    layer_slot_bytes_.resize(n_layers, 0);
    size_t max_layer_bytes = 0;
    int n_offloaded = 0;
    for (int i = 0; i < n_layers; i++) {
        if (!offloaded_[i])
            continue;
        scan_layer_weights(i);
        max_layer_bytes = std::max(max_layer_bytes, layer_slot_bytes_[i]);
        n_offloaded++;
    }

    if (n_offloaded == 0 || max_layer_bytes == 0) {
        enabled_ = false;
        IMP_LOG_INFO("Layer offloading: nothing to offload");
        return true;
    }

    // Allocate double-buffered GPU staging slots
    for (int s = 0; s < 2; s++) {
        cudaError_t err = cudaMalloc(&slots_[s].gpu_buf, max_layer_bytes);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate offload slot %d (%zu bytes): %s", s, max_layer_bytes,
                          cudaGetErrorString(err));
            return false;
        }
        slots_[s].buf_size = max_layer_bytes;
        slots_[s].loaded_layer = -1;

        err = slots_[s].ready_event.try_create(cudaEventDisableTiming);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to create offload event: %s", cudaGetErrorString(err));
            return false;
        }

        err = slots_[s].transfer_stream.try_create(cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to create offload stream: %s", cudaGetErrorString(err));
            return false;
        }
    }

    enabled_ = true;

    int n_attn = 0, n_ssm = 0, n_other = 0;
    for (int i = 0; i < n_layers; i++) {
        if (offloaded_[i]) {
            const auto& ly = model->layer(i);
            if (ly.wq.data)
                n_attn++;
            else if (ly.ssm_in.data)
                n_ssm++;
            else
                n_other++;
        }
    }

    IMP_LOG_INFO(
        "Layer offloading: %d/%d layers offloaded (%d attn, %d ssm, %d other), "
        "staging=2x%.2f MiB = %.2f MiB overhead",
        n_offloaded, n_layers, n_attn, n_ssm, n_other, max_layer_bytes / (1024.0 * 1024.0),
        2 * max_layer_bytes / (1024.0 * 1024.0));

    return true;
}

void LayerOffloadManager::upload_layer_to_slot(int layer, int slot_idx) {
    auto& slot = slots_[slot_idx];
    const auto& entries = layer_entries_[layer];

    char* dst_base = static_cast<char*>(slot.gpu_buf);

    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attrs.srcLocHint     = { cudaMemLocationTypeHostNumaCurrent, 0 };
    attrs.dstLocHint     = { cudaMemLocationTypeDevice, 0 };
    attrs.flags          = 0;
    for (const auto& e : entries) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyWithAttributesAsync(dst_base + e.offset_in_slot, e.host_ptr,
                                                         e.raw_bytes, &attrs, slot.transfer_stream));
    }

    // Record event so compute stream can wait on it
    IMP_CUDA_CHECK_LOG(cudaEventRecord(slot.ready_event, slot.transfer_stream));
    slot.loaded_layer = layer;
}

void LayerOffloadManager::ensure_layer(int layer, cudaStream_t compute_stream) {
    if (!enabled_ || !offloaded_[layer])
        return;

    // Check if already loaded in a slot
    for (int s = 0; s < 2; s++) {
        if (slots_[s].loaded_layer == layer) {
            // Wait for transfer to complete before compute
            IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(compute_stream, slots_[s].ready_event, 0));

            // Remap layer tensor pointers to staging buffer
            char* base = static_cast<char*>(slots_[s].gpu_buf);
            for (auto& e : layer_entries_[layer]) {
                e.tensor->data = base + e.offset_in_slot;
                e.tensor->on_device = true;
            }

            active_slot_ = s;
            return;
        }
    }

    // Not in any slot — need to load synchronously.
    // Use the inactive slot (the one not currently in use).
    int target_slot = 1 - active_slot_;
    upload_layer_to_slot(layer, target_slot);

    // Wait for transfer
    IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(compute_stream, slots_[target_slot].ready_event, 0));

    // Remap tensor pointers
    char* base = static_cast<char*>(slots_[target_slot].gpu_buf);
    for (auto& e : layer_entries_[layer]) {
        e.tensor->data = base + e.offset_in_slot;
        e.tensor->on_device = true;
    }

    active_slot_ = target_slot;
}

void LayerOffloadManager::prefetch_layer(int next_layer) {
    if (!enabled_ || !offloaded_[next_layer])
        return;

    // Check if already loaded
    for (int s = 0; s < 2; s++) {
        if (slots_[s].loaded_layer == next_layer)
            return;
    }

    // Start prefetch into the inactive slot
    int target = 1 - active_slot_;
    upload_layer_to_slot(next_layer, target);
}

void LayerOffloadManager::release_layer(int layer) {
    if (!enabled_ || !offloaded_[layer])
        return;

    // Restore original host pointers so the model state remains consistent
    for (auto& e : layer_entries_[layer]) {
        e.tensor->data = e.host_ptr;
        e.tensor->on_device = false;
    }
}

}  // namespace imp
