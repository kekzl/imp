#include "memory/layer_offload.h"
#include "core/logging.h"
#include <algorithm>
#include <numeric>

namespace imp {

LayerOffloadManager::~LayerOffloadManager() {
    for (auto& slot : slots_) {
        if (slot.gpu_buf) {
            cudaFree(slot.gpu_buf);
            slot.gpu_buf = nullptr;
        }
        if (slot.ready_event) {
            cudaEventDestroy(slot.ready_event);
            slot.ready_event = nullptr;
        }
        if (slot.transfer_stream) {
            cudaStreamDestroy(slot.transfer_stream);
            slot.transfer_stream = nullptr;
        }
    }
}

void LayerOffloadManager::scan_layer_weights(int layer) {
    auto& ly = model_->layer(layer);
    auto& entries = layer_entries_[layer];
    entries.clear();

    size_t offset = 0;
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    // Helper: add a tensor if it has data
    auto add = [&](Tensor& t) {
        if (!t.data || t.on_device) return;
        size_t nbytes = t.nbytes();
        if (nbytes == 0) return;
        entries.push_back({&t, t.data, nbytes, offset});
        offset = align256(offset + nbytes);
    };

    // Attention weights
    add(ly.wq); add(ly.wk); add(ly.wv); add(ly.wo);
    add(ly.attn_norm);
    add(ly.attn_q_norm); add(ly.attn_k_norm);

    // Dense FFN weights
    add(ly.w_gate); add(ly.w_up); add(ly.w_down);
    add(ly.ffn_norm);

    // MoE weights (gate router, shared expert)
    add(ly.moe_gate);
    add(ly.w_up_shared); add(ly.w_down_shared); add(ly.w_gate_shared);
    add(ly.moe_router_bias);

    // Packed expert tensors (these are the big ones, but often already offloaded)
    add(ly.expert_gate_packed);
    add(ly.expert_up_packed);
    add(ly.expert_down_packed);

    // SSM weights
    add(ly.ssm_in); add(ly.ssm_out);
    add(ly.ssm_conv1d_w); add(ly.ssm_conv1d_b);
    add(ly.ssm_dt_b); add(ly.ssm_a); add(ly.ssm_d);
    add(ly.ssm_norm_w);

    // Scale tensors (small, on GPU already from upload)
    // These are skipped since they're always on device after upload.

    layer_slot_bytes_[layer] = offset;
}

bool LayerOffloadManager::init(Model* model, int gpu_layers) {
    model_ = model;
    int n_layers = model->n_layers();

    if (gpu_layers < 0 || gpu_layers >= n_layers) {
        enabled_ = false;
        return true;  // all on GPU, nothing to offload
    }

    // Determine which layers to keep on GPU (priority: attention > SSM > MoE-only)
    struct LayerPriority {
        int layer_idx;
        int priority;  // 0 = attention, 1 = SSM, 2 = MoE-only / other
    };
    std::vector<LayerPriority> priorities(n_layers);
    for (int i = 0; i < n_layers; i++) {
        const auto& ly = model->layer(i);
        int prio = 2;  // default: lowest
        if (ly.wq.data != nullptr) prio = 0;       // attention
        else if (ly.ssm_in.data != nullptr) prio = 1;  // SSM
        priorities[i] = {i, prio};
    }

    // Sort by priority (stable to preserve layer order within same priority)
    std::stable_sort(priorities.begin(), priorities.end(),
                     [](const auto& a, const auto& b) { return a.priority < b.priority; });

    // First gpu_layers (by priority) stay on GPU
    offloaded_.resize(n_layers, true);
    for (int i = 0; i < std::min(gpu_layers, n_layers); i++) {
        offloaded_[priorities[i].layer_idx] = false;
    }

    // Scan offloaded layers to compute slot sizes
    layer_entries_.resize(n_layers);
    layer_slot_bytes_.resize(n_layers, 0);
    size_t max_layer_bytes = 0;
    int n_offloaded = 0;

    for (int i = 0; i < n_layers; i++) {
        if (!offloaded_[i]) continue;
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
            IMP_LOG_ERROR("Failed to allocate offload slot %d (%zu bytes): %s",
                          s, max_layer_bytes, cudaGetErrorString(err));
            return false;
        }
        slots_[s].buf_size = max_layer_bytes;
        slots_[s].loaded_layer = -1;

        err = cudaEventCreateWithFlags(&slots_[s].ready_event, cudaEventDisableTiming);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to create offload event: %s", cudaGetErrorString(err));
            return false;
        }

        err = cudaStreamCreateWithFlags(&slots_[s].transfer_stream, cudaStreamNonBlocking);
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
            if (ly.wq.data) n_attn++;
            else if (ly.ssm_in.data) n_ssm++;
            else n_other++;
        }
    }

    IMP_LOG_INFO("Layer offloading: %d/%d layers offloaded (%d attn, %d ssm, %d other), "
                 "staging=2x%.2f MiB = %.2f MiB overhead",
                 n_offloaded, n_layers, n_attn, n_ssm, n_other,
                 max_layer_bytes / (1024.0 * 1024.0),
                 2 * max_layer_bytes / (1024.0 * 1024.0));

    return true;
}

void LayerOffloadManager::upload_layer_to_slot(int layer, int slot_idx) {
    auto& slot = slots_[slot_idx];
    const auto& entries = layer_entries_[layer];

    char* dst_base = static_cast<char*>(slot.gpu_buf);

    for (const auto& e : entries) {
        cudaMemcpyAsync(dst_base + e.offset_in_slot, e.host_ptr, e.raw_bytes,
                        cudaMemcpyHostToDevice, slot.transfer_stream);
    }

    // Record event so compute stream can wait on it
    cudaEventRecord(slot.ready_event, slot.transfer_stream);
    slot.loaded_layer = layer;
}

void LayerOffloadManager::ensure_layer(int layer, cudaStream_t compute_stream) {
    if (!enabled_ || !offloaded_[layer]) return;

    // Check if already loaded in a slot
    for (int s = 0; s < 2; s++) {
        if (slots_[s].loaded_layer == layer) {
            // Wait for transfer to complete before compute
            cudaStreamWaitEvent(compute_stream, slots_[s].ready_event, 0);

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
    cudaStreamWaitEvent(compute_stream, slots_[target_slot].ready_event, 0);

    // Remap tensor pointers
    char* base = static_cast<char*>(slots_[target_slot].gpu_buf);
    for (auto& e : layer_entries_[layer]) {
        e.tensor->data = base + e.offset_in_slot;
        e.tensor->on_device = true;
    }

    active_slot_ = target_slot;
}

void LayerOffloadManager::prefetch_layer(int next_layer) {
    if (!enabled_ || !offloaded_[next_layer]) return;

    // Check if already loaded
    for (int s = 0; s < 2; s++) {
        if (slots_[s].loaded_layer == next_layer) return;
    }

    // Start prefetch into the inactive slot
    int target = 1 - active_slot_;
    upload_layer_to_slot(next_layer, target);
}

void LayerOffloadManager::release_layer(int layer) {
    if (!enabled_ || !offloaded_[layer]) return;

    // Restore original host pointers so the model state remains consistent
    for (auto& e : layer_entries_[layer]) {
        e.tensor->data = e.host_ptr;
        e.tensor->on_device = false;
    }
}

} // namespace imp
