#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

class VRAMAllocator;

// Manages per-sequence, per-SSM-layer state for Mamba2 models.
// Two state types per (sequence, layer):
//   - conv_state: [conv_channels, conv_kernel] float (sliding window for causal conv1d)
//   - h_state:    [n_heads, head_dim_ssm, state_size] in h_dtype (SSM recurrent state)
//
// conv_state is always FP32 (small, needs precision for convolution).
// h_state dtype is configurable: FP32 (default) or FP16 (saves ~50% VRAM for h_state).
// SSM scan computes in FP32 regardless; FP16 h_state uses FP16 load/store only.
class SSMState {
public:
    SSMState() = default;
    ~SSMState();

    // Allocate state for the given configuration.
    // h_dtype: QType::F32 (default) or QType::F16 for h_state storage.
    [[nodiscard]] bool init(int n_ssm_layers, int max_sequences, int conv_channels, int conv_kernel,
                            int n_heads, int head_dim_ssm, int state_size, QType h_dtype = QType::F32,
                            VRAMAllocator* alloc = nullptr);

    // Get pointers into the state pool for a given sequence and SSM layer index.
    void* conv_state(int seq_id, int ssm_layer_idx);
    void* h_state(int seq_id, int ssm_layer_idx);

    // Zero-initialize all state for a sequence (on new request).
    void reset_sequence(int seq_id, cudaStream_t stream);

    // Base pointer / size of one sequence's full state slab (all layers,
    // conv + h contiguous) — the unit copied by recurrent-state snapshots.
    void* seq_base(int seq_id) {
        return static_cast<char*>(pool_) + static_cast<size_t>(seq_id) * per_seq_bytes_;
    }
    size_t per_seq_bytes() const { return per_seq_bytes_; }

    // The same per-layer layout applied to an arbitrary slab of per_seq_bytes()
    // — a snapshot scratch rather than a live sequence. The speculative verify
    // writes a second, mid-chunk state into one of these so a partial
    // acceptance can adopt it instead of re-forwarding to reach it.
    void* conv_state_in(void* slab, int ssm_layer_idx) const {
        return static_cast<char*>(slab) + static_cast<size_t>(ssm_layer_idx) * per_layer_bytes_;
    }
    void* h_state_in(void* slab, int ssm_layer_idx) const {
        return static_cast<char*>(slab) + static_cast<size_t>(ssm_layer_idx) * per_layer_bytes_ + conv_bytes_;
    }

    int max_sequences() const { return max_sequences_; }
    int n_ssm_layers() const { return n_ssm_layers_; }
    QType h_dtype() const { return h_dtype_; }

private:
    VRAMAllocator* alloc_ = nullptr;
    void* pool_ = nullptr;
    int n_ssm_layers_ = 0;
    int max_sequences_ = 0;
    QType h_dtype_ = QType::F32;
    size_t conv_bytes_ = 0;       // per (seq, layer) conv state
    size_t h_bytes_ = 0;          // per (seq, layer) h state
    size_t per_layer_bytes_ = 0;  // conv_bytes_ + h_bytes_
    size_t per_seq_bytes_ = 0;    // per_layer_bytes_ * n_ssm_layers_
    size_t total_bytes_ = 0;
};

}  // namespace imp
