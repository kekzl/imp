#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

class VRAMAllocator;

// Manages per-sequence, per-GDN-layer recurrent state for Gated DeltaNet.
// State per (sequence, layer): S[n_heads, head_dim, state_dim] in FP32.
// The delta rule updates S incrementally: S = α*S + β*(v - S^T k)⊗k.
// Unlike Mamba2, GDN has no conv1d state — only the recurrent state matrix.
class GDNState {
public:
    GDNState() = default;
    ~GDNState();

    // Allocate state pool.
    // n_heads: number of recurrence heads (= ssm_dt_rank from GGUF)
    // head_dim: dimension per head (= ssm_inner_size / n_heads)
    // state_dim: state matrix second dimension (= ssm_state_size from GGUF)
    bool init(int n_gdn_layers, int max_sequences, int n_heads, int head_dim, int state_dim,
              VRAMAllocator* alloc = nullptr);

    // Get pointer to S state for a given sequence and GDN layer.
    // Returns float* of shape [n_heads, head_dim, state_dim].
    void* s_state(int seq_id, int gdn_layer_idx);

    // Zero-initialize all state for a sequence (on new request).
    void reset_sequence(int seq_id, cudaStream_t stream);

    int max_sequences() const { return max_sequences_; }
    int n_gdn_layers() const { return n_gdn_layers_; }
    int n_heads() const { return n_heads_; }
    int head_dim() const { return head_dim_; }
    int state_dim() const { return state_dim_; }

private:
    VRAMAllocator* alloc_ = nullptr;
    void* pool_ = nullptr;
    int n_gdn_layers_ = 0;
    int max_sequences_ = 0;
    int n_heads_ = 0;
    int head_dim_ = 0;
    int state_dim_ = 0;
    size_t per_layer_bytes_ = 0;
    size_t per_seq_bytes_ = 0;
    size_t total_bytes_ = 0;
};

}  // namespace imp
