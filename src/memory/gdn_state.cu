#include "memory/gdn_state.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

GDNState::~GDNState() {
    if (pool_) {
        cudaFree(pool_);
        pool_ = nullptr;
    }
}

bool GDNState::init(int n_gdn_layers, int max_sequences,
                    int n_heads, int head_dim, int state_dim) {
    n_gdn_layers_ = n_gdn_layers;
    max_sequences_ = max_sequences;
    n_heads_ = n_heads;
    head_dim_ = head_dim;
    state_dim_ = state_dim;

    // State S[n_heads, head_dim, state_dim] in FP32
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    per_layer_bytes_ = align256(
        static_cast<size_t>(n_heads) * head_dim * state_dim * sizeof(float));
    per_seq_bytes_ = per_layer_bytes_ * n_gdn_layers;
    total_bytes_ = per_seq_bytes_ * max_sequences;

    cudaError_t err = cudaMalloc(&pool_, total_bytes_);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate GDN state pool (%zu bytes): %s",
                      total_bytes_, cudaGetErrorString(err));
        pool_ = nullptr;
        return false;
    }

    cudaMemset(pool_, 0, total_bytes_);

    IMP_LOG_INFO("GDN state: %d layers x %d sequences = %.2f MiB "
                 "(S=[%d,%d,%d] FP32, %.1f KB per layer)",
                 n_gdn_layers, max_sequences,
                 total_bytes_ / (1024.0 * 1024.0),
                 n_heads, head_dim, state_dim,
                 per_layer_bytes_ / 1024.0);
    return true;
}

void* GDNState::s_state(int seq_id, int gdn_layer_idx) {
    char* base = static_cast<char*>(pool_);
    return base + seq_id * per_seq_bytes_ + gdn_layer_idx * per_layer_bytes_;
}

void GDNState::reset_sequence(int seq_id, cudaStream_t stream) {
    if (!pool_ || seq_id < 0 || seq_id >= max_sequences_) return;
    char* base = static_cast<char*>(pool_) + seq_id * per_seq_bytes_;
    cudaMemsetAsync(base, 0, per_seq_bytes_, stream);
}

} // namespace imp
