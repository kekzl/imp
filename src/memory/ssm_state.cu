#include "memory/ssm_state.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

SSMState::~SSMState() {
    if (pool_) {
        cudaFree(pool_);
        pool_ = nullptr;
    }
}

bool SSMState::init(int n_ssm_layers, int max_sequences,
                    int conv_channels, int conv_kernel,
                    int n_heads, int head_dim_ssm, int state_size,
                    DType h_dtype) {
    n_ssm_layers_ = n_ssm_layers;
    max_sequences_ = max_sequences;
    h_dtype_ = h_dtype;

    // conv_state is always FP32 (small, needs precision)
    // h_state uses h_dtype (FP32 or FP16)
    conv_bytes_ = static_cast<size_t>(conv_channels) * conv_kernel * sizeof(float);
    h_bytes_ = static_cast<size_t>(n_heads) * head_dim_ssm * state_size * dtype_size(h_dtype_);

    // Align each sub-allocation to 256 bytes
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    conv_bytes_ = align256(conv_bytes_);
    h_bytes_ = align256(h_bytes_);

    per_layer_bytes_ = conv_bytes_ + h_bytes_;
    per_seq_bytes_ = per_layer_bytes_ * n_ssm_layers_;
    total_bytes_ = per_seq_bytes_ * max_sequences_;

    cudaError_t err = cudaMalloc(&pool_, total_bytes_);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate SSM state pool (%zu bytes): %s",
                      total_bytes_, cudaGetErrorString(err));
        pool_ = nullptr;
        return false;
    }

    // Zero-initialize all state
    cudaMemset(pool_, 0, total_bytes_);

    IMP_LOG_INFO("SSM state: %d layers x %d sequences = %.2f MiB "
                 "(conv=%.1f KB, h=%.1f KB [%s] per layer)",
                 n_ssm_layers_, max_sequences_,
                 total_bytes_ / (1024.0 * 1024.0),
                 conv_bytes_ / 1024.0, h_bytes_ / 1024.0,
                 dtype_name(h_dtype_));
    return true;
}

void* SSMState::conv_state(int seq_id, int ssm_layer_idx) {
    char* base = static_cast<char*>(pool_);
    return base + seq_id * per_seq_bytes_ + ssm_layer_idx * per_layer_bytes_;
}

void* SSMState::h_state(int seq_id, int ssm_layer_idx) {
    char* base = static_cast<char*>(pool_);
    return base + seq_id * per_seq_bytes_ + ssm_layer_idx * per_layer_bytes_ + conv_bytes_;
}

void SSMState::reset_sequence(int seq_id, cudaStream_t stream) {
    if (!pool_ || seq_id < 0 || seq_id >= max_sequences_) return;
    char* base = static_cast<char*>(pool_) + seq_id * per_seq_bytes_;
    cudaMemsetAsync(base, 0, per_seq_bytes_, stream);
}

} // namespace imp
