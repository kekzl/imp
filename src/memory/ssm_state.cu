#include "memory/ssm_state.h"

#include <algorithm>
#include "memory/ssm_state_size.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

SSMState::~SSMState() {
    if (pool_) {
        if (alloc_)
            alloc_->free(pool_);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(pool_));
        pool_ = nullptr;
    }
}

bool SSMState::init(int n_ssm_layers, int max_sequences, int conv_channels, int conv_kernel, int n_heads,
                    int head_dim_ssm, int state_size, QType h_dtype, VRAMAllocator* alloc, int n_reserved) {
    n_ssm_layers_ = n_ssm_layers;
    max_sequences_ = max_sequences;
    n_reserved_ = std::max(0, n_reserved);
    h_dtype_ = h_dtype;
    alloc_ = alloc;

    // Geometry and byte counts: memory/ssm_state_size.h, the ONE formula. The
    // plan charges the same header, so what is charged is what is taken.
    const SsmStateGeometry geom{n_ssm_layers_, conv_channels, conv_kernel,
                                n_heads,       head_dim_ssm, state_size, h_dtype_};
    conv_bytes_ = ssm_conv_bytes_per_layer(geom);
    h_bytes_ = ssm_h_bytes_per_layer(geom);
    per_layer_bytes_ = ssm_bytes_per_layer(geom);
    per_seq_bytes_ = ssm_bytes_per_slot(geom);
    total_bytes_ = ssm_pool_bytes(geom, max_sequences_, n_reserved_);

    if (alloc_) {
        // No raw-cudaMalloc retry behind the allocator's back. That hatch is
        // how 5088 MiB of state landed past the headroom on Qwen3.8-27B-NVFP4
        // and spilled the KV pool to 528 GB/s (MEMORY.md D14): the allocator
        // said no, the pool took the memory anyway, and nothing downstream
        // knew. A rejected pool is a refused configuration now, with the
        // numbers and the lever.
        pool_ = alloc_->allocate(total_bytes_, "ssm_state");
    } else {
        cudaError_t err = cudaMalloc(&pool_, total_bytes_);
        if (err != cudaSuccess)
            pool_ = nullptr;
    }
    if (!pool_) {
        size_t free_bytes = 0, total_device = 0;
        if (cudaMemGetInfo(&free_bytes, &total_device) != cudaSuccess)
            free_bytes = 0;
        IMP_LOG_ERROR("%s", ssm_pool_failure_message(total_bytes_, max_sequences_, n_reserved_,
                                                     n_ssm_layers_, free_bytes)
                                .c_str());
        return false;
    }

    // Zero-initialize all state
    IMP_CUDA_CHECK_LOG(cudaMemset(pool_, 0, total_bytes_));

    IMP_LOG_INFO(
        "SSM state: %d layers x %d sequences = %.2f MiB "
        "(conv=%.1f KB, h=%.1f KB [%s] per layer)",
        n_ssm_layers_, max_sequences_, total_bytes_ / (1024.0 * 1024.0), conv_bytes_ / 1024.0,
        h_bytes_ / 1024.0, dtype_name(h_dtype_));
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
    if (!pool_ || seq_id < 0 || seq_id >= max_sequences_ + n_reserved_)
        return;
    char* base = static_cast<char*>(pool_) + seq_id * per_seq_bytes_;
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(base, 0, per_seq_bytes_, stream));
}

}  // namespace imp
