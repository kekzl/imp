#pragma once

#include "core/tensor.h"
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// CPU-side batch data (built by BatchBuilder)
struct Batch {
    std::vector<int32_t> token_ids;     // flattened tokens
    std::vector<int32_t> positions;     // position ids
    std::vector<int32_t> seq_offsets;   // [n_seqs+1] ragged offsets
    std::vector<int32_t> block_tables;  // flattened KV block tables (padded per seq)
    std::vector<int32_t> context_lens;  // [n_seqs] context lengths

    int n_sequences = 0;
    int total_tokens = 0;
    int max_blocks_per_seq = 0;
};

// GPU-side batch data (device pointers)
struct GPUBatch {
    int32_t* d_token_ids = nullptr;     // [total_tokens]
    int* d_positions = nullptr;          // [total_tokens]
    int* d_seq_offsets = nullptr;        // [n_sequences+1]
    int* d_block_tables = nullptr;       // [n_sequences, max_blocks_per_seq] padded
    int* d_context_lens = nullptr;       // [n_sequences]

    int n_sequences = 0;
    int total_tokens = 0;
    int max_blocks_per_seq = 0;

    // Upload CPU Batch to GPU (async)
    void upload(const Batch& batch, cudaStream_t stream = nullptr);

    // Free device memory
    void free();

    bool is_valid() const { return d_token_ids != nullptr; }
};

// Pre-allocated GPU batch pool for CUDA Graph compatibility.
// Allocates device memory once for a max batch configuration and reuses
// it across decode steps, providing stable pointers required by CUDA Graphs.
class GPUBatchPool {
public:
    GPUBatchPool() = default;
    ~GPUBatchPool();

    // Allocate pool for the given max configuration. Call once at init.
    void allocate(int max_batch_size, int max_blocks_per_seq);

    // Upload a CPU batch into the pre-allocated pool (async).
    // Returns a GPUBatch with pointers into the pool (caller must NOT free them).
    GPUBatch upload_into_pool(const Batch& batch, cudaStream_t stream = nullptr);

    // Free all pool memory.
    void free_pool();

    bool is_allocated() const { return pool_ != nullptr; }

    // Pre-allocated single int32 result buffer for sampling kernels
    int32_t* d_sample_result() const { return d_sample_result_; }

private:
    void* pool_ = nullptr;
    size_t pool_size_ = 0;

    // Offsets into pool_
    int32_t* d_token_ids_ = nullptr;
    int* d_positions_ = nullptr;
    int* d_seq_offsets_ = nullptr;
    int* d_block_tables_ = nullptr;
    int* d_context_lens_ = nullptr;
    int32_t* d_sample_result_ = nullptr;

    int max_batch_size_ = 0;
    int max_blocks_per_seq_ = 0;
};

class BatchBuilder {
public:
    BatchBuilder() = default;

    void reset();

    // Add a prefill sequence (multiple tokens)
    void add_prefill_sequence(const int32_t* tokens, int n_tokens,
                              const int* block_table, int n_blocks,
                              int start_pos);

    // Add a decode sequence (single token)
    void add_decode_sequence(int32_t token, int position,
                             const int* block_table, int n_blocks,
                             int context_len);

    Batch build();

private:
    Batch batch_;
    std::vector<std::pair<const int*, int>> raw_block_tables_;  // (ptr, n_blocks)
};

} // namespace imp
