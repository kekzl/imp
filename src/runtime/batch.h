#pragma once

#include "core/tensor.h"
#include <span>
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
    // Parallel SWA-group block tables (kv_cache.swa_sizing): same
    // [n_seqs, max_blocks_per_seq] shape/stride as block_tables, -1 = hole.
    // Empty when SWA sizing is off.
    std::vector<int32_t> block_tables_swa;
    std::vector<int32_t> context_lens;  // [n_seqs] context lengths

    int n_sequences = 0;
    int total_tokens = 0;
    int max_blocks_per_seq = 0;
    int actual_blocks_per_seq = 0;  // Pre-padding block count (for upload cache invalidation)
};

// GPU-side batch data (device pointers)
struct GPUBatch {
    int32_t* d_token_ids = nullptr;  // [total_tokens]
    int* d_positions = nullptr;      // [total_tokens]
    int* d_seq_offsets = nullptr;    // [n_sequences+1]
    int* d_block_tables = nullptr;   // [n_sequences, max_blocks_per_seq] padded
    int* d_block_tables_swa = nullptr;  // SWA-group tables (same shape) or nullptr
    int* d_context_lens = nullptr;   // [n_sequences]

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
    // with_swa_tables adds a second block-table region for the SWA-group
    // tables (kv_cache.swa_sizing): same stride, stable pointer.
    // One contiguous engine-lifetime slab from the T2 arena, sized from config.
    // demand_bytes() is the single source of that size formula; allocate()
    // calls it too, so Engine::init can reserve before the pool exists.
    static size_t demand_bytes(int max_batch_size, int max_blocks_per_seq, bool with_swa_tables);

    void allocate(int max_batch_size, int max_blocks_per_seq, bool with_swa_tables);

    // Upload a CPU batch into the pre-allocated pool (async).
    // Returns a GPUBatch with pointers into the pool (caller must NOT free them).
    GPUBatch upload_into_pool(const Batch& batch, cudaStream_t stream = nullptr);

    // Free all pool memory.
    void free_pool();

    bool is_allocated() const { return pool_ != nullptr; }
    int max_blocks_per_seq() const { return max_blocks_per_seq_; }
    void reset_upload_cache() {
        last_upload_block_tables_.clear();
        last_upload_block_tables_swa_.clear();
    }

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
    int* d_block_tables_swa_ = nullptr;
    int* d_context_lens_ = nullptr;
    int32_t* d_sample_result_ = nullptr;

    int max_batch_size_ = 0;
    int max_blocks_per_seq_ = 0;

    // Track block_table content for single-seq decode: skip re-upload only
    // when content is unchanged. A (count, first-block-ID) proxy fails under
    // prefix-caching reuse+eviction reordering (#536). Compare costs
    // nanoseconds per decode step for <=max_blocks ints.
    std::vector<int> last_upload_block_tables_;
    std::vector<int> last_upload_block_tables_swa_;
};

class BatchBuilder {
public:
    BatchBuilder() = default;

    void reset();

    // Add a prefill sequence (multiple tokens). swa_block_table (optional) is
    // the parallel SWA-group table; pass an empty span when SWA sizing is off.
    // Spans, not (pointer, count): build() re-checks the pointer, so a null
    // pointer with a nonzero count (SWA inactive) stays harmless.
    void add_prefill_sequence(std::span<const int32_t> tokens, std::span<const int> block_table,
                              int start_pos, std::span<const int> swa_block_table = {});

    // Add a decode sequence (single token)
    void add_decode_sequence(int32_t token, int position, std::span<const int> block_table, int context_len,
                             std::span<const int> swa_block_table = {});

    Batch build();

private:
    Batch batch_;
    std::vector<std::span<const int>> raw_block_tables_;
    std::vector<std::span<const int>> raw_swa_block_tables_;
    bool any_swa_tables_ = false;
};

}  // namespace imp
