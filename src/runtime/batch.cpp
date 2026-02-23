#include "runtime/batch.h"
#include <algorithm>
#include <cstring>

namespace imp {

// ---------------------------------------------------------------------------
// GPUBatch
// ---------------------------------------------------------------------------

void GPUBatch::upload(const Batch& batch, cudaStream_t stream) {
    free();  // Clean up any previous allocation

    n_sequences = batch.n_sequences;
    total_tokens = batch.total_tokens;
    max_blocks_per_seq = batch.max_blocks_per_seq;

    if (total_tokens <= 0 || n_sequences <= 0) return;

    // Allocate device memory
    cudaMalloc(&d_token_ids, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, total_tokens * sizeof(int));
    cudaMalloc(&d_context_lens, n_sequences * sizeof(int));

    if (n_sequences > 1) {
        cudaMalloc(&d_seq_offsets, (n_sequences + 1) * sizeof(int));
    }

    if (max_blocks_per_seq > 0) {
        cudaMalloc(&d_block_tables, n_sequences * max_blocks_per_seq * sizeof(int));
    }

    // Async copy
    cudaMemcpyAsync(d_token_ids, batch.token_ids.data(),
                    total_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_positions, batch.positions.data(),
                    total_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_context_lens, batch.context_lens.data(),
                    n_sequences * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    if (d_seq_offsets && !batch.seq_offsets.empty()) {
        cudaMemcpyAsync(d_seq_offsets, batch.seq_offsets.data(),
                        (n_sequences + 1) * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    if (d_block_tables && !batch.block_tables.empty()) {
        cudaMemcpyAsync(d_block_tables, batch.block_tables.data(),
                        n_sequences * max_blocks_per_seq * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }
}

void GPUBatch::free() {
    if (d_token_ids)   { cudaFree(d_token_ids);   d_token_ids = nullptr; }
    if (d_positions)   { cudaFree(d_positions);    d_positions = nullptr; }
    if (d_seq_offsets) { cudaFree(d_seq_offsets);  d_seq_offsets = nullptr; }
    if (d_block_tables){ cudaFree(d_block_tables); d_block_tables = nullptr; }
    if (d_context_lens){ cudaFree(d_context_lens); d_context_lens = nullptr; }
    n_sequences = 0;
    total_tokens = 0;
    max_blocks_per_seq = 0;
}

// ---------------------------------------------------------------------------
// BatchBuilder
// ---------------------------------------------------------------------------

void BatchBuilder::reset() {
    batch_.token_ids.clear();
    batch_.positions.clear();
    batch_.seq_offsets.clear();
    batch_.block_tables.clear();
    batch_.context_lens.clear();
    batch_.n_sequences = 0;
    batch_.total_tokens = 0;
    batch_.max_blocks_per_seq = 0;
    raw_block_tables_.clear();

    batch_.seq_offsets.push_back(0);
}

void BatchBuilder::add_prefill_sequence(const int32_t* tokens, int n_tokens,
                                        const int* block_table, int n_blocks,
                                        int start_pos) {
    for (int i = 0; i < n_tokens; ++i) {
        batch_.token_ids.push_back(tokens[i]);
        batch_.positions.push_back(start_pos + i);
    }

    batch_.context_lens.push_back(start_pos + n_tokens);
    raw_block_tables_.push_back({block_table, n_blocks});

    batch_.total_tokens += n_tokens;
    batch_.n_sequences++;
    batch_.seq_offsets.push_back(batch_.total_tokens);
}

void BatchBuilder::add_decode_sequence(int32_t token, int position,
                                       const int* block_table, int n_blocks,
                                       int context_len) {
    batch_.token_ids.push_back(token);
    batch_.positions.push_back(position);
    batch_.context_lens.push_back(context_len);
    raw_block_tables_.push_back({block_table, n_blocks});

    batch_.total_tokens += 1;
    batch_.n_sequences++;
    batch_.seq_offsets.push_back(batch_.total_tokens);
}

Batch BatchBuilder::build() {
    // Compute max_blocks_per_seq and build padded block_tables
    int max_blocks = 0;
    for (auto& [ptr, n] : raw_block_tables_) {
        max_blocks = std::max(max_blocks, n);
    }
    batch_.max_blocks_per_seq = max_blocks;

    // Build padded 2D block table: [n_sequences, max_blocks_per_seq]
    batch_.block_tables.clear();
    batch_.block_tables.resize(batch_.n_sequences * max_blocks, 0);

    for (int s = 0; s < batch_.n_sequences; s++) {
        auto& [ptr, n] = raw_block_tables_[s];
        for (int b = 0; b < n; b++) {
            batch_.block_tables[s * max_blocks + b] = ptr[b];
        }
    }

    return batch_;
}

// ---------------------------------------------------------------------------
// GPUBatchPool -- pre-allocated device memory for stable CUDA Graph pointers
// ---------------------------------------------------------------------------

GPUBatchPool::~GPUBatchPool() {
    free_pool();
}

void GPUBatchPool::allocate(int max_batch_size, int max_blocks_per_seq) {
    free_pool();

    max_batch_size_ = max_batch_size;
    max_blocks_per_seq_ = max_blocks_per_seq;

    // Compute sizes with 256-byte alignment per sub-buffer
    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    size_t token_ids_sz   = align256(static_cast<size_t>(max_batch_size) * sizeof(int32_t));
    size_t positions_sz   = align256(static_cast<size_t>(max_batch_size) * sizeof(int));
    size_t seq_offsets_sz = align256(static_cast<size_t>(max_batch_size + 1) * sizeof(int));
    size_t block_tab_sz   = align256(static_cast<size_t>(max_batch_size) * max_blocks_per_seq * sizeof(int));
    size_t ctx_lens_sz    = align256(static_cast<size_t>(max_batch_size) * sizeof(int));
    size_t sample_res_sz  = align256(sizeof(int32_t));

    pool_size_ = token_ids_sz + positions_sz + seq_offsets_sz +
                 block_tab_sz + ctx_lens_sz + sample_res_sz;

    cudaError_t err = cudaMalloc(&pool_, pool_size_);
    if (err != cudaSuccess) {
        pool_ = nullptr;
        pool_size_ = 0;
        return;
    }

    char* ptr = static_cast<char*>(pool_);
    d_token_ids_    = reinterpret_cast<int32_t*>(ptr); ptr += token_ids_sz;
    d_positions_    = reinterpret_cast<int*>(ptr);     ptr += positions_sz;
    d_seq_offsets_  = reinterpret_cast<int*>(ptr);     ptr += seq_offsets_sz;
    d_block_tables_ = reinterpret_cast<int*>(ptr);     ptr += block_tab_sz;
    d_context_lens_ = reinterpret_cast<int*>(ptr);     ptr += ctx_lens_sz;
    d_sample_result_= reinterpret_cast<int32_t*>(ptr); ptr += sample_res_sz;
}

GPUBatch GPUBatchPool::upload_into_pool(const Batch& batch, cudaStream_t stream) {
    GPUBatch gpu;
    if (!pool_ || batch.n_sequences <= 0) return gpu;

    gpu.n_sequences = batch.n_sequences;
    gpu.total_tokens = batch.total_tokens;
    gpu.max_blocks_per_seq = batch.max_blocks_per_seq;

    // Point to pre-allocated memory (stable addresses)
    gpu.d_token_ids   = d_token_ids_;
    gpu.d_positions   = d_positions_;
    gpu.d_seq_offsets = d_seq_offsets_;
    gpu.d_block_tables = d_block_tables_;
    gpu.d_context_lens = d_context_lens_;

    // Async copy data into pool
    cudaMemcpyAsync(d_token_ids_, batch.token_ids.data(),
                    batch.total_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_positions_, batch.positions.data(),
                    batch.total_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_context_lens_, batch.context_lens.data(),
                    batch.n_sequences * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    if (batch.n_sequences > 1 && !batch.seq_offsets.empty()) {
        cudaMemcpyAsync(d_seq_offsets_, batch.seq_offsets.data(),
                        (batch.n_sequences + 1) * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    if (batch.max_blocks_per_seq > 0 && !batch.block_tables.empty()) {
        cudaMemcpyAsync(d_block_tables_, batch.block_tables.data(),
                        batch.n_sequences * batch.max_blocks_per_seq * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    return gpu;
}

void GPUBatchPool::free_pool() {
    if (pool_) {
        cudaFree(pool_);
        pool_ = nullptr;
    }
    pool_size_ = 0;
    d_token_ids_ = nullptr;
    d_positions_ = nullptr;
    d_seq_offsets_ = nullptr;
    d_block_tables_ = nullptr;
    d_context_lens_ = nullptr;
    d_sample_result_ = nullptr;
}

} // namespace imp
