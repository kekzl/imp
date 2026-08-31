#include "runtime/batch.h"
#include "memory/engine_arena.h"
#include "memory/vram_allocator.h"
#include "core/logging.h"
#include <algorithm>
#include <ranges>
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

    if (total_tokens <= 0 || n_sequences <= 0)
        return;

    // Allocate device memory
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_token_ids, total_tokens * sizeof(int32_t)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_positions, total_tokens * sizeof(int)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_context_lens, n_sequences * sizeof(int)));

    if (n_sequences > 1) {
        IMP_CUDA_CHECK_LOG(cudaMalloc(&d_seq_offsets, (n_sequences + 1) * sizeof(int)));
    }

    if (max_blocks_per_seq > 0) {
        IMP_CUDA_CHECK_LOG(
            cudaMalloc(&d_block_tables, static_cast<unsigned long>(n_sequences) * max_blocks_per_seq * sizeof(int)));
    }

    // Async copy
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_ids, batch.token_ids.data(), total_tokens * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_positions, batch.positions.data(), total_tokens * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_context_lens, batch.context_lens.data(), n_sequences * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));

    if (d_seq_offsets && !batch.seq_offsets.empty()) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_seq_offsets, batch.seq_offsets.data(),
                                           (n_sequences + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    if (d_block_tables && !batch.block_tables.empty()) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_block_tables, batch.block_tables.data(),
                                           static_cast<unsigned long>(n_sequences) * max_blocks_per_seq * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    }
}

void GPUBatch::free() {
    if (d_token_ids) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_ids));
        d_token_ids = nullptr;
    }
    if (d_positions) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_positions));
        d_positions = nullptr;
    }
    if (d_seq_offsets) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_seq_offsets));
        d_seq_offsets = nullptr;
    }
    if (d_block_tables) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_block_tables));
        d_block_tables = nullptr;
    }
    if (d_context_lens) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_context_lens));
        d_context_lens = nullptr;
    }
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
    batch_.block_tables_swa.clear();
    batch_.context_lens.clear();
    batch_.n_sequences = 0;
    batch_.total_tokens = 0;
    batch_.max_blocks_per_seq = 0;
    batch_.actual_blocks_per_seq = 0;
    raw_block_tables_.clear();
    raw_swa_block_tables_.clear();
    any_swa_tables_ = false;

    batch_.seq_offsets.push_back(0);
}

void BatchBuilder::add_prefill_sequence(std::span<const int32_t> tokens, std::span<const int> block_table,
                                        int start_pos, std::span<const int> swa_block_table) {
    const int n_tokens = static_cast<int>(tokens.size());
    for (const auto [i, tok] : std::views::enumerate(tokens)) {
        batch_.token_ids.push_back(tok);
        batch_.positions.push_back(start_pos + static_cast<int>(i));
    }

    batch_.context_lens.push_back(start_pos + n_tokens);
    raw_block_tables_.push_back(block_table);
    raw_swa_block_tables_.push_back(swa_block_table);
    any_swa_tables_ |= !swa_block_table.empty();

    batch_.total_tokens += n_tokens;
    batch_.n_sequences++;
    batch_.seq_offsets.push_back(batch_.total_tokens);
}

void BatchBuilder::add_decode_sequence(int32_t token, int position, std::span<const int> block_table,
                                       int context_len, std::span<const int> swa_block_table) {
    batch_.token_ids.push_back(token);
    batch_.positions.push_back(position);
    batch_.context_lens.push_back(context_len);
    raw_block_tables_.push_back(block_table);
    raw_swa_block_tables_.push_back(swa_block_table);
    any_swa_tables_ |= !swa_block_table.empty();

    batch_.total_tokens += 1;
    batch_.n_sequences++;
    batch_.seq_offsets.push_back(batch_.total_tokens);
}

Batch BatchBuilder::build() {
    // Compute max_blocks_per_seq and build padded block_tables
    int max_blocks = 0;
    for (const auto& table : raw_block_tables_)
        max_blocks = std::max(max_blocks, static_cast<int>(table.size()));
    batch_.max_blocks_per_seq = max_blocks;
    batch_.actual_blocks_per_seq = max_blocks;

    // Build padded 2D block table: [n_sequences, max_blocks_per_seq]
    batch_.block_tables.clear();
    batch_.block_tables.resize(static_cast<unsigned long>(batch_.n_sequences) * max_blocks, 0);

    for (int s = 0; s < batch_.n_sequences; s++) {
        const auto& table = raw_block_tables_[static_cast<size_t>(s)];
        for (const auto [b, block] : std::views::enumerate(table))
            batch_.block_tables[s * max_blocks + static_cast<int>(b)] = block;
    }

    // Parallel SWA tables (same shape/stride). Padded with -1 — a padded slot
    // must read as a hole, never as SWA block 0.
    if (any_swa_tables_) {
        batch_.block_tables_swa.assign(static_cast<unsigned long>(batch_.n_sequences) * max_blocks, -1);
        for (int s = 0; s < batch_.n_sequences; s++) {
            const auto full = raw_swa_block_tables_[static_cast<size_t>(s)];
            const auto table = full.first(std::min(full.size(), static_cast<size_t>(max_blocks)));
            for (const auto [b, block] : std::views::enumerate(table))
                batch_.block_tables_swa[s * max_blocks + static_cast<int>(b)] = block;
        }
    }

    return std::move(batch_);
}

// ---------------------------------------------------------------------------
// GPUBatchPool -- pre-allocated device memory for stable CUDA Graph pointers
// ---------------------------------------------------------------------------

GPUBatchPool::~GPUBatchPool() { free_pool(); }

// 256-byte alignment per sub-buffer, so the offsets below stay valid.
static size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

size_t GPUBatchPool::demand_bytes(int max_batch_size, int max_blocks_per_seq, bool with_swa_tables) {
    const size_t block_tab_sz =
        align256(static_cast<size_t>(max_batch_size) * max_blocks_per_seq * sizeof(int));
    return align256(static_cast<size_t>(max_batch_size) * sizeof(int32_t)) +      // token ids
           align256(static_cast<size_t>(max_batch_size) * sizeof(int)) +          // positions
           align256(static_cast<size_t>(max_batch_size + 1) * sizeof(int)) +      // seq offsets
           block_tab_sz + (with_swa_tables ? block_tab_sz : 0) +                  // block tables
           align256(static_cast<size_t>(max_batch_size) * sizeof(int)) +          // ctx lens
           align256(sizeof(int32_t));                                             // sample result
}

void GPUBatchPool::allocate(int max_batch_size, int max_blocks_per_seq, bool with_swa_tables) {
    free_pool();

    max_batch_size_ = max_batch_size;
    max_blocks_per_seq_ = max_blocks_per_seq;
    last_upload_block_tables_.clear();
    last_upload_block_tables_swa_.clear();

    size_t token_ids_sz = align256(static_cast<size_t>(max_batch_size) * sizeof(int32_t));
    size_t positions_sz = align256(static_cast<size_t>(max_batch_size) * sizeof(int));
    size_t seq_offsets_sz = align256(static_cast<size_t>(max_batch_size + 1) * sizeof(int));
    size_t block_tab_sz = align256(static_cast<size_t>(max_batch_size) * max_blocks_per_seq * sizeof(int));
    size_t swa_tab_sz = with_swa_tables ? block_tab_sz : 0;
    size_t ctx_lens_sz = align256(static_cast<size_t>(max_batch_size) * sizeof(int));

    pool_size_ = demand_bytes(max_batch_size, max_blocks_per_seq, with_swa_tables);

    auto slab = engine_arena().take_bytes(pool_size_);
    if (slab.empty()) {
        IMP_LOG_ERROR("decode batch pool: engine arena exhausted for %zu bytes — the arena was "
                      "reserved without this pool",
                      pool_size_);
        pool_ = nullptr;
        pool_size_ = 0;
        return;
    }
    pool_ = slab.data();

    char* ptr = static_cast<char*>(pool_);
    d_token_ids_ = reinterpret_cast<int32_t*>(ptr);
    ptr += token_ids_sz;
    d_positions_ = reinterpret_cast<int*>(ptr);
    ptr += positions_sz;
    d_seq_offsets_ = reinterpret_cast<int*>(ptr);
    ptr += seq_offsets_sz;
    d_block_tables_ = reinterpret_cast<int*>(ptr);
    ptr += block_tab_sz;
    if (with_swa_tables) {
        d_block_tables_swa_ = reinterpret_cast<int*>(ptr);
        ptr += swa_tab_sz;
    }
    d_context_lens_ = reinterpret_cast<int*>(ptr);
    ptr += ctx_lens_sz;
    d_sample_result_ = reinterpret_cast<int32_t*>(ptr);

    // Pinned mirror of everything up to (excluding) the sample result.
    h_used_bytes_ = pool_size_ - align256(sizeof(int32_t));
    h_pool_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(), h_used_bytes_);
    if (h_pool_.empty())
        IMP_LOG_WARN("decode batch pool: pinned mirror alloc failed (%zu bytes) - "
                     "per-step uploads stay per-buffer",
                     h_used_bytes_);
}

GPUBatch GPUBatchPool::upload_into_pool(const Batch& batch, cudaStream_t stream) {
    GPUBatch gpu;
    if (!pool_ || batch.n_sequences <= 0)
        return gpu;

    gpu.n_sequences = batch.n_sequences;
    gpu.total_tokens = batch.total_tokens;
    gpu.max_blocks_per_seq = batch.max_blocks_per_seq;

    // Point to pre-allocated memory (stable addresses)
    gpu.d_token_ids = d_token_ids_;
    gpu.d_positions = d_positions_;
    gpu.d_seq_offsets = d_seq_offsets_;
    gpu.d_block_tables = d_block_tables_;
    gpu.d_block_tables_swa =
        (!batch.block_tables_swa.empty() && d_block_tables_swa_) ? d_block_tables_swa_ : nullptr;
    gpu.d_context_lens = d_context_lens_;

    // One consolidated H2D at batch > 1: the vectors are memcpy'd into the
    // pinned mirror at the same offsets the device pool uses, then a single
    // async copy covers token ids .. ctx lens (block-table dedupe only pays
    // at n == 1, where the old path below still runs).
    if (batch.n_sequences > 1 && !h_pool_.empty()) {
        char* hb = static_cast<char*>(h_pool_.data());
        char* db = static_cast<char*>(pool_);
        auto put = [&](const void* src, size_t bytes, const void* dst) {
            memcpy(hb + (static_cast<const char*>(dst) - db), src, bytes);
        };
        put(batch.token_ids.data(), batch.total_tokens * sizeof(int32_t), d_token_ids_);
        put(batch.positions.data(), batch.total_tokens * sizeof(int), d_positions_);
        put(batch.context_lens.data(), batch.n_sequences * sizeof(int), d_context_lens_);
        if (!batch.seq_offsets.empty())
            put(batch.seq_offsets.data(), (batch.n_sequences + 1) * sizeof(int), d_seq_offsets_);
        if (batch.max_blocks_per_seq > 0 && !batch.block_tables.empty())
            put(batch.block_tables.data(),
                static_cast<size_t>(batch.n_sequences) * batch.max_blocks_per_seq * sizeof(int),
                d_block_tables_);
        if (gpu.d_block_tables_swa && !batch.block_tables_swa.empty())
            put(batch.block_tables_swa.data(),
                static_cast<size_t>(batch.n_sequences) * batch.max_blocks_per_seq * sizeof(int),
                d_block_tables_swa_);
        // Span: from the pool base through the end of ctx lens (the layout
        // puts ctx lens after both block-table regions - see allocate()).
        const size_t span = static_cast<const char*>(static_cast<const void*>(d_context_lens_)) - db +
                            align256(static_cast<size_t>(max_batch_size_) * sizeof(int));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(pool_, hb, std::min(span, h_used_bytes_),
                                           cudaMemcpyHostToDevice, stream));
        last_upload_block_tables_.clear();
        last_upload_block_tables_swa_.clear();
        return gpu;
    }
    // Async copy data into pool
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_ids_, batch.token_ids.data(),
                                       batch.total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_positions_, batch.positions.data(), batch.total_tokens * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_context_lens_, batch.context_lens.data(),
                                       batch.n_sequences * sizeof(int), cudaMemcpyHostToDevice, stream));

    if (batch.n_sequences > 1 && !batch.seq_offsets.empty()) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_seq_offsets_, batch.seq_offsets.data(),
                                           (batch.n_sequences + 1) * sizeof(int), cudaMemcpyHostToDevice,
                                           stream));
    }

    if (batch.max_blocks_per_seq > 0 && !batch.block_tables.empty()) {
        // For single-seq decode, skip the block_table re-upload only when the
        // CONTENT matches the last upload exactly. Size/first-block proxies are
        // NOT sufficient — see last_upload_block_tables_ in batch.h (#536).
        if (batch.n_sequences == 1 && batch.block_tables == last_upload_block_tables_) {
            // Skip — block_table content is identical to last upload
        } else {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_block_tables_, batch.block_tables.data(),
                                               static_cast<unsigned long>(batch.n_sequences) *
                                                   batch.max_blocks_per_seq * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
            if (batch.n_sequences == 1)
                last_upload_block_tables_ = batch.block_tables;
            else
                last_upload_block_tables_.clear();
        }
    }

    // SWA-group tables: same shape/dedupe as the main table.
    if (gpu.d_block_tables_swa && batch.max_blocks_per_seq > 0) {
        if (batch.n_sequences == 1 && batch.block_tables_swa == last_upload_block_tables_swa_) {
            // Skip — content identical to last upload
        } else {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_block_tables_swa_, batch.block_tables_swa.data(),
                                               static_cast<unsigned long>(batch.n_sequences) *
                                                   batch.max_blocks_per_seq * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
            if (batch.n_sequences == 1)
                last_upload_block_tables_swa_ = batch.block_tables_swa;
            else
                last_upload_block_tables_swa_.clear();
        }
    }

    return gpu;
}

// An arena slice: released wholesale when the arena closes, so this only drops
// pointers. allocate() runs once per engine, so nothing is leaked by not freeing.
void GPUBatchPool::free_pool() {
    h_pool_.reset();
    h_used_bytes_ = 0;
    pool_ = nullptr;
    pool_size_ = 0;
    d_token_ids_ = nullptr;
    d_positions_ = nullptr;
    d_seq_offsets_ = nullptr;
    d_block_tables_ = nullptr;
    d_block_tables_swa_ = nullptr;
    d_context_lens_ = nullptr;
    d_sample_result_ = nullptr;
    last_upload_block_tables_.clear();
    last_upload_block_tables_swa_.clear();
}

}  // namespace imp
