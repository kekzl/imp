#include "memory/block_pool.h"
#include "core/logging.h"

#include <cstdlib>

namespace imp {

// --- BlockRef ---

BlockRef BlockRef::share() const {
    if (!valid())
        return BlockRef();
    pool_->inc_ref_(id_);
    return BlockRef(pool_, id_);
}

void BlockRef::reset() {
    if (pool_ && id_ >= 0)
        pool_->dec_ref_(id_);
    pool_ = nullptr;
    id_ = -1;
}

int BlockRef::release() {
    const int id = id_;
    pool_ = nullptr;
    id_ = -1;
    return id;
}

// --- BlockPool ---

BlockPool::~BlockPool() { close(); }

void BlockPool::init_slots_(int num_blocks) {
    open_ = true;
    num_blocks_ = num_blocks;
    refcount_.assign(static_cast<size_t>(num_blocks), 0);
    free_list_.clear();
    free_list_.reserve(static_cast<size_t>(num_blocks));
    // Descending so the first acquire() hands out block 0 — deterministic ids
    // make a journal replay comparable across runs, and it matches the order
    // KVCache's own free list handed out before the migration.
    for (int i = num_blocks - 1; i >= 0; --i)
        free_list_.push_back(i);
}

MemError BlockPool::open(Backend& backend, size_t block_bytes, int num_blocks, RegionTag tag) {
    std::lock_guard<std::mutex> lock(mu_);
    if (open_)
        return MemError::InvalidArgument;
    if (block_bytes == 0 || num_blocks <= 0)
        return MemError::InvalidArgument;

    auto res = backend.acquire(block_bytes * static_cast<size_t>(num_blocks), 256, tag);
    if (!res)
        return res.error;

    region_ = std::move(res.region);
    block_bytes_ = block_bytes;
    init_slots_(num_blocks);
    return MemError::Ok;
}

MemError BlockPool::open_slots(int num_blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    if (open_)
        return MemError::InvalidArgument;
    if (num_blocks <= 0)
        return MemError::InvalidArgument;
    block_bytes_ = 0;
    init_slots_(num_blocks);
    return MemError::Ok;
}

void BlockPool::close() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!open_)
        return;

    const size_t outstanding = refcount_.size() - free_list_.size();
    if (outstanding != 0) {
        IMP_LOG_ERROR("BlockPool::close with %zu blocks still referenced — a block outlived its "
                      "owner (docs/internals/MEMORY.md A5.1)",
                      outstanding);
#ifndef NDEBUG
        std::abort();
#endif
    }

    region_.reset();
    open_ = false;
    block_bytes_ = 0;
    num_blocks_ = 0;
    refcount_.clear();
    free_list_.clear();
}

bool BlockPool::is_open() const {
    std::lock_guard<std::mutex> lock(mu_);
    return open_;
}

BlockRef BlockPool::acquire() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_list_.empty())
        return BlockRef();
    const int id = free_list_.back();
    free_list_.pop_back();
    refcount_[static_cast<size_t>(id)] = 1;
    return BlockRef(this, id);
}

void BlockPool::inc_ref_(int id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (id < 0 || id >= num_blocks_)
        return;
    ++refcount_[static_cast<size_t>(id)];
}

void BlockPool::dec_ref_(int id) { dec_ref_impl_(id, /*strict=*/true); }

void BlockPool::dec_ref_impl_(int id, bool strict) {
    std::lock_guard<std::mutex> lock(mu_);
    if (id < 0 || id >= num_blocks_)
        return;
    int& rc = refcount_[static_cast<size_t>(id)];
    if (rc <= 0) {
        // The int-based KV API tolerates a free of an already-free block
        // (KVCache::free_block returns silently). Keep that exact behaviour
        // for the raw path so the migration cannot change semantics; a drop
        // through a BlockRef is strict, because there it IS a double-free.
        if (!strict)
            return;
        IMP_LOG_ERROR("BlockPool: dec_ref on block %d with refcount %d (double release)", id, rc);
#ifndef NDEBUG
        std::abort();
#endif
        return;
    }
    if (--rc == 0)
        free_list_.push_back(id);
}

void BlockPool::acquire_raw(int id) { inc_ref_(id); }

void BlockPool::release_raw(int id) { dec_ref_impl_(id, /*strict=*/false); }

BlockRef BlockPool::share_by_id(int id) {
    std::lock_guard<std::mutex> lock(mu_);
    if (id < 0 || id >= num_blocks_ || refcount_[static_cast<size_t>(id)] <= 0)
        return BlockRef();
    ++refcount_[static_cast<size_t>(id)];
    return BlockRef(this, id);
}

StableSpan<std::byte> BlockPool::block(int id) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!region_.valid() || id < 0 || id >= num_blocks_)
        return StableSpan<std::byte>();
    auto* base = static_cast<std::byte*>(region_.base()) + static_cast<size_t>(id) * block_bytes_;
    return StableSpan<std::byte>(detail::StableKey{}, base, block_bytes_);
}

int BlockPool::num_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return num_blocks_;
}

int BlockPool::free_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(free_list_.size());
}

int BlockPool::live_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return num_blocks_ - static_cast<int>(free_list_.size());
}

uint64_t BlockPool::total_refs() const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t sum = 0;
    for (int rc : refcount_)
        sum += static_cast<uint64_t>(rc);
    return sum;
}

int BlockPool::ref_count(int id) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (id < 0 || id >= num_blocks_)
        return 0;
    return refcount_[static_cast<size_t>(id)];
}

void BlockPool::abandon() {
    std::lock_guard<std::mutex> lock(mu_);
    region_.reset();
    open_ = false;
    block_bytes_ = 0;
    num_blocks_ = 0;
    refcount_.clear();
    free_list_.clear();
}

size_t BlockPool::block_bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return block_bytes_;
}

}  // namespace imp
