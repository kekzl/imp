#pragma once

// T3 (pooled fixed-block) of the lifetime taxonomy
// (docs/MEMORY_ARCHITECTURE.md §A2/§A3.4/§A5.1).
//
// One contiguous Region carved into fixed-size blocks, handed out through a
// free list. Fixed size is the whole point: a pool of identical blocks cannot
// fragment, so the KV pool's failure mode is exhaustion (a clean, typed,
// admission-controllable condition) and never "there is memory but not in one
// piece".
//
// The pool OWNS the memory; nobody else does. A KV block has three concurrent
// referents in imp — the owning sequence's block table, the content-addressed
// prefix cache, and the agentic pin set (plus, out of process, the persisted
// prefix cache). Today each of those frees by hand, which is why
// free_block_dropping_stale_hash() exists and documents the double-ownership
// bug it prevents. Here they all hold BlockRefs and a block returns to the free
// list when, and only when, its last ref is destroyed. Cancellation, client
// disconnect and error paths stop being special cases: the refs unwind.
//
// BlockRef is move-only. Every additional referent is therefore a visible,
// greppable share() call rather than an accidental copy.

#include "memory/backend.h"
#include "memory/span.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace imp {

class BlockPool;

// RAII reference to one block. Move-only; alias explicitly via share().
class BlockRef {
public:
    BlockRef() = default;
    ~BlockRef() { reset(); }

    BlockRef(BlockRef&& o) noexcept : pool_(o.pool_), id_(o.id_) {
        o.pool_ = nullptr;
        o.id_ = -1;
    }
    BlockRef& operator=(BlockRef&& o) noexcept {
        if (this != &o) {
            reset();
            pool_ = o.pool_;
            id_ = o.id_;
            o.pool_ = nullptr;
            o.id_ = -1;
        }
        return *this;
    }
    BlockRef(const BlockRef&) = delete;
    BlockRef& operator=(const BlockRef&) = delete;

    // The ONLY way to create an additional referent. Explicit on purpose.
    [[nodiscard]] BlockRef share() const;

    void reset();

    int id() const { return id_; }
    bool valid() const { return pool_ != nullptr && id_ >= 0; }
    explicit operator bool() const { return valid(); }

private:
    friend class BlockPool;
    BlockRef(BlockPool* pool, int id) : pool_(pool), id_(id) {}

    BlockPool* pool_ = nullptr;
    int id_ = -1;
};

class BlockPool {
public:
    BlockPool() = default;
    ~BlockPool();

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;

    // Acquire num_blocks x block_bytes from `backend` as one region.
    [[nodiscard]] MemError open(Backend& backend, size_t block_bytes, int num_blocks, RegionTag tag);

    // Release the region. Asserts (debug) that no refs are outstanding — a
    // non-zero count here is exactly the "block outlived its request" bug.
    void close();

    bool is_open() const;

    // Take a free block with refcount 1. Invalid BlockRef when exhausted.
    [[nodiscard]] BlockRef acquire();

    // Bytes of one block. Stable for the pool's lifetime (I3): the region
    // never moves and blocks never migrate between ids.
    StableSpan<std::byte> block(int id) const;

    int num_blocks() const;
    int free_count() const;
    // Blocks with at least one live ref. free_count() + live_blocks() is
    // invariant and equals num_blocks() — the V4 conservation property.
    int live_blocks() const;
    // Sum of all refcounts. A request-scoped leak shows up here first.
    uint64_t total_refs() const;
    size_t block_bytes() const;

private:
    friend class BlockRef;
    void inc_ref_(int id);
    void dec_ref_(int id);

    mutable std::mutex mu_;
    Region region_;
    size_t block_bytes_ = 0;
    int num_blocks_ = 0;
    std::vector<int> refcount_;   // per block id
    std::vector<int> free_list_;  // ids with refcount 0
};

}  // namespace imp
