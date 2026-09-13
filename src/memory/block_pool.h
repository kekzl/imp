#pragma once

// T3 (pooled fixed-block) of the lifetime taxonomy (MEMORY.md A2/A3.4/A5.1). One
// contiguous Region carved into fixed-size blocks via a free list: fixed size means the
// pool cannot fragment, so its failure mode is clean exhaustion, never "memory exists but
// not contiguous".
// The pool OWNS the memory. A KV block has multiple concurrent referents (owning
// sequence's block table, prefix cache, agentic pin set); each holds a BlockRef, and a
// block returns to the free list only when its last ref is destroyed, so cancellation
// and error paths unwind instead of being special cases. BlockRef is move-only, so every
// extra referent is a visible share() call, not an accidental copy.

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

    // Relinquish the handle WITHOUT dropping the reference, returning the id: converts a
    // tracked ref into an untracked one that must later be dropped through the pool's raw
    // API. Migration scaffolding only (A7 step 3); nothing new should use it.
    [[nodiscard]] int release();

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

    // Id space only: the pool owns block ids and refcounts, the CALLER owns the memory and
    // computes its own addresses (block() returns an empty span). Needed because the KV
    // cache's pool is layer-major: one block id's bytes scatter across per-layer K/V regions
    // of differing size (Gemma-4 dual geometry, SWA trailing-window groups), which a uniform
    // stride cannot express.
    [[nodiscard]] MemError open_slots(int num_blocks);

    // Add ids above the current range, for a pool whose memory grew underneath it.
    // Slot-only pools only: a backed pool owns its Region and would have to grow that too.
    // Existing ids and refcounts are untouched, so growth is invisible to every outstanding
    // BlockRef.
    [[nodiscard]] MemError grow_slots(int new_num_blocks);

    // Release the region. Asserts (debug) that no refs are outstanding — a
    // non-zero count here is exactly the "block outlived its request" bug.
    void close();

    bool is_open() const;

    // Take a free block with refcount 1. Invalid BlockRef when exhausted.
    [[nodiscard]] BlockRef acquire();

    // Take a tracked reference to a block that already has at least one
    // holder — the prefix-reuse case where a second sequence shares a block a
    // live sequence is still using. Invalid handle if the block is free.
    [[nodiscard]] BlockRef share_by_id(int id);

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
    int ref_count(int id) const;

    // Untracked, id-based refcounting: backs KVCache's int API (allocate_block/free_block/
    // inc_ref), exercised directly by its own tests. Every OWNER above this layer holds a
    // BlockRef; these are for the id-based surface only.
    void acquire_raw(int id);
    void release_raw(int id);
    // Close without the outstanding-ref check, for an owner whose referents
    // hold untracked refs — a bare KVCache used through the int API.
    void abandon();

private:
    friend class BlockRef;
    void inc_ref_(int id);
    void dec_ref_(int id);
    void dec_ref_impl_(int id, bool strict);
    void init_slots_(int num_blocks);  // caller holds mu_

    mutable std::mutex mu_;
    Region region_;      // empty in open_slots() mode — the caller owns the memory
    bool open_ = false;
    size_t block_bytes_ = 0;
    int num_blocks_ = 0;
    std::vector<int> refcount_;   // per block id
    std::vector<int> free_list_;  // ids with refcount 0
};

}  // namespace imp
