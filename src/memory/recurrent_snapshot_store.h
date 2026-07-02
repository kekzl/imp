#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace imp {

// One stored recurrent-state snapshot: the full per-sequence SSM/GDN state
// slab as it was after prefilling exactly `n_tokens` tokens. `key` is the
// chained KV block hash of those tokens (n_tokens is always a multiple of
// the KV block size, so the key identifies the byte-exact token prefix).
struct RecurrentSnapshotEntry {
    size_t key = 0;
    int n_tokens = 0;
    void* data = nullptr;  // device buffer of entry_bytes
};

// Device-side LRU store of recurrent-state snapshots for hybrid (SSM/GDN)
// models. Dense models reuse KV blocks at block granularity; recurrent state
// is cumulative, so a prefix can only be skipped when the state at exactly
// that boundary was saved. The engine saves one snapshot per prefill (at the
// largest block-aligned prompt position) and restores the longest match on
// admission, turning multi-turn full-history re-prefill into a tail-only
// prefill.
//
// Threading: save/find/clear run on the engine worker thread. Entries are
// handed out as shared_ptr; a Request may hold one across steps and release
// it from another thread (server request teardown), so buffer recycling goes
// through a mutex-guarded pool shared with the entry deleters. An evicted
// entry's buffer is recycled only after the last holder releases it — an
// in-flight restore can never read a reused buffer.
class RecurrentSnapshotStore {
public:
    ~RecurrentSnapshotStore();

    // entry_bytes: size of one snapshot (the SSM per-seq slab).
    // budget_bytes: total device memory cap; capacity = budget / entry_bytes.
    void init(size_t entry_bytes, size_t budget_bytes);

    bool enabled() const { return capacity_ > 0; }
    size_t entry_bytes() const { return entry_bytes_; }
    int capacity() const { return capacity_; }
    int size() const { return static_cast<int>(entries_.size()); }

    bool contains(size_t key) const { return entries_.count(key) != 0; }

    // Look up a snapshot by key and mark it most-recently-used.
    std::shared_ptr<const RecurrentSnapshotEntry> find(size_t key);

    // Copy entry_bytes from `src` (device) into the store on `stream`.
    // Evicts the LRU entry if at capacity. Returns false when no buffer is
    // available (all entries held by in-flight requests) or on alloc failure.
    bool save(size_t key, int n_tokens, const void* src, cudaStream_t stream);

    // Drop all entries (context reset). Outstanding request-held entries
    // stay valid until released.
    void clear();

private:
    // Shared with entry deleters so buffers outlive the store if needed.
    struct BufferPool {
        std::mutex mu;
        std::vector<void*> free_bufs;
        bool shutdown = false;  // deleters cudaFree instead of recycling
    };

    void* acquire_buffer_();

    std::shared_ptr<BufferPool> pool_;
    size_t entry_bytes_ = 0;
    int capacity_ = 0;
    int allocated_bufs_ = 0;  // pre-allocated at init (== capacity_)

    std::unordered_map<size_t, std::shared_ptr<RecurrentSnapshotEntry>> entries_;
    // LRU: most-recently-used at the tail.
    std::list<size_t> lru_;
    std::unordered_map<size_t, std::list<size_t>::iterator> lru_map_;
};

}  // namespace imp
