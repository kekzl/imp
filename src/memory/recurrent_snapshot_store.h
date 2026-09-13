#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace imp {

// One stored recurrent-state snapshot: the full per-sequence SSM/GDN state slab as it
// was after prefilling exactly n_tokens tokens. key is the chained KV block hash of
// those tokens (n_tokens is always a multiple of the KV block size, so the key
// identifies the byte-exact token prefix).
struct RecurrentSnapshotEntry {
    size_t key = 0;
    int n_tokens = 0;
    void* data = nullptr;  // entry_bytes: a device buffer, or pinned host memory when on_host
    bool on_host = false;  // host-tier copy: restore with cudaMemcpyDefault (pinned H2D)
};

// Device-side LRU store of recurrent-state snapshots for hybrid (SSM/GDN) models. Dense
// models reuse KV blocks at block granularity; recurrent state is cumulative, so a
// prefix can only be skipped when the state at exactly that boundary was saved. The
// engine saves one snapshot per prefill (largest block-aligned position) and restores
// the longest match on admission, turning multi-turn full-history re-prefill into a
// tail-only prefill.
// Threading: save/find/clear run on the engine worker thread; entries are shared_ptr, a
// Request may hold one across steps and release it from another thread, so buffer
// recycling goes through a mutex-guarded pool shared with the entry deleters. An evicted
// entry's buffer recycles only after the last holder releases it.
// Host tier (server.recurrent_snapshot_host_mb): the device tier is only a few slots, so
// with more concurrent multi-turn sessions than slots every snapshot would be evicted
// before its next turn. An entry evicted from the device tier is copied into pinned host
// memory instead of dropped (stream-ordered D2H); find() serves host-tier entries
// (on_host = true), restored via cudaMemcpyDefault. Host entries are never promoted back
// (a restore is one H2D of one slab). A save finding every device slab held by an
// in-flight restore goes straight into the host tier (save_to_host_).
class RecurrentSnapshotStore {
public:
    ~RecurrentSnapshotStore();

    // entry_bytes: size of one snapshot (the SSM per-seq slab).
    // budget_bytes: total device memory cap; capacity = budget / entry_bytes.
    // host_budget_bytes: pinned host memory cap for evicted entries (0 = off).
    void init(size_t entry_bytes, size_t budget_bytes, size_t host_budget_bytes = 0);

    bool enabled() const { return capacity_ > 0; }
    size_t entry_bytes() const { return entry_bytes_; }
    int capacity() const { return capacity_; }
    int host_capacity() const { return host_capacity_; }
    int size() const { return static_cast<int>(entries_.size()); }
    int host_size() const { return static_cast<int>(host_entries_.size()); }

    bool contains(size_t key) const { return entries_.count(key) != 0 || host_entries_.count(key) != 0; }

    // Look up a snapshot by key and mark it most-recently-used.
    std::shared_ptr<const RecurrentSnapshotEntry> find(size_t key);

        // Copy entry_bytes from `src` (device) into the store on `stream`.
    // Evicts the LRU entry if at capacity (into the host tier when configured, D2H on
    // `stream`). When every device slab is held by an in-flight request, the save goes
    // straight into the host tier instead. Returns false only when no slab of either tier is
    // free or on a copy failure.
    bool save(size_t key, int n_tokens, const void* src, cudaStream_t stream);
    // Saves that landed in the host tier because no device slab was free, and
    // saves dropped because no slab of either tier was.
    int host_direct_saves() const { return host_direct_saves_; }
    int dropped_saves() const { return dropped_saves_; }

    // Drop all entries (context reset). Outstanding request-held entries
    // stay valid until released.
    void clear();

private:
    // Shared with entry deleters so buffers outlive the store if needed.
        struct BufferPool {
        std::mutex mu;
        std::vector<void*> free_bufs;       // device slabs
        std::vector<void*> free_host_bufs;  // pinned host slabs
        bool shutdown = false;              // deleters free instead of recycling
    };

    void* acquire_buffer_(cudaStream_t stream);
    void* acquire_host_buffer_();
    void evict_device_lru_(cudaStream_t stream);
    bool save_to_host_(size_t key, int n_tokens, const void* src, cudaStream_t stream);
    int host_direct_saves_ = 0;
    int dropped_saves_ = 0;
    std::shared_ptr<RecurrentSnapshotEntry> make_entry_(size_t key, int n_tokens, void* buf, bool on_host);

    std::shared_ptr<BufferPool> pool_;
    size_t entry_bytes_ = 0;
    int capacity_ = 0;
    int host_capacity_ = 0;
    int allocated_bufs_ = 0;  // pre-allocated at init (== capacity_)

    std::unordered_map<size_t, std::shared_ptr<RecurrentSnapshotEntry>> entries_;
    // LRU: most-recently-used at the tail.
    std::list<size_t> lru_;
    std::unordered_map<size_t, std::list<size_t>::iterator> lru_map_;
    // Host tier, same bookkeeping.
    std::unordered_map<size_t, std::shared_ptr<RecurrentSnapshotEntry>> host_entries_;
    std::list<size_t> host_lru_;
    std::unordered_map<size_t, std::list<size_t>::iterator> host_lru_map_;
};

}  // namespace imp
