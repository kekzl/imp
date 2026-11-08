#pragma once

#include <cuda_runtime.h>

#include "memory/kv_chain_link.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>

namespace imp {

class KVCache;

// One KV block across every layer as host bytes: per layer K, V, then K/V scales when the dtype
// has them, then the sparse key min/max page metadata when that pool exists. False when the cache
// cannot round-trip a block: an SWA layer (separate id space).
[[nodiscard]] bool kv_host_spill_supported(KVCache& cache);
size_t kv_host_block_bytes(KVCache& cache);
// Async on stream; the caller synchronizes before the block or the host bytes are reused.
// False: a copy failed to enqueue (the rest were still enqueued).
[[nodiscard]] bool kv_block_to_host(KVCache& cache, int block_id, void* dst, cudaStream_t stream);
[[nodiscard]] bool kv_block_from_host(KVCache& cache, const void* src, int block_id, cudaStream_t stream);

// Host-RAM tier for evicted prefix-cache KV blocks (#2203): a reclaimed block's KV is copied into
// a pinned slot keyed by its binding (slot_key); a later prefix hit on that hash copies it back instead of
// re-prefilling. Fixed slot count, LRU replacement. Host logic only: the byte copies are the
// caller's (KVCache::copy_block_to_host / copy_block_from_host).
class KVHostSpill {
public:
    using Alloc = std::function<void*(size_t)>;
    using Free = std::function<void(void*)>;

    // slot_bytes: one block across all layers (KVCache::host_spill_block_bytes()). Slot count =
    // budget_bytes / slot_bytes; 0 slots = disabled. alloc/free default to cudaHostAlloc/cudaFreeHost.
    KVHostSpill(size_t budget_bytes, size_t slot_bytes, Alloc alloc = {}, Free free = {});
    ~KVHostSpill();
    KVHostSpill(const KVHostSpill&) = delete;
    KVHostSpill& operator=(const KVHostSpill&) = delete;

    bool enabled() const { return slots_ > 0 && arena_ != nullptr; }
    int slots() const { return slots_; }
    size_t slot_bytes() const { return slot_bytes_; }

    // A slot is keyed by the binding its KV came from: two forwards of one hash (a primary and a
    // shadow, #2409) keep separate slots, so each recurrent snapshot chain (#2174) finds its own.
    static size_t slot_key(const KvChainLink& origin) {
        return origin.hash ^ (origin.serial * 0x9E3779B97F4A7C15ULL);
    }

    // Slot buffer to fill for key (replaces the LRU entry when full, reuses the key's own slot).
    // nullptr when disabled. The entry counts only after commit(key).
    void* reserve(size_t key);
    // origin: the binding whose KV the slot holds; it becomes the newest slot of origin.hash.
    void commit(size_t key, const KvChainLink& origin = {});
    // Undo a reserve() whose copy failed.
    void abandon(size_t key);

    // Slot buffer holding key, nullptr on a miss. Marks it most recently used.
    const void* find(size_t key);
    // Newest committed slot of hash (any forward's KV): false on a miss.
    [[nodiscard]] bool latest_key(size_t hash, size_t* key) const;
    [[nodiscard]] bool contains(size_t hash) const {
        size_t k = 0;
        return latest_key(hash, &k);
    }
    // The binding the newest slot of hash came from; block_id -1 on a miss.
    [[nodiscard]] KvChainLink origin(size_t hash) const;
    [[nodiscard]] KvChainLink origin_of_key(size_t key) const {
        auto it = index_.find(key);
        return it != index_.end() && it->second.committed ? it->second.origin : KvChainLink{};
    }
    // True when a committed slot holds the KV of link's binding.
    [[nodiscard]] bool holds(const KvChainLink& link) const {
        const KvChainLink o = origin_of_key(slot_key(link));
        return o.hash == link.hash && o.block_id == link.block_id && o.serial == link.serial;
    }
    void erase(size_t key);
    void clear();
    int size() const { return static_cast<int>(index_.size()); }

    uint64_t saves() const { return saves_.load(std::memory_order_relaxed); }
    uint64_t restores() const { return restores_.load(std::memory_order_relaxed); }
    uint64_t replacements() const { return replacements_.load(std::memory_order_relaxed); }
    void count_restore() { restores_.fetch_add(1, std::memory_order_relaxed); }

private:
    struct Entry {
        int slot;
        bool committed;
        std::list<size_t>::iterator lru;
        KvChainLink origin{};
    };
    char* slot_ptr(int slot) const { return arena_ + (static_cast<size_t>(slot) * slot_bytes_); }
    // Drops the hash -> newest-slot alias when it names key.
    void drop_alias_(size_t key, const Entry& e);

    size_t slot_bytes_ = 0;
    int slots_ = 0;
    char* arena_ = nullptr;
    Free free_;
    std::vector<int> free_slots_;
    std::list<size_t> lru_;  // front = least recently used
    std::unordered_map<size_t, Entry> index_;
    std::unordered_map<size_t, size_t> latest_;  // hash -> key of its newest committed slot
    // Read by /metrics while the engine thread writes them.
    std::atomic<uint64_t> saves_{0}, restores_{0}, replacements_{0};
};

}  // namespace imp
