#include "memory/recurrent_snapshot_store.h"
#include "core/logging.h"

namespace imp {

RecurrentSnapshotStore::~RecurrentSnapshotStore() {
    if (!pool_)
        return;
    std::vector<void*> to_free, to_free_host;
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->shutdown = true;
        to_free.swap(pool_->free_bufs);
        to_free_host.swap(pool_->free_host_bufs);
    }
    for (void* p : to_free)
        cudaFree(p);
    for (void* p : to_free_host)
        cudaFreeHost(p);
    // Entries still held by requests free their buffers via the deleter.
}

void RecurrentSnapshotStore::init(size_t entry_bytes, size_t budget_bytes, size_t host_budget_bytes) {
    entry_bytes_ = entry_bytes;
    int want = (entry_bytes > 0) ? static_cast<int>(budget_bytes / entry_bytes) : 0;
    if (want <= 0) {
        IMP_LOG_INFO("RecurrentSnapshotStore: disabled (budget %.0f MiB < one %.1f MiB slot)",
                     budget_bytes / (1024.0 * 1024.0), entry_bytes / (1024.0 * 1024.0));
        return;
    }
    // Allocate the buffers EAGERLY: engine init sizes weight caches, KV clamp and workspace
    // pools to fill the card (the async mempool retains the slack), so at serving time free
    // VRAM is ~0 by design; lazy allocation would never get a byte on tight models. Claiming
    // the budget here makes downstream sizing account for it; a failed malloc just caps the
    // slot count.
    pool_ = std::make_shared<BufferPool>();
    for (int i = 0; i < want; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, entry_bytes_) != cudaSuccess)
            break;
        pool_->free_bufs.push_back(p);
        allocated_bufs_++;
    }
    capacity_ = allocated_bufs_;
    if (capacity_ == 0) {
        // Was a silent `return`. The caller reads enabled() == false and turns hybrid prefix
        // caching off with a message about the BUDGET, which was not the problem: the budget
        // bought `want` slots and the device refused every one of them.
        size_t free_bytes = 0, total_device = 0;
        if (cudaMemGetInfo(&free_bytes, &total_device) != cudaSuccess)
            free_bytes = 0;
        IMP_LOG_ERROR(
            "RecurrentSnapshotStore: 0/%d slots x %.1f MiB allocated (budget %.0f MiB, %.0f MiB "
            "free) - hybrid prefix caching stays OFF; every turn re-prefills its whole prompt",
            want, entry_bytes / (1024.0 * 1024.0), budget_bytes / (1024.0 * 1024.0),
            free_bytes / (1024.0 * 1024.0));
        pool_.reset();
        return;
    }
    if (capacity_ < want) {
        // A partial store is not a failure, but it is not what was configured
        // either: the hit rate is set by the slot count, and 6/8 turn-2 hits
        // came from exactly this shortfall once before (#1937/#1938).
        IMP_LOG_WARN(
            "RecurrentSnapshotStore: only %d of %d slots x %.1f MiB allocated (budget %.0f MiB) - "
            "the device refused the rest; fewer snapshots means more full re-prefills",
            capacity_, want, entry_bytes / (1024.0 * 1024.0), budget_bytes / (1024.0 * 1024.0));
    }
    IMP_LOG_INFO("RecurrentSnapshotStore: %d/%d slots x %.1f MiB pre-allocated (budget %.0f MiB)",
                 capacity_, want, entry_bytes / (1024.0 * 1024.0), budget_bytes / (1024.0 * 1024.0));
    // Host tier: pinned, so the eviction D2H and the restore H2D run as
    // stream-ordered async copies. Same eager policy, a failed pin caps it.
    const int want_host = static_cast<int>(host_budget_bytes / entry_bytes);
    for (int i = 0; i < want_host; ++i) {
        void* p = nullptr;
        if (cudaHostAlloc(&p, entry_bytes_, cudaHostAllocDefault) != cudaSuccess)
            break;
        pool_->free_host_bufs.push_back(p);
        host_capacity_++;
    }
    if (want_host > 0) {
        IMP_LOG_INFO("RecurrentSnapshotStore: host tier %d/%d slots x %.1f MiB pinned (budget %.0f MiB)",
                     host_capacity_, want_host, entry_bytes / (1024.0 * 1024.0),
                     host_budget_bytes / (1024.0 * 1024.0));
        if (host_capacity_ < want_host)
            IMP_LOG_WARN(
                "RecurrentSnapshotStore: host tier short by %d of %d slots (cudaHostAlloc refused) - "
                "evicted snapshots are dropped instead of demoted",
                want_host - host_capacity_, want_host);
    }
}

std::shared_ptr<const RecurrentSnapshotEntry> RecurrentSnapshotStore::find(size_t key) {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
        auto lit = lru_map_.find(key);
        if (lit != lru_map_.end()) {
            lru_.erase(lit->second);
            lru_.push_back(key);
            lit->second = std::prev(lru_.end());
        }
        return it->second;
    }
    auto hit = host_entries_.find(key);
    if (hit == host_entries_.end())
        return nullptr;
    auto lit = host_lru_map_.find(key);
    if (lit != host_lru_map_.end()) {
        host_lru_.erase(lit->second);
        host_lru_.push_back(key);
        lit->second = std::prev(host_lru_.end());
    }
    return hit->second;
}

std::shared_ptr<RecurrentSnapshotEntry> RecurrentSnapshotStore::make_entry_(size_t key, int n_tokens,
                                                                            void* buf, bool on_host) {
    auto pool = pool_;
    return std::shared_ptr<RecurrentSnapshotEntry>(
        new RecurrentSnapshotEntry{key, n_tokens, buf, on_host}, [pool](RecurrentSnapshotEntry* e) {
            {
                std::lock_guard<std::mutex> lk(pool->mu);
                if (!pool->shutdown) {
                    (e->on_host ? pool->free_host_bufs : pool->free_bufs).push_back(e->data);
                    delete e;
                    return;
                }
            }
            if (e->on_host)
                cudaFreeHost(e->data);
            else
                cudaFree(e->data);
            delete e;
        });
}

void* RecurrentSnapshotStore::acquire_host_buffer_() {
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        if (!pool_->free_host_bufs.empty()) {
            void* p = pool_->free_host_bufs.back();
            pool_->free_host_bufs.pop_back();
            return p;
        }
    }
    while (!host_lru_.empty()) {
        size_t victim = host_lru_.front();
        host_lru_.pop_front();
        host_lru_map_.erase(victim);
        host_entries_.erase(victim);  // may run the deleter right here
        std::lock_guard<std::mutex> lk(pool_->mu);
        if (!pool_->free_host_bufs.empty()) {
            void* p = pool_->free_host_bufs.back();
            pool_->free_host_bufs.pop_back();
            return p;
        }
    }
    return nullptr;
}

// Drop the device LRU-head entry. With a host tier, copy it out first on
// `stream`: every store copy runs on the prefill stream, so the D2H precedes
// any later save into the recycled device buffer.
void RecurrentSnapshotStore::evict_device_lru_(cudaStream_t stream) {
    size_t victim = lru_.front();
    lru_.pop_front();
    lru_map_.erase(victim);
    auto it = entries_.find(victim);
    if (it == entries_.end())
        return;
    std::shared_ptr<RecurrentSnapshotEntry> dev = it->second;
    entries_.erase(it);
    if (host_capacity_ > 0 && host_entries_.count(victim) == 0) {
        void* hbuf = acquire_host_buffer_();
        if (hbuf &&
            cudaMemcpyAsync(hbuf, dev->data, entry_bytes_, cudaMemcpyDeviceToHost, stream) == cudaSuccess) {
            host_entries_[victim] = make_entry_(victim, dev->n_tokens, hbuf, /*on_host=*/true);
            host_lru_.push_back(victim);
            host_lru_map_[victim] = std::prev(host_lru_.end());
        } else if (hbuf) {
            std::lock_guard<std::mutex> lk(pool_->mu);
            pool_->free_host_bufs.push_back(hbuf);
        }
    }
    // `dev` goes out of scope here: its buffer recycles now, or when the last
    // in-flight holder releases it.
}

void* RecurrentSnapshotStore::acquire_buffer_(cudaStream_t stream) {
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        if (!pool_->free_bufs.empty()) {
            void* p = pool_->free_bufs.back();
            pool_->free_bufs.pop_back();
            return p;
        }
    }
    // All buffers were pre-allocated at init, none free: evict LRU-head entries until a
    // buffer comes back. An entry whose buffer is still held by a request releases it later;
    // keep evicting map entries until one deleter actually recycles.
    while (!lru_.empty()) {
        evict_device_lru_(stream);
        std::lock_guard<std::mutex> lk(pool_->mu);
        if (!pool_->free_bufs.empty()) {
            void* p = pool_->free_bufs.back();
            pool_->free_bufs.pop_back();
            return p;
        }
    }
    return nullptr;  // every buffer is held by an in-flight request
}

bool RecurrentSnapshotStore::save(size_t key, int n_tokens, const void* src, cudaStream_t stream) {
    if (!enabled() || src == nullptr || n_tokens <= 0)
        return false;
    if (entries_.count(key) != 0 || host_entries_.count(key) != 0)
        return true;  // identical prefix already snapshotted (either tier)
    void* buf = acquire_buffer_(stream);
    if (!buf)
        return save_to_host_(key, n_tokens, src, stream);
    if (cudaMemcpyAsync(buf, src, entry_bytes_, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->free_bufs.push_back(buf);
        return false;
    }
    entries_[key] = make_entry_(key, n_tokens, buf, /*on_host=*/false);
    lru_.push_back(key);
    lru_map_[key] = std::prev(lru_.end());
    return true;
}

// Every device slab can be held by an in-flight restore (the steady state of concurrent
// multi-turn sessions); dropping the save in that case cost every session its next-turn
// prefix, since the host tier had free slots the whole time. Save straight into the host
// tier instead: the same one-slab D2H an eviction issues, and find() already serves host
// entries.
bool RecurrentSnapshotStore::save_to_host_(size_t key, int n_tokens, const void* src, cudaStream_t stream) {
    void* hbuf = host_capacity_ > 0 ? acquire_host_buffer_() : nullptr;
    if (!hbuf) {
        ++dropped_saves_;
        if (dropped_saves_ == 1)
            IMP_LOG_WARN(
                "RecurrentSnapshotStore: snapshot dropped, every device slab is held by an "
                "in-flight request and no host slab is free (%d device / %d host slots); the "
                "next turn of this session prefills its whole history",
                capacity_, host_capacity_);
        return false;
    }
    if (cudaMemcpyAsync(hbuf, src, entry_bytes_, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->free_host_bufs.push_back(hbuf);
        return false;
    }
    host_entries_[key] = make_entry_(key, n_tokens, hbuf, /*on_host=*/true);
    host_lru_.push_back(key);
    host_lru_map_[key] = std::prev(host_lru_.end());
    ++host_direct_saves_;
    return true;
}

void RecurrentSnapshotStore::clear() {
    entries_.clear();
    lru_.clear();
    lru_map_.clear();
    host_entries_.clear();
    host_lru_.clear();
    host_lru_map_.clear();
}

}  // namespace imp
