#include "memory/recurrent_snapshot_store.h"
#include "memory/vram_query.h"
#include "memory/device_alloc_all.h"
#include "memory/backend.h"
#include "memory/host_pinned.h"
#include "core/logging.h"
#include "memory/mem_account.h"
#include "memory/weight_snapshot.h"

namespace imp {

RecurrentSnapshotStore::~RecurrentSnapshotStore() {
    if (!pool_)
        return;
    stop_host_refill_();
    MemAccount::instance().note("recurrent_snapshots",
                                -static_cast<std::ptrdiff_t>(static_cast<size_t>(allocated_bufs_) *
                                                             buf_bytes_));
    std::vector<void*> to_free, to_free_host;
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->shutdown = true;
        to_free.swap(pool_->free_bufs);
        to_free_host.swap(pool_->free_host_bufs);
    }
    for (void* p : to_free)
        device_free_all(p);
    for (void* p : to_free_host) {
        host_pinned_forget(p);
        IMP_CUDA_CHECK_LOG(cudaFreeHost(p));
    }
    // Entries still held by requests free their buffers via the deleter.
}

void RecurrentSnapshotStore::init(size_t entry_bytes, size_t budget_bytes, size_t host_budget_bytes,
                                  size_t sidecar_bytes) {
    entry_bytes_ = entry_bytes;
    sidecar_bytes_ = sidecar_bytes;
    buf_bytes_ = entry_bytes + sidecar_bytes;
    int want = (entry_bytes > 0) ? static_cast<int>(budget_bytes / buf_bytes_) : 0;
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
        if (device_alloc_all(dev_req(p, buf_bytes_)) != cudaSuccess)
            break;
        pool_->free_bufs.push_back(p);
        allocated_bufs_++;
    }
    capacity_ = allocated_bufs_;
    // The plan charges these slots (recurrent_snapshot_bytes); unnoted they also read as library
    // reserve, charged a second time on the next start (Qwen3.6-35B-A3B GGUF server: 4 x 63.8 MiB).
    MemAccount::instance().note("recurrent_snapshots",
                                static_cast<std::ptrdiff_t>(static_cast<size_t>(allocated_bufs_) *
                                                            buf_bytes_));
    if (capacity_ == 0) {
        // Was a silent `return`. The caller reads enabled() == false and turns hybrid prefix
        // caching off with a message about the BUDGET, which was not the problem: the budget
        // bought `want` slots and the device refused every one of them.
        size_t free_bytes = 0, total_device = 0;
        (void)vram_diag_mem_info(&free_bytes, &total_device);
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
    // Host tier: pinned, so the eviction D2H and the restore H2D run as stream-ordered async copies.
    // A slab is pinned when a demotion first needs it (103 slabs up front held a start for 10 s).
    host_capacity_ = static_cast<int>(host_budget_bytes / buf_bytes_);
    if (host_capacity_ <= 0)
        return;
    IMP_LOG_INFO(
        "RecurrentSnapshotStore: host tier up to %d slots x %.1f MiB, %d pinned ahead by a worker "
        "(budget %.0f MiB)",
        host_capacity_.load(), entry_bytes / (1024.0 * 1024.0), host_spare_,
        host_budget_bytes / (1024.0 * 1024.0));
    if (host_spare_ > 0 && cudaGetDevice(&refill_device_) == cudaSuccess)
        host_refill_ = std::thread(&RecurrentSnapshotStore::host_refill_loop_, this);
}

int RecurrentSnapshotStore::host_slabs_ready() const {
    if (!pool_)
        return 0;
    std::lock_guard<std::mutex> lk(pool_->mu);
    return static_cast<int>(pool_->free_host_bufs.size());
}

void* RecurrentSnapshotStore::pin_host_slab_() {
    if (host_allocated_.load() >= host_capacity_.load())
        return nullptr;
    void* p = nullptr;
    if (host_ram_spares(buf_bytes_) && cudaHostAlloc(&p, buf_bytes_, cudaHostAllocDefault) == cudaSuccess) {
        ++host_allocated_;
        host_pinned_note(p, buf_bytes_, "recurrent_snapshots");
        return p;
    }
    (void)cudaGetLastError();
    IMP_LOG_WARN(
        "RecurrentSnapshotStore: the host refused slab %d of %d (%.1f MiB each), the host tier "
        "stays at %d; fewer snapshots means more full re-prefills",
        host_allocated_.load() + 1, host_capacity_.load(), buf_bytes_ / (1024.0 * 1024.0),
        host_allocated_.load());
    host_capacity_ = host_allocated_.load();
    return nullptr;
}

// Keeps host_spare_ free pinned slabs while the tier is below its ceiling. Pinning 79.5 MiB is a
// page-locking call: here it runs beside the requests, not inside a demotion.
void RecurrentSnapshotStore::host_refill_loop_() {
    if (cudaSetDevice(refill_device_) != cudaSuccess)
        return;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(pool_->mu);
            pool_->cv.wait(lk, [this] {
                return pool_->refill_stop || (static_cast<int>(pool_->free_host_bufs.size()) < host_spare_ &&
                                              host_allocated_.load() < host_capacity_.load());
            });
            if (pool_->refill_stop)
                return;
        }
        void* p = nullptr;
        {
            PlannedHostAllocScope planned;
            std::lock_guard<std::mutex> pl(pin_mu_);
            p = pin_host_slab_();
        }
        if (!p)
            continue;  // at the ceiling now: the predicate holds the worker until a slab is released
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->free_host_bufs.push_back(p);
    }
}

void RecurrentSnapshotStore::stop_host_refill_() {
    if (!host_refill_.joinable())
        return;
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->refill_stop = true;
    }
    pool_->cv.notify_all();
    host_refill_.join();
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

std::shared_ptr<RecurrentSnapshotEntry> RecurrentSnapshotStore::make_entry_(
    size_t key, int n_tokens, void* buf, bool on_host, std::vector<KvChainLink> kv_chain) {
    auto pool = pool_;
    return std::shared_ptr<RecurrentSnapshotEntry>(
        new RecurrentSnapshotEntry{key, n_tokens, buf, on_host, std::move(kv_chain)},
        [pool](RecurrentSnapshotEntry* e) {
            {
                std::lock_guard<std::mutex> lk(pool->mu);
                if (!pool->shutdown) {
                    (e->on_host ? pool->free_host_bufs : pool->free_bufs).push_back(e->data);
                    delete e;
                    return;
                }
            }
            if (e->on_host) {
                host_pinned_forget(e->data);
                IMP_CUDA_CHECK_LOG(cudaFreeHost(e->data));
            } else
                device_free_all(e->data);
            delete e;
        });
}

void* RecurrentSnapshotStore::acquire_host_buffer_() {
    const auto take_ready = [this]() -> void* {
        std::lock_guard<std::mutex> lk(pool_->mu);
        if (pool_->free_host_bufs.empty())
            return nullptr;
        void* p = pool_->free_host_bufs.back();
        pool_->free_host_bufs.pop_back();
        return p;
    };
    void* p = take_ready();
    if (!p) {
        // The worker was outrun (or is off): pin here, after a pin of its own that may be in flight.
        std::lock_guard<std::mutex> pl(pin_mu_);
        p = take_ready();
        if (!p) {
            p = pin_host_slab_();
            host_inline_pins_.fetch_add(p != nullptr ? 1 : 0, std::memory_order_relaxed);
        }
    }
    pool_->cv.notify_one();
    if (p)
        return p;
    // Protected keys rotate to the tail; one pass over the list bounds the walk.
    for (size_t n = host_lru_.size(); n > 0 && !host_lru_.empty(); --n) {
        size_t victim = host_lru_.front();
        if (is_protected(victim)) {
            host_lru_.splice(host_lru_.end(), host_lru_, host_lru_.begin());
            continue;
        }
        host_lru_.pop_front();
        host_lru_map_.erase(victim);
        host_entries_.erase(victim);  // may run the deleter right here
        ++evictions_;
        if (void* freed = take_ready())
            return freed;
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
    if (host_capacity_ > 0 && !host_entries_.contains(victim)) {
        void* hbuf = acquire_host_buffer_();
        if (hbuf &&
            cudaMemcpyAsync(hbuf, dev->data, buf_bytes_, cudaMemcpyDeviceToHost, stream) == cudaSuccess) {
            host_entries_[victim] = make_entry_(victim, dev->n_tokens, hbuf, /*on_host=*/true, dev->kv_chain);
            host_lru_.push_back(victim);
            host_lru_map_[victim] = std::prev(host_lru_.end());
            ++demotions_;
            return;
        }
        if (hbuf) {
            std::lock_guard<std::mutex> lk(pool_->mu);
            pool_->free_host_bufs.push_back(hbuf);
        }
    }
    if (!host_entries_.contains(victim))
        ++evictions_;
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

bool RecurrentSnapshotStore::copy_in_(void* buf, const void* src, const void* sidecar_src, cudaMemcpyKind kind,
                                      cudaStream_t stream) const {
    if (cudaMemcpyAsync(buf, src, entry_bytes_, kind, stream) != cudaSuccess)
        return false;
    return sidecar_bytes_ == 0 || cudaMemcpyAsync(static_cast<char*>(buf) + entry_bytes_, sidecar_src,
                                                  sidecar_bytes_, kind, stream) == cudaSuccess;
}

bool RecurrentSnapshotStore::save(size_t key, int n_tokens, const void* src, cudaStream_t stream,
                                  const void* sidecar_src, std::vector<KvChainLink> kv_chain) {
    if (!enabled() || src == nullptr || n_tokens <= 0 || (sidecar_bytes_ > 0 && sidecar_src == nullptr))
        return false;
    if (entries_.contains(key) || host_entries_.contains(key))
        return true;  // identical prefix already snapshotted (either tier)
    void* buf = acquire_buffer_(stream);
    if (!buf)
        return save_to_host_(key, n_tokens, src, sidecar_src, stream, std::move(kv_chain));
    if (!copy_in_(buf, src, sidecar_src, cudaMemcpyDeviceToDevice, stream)) {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->free_bufs.push_back(buf);
        return false;
    }
    entries_[key] = make_entry_(key, n_tokens, buf, /*on_host=*/false, std::move(kv_chain));
    lru_.push_back(key);
    lru_map_[key] = std::prev(lru_.end());
    ++saves_;
    return true;
}

// Every device slab can be held by an in-flight restore (the steady state of concurrent
// multi-turn sessions); dropping the save in that case cost every session its next-turn
// prefix, since the host tier had free slots the whole time. Save straight into the host
// tier instead: the same one-slab D2H an eviction issues, and find() already serves host
// entries.
bool RecurrentSnapshotStore::save_to_host_(size_t key, int n_tokens, const void* src, const void* sidecar_src,
                                           cudaStream_t stream, std::vector<KvChainLink> kv_chain) {
    void* hbuf = host_capacity_ > 0 ? acquire_host_buffer_() : nullptr;
    if (!hbuf) {
        ++dropped_saves_;
        if (dropped_saves_ == 1)
            IMP_LOG_WARN(
                "RecurrentSnapshotStore: snapshot dropped, every device slab is held by an "
                "in-flight request and no host slab is free (%d device / %d host slots); the "
                "next turn of this session prefills its whole history",
                capacity_, host_capacity_.load());
        return false;
    }
    if (!copy_in_(hbuf, src, sidecar_src, cudaMemcpyDeviceToHost, stream)) {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->free_host_bufs.push_back(hbuf);
        return false;
    }
    host_entries_[key] = make_entry_(key, n_tokens, hbuf, /*on_host=*/true, std::move(kv_chain));
    host_lru_.push_back(key);
    host_lru_map_[key] = std::prev(host_lru_.end());
    ++host_direct_saves_;
    ++saves_;
    return true;
}

bool RecurrentSnapshotStore::save_gather(size_t key, int n_tokens, std::span<const SnapshotPiece> pieces,
                                         cudaStream_t stream, std::vector<KvChainLink> kv_chain) {
    if (!enabled() || n_tokens <= 0 || pieces.empty())
        return false;
    for (const SnapshotPiece& p : pieces)
        if (p.src == nullptr || p.off > buf_bytes_ || p.bytes > buf_bytes_ - p.off)
            return false;
    if (contains(key))
        return true;
    void* buf = acquire_buffer_(stream);
    const bool on_host = buf == nullptr;
    if (on_host && host_capacity_ > 0)
        buf = acquire_host_buffer_();
    if (!buf) {
        ++dropped_saves_;
        return false;
    }
    for (const SnapshotPiece& p : pieces)
        if (cudaMemcpyAsync(static_cast<char*>(buf) + p.off, p.src, p.bytes, cudaMemcpyDefault, stream) !=
            cudaSuccess) {
            std::lock_guard<std::mutex> lk(pool_->mu);
            (on_host ? pool_->free_host_bufs : pool_->free_bufs).push_back(buf);
            return false;
        }
    auto& map = on_host ? host_entries_ : entries_;
    auto& lru = on_host ? host_lru_ : lru_;
    auto& lru_map = on_host ? host_lru_map_ : lru_map_;
    map[key] = make_entry_(key, n_tokens, buf, on_host, std::move(kv_chain));
    lru.push_back(key);
    lru_map[key] = std::prev(lru.end());
    host_direct_saves_ += on_host ? 1 : 0;
    ++saves_;
    return true;
}

void RecurrentSnapshotStore::erase(size_t key) {
    if (auto lit = lru_map_.find(key); lit != lru_map_.end()) {
        lru_.erase(lit->second);
        lru_map_.erase(lit);
    }
    entries_.erase(key);
    if (auto lit = host_lru_map_.find(key); lit != host_lru_map_.end()) {
        host_lru_.erase(lit->second);
        host_lru_map_.erase(lit);
    }
    host_entries_.erase(key);
}

void RecurrentSnapshotStore::unprotect(int owner) {
    auto it = protected_.find(owner);
    if (it == protected_.end())
        return;
    const size_t key = it->second;
    protected_.erase(it);
    (void)find(key);  // MRU: the owner's next turn is the likeliest next restore
}

bool RecurrentSnapshotStore::is_protected(size_t key) const {
    for (const auto& [owner, k] : protected_)
        if (k == key)
            return true;
    return false;
}

void RecurrentSnapshotStore::clear() {
    entries_.clear();
    lru_.clear();
    lru_map_.clear();
    host_entries_.clear();
    host_lru_.clear();
    host_lru_map_.clear();
}

int RecurrentSnapshotStore::release_host_tier() {
    clear();
    if (!pool_)
        return 0;
    std::vector<void*> slabs;
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        slabs.swap(pool_->free_host_bufs);
    }
    for (void* p : slabs) {
        host_pinned_forget(p);
        IMP_CUDA_CHECK_LOG(cudaFreeHost(p));
    }
    // A slab a request still holds stays counted and returns to the pool with its entry.
    host_allocated_ -= static_cast<int>(slabs.size());
    pool_->cv.notify_one();  // the worker pins its spare again, beside the requests
    return static_cast<int>(slabs.size());
}

void log_swa_snapshot_store(const RecurrentSnapshotStore& store, int budget_mb) {
    const size_t kib = store.entry_bytes() >> 10;
    IMP_LOG_INFO("SWA snapshots: %d MiB budget, %zu KiB/snapshot, capacity %d", budget_mb, kib,
                 store.capacity());
    if (store.capacity() < 2)
        IMP_LOG_WARN(
            "SWA snapshots: capacity %d restores nothing across sessions; set "
            "kv_cache.swa_snapshot_mb >= %zu (2 snapshots per concurrent session)",
            store.capacity(), 2 * (kib >> 10));
}

}  // namespace imp
