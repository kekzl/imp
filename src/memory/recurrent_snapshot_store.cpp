#include "memory/recurrent_snapshot_store.h"
#include "core/logging.h"

namespace imp {

RecurrentSnapshotStore::~RecurrentSnapshotStore() {
    if (!pool_)
        return;
    std::vector<void*> to_free;
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->shutdown = true;
        to_free.swap(pool_->free_bufs);
    }
    for (void* p : to_free)
        cudaFree(p);
    // Entries still held by requests free their buffers via the deleter.
}

void RecurrentSnapshotStore::init(size_t entry_bytes, size_t budget_bytes) {
    entry_bytes_ = entry_bytes;
    int want = (entry_bytes > 0) ? static_cast<int>(budget_bytes / entry_bytes) : 0;
    if (want <= 0) {
        IMP_LOG_INFO("RecurrentSnapshotStore: disabled (budget %.0f MiB < one %.1f MiB slot)",
                     budget_bytes / (1024.0 * 1024.0), entry_bytes / (1024.0 * 1024.0));
        return;
    }
    // Allocate the buffers EAGERLY: engine init sizes the weight caches, KV
    // clamp and workspace pools to fill the card (and the async mempool
    // retains the slack), so at serving time free VRAM is ~0 by design —
    // lazy allocation would never get a byte on tight models. Claiming the
    // budget here makes the downstream sizing account for it; a failed
    // malloc just caps the slot count.
    pool_ = std::make_shared<BufferPool>();
    for (int i = 0; i < want; ++i) {
        void* p = nullptr;
        if (cudaMalloc(&p, entry_bytes_) != cudaSuccess)
            break;
        pool_->free_bufs.push_back(p);
        allocated_bufs_++;
    }
    capacity_ = allocated_bufs_;
    if (capacity_ == 0)
        pool_.reset();
    IMP_LOG_INFO("RecurrentSnapshotStore: %d/%d slots x %.1f MiB pre-allocated (budget %.0f MiB)",
                 capacity_, want, entry_bytes / (1024.0 * 1024.0), budget_bytes / (1024.0 * 1024.0));
}

std::shared_ptr<const RecurrentSnapshotEntry> RecurrentSnapshotStore::find(size_t key) {
    auto it = entries_.find(key);
    if (it == entries_.end())
        return nullptr;
    auto lit = lru_map_.find(key);
    if (lit != lru_map_.end()) {
        lru_.erase(lit->second);
        lru_.push_back(key);
        lit->second = std::prev(lru_.end());
    }
    return it->second;
}

void* RecurrentSnapshotStore::acquire_buffer_() {
    {
        std::lock_guard<std::mutex> lk(pool_->mu);
        if (!pool_->free_bufs.empty()) {
            void* p = pool_->free_bufs.back();
            pool_->free_bufs.pop_back();
            return p;
        }
    }
    // All buffers were pre-allocated at init. None free: evict LRU-head
    // entries until a buffer comes back. An
    // entry whose buffer is still held by a request releases it later —
    // keep evicting map entries until one deleter actually recycles.
    while (!lru_.empty()) {
        size_t victim = lru_.front();
        lru_.pop_front();
        lru_map_.erase(victim);
        entries_.erase(victim);  // may run the deleter right here
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
    if (entries_.count(key) != 0)
        return true;  // identical prefix already snapshotted
    void* buf = acquire_buffer_();
    if (!buf)
        return false;
    if (cudaMemcpyAsync(buf, src, entry_bytes_, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        std::lock_guard<std::mutex> lk(pool_->mu);
        pool_->free_bufs.push_back(buf);
        return false;
    }
    auto pool = pool_;
    auto entry = std::shared_ptr<RecurrentSnapshotEntry>(new RecurrentSnapshotEntry{key, n_tokens, buf},
                                                         [pool](RecurrentSnapshotEntry* e) {
                                                             {
                                                                 std::lock_guard<std::mutex> lk(pool->mu);
                                                                 if (!pool->shutdown) {
                                                                     pool->free_bufs.push_back(e->data);
                                                                     delete e;
                                                                     return;
                                                                 }
                                                             }
                                                             cudaFree(e->data);
                                                             delete e;
                                                         });
    entries_[key] = entry;
    lru_.push_back(key);
    lru_map_[key] = std::prev(lru_.end());
    return true;
}

void RecurrentSnapshotStore::clear() {
    entries_.clear();
    lru_.clear();
    lru_map_.clear();
}

}  // namespace imp
