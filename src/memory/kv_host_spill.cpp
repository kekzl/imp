#include "memory/kv_host_spill.h"

#include "core/logging.h"
#include "memory/kv_cache.h"

#include <cuda_runtime.h>

namespace imp {

bool kv_host_spill_supported(KVCache& cache) {
    if (cache.key_minmax_enabled())
        return false;
    for (int l = 0; l < cache.n_layers(); ++l)
        if (cache.layer_is_swa(l))
            return false;
    return cache.n_layers() > 0;
}

namespace {

enum class Part { K, V, KScale, VScale };

void* part_ptr(KVCache& c, Part p, int layer, int block_id) {
    switch (p) {
        case Part::K:
            return c.k_ptr(layer, block_id);
        case Part::V:
            return c.v_ptr(layer, block_id);
        case Part::KScale:
            return c.k_scale_ptr(layer, block_id);
        case Part::VScale:
            return c.v_scale_ptr(layer, block_id);
    }
    return nullptr;
}

size_t part_bytes(KVCache& c, Part p, int layer) {
    return (p == Part::K || p == Part::V) ? c.block_bytes(layer) : c.scale_block_bytes(layer);
}

// Host layout: all layers' K, then V, then K scales, then V scales. A part whose layers share one
// size and a constant device stride moves as a single 2D copy (2 to 4 copies per block instead of
// 2-4 per layer); anything else falls back to one copy per layer.
template <typename F2D, typename F1D>
size_t for_each_part(KVCache& c, int block_id, F2D&& copy2d, F1D&& copy1d) {
    const int n = c.n_layers();
    size_t host_off = 0;
    for (Part p : {Part::K, Part::V, Part::KScale, Part::VScale}) {
        if (!part_ptr(c, p, 0, block_id))
            continue;  // no scale planes for this dtype
        const size_t bytes = part_bytes(c, p, 0);
        const auto* first = static_cast<const char*>(part_ptr(c, p, 0, block_id));
        const ptrdiff_t stride = n > 1 ? static_cast<const char*>(part_ptr(c, p, 1, block_id)) - first : 0;
        bool uniform = stride >= static_cast<ptrdiff_t>(bytes) || n == 1;
        for (int l = 1; uniform && l < n; ++l)
            uniform = part_bytes(c, p, l) == bytes &&
                      static_cast<const char*>(part_ptr(c, p, l, block_id)) - first == stride * l;
        if (uniform) {
            copy2d(host_off, const_cast<char*>(first), static_cast<size_t>(stride), bytes, n);
            host_off += bytes * static_cast<size_t>(n);
        } else {
            for (int l = 0; l < n; ++l) {
                const size_t b = part_bytes(c, p, l);
                copy1d(host_off, part_ptr(c, p, l, block_id), b);
                host_off += b;
            }
        }
    }
    return host_off;
}

}  // namespace

size_t kv_host_block_bytes(KVCache& cache) {
    return for_each_part(cache, 0, [](size_t, char*, size_t, size_t, int) {}, [](size_t, void*, size_t) {});
}

void kv_block_to_host(KVCache& cache, int block_id, void* dst, cudaStream_t stream) {
    char* out = static_cast<char*>(dst);
    for_each_part(
        cache, block_id,
        [&](size_t off, char* dev, size_t pitch, size_t width, int rows) {
            cudaMemcpy2DAsync(out + off, width, dev, pitch, width, rows, cudaMemcpyDeviceToHost, stream);
        },
        [&](size_t off, void* dev, size_t bytes) {
            cudaMemcpyAsync(out + off, dev, bytes, cudaMemcpyDeviceToHost, stream);
        });
}

void kv_block_from_host(KVCache& cache, const void* src, int block_id, cudaStream_t stream) {
    const char* in = static_cast<const char*>(src);
    for_each_part(
        cache, block_id,
        [&](size_t off, char* dev, size_t pitch, size_t width, int rows) {
            cudaMemcpy2DAsync(dev, pitch, in + off, width, width, rows, cudaMemcpyHostToDevice, stream);
        },
        [&](size_t off, void* dev, size_t bytes) {
            cudaMemcpyAsync(dev, in + off, bytes, cudaMemcpyHostToDevice, stream);
        });
}

KVHostSpill::KVHostSpill(size_t budget_bytes, size_t slot_bytes, Alloc alloc, Free free)
    : slot_bytes_(slot_bytes), free_(std::move(free)) {
    if (slot_bytes == 0 || budget_bytes < slot_bytes)
        return;
    const size_t n = budget_bytes / slot_bytes;
    if (!alloc)
        alloc = [](size_t bytes) -> void* {
            void* p = nullptr;
            return cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess ? p : nullptr;
        };
    if (!free_)
        free_ = [](void* p) { cudaFreeHost(p); };
    arena_ = static_cast<char*>(alloc(n * slot_bytes));
    if (!arena_) {
        IMP_LOG_WARN("KV host spill: pinned alloc of %.1f MiB failed; tier disabled",
                     (n * slot_bytes) / (1024.0 * 1024.0));
        return;
    }
    slots_ = static_cast<int>(n);
    free_slots_.reserve(n);
    for (int s = slots_ - 1; s >= 0; --s)
        free_slots_.push_back(s);
}

KVHostSpill::~KVHostSpill() {
    if (arena_ && free_)
        free_(arena_);
}

void* KVHostSpill::reserve(size_t hash) {
    if (!enabled())
        return nullptr;
    auto it = index_.find(hash);
    if (it != index_.end()) {  // same content re-spilled: overwrite in place
        lru_.splice(lru_.end(), lru_, it->second.lru);
        it->second.committed = false;
        return slot_ptr(it->second.slot);
    }
    int slot;
    if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
    } else {
        const size_t victim = lru_.front();
        lru_.pop_front();
        auto v = index_.find(victim);
        slot = v->second.slot;
        index_.erase(v);
        ++replacements_;
    }
    lru_.push_back(hash);
    index_.emplace(hash, Entry{slot, false, std::prev(lru_.end())});
    return slot_ptr(slot);
}

void KVHostSpill::commit(size_t hash) {
    auto it = index_.find(hash);
    if (it != index_.end() && !it->second.committed) {
        it->second.committed = true;
        ++saves_;
    }
}

void KVHostSpill::abandon(size_t hash) { erase(hash); }

const void* KVHostSpill::find(size_t hash) {
    auto it = index_.find(hash);
    if (it == index_.end() || !it->second.committed)
        return nullptr;
    lru_.splice(lru_.end(), lru_, it->second.lru);
    return slot_ptr(it->second.slot);
}

void KVHostSpill::erase(size_t hash) {
    auto it = index_.find(hash);
    if (it == index_.end())
        return;
    free_slots_.push_back(it->second.slot);
    lru_.erase(it->second.lru);
    index_.erase(it);
}

void KVHostSpill::clear() {
    for (auto& [h, e] : index_)
        free_slots_.push_back(e.slot);
    index_.clear();
    lru_.clear();
}

}  // namespace imp
