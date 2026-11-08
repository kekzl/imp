#include "memory/kv_host_spill.h"

#include "core/logging.h"
#include "memory/kv_cache.h"

#include <cuda_runtime.h>

namespace imp {

bool kv_host_spill_supported(KVCache& cache) {
    for (int l = 0; l < cache.n_layers(); ++l)
        if (cache.layer_is_swa(l))
            return false;
    return cache.n_layers() > 0;
}

namespace {

// KeyMinMax: the sparse attention page metadata of the block's keys, a pure function of its rows.
enum class Part { K, V, KScale, VScale, KeyMinMax };

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
        case Part::KeyMinMax:
            return c.key_minmax_ptr(layer, block_id);
    }
    return nullptr;
}

size_t part_bytes(KVCache& c, Part p, int layer) {
    if (p == Part::KeyMinMax)
        return KVCache::minmax_block_bytes(c.n_kv_heads(), c.head_dim());
    return (p == Part::K || p == Part::V) ? c.block_bytes(layer) : c.scale_block_bytes(layer);
}

// Host layout: all layers' K, then V, K scales, V scales, key min/max pages. A part whose layers share one
// size and a constant device stride moves as a single 2D copy (2 to 4 copies per block instead of
// 2-4 per layer); anything else falls back to one copy per layer.
template <typename F2D, typename F1D>
size_t for_each_part(KVCache& c, int block_id, F2D&& copy2d, F1D&& copy1d) {
    const int n = c.n_layers();
    size_t host_off = 0;
    for (Part p : {Part::K, Part::V, Part::KScale, Part::VScale, Part::KeyMinMax}) {
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

namespace {
// First failed copy only: the error is consumed so it does not surface at an unrelated check.
bool copy_ok(cudaError_t e, const char* dir, int block_id, size_t off, size_t pitch, size_t width, int rows) {
    if (e == cudaSuccess)
        return true;
    static std::atomic<bool> logged{false};
    (void)cudaGetLastError();
    if (!logged.exchange(true))
        IMP_LOG_ERROR(
            "KV host spill: %s copy of block %d failed (%s): host off %zu, pitch %zu, width %zu, rows %d",
            dir, block_id, cudaGetErrorString(e), off, pitch, width, rows);
    return false;
}

// A 2D copy over a 125 MiB layer pitch failed as invalid argument (Qwen3.8-27B, pool at 2000
// blocks during warmup): one copy per layer then.
cudaError_t copy_2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, int rows,
                    cudaMemcpyKind kind, cudaStream_t stream) {
    if (cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, rows, kind, stream) == cudaSuccess)
        return cudaSuccess;
    (void)cudaGetLastError();
    cudaError_t e = cudaSuccess;
    for (int r = 0; r < rows && e == cudaSuccess; ++r)
        e = cudaMemcpyAsync(static_cast<char*>(dst) + (dpitch * r),
                            static_cast<const char*>(src) + (spitch * r), width, kind, stream);
    return e;
}
}  // namespace

bool kv_block_to_host(KVCache& cache, int block_id, void* dst, cudaStream_t stream) {
    char* out = static_cast<char*>(dst);
    bool ok = true;
    for_each_part(
        cache, block_id,
        [&](size_t off, char* dev, size_t pitch, size_t width, int rows) {
            ok &= copy_ok(copy_2d(out + off, width, dev, pitch, width, rows, cudaMemcpyDeviceToHost, stream),
                          "D2H", block_id, off, pitch, width, rows);
        },
        [&](size_t off, void* dev, size_t bytes) {
            ok &= copy_ok(cudaMemcpyAsync(out + off, dev, bytes, cudaMemcpyDeviceToHost, stream), "D2H",
                          block_id, off, 0, bytes, 1);
        });
    return ok;
}

bool kv_block_from_host(KVCache& cache, const void* src, int block_id, cudaStream_t stream) {
    const char* in = static_cast<const char*>(src);
    bool ok = true;
    for_each_part(
        cache, block_id,
        [&](size_t off, char* dev, size_t pitch, size_t width, int rows) {
            ok &= copy_ok(copy_2d(dev, pitch, in + off, width, width, rows, cudaMemcpyHostToDevice, stream),
                          "H2D", block_id, off, pitch, width, rows);
        },
        [&](size_t off, void* dev, size_t bytes) {
            ok &= copy_ok(cudaMemcpyAsync(dev, in + off, bytes, cudaMemcpyHostToDevice, stream), "H2D",
                          block_id, off, 0, bytes, 1);
        });
    return ok;
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

void KVHostSpill::drop_alias_(size_t key, const Entry& e) {
    if (!e.committed || e.origin.block_id < 0)
        return;
    if (auto a = latest_.find(e.origin.hash); a != latest_.end() && a->second == key)
        latest_.erase(a);
}

void* KVHostSpill::reserve(size_t key) {
    if (!enabled())
        return nullptr;
    auto it = index_.find(key);
    if (it != index_.end()) {  // same binding re-spilled: overwrite in place
        lru_.splice(lru_.end(), lru_, it->second.lru);
        drop_alias_(key, it->second);
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
        drop_alias_(victim, v->second);
        index_.erase(v);
        ++replacements_;
    }
    lru_.push_back(key);
    index_.emplace(key, Entry{slot, false, std::prev(lru_.end()), KvChainLink{}});
    return slot_ptr(slot);
}

void KVHostSpill::commit(size_t key, const KvChainLink& origin) {
    auto it = index_.find(key);
    if (it != index_.end() && !it->second.committed) {
        it->second.committed = true;
        it->second.origin = origin;
        if (origin.block_id >= 0)
            latest_[origin.hash] = key;
        ++saves_;
    }
}

void KVHostSpill::abandon(size_t key) { erase(key); }

const void* KVHostSpill::find(size_t key) {
    auto it = index_.find(key);
    if (it == index_.end() || !it->second.committed)
        return nullptr;
    lru_.splice(lru_.end(), lru_, it->second.lru);
    return slot_ptr(it->second.slot);
}

bool KVHostSpill::latest_key(size_t hash, size_t* key) const {
    auto a = latest_.find(hash);
    if (a == latest_.end())
        return false;
    *key = a->second;
    return true;
}

KvChainLink KVHostSpill::origin(size_t hash) const {
    size_t key = 0;
    return latest_key(hash, &key) ? origin_of_key(key) : KvChainLink{};
}

void KVHostSpill::erase(size_t key) {
    auto it = index_.find(key);
    if (it == index_.end())
        return;
    drop_alias_(key, it->second);
    free_slots_.push_back(it->second.slot);
    lru_.erase(it->second.lru);
    index_.erase(it);
}

void KVHostSpill::clear() {
    for (auto& [h, e] : index_)
        free_slots_.push_back(e.slot);
    index_.clear();
    latest_.clear();
    lru_.clear();
}

}  // namespace imp
