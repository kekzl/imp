#include "memory/kv_host_spill.h"

#include "core/logging.h"
#include "memory/kv_cache.h"

#include <cuda_runtime.h>

#include <vector>

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

// Host layout: all layers' K, then V, K scales, V scales, key min/max pages. copy2d gets a part whose
// layers share one size and a constant device stride (rows = layers); anything else goes to copy1d
// one layer at a time.
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

std::atomic<bool> g_batch_copy_ok{true};

// One cudaMemcpyBatchAsync per block, every layer row a segment. A growable pool's layer pitch
// (Qwen3.8-27B: 17223-block ceiling) refuses cudaMemcpy2DAsync, and 80 single copies cost 668 us
// per block vs 75 us on a fixed pool (16 layers, 4 kv heads, hd 256, NVFP4; batched: 55 us).
bool copy_block(KVCache& c, int block_id, char* host, bool to_host, cudaStream_t stream) {
    std::vector<void*> dsts;
    std::vector<const void*> srcs;
    std::vector<size_t> sizes;
    const auto add = [&](size_t off, char* dev, size_t bytes) {
        dsts.push_back(to_host ? host + off : dev);
        srcs.push_back(to_host ? dev : host + off);
        sizes.push_back(bytes);
    };
    for_each_part(
        c, block_id,
        [&](size_t off, char* dev, size_t pitch, size_t width, int rows) {
            for (int r = 0; r < rows; ++r)
                add(off + (width * r), dev + (pitch * r), width);
        },
        [&](size_t off, void* dev, size_t bytes) { add(off, static_cast<char*>(dev), bytes); });
    const char* dir = to_host ? "D2H" : "H2D";
    // The legacy NULL stream takes no batch copy (tests pass nullptr).
    if (stream != nullptr && g_batch_copy_ok.load(std::memory_order_relaxed)) {
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attr_idx = 0;
        const cudaError_t e = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attr,
                                                   &attr_idx, 1, stream);
        if (e == cudaSuccess)
            return true;
        (void)cudaGetLastError();
        if (g_batch_copy_ok.exchange(false))
            IMP_LOG_WARN("KV host spill: cudaMemcpyBatchAsync refused (%s), one copy per segment from now on",
                         cudaGetErrorString(e));
    }
    const cudaMemcpyKind kind = to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    bool ok = true;
    for (size_t i = 0; i < dsts.size(); ++i) {
        const auto off = static_cast<size_t>(static_cast<const char*>(to_host ? dsts[i] : srcs[i]) - host);
        ok &= copy_ok(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], kind, stream), dir, block_id, off, 0,
                      sizes[i], 1);
    }
    return ok;
}
}  // namespace

bool kv_block_to_host(KVCache& cache, int block_id, void* dst, cudaStream_t stream) {
    return copy_block(cache, block_id, static_cast<char*>(dst), true, stream);
}

bool kv_block_from_host(KVCache& cache, const void* src, int block_id, cudaStream_t stream) {
    return copy_block(cache, block_id, const_cast<char*>(static_cast<const char*>(src)), false, stream);
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
