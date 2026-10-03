// KV swap (#2486): admission reserves an expected decode length, so a dry pool at decode is
// expected, not fatal. The request's blocks go to mapped pinned host memory in one batched
// copy, the request waits as SWAPPED (recurrent slot, MTP state and constraints kept) and the
// scheduler brings it back, oldest first, before admitting anything new.

#include "runtime/engine.h"
#include "runtime/graph_eligibility.h"
#include "runtime/scheduler.h"
#include "memory/host_pinned.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "core/logging.h"

#include <cuda_runtime.h>

namespace imp {

struct KvSwapState {
    PinnedBuffer image;  // n_blocks * swap_bytes_per_block(), then the CopyDesc table
    int n_blocks = 0;
    size_t descs_offset = 0;
};

namespace {

size_t align16(size_t n) { return (n + 15) & ~static_cast<size_t>(15); }

}  // namespace

bool Engine::kv_swap_supported_() const {
    // SWA layers live in a second id space and StreamingLLM tables hold -1 sentinels: neither
    // round-trips through a flat block image.
    return runtime_config_.runtime.kv_swap && kv_manager_ && kv_cache_raw_ && !swa_sizing_active_ &&
           !config_.streaming_kv_enabled;
}

bool Engine::kv_valve_armed_(int free_blocks, int reclaimable, int pool_total, int unmet, int live) const {
    // A dry pool with another live sequence swaps one out; dropping every sequence's middle
    // context is for a lone request only.
    if (kv_swap_supported_() && live > 1)
        return false;
    return kv_pressure_demotes_graphs(free_blocks, reclaimable, pool_total, unmet);
}

void Engine::wire_kv_swap_() {
    if (!scheduler_)
        return;
    if (!kv_swap_supported_()) {
        scheduler_->set_admission_decode_mode(-1);
        return;
    }
    scheduler_->set_swap_in([this](Request& r) { return kv_swap_in_(r); });
    scheduler_->set_admission_decode_mode(runtime_config_.runtime.admission_decode_tokens);
}

bool Engine::kv_swap_out_(Request& req) {
    if (!kv_swap_supported_() || req.evicted_kv_tokens > 0 || req.status != RequestStatus::DECODING)
        return false;
    // Alone in the pool, giving the blocks back frees nothing anyone else could use.
    if (kv_manager_->stats().active_sequences <= 1)
        return false;
    const std::vector<int> blocks = kv_manager_->block_table(req.id);
    if (blocks.empty())
        return false;
    const size_t per_block = kv_cache_raw_->swap_bytes_per_block();
    const size_t n_descs = blocks.size() * static_cast<size_t>(kv_cache_raw_->swap_descs_per_block());
    auto st = std::make_shared<KvSwapState>();
    st->n_blocks = static_cast<int>(blocks.size());
    st->descs_offset = align16(blocks.size() * per_block);
    const size_t bytes = st->descs_offset + n_descs * sizeof(KVCache::CopyDesc);
    st->image = PinnedBuffer::acquire(cuda_host_pinned_allocator(), bytes, HostPinnedKind::Mapped);
    if (!st->image || st->image.device() == nullptr) {
        IMP_LOG_WARN("KV swap: %zu MiB pinned host for seq %d unavailable", bytes >> 20, req.id);
        return false;
    }
    char* host = static_cast<char*>(st->image.data());
    char* dev = static_cast<char*>(st->image.device());
    // Every stream: a kernel still reading these blocks must finish before they are freed.
    const cudaStream_t s = decode_stream();
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        kv_cache_raw_->swap_copy(blocks, dev, reinterpret_cast<KVCache::CopyDesc*>(host + st->descs_offset),
                                 reinterpret_cast<const KVCache::CopyDesc*>(dev + st->descs_offset), true, s);
        err = cudaStreamSynchronize(s);
    }
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("KV swap-out of seq %d failed: %s", req.id, cudaGetErrorString(err));
        return false;
    }
    kv_manager_->free_sequence(req.id);
    req.kv_swap = std::move(st);
    req.status = RequestStatus::SWAPPED;
    kv_swaps_out_.fetch_add(1, std::memory_order_relaxed);
    IMP_LOG_INFO("KV swap-out: seq %d, %zu blocks (%zu MiB) to host at ctx %d, out %zu of %d", req.id,
                 blocks.size(), (blocks.size() * per_block) >> 20, req.context_len(),
                 req.output_tokens.size(), req.max_tokens);
    return true;
}

bool Engine::kv_swap_in_(Request& req) {
    if (!req.kv_swap || !kv_manager_ || !kv_cache_raw_)
        return false;
    KvSwapState& st = *req.kv_swap;
    // One block of headroom: the next decode step appends, and must not swap straight back out.
    if (!kv_manager_->can_allocate(st.n_blocks + 1) || !kv_manager_->allocate_blocks(req.id, st.n_blocks))
        return false;
    const std::vector<int> blocks = kv_manager_->block_table(req.id);
    char* host = static_cast<char*>(st.image.data());
    char* dev = static_cast<char*>(st.image.device());
    const cudaStream_t s = decode_stream();
    kv_cache_raw_->swap_copy(blocks, dev, reinterpret_cast<KVCache::CopyDesc*>(host + st.descs_offset),
                             reinterpret_cast<const KVCache::CopyDesc*>(dev + st.descs_offset), false, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        IMP_LOG_ERROR("KV swap-in of seq %d failed: %s", req.id, cudaGetErrorString(cudaGetLastError()));
        kv_manager_->free_sequence(req.id);
        return false;
    }
    IMP_LOG_INFO("KV swap-in: seq %d, %d blocks back at ctx %d", req.id, st.n_blocks, req.context_len());
    req.kv_swap.reset();
    return true;
}

}  // namespace imp
