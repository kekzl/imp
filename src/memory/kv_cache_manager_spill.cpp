// Host spill tier of KVCacheManager (#2203), split out of kv_cache_manager.cpp (file-size gate).

#include "memory/kv_cache_manager.h"

#include "core/logging.h"

namespace imp {

bool KVCacheManager::enable_host_spill(size_t budget_bytes) {
    if (budget_bytes == 0)
        return false;
    const char* why = !prefix_caching_enabled_                       ? "prefix caching is off"
                      : !cache_ || !kv_host_spill_supported(*cache_) ? "SWA layers"
                                                                     : nullptr;
    std::unique_ptr<KVHostSpill> spill;
    if (!why) {
        spill = std::make_unique<KVHostSpill>(budget_bytes, kv_host_block_bytes(*cache_));
        if (!spill->enabled() || !spill_stream_.create())
            why = "pinned alloc or stream create failed";
    }
    if (why) {
        IMP_LOG_WARN("kv_cache.host_spill_mb=%zu: host spill tier off (%s)", budget_bytes >> 20, why);
        return false;
    }
    IMP_LOG_INFO("KV host spill: %d blocks x %.1f KiB pinned (%.1f MiB)", spill->slots(),
                 spill->slot_bytes() / 1024.0, (spill->slots() * spill->slot_bytes()) / (1024.0 * 1024.0));
    host_spill_ = std::move(spill);
    return true;
}

// The block's KV was written by a forward that finished before its sequence was freed (tokens are
// read back to the host first), so the copy needs no wait on the compute stream.
void KVCacheManager::spill_block_(int block_id, size_t hash) {
    void* dst = host_spill_->reserve(hash);
    if (!dst)
        return;
    const bool enqueued = kv_block_to_host(*cache_, block_id, dst, spill_stream_);
    if (cudaStreamSynchronize(spill_stream_) == cudaSuccess && enqueued)
        host_spill_->commit(hash, identity_(hash, block_id));
    else
        host_spill_->abandon(hash);
}

bool KVCacheManager::restore_spilled_(size_t hash, int block_id) {
    const void* src = host_spill_->find(hash);
    if (!src)
        return false;
    // Pending even on failure: flush_spill_restores_ must retire the copies that did enqueue.
    spill_restores_pending_ = true;
    if (!kv_block_from_host(*cache_, src, block_id, spill_stream_)) {
        IMP_LOG_ERROR("KV host spill: restore enqueue failed for block %d", block_id);
        return false;
    }
    host_spill_->count_restore();
    return true;
}

KvChainLink KVCacheManager::identity_(size_t hash, int block_id) const {
    if (auto o = block_origin_.find(block_id); o != block_origin_.end())
        return o->second;
    auto s = block_bind_serial_.find(block_id);
    return {hash, block_id, s != block_bind_serial_.end() ? s->second : 0};
}

int KVCacheManager::device_block_of_(const KvChainLink& l) const {
    const auto same = [&](int id) {
        const KvChainLink k = identity_(l.hash, id);
        return k.block_id == l.block_id && k.serial == l.serial;
    };
    if (auto hit = block_hash_to_id_.find(l.hash); hit != block_hash_to_id_.end() && same(hit->second))
        return hit->second;
    if (auto sh = shadow_id_to_hash_.find(l.block_id);
        sh != shadow_id_to_hash_.end() && sh->second == l.hash && same(l.block_id))
        return l.block_id;
    return -1;
}

bool KVCacheManager::host_holds_(const KvChainLink& l) const {
    return host_spill_ && host_spill_->holds(l) && !block_hash_to_id_.contains(l.hash);
}

// A link holds while its KV is bound on the device or sits in the host tier.
bool KVCacheManager::chain_intact(std::span<const KvChainLink> chain) const {
    if (chain.empty())
        return false;
    for (const auto& l : chain)
        if (device_block_of_(l) < 0 && !host_holds_(l))
            return false;
    return true;
}

void KVCacheManager::flush_spill_restores_() {
    if (!spill_restores_pending_)
        return;
    spill_restores_pending_ = false;
    if (cudaStreamSynchronize(spill_stream_) != cudaSuccess)
        IMP_LOG_ERROR("KV host spill: restore copy failed: %s", cudaGetErrorString(cudaGetLastError()));
}

}  // namespace imp
