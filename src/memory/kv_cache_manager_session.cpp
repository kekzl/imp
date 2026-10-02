// Session pins of KVCacheManager (#2407), split out of kv_cache_manager.cpp (file-size gate).

#include "memory/kv_cache_manager.h"

#include "core/logging.h"

#include <string>
#include <vector>

namespace imp {

void KVCacheManager::pin_session(const std::string& session_id, int seq_id, int num_blocks, int64_t now_ms) {
    if (session_id.empty() || num_blocks <= 0)
        return;
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        const int owner = next_session_owner_;
        next_session_owner_ = owner == -0x7fffffff ? -1 : owner - 1;
        it = sessions_.emplace(session_id, SessionPin{owner, now_ms}).first;
        session_of_owner_[owner] = session_id;
        num_sessions_.store(static_cast<int>(sessions_.size()), std::memory_order_relaxed);
    }
    const int owner = it->second.owner;
    it->second.last_pin_ms = now_ms;
    pin_blocks_as_(owner, seq_id, num_blocks);  // may unpin + forget other sessions (budget FIFO)
    if (!pinned_seq_blocks_.contains(owner))
        forget_session_owner_(owner);  // nothing pinnable (seq gone or no full block)
}

int KVCacheManager::close_session(const std::string& session_id) {
    auto it = sessions_.find(session_id);
    if (it == sessions_.end())
        return 0;
    const int owner = it->second.owner;
    auto pins = pinned_seq_blocks_.find(owner);
    const int released = pins == pinned_seq_blocks_.end() ? 0 : static_cast<int>(pins->second.size());
    unpin_prefix(owner);
    forget_session_owner_(owner);
    IMP_LOG_DEBUG("Session '%s' closed: %d pinned blocks released", session_id.c_str(), released);
    return released;
}

int KVCacheManager::expire_sessions(int64_t now_ms, int64_t ttl_ms) {
    if (ttl_ms <= 0 || sessions_.empty())
        return 0;
    std::vector<std::string> idle;
    for (const auto& [id, s] : sessions_)
        if (now_ms - s.last_pin_ms > ttl_ms)
            idle.push_back(id);
    for (const auto& id : idle)
        (void)close_session(id);
    if (!idle.empty())
        IMP_LOG_DEBUG("Session TTL: %zu idle session(s) released", idle.size());
    return static_cast<int>(idle.size());
}

void KVCacheManager::forget_session_owner_(int owner) {
    auto it = session_of_owner_.find(owner);
    if (it == session_of_owner_.end())
        return;
    sessions_.erase(it->second);
    session_of_owner_.erase(it);
    num_sessions_.store(static_cast<int>(sessions_.size()), std::memory_order_relaxed);
}

}  // namespace imp
