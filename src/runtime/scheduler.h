#pragma once

#include "runtime/request.h"
#include <vector>
#include <deque>
#include <functional>
#include <memory>

namespace imp {

class KVCacheManager;  // forward declare

class Scheduler {
public:
    explicit Scheduler(int max_batch_size = 32);
    ~Scheduler() = default;

    void add_request(std::shared_ptr<Request> req);

    // Schedule next batch: returns requests for prefill and decode
    void schedule(std::vector<std::shared_ptr<Request>>& prefill_batch,
                  std::vector<std::shared_ptr<Request>>& decode_batch);

    [[nodiscard]] bool has_pending() const;
    [[nodiscard]] int active_count() const;

    // Memory-aware scheduling: set KV cache manager to check budget
    void set_kv_manager(KVCacheManager* mgr) { kv_manager_ = mgr; }

    // Hybrid (SSM/GDN) prefix caching: cap on reusable prefix blocks for a
    // request being admitted. The engine's hook finds the longest recurrent-
    // state snapshot matching the prompt, attaches it to the request, and
    // returns its block boundary (0 = no restorable prefix, so no KV reuse
    // — skipping prefill without the matching recurrent state would decode
    // from a zero state). Unset (dense models) = unlimited reuse.
    using PrefixReuseLimitFn = std::function<int(Request&)>;
    void set_prefix_reuse_limit(PrefixReuseLimitFn fn) { prefix_reuse_limit_ = std::move(fn); }

private:
    int max_batch_size_;
    bool pending_dirty_ = false;
    std::deque<std::shared_ptr<Request>> pending_;
    std::vector<std::shared_ptr<Request>> active_;
    KVCacheManager* kv_manager_ = nullptr;  // optional, for memory-aware scheduling
    PrefixReuseLimitFn prefix_reuse_limit_;  // optional, hybrid snapshot boundary
};

}  // namespace imp
