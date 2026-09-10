#pragma once

#include "runtime/request.h"
#include <algorithm>
#include <cstdint>
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

    // Rounds a request may be passed over before it sorts ahead of shorter
    // ones. See the sort in schedule() for why this exists (#1634).
    static constexpr int kAgingRounds = 32;

    // Schedule next batch: returns requests for prefill and decode
    void schedule(std::vector<std::shared_ptr<Request>>& prefill_batch,
                  std::vector<std::shared_ptr<Request>>& decode_batch);

    [[nodiscard]] bool has_pending() const;
    [[nodiscard]] int active_count() const;
    [[nodiscard]] int pending_count() const;  // admitted, not yet in a batch

    // Memory-aware scheduling: set KV cache manager to check budget
    void set_kv_manager(KVCacheManager* mgr) { kv_manager_ = mgr; }
    // Lower the admission cap after construction (the memory plan clamps
    // max_batch_size once the post-weight budget is known). Never raises it.
    void clamp_max_batch_size(int n) { max_batch_size_ = std::min(max_batch_size_, std::max(1, n)); }

    // Hybrid (SSM/GDN) prefix caching: cap on reusable prefix blocks for a
    // request being admitted. The engine's hook finds the longest recurrent-
    // state snapshot matching the prompt, attaches it to the request, and
    // returns its block boundary (0 = no restorable prefix, so no KV reuse
    // — skipping prefill without the matching recurrent state would decode
    // from a zero state). Unset (dense models) = unlimited reuse.
    using PrefixReuseLimitFn = std::function<int(Request&)>;
    void set_prefix_reuse_limit(PrefixReuseLimitFn fn) { prefix_reuse_limit_ = std::move(fn); }
    // Asked before a pending request takes KV blocks: can the engine seat
    // one more sequence right now? False ends this round's admission (the
    // resource is shared, so nothing behind it could be seated either) and
    // the request stays pending; it is retried next round. The lazy SSM slab
    // answers with a slot commit (docs/internals/MEMORY.md, lazy pools).
    using AdmissionGateFn = std::function<bool()>;
    void set_admission_gate(AdmissionGateFn fn) { admission_gate_ = std::move(fn); }

private:
    int max_batch_size_;
    bool pending_dirty_ = false;
    uint64_t round_ = 0;  // schedule() invocations, the clock the aging uses
    std::deque<std::shared_ptr<Request>> pending_;
    std::vector<std::shared_ptr<Request>> active_;
    KVCacheManager* kv_manager_ = nullptr;  // optional, for memory-aware scheduling
    PrefixReuseLimitFn prefix_reuse_limit_;  // optional, hybrid snapshot boundary
    AdmissionGateFn admission_gate_;         // optional, lazy recurrent-slot commit
};

}  // namespace imp
