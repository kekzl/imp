#pragma once

#include "runtime/decode_length_estimate.h"
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
    // Ids of the admitted requests still running (engine thread only, like schedule()).
    [[nodiscard]] std::vector<int> active_ids() const;
    // Live sequences whose KV StreamingLLM evicted (engine thread only, like schedule()).
    [[nodiscard]] int active_evicted_count() const;
    // Sum of kv_unmet_blocks (graph_eligibility.h) over the live sequences (engine thread only).
    [[nodiscard]] int active_unmet_kv_blocks() const;

    // Memory-aware scheduling: set KV cache manager to check budget
    void set_kv_manager(KVCacheManager* mgr) { kv_manager_ = mgr; }
    // Lower the admission cap after construction (the memory plan clamps
    // max_batch_size once the post-weight budget is known). Never raises it.
    void clamp_max_batch_size(int n) { max_batch_size_ = std::min(max_batch_size_, std::max(1, n)); }

    // Hybrid (SSM/GDN) prefix caching: cap on reusable prefix blocks. The
    // engine's hook finds the longest recurrent-state snapshot matching the
    // prompt and returns its block boundary (0 = no restorable prefix, since skipping prefill without matching state would decode from zero). Unset (dense models) = unlimited reuse.
    using PrefixReuseLimitFn = std::function<int(Request&)>;
    void set_prefix_reuse_limit(PrefixReuseLimitFn fn) { prefix_reuse_limit_ = std::move(fn); }
    // Asked before a pending request takes KV blocks: can the engine seat one
    // more sequence now? False ends this round's admission (shared resource,
    // nothing behind could be seated either); retried next round. Lazy SSM slab answers with a slot commit (docs/internals/MEMORY.md).
    using AdmissionGateFn = std::function<bool()>;
    void set_admission_gate(AdmissionGateFn fn) { admission_gate_ = std::move(fn); }

    // #2486: decode tokens reserved per admitted request (admission_decode_tokens: -1 =
    // max_tokens, 0 = DecodeLengthEstimate, > 0 fixed). The engine passes -1 unless it can swap.
    void set_admission_decode_mode(int mode) { admission_decode_mode_ = mode; }
    // Brings a SWAPPED request's KV back (engine thread). True = it holds its blocks again.
    using SwapInFn = std::function<bool(Request&)>;
    void set_swap_in(SwapInFn fn) { swap_in_ = std::move(fn); }
    [[nodiscard]] int reserve_decode_tokens(const Request& r) const {
        return admission_decode_tokens(admission_decode_mode_, r.max_tokens, decode_estimate_);
    }
    [[nodiscard]] const DecodeLengthEstimate& decode_estimate() const { return decode_estimate_; }
    [[nodiscard]] int swapped_count() const;

private:
    // Swap SWAPPED requests back in, oldest first; true while one still waits.
    [[nodiscard]] bool swap_in_waiting_();
    // Feeds FINISHED output lengths into decode_estimate_ (before schedule() erases them).
    void record_finished_();
    // Admission's decode promise for req (#1635), held until its blocks are written.
    void hold_decode_reservation_(const Request& req);

    int max_batch_size_;
    bool pending_dirty_ = false;
    uint64_t round_ = 0;  // schedule() invocations, the clock the aging uses
    std::deque<std::shared_ptr<Request>> pending_;
    std::vector<std::shared_ptr<Request>> active_;
    KVCacheManager* kv_manager_ = nullptr;  // optional, for memory-aware scheduling
    PrefixReuseLimitFn prefix_reuse_limit_;  // optional, hybrid snapshot boundary
    AdmissionGateFn admission_gate_;         // optional, lazy recurrent-slot commit
    SwapInFn swap_in_;                       // optional, KV swap-in (#2486)
    int admission_decode_mode_ = -1;
    DecodeLengthEstimate decode_estimate_;
};

}  // namespace imp
