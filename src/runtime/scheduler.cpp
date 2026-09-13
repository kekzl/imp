#include "runtime/scheduler.h"
#include "memory/kv_cache_manager.h"
#include "memory/kv_cache.h"
#include "core/logging.h"
#include <algorithm>
#include <chrono>
#include <ranges>

namespace imp {

Scheduler::Scheduler(int max_batch_size) : max_batch_size_(max_batch_size) {}

void Scheduler::add_request(std::shared_ptr<Request> req) {
    req->enqueued_round = round_;
    pending_.push_back(std::move(req));
    pending_dirty_ = true;
}

void Scheduler::schedule(std::vector<std::shared_ptr<Request>>& prefill_batch,
                         std::vector<std::shared_ptr<Request>>& decode_batch) {
    prefill_batch.clear();
    decode_batch.clear();

    // 1. Remove finished/cancelled requests from active_ AND pending_: a request
    // cancelled while queued must not be promoted (the promotion would overwrite
    // CANCELLED with PREFILLING and run a full generation for a gone client) (#1633).
    const auto is_done = [](const std::shared_ptr<Request>& r) {
        return r->status == RequestStatus::FINISHED || r->status == RequestStatus::CANCELLED;
    };
    std::erase_if(active_, is_done);
    std::erase_if(pending_, is_done);

    // 2. Sort pending: priority (Request::priority, default 0) is primary and
    // dominates STRICTLY (aging never lets a class overtake another, caller owns
    // cross-class starvation, #1634).
    // Within a class: shortest-first, aged (kAgingRounds) requests sort ahead of
    // unaged peers to bound starvation under sustained short traffic; ties break
    // by enqueue order among the aged, else by length.
    // Re-sorts every round, not only on pending_dirty_: the aging bucket changes
    // with time, not with arrivals.
    ++round_;
    const uint64_t now = round_;
    if (pending_dirty_ || !pending_.empty()) {
        std::ranges::sort(pending_, [now](const std::shared_ptr<Request>& a,
                                          const std::shared_ptr<Request>& b) {
            if (a->priority != b->priority)
                return a->priority < b->priority;
            const bool a_aged = now - a->enqueued_round >= static_cast<uint64_t>(kAgingRounds);
            const bool b_aged = now - b->enqueued_round >= static_cast<uint64_t>(kAgingRounds);
            if (a_aged != b_aged)
                return a_aged;  // waited long enough beats short
            if (a_aged)
                return a->enqueued_round < b->enqueued_round;  // among the aged, oldest first
            return a->input_tokens.size() < b->input_tokens.size();
        });
        pending_dirty_ = false;
    }

    // 3. Promote pending requests to prefill (up to max_batch_size_ budget)
    {
        auto it = pending_.begin();
        while (it != pending_.end() && static_cast<int>(active_.size()) < max_batch_size_) {
            auto& req = *it;
            // Aging's allocator half: nothing behind an aged head is admitted in a
            // round it could not get blocks in (AUDIT_arch_2026 C-2), else shorter
            // requests starve it on a pool that's nearly but never quite full. Only
            // the aged head holds the queue; infeasible still cancels, unaged still yields to what fits.
            const bool aged = now - req->enqueued_round >= static_cast<uint64_t>(kAgingRounds);

            // Before any KV is taken: a seat the engine cannot back right now
            // (lazy recurrent slot) ends the round for everyone behind too.
            if (admission_gate_ && !admission_gate_())
                break;

            // Memory-aware check: estimate KV blocks needed for this request
            if (kv_manager_) {
                int ctx_len = req->context_len();
                const int bs = kv_manager_->kv_cache()->block_size();
                int blocks_needed = (ctx_len + bs - 1) / bs;

                // Admit on prompt + generation, not prompt alone (#1635): context_len() is
                // NOW, so an all-fits batch can still run the pool dry mid-generation.
                // Clamped to the pool: a cache too small for prompt+max_tokens (16-block
                // floor) degrades to prompt-only admission instead of queuing forever.
                const int decode_blocks = (req->max_tokens + bs - 1) / bs + 1;
                const int pool_blocks = kv_manager_->kv_cache()->total_blocks();
                const int admit_blocks = std::min(blocks_needed + decode_blocks,
                                                  std::max(blocks_needed, pool_blocks));

                // Aggregate-pressure growth: the pool grows toward its ceiling here
                // because the ceiling IS the post-weight residual already clamped at
                // init (vram_budget), so growing up to it competes with nothing. Coarse steps, same as the decode-side trigger.
                if (!kv_manager_->can_allocate(admit_blocks)) {
                    auto* kvc = kv_manager_->kv_cache();
                    const int total_now = kvc->total_blocks();
                    if (kvc->ceiling_blocks() > total_now)
                        kvc->try_grow_to(total_now + std::max(admit_blocks, total_now / 4));
                }
                if (!kv_manager_->can_allocate(admit_blocks)) {
                    // If the request needs more blocks than the KV cache can ever
                    // hold, no eviction frees enough: leaving it in pending_ busy-loops
                    // the worker forever, so cancel up front instead of a 30s timeout.
                    int cap = kv_manager_->kv_cache()->total_blocks();
                    // Grows here only when the pool cannot hold the request AT ALL (a
                    // clamped startup, unfixable by waiting); ordinary contention among
                    // fitting requests still queues. This is the deliberate exception to
                    // invariant I2 (no driver calls during serving): bounded by the init ceiling, logged when it fires.
                    if (blocks_needed > cap) {
                        // Grow for the whole request, not just its prompt: context_len()
                        // is NOW, so growing to exactly that leaves zero decode headroom
                        // and cancels anyway, one block short, after paying for the growth.
                        cap = kv_manager_->kv_cache()->try_grow_to(blocks_needed + decode_blocks);
                    }
                    if (blocks_needed > cap) {
                        IMP_LOG_ERROR(
                            "Scheduler: request %d needs %d KV blocks but cache capacity is %d "
                            "(ctx_len=%d, block_size=%d) — cancelling (KV cache too small for prompt)",
                            req->id, blocks_needed, cap, ctx_len, bs);
                        req->status = RequestStatus::CANCELLED;
                        // Actionable, unlike every other cancellation: the
                        // caller can shorten the prompt or give the process
                        // more VRAM. Surfaces as IMP_ERROR_CAPACITY / HTTP 503.
                        req->cancel_reason = CancelReason::KvCapacity;
                        it = pending_.erase(it);
                        continue;
                    }
                    // Otherwise: not enough memory right now. An aged head
                    // keeps the queue; anyone else lets smaller requests by.
                    if (aged)
                        break;
                    ++it;
                    continue;
                }
                // Reserve blocks, using prefix caching when enabled. An image request
                // participates only through its content hash (every image token shares
                // one id, so two different pictures would otherwise share a prefix); no
                // hash means excluded, degrading to "no reuse" rather than "the previous picture".
                const bool has_image = req->image || !req->qwen_patches.empty() || req->vision_emb ||
                                       req->n_vision_tokens > 0;
                const bool cacheable = !has_image || req->vision_content_hash != 0;
                if (kv_manager_->prefix_caching_enabled() && cacheable) {
                    // Hybrid models cap reuse at the recurrent-snapshot
                    // boundary (and attach the snapshot to the request).
                    int max_reuse = prefix_reuse_limit_ ? prefix_reuse_limit_(*req) : -1;
                    int reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens,
                                                                          max_reuse, req->prefix_salt);
                    if (reused < 0) {
                        if (aged)
                            break;
                        ++it;
                        continue;
                    }
                    if (max_reuse >= 0 && reused != max_reuse) {
                        // Defensive: the snapshot boundary was probed moments ago, so this
                        // should not happen. Restore position no longer matches the reused
                        // KV prefix: release everything (a full prefill must not rewrite shared blocks) and fall back to plain allocation.
                        IMP_LOG_WARN(
                            "Scheduler: hybrid prefix reuse mismatch (reused=%d, snapshot=%d "
                            "blocks) — full prefill for req %d",
                            reused, max_reuse, req->id);
                        req->recurrent_restore.reset();
                        kv_manager_->free_sequence(req->id);
                        if (!kv_manager_->allocate_blocks(req->id, blocks_needed)) {
                            if (aged)
                                break;
                            ++it;
                            continue;
                        }
                        reused = 0;
                    }
                    // Skip prefill for tokens covered by reused blocks.
                    if (reused > 0) {
                        int skip = reused * bs;
                        int total = static_cast<int>(req->input_tokens.size());
                        if (skip >= total)
                            skip = (total / bs) * bs;
                        if (skip >= total)
                            // Full prefix hit: still forward the last token (model needs
                            // logits for the next position). Re-prefill rewrites KV at
                            // total-1, possibly in a SHARED (ref>=2) block with no
                            // copy-on-write (F-A10); safe only because a prefix hit requires
                            // byte-identical tokens+positions and quantizing identical FP16
                            // is deterministic. A future input-dependent re-quant needs COW here first.
                            skip = total - 1;
                        req->prefill_offset = skip;
                        // Reporting: usage prompt_tokens_details / Anthropic
                        // cache_read_input_tokens read this off the request.
                        req->cached_tokens = skip;
                    }
                } else {
                    if (!kv_manager_->allocate_blocks(req->id, blocks_needed)) {
                        if (aged)
                            break;
                        ++it;
                        continue;
                    }
                }
            }

            if (kv_manager_) {
                // Hold the promise, do not just test it: without this the next request is
                // admitted against blocks this one has not written yet (#1635). Decays as
                // blocks are appended, dropped by free_sequence().
                const int bs = kv_manager_->kv_cache()->block_size();
                const int prompt_blocks = (req->context_len() + bs - 1) / bs;
                const int decode_blocks = (req->max_tokens + bs - 1) / bs + 1;
                const int pool_blocks = kv_manager_->kv_cache()->total_blocks();
                kv_manager_->set_decode_reservation(req->id, std::min(prompt_blocks + decode_blocks,
                                                                      std::max(prompt_blocks, pool_blocks)));
            }

            auto r = *it;
            it = pending_.erase(it);
            r->status = RequestStatus::PREFILLING;
            r->t_scheduled = std::chrono::steady_clock::now();
            prefill_batch.push_back(r);
            active_.push_back(r);
        }
    }

    // 3. Re-schedule incomplete PREFILLING requests (chunked prefill), skipping
    //    ones already in prefill_batch. `>= 0`, not `> 0` (#1643): a request
    //    promoted but not served this tick still has offset 0, and `> 0` dropped it
    //    here forever, holding KV with no batch. already_queued avoids double-adding.
    for (auto& req : active_) {
        if (req->status == RequestStatus::PREFILLING && req->prefill_offset >= 0 &&
            req->prefill_offset < static_cast<int>(req->input_tokens.size())) {
            bool already_queued = false;
            for (const auto& pf : prefill_batch) {
                if (pf.get() == req.get()) {
                    already_queued = true;
                    break;
                }
            }
            if (!already_queued) {
                prefill_batch.push_back(req);
            }
        }
    }

    // 4. All active decoding requests go to the decode batch
    for (auto& req : active_) {
        if (req->status == RequestStatus::DECODING) {
            decode_batch.push_back(req);
        }
    }
}

bool Scheduler::has_pending() const { return !pending_.empty(); }

int Scheduler::active_count() const { return static_cast<int>(active_.size()); }
int Scheduler::pending_count() const { return static_cast<int>(pending_.size()); }

}  // namespace imp
