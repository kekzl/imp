#include "runtime/scheduler.h"
#include "memory/kv_cache_manager.h"
#include "memory/kv_cache.h"
#include "core/logging.h"
#include <algorithm>
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

    // 1. Remove finished/cancelled requests from active_ AND from pending_.
    //
    // pending_ was not filtered (#1633). A request cancelled while queued -
    // which is what the server does when the client disconnects - was promoted
    // anyway a few lines below, and the promotion overwrote CANCELLED with
    // PREFILLING, so nothing downstream could tell either. It then ran a full
    // generation, holding KV and a batch slot, for a client that was gone.
    const auto is_done = [](const std::shared_ptr<Request>& r) {
        return r->status == RequestStatus::FINISHED || r->status == RequestStatus::CANCELLED;
    };
    std::erase_if(active_, is_done);
    std::erase_if(pending_, is_done);

    // 2. Sort pending shortest-first, with aging.
    //
    // Shortest-first reduces head-of-line blocking, which is why it is here.
    // On its own it also starves: the queue is re-sorted on every arrival, so
    // a long prompt is passed over by every shorter one that shows up while
    // the batch is full, for as long as that lasts. Under sustained short
    // traffic "for as long as that lasts" has no bound (#1634).
    //
    // Aging puts a bound on it without giving up the property: a request that
    // has been waiting kAgingRounds scheduling rounds sorts ahead of every
    // request that has not, and ties fall back to length. So the ordering is
    // shortest-first among peers, and arrival order across the aging boundary.
    //
    // The sort has to run whenever the round advances, not only when the queue
    // changed - the aging bucket of a request changes with time, not with
    // arrivals, and `pending_dirty_` cannot see time passing.
    ++round_;
    const uint64_t now = round_;
    if (pending_dirty_ || !pending_.empty()) {
        std::ranges::sort(pending_, [now](const std::shared_ptr<Request>& a,
                                          const std::shared_ptr<Request>& b) {
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

            // Memory-aware check: estimate KV blocks needed for this request
            if (kv_manager_) {
                int ctx_len = req->context_len();
                const int bs = kv_manager_->kv_cache()->block_size();
                int blocks_needed = (ctx_len + bs - 1) / bs;

                // Admit on prompt + generation, not on the prompt alone
                // (#1635). context_len() counts what exists NOW, so a batch
                // whose prompts all fit could still run the pool dry mid-
                // generation, and the loser is cancelled after the client has
                // already received part of the answer. The grow branch below
                // has computed the decode half correctly since it was written;
                // the admission test above it did not read it.
                //
                // Clamped to the pool: on a cache too small to ever hold
                // prompt + max_tokens (the 16-block floor case) the full
                // reserve would queue every request forever. There the
                // guarantee degrades to the old prompt-only admission rather
                // than to a refusal, and the mid-stream cancel stays possible.
                const int decode_blocks = (req->max_tokens + bs - 1) / bs + 1;
                const int pool_blocks = kv_manager_->kv_cache()->total_blocks();
                const int admit_blocks = std::min(blocks_needed + decode_blocks,
                                                  std::max(blocks_needed, pool_blocks));

                // Aggregate-pressure growth (2026-08-27): requests that each
                // fit used to queue while the pool sat at its initial commit:
                // 32 x 8k-token concurrent measured effectively ~7-way with
                // 4437 of the 6483 ceiling blocks never committed (45.2 s
                // wall). The old rule "ordinary contention is left to queue so
                // growth never competes with the weight caches" is stricter
                // than its own reason: the ceiling IS the post-weight residual
                // the planner clamped at init (vram_budget "KV clamped ... to
                // fit post-weight VRAM"), so growing up to it competes with
                // nothing. Coarse steps, same as the decode-side trigger.
                if (!kv_manager_->can_allocate(admit_blocks)) {
                    auto* kvc = kv_manager_->kv_cache();
                    const int total_now = kvc->total_blocks();
                    if (kvc->ceiling_blocks() > total_now)
                        kvc->try_grow_to(total_now + std::max(admit_blocks, total_now / 4));
                }
                if (!kv_manager_->can_allocate(admit_blocks)) {
                    // If the request needs more blocks than the KV cache can
                    // ever hold, no eviction will free enough — leaving the
                    // request in pending_ would busy-loop the worker forever
                    // (observed on Nemotron-H NVFP4 where the KV cache fell
                    // back to the 16-block / 512-token floor and any longer
                    // prompt looped here indefinitely). Cancel up front so
                    // the caller gets a clear error instead of a 30s timeout.
                    int cap = kv_manager_->kv_cache()->total_blocks();
                    // A growable pool is allowed to answer this with memory
                    // rather than with a refusal. Only here, where the pool
                    // cannot hold the request AT ALL: that is the condition a
                    // clamped startup produces and the one no amount of
                    // waiting fixes. Ordinary contention between requests that
                    // each fit is left to queue, so growth never competes with
                    // the weight caches for VRAM on a merely busy server.
                    //
                    // This does ask the driver for memory during serving,
                    // which invariant I2 otherwise forbids. It is the
                    // exception the growable pool is for, it is bounded by the
                    // ceiling reserved at init, and it is logged when it
                    // happens.
                    if (blocks_needed > cap) {
                        // Grow for the whole request, not for its prompt.
                        // context_len() counts what exists NOW, so growing to
                        // exactly that produced a pool with zero decode
                        // headroom and the request was cancelled anyway, one
                        // block short, after paying for the growth. Measured on
                        // a 25 222-token prompt: grew 810 -> 1577 blocks, then
                        // refused.
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
                    // Otherwise: not enough memory right now, try smaller requests
                    ++it;
                    continue;
                }
                // Reserve blocks, using prefix caching when enabled.
                //
                // An image request participates only through its content hash:
                // the cache is addressed by TOKEN IDS, every image token carries
                // the SAME id, and two different pictures would otherwise share
                // a long prefix. A request that carries an image but reports no
                // hash is excluded outright, so a missed plumbing site degrades
                // to "no reuse" rather than "the previous picture".
                const bool has_image = req->image || !req->qwen_patches.empty() || req->vision_emb ||
                                       req->n_vision_tokens > 0;
                const bool cacheable = !has_image || req->vision_content_hash != 0;
                if (kv_manager_->prefix_caching_enabled() && cacheable) {
                    // Hybrid models cap reuse at the recurrent-snapshot
                    // boundary (and attach the snapshot to the request).
                    int max_reuse = prefix_reuse_limit_ ? prefix_reuse_limit_(*req) : -1;
                    int reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens,
                                                                          max_reuse,
                                                                          req->vision_content_hash);
                    if (reused < 0) {
                        ++it;
                        continue;
                    }
                    if (max_reuse >= 0 && reused != max_reuse) {
                        // Defensive: the snapshot boundary was probed against
                        // the cache a moment ago, so this should not happen.
                        // The restore position no longer matches the reused
                        // KV prefix — release everything (a full prefill must
                        // not re-write blocks still shared with other seqs)
                        // and fall back to plain allocation.
                        IMP_LOG_WARN(
                            "Scheduler: hybrid prefix reuse mismatch (reused=%d, snapshot=%d "
                            "blocks) — full prefill for req %d",
                            reused, max_reuse, req->id);
                        req->recurrent_restore.reset();
                        kv_manager_->free_sequence(req->id);
                        if (!kv_manager_->allocate_blocks(req->id, blocks_needed)) {
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
                            // Full prefix hit: still forward the last token (the
                            // model needs logits for the next position). That
                            // re-prefill re-writes KV at total-1, which may sit
                            // in a SHARED (ref>=2) cached block — there is no
                            // copy-on-write here (F-A10). It is safe because the
                            // write is idempotent: a prefix hit requires
                            // byte-identical tokens+positions (chained block
                            // hash), and quantizing identical FP16 source is
                            // deterministic, so the shared holders see the same
                            // KV bytes. If a future KV-quant scheme makes
                            // re-quant input-dependent on neighbouring tokens
                            // (non-idempotent), this site needs COW of the last
                            // block before the re-prefill.
                            skip = total - 1;
                        req->prefill_offset = skip;
                        // Reporting: usage prompt_tokens_details / Anthropic
                        // cache_read_input_tokens read this off the request.
                        req->cached_tokens = skip;
                    }
                } else {
                    if (!kv_manager_->allocate_blocks(req->id, blocks_needed)) {
                        ++it;
                        continue;
                    }
                }
            }

            if (kv_manager_) {
                // Hold the promise, do not just test it. Without this the next
                // request is admitted against the blocks this one has not
                // written yet, which is the same over-admission one round
                // later (#1635). It decays as the blocks are appended and is
                // dropped by free_sequence().
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
            prefill_batch.push_back(r);
            active_.push_back(r);
        }
    }

    // 3. Re-schedule incomplete PREFILLING requests (chunked prefill).
    //    Skip requests already in prefill_batch (just promoted from pending).
    //
    //    `>= 0`, not `> 0` (#1643): a request promoted in step 2 but not served
    //    that tick still has offset 0, and the old condition dropped it here -
    //    PREFILLING, admitted, holding KV, and in no batch ever again. Nothing
    //    hit it while every promoted request was served immediately; the
    //    per-step prefill cap makes "not served this tick" a normal state, and
    //    two of three concurrent ingests then hung until the 300 s request
    //    timeout. The `already_queued` scan below is what keeps the
    //    just-promoted ones from being added twice.
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

}  // namespace imp
