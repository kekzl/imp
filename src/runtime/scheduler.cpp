#include "runtime/scheduler.h"
#include "memory/kv_cache_manager.h"
#include "memory/kv_cache.h"
#include "core/logging.h"
#include <algorithm>
#include <ranges>

namespace imp {

Scheduler::Scheduler(int max_batch_size) : max_batch_size_(max_batch_size) {}

void Scheduler::add_request(std::shared_ptr<Request> req) {
    pending_.push_back(std::move(req));
    pending_dirty_ = true;
}

void Scheduler::schedule(std::vector<std::shared_ptr<Request>>& prefill_batch,
                         std::vector<std::shared_ptr<Request>>& decode_batch) {
    prefill_batch.clear();
    decode_batch.clear();

    // 1. Remove finished/cancelled requests from active_
    std::erase_if(active_, [](const std::shared_ptr<Request>& r) {
        return r->status == RequestStatus::FINISHED || r->status == RequestStatus::CANCELLED;
    });

    // 2. Sort pending by ascending input token count (shortest-first)
    //    to reduce head-of-line blocking in continuous batching.
    if (pending_dirty_) {
        std::ranges::sort(pending_, [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
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
                if (!kv_manager_->can_allocate(blocks_needed)) {
                    // If the request needs more blocks than the KV cache can
                    // ever hold, no eviction will free enough — leaving the
                    // request in pending_ would busy-loop the worker forever
                    // (observed on Nemotron-H NVFP4 where the KV cache fell
                    // back to the 16-block / 512-token floor and any longer
                    // prompt looped here indefinitely). Cancel up front so
                    // the caller gets a clear error instead of a 30s timeout.
                    int cap = kv_manager_->kv_cache()->total_blocks();
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
                const bool has_image = req->image || req->qwen_patches || req->vision_emb ||
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

            auto r = *it;
            it = pending_.erase(it);
            r->status = RequestStatus::PREFILLING;
            prefill_batch.push_back(r);
            active_.push_back(r);
        }
    }

    // 3. Re-schedule incomplete PREFILLING requests (chunked prefill).
    //    Skip requests already in prefill_batch (just promoted from pending).
    for (auto& req : active_) {
        if (req->status == RequestStatus::PREFILLING && req->prefill_offset > 0 &&
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
