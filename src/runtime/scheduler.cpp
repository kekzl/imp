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
                        it = pending_.erase(it);
                        continue;
                    }
                    // Otherwise: not enough memory right now, try smaller requests
                    ++it;
                    continue;
                }
                // Reserve blocks, using prefix caching when enabled.
                if (kv_manager_->prefix_caching_enabled()) {
                    int reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens);
                    if (reused < 0) {
                        ++it;
                        continue;
                    }
                    // Skip prefill for tokens covered by reused blocks.
                    if (reused > 0) {
                        int skip = reused * bs;
                        int total = static_cast<int>(req->input_tokens.size());
                        if (skip >= total)
                            skip = (total / bs) * bs;
                        if (skip >= total)
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
