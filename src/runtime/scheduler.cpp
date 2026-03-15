#include "runtime/scheduler.h"
#include "memory/kv_cache_manager.h"
#include "memory/kv_cache.h"  // for kKVBlockSize
#include <algorithm>

namespace imp {

Scheduler::Scheduler(int max_batch_size)
    : max_batch_size_(max_batch_size) {}

void Scheduler::add_request(std::shared_ptr<Request> req) {
    pending_.push_back(std::move(req));
}

void Scheduler::schedule(std::vector<std::shared_ptr<Request>>& prefill_batch,
                         std::vector<std::shared_ptr<Request>>& decode_batch) {
    prefill_batch.clear();
    decode_batch.clear();

    // 1. Remove finished/cancelled requests from active_
    active_.erase(
        std::remove_if(active_.begin(), active_.end(),
            [](const std::shared_ptr<Request>& r) {
                return r->status == RequestStatus::FINISHED ||
                       r->status == RequestStatus::CANCELLED;
            }),
        active_.end());

    // 2. Sort pending by ascending input token count (shortest-first)
    //    to reduce head-of-line blocking in continuous batching.
    std::sort(pending_.begin(), pending_.end(),
              [](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
                  return a->input_tokens.size() < b->input_tokens.size();
              });

    // 3. Promote pending requests to prefill (up to max_batch_size_ budget)
    while (!pending_.empty() &&
           static_cast<int>(active_.size()) < max_batch_size_) {
        auto& req = pending_.front();

        // Memory-aware check: estimate KV blocks needed for this request
        if (kv_manager_) {
            int ctx_len = req->context_len();
            int blocks_needed = (ctx_len + kKVBlockSize - 1) / kKVBlockSize;
            if (!kv_manager_->can_allocate(blocks_needed)) {
                // Not enough memory even with eviction -- stop admitting
                break;
            }
            // Reserve blocks, using prefix caching when enabled.
            if (kv_manager_->prefix_caching_enabled()) {
                int reused = kv_manager_->allocate_blocks_with_prefix(
                    req->id, req->input_tokens.data(),
                    static_cast<int>(req->input_tokens.size()));
                if (reused < 0) break;
                // Skip prefill for tokens covered by reused blocks.
                if (reused > 0) {
                    int skip = reused * kKVBlockSize;
                    int total = static_cast<int>(req->input_tokens.size());
                    if (skip >= total) skip = (total / kKVBlockSize) * kKVBlockSize;
                    if (skip >= total) skip = total - 1;
                    req->prefill_offset = skip;
                }
            } else {
                if (!kv_manager_->allocate_blocks(req->id, blocks_needed)) {
                    break;
                }
            }
        }

        auto r = pending_.front();
        pending_.pop_front();
        r->status = RequestStatus::PREFILLING;
        prefill_batch.push_back(r);
        active_.push_back(r);
    }

    // 3. Re-schedule incomplete PREFILLING requests (chunked prefill).
    //    Skip requests already in prefill_batch (just promoted from pending).
    for (auto& req : active_) {
        if (req->status == RequestStatus::PREFILLING &&
            req->prefill_offset > 0 &&
            req->prefill_offset < static_cast<int>(req->input_tokens.size())) {
            bool already_queued = false;
            for (const auto& pf : prefill_batch) {
                if (pf.get() == req.get()) { already_queued = true; break; }
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

bool Scheduler::has_pending() const {
    return !pending_.empty();
}

int Scheduler::active_count() const {
    return static_cast<int>(active_.size());
}

} // namespace imp
