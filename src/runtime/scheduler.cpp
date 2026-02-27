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
            // Reserve blocks under req->id so subsequent requests see reduced
            // availability and the engine can reuse them during prefill.
            if (!kv_manager_->allocate_blocks(req->id, blocks_needed)) {
                break;
            }
        }

        auto r = pending_.front();
        pending_.pop_front();
        r->status = RequestStatus::PREFILLING;
        prefill_batch.push_back(r);
        active_.push_back(r);
    }

    // 3. All active decoding requests go to the decode batch
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
