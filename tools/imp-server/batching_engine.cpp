#include "batching_engine.h"
#include "core/logging.h"
#include "model/tokenizer.h"

#include <algorithm>
#include <chrono>

BatchingEngine::~BatchingEngine() {
    stop();
}

void BatchingEngine::start(ImpContext ctx) {
    if (running_.load()) {
        stop();
    }
    ctx_ = ctx;
    stop_requested_.store(false);
    running_.store(true);
    worker_thread_ = std::thread(&BatchingEngine::worker_loop, this);
}

void BatchingEngine::stop() {
    if (!running_.load()) return;
    stop_requested_.store(true);
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    running_.store(false);

    // Cancel any remaining pending requests
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto& sr : pending_queue_) {
            sr->push_finish("cancelled");
        }
        pending_queue_.clear();
    }

    // Cancel active requests
    for (auto& sr : active_requests_) {
        sr->push_finish("cancelled");
    }
    active_requests_.clear();
}

void BatchingEngine::submit(std::shared_ptr<ServerRequest> req) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_queue_.push_back(std::move(req));
    }
    queue_cv_.notify_one();
}

int BatchingEngine::queue_depth() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return static_cast<int>(pending_queue_.size()) +
           static_cast<int>(active_requests_.size());
}

void BatchingEngine::worker_loop() {
    imp::Engine* engine = ctx_->engine.get();
    imp::KVCacheManager* kv_mgr = engine->kv_manager();

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        // 1. Drain incoming queue and add new requests to the scheduler
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // If no active work, wait for new requests (with timeout to check stop)
            if (active_requests_.empty() && pending_queue_.empty()) {
                queue_cv_.wait_for(lock, std::chrono::milliseconds(10),
                    [this] {
                        return !pending_queue_.empty() ||
                               stop_requested_.load(std::memory_order_relaxed);
                    });
                if (stop_requested_.load(std::memory_order_relaxed)) break;
            }

            // Move all pending requests to active
            while (!pending_queue_.empty()) {
                auto sr = std::move(pending_queue_.front());
                pending_queue_.pop_front();

                // Clear any stale active_request on the context.
                // The batching engine manages requests through the scheduler,
                // not through ctx->active_request.
                if (ctx_->active_request) {
                    kv_mgr->free_sequence(ctx_->active_request->id);
                    engine->reset_ssm_state(ctx_->active_request->id);
                    ctx_->active_request->status = imp::RequestStatus::CANCELLED;
                    ctx_->active_request = nullptr;
                }

                // Initialize notified_count to current output size (usually 0)
                sr->notified_count = sr->request->output_tokens.size();

                engine->add_request(sr->request);
                active_requests_.push_back(std::move(sr));
            }
        }

        if (active_requests_.empty()) continue;

        // 2. Check for cancelled requests before stepping
        for (auto& sr : active_requests_) {
            if (sr->is_cancelled() &&
                sr->request->status != imp::RequestStatus::FINISHED &&
                sr->request->status != imp::RequestStatus::CANCELLED) {
                sr->request->status = imp::RequestStatus::CANCELLED;
                kv_mgr->free_sequence(sr->request->id);
                engine->reset_ssm_state(sr->request->id);
            }
        }

        // 3. Run one engine step (processes all scheduled requests)
        engine->step();

        // 4. Deliver new tokens and check for completion
        imp::Tokenizer* tok = engine->model()->tokenizer();
        const auto& stop_ids = engine->chat_template().stop_token_ids();

        auto it = active_requests_.begin();
        while (it != active_requests_.end()) {
            auto& sr = *it;
            auto& req = sr->request;
            size_t current_count = req->output_tokens.size();
            bool is_done = (req->status == imp::RequestStatus::FINISHED ||
                            req->status == imp::RequestStatus::CANCELLED);
            bool had_new_tokens = (current_count > sr->notified_count);

            // Deliver new tokens
            for (size_t i = sr->notified_count; i < current_count; i++) {
                int32_t token = req->output_tokens[i];
                bool is_last_token = is_done && (i == current_count - 1);

                if (is_last_token) {
                    // Determine finish reason
                    const char* reason = "length";
                    if (req->status == imp::RequestStatus::CANCELLED) {
                        reason = "cancelled";
                    } else {
                        if (token == tok->eos_id()) {
                            reason = "stop";
                        } else {
                            for (int32_t sid : stop_ids) {
                                if (token == sid) {
                                    reason = "stop";
                                    break;
                                }
                            }
                        }
                    }
                    sr->push_token(token, true, reason);
                } else {
                    sr->push_token(token, false, nullptr);
                }
            }
            sr->notified_count = current_count;

            if (is_done) {
                // If we didn't push a finish event via the token loop
                // (request ended with no new tokens this step), push one now.
                if (!had_new_tokens) {
                    const char* reason = (req->status == imp::RequestStatus::CANCELLED)
                                         ? "cancelled" : "length";
                    if (!req->output_tokens.empty()) {
                        int32_t last = req->output_tokens.back();
                        if (last == tok->eos_id()) reason = "stop";
                        for (int32_t sid : stop_ids) {
                            if (last == sid) { reason = "stop"; break; }
                        }
                    }
                    sr->push_finish(reason);
                }
                it = active_requests_.erase(it);
            } else {
                ++it;
            }
        }
    }
}
