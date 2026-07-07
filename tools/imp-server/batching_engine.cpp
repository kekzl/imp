#include "batching_engine.h"
#include "core/logging.h"
#include "model/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <exception>
#include <stdexcept>

namespace {
// A kernel illegal-access (or similar) poisons the whole process-wide CUDA
// context: every later launch returns the same sticky error, which forward()
// silently clears and ignores, so the engine would serve garbage forever. These
// classes are NOT recoverable without a process restart. A plain host-side throw
// (the common case the step() try/catch was written for) leaves the device clean
// — cudaDeviceSynchronize() returns success — so we recover and continue as before.
bool cuda_error_is_unrecoverable(cudaError_t e) {
    switch (e) {
        case cudaErrorIllegalAddress:
        case cudaErrorMisalignedAddress:
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction:
        case cudaErrorECCUncorrectable:
        case cudaErrorExternalDevice:
            return true;
        default:
            return false;  // conservative: don't kill the server on recoverable/unknown errors
    }
}
}  // namespace

BatchingEngine::~BatchingEngine() { stop(); }

void BatchingEngine::start(ImpContext ctx) {
    if (running_.load()) {
        stop();
    }
    ctx_ = ctx;
    stop_requested_.store(false);
    faulted_.store(false);
    pause_requested_.store(false);
    paused_.store(false);
    running_.store(true);
    worker_thread_ = std::thread(&BatchingEngine::worker_loop, this);
}

void BatchingEngine::stop() {
    if (!running_.load())
        return;
    stop_requested_.store(true);
    queue_cv_.notify_all();
    pause_cv_.notify_all();  // wake the worker if it is parked in pause()
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

bool BatchingEngine::pause(int timeout_ms) {
    if (!running_.load())
        return true;  // nothing running → caller already has exclusive access
    pause_requested_.store(true);
    queue_cv_.notify_all();  // wake the worker if it is idle-waiting
    std::unique_lock<std::mutex> lock(pause_mutex_);
    bool ok = pause_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return paused_.load() || !running_.load() || stop_requested_.load();
    });
    if (!ok) {
        // Timed out waiting for the worker to drain in-flight work. Do NOT
        // proceed (driving engine->step() while the worker is live would race);
        // clear the request so the worker keeps running normally.
        pause_requested_.store(false);
        queue_cv_.notify_all();
        return false;
    }
    return true;
}

void BatchingEngine::resume() {
    pause_requested_.store(false);
    pause_cv_.notify_all();
    queue_cv_.notify_all();
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
    return static_cast<int>(pending_queue_.size()) + static_cast<int>(active_requests_.size());
}

void BatchingEngine::worker_loop() {
    imp::Engine* engine = ctx_->engine.get();
    imp::KVCacheManager* kv_mgr = engine->kv_manager();

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        // 0. Graceful pause handshake. When a caller wants exclusive engine
        //    access (embeddings / blocking vision), it calls pause(). We must
        //    NOT cancel in-flight generations — instead we keep stepping until
        //    active work has drained, then park here until resume(). New
        //    pending requests are left queued (not admitted, not cancelled) so
        //    they run after resume. The caller holds state.mtx, so no new chat
        //    can be submitted while we are parked.
        if (pause_requested_.load(std::memory_order_relaxed) && active_requests_.empty()) {
            std::unique_lock<std::mutex> lock(pause_mutex_);
            paused_.store(true);
            pause_cv_.notify_all();
            pause_cv_.wait(lock, [this] {
                return !pause_requested_.load(std::memory_order_relaxed) ||
                       stop_requested_.load(std::memory_order_relaxed);
            });
            paused_.store(false);
            continue;
        }

        // 1. Drain incoming queue and add new requests to the scheduler
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // If no active work, wait for new requests (with timeout to check
            // stop / pause). When a pause is pending we still need to wake so
            // step 0 can park, hence the predicate also watches pause_requested_.
            if (active_requests_.empty() && pending_queue_.empty()) {
                queue_cv_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                    return !pending_queue_.empty() ||
                           stop_requested_.load(std::memory_order_relaxed) ||
                           pause_requested_.load(std::memory_order_relaxed);
                });
                if (stop_requested_.load(std::memory_order_relaxed))
                    break;
            }

            // While a pause is pending, do not admit new pending requests —
            // let active work drain so step 0 can park. Leaving them queued
            // preserves them for after resume().
            if (!pause_requested_.load(std::memory_order_relaxed)) {
                // Move all pending requests to active
                while (!pending_queue_.empty()) {
                    auto sr = std::move(pending_queue_.front());
                    pending_queue_.pop_front();

                    // Per-request vision: encode the image on THIS worker thread
                    // (the encode is serialized + uses the shared encoder
                    // workspace, so it must not run concurrently with step()).
                    // The result lands in request->vision_emb; the request then
                    // batches with text like any other — no engine pause.
                    if (sr->request->image && !engine->encode_image_for(*sr->request)) {
                        sr->request->status = imp::RequestStatus::CANCELLED;
                        sr->push_finish("image_encode_failed");
                        continue;
                    }

                    sr->notified_count = sr->request->output_tokens.size();
                    engine->add_request(sr->request);
                    active_requests_.push_back(std::move(sr));
                }
            }
        }

        if (active_requests_.empty())
            continue;

        // 2. Check for cancelled requests before stepping
        for (auto& sr : active_requests_) {
            if (sr->is_cancelled() && sr->request->status != imp::RequestStatus::FINISHED &&
                sr->request->status != imp::RequestStatus::CANCELLED) {
                sr->request->status = imp::RequestStatus::CANCELLED;
                kv_mgr->free_sequence(sr->request->id);
                engine->reset_ssm_state(sr->request->id);
            }
        }

        // 3. Run one engine step (processes all scheduled requests).
        // Wrap in try/catch — a bug in any model/quant/cuda path that throws
        // mid-forward used to call std::terminate and kill the container,
        // taking every other in-flight request with it. Now we cancel all
        // active requests with a clear reason and keep the worker alive.
        try {
            (void)engine->step();
        } catch (const std::exception& e) {
            IMP_LOG_ERROR("BatchingEngine: engine->step() threw %s — cancelling %zu active request(s)",
                          e.what(), active_requests_.size());
            for (auto& sr : active_requests_) {
                if (sr->request->status != imp::RequestStatus::FINISHED &&
                    sr->request->status != imp::RequestStatus::CANCELLED) {
                    sr->request->status = imp::RequestStatus::CANCELLED;
                    kv_mgr->free_sequence(sr->request->id);
                    engine->reset_ssm_state(sr->request->id);
                }
                sr->push_finish("internal_error");
            }
            // Distinguish a host-side throw (device clean — recover) from a CUDA
            // fault that poisoned the shared context (no recovery without a
            // restart). Without this, a poisoned context is silently cleared at
            // the next forward() and the server returns garbage on every request.
            {
                cudaError_t sync_err = cudaDeviceSynchronize();
                cudaError_t sticky = cudaGetLastError();
                if (cuda_error_is_unrecoverable(sync_err) || cuda_error_is_unrecoverable(sticky)) {
                    IMP_LOG_ERROR("BatchingEngine: CUDA context poisoned (%s / %s) — stopping the "
                                  "worker; the server must be restarted.",
                                  cudaGetErrorString(sync_err), cudaGetErrorString(sticky));
                    faulted_.store(true, std::memory_order_relaxed);
                    stop_requested_.store(true, std::memory_order_relaxed);
                }
            }
            // Reset engine-level transient state so the next request starts clean.
            engine->invalidate_graphs();
            engine->reset_batch_pool_cache();
            active_requests_.clear();
            continue;
        } catch (...) {
            IMP_LOG_ERROR(
                "BatchingEngine: engine->step() threw non-std exception — cancelling %zu active request(s)",
                active_requests_.size());
            for (auto& sr : active_requests_) {
                if (sr->request->status != imp::RequestStatus::FINISHED &&
                    sr->request->status != imp::RequestStatus::CANCELLED) {
                    sr->request->status = imp::RequestStatus::CANCELLED;
                    kv_mgr->free_sequence(sr->request->id);
                    engine->reset_ssm_state(sr->request->id);
                }
                sr->push_finish("internal_error");
            }
            {
                cudaError_t sync_err = cudaDeviceSynchronize();
                cudaError_t sticky = cudaGetLastError();
                if (cuda_error_is_unrecoverable(sync_err) || cuda_error_is_unrecoverable(sticky)) {
                    IMP_LOG_ERROR("BatchingEngine: CUDA context poisoned (%s / %s) — stopping the "
                                  "worker; the server must be restarted.",
                                  cudaGetErrorString(sync_err), cudaGetErrorString(sticky));
                    faulted_.store(true, std::memory_order_relaxed);
                    stop_requested_.store(true, std::memory_order_relaxed);
                }
            }
            engine->invalidate_graphs();
            engine->reset_batch_pool_cache();
            active_requests_.clear();
            continue;
        }

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
                    bool is_stop_token = false;
                    if (req->status == imp::RequestStatus::CANCELLED) {
                        reason = "cancelled";
                    } else {
                        if (token == tok->eos_id()) {
                            reason = "stop";
                            is_stop_token = true;
                        } else {
                            for (int32_t sid : stop_ids) {
                                if (token == sid) {
                                    reason = "stop";
                                    is_stop_token = true;
                                    break;
                                }
                            }
                        }
                        // Banned tokens (e.g. <pad>) that triggered stop
                        if (!is_stop_token) {
                            for (int32_t bid : engine->banned_token_ids()) {
                                if (token == bid) {
                                    reason = "stop";
                                    is_stop_token = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (is_stop_token) {
                        // Don't include stop/banned tokens in output text
                        sr->push_finish(reason);
                    } else {
                        sr->push_token(token, true, reason);
                    }
                } else {
                    sr->push_token(token, false, nullptr);
                }
            }
            sr->notified_count = current_count;

            if (is_done) {
                // If we didn't push a finish event via the token loop
                // (request ended with no new tokens this step), push one now.
                if (!had_new_tokens) {
                    const char* reason = (req->status == imp::RequestStatus::CANCELLED) ? "cancelled"
                                                                                        : "length";
                    if (!req->output_tokens.empty()) {
                        int32_t last = req->output_tokens.back();
                        if (last == tok->eos_id())
                            reason = "stop";
                        for (int32_t sid : stop_ids) {
                            if (last == sid) {
                                reason = "stop";
                                break;
                            }
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
