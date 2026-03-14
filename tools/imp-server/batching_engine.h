#pragma once

#include "runtime/engine.h"
#include "runtime/request.h"
#include "api/imp_internal.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Token delivered from the worker thread to the HTTP handler.
struct TokenEvent {
    int32_t token_id;
    bool is_last;           // true if this is the final delivery (request done)
    const char* finish_reason;  // non-null on last token: "stop", "length", "cancelled"
};

// A server-level request submitted to the batching engine.
// Wraps an imp::Request and adds a token queue for the HTTP handler to read.
struct ServerRequest {
    std::shared_ptr<imp::Request> request;

    // Token queue: worker pushes, HTTP handler pops.
    // Protected by token_mutex / token_cv.
    std::mutex token_mutex;
    std::condition_variable token_cv;
    std::deque<TokenEvent> token_queue;
    bool cancelled = false;  // set by HTTP handler to cancel

    // Track how many output tokens we have already delivered
    size_t notified_count = 0;

    // Push a token event (called from worker thread)
    void push_token(int32_t token_id, bool is_last, const char* reason) {
        std::lock_guard<std::mutex> lock(token_mutex);
        token_queue.push_back({token_id, is_last, reason});
        token_cv.notify_one();
    }

    // Push a completion event with no token (called from worker thread)
    void push_finish(const char* reason) {
        std::lock_guard<std::mutex> lock(token_mutex);
        token_queue.push_back({-1, true, reason});
        token_cv.notify_one();
    }

    // Pop next token event, blocking until available or timeout.
    // Returns false on timeout (caller should check client disconnect).
    bool pop_token(TokenEvent& out, int timeout_ms = 500) {
        std::unique_lock<std::mutex> lock(token_mutex);
        if (!token_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                               [this] { return !token_queue.empty(); })) {
            return false;  // timeout — caller should check is_writable
        }
        out = token_queue.front();
        token_queue.pop_front();
        return true;
    }

    // Cancel the request (called from HTTP handler if client disconnects)
    void cancel() {
        std::lock_guard<std::mutex> lock(token_mutex);
        cancelled = true;
    }

    bool is_cancelled() {
        std::lock_guard<std::mutex> lock(token_mutex);
        return cancelled;
    }
};

// Continuous batching engine that runs inference in a background thread.
// HTTP handlers submit ServerRequest objects; the worker thread runs the
// engine step loop, processing multiple requests simultaneously via the
// scheduler.
class BatchingEngine {
public:
    BatchingEngine() = default;
    ~BatchingEngine();

    // Initialize with an existing ImpContext (takes non-owning reference).
    // Starts the background worker thread.
    void start(ImpContext ctx);

    // Stop the background worker thread. Must be called before destroying
    // the ImpContext. Waits for the worker to finish.
    void stop();

    // Submit a request for inference. Thread-safe.
    // The request will be picked up by the worker on the next iteration.
    void submit(std::shared_ptr<ServerRequest> req);

    // Returns the number of active + pending requests.
    int queue_depth() const;

    bool is_running() const { return running_.load(std::memory_order_relaxed); }

private:
    void worker_loop();

    ImpContext ctx_ = nullptr;  // non-owning

    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    // Incoming request queue (HTTP threads -> worker thread)
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<ServerRequest>> pending_queue_;

    // Active requests being processed by the engine
    std::vector<std::shared_ptr<ServerRequest>> active_requests_;
};
