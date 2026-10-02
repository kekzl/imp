#pragma once

#include "runtime/request.h"
#include "api/imp_internal.h"
#include "rate_limit.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <string>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace imp {
class KVCacheManager;
}  // namespace imp

// Token delivered from the worker thread to the HTTP handler.
struct TokenEvent {
    int32_t token_id;
    bool is_last;               // true if this is the final delivery (request done)
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
    std::atomic<bool> cancelled{false};  // set by HTTP handler to cancel

    // Track how many output tokens we have already delivered
    size_t notified_count = 0;

    // Queue observability (#1580): t_submit set by submit(); queue_ms filled when the worker
    // moves this request from pending to active, i.e. time waiting behind other requests, NOT
    // prefill. Previously unmeasured, so "the server is slow" and "the server is busy" looked
    // the same from outside.
    std::chrono::steady_clock::time_point t_submit{};
    std::atomic<double> queue_ms{-1.0};

    // --max-queued-tokens reservation (#2408); the worker releases it at the first output token
    // or at finish, the destructor on every other path. Null = not counted.
    std::shared_ptr<QueuedTokenLease> queued_lease;

    // POST /v1/requests/{id}/end_thinking (#2420). public_ids: completion/message/response id and
    // the client X-Request-Id, set before submit. HTTP sets end_thinking; the worker copies it into
    // request->end_thinking before its next step and publishes think_phase after each step.
    enum ThinkPhase : int {
        kThinkUnknown = -1,
        kThinkOff = 0,
        kThinkOn = 1,
        kThinkNoCloser = 2,
        kThinkDone = 3
    };
    std::vector<std::string> public_ids;
    std::atomic<bool> end_thinking{false};
    std::atomic<int> think_phase{kThinkUnknown};

    // Push a token event (called from worker thread)
    void push_token(int32_t token_id, bool is_last, const char* reason) {
        if (is_last)
            think_phase.store(kThinkDone, std::memory_order_release);
        std::lock_guard<std::mutex> lock(token_mutex);
        token_queue.push_back({token_id, is_last, reason});
        token_cv.notify_one();
    }

    // Push a completion event with no token (called from worker thread)
    void push_finish(const char* reason) {
        think_phase.store(kThinkDone, std::memory_order_release);
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
    void cancel() { cancelled.store(true, std::memory_order_release); }

    bool is_cancelled() const { return cancelled.load(std::memory_order_acquire); }
};

// Continuous batching engine running inference in a background thread: HTTP handlers submit
// ServerRequest objects, the worker thread runs the engine step loop, processing multiple
// requests simultaneously via the scheduler.
class BatchingEngine {
public:
    // Decode-batch observability (#1580): the knob that bounds batch size is
    // configurable and the result was unobservable. Counter pair rather than an
    // average, so a dashboard can rate() both and divide over any window.
    std::atomic<int64_t> decode_steps{0};
    std::atomic<int64_t> decode_rows{0};
    std::atomic<int64_t> decode_batch_max{0};
    // Rows of the most recent step, 0 while the worker idles: the sequences
    // decoding together right now. The counter pair above gives only the
    // windowed mean and decode_batch_max never resets.
    std::atomic<int64_t> decode_batch_last{0};
    // Queue split for /metrics (AUDIT_arch_2026 C-5): waiting = submitted but not yet in a
    // prefill/decode batch (pending_queue_, or held behind max_batch_size/KV admission); running =
    // in a batch. queue_depth() is their sum, refreshed by the worker every loop so a scrape never
    // reads scheduler state directly.
    std::atomic<int64_t> queue_waiting{0};
    std::atomic<int64_t> queue_running{0};

    BatchingEngine() = default;
    ~BatchingEngine();

    // Initialize with an existing ImpContext (takes non-owning reference).
    // Starts the background worker thread.
    void start(ImpContext ctx);

    // Stops the background worker thread; must be called before destroying the ImpContext, waits
    // for the worker to finish. Cancels any in-flight requests - use pause()/resume() instead when
    // generations must survive (e.g. embeddings/vision exclusive-access windows).
    void stop();

    // Graceful exclusive-access handshake: pause() lets the worker FINISH in-flight requests
    // (never cancels), then parks the worker thread so the caller can drive engine->step() directly
    // without racing it; resume() unparks. Caller MUST hold state.mtx during the window (chat
    // submit also takes it). pause() blocks until parked, bounded by timeout_ms (0 = no work to
    // drain, returns immediately). Returns true once parked.
    bool pause(int timeout_ms = 60000);
    void resume();

    // Submit a request for inference. Thread-safe.
    // The request will be picked up by the worker on the next iteration.
    void submit(std::shared_ptr<ServerRequest> req);
    // All of `reqs` enter the queue under one lock: the worker admits them in the same
    // iteration, so a /v1/decide shared wave has one batch composition (#2198).
    void submit_all(const std::vector<std::shared_ptr<ServerRequest>>& reqs);

    // POST /v1/sessions/{id}/close (#2407). Thread-safe: the worker releases the pin before its next step.
    void close_session(std::string session_id);

    // POST /v1/requests/{id}/end_thinking (#2420). Thread-safe, idempotent. NotFound: no pending or
    // running request carries `id`. Ending: flag set (also before admission). AlreadyClosed: not in a
    // think block. NoCloser: in a think block the model has no closer id for.
    enum class EndThinking { NotFound, Ending, AlreadyClosed, NoCloser };
    // first_call: true when this call set the flag (a repeat returns Ending with false).
    EndThinking end_thinking(const std::string& id, bool* first_call = nullptr);

    // Returns the number of active + pending requests.
    int queue_depth() const;

    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    // True after the worker declares the CUDA context poisoned and stops (#874): /health reports
    // unhealthy so an orchestrator can restart the process, instead of answering "ok" while every
    // request fails with internal_error.
    bool faulted() const { return faulted_.load(std::memory_order_relaxed); }

private:
    void worker_loop();
    // Cancels every active request with finish "internal_error", re-probes the device and, on
    // an unrecoverable class, sets faulted_ and stops the worker (#874, AUDIT_arch_2026 D-1).
    // `observed` is the error the caller already holds (cudaSuccess when the trigger was a host throw).
    void fail_active_requests_(const std::string& why, cudaError_t observed);

    ImpContext ctx_ = nullptr;  // non-owning

    std::thread worker_thread_;
    // Deferred delivery: every push_token's notify_one used to wake an SSE handler that ran
    // (detokenise + socket write) before the worker regained the core - at 32 streams that
    // serialised ~6.4ms/step of handler work INTO the GPU driver loop (19% of the step period). The
    // worker now hands the step's events to this thread in one batch and proceeds; the notifier
    // wakes clients while the GPU is busy. Per-request ordering preserved (one notifier, FIFO).
    struct PendingDelivery {
        std::shared_ptr<ServerRequest> sr;
        int32_t token_id;
        bool is_last;
        const char* reason;  // nullptr = plain token; is_last: finish reason
        bool finish_only;    // push_finish instead of push_token
    };
    std::thread notify_thread_;
    std::mutex nq_mutex_;
    std::condition_variable nq_cv_;
    std::vector<PendingDelivery> nq_;
    void notify_loop_();
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> faulted_{false};

    // Graceful pause handshake (see pause()/resume()). pause_requested_ tells
    // the worker to drain in-flight work and park; paused_ reports that it has.
    std::atomic<bool> pause_requested_{false};
    std::atomic<bool> paused_{false};
    std::mutex pause_mutex_;
    std::condition_variable pause_cv_;

    // Incoming request queue (HTTP threads -> worker thread)
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<ServerRequest>> pending_queue_;

    // Active requests being processed by the engine
    std::vector<std::shared_ptr<ServerRequest>> active_requests_;

    // Session closes (HTTP threads -> worker) and the TTL sweep, run at most once per second.
    std::mutex session_mutex_;
    std::vector<std::string> session_closes_;
    std::chrono::steady_clock::time_point next_session_sweep_{};
    void apply_session_ops_(imp::KVCacheManager* kv, int ttl_s);

    // public id -> request (#2420). Weak: a finished request expires; retire_ids_ drops it at finish.
    std::mutex ids_mutex_;
    std::unordered_map<std::string, std::weak_ptr<ServerRequest>> by_public_id_;
    void register_ids_(const std::shared_ptr<ServerRequest>& sr);
    void retire_ids_(const std::shared_ptr<ServerRequest>& sr);
};
