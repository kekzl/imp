#pragma once

// Per-peer rate limiting extracted from ServerState so it is testable in the CPU lane (#1614,
// same move as bearer_token_matches) - ServerState pulls in the engine, which cannot construct there.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// InflightGate: atomic in-flight admission (AUDIT_arch_2026 E-2) - a raw queue-depth read then
// admit let N workers reading "63 in flight" simultaneously all pass. Pre-routing hook enters,
// post-routing hook (same thread) leaves. Header-only so the CPU lane can drive it.
class InflightGate {
public:
    // True and counted when there is room; false and NOT counted when
    // `limit` handlers are already in flight. limit <= 0 means unlimited.
    bool try_enter(int limit) {
        const int n = inflight_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (limit > 0 && n > limit) {
            inflight_.fetch_sub(1, std::memory_order_acq_rel);
            return false;
        }
        return true;
    }
    void leave() { inflight_.fetch_sub(1, std::memory_order_acq_rel); }
    int inflight() const { return inflight_.load(std::memory_order_relaxed); }

private:
    std::atomic<int> inflight_{0};
};

// QueuedTokenGate (#2408): prompt tokens admitted and not yet prefilled, bounded by
// --max-queued-tokens. Refuses when queued + n > cap; an empty queue always admits (a lone
// prompt over the cap is --max-input-tokens' job). cap <= 0 = off, still counted for /metrics.
class QueuedTokenGate {
public:
    bool try_reserve(int64_t n, int64_t cap) {
        int64_t cur = queued_.load(std::memory_order_acquire);
        do {
            if (cap > 0 && cur > 0 && cur + n > cap)
                return false;
        } while (!queued_.compare_exchange_weak(cur, cur + n, std::memory_order_acq_rel,
                                                std::memory_order_acquire));
        return true;
    }
    void release(int64_t n) { queued_.fetch_sub(n, std::memory_order_acq_rel); }
    int64_t queued() const { return queued_.load(std::memory_order_relaxed); }

private:
    std::atomic<int64_t> queued_{0};
};

// One request's reservation. release() is idempotent: the batching worker calls it at the first
// output token, the destructor covers every path that never got one.
class QueuedTokenLease {
public:
    QueuedTokenLease(QueuedTokenGate& gate, int64_t n) : gate_(&gate), n_(n) {}
    ~QueuedTokenLease() { release(); }
    QueuedTokenLease(const QueuedTokenLease&) = delete;
    QueuedTokenLease& operator=(const QueuedTokenLease&) = delete;
    void release() {
        const int64_t n = n_.exchange(0, std::memory_order_acq_rel);
        if (n != 0)
            gate_->release(n);
    }

private:
    QueuedTokenGate* gate_;
    std::atomic<int64_t> n_;
};

// Null when `n` does not fit under `cap`; nothing is counted then.
inline std::shared_ptr<QueuedTokenLease> reserve_queued_tokens(QueuedTokenGate& gate, int64_t n,
                                                               int64_t cap) {
    if (!gate.try_reserve(n, cap))
        return nullptr;
    return std::make_shared<QueuedTokenLease>(gate, n);
}

namespace httplib {
struct Response;
}  // namespace httplib

// Retry-After on the queued-token refusal: 1 s, the smallest whole value the header carries.
inline constexpr int kQueuedTokensRetryAfterSeconds = 1;

// Reserves `n` prompt tokens, or answers 429 overloaded_error (dialect of `path`) with
// Retry-After and returns null.
std::shared_ptr<QueuedTokenLease> admit_queued_tokens(const std::string& path, httplib::Response& res,
                                                      QueuedTokenGate& gate, int64_t n, int64_t cap);

class RateLimiter {
public:
    int limit = 0;

    // Peers whose X-Forwarded-For is believed. Empty = the header is ignored.
    std::set<std::string> trusted_proxies;

    // key(): X-Forwarded-For is trusted only from an operator-named peer - otherwise a client-
    // writable header would let one client become unlimited rate-limit buckets. A proxy appends, so
    // the first element is the original client.
    std::string key(const std::string& remote_addr, const std::string& xff) const;

    // allow(): `now` is a parameter so the 60-second window and eviction sweep are testable without
    // a real 60-second wait; production calls use the default (wall clock).
    bool allow(const std::string& k,
               std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    size_t tracked();

private:
    std::mutex mtx_;
    std::unordered_map<std::string, std::vector<std::chrono::steady_clock::time_point>> tracker_;
    int sweep_counter_ = 0;
};
