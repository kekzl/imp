#pragma once

// Per-peer request rate limiting, extracted from ServerState so it can be
// tested in the CPU lane (#1614).
//
// The extraction is the same move `bearer_token_matches` made: ServerState
// pulls in BatchingEngine and the engine, so nothing that lives on it can be
// constructed in the only lane CI runs. A limit whose test cannot run is a
// limit that regresses silently.

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Requests per minute per key. 0 disables the limit entirely.
// In-flight admission for the inference routes (AUDIT_arch_2026 E-2). The
// --max-concurrent check read the engine's queue depth and admitted, with
// nothing between the read and the handler's submit, so N workers that read
// "63 in flight" at once all went through. One atomic counts admitted
// handlers instead: the pre-routing hook enters, the post-routing hook on the
// same worker thread leaves. Header-only so the CPU lane can drive it.
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

class RateLimiter {
public:
    int limit = 0;

    // Peers whose X-Forwarded-For is believed. Empty = the header is ignored.
    std::set<std::string> trusted_proxies;

    // The key a request counts against.
    //
    // X-Forwarded-For is believed only from a peer the operator named.
    // Otherwise it is a string the client writes, and believing it means one
    // client is an unlimited number of buckets: it both bypasses the limit and
    // grows the tracker without bound. A proxy appends, so the first element
    // is the original client and the rest are the chain.
    std::string key(const std::string& remote_addr, const std::string& xff) const;

    // True when the request is admitted.
    //
    // `now` is a parameter so the 60-second window and the eviction sweep are
    // testable without a 60-second test. Production calls take the default.
    bool allow(const std::string& k,
               std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    size_t tracked();

private:
    std::mutex mtx_;
    std::unordered_map<std::string, std::vector<std::chrono::steady_clock::time_point>> tracker_;
    int sweep_counter_ = 0;
};
