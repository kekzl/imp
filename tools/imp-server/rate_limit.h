#pragma once

// Per-peer rate limiting extracted from ServerState so it is testable in the CPU lane (#1614,
// same move as bearer_token_matches) - ServerState pulls in the engine, which cannot construct there.

#include <atomic>
#include <chrono>
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
