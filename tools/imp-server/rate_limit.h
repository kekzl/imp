#pragma once

// Per-peer request rate limiting, extracted from ServerState so it can be
// tested in the CPU lane (#1614).
//
// The extraction is the same move `bearer_token_matches` made: ServerState
// pulls in BatchingEngine and the engine, so nothing that lives on it can be
// constructed in the only lane CI runs. A limit whose test cannot run is a
// limit that regresses silently.

#include <chrono>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Requests per minute per key. 0 disables the limit entirely.
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
