#include "rate_limit.h"

#include <algorithm>
#include <iterator>

namespace {
constexpr size_t kMaxKeyLen = 64;
constexpr int kWindowSeconds = 60;
// Amortised: one O(map) sweep per this many admissions, not per request.
constexpr int kSweepEvery = 256;
}  // namespace

std::string RateLimiter::key(const std::string& remote_addr, const std::string& xff) const {
    if (xff.empty() || trusted_proxies.find(remote_addr) == trusted_proxies.end())
        return remote_addr;

    const size_t comma = xff.find(',');
    std::string first = (comma == std::string::npos) ? xff : xff.substr(0, comma);
    const size_t b = first.find_first_not_of(" \t");
    const size_t e = first.find_last_not_of(" \t");
    if (b == std::string::npos)
        return remote_addr;
    first = first.substr(b, e - b + 1);
    // The value still comes off the wire and becomes a map key that outlives
    // the request.
    if (first.size() > kMaxKeyLen)
        first.resize(kMaxKeyLen);
    return first;
}

bool RateLimiter::allow(const std::string& k, std::chrono::steady_clock::time_point now) {
    if (limit <= 0)
        return true;
    std::lock_guard<std::mutex> lock(mtx_);
    const auto cutoff = now - std::chrono::seconds(kWindowSeconds);

    // The sweep is what keeps this bounded. Before it, only the bucket being
    // asked about was pruned, so every key ever seen stayed for the life of
    // the process - and the client chose the keys.
    if (++sweep_counter_ >= kSweepEvery) {
        sweep_counter_ = 0;
        for (auto it = tracker_.begin(); it != tracker_.end();) {
            auto& v = it->second;
            v.erase(std::remove_if(v.begin(), v.end(), [&](const auto& t) { return t < cutoff; }), v.end());
            it = v.empty() ? tracker_.erase(it) : std::next(it);
        }
    }

    auto& stamps = tracker_[k];
    stamps.erase(std::remove_if(stamps.begin(), stamps.end(), [&](const auto& t) { return t < cutoff; }),
                 stamps.end());
    if (static_cast<int>(stamps.size()) >= limit)
        return false;
    stamps.push_back(now);
    return true;
}

size_t RateLimiter::tracked() {
    std::lock_guard<std::mutex> lock(mtx_);
    return tracker_.size();
}
