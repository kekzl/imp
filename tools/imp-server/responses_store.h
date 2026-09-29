// In-memory store behind Responses `store: true` / `previous_response_id` (#2206).
// Bounded by TTL (from insertion), entry count and bytes; LRU eviction. Own mutex, held only for
// map/list updates, never across generation. Entries are immutable shared_ptrs.

#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace imp_server::responses {

using json = nlohmann::json;

struct StoredResponse {
    json input_items;   // full conversation input as the model saw it (chained turns flattened)
    json output_items;  // this turn's output items (reasoning, message, function_call)
    json response;      // the response object as returned, served by GET /v1/responses/{id}
    std::string model;
    int64_t created_at = 0;  // unix seconds
};

struct ResponseStoreLimits {
    int64_t ttl_seconds = 3600;
    size_t max_entries = 1000;
    size_t max_bytes = size_t{256} << 20;
    // Any limit at 0 disables the store: store=true is then refused with 400.
    bool enabled() const { return ttl_seconds > 0 && max_entries > 0 && max_bytes > 0; }
};

class ResponseStore {
public:
    using NowMs = std::function<int64_t()>;  // monotonic milliseconds, injectable for tests

    explicit ResponseStore(ResponseStoreLimits limits = {}, NowMs now_ms = {});

    void set_limits(const ResponseStoreLimits& limits);
    ResponseStoreLimits limits() const;
    bool enabled() const { return limits().enabled(); }

    // Inserts (or replaces) `id`, then evicts least recently used entries until both caps hold.
    // False when the entry alone exceeds max_bytes or the store is disabled: nothing is stored.
    bool put(const std::string& id, StoredResponse entry);

    // nullptr for unknown or expired ids; a hit refreshes LRU order, not the TTL.
    std::shared_ptr<const StoredResponse> get(const std::string& id);

    bool erase(const std::string& id);

    // Drops expired entries now (otherwise done lazily by put/get/erase); /metrics calls it.
    void purge_expired();

    size_t size() const;
    size_t bytes() const;
    uint64_t evictions() const;    // removed by the entry or byte cap
    uint64_t expirations() const;  // removed by the TTL

    // Approximate footprint of one entry: serialized JSON sizes plus the key.
    static size_t entry_bytes(const std::string& id, const StoredResponse& e);

private:
    struct Slot {
        std::shared_ptr<const StoredResponse> value;
        size_t bytes = 0;
        int64_t expires_ms = 0;
        std::list<std::string>::iterator lru_it;  // front = most recently used
        std::list<std::string>::iterator age_it;  // front = oldest insertion
    };

    void purge_expired_locked(int64_t now);
    void erase_locked(std::unordered_map<std::string, Slot>::iterator it);

    mutable std::mutex mu_;
    ResponseStoreLimits limits_;
    NowMs now_ms_;
    std::unordered_map<std::string, Slot> map_;
    std::list<std::string> lru_;
    std::list<std::string> age_;  // TTL is constant, so insertion order is expiry order
    size_t bytes_ = 0;
    uint64_t evictions_ = 0;
    uint64_t expirations_ = 0;
};

}  // namespace imp_server::responses
