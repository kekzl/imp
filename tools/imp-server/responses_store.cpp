#include "responses_store.h"

#include <chrono>
#include <utility>

namespace imp_server::responses {

namespace {

int64_t steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

size_t json_bytes(const json& j) {
    // replace: model output may carry invalid UTF-8, and a size estimate must not throw.
    return j.dump(-1, ' ', false, json::error_handler_t::replace).size();
}

}  // namespace

ResponseStore::ResponseStore(ResponseStoreLimits limits, NowMs now_ms)
    : limits_(limits), now_ms_(now_ms ? std::move(now_ms) : NowMs(steady_now_ms)) {}

void ResponseStore::set_limits(const ResponseStoreLimits& limits) {
    std::lock_guard<std::mutex> lock(mu_);
    limits_ = limits;
}

ResponseStoreLimits ResponseStore::limits() const {
    std::lock_guard<std::mutex> lock(mu_);
    return limits_;
}

size_t ResponseStore::entry_bytes(const std::string& id, const StoredResponse& e) {
    return id.size() + e.model.size() + json_bytes(e.input_items) + json_bytes(e.output_items) +
           json_bytes(e.response);
}

void ResponseStore::erase_locked(std::unordered_map<std::string, Slot>::iterator it) {
    bytes_ -= it->second.bytes;
    lru_.erase(it->second.lru_it);
    age_.erase(it->second.age_it);
    map_.erase(it);
}

void ResponseStore::purge_expired_locked(int64_t now) {
    while (!age_.empty()) {
        auto it = map_.find(age_.front());
        if (it == map_.end() || it->second.expires_ms > now)
            break;
        erase_locked(it);
        ++expirations_;
    }
}

bool ResponseStore::put(const std::string& id, StoredResponse entry) {
    // Serialization runs outside the lock: it is the only costly step.
    const size_t sz = entry_bytes(id, entry);
    auto value = std::make_shared<const StoredResponse>(std::move(entry));

    std::lock_guard<std::mutex> lock(mu_);
    if (!limits_.enabled() || sz > limits_.max_bytes)
        return false;
    const int64_t now = now_ms_();
    purge_expired_locked(now);
    if (auto old = map_.find(id); old != map_.end())
        erase_locked(old);

    lru_.push_front(id);
    age_.push_back(id);
    Slot slot;
    slot.value = std::move(value);
    slot.bytes = sz;
    slot.expires_ms = now + limits_.ttl_seconds * 1000;
    slot.lru_it = lru_.begin();
    slot.age_it = std::prev(age_.end());
    map_.emplace(id, std::move(slot));
    bytes_ += sz;

    while (map_.size() > limits_.max_entries || bytes_ > limits_.max_bytes) {
        erase_locked(map_.find(lru_.back()));
        ++evictions_;
    }
    return true;
}

std::shared_ptr<const StoredResponse> ResponseStore::get(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    purge_expired_locked(now_ms_());
    auto it = map_.find(id);
    if (it == map_.end())
        return nullptr;
    lru_.splice(lru_.begin(), lru_, it->second.lru_it);
    return it->second.value;
}

bool ResponseStore::erase(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    purge_expired_locked(now_ms_());
    auto it = map_.find(id);
    if (it == map_.end())
        return false;
    erase_locked(it);
    return true;
}

void ResponseStore::purge_expired() {
    std::lock_guard<std::mutex> lock(mu_);
    purge_expired_locked(now_ms_());
}

size_t ResponseStore::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return map_.size();
}

size_t ResponseStore::bytes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return bytes_;
}

uint64_t ResponseStore::evictions() const {
    std::lock_guard<std::mutex> lock(mu_);
    return evictions_;
}

uint64_t ResponseStore::expirations() const {
    std::lock_guard<std::mutex> lock(mu_);
    return expirations_;
}

}  // namespace imp_server::responses
