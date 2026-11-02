#include "runtime/prefill_graph_cache.h"

#include <algorithm>

namespace imp {

void PrefillGraphCache::set_short_max(int tokens) {
    short_max_ = std::max(0, tokens);
    std::erase_if(entries_, [this](const Entry& e) { return is_short_(e.key); });
}

CudaGraphRunner* PrefillGraphCache::select(const Key& key) {
    for (auto& e : entries_) {
        if (e.key == key)
            return e.runner.get();
    }
    if (!is_short_(key)) {
        // Long keys capture on a second CONSECUTIVE sighting: alternating lengths stay eager
        // instead of re-capturing the shared runner every request.
        if (key != pending_long_) {
            pending_long_ = key;
            return nullptr;
        }
        pending_long_ = Key{-1, -1, 0, -1};
        // The shared runner: its exec is updated in place when the topology matches.
        for (auto& e : entries_) {
            if (!is_short_(e.key)) {
                e.key = key;
                e.runner->invalidate_for_update();
                return e.runner.get();
            }
        }
    } else {
        auto seen = std::ranges::find(seen_, key);
        if (seen == seen_.end()) {
            seen_.push_back(key);
            if (seen_.size() > kSeenMax)
                seen_.pop_front();
            return nullptr;
        }
        seen_.erase(seen);
    }
    auto runner = std::make_unique<CudaGraphRunner>();
    // The first sighting ran eager (cuBLASLt probed this M outside any capture): capture now.
    runner->set_warmup_steps(0);
    entries_.push_back(Entry{key, std::move(runner)});
    return entries_.back().runner.get();
}

bool PrefillGraphCache::warm(const Key& key) const {
    return std::ranges::any_of(entries_, [&](const Entry& e) { return e.key == key; }) ||
           key == pending_long_ || std::ranges::find(seen_, key) != seen_.end();
}

void PrefillGraphCache::invalidate_all() {
    // One eager step before the re-capture, as the single runner did: the forward may have new
    // kernels (LoRA, control vector) whose cuBLASLt shapes were never probed.
    for (auto& e : entries_) {
        e.runner->set_warmup_steps(1);
        e.runner->invalidate();
    }
}

int PrefillGraphCache::captures() const {
    int n = 0;
    for (const auto& e : entries_)
        n += e.runner->capture_count();
    return n;
}

int PrefillGraphCache::replays() const {
    int n = 0;
    for (const auto& e : entries_)
        n += e.runner->replay_count();
    return n;
}

}  // namespace imp
