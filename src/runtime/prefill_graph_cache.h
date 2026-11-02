#pragma once

// Serial offset-0 prefill graphs: a key runs eager on its first sighting, captures on its second,
// replays after. Prompts up to short_max tokens (runtime.prefill_graph_prewarm) keep a runner each;
// longer ones share one runner, re-captured per key through cudaGraphExecUpdate (an update costs a
// fraction of an instantiate: 26.5 ms instantiate at pp512 on Qwen3-30B-A3B).

#include "runtime/cuda_graph.h"

#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <tuple>

namespace imp {

class PrefillGraphCache {
public:
    // (chunk_len, block count, FP8 KV calibration generation, recurrent slot or -1)
    using Key = std::tuple<int, int, uint64_t, int>;

    void set_short_max(int tokens);
    [[nodiscard]] int short_max() const { return short_max_; }
    // The runner to execute this key through, or nullptr: run eager (first sighting).
    [[nodiscard]] CudaGraphRunner* select(const Key& key);
    // True when the key replays or captures on its next select (no eager first sighting left).
    [[nodiscard]] bool warm(const Key& key) const;
    // Drop every captured graph (pointer moves, LoRA, control vectors).
    void invalidate_all();
    [[nodiscard]] size_t size() const { return entries_.size(); }
    [[nodiscard]] int captures() const;
    [[nodiscard]] int replays() const;

private:
    struct Entry {
        Key key;
        std::unique_ptr<CudaGraphRunner> runner;
    };
    [[nodiscard]] bool is_short_(const Key& k) const { return std::get<0>(k) <= short_max_; }
    std::list<Entry> entries_;         // at most one entry for a long key
    std::deque<Key> seen_;             // first sightings of short keys, bounded
    Key pending_long_{-1, -1, 0, -1};  // the last long key that ran eager
    int short_max_ = 0;
    static constexpr size_t kSeenMax = 256;
};

}  // namespace imp
