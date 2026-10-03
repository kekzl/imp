#pragma once

// Piecewise prefill graphs (#2435): the FFN/MoE phase of one layer replays from a CUDA graph keyed
// by (layer, padded rows); attention stays eager. Removes the FFN half of an eager prefill's
// launches (Qwen3-30B-A3B NVFP4: 2/3 of the GPU idle between kernels sits in that half).

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <map>
#include <utility>

namespace imp {

class FfnGraphCache {
public:
    static constexpr int kMaxRowBuckets = 8;
    static constexpr int kMaxRows = 4096;  // above: launch overhead < 1 % of the forward, stay eager

    // Rows an n-row prefill FFN runs at: 16-row steps to 64, 64 to 1024, 256 to kMaxRows; 0 = eager.
    [[nodiscard]] static int bucket_rows(int n);

    FfnGraphCache() = default;
    FfnGraphCache(const FfnGraphCache&) = delete;
    FfnGraphCache& operator=(const FfnGraphCache&) = delete;
    ~FfnGraphCache() { clear(); }

    // fn runs eagerly on the first sighting of (layer, rows), is captured on the second and replayed
    // after. A generation change drops every graph; a capture or launch failure, or fn throwing
    // under capture, runs fn eagerly and disables the cache for the process.
    void run(int layer, int rows, uint64_t generation, cudaStream_t stream, const std::function<void()>& fn);
    void clear();

    [[nodiscard]] bool disabled() const { return disabled_; }
    [[nodiscard]] uint64_t replays() const { return replays_; }
    [[nodiscard]] size_t graphs() const;

private:
    struct Entry {
        cudaGraphExec_t exec = nullptr;
        int sightings = 0;
        uint64_t last_use = 0;
    };
    void disable_(const char* what, cudaError_t err);
    void evict_oldest_bucket_();
    std::map<std::pair<int, int>, Entry> entries_;  // (rows, layer)
    uint64_t generation_ = 0;
    uint64_t clock_ = 0;
    uint64_t replays_ = 0;
    bool disabled_ = false;
};

}  // namespace imp
