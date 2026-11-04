#pragma once
// CUDA graphs of the batched MTP head pass (Engine::mtp_head_pass_, speculative.mtp_head_graph): one per
// (fed rows, requests, chain). First use eager (cuBLASLt picks and caches the shapes' algos), second use
// captured, then replayed; the scan nodes' smem follows the pass's largest position
// (mtp_scan_nodes_set_rows).

#include "compute/mtp_forward.h"  // MtpScanNodes
#include "core/cuda_raii.h"
#include "memory/host_pinned.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace imp {

struct MtpHeadGraphKey {
    int n = 0;  // fed rows
    int m = 0;  // bound requests (draft rows)
    bool chain = false;
    auto operator<=>(const MtpHeadGraphKey&) const = default;
};

// Bound: mtp_batch_verify <= 4 requests x 1..3 fed rows each = 24 (n, m) pairs, x chain on/off = 48.
inline constexpr size_t kMtpHeadGraphMax = 48;
// Consecutive capture / instantiate / launch failures before the cache turns itself off.
inline constexpr int kMtpHeadGraphMaxFailures = 2;

enum class MtpHeadGraphStep { Eager, Capture, Replay };

// What a pass does: replay a cached graph; capture on the second use of a key (the first warmed cuBLASLt);
// eager when the cache is off, full, or the key is new.
constexpr MtpHeadGraphStep mtp_head_graph_step(bool cached, int eager_uses, size_t n_graphs, bool doomed) {
    if (doomed)
        return MtpHeadGraphStep::Eager;
    if (cached)
        return MtpHeadGraphStep::Replay;
    return eager_uses >= 1 && n_graphs < kMtpHeadGraphMax ? MtpHeadGraphStep::Capture
                                                          : MtpHeadGraphStep::Eager;
}

struct MtpHeadGraph {
    CudaGraph graph;  // kept: scan.params point into it
    CudaGraphExec exec;
    MtpScanNodes scan;
    int eager_uses = 0;
    int n_out = 0;  // draft tokens the pass writes (m, or 2m with the chain)
    // Captured device pointers that are not fixed by the workspace: a change drops the graph.
    const void* hidden = nullptr;
    const void* head = nullptr;
};

struct MtpHeadGraphs {
    std::map<MtpHeadGraphKey, MtpHeadGraph> by_key;
    size_t n_graphs = 0;
    int failures = 0;
    bool doomed = false;
    long long replays = 0;
    PinnedBuffer h_out;                   // the drafts' D2H destination (kMtpMaxTopW * kMtpMaxChainK ints)
    std::vector<int32_t> h_out_pageable;  // the destination when the pinned allocation failed
};

}  // namespace imp
