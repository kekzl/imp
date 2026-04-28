#pragma once

// Diagnostics for CUDA-graph capture/replay (Milestone A1).
// Activated via imp.conf:
//   [diagnostics] graph_diag = true        — post-launch error checks + logging
//   [diagnostics] graph_dump_dir = "<path>" — dump verbose DOT per graph
// Off by default; no overhead when unset.

#include "core/logging.h"
#include "runtime/config.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace imp::graph_diag {

enum class Phase { NORMAL, CAPTURE, REPLAY };

inline thread_local Phase g_phase = Phase::NORMAL;

inline bool enabled() {
    return RuntimeConfig::current().diagnostics.graph_diag;
}

inline const char* dump_path() {
    const std::string& d = RuntimeConfig::current().diagnostics.graph_dump_dir;
    return d.empty() ? nullptr : d.c_str();
}

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::CAPTURE: return "capture";
        case Phase::REPLAY:  return "replay";
        default:             return "normal";
    }
}

inline Phase phase() { return g_phase; }

struct PhaseScope {
    Phase prev;
    explicit PhaseScope(Phase p) : prev(g_phase) { g_phase = p; }
    ~PhaseScope() { g_phase = prev; }
};

// Post-launch sync + error check. Only runs when IMP_GRAPH_DIAG is set.
// Uses cudaGetLastError (read+clear) to avoid false positives from sticky errors
// set earlier in the process (e.g. failed cudaGraphDebugDotPrint on WSL2).
inline void check_post_launch(cudaStream_t stream, const char* label) {
    if (!enabled()) return;
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        IMP_LOG_ERROR("[graph_diag:%s] cudaGetLastError after launch: %s",
                      label, cudaGetErrorString(launch_err));
    }
    cudaError_t sync = cudaStreamSynchronize(stream);
    if (sync != cudaSuccess) {
        IMP_LOG_ERROR("[graph_diag:%s] cudaStreamSynchronize after launch: %s",
                      label, cudaGetErrorString(sync));
    }
}

// Dump verbose DOT of a captured graph if IMP_GRAPH_DUMP is set.
// The label is appended to the configured path to disambiguate multiple graphs.
// On WSL2, cudaGraphDebugDotPrint may return an error that sets the sticky
// CUDA error state; we always clear it with cudaGetLastError so downstream
// checks don't see a stale error.
inline void dump_graph(cudaGraph_t g, const char* label) {
    const char* base = dump_path();
    if (!base || !g) return;
    std::string path = std::string(base) + "." + label + ".dot";
    cudaError_t err = cudaGraphDebugDotPrint(g, path.c_str(),
                                             cudaGraphDebugDotFlagsVerbose);
    (void) cudaGetLastError();  // clear sticky state unconditionally
    if (err != cudaSuccess) {
        static bool warned = false;
        if (!warned) {
            IMP_LOG_WARN("[graph_diag:%s] cudaGraphDebugDotPrint(%s) failed: %s "
                         "(known WSL2 limitation; kernel-node summary is the fallback)",
                         label, path.c_str(), cudaGetErrorString(err));
            warned = true;
        }
    } else {
        IMP_LOG_INFO("[graph_diag:%s] wrote %s", label, path.c_str());
    }
}

// Walk all kernel nodes of a graph and log (funcPtr, gridDim, blockDim, smem).
// Useful for sanity-checking per-layer kernel counts (e.g. 32 attention nodes
// on Gemma-4, split as 26 SWA + 6 global).
inline void log_kernel_nodes(cudaGraph_t g, const char* label) {
    if (!enabled() || !g) return;

    size_t num_nodes = 0;
    if (cudaGraphGetNodes(g, nullptr, &num_nodes) != cudaSuccess || num_nodes == 0) {
        IMP_LOG_INFO("[graph_diag:%s] graph has 0 nodes", label);
        return;
    }
    std::vector<cudaGraphNode_t> nodes(num_nodes);
    if (cudaGraphGetNodes(g, nodes.data(), &num_nodes) != cudaSuccess) return;

    int n_kernel = 0, n_memcpy = 0, n_memset = 0, n_host = 0,
        n_alloc = 0, n_free = 0, n_cond = 0, n_child = 0, n_other = 0;

    for (size_t i = 0; i < num_nodes; ++i) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) continue;
        switch (type) {
            case cudaGraphNodeTypeKernel:       ++n_kernel; break;
            case cudaGraphNodeTypeMemcpy:       ++n_memcpy; break;
            case cudaGraphNodeTypeMemset:       ++n_memset; break;
            case cudaGraphNodeTypeHost:         ++n_host;   break;
            case cudaGraphNodeTypeMemAlloc:     ++n_alloc;  break;
            case cudaGraphNodeTypeMemFree:      ++n_free;   break;
            case cudaGraphNodeTypeConditional:  ++n_cond;   break;
            case cudaGraphNodeTypeGraph:        ++n_child;  break;
            default:                            ++n_other;  break;
        }
    }

    IMP_LOG_INFO("[graph_diag:%s] %zu nodes (kernel=%d memcpy=%d memset=%d "
                 "host=%d alloc=%d free=%d cond=%d child=%d other=%d)",
                 label, num_nodes, n_kernel, n_memcpy, n_memset, n_host,
                 n_alloc, n_free, n_cond, n_child, n_other);

    // Per-kernel summary — at INFO level when total is modest, DEBUG otherwise.
    bool verbose = (n_kernel <= 128);
    int idx = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) continue;
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams kp{};
        if (cudaGraphKernelNodeGetParams(nodes[i], &kp) != cudaSuccess) continue;
        if (verbose) {
            IMP_LOG_INFO("[graph_diag:%s]   kernel[%d] func=%p grid=(%u,%u,%u) "
                         "block=(%u,%u,%u) smem=%u",
                         label, idx, kp.func,
                         kp.gridDim.x, kp.gridDim.y, kp.gridDim.z,
                         kp.blockDim.x, kp.blockDim.y, kp.blockDim.z,
                         kp.sharedMemBytes);
        } else {
            IMP_LOG_DEBUG("[graph_diag:%s]   kernel[%d] func=%p grid=(%u,%u,%u)",
                          label, idx, kp.func,
                          kp.gridDim.x, kp.gridDim.y, kp.gridDim.z);
        }
        ++idx;
    }
}

} // namespace imp::graph_diag
