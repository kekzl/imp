#pragma once

// cuBLASLt algo choice for a cold shape whose first call lands inside a stream capture (#2396).
// Pure host logic, no CUDA types: covered by the CPU lane (tests/test_gemm_algo_capture.cpp).

#include <cstddef>
#include <span>

namespace imp {

// Probe launches (warmup + event timing) inside a capture invalidate it: allowed only when the
// capture query succeeded and reported no capture.
constexpr bool gemm_algo_probe_allowed(bool query_ok, bool capturing) { return query_ok && !capturing; }

// One heuristic candidate as cublasLtMatmulAlgoCheck sees it on the host.
struct GemmAlgoHostCheck {
    bool supported = false;
    size_t workspace = 0;
};

// First candidate in heuristic order that the host check accepts within `max_workspace`; -1 if none.
constexpr int gemm_capture_safe_pick(std::span<const GemmAlgoHostCheck> cands, size_t max_workspace) {
    for (size_t i = 0; i < cands.size(); i++) {
        if (cands[i].supported && cands[i].workspace <= max_workspace)
            return static_cast<int>(i);
    }
    return -1;
}

}  // namespace imp
