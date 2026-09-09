#pragma once

#include <cstddef>

// Which NVFP4 planes the gemm_nvfp4 dequant workspace has to cover, and what
// that costs prefill graph capture.
//
// The 512 MiB cap guards exactly ONE path: gemm_nvfp4's M>1 fallback, which
// dequantizes a whole weight to FP16 before the GEMM. Under capture that
// fallback throws, so a plane larger than the cap disables prefill graph
// capture for the model.
//
// A plane that can never REACH that fallback therefore has no business vetoing
// capture for the planes that can. The LM head is exactly such a plane: at
// M > 1 executor_forward.cu runs the smallm v2 kernel, the CUTLASS NVFP4 GEMM
// or the batched K-par GEMV, and none of the three dequantizes the weight.
// Counting it turned the cap into a blanket disable on any NVFP4 checkpoint
// whose vocab x d_model exceeds 512 MiB in FP16 - 2425 MiB on
// Qwen3.8-27B-NVFP4, i.e. every prefill on that model ran eager because of a
// guard for a path it does not take.
//
// Scope: this is the WEIGHT-plane blocker only. On a Qwen3.5-family hybrid a
// second, independent blocker keeps the prefill graph off (the NVFP4 KV append
// syncs its scale to the host per chunk and cannot be captured,
// engine_init_resolver.cpp), so the TTFT effect of this exclusion is
// measurable with kv_cache.dtype=fp16 or on a dense NVFP4 model, not on
// Qwen3.8-27B-NVFP4 at its default KV dtype.
//
// Pure so the arithmetic is pinned in the CPU lane.

namespace imp {

struct DequantCapInputs {
    size_t max_eligible_bytes = 0;  // largest plane that CAN take the M>1 fallback
    size_t max_excluded_bytes = 0;  // largest plane that cannot (the LM head), for the log
    size_t cap_bytes = 0;           // the workspace cap
    bool ignore_cap = false;        // diagnostics.prefill_graph_ignore_dequant_cap
};

struct DequantCapDecision {
    bool over_cap = false;         // an ELIGIBLE plane exceeds the cap
    bool graph_capture_ok = true;  // prefill graph capture survives this model
};

constexpr DequantCapDecision dequant_cap_decide(const DequantCapInputs& in) {
    DequantCapDecision d;
    d.over_cap = in.max_eligible_bytes > in.cap_bytes;
    d.graph_capture_ok = !d.over_cap || in.ignore_cap;
    return d;
}

}  // namespace imp
