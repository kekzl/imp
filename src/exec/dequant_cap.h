#pragma once

#include <cstddef>

// The 512 MiB dequant-workspace cap guards only gemm_nvfp4's M>1 fallback
// (dequantizes a whole weight to FP16, illegal under capture). A plane
// that never reaches that fallback must not veto capture for planes that
// can: the LM head at M>1 always takes smallm v2/CUTLASS/K-par GEMV, none
// of which dequantize, so it is excluded from the cap. Weight-plane blocker
// only: a hybrid's NVFP4 KV-append host sync is a separate, independent capture blocker. Pure function.

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
