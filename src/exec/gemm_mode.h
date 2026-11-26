#pragma once

namespace imp {

// Per-forward GEMM mode (audit D-7): forward_logits fills it once, every GemmContext carries a copy,
// GEMM dispatch reads it from the context instead of executor members. All false = prompt prefill.
struct GemmMode {
    bool force_fp16 = false;     // InferenceState::force_fp16_gemm: bypasses the FP8 GEMM paths
    bool spec_verify = false;    // InferenceState::spec_verify_chunk (#998)
    bool verify_smallm = false;  // InferenceState::spec_verify_smallm: captured verify on the small-M block
    bool decode_rows = false;    // rows are generated tokens, not prompt rows (#1897)
};

// Verify chunks forward with is_prefill=true (KV-write semantics) but their rows are draft tokens,
// so they count as decode rows; M alone cannot tell a decode step from a short prefill (#1897).
[[nodiscard]] constexpr bool gemm_decode_rows(int n, bool is_prefill, bool spec_verify_chunk) {
    return n > 1 && (!is_prefill || spec_verify_chunk);
}

}  // namespace imp
