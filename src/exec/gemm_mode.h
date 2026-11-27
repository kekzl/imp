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

// Own FP8 x FP8 small-M GEMM rows: decode / verify rows, or any row when gemm.fp8_smallm_m1 set the
// FP8 row floor to 1; prompt rows stay on the M-invariant cuBLASLt pin (#2152).
[[nodiscard]] constexpr bool fp8_smallm_mode_rows(const GemmMode& m, int fp8_min_rows) {
    return m.decode_rows || fp8_min_rows == 1;
}

// FP32-residual sandwich fusion: the M=1 decode row or decode / verify rows, never prompt rows.
[[nodiscard]] constexpr bool fp32_resid_mode_rows(const GemmMode& m, int n) {
    return n >= 1 && (n == 1 || m.decode_rows);
}

// M=1 launch fusions (GDN input, GDN out + residual, FP8 fold, attention q|k|v): no verify chunk.
[[nodiscard]] constexpr bool m1_fusion_mode_ok(const GemmMode& m) { return !m.spec_verify; }

}  // namespace imp
