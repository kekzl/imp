#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Generation {
    bool no_logit_softcap = false;
    // lm_dequant_fp16 moved to diagnostics.lm_dequant_fp16 (AUDIT_arch_2026 A2-9).
    bool force_bos = false;
    // Disable banned-token list (debug).
    bool no_ban = false;
    // Disable RoPE inside the MTP draft head (diagnostic).
    bool mtp_no_rope = false;
};
}  // namespace imp::cfg
