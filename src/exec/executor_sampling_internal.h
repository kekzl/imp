// Shared between executor.cu (forward/masked_sample/decode-async paths) and
// executor_sampling.cu (the batched per-row sampling family) after the
// 2026-08-27 TU split. Host-inline helper + the launcher for the ban kernel
// that stays defined in executor.cu.
#pragma once

#include "core/tensor.h"
#include "exec/executor.h"
#include "compute/json_constrain.h"
#include "compute/schema_constrain.h"
#include "compute/regex_constrain.h"
#include "compute/grammar_constrain.h"

#include <cuda_runtime.h>

namespace imp {

// Defined in executor.cu (wraps ban_logits_kernel, which lives there).
void launch_ban_logits(float* logits, const int32_t* banned_ids, int n_banned, int vocab_size,
                       cudaStream_t stream);

// The one place that decides which constrainer masks a step. There are four
// sampling paths across these two TUs and they used to carry four copies of
// this chain — a new constrainer then had to be added to all four, and the
// two easy-to-miss ones are exactly how an unmasked path ships. Precedence
// mirrors ConstraintManager: grammar > regex > schema > json.
inline void apply_constraint_mask(const imp::InferenceState& st, float* logits, int vocab,
                                  cudaStream_t stream) {
    if (st.grammar_constrainer)
        st.grammar_constrainer->apply_mask(logits, vocab, stream);
    else if (st.regex_constrainer)
        st.regex_constrainer->apply_mask(logits, vocab, stream);
    else if (st.schema_constrainer)
        st.schema_constrainer->apply_mask(logits, vocab, stream);
    else if (st.json_constrainer) {
        // Budget-aware close (#1104): hand the remaining output allowance to
        // the FSM so it can force the document shut before max_tokens cuts it
        // mid-structure. Harmless when the engine leaves it at -1.
        st.json_constrainer->set_remaining_budget(st.constrain_remaining_tokens);
        st.json_constrainer->apply_mask(logits, vocab, stream);
    }
}

}  // namespace imp
