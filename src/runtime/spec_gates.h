#pragma once

// Which speculation sources are live, as a pure function of configuration.
//
// The verify step is shared: the n-gram/suffix matcher, the trained MTP head
// and token recycling all fill the same draft chunk, and which one filled it
// is decided inside step_spec_verify_. Entering that step is therefore a
// question about ALL of them, while running the matcher is a question about
// one. Conflating the two is how `speculative.ngram=false` came to disable
// MTP outright: mtp_k=2 drafted nothing, because the step that consumes its
// chain was never reached, and nothing in the logs said so.
//
// Kept here as a free function so the truth table is testable without a GPU
// or a live Engine. The engine holds the state; this file holds the rule.

namespace imp {

struct SpecDrafterState {
    bool ngram_on = false;      // speculative.ngram, after the per-request override
    bool mtp_on = false;        // speculative.mtp_k > 0
    bool recycling_on = false;  // speculative.token_recycling
    // Model-level facts that no flag can overrule (recurrent state without
    // speculative.hybrid, GGUF-MoE, no chunked prefill). False here means this
    // model never speculates, whatever is switched on.
    bool model_capable = false;
};

// Is any drafter able to feed the verify step for this request?
constexpr bool spec_any_drafter(const SpecDrafterState& s) {
    return s.model_capable && (s.ngram_on || s.mtp_on || s.recycling_on);
}

// Is the history matcher itself live? Narrower than spec_any_drafter on
// purpose: with n-gram off but MTP on, the step runs and the matcher does not.
constexpr bool spec_ngram_source(const SpecDrafterState& s) {
    return s.model_capable && s.ngram_on;
}

// Round-robin batched speculation (speculative.batch_rr, #1003 stage 1): at
// batch > 1 one request per step runs its verify while the rest decode
// batched. Entering it is the same question as entering any verify step - is
// there A DRAFT SOURCE - and it used to demand `speculative.ngram` by name.
// That is exactly the key the MTP recipe turns off (docs/MODELS.md: `--set
// speculative.mtp_k=2 --set speculative.ngram=false` as a pair), so an
// MTP-drafting server lost batched speculation to the flag that was supposed
// to arm the head. On the hybrid the `recurrent` term hides it; on every dense
// model it was live.
struct SpecBatchRrState {
    bool enabled = false;      // speculative.batch_rr
    bool recurrent = false;    // ssm_state_ != nullptr (the verify shape differs)
    int batch_rows = 0;        // rows in this step's decode batch
    bool any_drafter = false;  // spec_any_drafter() over the server-wide config
};

constexpr bool spec_batch_rr_active(const SpecBatchRrState& s) {
    return s.enabled && !s.recurrent && s.batch_rows > 1 && s.any_drafter;
}

}  // namespace imp
