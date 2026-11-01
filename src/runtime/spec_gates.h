#pragma once

// Which speculation sources are live, as a pure function of configuration.
//
// The verify step is shared across n-gram/suffix, MTP and token recycling:
// entering it is a question about ALL drafters, running the matcher is a
// question about one. Conflating the two silently disables MTP when
// speculative.ngram=false.
//
// Kept here as a free function so the truth table is testable without a GPU
// or a live Engine. The engine holds the state; this file holds the rule.

namespace imp {

struct SpecDrafterState {
    bool ngram_on = false;      // speculative.ngram, after the per-request override
    bool mtp_on = false;        // speculative.mtp_k > 0
    bool recycling_on = false;  // speculative.token_recycling
    bool dspark_on = false;     // speculative.dspark_model loaded
    // Model-level facts that no flag can overrule (recurrent state without
    // speculative.hybrid, GGUF-MoE, no chunked prefill). False here means this
    // model never speculates, whatever is switched on.
    bool model_capable = false;
};

// Is any drafter able to feed the verify step for this request?
constexpr bool spec_any_drafter(const SpecDrafterState& s) {
    return s.model_capable && (s.ngram_on || s.mtp_on || s.recycling_on || s.dspark_on);
}

// Is the history matcher itself live? Narrower than spec_any_drafter on
// purpose: with n-gram off but MTP on, the step runs and the matcher does not.
constexpr bool spec_ngram_source(const SpecDrafterState& s) {
    return s.model_capable && s.ngram_on;
}

// Does the MTP head bind this request at prefill? Only a request that can verify: a bound one runs
// the eager MTP path every step and skips the constrained pipeline (Qwen3.8-27B-NVFP4: sampled
// 70.8 vs 93.5 tok/s, json_schema 69.0 vs 92.6).
struct MtpBindState {
    bool mtp_armed = false;         // mtp_spec_decode_enabled() and a workspace
    bool constrained = false;       // constrained, verify refuses it (sampled or constrained=off)
    bool sampling_refused = false;  // spec_sampling_refusal_ said no
};

constexpr bool mtp_binds_request(const MtpBindState& s) {
    return s.mtp_armed && !s.constrained && !s.sampling_refused;
}

// Round-robin batched speculation (speculative.batch_rr, #1003 stage 1): at
// batch > 1 one request per step runs its verify while the rest decode
// batched. Entering it is the same question as any verify step: is there a
// draft source, not just `speculative.ngram` by name - the MTP recipe
// (docs/MODELS.md) turns that key off, which silently loses batched
// speculation on an MTP-drafting server.
struct SpecBatchRrState {
    bool enabled = false;      // speculative.batch_rr
    bool recurrent = false;    // ssm_state_ != nullptr (the verify shape differs)
    int batch_rows = 0;        // rows in this step's decode batch
    bool any_drafter = false;  // spec_any_drafter() over the server-wide config
};

constexpr bool spec_batch_rr_active(const SpecBatchRrState& s) {
    return s.enabled && !s.recurrent && s.batch_rows > 1 && s.any_drafter;
}

// Hybrid partial-accept replay: device KV length to stage. Capture-mode FA2 derives
// q_offset = kv_len - rows, so the graph (rows = chunk_pad) needs p0 + chunk_pad and the
// eager replay (rows = matched + 1) needs p0 + 1 + matched; both give q_offset = p0.
constexpr int spec_replay_kv_len(bool graph_replay, int p0, int chunk_pad, int matched) {
    return graph_replay ? p0 + chunk_pad : p0 + 1 + matched;
}

// A hybrid verify chunk leaves the live recurrent slab at the last real row of the chunk
// (linear) or of candidate group 0 (grouped). A request finishing at emit row j is only at
// that row when j is it; otherwise the slab is ahead of the transcript.
constexpr bool spec_hybrid_slab_at_row(bool grouped, int winner, int j, int K, int rows_per_cand) {
    return grouped ? (winner == 0 && j == rows_per_cand - 1) : j == K;
}

// Speculation economics after a request's verify step: stop drafting? `final` = for the rest of the
// request. N-gram drafts are free, so below 15 % of drafted tokens accepted a verify cannot amortize (final).
// A block drafter pays per verify: emitted / verify below min_emit stops it until the next burst re-arm.
// Qwen3-8B-NVFP4 prose: 2.34 tok/verify at 19 % accepted, 348 vs 282 tok/s nospec; the 15 % rule stopped it.
struct SpecEconState {
    int verifies = 0;
    long long drafted = 0, accepted = 0, emitted = 0;
    bool block_drafter = false;  // speculative.dspark_model loaded
    float min_emit = 0.0f;       // speculative.dspark_min_emit, 0 = the n-gram rule
};
struct SpecEconVerdict {
    bool stop = false;
    bool final = false;
};

constexpr SpecEconVerdict spec_economics(const SpecEconState& s) {
    constexpr int kSample = 8;  // verifies before judging
    if (s.verifies < kSample)
        return {};
    if (s.block_drafter && s.min_emit > 0.0f)
        return {.stop = static_cast<float>(s.emitted) < static_cast<float>(s.verifies) * s.min_emit,
                .final = false};
    return {.stop = s.accepted * 100 < s.drafted * 15, .final = true};
}

}  // namespace imp
