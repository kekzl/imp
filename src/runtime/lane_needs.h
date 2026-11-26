#pragma once

// Decode-lane eligibility as one table (audit C-8): 11 need bits, 5 lanes each refuse a mask.
// Admission bits are set once by Engine::add_request; live bits are re-read per check, since
// logit_bias grows mid-run (think-opener ban), cfg drops with its shadow, forced decode runs out.

#include <cstdint>

namespace imp {

using LaneNeeds = uint32_t;

namespace lane_need {
// ── admission ──────────────────────────────────────────────────────
inline constexpr LaneNeeds kLogprobs = 1u << 0;        // host copy of the logits each step
inline constexpr LaneNeeds kReadout = 1u << 1;         // logit lens / SAE readout after each eager step
inline constexpr LaneNeeds kGrammar = 1u << 2;         // json_mode, json_schema, regex, grammar FSM
inline constexpr LaneNeeds kToolConstraint = 1u << 3;  // enforced tool-call envelope FSM
inline constexpr LaneNeeds kMirostat = 1u << 4;
inline constexpr LaneNeeds kDry = 1u << 5;             // dry_multiplier > 0 (the sampler's own test)
inline constexpr LaneNeeds kMinP = 1u << 6;            // min_p != 0
inline constexpr LaneNeeds kTypicalP = 1u << 7;        // typical_p < 1
inline constexpr LaneNeeds kBoundedPenalty = 1u << 8;  // penalties over a sliding repeat_last_n window
// ── live ───────────────────────────────────────────────────────────
inline constexpr LaneNeeds kHostShaping = 1u << 9;  // logit_bias, phrase ban, cfg
inline constexpr LaneNeeds kForcing = 1u << 10;     // teacher-forced decode tokens left

inline constexpr LaneNeeds kConstrained = kGrammar | kToolConstraint;
inline constexpr LaneNeeds kHostLogits = kLogprobs | kReadout;
}  // namespace lane_need

// What each lane refuses. Dynamic conditions (status, MTP binding, think markers, top-k vs vocab,
// a live constraint manager) stay at the lane: they are not request fields.
namespace lane_refuses {
using namespace lane_need;
// Single-request async graph loop (engine_scheduler.cpp).
inline constexpr LaneNeeds kGraphLoop = kHostLogits | kConstrained | kHostShaping | kMirostat | kDry | kMinP |
                                        kTypicalP;
// Constrained pipeline (engine_scheduler.cpp); also requires kConstrained.
inline constexpr LaneNeeds kConstrainedPipeline = kHostLogits | kHostShaping | kMirostat | kDry | kMinP |
                                                  kTypicalP;
// Pipelined batched decode row (engine_decode_pipeline.cpp); min_p/typical_p run device-side there.
inline constexpr LaneNeeds kPipelineRow = kHostLogits | kGrammar | kHostShaping | kMirostat | kDry | kForcing;
// Batched MTP verify row (engine_spec_batch_verify.cpp).
inline constexpr LaneNeeds kBatchVerifyRow = kHostLogits | kConstrained | kHostShaping | kMirostat | kDry |
                                             kBoundedPenalty;
// n-gram speculation (engine_spec_ngram.cpp), one mask per refusal reason it logs.
inline constexpr LaneNeeds kNgramWindow = kBoundedPenalty;
inline constexpr LaneNeeds kNgramShaping = kHostShaping | kMirostat | kDry;
inline constexpr LaneNeeds kNgramHostLogits = kHostLogits;
}  // namespace lane_refuses

}  // namespace imp
