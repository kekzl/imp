// Audit C-8: decode-lane eligibility was a request-field matrix kept by hand at 5 gate sites, and
// logit lens / SAE set logprobs = true to stay eager. The lanes now test one bitset
// (runtime/lane_needs.h). Every combination of the 17 fields the old predicates read lands on the
// same lanes as the old predicates (copied below as the oracle, lens/SAE still implying logprobs).

#include "runtime/lane_needs.h"
#include "runtime/request.h"

#include <gtest/gtest.h>

#include <string>

using imp::LaneNeeds;
using imp::Request;
namespace need = imp::lane_need;
namespace refuses = imp::lane_refuses;

namespace {

constexpr int kFields = 17;

void set_fields(Request& r, unsigned m) {
    const auto on = [m](int i) { return ((m >> i) & 1u) != 0; };
    r.logprobs = on(0);
    r.logit_lens = on(1) ? 5 : 0;
    r.sae_top = on(2) ? 8 : 0;
    r.sae_clamp.clear();
    if (on(3))
        r.sae_clamp.emplace_back(7, 2.0f);
    r.json_mode = on(4);
    r.json_schema = on(5) ? std::string("{}") : std::string();
    r.regex_pattern = on(6) ? std::string("a+") : std::string();
    r.grammar = on(7) ? std::string("root ::= \"a\"") : std::string();
    r.tool_constraint_tools.clear();
    if (on(8))
        r.tool_constraint_tools.emplace_back("f", "{}");
    r.mirostat = on(9) ? 2 : 0;
    r.dry_multiplier = on(10) ? 0.8f : 0.0f;
    r.min_p = on(11) ? 0.05f : 0.0f;
    r.typical_p = on(12) ? 0.9f : 1.0f;
    r.repetition_penalty = on(13) ? 1.05f : 1.0f;
    r.repeat_last_n = on(14) ? 64 : 0;
    r.logit_bias.clear();
    if (on(15))
        r.logit_bias.emplace_back(3, -5.0f);
    r.forced_decode.clear();
    r.forced_pos = 0;
    if (on(16))
        r.forced_decode.push_back(11);
}

// The 5 lanes before C-8, as they read the fields (lens and SAE had set logprobs = true).
struct OldLanes {
    bool graph_loop, constrained_pipeline, pipeline_row, batch_verify_row, batch_constrained, prefill_logits,
        ragged;
    const char* ngram;
};

OldLanes old_lanes(const Request& r) {
    const bool lp = r.logprobs || r.logit_lens > 0 || r.sae_top > 0 || !r.sae_clamp.empty();
    const bool host = r.host_logit_shaping();
    const bool grammar = r.json_mode || !r.json_schema.empty() || !r.regex_pattern.empty() ||
                         !r.grammar.empty();
    const bool constrained = grammar || !r.tool_constraint_tools.empty();
    const bool sampler_ok = !host && r.mirostat == 0 && r.dry_multiplier == 0.0f && r.min_p == 0.0f &&
                            r.typical_p >= 1.0f;
    const bool penalties = r.repetition_penalty != 1.0f || r.frequency_penalty != 0.0f ||
                           r.presence_penalty != 0.0f;
    const bool window = penalties && r.repeat_last_n != 0;
    const bool shaping = r.dry_multiplier != 0.0f || r.mirostat != 0 || host;
    OldLanes o{};
    o.graph_loop = !lp && !constrained && sampler_ok;
    o.constrained_pipeline = !lp && constrained && sampler_ok;
    o.pipeline_row = !(lp || grammar) && !(r.mirostat != 0 || host || r.forcing()) &&
                     !(r.dry_multiplier > 0.0f);
    o.batch_verify_row = !window && !shaping && !(lp || constrained);
    o.batch_constrained = constrained;
    o.prefill_logits = lp || host;
    o.ragged = !lp && !constrained;
    o.ngram = window    ? "bounded_repeat_window"
              : shaping ? "logit_shaping"
              : lp      ? "constrained_decode"
                        : nullptr;
    return o;
}

OldLanes new_lanes(const Request& r) {
    const LaneNeeds n = r.lane_needs_now();
    OldLanes o{};
    o.graph_loop = (n & refuses::kGraphLoop) == 0;
    o.constrained_pipeline = (n & need::kConstrained) != 0 && (n & refuses::kConstrainedPipeline) == 0;
    o.pipeline_row = (n & refuses::kPipelineRow) == 0;
    o.batch_verify_row = (n & refuses::kBatchVerifyRow) == 0;
    o.batch_constrained = (n & need::kConstrained) != 0;
    o.prefill_logits = r.needs_prefill_logits();
    o.ragged = r.ragged_prefill_allowed();
    o.ngram = (n & refuses::kNgramWindow) != 0       ? "bounded_repeat_window"
              : (n & refuses::kNgramShaping) != 0    ? "logit_shaping"
              : (n & refuses::kNgramHostLogits) != 0 ? "constrained_decode"
                                                     : nullptr;
    return o;
}

}  // namespace

TEST(LaneNeeds, EveryFieldCombinationLandsOnTheOldLanes) {
    Request r;
    int mismatches = 0;
    for (unsigned m = 0; m < (1u << kFields) && mismatches < 5; ++m) {
        set_fields(r, m);
        r.lane_needs = r.admission_lane_needs();
        const OldLanes want = old_lanes(r);
        const OldLanes got = new_lanes(r);
        const std::string want_ng = want.ngram ? want.ngram : "-";
        const std::string got_ng = got.ngram ? got.ngram : "-";
        const bool same = want.graph_loop == got.graph_loop &&
                          want.constrained_pipeline == got.constrained_pipeline &&
                          want.pipeline_row == got.pipeline_row &&
                          want.batch_verify_row == got.batch_verify_row &&
                          want.batch_constrained == got.batch_constrained &&
                          want.prefill_logits == got.prefill_logits && want.ragged == got.ragged &&
                          want_ng == got_ng;
        if (!same) {
            ++mismatches;
            ADD_FAILURE() << "field mask 0x" << std::hex << m << std::dec << " lane_needs 0x" << std::hex
                          << r.lane_needs_now() << std::dec << ": graph_loop " << want.graph_loop << "/"
                          << got.graph_loop << " cpipe " << want.constrained_pipeline << "/"
                          << got.constrained_pipeline << " pipeline " << want.pipeline_row << "/"
                          << got.pipeline_row << " batch_verify " << want.batch_verify_row << "/"
                          << got.batch_verify_row << " constrained " << want.batch_constrained << "/"
                          << got.batch_constrained << " prefill_logits " << want.prefill_logits << "/"
                          << got.prefill_logits << " ragged " << want.ragged << "/" << got.ragged << " ngram "
                          << want_ng << "/" << got_ng;
        }
    }
    EXPECT_EQ(mismatches, 0);
}

// Lens and SAE keep their own bit: logprobs stays what the client asked for.
TEST(LaneNeeds, ReadoutDoesNotTurnOnLogprobs) {
    Request r;
    r.logit_lens = 5;
    r.lane_needs = r.admission_lane_needs();
    EXPECT_FALSE(r.logprobs);
    EXPECT_EQ(r.lane_needs, need::kReadout);
    EXPECT_NE(r.lane_needs & refuses::kGraphLoop, 0u);
}

// The admission bits are frozen; the live ones follow the fields mid-run (the think-opener ban
// appends to logit_bias after admission, forced decode runs out).
TEST(LaneNeeds, LiveBitsFollowMidRunChanges) {
    Request r;
    r.lane_needs = r.admission_lane_needs();
    EXPECT_EQ(r.lane_needs_now(), 0u);
    r.logit_bias.emplace_back(9, -1e30f);
    EXPECT_EQ(r.lane_needs_now(), need::kHostShaping);
    r.logit_bias.clear();
    r.forced_decode = {4};
    EXPECT_EQ(r.lane_needs_now(), need::kForcing);
    r.forced_pos = 1;
    EXPECT_EQ(r.lane_needs_now(), 0u);
}

// DRY is active only above 0 (executor_sampling.cu); a negative multiplier is a no-op and sets no bit.
TEST(LaneNeeds, NegativeDryIsNoNeed) {
    Request r;
    r.dry_multiplier = -1.0f;
    EXPECT_EQ(r.admission_lane_needs(), 0u);
}
