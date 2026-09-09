// Speculation as a per-request contract.
//
// Three rules live here, all pure, all previously unreachable from any CI lane:
//   1. mtp_resolve_request  - what depth a request gets and why it did not get
//                             what it asked for. The server and the engine both
//                             call it, so they cannot disagree about a request.
//   2. spec_batch_rr_active - round-robin batched speculation needs a DRAFT
//                             SOURCE, not `speculative.ngram` by name.
//   3. dequant_cap_decide   - the 512 MiB NVFP4 dequant cap counts only planes
//                             that can take the M>1 fallback it guards.
//
// CI has no GPU; every one of these decides something on a GPU path.

#include "exec/dequant_cap.h"
#include "runtime/spec_gates.h"
#include "runtime/spec_request.h"

#include <gtest/gtest.h>

using imp::DequantCapInputs;
using imp::dequant_cap_decide;
using imp::MtpRequestState;
using imp::mtp_resolve_request;
using imp::spec_batch_rr_active;
using imp::SpecBatchRrState;
using imp::SpecDecline;
using imp::spec_decline_is_reportable;
using imp::spec_decline_name;

namespace {

// A server serving the documented Qwen3.8 recipe: head loaded, depth 2 armed.
MtpRequestState armed(int requested_k) {
    MtpRequestState s;
    s.requested_k = requested_k;
    s.armed_k = 2;
    s.head_present = true;
    s.head_loaded = true;
    return s;
}

// ---------------------------------------------------------------- resolution

TEST(SpecRequest, UnsetTakesTheServerDefaultAndDeclinesNothing) {
    const auto r = mtp_resolve_request(armed(-1));
    EXPECT_EQ(r.k, 2);
    EXPECT_EQ(r.reason, SpecDecline::kNone);
}

// A request that asked for nothing on a server without a head is the ordinary
// case, not a decline: reporting one would put a field on every response of
// every model that ships no MTP tensors.
TEST(SpecRequest, SilenceOnAHeadlessServerIsNotADecline) {
    MtpRequestState s;
    s.requested_k = -1;
    const auto r = mtp_resolve_request(s);
    EXPECT_EQ(r.k, 0);
    EXPECT_EQ(r.reason, SpecDecline::kNone);
}

TEST(SpecRequest, RequestCanLowerTheDepth) {
    const auto r = mtp_resolve_request(armed(1));
    EXPECT_EQ(r.k, 1);
    EXPECT_EQ(r.reason, SpecDecline::kNone);
}

TEST(SpecRequest, ZeroTurnsMtpOffForThisRequestOnly) {
    const auto r = mtp_resolve_request(armed(0));
    EXPECT_EQ(r.k, 0);
    EXPECT_EQ(r.reason, SpecDecline::kRequestOff);
    EXPECT_FALSE(spec_decline_is_reportable(r.reason));
}

TEST(SpecRequest, SpeculativeFalseBeatsADepth) {
    MtpRequestState s = armed(2);
    s.forced_off = true;
    EXPECT_EQ(mtp_resolve_request(s).k, 0);
    // ... and it also suppresses the server default.
    MtpRequestState unset = armed(-1);
    unset.forced_off = true;
    EXPECT_EQ(mtp_resolve_request(unset).k, 0);
}

TEST(SpecRequest, DeeperThanArmedClampsToTheArmedChain) {
    const auto r = mtp_resolve_request(armed(4));
    EXPECT_EQ(r.k, 2) << "a request must not exceed the chain buffer the process allocated";
    EXPECT_EQ(r.reason, SpecDecline::kDepthClamped);
    EXPECT_TRUE(spec_decline_is_reportable(r.reason));
}

// The decline an operator actually hits: concurrent serving declines the head
// at load time, so `{"speculative": {"mtp_k": 2}}` cannot be honoured. Before
// this, the only trace was one INFO line in the startup log.
TEST(SpecRequest, ConcurrentServerDeclineIsNamedAndReportable) {
    MtpRequestState s;
    s.requested_k = 2;
    s.head_present = true;   // the checkpoint ships one
    s.head_loaded = false;   // this process did not upload it
    const auto r = mtp_resolve_request(s);
    EXPECT_EQ(r.k, 0);
    EXPECT_EQ(r.reason, SpecDecline::kHeadNotLoaded);
    EXPECT_TRUE(spec_decline_is_reportable(r.reason));
    EXPECT_STREQ(spec_decline_name(r.reason), "mtp_head_not_loaded");
}

TEST(SpecRequest, NoHeadAtAllIsADifferentReasonThanNotLoaded) {
    MtpRequestState s;
    s.requested_k = 2;
    const auto r = mtp_resolve_request(s);
    EXPECT_EQ(r.reason, SpecDecline::kNoHead);
    EXPECT_STREQ(spec_decline_name(r.reason), "no_mtp_head");
}

TEST(SpecRequest, LoadedButUnarmedHeadIsItsOwnReason) {
    MtpRequestState s;
    s.requested_k = 2;
    s.head_present = true;
    s.head_loaded = true;
    s.armed_k = 0;  // speculative.mtp_k=0
    EXPECT_EQ(mtp_resolve_request(s).reason, SpecDecline::kHeadNotArmed);
}

// The resolution never hands back a depth the process cannot serve, whatever
// the caller asked for.
TEST(SpecRequest, ResolvedDepthNeverExceedsTheArmedChain) {
    for (int armed_k = 0; armed_k <= 4; ++armed_k) {
        for (int req = -1; req <= 8; ++req) {
            MtpRequestState s;
            s.requested_k = req;
            s.armed_k = armed_k;
            s.head_present = true;
            s.head_loaded = true;
            EXPECT_LE(mtp_resolve_request(s).k, armed_k) << "armed=" << armed_k << " req=" << req;
            EXPECT_GE(mtp_resolve_request(s).k, 0) << "armed=" << armed_k << " req=" << req;
        }
    }
}

// ------------------------------------------------------------------ batch_rr

// The defect: batch_rr demanded `speculative.ngram`, the one key the measured
// MTP recipe sets to false.
TEST(SpecBatchRr, MtpOnlyServerStillGetsRoundRobinVerify) {
    SpecBatchRrState s;
    s.enabled = true;
    s.batch_rows = 4;
    s.any_drafter = true;  // ngram off, MTP on
    EXPECT_TRUE(spec_batch_rr_active(s));
}

TEST(SpecBatchRr, NoDrafterNoRoundRobin) {
    SpecBatchRrState s;
    s.enabled = true;
    s.batch_rows = 4;
    s.any_drafter = false;
    EXPECT_FALSE(spec_batch_rr_active(s));
}

TEST(SpecBatchRr, NeedsABatchAndANonRecurrentModel) {
    SpecBatchRrState s;
    s.enabled = true;
    s.any_drafter = true;
    s.batch_rows = 1;
    EXPECT_FALSE(spec_batch_rr_active(s)) << "batch 1 takes the plain verify dispatch";
    s.batch_rows = 4;
    s.recurrent = true;
    EXPECT_FALSE(spec_batch_rr_active(s));
    s.recurrent = false;
    EXPECT_TRUE(spec_batch_rr_active(s));
    s.enabled = false;
    EXPECT_FALSE(spec_batch_rr_active(s));
}

// --------------------------------------------------------------- dequant cap

constexpr size_t kMiB = 1024ULL * 1024;

// Qwen3.8-27B-NVFP4: the LM head dequants to 2425 MiB, every layer plane fits
// in 512 MiB. Counting the LM head disabled prefill graph capture for the
// whole model, for a fallback the LM head never takes.
TEST(DequantCap, LmHeadDoesNotDisableGraphCaptureForTheLayers) {
    DequantCapInputs in;
    in.max_eligible_bytes = 96 * kMiB;
    in.max_excluded_bytes = 2425 * kMiB;
    in.cap_bytes = 512 * kMiB;
    const auto d = dequant_cap_decide(in);
    EXPECT_FALSE(d.over_cap);
    EXPECT_TRUE(d.graph_capture_ok);
}

TEST(DequantCap, AnEligiblePlaneOverTheCapStillDisablesCapture) {
    DequantCapInputs in;
    in.max_eligible_bytes = 513 * kMiB;
    in.cap_bytes = 512 * kMiB;
    const auto d = dequant_cap_decide(in);
    EXPECT_TRUE(d.over_cap);
    EXPECT_FALSE(d.graph_capture_ok);
}

TEST(DequantCap, ExactlyAtTheCapIsCovered) {
    DequantCapInputs in;
    in.max_eligible_bytes = 512 * kMiB;
    in.cap_bytes = 512 * kMiB;
    EXPECT_FALSE(dequant_cap_decide(in).over_cap);
}

TEST(DequantCap, TheDiagnosticOverrideKeepsCaptureOnButNotTheVerdict) {
    DequantCapInputs in;
    in.max_eligible_bytes = 900 * kMiB;
    in.cap_bytes = 512 * kMiB;
    in.ignore_cap = true;
    const auto d = dequant_cap_decide(in);
    EXPECT_TRUE(d.over_cap) << "the override must not rewrite the measurement";
    EXPECT_TRUE(d.graph_capture_ok);
}

}  // namespace
