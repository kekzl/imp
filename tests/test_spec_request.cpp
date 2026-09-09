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

#include "compute/attention_paged.h"
#include "exec/dequant_cap.h"
#include "runtime/spec_gates.h"
#include "runtime/spec_request.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

using imp::DequantCapInputs;
using imp::dequant_cap_decide;
using imp::MtpRequestState;
using imp::mtp_resolve_request;
using imp::spec_batch_rr_active;
using imp::SpecBatchRrState;
using imp::SpecDecline;
using imp::spec_decline_is_reportable;
using imp::spec_decline_name;
using imp::paged_attention_serves_head_dim;
using imp::paged_fp8_decode_has_fast_kernel;

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

// ------------------------------------------------------- FP8 decode kernels

// The pin `kv_cache.dtype=fp8` passes init on a head_dim-256 model
// (paged_attention_serves_head_dim says FP8 covers 64/96/128/256/512) and then
// lands on the scalar template, because the four-token and GQA-lane kernels are
// head_dim-128 instances: 4 FP8 bytes per lane is what makes one uint32 load
// per lane work (attention_paged_fp8_multitok.cu's static_assert, and
// paged_attention_fp8_multitok_heads_per_cta's `if (head_dim != 128) return 0`).
// Serving and serving fast are two questions and only the first had a predicate.
TEST(PagedFp8Decode, FastKernelsAreHeadDim128Only) {
    EXPECT_TRUE(paged_fp8_decode_has_fast_kernel(128));
    for (int hd : {64, 96, 192, 256, 512})
        EXPECT_FALSE(paged_fp8_decode_has_fast_kernel(hd)) << "hd=" << hd;
}

// The two questions must not be confused: FP8 SERVES head_dim 256 (the scalar
// kernel writes a correct answer), it just does not serve it fast. A predicate
// that answered "no" to both would have init fall back to FP16 KV instead of
// logging, which is a different and wrong behaviour.
TEST(PagedFp8Decode, ServingAndServingFastAreDifferentQuestions) {
    EXPECT_TRUE(paged_attention_serves_head_dim(imp::QType::FP8_E4M3, 256));
    EXPECT_FALSE(paged_fp8_decode_has_fast_kernel(256));
}

// ------------------------------------------------------------- the wiring

// Two call sites decide whether batch>1 speculation happens at all, and both
// used to ask `speculative.ngram` - the one key the measured MTP recipe sets to
// false. Fixing the rule does not fix the wiring: a mutant that points either
// site back at the n-gram predicate survives the entire CPU lane, because
// answering the question at runtime needs a GPU, a model and a batch.
//
// So this guard reads the source. It is the same trade the guard_* ctest
// entries make (a literal filter copy that no test execution can compare), and
// it is the only lane CI has.
std::string read_source(const char* rel) {
    std::ifstream in(std::string(IMP_TEST_SOURCE_ROOT) + "/" + rel);
    EXPECT_TRUE(in.good()) << "cannot read " << rel;
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The block between `first` and the next line containing `end_marker`.
std::string block_after(const std::string& src, const std::string& first, const std::string& end_marker) {
    const size_t a = src.find(first);
    EXPECT_NE(a, std::string::npos) << "anchor not found: " << first;
    if (a == std::string::npos)
        return {};
    const size_t b = src.find(end_marker, a);
    EXPECT_NE(b, std::string::npos) << "end marker not found: " << end_marker;
    return src.substr(a, b == std::string::npos ? std::string::npos : b - a);
}

TEST(SpecBatchRrWiring, SchedulerRoundRobinAsksForADrafterNotForNgram) {
    const std::string src = read_source("src/runtime/engine_scheduler.cpp");
    const std::string blk = block_after(src, "SpecBatchRrState rr_state", "spec_rr_yield_interval_ =");
    EXPECT_NE(blk.find("spec_any_drafter_enabled_"), std::string::npos)
        << "the round-robin branch must select rows by 'can anyone draft', not by the n-gram flag";
    EXPECT_EQ(blk.find("spec_ngram_enabled_"), std::string::npos)
        << "spec_ngram_enabled_ is back in the round-robin branch: with the documented MTP pair "
           "(mtp_k=2, ngram=false) it selects no row and batch>1 speculation never fires";
    EXPECT_EQ(blk.find("speculative.ngram"), std::string::npos)
        << "the batch_rr entry gate must not read speculative.ngram";
}

TEST(SpecBatchRrWiring, PipelineYieldAsksForADrafterNotForNgram) {
    const std::string src = read_source("src/runtime/engine_decode_pipeline.cpp");
    const std::string blk =
        block_after(src, "speculative.batch_rr", "const int next_parity");
    EXPECT_NE(blk.find("spec_any_drafter_enabled_"), std::string::npos)
        << "the #1003 yield is what lets the round-robin verify get a turn; asking the n-gram "
           "question here starves it on a dense model drafting with MTP alone";
    EXPECT_EQ(blk.find("spec_ngram_enabled_"), std::string::npos)
        << "spec_ngram_enabled_ is back in the pipeline spec yield";
    EXPECT_EQ(blk.find("speculative.ngram"), std::string::npos)
        << "the pipeline spec yield must not read speculative.ngram";
}

}  // namespace
