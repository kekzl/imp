// AUDIT_arch_2026 C-6: the per-request speculation counters reach `usage`
// under vendor-prefixed keys, and only when a verify step ran - plus the two
// halves that were missing: the DECLINE (a request that asked for MTP and did
// not get it), and the `"speculative"` request field in its object form.
#include <gtest/gtest.h>
#include "handlers_internal.h"  // tools/imp-server/handlers_internal.h
#include "runtime/request.h"
#include <memory>
#include <string>

namespace {

TEST(SpecUsage, AbsentWithoutAVerifyStep) {
    auto req = std::make_shared<imp::Request>();
    nlohmann::json usage = {{"prompt_tokens", 3}, {"completion_tokens", 2}};
    add_spec_usage_(usage, req);
    EXPECT_FALSE(usage.contains("completion_tokens_details"));
    add_spec_usage_(usage, nullptr);
    EXPECT_FALSE(usage.contains("completion_tokens_details"));
}

TEST(SpecUsage, CountersNextToTheExistingDetails) {
    auto req = std::make_shared<imp::Request>();
    req->spec_verifies = 7;
    req->spec_drafted = 21;
    req->spec_accepted = 13;
    req->spec_emitted = 19;
    nlohmann::json usage = {{"prompt_tokens", 3},
                            {"completion_tokens", 2},
                            {"completion_tokens_details", {{"reasoning_tokens", 5}}}};
    add_spec_usage_(usage, req);
    const auto& d = usage["completion_tokens_details"];
    EXPECT_EQ(d["reasoning_tokens"], 5);
    EXPECT_EQ(d["imp_spec_drafted"], 21);
    EXPECT_EQ(d["imp_spec_accepted"], 13);
    // emitted, not accepted, is what the caller bought: a verify emits the
    // accepted prefix PLUS the bonus token, and only this number prices the step.
    EXPECT_EQ(d["imp_spec_emitted"], 19);
    EXPECT_EQ(d["imp_spec_verify_steps"], 7);
    EXPECT_FALSE(d.contains("imp_spec_declined"));
}

// A request that ASKED for MTP and did not get it must say so, even though no
// verify step ran. Until now the refusal lived in one startup INFO line, so a
// caller sending {"speculative": {"mtp_k": 2}} to a concurrent server got a
// plain decode indistinguishable from a served one.
TEST(SpecUsage, DeclineIsReportedWithoutAnyVerifyStep) {
    auto req = std::make_shared<imp::Request>();
    req->spec_decline = imp::SpecDecline::kHeadNotLoaded;
    nlohmann::json usage = {{"prompt_tokens", 3}, {"completion_tokens", 2}};
    add_spec_usage_(usage, req);
    ASSERT_TRUE(usage.contains("completion_tokens_details"));
    const auto& d = usage["completion_tokens_details"];
    EXPECT_EQ(d["imp_spec_declined"], "mtp_head_not_loaded");
    EXPECT_NE(std::string(d["imp_spec_declined_detail"]).find("single-stream"), std::string::npos);
    EXPECT_EQ(d["imp_spec_drafted"], 0);
}

// "speculative": false is not news: the caller switched it off.
TEST(SpecUsage, TheCallersOwnOptOutIsNotReported) {
    auto req = std::make_shared<imp::Request>();
    req->spec_decline = imp::SpecDecline::kRequestOff;
    nlohmann::json usage = {{"prompt_tokens", 3}, {"completion_tokens", 2}};
    add_spec_usage_(usage, req);
    EXPECT_FALSE(usage.contains("completion_tokens_details"));
}

// ------------------------------------------------------ the request field

TEST(SpecField, AbsentLeavesBothTriStatesUnset) {
    const auto p = parse_spec_field_(nlohmann::json::object(), /*armed_mtp_k=*/2);
    EXPECT_TRUE(p.ok);
    EXPECT_EQ(p.spec_override, -1);
    EXPECT_EQ(p.mtp_k, -1);
}

TEST(SpecField, BooleanKeepsTheOldMeaning) {
    EXPECT_EQ(parse_spec_field_({{"speculative", true}}, 2).spec_override, 1);
    EXPECT_EQ(parse_spec_field_({{"speculative", false}}, 2).spec_override, 0);
    EXPECT_EQ(parse_spec_field_({{"speculative", true}}, 2).mtp_k, -1);
}

TEST(SpecField, ObjectSetsTheDepthAndLeavesTheBooleanAlone) {
    const auto p = parse_spec_field_({{"speculative", {{"mtp_k", 1}}}}, 2);
    EXPECT_TRUE(p.ok);
    EXPECT_EQ(p.mtp_k, 1);
    EXPECT_EQ(p.spec_override, -1) << "the object form addresses the head, not the matcher";
}

TEST(SpecField, ZeroIsAcceptedAndMeansOffForThisRequest) {
    const auto p = parse_spec_field_({{"speculative", {{"mtp_k", 0}}}}, 2);
    EXPECT_TRUE(p.ok);
    EXPECT_EQ(p.mtp_k, 0);
}

TEST(SpecField, OutOfRangeIs400WithTheRangeInTheMessage) {
    const auto over = parse_spec_field_({{"speculative", {{"mtp_k", 5}}}}, 2);
    EXPECT_FALSE(over.ok);
    EXPECT_NE(over.error.find("0..2"), std::string::npos) << over.error;
    const auto neg = parse_spec_field_({{"speculative", {{"mtp_k", -1}}}}, 2);
    EXPECT_FALSE(neg.ok);
    EXPECT_NE(neg.error.find("0..2"), std::string::npos) << neg.error;
}

// The model-less validation lane has no armed depth, and must still answer.
// The bound there is the device chain buffer, not zero: a server that answers
// "0..0" to every depth teaches the caller the wrong contract.
TEST(SpecField, WithoutAnArmedHeadTheBoundIsTheDeviceChainCap) {
    const auto ok = parse_spec_field_({{"speculative", {{"mtp_k", 2}}}}, /*armed_mtp_k=*/0);
    EXPECT_TRUE(ok.ok) << ok.error;
    const auto over = parse_spec_field_({{"speculative", {{"mtp_k", 99}}}}, 0);
    EXPECT_FALSE(over.ok);
    EXPECT_NE(over.error.find("0..16"), std::string::npos) << over.error;
}

TEST(SpecField, WrongTypesAreRefusedRatherThanIgnored) {
    EXPECT_FALSE(parse_spec_field_({{"speculative", "yes"}}, 2).ok);
    EXPECT_FALSE(parse_spec_field_({{"speculative", 1}}, 2).ok);
    EXPECT_FALSE(parse_spec_field_({{"speculative", {{"mtp_k", "2"}}}}, 2).ok);
    EXPECT_FALSE(parse_spec_field_({{"speculative", {{"mtp_k", 1.5}}}}, 2).ok);
}

}  // namespace
