// AUDIT_arch_2026 C-6: the per-request speculation counters reach `usage`
// under vendor-prefixed keys, and only when a verify step ran.
#include <gtest/gtest.h>
#include "handlers_internal.h"  // tools/imp-server/handlers_internal.h
#include "runtime/request.h"
#include <memory>

namespace {

TEST(SpecUsage, AbsentWithoutAVerifyStep) {
    auto req = std::make_shared<imp::Request>();
    nlohmann::json usage = {{"prompt_tokens", 3}, {"completion_tokens", 2}};
    add_spec_usage_(usage, req);
    EXPECT_FALSE(usage.contains("completion_tokens_details"));
    add_spec_usage_(usage, nullptr);
    EXPECT_FALSE(usage.contains("completion_tokens_details"));
}

TEST(SpecUsage, ThreeCountersNextToTheExistingDetails) {
    auto req = std::make_shared<imp::Request>();
    req->spec_verifies = 7;
    req->spec_drafted = 21;
    req->spec_accepted = 13;
    nlohmann::json usage = {{"prompt_tokens", 3},
                            {"completion_tokens", 2},
                            {"completion_tokens_details", {{"reasoning_tokens", 5}}}};
    add_spec_usage_(usage, req);
    const auto& d = usage["completion_tokens_details"];
    EXPECT_EQ(d["reasoning_tokens"], 5);
    EXPECT_EQ(d["imp_spec_drafted"], 21);
    EXPECT_EQ(d["imp_spec_accepted"], 13);
    EXPECT_EQ(d["imp_spec_verify_steps"], 7);
}

}  // namespace
