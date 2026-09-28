// max_batch_size with host-resident MoE experts (runtime/expert_batch_policy.h): auto resolves
// to 1, an explicit batch is kept and warned about with the expert budget it leaves.

#include <gtest/gtest.h>

#include "runtime/expert_batch_policy.h"

#include <string>

namespace imp {
namespace {

TEST(ExpertBatchPolicy, AutoResolvesToOneWithHostExperts) {
    EXPECT_EQ(auto_batch_for_host_experts(32, 48), 1);  // Qwen3.8-Flash-Next: 48 host layers
    EXPECT_EQ(auto_batch_for_host_experts(4, 1), 1);
    EXPECT_EQ(auto_batch_for_host_experts(1, 48), 1);
}

TEST(ExpertBatchPolicy, AutoUnchangedWithoutHostExperts) {
    EXPECT_EQ(auto_batch_for_host_experts(32, 0), 32);
    EXPECT_EQ(auto_batch_for_host_experts(8, 0), 8);
    EXPECT_EQ(auto_batch_for_host_experts(1, 0), 1);
}

TEST(ExpertBatchPolicy, ExplicitBatchWarnsWithBudgetAndSlots) {
    // 361 slots/layer x 48 layers x 0.88 MiB (Flash-Next at batch 2): 14.9 GiB.
    const size_t bytes = 361ull * 48 * 922746;
    const std::string w = host_expert_batch_warning(8, true, bytes, 361);
    EXPECT_NE(w.find("max_batch_size 8"), std::string::npos) << w;
    EXPECT_NE(w.find("expert cache budget is 14.89 GiB"), std::string::npos) << w;
    EXPECT_NE(w.find("361 slots/layer"), std::string::npos) << w;
}

TEST(ExpertBatchPolicy, NoWarningAtBatchOneOrWithoutHostExperts) {
    EXPECT_TRUE(host_expert_batch_warning(1, true, 1ull << 30, 100).empty());
    EXPECT_TRUE(host_expert_batch_warning(8, false, 1ull << 30, 100).empty());
    EXPECT_TRUE(host_expert_batch_warning(0, true, 0, 0).empty());
}

}  // namespace
}  // namespace imp
