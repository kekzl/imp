// max_batch_size with host-resident MoE experts (exec/expert_batch_policy.h): auto takes up to
// kHostExpertAutoBatch rows when the host pins the experts, 1 on the host expert tier; an explicit
// batch the device cache cannot run as one set of rows is warned about.

#include <gtest/gtest.h>

#include "exec/expert_batch_policy.h"

#include <string>

namespace imp {
namespace {

TEST(ExpertBatchPolicy, AutoCapsAtDeviceRowsWithPinnedHostExperts) {
    EXPECT_EQ(auto_batch_for_host_experts(32, 48, true),
              kHostExpertAutoBatch);  // Qwen3.8-Flash-Next: 48 host layers
    EXPECT_EQ(auto_batch_for_host_experts(4, 1, true), 4);
    EXPECT_EQ(auto_batch_for_host_experts(1, 48, true), 1);
}

TEST(ExpertBatchPolicy, AutoResolvesToOneOnTheHostExpertTier) {
    EXPECT_EQ(auto_batch_for_host_experts(32, 48, false), 1);
    EXPECT_EQ(auto_batch_for_host_experts(4, 1, false), 1);
}

TEST(ExpertBatchPolicy, AutoDropsToOneWhenTheUploadLeftExpertsUnpinned) {
    // 64 GiB host: the resolver read enough RAM to pin, the upload did not.
    EXPECT_EQ(batch_after_host_expert_pin(8, true, true), 1);
    EXPECT_EQ(batch_after_host_expert_pin(8, true, false), 8);
    EXPECT_EQ(batch_after_host_expert_pin(8, false, true), 8);  // explicit batch kept (warned at KV init)
}

TEST(ExpertBatchPolicy, AutoUnchangedWithoutHostExperts) {
    EXPECT_EQ(auto_batch_for_host_experts(32, 0, true), 32);
    EXPECT_EQ(auto_batch_for_host_experts(8, 0, false), 8);
    EXPECT_EQ(auto_batch_for_host_experts(1, 0, true), 1);
}

TEST(ExpertBatchPolicy, KvCeilingIsOneContextWithHostNvfp4Experts) {
    EXPECT_EQ(kv_ceiling_seqs(8, true), kHostExpertKvSeqs);
    EXPECT_EQ(kv_ceiling_seqs(1, true), 1);
    EXPECT_EQ(kv_ceiling_seqs(8, false), 8);
}

TEST(ExpertBatchPolicy, WarnsPastDeviceRowsWithBudgetAndSlots) {
    // 422 slots/layer x 48 layers x 0.88 MiB (Flash-Next): 17.39 GiB.
    const size_t bytes = 422ull * 48 * 922746;
    const std::string w = host_expert_batch_warning(16, true, false, bytes, 422);
    EXPECT_NE(w.find("max_batch_size 16"), std::string::npos) << w;
    EXPECT_NE(w.find("past 8"), std::string::npos) << w;
    EXPECT_NE(w.find("expert cache 17.41 GiB"), std::string::npos) << w;
    EXPECT_NE(w.find("422 slots/layer"), std::string::npos) << w;
}

TEST(ExpertBatchPolicy, WarnsOnTieredLayersAtAnyBatch) {
    const std::string w = host_expert_batch_warning(2, true, true, 1ull << 30, 100);
    EXPECT_NE(w.find("on host expert tier layers"), std::string::npos) << w;
}

TEST(ExpertBatchPolicy, NoWarningWhenTheDeviceCacheRunsTheRows) {
    EXPECT_TRUE(host_expert_batch_warning(8, true, false, 1ull << 30, 100).empty());
    EXPECT_TRUE(host_expert_batch_warning(1, true, true, 1ull << 30, 100).empty());
    EXPECT_TRUE(host_expert_batch_warning(16, false, false, 1ull << 30, 100).empty());
    EXPECT_TRUE(host_expert_batch_warning(0, true, false, 0, 0).empty());
}

}  // namespace
}  // namespace imp
