// Batched speculative verify predicates (runtime/spec_batch_verify_state.h).
#include <gtest/gtest.h>

#include "model/model.h"
#include "runtime/config.h"
#include "runtime/spec_batch_verify_state.h"

namespace imp {
namespace {

// The factored spare reserves no recurrent slot and the verify still runs, so
// the MTP draft pool keeps one KV slot per batch slot. Keyed on the spare count
// it fell to 1: 128 drafts over 127 verify steps at 24 streams.
TEST(BatchVerifyPredicates, FactoredSpareKeepsOneMtpSlotPerBatchSlot) {
    Model m;
    m.config_.ssm_inner_size = 6144;
    RuntimeConfig cfg;
    cfg.speculative.batch_verify = true;
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 24), 24);
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &m, 24), 24);
    cfg.speculative.factored_spare = true;
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 24), 0);
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &m, 24), 24);
}

TEST(BatchVerifyPredicates, OneMtpSlotWhenTheBatchedVerifyDoesNotRun) {
    Model hybrid;
    hybrid.config_.ssm_inner_size = 6144;
    RuntimeConfig cfg;
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &hybrid, 24), 1);  // batch_verify off
    cfg.speculative.batch_verify = true;
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &hybrid, 1), 1);  // batch of one
    Model dense;                                        // no recurrent state
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &dense, 24), 1);
    cfg.speculative.hybrid = false;
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &hybrid, 24), 1);
}

}  // namespace
}  // namespace imp
