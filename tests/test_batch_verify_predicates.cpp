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

// Concurrent server, one head KV slot: the first prompt takes the head (serial prefill), every
// prompt behind the active binding rides the ragged forward; the slot frees when it unbinds.
TEST(BatchVerifyPredicates, OnlyTheFirstPromptTakesTheSingleHeadSlot) {
    MtpBind active;
    MtpBindPool pool;
    EXPECT_TRUE(mtp_slot_available(active, pool, 1, 7)) << "fresh pool, nothing bound";
    active.req = 7;
    EXPECT_TRUE(mtp_slot_available(active, pool, 1, 7)) << "its own later chunks stay serial";
    EXPECT_FALSE(mtp_slot_available(active, pool, 1, 8)) << "the active binding never yields its slot";
    EXPECT_TRUE(mtp_slot_available(active, pool, 4, 8)) << "an uninitialised multi-slot pool has room";
    pool.initialized = true;
    EXPECT_FALSE(mtp_slot_available(active, pool, 1, 8));
    pool.free_slots.push_back(0);
    active.req = -1;  // mtp_unbind_: the slot goes back to the pool
    EXPECT_TRUE(mtp_slot_available(active, pool, 1, 8));
    pool.free_slots.clear();
    pool.binds[3] = MtpBind{};
    EXPECT_TRUE(mtp_slot_available(active, pool, 1, 8)) << "a parked binding is evictable";
}

// A lone chat prompt takes the free head; a burst rides the ragged forward unbound; a holder keeps it.
TEST(BatchVerifyPredicates, OnlyALonePromptTakesTheHead) {
    EXPECT_TRUE(mtp_prompt_takes_head(/*bound=*/false, /*slot_available=*/true, 1));
    EXPECT_FALSE(mtp_prompt_takes_head(false, true, 4)) << "binding one of a burst costs a serial forward";
    EXPECT_FALSE(mtp_prompt_takes_head(false, false, 1)) << "the active binding keeps the only slot";
    EXPECT_TRUE(mtp_prompt_takes_head(true, true, 4)) << "a bound prompt's later chunks stay serial";
}

// A binding whose head cache lags its context (decoded in a batch) is dead; synced is not.
TEST(BatchVerifyPredicates, HeadCacheBehindAfterBatchedDecode) {
    EXPECT_FALSE(mtp_cache_behind(120, 120));
    EXPECT_TRUE(mtp_cache_behind(120, 135));
}

}  // namespace
}  // namespace imp
