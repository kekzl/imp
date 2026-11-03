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

TEST(BatchVerifyPredicates, HistorySlotsWhenTheBatchedVerifyDoesNotRun) {
    Model hybrid;
    hybrid.config_.ssm_inner_size = 6144;
    RuntimeConfig cfg;
    const int h = cfg.speculative.mtp_history_slots;
    EXPECT_EQ(h, 4);
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &hybrid, 24), h);  // batch_verify off
    EXPECT_EQ(mtp_bind_cap(cfg, &hybrid, 24), 1) << "history slots never bind a second request";
    cfg.speculative.batch_verify = true;
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &hybrid, 1), h);  // batch of one
    EXPECT_EQ(mtp_bind_cap(cfg, &hybrid, 24), 24);
    Model dense;                                        // no recurrent state
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &dense, 24), h);
    cfg.speculative.hybrid = false;
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &hybrid, 24), h);
    cfg.speculative.mtp_history_slots = 0;
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

// Four head slots, bind cap 1: free history slots never let a prompt bind behind the active one.
TEST(BatchVerifyPredicates, HistorySlotsDoNotBindASecondRequest) {
    MtpBind active;
    MtpBindPool pool;
    pool.initialized = true;
    pool.free_slots = {1, 2, 3};
    active.req = 7;
    EXPECT_FALSE(mtp_slot_available(active, pool, /*bind_cap=*/1, 8));
    EXPECT_TRUE(mtp_slot_available(active, pool, /*bind_cap=*/4, 8)) << "batched verify: one per batch slot";
    active.req = -1;
    EXPECT_TRUE(mtp_slot_available(active, pool, 1, 8));
}

std::vector<int32_t> seq(int from, int n) {
    std::vector<int32_t> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = from + i;
    return v;
}

// Chat A (slot 0) decoded 1500 tokens over a 900-token prompt; a UI task request between two turns
// shares only the template's first tokens with it. Qwen3.8-27B on main: the task request took A's only
// slot and every later turn of A decoded without drafts (96 vs 120 tok/s).
TEST(BatchVerifyPredicates, PickSlotKeepsAChatsHeadCacheAcrossASideRequest) {
    const auto a_hist = seq(0, 2400);  // turn-3 transcript: prompt 0..899 + answer
    std::vector<int32_t> empty;
    std::vector<MtpSlotCand> c = {{.pos = 2399, .history = &a_hist, .last_use = 1},
                                  {.pos = 0, .history = &empty, .last_use = 0}};
    auto side = seq(0, 3);  // "<|im_start|>user\n" shared, then the task text
    side.insert(side.end(), 500, 99999);
    EXPECT_EQ(mtp_pick_slot(c, side, 0), 1) << "a 2-pair resume is not worth chat A's history";
    // A's next turn: the template dropped the old thinking, so the match ends at the old prompt
    // (900 tokens); the prefix cache resumes at its block-aligned snapshot 864.
    auto a_next = seq(0, 900);
    a_next.insert(a_next.end(), 600, 77777);
    EXPECT_EQ(mtp_pick_slot(c, a_next, 864), 0);
    EXPECT_EQ(mtp_pick_slot(c, a_next, 0), 0) << "a long resume wins on a fresh prompt too";
    EXPECT_EQ(mtp_pick_slot(c, a_next, 899), 0) << "resume min(pos, L - 1) = 899 covers offset 899";
    EXPECT_EQ(mtp_pick_slot(c, a_next, 900), -1) << "pair 899 needs h_899, the prefix never forwarded it";
    // The side request took slot 1; its history is no match for A, the decline keeps slot 0 intact.
    const auto side_hist = side;
    c[1] = {.pos = 502, .history = &side_hist, .last_use = 2};
    EXPECT_EQ(mtp_pick_slot(c, a_next, 864), 0);
    // A second chat B (fresh, short shared prefix) takes the least recently used slot, not A's.
    auto b = seq(0, 3);
    b.insert(b.end(), 200, 55555);
    EXPECT_EQ(mtp_pick_slot(c, b, 0), 0) << "slot 0 is the LRU one here";
    c[0].last_use = 3;
    EXPECT_EQ(mtp_pick_slot(c, b, 0), 1);
}

}  // namespace
}  // namespace imp
