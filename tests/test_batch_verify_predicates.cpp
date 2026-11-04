// Batched speculative verify predicates (runtime/spec_batch_verify_state.h).
#include <gtest/gtest.h>

#include "model/model.h"
#include "model/mtp_head.h"
#include "runtime/config.h"
#include "runtime/spec_batch_verify_state.h"
#include "runtime/spec_gates.h"

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
    EXPECT_EQ(mtp_bind_cap(cfg, &hybrid, 24), h) << "mtp_carry: every history slot can bind";
    EXPECT_EQ(mtp_bind_cap(cfg, &hybrid, 24, /*carry_live=*/false), 1) << "carry scratch refused";
    EXPECT_EQ(mtp_carry_rows(cfg, &hybrid, 24), kMtpCarryRows);
    cfg.speculative.mtp_carry = false;
    EXPECT_EQ(mtp_carry_rows(cfg, &hybrid, 24), 0);
    EXPECT_EQ(mtp_bind_cap(cfg, &hybrid, 24), 1) << "without the carry a second binding dies in a batch";
    cfg.speculative.batch_verify = true;
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &hybrid, 1), h);  // batch of one
    EXPECT_EQ(mtp_bind_cap(cfg, &hybrid, 24), 24);
    cfg.speculative.mtp_carry = true;
    EXPECT_EQ(mtp_carry_rows(cfg, &hybrid, 24), 0) << "the batched verify feeds every bound row itself";
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
    // speculative.mtp_batch_verify (4 rows): a c4 burst binds and verifies together; c32 stays ragged.
    EXPECT_TRUE(mtp_prompt_takes_head(false, true, 4, /*burst_bind=*/4));
    EXPECT_FALSE(mtp_prompt_takes_head(false, true, 5, 4));
    EXPECT_FALSE(mtp_prompt_takes_head(false, false, 2, 4)) << "every bind place taken";
    EXPECT_FALSE(mtp_prompt_takes_head(false, true, 0, 4));
}

// Qwen3.8-27B-NVFP4, 2 overlapping chats: req 35 unbound at its 8th lone verify ("verifies=8 emitted=12
// min_emit=1.60"); its response reported drafted 1138, accepted 659, almost all from batched verifies.
TEST(BatchVerifyPredicates, EconomicsGuardCountsBatchedVerifies) {
    EXPECT_FLOAT_EQ(mtp_econ_min_emit(8, 12, -1.0f), 1.60f);
    EXPECT_TRUE(mtp_econ_uneconomic(8, 12, 12, -1.0f)) << "the lone tail alone looks uneconomic";
    MtpBind active;
    MtpBindPool pool;
    active.req = 34;
    pool.binds[35] = MtpBind{};
    EXPECT_EQ(mtp_find_bind(active, pool, 34), &active);
    ASSERT_EQ(mtp_find_bind(active, pool, 35), &pool.binds[35]);
    EXPECT_EQ(mtp_find_bind(active, pool, 36), nullptr) << "a finished request has no tally";
    EXPECT_EQ(mtp_find_bind(active, pool, -1), nullptr);
    MtpBind& b = *mtp_find_bind(active, pool, 35);
    for (int i = 0; i < 563; ++i)  // 563 batched verifies x 2 drafts, 655 accepted
        mtp_econ_count(b, i < 92 ? 3 : 2, 2, /*batched=*/true);
    for (int i = 0; i < 8; ++i)  // the lone tail of the log line: 12 rows, 12 emitted
        mtp_econ_count(b, i < 4 ? 2 : 1, i < 4 ? 2 : 1, /*batched=*/false);
    EXPECT_EQ(b.econ_verifies, 571);
    EXPECT_EQ(b.econ_batched, 563);
    EXPECT_EQ(b.econ_rows, 1138);
    EXPECT_EQ(b.econ_emitted, 571 + 659);
    EXPECT_FALSE(mtp_econ_uneconomic(b.econ_verifies, b.econ_emitted, b.econ_rows, -1.0f));
    EXPECT_EQ(active.econ_verifies, 0) << "another binding's tally is untouched";
}

// speculative.mtp_econ_min_emit: 0 = off, positive = absolute floor, no verdict before 8 verifies.
TEST(BatchVerifyPredicates, EconomicsGuardConfig) {
    EXPECT_FALSE(mtp_econ_uneconomic(8, 8, 16, 0.0f));
    EXPECT_TRUE(mtp_econ_uneconomic(8, 15, 16, 2.0f));
    EXPECT_FALSE(mtp_econ_uneconomic(8, 16, 16, 2.0f));
    EXPECT_FALSE(mtp_econ_uneconomic(7, 7, 14, -1.0f));
    EXPECT_FLOAT_EQ(mtp_econ_min_emit(10, 30, -1.0f), 2.20f);
}

// A binding whose head cache lags its context (decoded in a batch) is dead; synced is not.
TEST(BatchVerifyPredicates, HeadCacheBehindAfterBatchedDecode) {
    EXPECT_FALSE(mtp_cache_behind(120, 120));
    EXPECT_TRUE(mtp_cache_behind(120, 135));
}

// Two chats decoding together (Qwen3.8-27B on main: 2 of 12 overlapped turns drafted). Chat A bound at
// mtp_pos 400; batched steps forward its inputs at 400, 401, ... and carry those hidden rows.
TEST(BatchVerifyPredicates, CarryTakesEveryBatchedStepInOrder) {
    EXPECT_EQ(mtp_carry_index(400, 0, 400), 0) << "first batched step after the lone feed";
    EXPECT_EQ(mtp_carry_index(400, 3, 403), 3) << "the next step appends";
    EXPECT_EQ(mtp_carry_index(400, 3, 402), 2) << "an abandoned chained step runs again: overwrite";
    EXPECT_EQ(mtp_carry_index(400, 3, 404), -1) << "a step without a row (loop burst, rider) breaks it";
    EXPECT_EQ(mtp_carry_index(400, 3, 399), -1) << "below the head cache";
    EXPECT_EQ(mtp_carry_index(400, -1, 400), -1) << "broken stays broken";
}

// Pair p = (token p + 1, h_p): a per-step batch knows every carried row's next token, a chained
// pipeline step (input = the in-flight token) one row less.
TEST(BatchVerifyPredicates, CarryFeedsOnlyRowsWhoseNextTokenIsKnown) {
    EXPECT_EQ(mtp_carry_feedable(400, 3, 404), 3) << "lone step at 403 after rows 400..402";
    EXPECT_EQ(mtp_carry_feedable(400, 4, 404), 3) << "row 403 is the in-flight step's";
    EXPECT_EQ(mtp_carry_feedable(400, 0, 404), 0);
    EXPECT_EQ(mtp_carry_feedable(400, -1, 404), 0);
    EXPECT_EQ(mtp_carry_feedable(400, 256, 900), 256) << "a finished request feeds the whole carry";
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

// speculative.mtp_batch_verify: two to four chats decoding on a default Qwen3.8-27B server verify their bound
// drafts in one step; the head keeps its history slots and the carry, one slot-swapping spare per row.
TEST(BatchVerifyPredicates, BoundVerifyKeepsTheHistorySlotsAndOneSparePerRow) {
    Model m;
    m.config_.ssm_inner_size = 6144;
    m.mtp_ = std::make_unique<MtpHead>();
    m.mtp_->loaded = true;
    RuntimeConfig cfg;
    EXPECT_EQ(cfg.speculative.mtp_batch_verify, 4) << "default on: 2-4 chats verify together";
    const int h = cfg.speculative.mtp_history_slots;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 4);
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 2) << "default: 2 drafts, 3 rows per request";
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 8) << "slot-swapping spares, two per row at 2 drafts";
    cfg.speculative.mtp_batch_verify_k = 1;
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 4) << "one draft: one spare per row";
    EXPECT_TRUE(batch_verify_wants_bufs(cfg, &m, 32));
    EXPECT_FALSE(batch_verify_on(cfg, &m)) << "the full batch_verify stays off";
    EXPECT_EQ(mtp_draft_kv_slots(cfg, &m, 32), h) << "not one head KV slot per batch slot";
    EXPECT_EQ(mtp_bind_cap(cfg, &m, 32), h);
    EXPECT_EQ(mtp_carry_rows(cfg, &m, 32), kMtpCarryRows) << "plain steps above the rows still carry";
    cfg.speculative.factored_spare = true;
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 4) << "a factored row needs every next step to apply it";
    cfg.speculative.mtp_batch_verify = 8;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), h) << "capped at the requests that can bind";
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 3), 3);
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 1), 0) << "batch of one: the single verify";
    cfg.speculative.mtp_batch_verify = 1;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0);
    EXPECT_FALSE(batch_verify_wants_bufs(cfg, &m, 32));
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 0);
    cfg.speculative.mtp_batch_verify = 4;
    cfg.speculative.mtp_carry = false;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0) << "one binding: nothing to batch";
    cfg.speculative.mtp_carry = true;
    cfg.speculative.mtp_k = 0;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0) << "imp-cli --bench pins mtp_k=0: no spares priced";
    cfg.speculative.mtp_k = -1;
    m.mtp_->loaded = false;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0) << "no head, no spares";
    m.mtp_->loaded = true;
    cfg.speculative.batch_verify = true;
    cfg.speculative.factored_spare = false;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0) << "the full batch_verify owns the step";
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 32);
    cfg.speculative.batch_verify = false;
    Model dense;
    dense.mtp_ = std::make_unique<MtpHead>();
    dense.mtp_->loaded = true;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &dense, 32), 0) << "no grouped verify scan to batch";
}

// An undrafted row forwards a pad that is never accepted: half the rows must draft.
TEST(BatchVerifyPredicates, BoundVerifyRunsWhenHalfTheRowsDraft) {
    EXPECT_TRUE(mtp_bound_verify_fits(4, 2, 1));
    EXPECT_TRUE(mtp_bound_verify_fits(4, 2, 2));
    EXPECT_FALSE(mtp_bound_verify_fits(4, 3, 1));
    EXPECT_TRUE(mtp_bound_verify_fits(4, 3, 2));
    EXPECT_TRUE(mtp_bound_verify_fits(4, 4, 2));
    EXPECT_FALSE(mtp_bound_verify_fits(4, 5, 5)) << "above the rows: plain batched decode";
    EXPECT_FALSE(mtp_bound_verify_fits(4, 2, 0));
    EXPECT_FALSE(mtp_bound_verify_fits(4, 1, 1)) << "a lone request takes the single verify";
    EXPECT_FALSE(mtp_bound_verify_fits(0, 2, 2)) << "off";
}

// speculative.mtp_batch_verify_solo: a lone bound request with a draft is a group of one; no draft, no step.
TEST(BatchVerifyPredicates, LoneBoundRequestVerifiesAsAGroupOfOne) {
    EXPECT_TRUE(mtp_bound_verify_fits(4, 1, 1, /*solo=*/true));
    EXPECT_FALSE(mtp_bound_verify_fits(4, 1, 0, /*solo=*/true));
    EXPECT_FALSE(mtp_bound_verify_fits(0, 1, 1, /*solo=*/true)) << "bound verify off";
    EXPECT_TRUE(mtp_bound_verify_solo(true, 1, true));
    EXPECT_FALSE(mtp_bound_verify_solo(false, 1, true)) << "key off";
    EXPECT_FALSE(mtp_bound_verify_solo(true, 2, true)) << "two decode: the plain bound verify";
    EXPECT_FALSE(mtp_bound_verify_solo(true, 1, false)) << "not bound: nothing drafts";
    RuntimeConfig cfg;
    EXPECT_TRUE(cfg.speculative.mtp_batch_verify_solo);
}

// speculative.mtp_batch_verify_k: 1 + k accepted rows name the slot that holds the state after them.
TEST(BatchVerifyPredicates, AcceptedDraftsPickTheSlotHoldingTheirState) {
    EXPECT_EQ(mtp_bound_verify_depth(0), 1);
    EXPECT_EQ(mtp_bound_verify_depth(2), 2);
    EXPECT_EQ(mtp_bound_verify_depth(5), 2) << "the step stages two snapshots at most";
    // One draft: reject keeps the live slot (row-1 snapshot), accept takes the spare (both rows).
    EXPECT_EQ(bound_verify_state_slot(0, 1), 0);
    EXPECT_EQ(bound_verify_state_slot(1, 1), 1);
    // Two drafts: 1 accepted takes the second spare (row-2 snapshot), 2 the spare (all 3 rows).
    EXPECT_EQ(bound_verify_state_slot(0, 2), 0);
    EXPECT_EQ(bound_verify_state_slot(1, 2), 2);
    EXPECT_EQ(bound_verify_state_slot(2, 2), 1);
    // Stage: 3 ints per row, 4 per request (seq, out, 2 snapshots), 3 row counts.
    EXPECT_EQ(batch_verify_stage_ints(12, 4), 36u + 16u + 3u);
}

// speculative.mtp_adaptive_k on the bound verify: AIMD between 1 and the ceiling (the chunk path's rule).
TEST(BatchVerifyPredicates, AdaptiveDepthShedsOnRejectAndGrowsOnFullAccept) {
    EXPECT_EQ(mtp_bound_verify_next_depth(2, 2, 2), 2) << "full accept at the ceiling stays";
    EXPECT_EQ(mtp_bound_verify_next_depth(2, 1, 2), 1) << "second draft rejected: one draft next";
    EXPECT_EQ(mtp_bound_verify_next_depth(2, 0, 2), 1);
    EXPECT_EQ(mtp_bound_verify_next_depth(1, 1, 2), 2) << "one draft accepted: the chain comes back";
    EXPECT_EQ(mtp_bound_verify_next_depth(1, 0, 2), 1) << "floor 1";
    EXPECT_EQ(mtp_bound_verify_next_depth(1, 1, 1), 1) << "a request that asked for mtp_k=1 stays at 1";
    EXPECT_EQ(mtp_bound_verify_next_depth(2, 2, 0), 1) << "floor holds against a zero ceiling";
}

}  // namespace
}  // namespace imp
