// Batched speculative verify predicates (runtime/spec_batch_verify_state.h).
#include <gtest/gtest.h>

#include "model/model.h"
#include "model/mtp_head.h"
#include "runtime/config.h"
#include "runtime/spec_batch_verify_state.h"
#include "runtime/spec_gates.h"

namespace imp {
namespace {

// MTP draft KV slots and bindings: the history slots, one binding without the carry scratch.
TEST(BatchVerifyPredicates, HistorySlotsBoundTheHeadSlotsAndBindings) {
    RuntimeConfig cfg;
    const int h = cfg.speculative.mtp_history_slots;
    EXPECT_EQ(h, 4);
    EXPECT_EQ(mtp_draft_kv_slots(cfg), h);
    EXPECT_EQ(mtp_bind_cap(cfg), h) << "the carry: every history slot can bind";
    EXPECT_EQ(mtp_bind_cap(cfg, /*carry_live=*/false), 1) << "carry scratch refused";
    cfg.speculative.mtp_history_slots = 0;
    EXPECT_EQ(mtp_draft_kv_slots(cfg), 1);
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
    EXPECT_EQ(mtp_pick_slot(c, a_next, 900), 0)
        << "pair 899 is past the prefix: the chat's own stale slot binds at 900";
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

// familiar, 4 agents on Qwen3.8-27B: chats 2-4 hit the shared system prompt past a wrapped ring, then every
// turn's hit reached past the slot's rows, 3851 of 4567 requests drafted nothing.
TEST(BatchVerifyPredicates, GapBindPicksTheLruSlotForANewChatOverASharedPrompt) {
    const auto a_hist = seq(0, 2400), b_hist = seq(0, 3000);
    std::vector<MtpSlotCand> c = {{.pos = 2399, .history = &a_hist, .last_use = 2},
                                  {.pos = 2999, .history = &b_hist, .last_use = 1}};
    auto d = seq(0, 900);  // the shared 900-token system prompt, then its own turn
    d.insert(d.end(), 300, 77777);
    EXPECT_EQ(mtp_pick_slot(c, d, 899), 1) << "both resume 899: the older slot";
    EXPECT_EQ(mtp_pick_slot(c, d, 1000), 1) << "same tie with the hit past both resumes";
    std::vector<int32_t> nothing;
    auto x = seq(5000, 40);
    std::vector<MtpSlotCand> two = {{.pos = 0, .history = &nothing, .last_use = 5},
                                    {.pos = 0, .history = &nothing, .last_use = 4}};
    EXPECT_EQ(mtp_pick_slot(two, x, 30), 1) << "no resume at all: LRU";
    EXPECT_EQ(mtp_pick_slot({}, x, 30), -1) << "no free slot";
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
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k, 4) << "default: 4 drafts, 5 rows per request";
    EXPECT_EQ(cfg.speculative.mtp_batch_verify_k_wide, 3)
        << "default: 3 drafts once 4 requests share the step";
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 4) << "factor row: one spare per row";
    cfg.speculative.mtp_batch_verify_k = 1;
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 4) << "one draft: one spare per row";
    EXPECT_EQ(mtp_draft_kv_slots(cfg), h) << "not one head KV slot per batch slot";
    EXPECT_EQ(mtp_bind_cap(cfg), h);
    cfg.speculative.mtp_batch_verify = 8;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), h) << "capped at the requests that can bind";
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 3), 3);
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 1), 0) << "batch of one: the single verify";
    cfg.speculative.mtp_batch_verify = 1;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0);
    EXPECT_EQ(batch_verify_spare_slots(cfg, &m, 32), 0);
    cfg.speculative.mtp_batch_verify = 4;
    cfg.speculative.mtp_k = 0;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0) << "imp-cli --bench pins mtp_k=0: no spares priced";
    cfg.speculative.mtp_k = -1;
    m.mtp_->loaded = false;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0) << "no head, no spares";
    m.mtp_->loaded = true;
    Model dense;
    dense.mtp_ = std::make_unique<MtpHead>();
    dense.mtp_->loaded = true;
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &dense, 32), 0) << "no grouped verify scan to batch";
    EXPECT_EQ(mtp_bound_verify_config_rows(cfg), 4) << "the config half does not see the model";
    cfg.speculative.hybrid = false;
    EXPECT_EQ(mtp_bound_verify_config_rows(cfg), 0);
    EXPECT_EQ(mtp_bound_verify_rows(cfg, &m, 32), 0);
}

// Engine::init applies it too: an embedder's mtp_k=2 does not forward a pad row per request.
TEST(BatchVerifyPredicates, UnsetVerifyDepthFollowsASmallerDraftDepth) {
    const auto depth = [](std::vector<std::string> sets) {
        RuntimeConfig cfg;
        EXPECT_TRUE(cfg.apply_overrides(sets).empty());
        mtp_verify_depth_follows_k(cfg);
        return cfg.speculative.mtp_batch_verify_k;
    };
    EXPECT_EQ(depth({}), 4) << "mtp_k=auto is not a depth";
    EXPECT_EQ(depth({"speculative.mtp_k=0"}), 4);
    EXPECT_EQ(depth({"speculative.mtp_k=2"}), 2);
    EXPECT_EQ(depth({"speculative.mtp_k=1"}), 1);
    EXPECT_EQ(depth({"speculative.mtp_k=3"}), 3);
    EXPECT_EQ(depth({"speculative.mtp_k=6"}), 4);
    EXPECT_EQ(depth({"speculative.mtp_k=1", "speculative.mtp_batch_verify_k=3"}), 3) << "the operator set it";
    EXPECT_EQ(depth({"speculative.mtp_k=2", "speculative.mtp_batch_verify_k=1"}), 1);
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

// A lone bound request with a draft is a group of one; no draft, no step.
TEST(BatchVerifyPredicates, LoneBoundRequestVerifiesAsAGroupOfOne) {
    EXPECT_TRUE(mtp_bound_verify_fits(4, 1, 1, /*solo=*/true));
    EXPECT_FALSE(mtp_bound_verify_fits(4, 1, 0, /*solo=*/true));
    EXPECT_FALSE(mtp_bound_verify_fits(0, 1, 1, /*solo=*/true)) << "bound verify off";
    EXPECT_TRUE(mtp_bound_verify_solo(1, true));
    EXPECT_FALSE(mtp_bound_verify_solo(2, true)) << "two decode: the plain bound verify";
    EXPECT_FALSE(mtp_bound_verify_solo(1, false)) << "not bound: nothing drafts";
    RuntimeConfig cfg;
}

// speculative.mtp_batch_verify_k: 1 + k accepted rows name the slot that holds the state after them.
TEST(BatchVerifyPredicates, AcceptedDraftsPickTheSlotHoldingTheirState) {
    EXPECT_EQ(mtp_bound_verify_depth(0, true), 1);
    EXPECT_EQ(mtp_bound_verify_depth(1, false), 1);
    EXPECT_EQ(mtp_bound_verify_depth(5, false), 1) << "without factor rows the step stages one draft";
    EXPECT_EQ(mtp_bound_verify_depth(3, true), 3);
    EXPECT_EQ(mtp_bound_verify_depth(4, true), 4);
    EXPECT_EQ(mtp_bound_verify_depth(8, true), 8);
    EXPECT_EQ(mtp_bound_verify_depth(9, true), 8) << "one fold applies kGdnFactorFoldMaxRows = 7 rows";
    // One draft: reject keeps the live slot (row-1 snapshot), accept takes the spare (both rows).
    EXPECT_EQ(bound_verify_kept_slot(0, 1), 0);
    EXPECT_EQ(bound_verify_kept_slot(1, 1), 1);
    // Factor row at two drafts: 2 accepted take the one spare (all 3 rows), 1 accepted keeps the live slot
    // and folds draft 1 into it, 0 keeps the live slot as is.
    EXPECT_EQ(bound_verify_kept_slot(0, 2), 0);
    EXPECT_EQ(bound_verify_kept_slot(1, 2), 0);
    EXPECT_EQ(bound_verify_kept_slot(2, 2), 1);
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 1, 2), 1);
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 0, 2), 0);
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 2, 2), 0) << "the spare already holds all 3 rows";
    // Three drafts: 1 or 2 accepted fold that many rows, 3 take the spare (all 4 rows).
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 1, 3), 1);
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 2, 3), 2);
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 3, 3), 0);
    EXPECT_EQ(bound_verify_kept_slot(2, 3), 0);
    EXPECT_EQ(bound_verify_kept_slot(3, 3), 1);
    // Four drafts: 1..3 accepted fold that many rows, 4 take the spare (all 5 rows).
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 3, 4), 3);
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 4, 4), 0);
    EXPECT_EQ(bound_verify_kept_slot(3, 4), 0);
    EXPECT_EQ(bound_verify_kept_slot(4, 4), 1);
    EXPECT_EQ(bound_verify_factor_row_folds(true, true, 1, 2), 0)
        << "a finished request keeps its slot as is";
    EXPECT_EQ(bound_verify_factor_row_folds(false, false, 1, 2), 0);
    EXPECT_EQ(bound_verify_factor_row_folds(true, false, 1, 1), 0) << "one draft: accept takes the spare";
    // Qwen3.8-27B: 48 layers x (48 heads x 260 floats + 10240 halfs) per draft before the last + 3 list
    // ints per group.
    EXPECT_EQ(bound_verify_factor_row_bytes(48, 1, 1, 48, 128, 128, 10240), 3379212u);
    EXPECT_EQ(bound_verify_factor_row_bytes(48, 4, 2, 48, 128, 128, 10240), (8u * 3379200u) + 48u);
    RuntimeConfig cfg;
    EXPECT_TRUE(mtp_bound_verify_factor_row(cfg));
    cfg.speculative.mtp_batch_verify_k = 1;
    EXPECT_FALSE(mtp_bound_verify_factor_row(cfg)) << "one draft: nothing to factor";
    cfg.speculative.mtp_batch_verify_k = 2;
    EXPECT_TRUE(mtp_bound_verify_factor_row(cfg)) << "two drafts: factor rows";
    // Stage: 3 ints per row, 3 per request (seq, out, snapshot), 2 row counts.
    EXPECT_EQ(batch_verify_stage_ints(12, 4), 36u + 12u + 2u);
}

// speculative.mtp_batch_verify_k_wide: a step of kMtpBoundVerifyWideRows requests or more verifies the wide
// depth.
TEST(BatchVerifyPredicates, StepDepthFollowsTheRequestsInTheStep) {
    EXPECT_EQ(kMtpBoundVerifyWideRows, 4);
    for (int n = 1; n < kMtpBoundVerifyWideRows; ++n) {
        EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, n), 4) << n;
    }
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 4), 3);
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 8), 3);
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 4, 4), 4) << "wide key = depth: one depth for every step";
    // speculative.mtp_batch_verify_k_streak: the step after a full accept, below the wide tier only.
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 1, 6), 6);
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 2, 6), 6);
    // The 8-row GEMM tile: 1 or 2 requests reach 7 drafts (8, 16 rows), 3 stay at 4 (15 rows, 6 = 21).
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 1, 8), 7);
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 2, 8), 7);
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 3, 6), 4);
    EXPECT_EQ(mtp_bound_verify_tile_depth(5, 3, 8), 7) << "18 rows sit in the 24-row tile";
    EXPECT_EQ(mtp_bound_verify_tile_depth(7, 1, 8), 7) << "a 9th row opens the second tile";
    EXPECT_EQ(mtp_bound_verify_tile_depth(8, 1, 6), 8) << "the depth itself is never cut";
    EXPECT_EQ(mtp_bound_verify_tile_depth(4, 0, 6), 6);
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 4, 6), 3) << "4 requests keep the wide depth";
    EXPECT_EQ(mtp_bound_verify_step_depth(4, 3, 1, 0), 4) << "no full accept before it";
    EXPECT_EQ(mtp_bound_verify_streak_depth(6, 4, true), 6);
    EXPECT_EQ(mtp_bound_verify_streak_depth(9, 4, true), kMtpBoundVerifyMaxDepth);
    EXPECT_EQ(mtp_bound_verify_streak_depth(0, 4, true), 4) << "0 = mtp_batch_verify_k";
    EXPECT_EQ(mtp_bound_verify_streak_depth(3, 4, true), 4) << "never below the depth";
    EXPECT_EQ(mtp_bound_verify_streak_depth(6, 2, false), 2) << "spares alone stage 2 drafts";
    EXPECT_TRUE(mtp_bound_verify_full_accept(4, 4, false));
    EXPECT_FALSE(mtp_bound_verify_full_accept(4, 3, false));
    EXPECT_FALSE(mtp_bound_verify_full_accept(0, 0, false)) << "an undrafted row accepts nothing";
    EXPECT_FALSE(mtp_bound_verify_full_accept(4, 4, true)) << "a finished request drafts no next step";
    // The ladder: a full accept of N drafts earns N + kMtpBoundVerifyStreakStep, up to the streak depth.
    EXPECT_EQ(mtp_bound_verify_streak_next(4, 8), 6);
    EXPECT_EQ(mtp_bound_verify_streak_next(6, 8), 8);
    EXPECT_EQ(mtp_bound_verify_streak_next(8, 8), 8);
    EXPECT_EQ(mtp_bound_verify_streak_next(4, 6), 6);
    EXPECT_EQ(mtp_bound_verify_streak_next(6, 6), 6);
    EXPECT_EQ(mtp_bound_verify_streak_next(0, 8), 0) << "no full accept: back to the depth";
    EXPECT_EQ(mtp_bound_verify_streak_next(4, 4), 4) << "key off: the streak depth is the depth";
    EXPECT_EQ(mtp_bound_verify_pin_depth(8, 4, true), 8);
    EXPECT_EQ(mtp_bound_verify_pin_depth(0, 4, true), 0);
    EXPECT_EQ(mtp_bound_verify_pin_depth(2, 4, true), 4);
    EXPECT_EQ(mtp_bound_verify_pin_depth(11, 4, true), kMtpBoundVerifyMaxDepth);
    EXPECT_EQ(mtp_bound_verify_pin_depth(8, 2, false), 0) << "spares alone stage 2 drafts";
    {
        BatchVerifyState bv;
        bv.depth = 4, bv.wide_depth = 3, bv.streak_depth = 8;
        EXPECT_EQ(bound_verify_step_depth(bv, 2), 4);
        bv.streak = 6;
        EXPECT_EQ(bound_verify_step_depth(bv, 2), 6);
        EXPECT_EQ(bound_verify_step_depth(bv, 4), 3);
        EXPECT_EQ(bound_verify_stage_depth(bv, 2), 6);
        // diagnostics.mtp_verify_pin: every step stages the pinned rows, the head still drafts the step
        // depth.
        bv.streak = 0, bv.pin = 8;
        EXPECT_EQ(bound_verify_stage_depth(bv, 2), 8);
        EXPECT_EQ(bound_verify_stage_depth(bv, 4), 3);
        EXPECT_EQ(bound_verify_step_depth(bv, 2), 4);
        bv.streak = 6;
        EXPECT_EQ(bound_verify_stage_depth(bv, 2), 8);
        EXPECT_EQ(bound_verify_step_depth(bv, 2), 6);
    }
    const auto top = [](std::vector<std::string> sets) {
        RuntimeConfig cfg;
        EXPECT_TRUE(cfg.apply_overrides(sets).empty());
        mtp_verify_depth_follows_k(cfg);
        return mtp_bound_verify_top_depth(cfg);
    };
    EXPECT_EQ(top({}), 7) << "the default ladder: 4 -> 6 -> 7";
    EXPECT_EQ(top({"speculative.mtp_batch_verify_k_streak=0"}), 4);
    EXPECT_EQ(top({"speculative.mtp_batch_verify_k_streak=6"}), 6);
    EXPECT_EQ(top({"speculative.mtp_batch_verify_k_streak=6", "speculative.mtp_k=5"}), 5)
        << "mtp_k caps a pass";
    EXPECT_EQ(top({"speculative.mtp_batch_verify_k_streak=6", "speculative.mtp_k=2"}), 2);
    EXPECT_EQ(top({"speculative.mtp_batch_verify_k_streak=8"}), 8);
    EXPECT_EQ(top({"speculative.mtp_batch_verify_k_streak=12"}), kMtpBoundVerifyMaxDepth);
    EXPECT_EQ(top({"diagnostics.mtp_verify_pin=8"}), 8) << "the pin sizes the stage like a streak";
    EXPECT_EQ(top({"diagnostics.mtp_verify_pin=6", "speculative.mtp_batch_verify_k_streak=8"}), 8);
    // The wide depth stays in 2..depth and needs the factor rows.
    EXPECT_EQ(mtp_bound_verify_wide_depth(3, 4, true), 3);
    EXPECT_EQ(mtp_bound_verify_wide_depth(9, 4, true), 4);
    EXPECT_EQ(mtp_bound_verify_wide_depth(0, 4, true), 2);
    EXPECT_EQ(mtp_bound_verify_wide_depth(3, 2, true), 2) << "never above the depth";
    EXPECT_EQ(mtp_bound_verify_wide_depth(3, 1, true), 1) << "one draft: nothing to narrow";
    EXPECT_EQ(mtp_bound_verify_wide_depth(1, 2, false), 2) << "no factor rows: the depth";
}

// A wide step (3 drafts) of 4 groups in buffers allocated for 4 drafts: the scan wrote 2 rows per group, so
// group g folds from row 2 * g.
TEST(BatchVerifyPredicates, FoldListsUseTheStepsRowStride) {
    const int live[] = {7, 5, 3, 1};
    const int fold[] = {1, 0, 2, 5};
    int32_t out[12] = {};
    ASSERT_EQ(bound_verify_fold_lists(/*levels=*/2, live, fold, 4, out), 3);
    const int32_t want[] = {7, 3, 1, /*first row*/ 0, 4, 6, /*rows*/ 1, 2, 2};
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(out[i], want[i]) << i;
    }
    ASSERT_EQ(bound_verify_fold_lists(/*levels=*/3, live, fold, 4, out), 3);
    EXPECT_EQ(out[3 + 2], 9) << "4 drafts: 3 rows per group";
    EXPECT_EQ(out[6 + 2], 3);
    const int none[] = {0, 0, 0, 0};
    EXPECT_EQ(bound_verify_fold_lists(2, live, none, 4, out), 0);
}

}  // namespace
}  // namespace imp
