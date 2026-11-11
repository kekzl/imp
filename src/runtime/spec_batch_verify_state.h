#pragma once

// State of the batched speculative verify (engine_spec_batch_verify.cpp) and
// of the per-request MTP bindings (engine_spec_mtp.cpp), kept out of
// engine.h: docs/plans/2026-09-11-batched-mtp-verify.md.

#include "memory/host_pinned.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace imp {

// Device staging for one batched verify step, sized once from max_batch_size:
// [tok | pos | row ctx | seq | out | snap | chunk_len, snap_n], per-row block
// tables, argmax output (pinned host twins), and spare recurrent slots
// (reserved pool slots past the multi-candidate ones, one per batch slot,
// held for a request's lifetime; accept swaps it with the live slot).
struct BatchVerifyState {
    int cap_seq = 0;
    int cap_rows = 0;
    int table_cap = 0;
    bool bound_only = false;  // speculative.mtp_batch_verify: cap_seq = its rows, bound rows draft
    // Pipelined plain steps between two bound-verify attempts: 1 after a verify, doubling to 64 per decline.
    int yield_every = 1;
    int since_yield = 0;
    int32_t* d_stage = nullptr;
    int32_t* d_row_tables = nullptr;
    int32_t* d_argmax = nullptr;
    PinnedBuffer h_stage, h_row_tables, h_argmax;
    // Drafts per request (rows per request = depth + 1): 1, or speculative.mtp_batch_verify_k on the bound
    // verify.
    int depth = 1;
    std::unordered_map<int, int> spare_of;   // req.id -> spare slot (state after all depth + 1 rows)
    std::unordered_map<int, int>
        spare2_of;  // req.id -> second spare (state after 2 rows), depth 2, no factor row
    // Depth 2..3 with speculative.mtp_batch_verify_factor_row: no second spare, drafts 1..depth - 1 are also
    // rows of d_fac + d_tap at [group * fac_levels + level], folded into the live slot on a partial accept
    // (bound_verify_factor_row_fold).
    bool factor_row = false;
    int fac_levels = 0;  // depth - 1 with factor_row
    std::vector<int> free;
    // Factored spare (speculative.factored_spare,
    // docs/plans/2026-09-12-factored-verify-spare.md): drafted row as (g, k,
    // delta) per (layer, slot, head) plus one conv tap per (layer, slot),
    // instead of a second full recurrent slot. Indexed by recurrent SLOT: a
    // request keeps its slot across steps while its batch position moves.
    float* d_fac = nullptr;
    void* d_tap = nullptr;  // half
    int fac_stride = 0;     // floats per (slot, head)
    int64_t fac_layer_stride = 0;
    int64_t tap_layer_stride = 0;
    // Slots whose row the NEXT forward must apply (drafts accepted last
    // verify). Rebuilt every verify; the device twin drives the conv tap
    // apply, and the recurrent half reads the 0 sentinel a clear writes into
    // rows not listed.
    std::vector<int> pending;
    int* d_pending = nullptr;
    // Rows that must be cleared before anything can apply them: this verify's
    // rejected groups, plus slots released since the last one (e.g. a request
    // that finished right after an accept). Separate device buffer from
    // d_pending: the forward's tap apply may still be reading that one when
    // the clear is queued.
    std::vector<int> clear_slots;
    int* d_clear = nullptr;
    // Pinned staging for the two slot lists. A pageable cudaMemcpyAsync would
    // read the host vector after it is cleared, and a blocking cudaMemcpy on a
    // non-default stream synchronises the device once per step.
    PinnedBuffer h_pending, h_clear;
    bool initialized = false;
    mutable const char* last_refusal = nullptr;  // one log line per distinct reason
};

// One request's MTP binding: the ACTIVE one lives in Engine::mtp_active_
// (req >= 0); others park in MtpBindPool::binds while another request is
// active. With one KV slot (batched verify off) a new request evicts the
// parked one.
//
// Sync invariant of the active binding: ws->mtp_pos == req->context_len() - 1;
// pair i is (emb(t_{i+1}), h_i); history holds tokens t_0..t_P (size ==
// mtp_pos + 1); pending is the chain drafted for draft_ctx (the context_len
// it targets); chains is the multi-candidate form ([0] == pending).
struct MtpBind {
    int req = -1;
    int slot = 0;
    std::vector<int32_t> history, pending;
    std::vector<std::vector<int32_t>> chains;
    int draft_ctx = -1;
    // Economics guard (docs/plans/2026-08-31-mtp-multicandidate-hybrid.md):
    // emitted per MTP verify vs the k that ran; adaptive chain depth
    // (AIMD, 0 = not bound yet).
    int econ_verifies = 0;
    long long econ_emitted = 0, econ_rows = 0;
    int econ_batched = 0;  // of econ_verifies, from the batched verify (speculative.mtp_batch_verify)
    int k_live = 0;
    bool stale_logged = false;  // "drafting off" logged once per binding
    int carry_n = 0;  // speculative.mtp_carry: hidden rows waiting in the slot's carry, -1 = a step missed
};

struct MtpBindPool {
    std::unordered_map<int, MtpBind> binds;  // parked (never the active request)
    std::vector<int> free_slots;
    bool initialized = false;
    std::vector<std::vector<int32_t>> slot_history;  // per KV slot: tokens its cache covers
    std::vector<uint64_t> slot_used;                 // per KV slot: last acquire stamp (LRU), 0 = never
    uint64_t use_clock = 0;
    int bind_cap = 1;       // requests bound at once: mtp_bind_cap(); the other slots only keep history
    int declined_req = -1;  // last request refused a slot (one log line, not one per prefill chunk)
    // Margin-gate tallies (mtp_tree_width > 1) and the legacy one-step
    // prediction the accuracy counter scores.
    long long tree_branched = 0;
    long long tree_linear = 0;
    int pending_prediction = -1;
};

// Is req_id bound: the active binding or a parked one.
[[nodiscard]] inline bool mtp_bound(const MtpBind& active, const MtpBindPool& pool, int req_id) {
    return req_id >= 0 && (active.req == req_id || pool.binds.contains(req_id));
}

// req_id's binding, active or parked, without activating it; nullptr = not bound.
[[nodiscard]] inline MtpBind* mtp_find_bind(MtpBind& active, MtpBindPool& pool, int req_id) {
    if (req_id < 0)
        return nullptr;
    if (active.req == req_id)
        return &active;
    auto it = pool.binds.find(req_id);
    return it == pool.binds.end() ? nullptr : &it->second;
}

// One MTP-drafted verify of the binding: `rows` drafts, `emitted` tokens. The economics guard judges
// lone and batched verifies together; lone-only made 8 lone verifies after ~700 batched ones the sample.
inline void mtp_econ_count(MtpBind& b, int emitted, int rows, bool batched) {
    b.econ_verifies++;
    b.econ_emitted += emitted;
    b.econ_rows += rows;
    b.econ_batched += batched ? 1 : 0;
}

// Would req_id's prompt take a head KV slot (mtp_acquire_slot_): bound already, fewer than bind_cap
// requests bound, or a parked binding to evict. The ACTIVE binding never yields its slot, so a prompt
// arriving behind it drafts nothing and may ride the ragged prefill forward instead of the serial one.
// History-only slots (bind_cap < slots) never let a second request bind.
[[nodiscard]] inline bool mtp_slot_available(const MtpBind& active, const MtpBindPool& pool, int bind_cap,
                                             int req_id) {
    if (mtp_bound(active, pool, req_id))
        return true;
    const int held = (active.req >= 0 ? 1 : 0) + static_cast<int>(pool.binds.size());
    return held < bind_cap || !pool.binds.empty();
}

// Head-cache resume shorter than this on a fresh prompt (prefix-cache offset 0) is not worth a slot
// whose history another chat may still resume: the chat template's first tokens match every slot.
inline constexpr int kMtpResumeMinPairs = 64;

// Head KV slot for a prompt whose prefill starts at `offset` (prefix-cache hit), or -1. A slot resumes
// at min(pos, L - 1) pairs, L = common prefix of its history and the prompt. Longest resume wins; without
// a kMtpResumeMinPairs resume the least recently used slot. Pairs below `offset` need hidden rows the
// cached prefix never forwarded: with no slot covering it, gap_bind picks as above (the head starts at
// `offset`, mtp_bind_has_gap), else -1 = draft nothing and leave every history intact.
struct MtpSlotCand {
    int pos = 0;                                    // pairs its head KV holds
    const std::vector<int32_t>* history = nullptr;  // tokens t_0..t_pos
    uint64_t last_use = 0;
};
[[nodiscard]] inline int mtp_pick_slot(const std::vector<MtpSlotCand>& cands, const std::vector<int32_t>& in,
                                       int offset, bool gap_bind = true) {
    int best = -1, best_r = -1, lru = -1;
    for (int i = 0; i < static_cast<int>(cands.size()); ++i) {
        const auto& h = *cands[i].history;
        size_t L = 0;
        while (L < h.size() && L < in.size() && h[L] == in[L])
            ++L;
        const int r = L > 0 ? std::min(cands[i].pos, static_cast<int>(L) - 1) : -1;
        if (r > best_r || (r == best_r && best >= 0 && cands[i].last_use < cands[best].last_use)) {
            best = i;
            best_r = r;
        }
        if (lru < 0 || cands[i].last_use < cands[lru].last_use)
            lru = i;
    }
    if (offset > 0 && best_r >= offset)
        return best;
    if (offset > 0 && !gap_bind)
        return -1;
    return best_r >= kMtpResumeMinPairs ? best : lru;
}

// A prompt takes the head when it prefills alone (a lone chat) or already holds it; the prompts of
// a burst ride the ragged forward unbound (binding one would cost it a serial forward of its own),
// unless the burst is no wider than speculative.mtp_batch_verify's rows (burst_bind), which verify it.
[[nodiscard]] constexpr bool mtp_prompt_takes_head(bool bound, bool slot_available, int prompts_this_step,
                                                   int burst_bind = 1) {
    return bound ||
           (slot_available && prompts_this_step >= 1 && prompts_this_step <= std::max(1, burst_bind));
}

// A bound request drafts only while the head cache covers exactly its pre-step context. Tokens
// decoded without a host feed and without a carried row (speculative.mtp_carry off, a ragged rider row,
// a loop burst) leave no hidden rows to catch up from: such a binding is dead.
[[nodiscard]] constexpr bool mtp_cache_behind(int mtp_pos, int cur_pos) { return mtp_pos != cur_pos; }

// speculative.mtp_carry: hidden rows per head KV slot, fed in one pass when full (2.5 MiB per slot on
// Qwen3.8).
inline constexpr int kMtpCarryRows = 256;
// Carry row of a batched step whose input token sits at `pos` (carry rows are
// positions mtp_pos..mtp_pos+carry_n-1). Below the end = the step ran again (rows past it drop); -1 = rows
// missing in between or a carry already broken.
[[nodiscard]] constexpr int mtp_carry_index(int mtp_pos, int carry_n, int pos) {
    return carry_n < 0 || pos < mtp_pos || pos > mtp_pos + carry_n ? -1 : pos - mtp_pos;
}
// Carried rows the head can take: pair p needs token p + 1, known up to position context_len - 1.
[[nodiscard]] constexpr int mtp_carry_feedable(int mtp_pos, int carry_n, int context_len) {
    return carry_n <= 0 ? 0 : std::clamp(context_len - 1 - mtp_pos, 0, carry_n);
}

struct RuntimeConfig;
class Model;
// Config + model say yes (no batch-size term): speculative.batch_verify on a
// recurrent model with speculative.hybrid.
[[nodiscard]] bool batch_verify_on(const RuntimeConfig& cfg, const Model* model);
// Reserved recurrent slots the batched verify adds to the pool: one per
// batch slot, 0 when off or at batch 1.
// Whether the batched verify needs its staging buffers (the full batch_verify or the bound one). Separate
// from the spare-slot count because the factored form reserves no slots and still runs.
[[nodiscard]] bool batch_verify_wants_bufs(const RuntimeConfig& cfg, const Model* model, int max_batch_size);
// The factored path is live only once its rows exist: a geometry or allocation
// refusal clears the flag, and everything downstream reads this, not the flag.
[[nodiscard]] bool factored_spare_active(const RuntimeConfig& cfg, const BatchVerifyState& bv);
int batch_verify_spare_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size);
// speculative.mtp_batch_verify: the most decoding requests whose bound members verify their MTP draft in one
// batched step (slot-swapping spares, one per row), 0 = never. Off under the full batch_verify, without a
// loaded head or the carry (one binding), on a model without the grouped verify scan, at batch 1.
[[nodiscard]] int mtp_bound_verify_rows(const RuntimeConfig& cfg, const Model* model, int max_batch_size);
// Its config half, before the model and the batch size cap it (tools/common/mtp_auto.cpp): 0 = never.
[[nodiscard]] int mtp_bound_verify_config_rows(const RuntimeConfig& cfg);
// An unset speculative.mtp_batch_verify_k follows a smaller mtp_k down: a row past the drafted depth is a pad
// (Qwen3.8-27B-NVFP4, 1 chat x 20k, 1 draft: 124.11 tok/s at 3 rows, 137.69 at 2).
void mtp_verify_depth_follows_k(RuntimeConfig& cfg);
// Every decoding row is no wider than the bound verify takes, and the verify pays: at least one row drafts
// and drafted rows are at least half the batch (an undrafted row forwards a pad that is never accepted).
[[nodiscard]] constexpr bool mtp_bound_verify_fits(int rows_cap, int decoding, int drafted,
                                                   bool solo = false) {
    return rows_cap >= 2 && decoding >= (solo ? 1 : 2) && decoding <= rows_cap && drafted >= 1 &&
           2 * drafted >= decoding;
}
// speculative.mtp_batch_verify_solo: a lone bound request takes the batched verify step as a group of one.
[[nodiscard]] constexpr bool mtp_bound_verify_solo(bool solo_on, int decoding, bool bound) {
    return solo_on && decoding == 1 && bound;
}
// speculative.mtp_batch_verify_k clamped to what the step stages: 2 drafts per request on spares alone (two
// snapshots), kMtpBoundVerifyMaxDepth with factor rows (a fold applies kGdnFactorFoldMaxRows of them).
inline constexpr int kMtpBoundVerifyMaxDepth = 3;
[[nodiscard]] constexpr int mtp_bound_verify_depth(int key, bool factor_row) {
    return std::clamp(key, 1, factor_row ? kMtpBoundVerifyMaxDepth : 2);
}
// Which recurrent slot holds the state after a group's 1 + matched rows: 0 the live slot (its in-place
// snapshot at row 1), 1 the spare (commit at all depth + 1 rows), 2 the second spare (snapshot at row 2).
[[nodiscard]] constexpr int bound_verify_state_slot(int matched, int depth) {
    return matched <= 0 ? 0 : (matched >= depth ? 1 : 2);
}
// Factor rows (speculative.mtp_batch_verify_factor_row at 2+ drafts): the one spare holds the state after
// all depth + 1 rows, so only a full accept adopts it (role 1); a partial accept keeps the live slot and
// folds.
[[nodiscard]] constexpr int bound_verify_factor_row_slot(int matched, int depth) {
    return matched >= depth ? 1 : 0;
}
[[nodiscard]] constexpr int bound_verify_kept_slot(int matched, int depth, bool factor_row) {
    return factor_row ? bound_verify_factor_row_slot(matched, depth)
                      : bound_verify_state_slot(matched, depth);
}
// Factor rows and conv taps a partial accept folds into the live slot (1 row -> 1 + matched rows): the
// matched drafts, 0 on a full accept (the spare), a reject or a finished request.
[[nodiscard]] constexpr int bound_verify_factor_row_folds(bool factor_row, bool done, int matched,
                                                          int depth) {
    return factor_row && !done && matched >= 1 && matched < depth ? matched : 0;
}
// Spare recurrent slots per bound row.
[[nodiscard]] constexpr int bound_verify_spares_per_row(int depth, bool factor_row) {
    return factor_row ? 1 : depth;
}
// Device bytes of the factor rows: (4 + ss + hd) floats per (layer, group, level, head), ch halfs per (layer,
// group, level), 3 fold-list ints per group. Qwen3.8-27B (48 layers, 48 heads, 128/128, 10240 channels):
// 3.22 MiB per (group, level).
[[nodiscard]] constexpr size_t bound_verify_factor_row_bytes(int layers, int groups, int levels, int heads,
                                                             int ss, int hd, int ch) {
    const size_t per_layer = (static_cast<size_t>(heads) * static_cast<size_t>(4 + ss + hd) * sizeof(float)) +
                             (static_cast<size_t>(ch) * sizeof(uint16_t));
    return static_cast<size_t>(groups) *
           ((static_cast<size_t>(levels) * static_cast<size_t>(layers) * per_layer) + (3 * sizeof(int32_t)));
}
// The key, at 2+ drafts.
[[nodiscard]] bool mtp_bound_verify_factor_row(const RuntimeConfig& cfg);
// Plan charge of the factor rows: one per (verify group, draft before the last).
[[nodiscard]] size_t mtp_bound_verify_factor_row_plan_bytes(const RuntimeConfig& cfg, const Model* model,
                                                            int max_batch_size);
// Staged ints: [tok | pos | row ctx] per row, [seq | out | snap x2] per request, [chunk_len, snap_n x2].
[[nodiscard]] constexpr size_t batch_verify_stage_ints(int cap_rows, int cap_seq) {
    return (3ull * cap_rows) + (4ull * cap_seq) + 3;
}
// MTP draft KV slots: one per batch slot whenever the batched verify runs, else
// speculative.mtp_history_slots (one binds, the rest keep other chats' head caches).
// Keyed on the verify, not on batch_verify_spare_slots: the factored spare makes
// that 0, and the pool fell to 1 slot (1 draft per verify step at 24 streams).
int mtp_draft_kv_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size);
// Requests bound to the head at once: one per batch slot under the batched verify, else
// speculative.mtp_history_slots with speculative.mtp_carry (carry_live: its scratch exists), else 1.
int mtp_bind_cap(const RuntimeConfig& cfg, const Model* model, int max_batch_size, bool carry_live = true);
// Carry rows per head KV slot to allocate: kMtpCarryRows with speculative.mtp_carry and no batched verify.
int mtp_carry_rows(const RuntimeConfig& cfg, const Model* model, int max_batch_size);

}  // namespace imp
