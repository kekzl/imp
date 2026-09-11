#pragma once

// State of the batched speculative verify (engine_spec_batch_verify.cpp) and
// of the per-request MTP bindings (engine_spec_mtp.cpp), kept out of
// engine.h: docs/plans/2026-09-11-batched-mtp-verify.md.

#include "memory/host_pinned.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace imp {

// Device staging for one batched verify step, sized once from
// max_batch_size: [tok | pos | row ctx | seq | out | snap | chunk_len, snap_n],
// the per-row block tables, the argmax output (pinned host twins), and the
// spare recurrent slots: reserved pool slots past the multi-candidate ones,
// one per batch slot, held for a request's lifetime (accept swaps it with
// the live slot).
struct BatchVerifyState {
    int cap_seq = 0;
    int cap_rows = 0;
    int table_cap = 0;
    int32_t* d_stage = nullptr;
    int32_t* d_row_tables = nullptr;
    int32_t* d_argmax = nullptr;
    PinnedBuffer h_stage, h_row_tables, h_argmax;
    std::unordered_map<int, int> spare_of;  // req.id -> spare slot
    std::vector<int> free;
    bool initialized = false;
    mutable const char* last_refusal = nullptr;  // one log line per distinct reason
};

// One request's MTP binding: the ACTIVE one lives in Engine::mtp_active_
// (req >= 0), the others park in MtpBindPool::binds while another request
// is active. With one KV slot (batched verify off) a new request evicts the
// parked one, the behaviour the single workspace always had.
//
// Sync invariant of the active binding: ws->mtp_pos == req->context_len() - 1;
// pair i is (emb(t_{i+1}), h_i), history holds the covered tokens t_0..t_P
// (size == mtp_pos + 1); pending is the chain drafted for draft_ctx
// (context_len it targets), chains the multi-candidate form ([0] == pending).
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
    int k_live = 0;
    bool stale_logged = false;  // "drafting off" logged once per binding
};

struct MtpBindPool {
    std::unordered_map<int, MtpBind> binds;  // parked (never the active request)
    std::vector<int> free_slots;
    bool initialized = false;
    std::vector<std::vector<int32_t>> slot_history;  // per KV slot: tokens its cache covers
    // Margin-gate tallies (mtp_tree_width > 1) and the legacy one-step
    // prediction the accuracy counter scores.
    long long tree_branched = 0;
    long long tree_linear = 0;
    int pending_prediction = -1;
};

// Is req_id bound: the active binding or a parked one.
inline bool mtp_bound(const MtpBind& active, const MtpBindPool& pool, int req_id) {
    return req_id >= 0 && (active.req == req_id || pool.binds.count(req_id) != 0);
}

struct RuntimeConfig;
class Model;
// Config + model say yes (no batch-size term): speculative.batch_verify on a
// recurrent model with speculative.hybrid.
bool batch_verify_on(const RuntimeConfig& cfg, const Model* model);
// Reserved recurrent slots the batched verify adds to the pool: one per
// batch slot, 0 when off or at batch 1.
int batch_verify_spare_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size);

}  // namespace imp
