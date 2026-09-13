#pragma once

// State of the batched speculative verify (engine_spec_batch_verify.cpp) and
// of the per-request MTP bindings (engine_spec_mtp.cpp), kept out of
// engine.h: docs/plans/2026-09-11-batched-mtp-verify.md.

#include "memory/host_pinned.h"

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
    int32_t* d_stage = nullptr;
    int32_t* d_row_tables = nullptr;
    int32_t* d_argmax = nullptr;
    PinnedBuffer h_stage, h_row_tables, h_argmax;
    std::unordered_map<int, int> spare_of;  // req.id -> spare slot
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
// Whether the batched verify needs its staging buffers. Separate from the
// spare-slot count because the factored form reserves no slots and still runs.
bool batch_verify_wants_bufs(const RuntimeConfig& cfg, const Model* model, int max_batch_size);
// The factored path is live only once its rows exist: a geometry or allocation
// refusal clears the flag, and everything downstream reads this, not the flag.
bool factored_spare_active(const RuntimeConfig& cfg, const BatchVerifyState& bv);
int batch_verify_spare_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size);
// MTP draft KV slots: one per batch slot whenever the batched verify runs, else 1.
// Keyed on the verify, not on batch_verify_spare_slots: the factored spare makes
// that 0, and the pool fell to 1 slot (1 draft per verify step at 24 streams).
int mtp_draft_kv_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size);

}  // namespace imp
