// engine_spec_batch_verify.cpp: batched speculative verify on the GDN hybrid.
//
// One forward per decode step for N requests, D + 1 rows each: the request's
// last emitted token and up to D drafts (D = 1; the bound verify takes
// speculative.mtp_batch_verify_k, up to 4). Group g runs the GDN layers on its
// live slot L_g, commits the (D + 1)-row state into the spare P_g, the 1-row
// state in place into L_g and the states between as factor rows (or, D = 2 without
// them, the 2-row state into a second spare Q_g; compute/gdn.h out_slots /
// snap_slots / snap_rows). Accepting k drafts swaps the slot holding 1 + k rows
// with L_g on the host, or folds k factor rows into L_g. No state copies, no replay
// (docs/plans/2026-09-11-batched-mtp-verify.md).
// Attention rows take the decode-attention route with per-row block tables
// and context lengths, the same shape as the multi-candidate chunk.

#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "model/mtp_head.h"
#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "compute/gdn_factor.cuh"
#include "runtime/ngram_draft.h"
#include "runtime/request.h"
#include "runtime/spec_accept.h"
#include "runtime/spec_trace.h"
#include "runtime/spec_verify_factor_rows.h"
#include "runtime/stop_mask.h"
#include "runtime/suffix_draft.h"
#include "runtime/think_stop_logic.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <span>

namespace imp {

namespace {
// Per-row conditions the batched verify replicates: rep / freq / presence
// penalties over the unbounded window (per-row device histories), no logit
// shaping, no constraint FSM, no logprobs. Sampling: spec_sampling_refusal_.
bool batch_verify_row_ok(const Request& r) {
    const bool penalties = r.repetition_penalty != 1.0f || r.frequency_penalty != 0.0f ||
                           r.presence_penalty != 0.0f;
    if (penalties && r.repeat_last_n != 0)
        return false;  // a bounded window slides per chunk row; not replicated
    if (r.dry_multiplier != 0.0f || r.mirostat != 0 || r.host_logit_shaping())
        return false;
    if (r.logprobs || r.json_mode || !r.json_schema.empty() || !r.regex_pattern.empty() ||
        !r.grammar.empty() || !r.tool_constraint_tools.empty())
        return false;
    return r.status == RequestStatus::DECODING && !r.output_tokens.empty();
}

// speculative.mtp_batch_verify refusals that flip from step to step: never logged.
constexpr const char* kAboveBoundRows = "above_bound_rows";
constexpr const char* kTooFewBound = "too_few_bound";

int bound_rows(const std::vector<std::shared_ptr<Request>>& batch, const MtpBind& active,
               const MtpBindPool& pool) {
    return static_cast<int>(std::count_if(batch.begin(), batch.end(),
                                          [&](const auto& r) { return mtp_bound(active, pool, r->id); }));
}

// A lone bound request verifies in the bound batched step, a group of one.
bool bound_solo(const RuntimeConfig& cfg, const Model* model, const BatchVerifyState& bv,
                const MtpBind& active, const MtpBindPool& pool,
                const std::vector<std::shared_ptr<Request>>& batch) {
    return bv.bound_only && !batch.empty() && !batch_verify_on(cfg, model) &&
           mtp_bound_verify_solo(static_cast<int>(batch.size()), mtp_bound(active, pool, batch[0]->id));
}

// Mode and batch-shape gates ahead of the per-row ones: the full batch_verify takes any batch of two or more;
// the bound one (speculative.mtp_batch_verify) up to its rows, with a loaded head and enough bound rows; solo
// = a lone bound request under the solo bound verify.
const char* batch_mode_refusal(bool full, const BatchVerifyState& bv, int rows, int bound, bool head_on,
                               bool solo) {
    if (!full && !bv.bound_only)
        return "off";
    if (rows < 2 && (full || !solo))
        return "batch_of_one";
    if (full)
        return nullptr;
    if (!head_on || rows > bv.cap_seq)
        return kAboveBoundRows;
    return mtp_bound_verify_fits(bv.cap_seq, rows, bound, solo) ? nullptr : kTooFewBound;
}

bool refusal_logged(bool full, const BatchVerifyState& bv, const char* why) {
    return full || (bv.bound_only && why != kAboveBoundRows && why != kTooFewBound);
}

// After drafting: nothing to verify, or (bound verify) too few drafted rows to pay for the pads.
const char* draft_refusal(const BatchVerifyState& bv, int rows, int drafted, bool solo) {
    if (drafted == 0)
        return "no_draft_in_batch";
    return bv.bound_only && !mtp_bound_verify_fits(bv.cap_seq, rows, drafted, solo) ? "too_few_drafts"
                                                                                    : nullptr;
}

// Group g's T rows: t0 at p0, then its nd drafts; rows past them are pads (token t0, attending one token).
void stage_group_rows(int32_t* tok, int32_t* pos, int32_t* ctx, int T, int g, int32_t t0, const int32_t* dr,
                      int nd, int p0) {
    for (int j = 0; j < T; ++j) {
        const size_t r = (static_cast<size_t>(T) * g) + j;
        const bool pad = j > nd;
        tok[r] = j == 0 || pad ? t0 : dr[j - 1];
        pos[r] = p0 + j;
        ctx[r] = pad ? 1 : p0 + 1 + j;
    }
}

// Step price per request count, logged every 100 verify steps: forward + argmax (host wall through the argmax
// sync), the MTP head pass after it (feed, drafts, sync) and the tokens emitted per step.
void log_step_price(int n, int rows_per_req, double fwd_ms, double head_ms, long long emitted) {
    constexpr int kMaxN = 8;
    struct Price {
        double fwd_ms = 0.0, head_ms = 0.0;
        long long steps = 0, emitted = 0;
    };
    static Price price[kMaxN + 1];
    static long long total = 0;
    Price& p = price[std::clamp(n, 0, kMaxN)];
    p.fwd_ms += fwd_ms;
    p.head_ms += head_ms;
    p.emitted += emitted;
    p.steps++;
    if (++total % 100 != 0)
        return;
    for (int i = 2; i <= kMaxN; ++i) {
        const Price& q = price[i];
        if (q.steps > 0)
            IMP_LOG_INFO(
                "spec-batch price: %d requests x %d rows, %lld steps: forward+argmax %.2f ms, head %.2f "
                "ms, %.2f tokens/step",
                i, rows_per_req, q.steps, q.fwd_ms / q.steps, q.head_ms / q.steps,
                static_cast<double>(q.emitted) / q.steps);
    }
}

// Every row of group g attends through the request's block table.
void stage_group_tables(int32_t* tab, int table_cap, int T, int g, const std::vector<int>& bt) {
    for (int j = 0; j < T; ++j)
        std::copy(bt.begin(), bt.end(), tab + (((static_cast<size_t>(T) * g) + j) * table_cap));
}
}  // namespace

namespace {
// Mamba2 layers have no grouped verify scan (run_ssm decodes the slot table one token per row).
bool grouped_verify_model(const RuntimeConfig& cfg, const Model* model) {
    return cfg.speculative.hybrid && model != nullptr && model->config().ssm_inner_size > 0 &&
           !model->profile().has_pure_ssm;
}

// The full batch_verify (every decoding row, one head KV slot per batch slot) at a batch above one.
bool batch_verify_full(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    return batch_verify_on(cfg, model) && max_batch_size > 1;
}
}  // namespace

bool batch_verify_on(const RuntimeConfig& cfg, const Model* model) {
    return cfg.speculative.batch_verify && grouped_verify_model(cfg, model);
}

int mtp_bound_verify_config_rows(const RuntimeConfig& cfg) {
    const auto& scfg = cfg.speculative;
    if (scfg.mtp_batch_verify < 2 || scfg.mtp_k == 0 || !scfg.hybrid || scfg.batch_verify)
        return 0;
    const int rows = std::min(scfg.mtp_batch_verify, std::max(1, scfg.mtp_history_slots));
    return rows >= 2 ? rows : 0;
}

int mtp_bound_verify_rows(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    if (!grouped_verify_model(cfg, model) || !model->mtp_ || !model->mtp_->loaded)
        return 0;
    const int rows = std::min(mtp_bound_verify_config_rows(cfg), max_batch_size);
    return rows >= 2 ? rows : 0;
}

void mtp_verify_depth_follows_k(RuntimeConfig& cfg) {
    if (cfg.speculative.mtp_k <= 0)
        return;
    if (!cfg.was_set("speculative.mtp_batch_verify_k"))
        cfg.speculative.mtp_batch_verify_k = std::min(cfg.speculative.mtp_batch_verify_k,
                                                      cfg.speculative.mtp_k);
    // mtp_k caps what a request keeps of a pass: a deeper streak would draft rows nobody verifies.
    cfg.speculative.mtp_batch_verify_k_streak = std::min(cfg.speculative.mtp_batch_verify_k_streak,
                                                         cfg.speculative.mtp_k);
}

int mtp_bound_verify_top_depth(const RuntimeConfig& cfg) {
    const bool factor_row = mtp_bound_verify_factor_row(cfg);
    const int depth = mtp_bound_verify_depth(cfg.speculative.mtp_batch_verify_k, factor_row);
    return std::max(mtp_bound_verify_streak_depth(cfg.speculative.mtp_batch_verify_k_streak, depth,
                                                  factor_row),
                    mtp_bound_verify_pin_depth(cfg.diagnostics.mtp_verify_pin, depth, factor_row));
}

bool batch_verify_wants_bufs(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    return batch_verify_full(cfg, model, max_batch_size) ||
           mtp_bound_verify_rows(cfg, model, max_batch_size) > 0;
}

int mtp_draft_kv_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    return batch_verify_full(cfg, model, max_batch_size) ? max_batch_size
                                                         : std::max(1, cfg.speculative.mtp_history_slots);
}

int mtp_bind_cap(const RuntimeConfig& cfg, const Model* model, int max_batch_size, bool carry_live) {
    if (batch_verify_full(cfg, model, max_batch_size))
        return max_batch_size;
    return carry_live ? std::max(1, cfg.speculative.mtp_history_slots) : 1;
}

int mtp_carry_rows(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    return !batch_verify_full(cfg, model, max_batch_size) ? kMtpCarryRows : 0;
}

int batch_verify_spare_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    if (!batch_verify_full(cfg, model, max_batch_size)) {  // slot-swapping: no factored rows to apply
        const bool factor_row = mtp_bound_verify_factor_row(cfg);
        return mtp_bound_verify_rows(cfg, model, max_batch_size) *
               bound_verify_spares_per_row(mtp_bound_verify_depth(cfg.speculative.mtp_batch_verify_k,
                                                                  factor_row),
                                           factor_row);
    }
    // The whole point of the factored form: the drafted row is (g, k, delta)
    // plus a conv tap, so the pool stops carrying a second slot per batch slot
    // and the planner stops charging one.
    if (cfg.speculative.factored_spare)
        return 0;
    return max_batch_size;
}

bool Engine::ensure_batch_verify_bufs_(cudaStream_t stream) {
    if (!batch_verify_wants_bufs(runtime_config_, model_.get(), config_.max_batch_size))
        return true;  // nothing to stage, and that is not a failure
    if (bv_.d_stage != nullptr)
        return true;
    const bool full = batch_verify_full(runtime_config_, model_.get(), config_.max_batch_size);
    const int cap_seq = full ? config_.max_batch_size
                             : mtp_bound_verify_rows(runtime_config_, model_.get(), config_.max_batch_size);
    const int table_cap = decode_batch_pool_.is_allocated() ? decode_batch_pool_.max_blocks_per_seq() : 0;
    if (cap_seq < 2 || table_cap <= 0)
        return false;
    // Factor rows: the pool reserved one spare per row, so without its rows the step verifies 1 draft.
    if (!full && mtp_bound_verify_factor_row(runtime_config_)) {
        const int levels = mtp_bound_verify_top_depth(runtime_config_) - 1;
        bv_.factor_row = bound_verify_factor_row_alloc(model_->config(), ssm_state_.get(), vram_alloc_, bv_,
                                                       cap_seq, levels, stream);
        if (!bv_.factor_row)
            runtime_config_.speculative.mtp_batch_verify_k = 1;
    }
    const int depth = full ? 1
                           : mtp_bound_verify_depth(runtime_config_.speculative.mtp_batch_verify_k,
                                                    bv_.factor_row);
    const int streak_depth =
        full ? 1
             : mtp_bound_verify_streak_depth(runtime_config_.speculative.mtp_batch_verify_k_streak, depth,
                                             bv_.factor_row);
    const int pin_depth = full ? 0
                               : mtp_bound_verify_pin_depth(runtime_config_.diagnostics.mtp_verify_pin, depth,
                                                            bv_.factor_row);
    const int cap_rows = (std::max(streak_depth, pin_depth) + 1) * cap_seq;
    const size_t stage_ints = batch_verify_stage_ints(cap_rows, cap_seq);
    const size_t table_ints = static_cast<size_t>(cap_rows) * table_cap;
    auto pin = [](PinnedBuffer& b, size_t bytes) {
        b = PinnedBuffer::acquire(cuda_host_pinned_allocator(), bytes);
        return !b.empty();
    };
    bv_.d_stage = static_cast<int32_t*>(
        vram_alloc_.allocate(stage_ints * sizeof(int32_t), "spec_batch_stage"));
    bv_.d_row_tables = static_cast<int32_t*>(
        vram_alloc_.allocate(table_ints * sizeof(int32_t), "spec_batch_row_tables"));
    bv_.d_argmax = static_cast<int32_t*>(
        vram_alloc_.allocate(static_cast<size_t>(cap_rows) * sizeof(int32_t), "spec_batch_argmax"));
    const bool ok = bv_.d_stage != nullptr && bv_.d_row_tables != nullptr && bv_.d_argmax != nullptr &&
                    pin(bv_.h_stage, stage_ints * sizeof(int32_t)) &&
                    pin(bv_.h_row_tables, table_ints * sizeof(int32_t)) &&
                    pin(bv_.h_argmax, static_cast<size_t>(cap_rows) * sizeof(int32_t));
    if (!ok) {
        IMP_LOG_WARN("spec-batch: buffer allocation failed (%d rows x %d blocks) - batched verify off",
                     cap_rows, table_cap);
        free_batch_verify_bufs_();
        return false;
    }
    bv_.cap_seq = cap_seq;
    bv_.cap_rows = cap_rows;
    bv_.table_cap = table_cap;
    bv_.depth = depth;
    bv_.streak_depth = streak_depth;
    bv_.pin = pin_depth;
    bv_.wide_depth = mtp_bound_verify_wide_depth(runtime_config_.speculative.mtp_batch_verify_k_wide, depth,
                                                 bv_.factor_row);
    bv_.bound_only = !full;
    if (!full) {
        IMP_LOG_INFO(
            "speculative.mtp_batch_verify: bound requests verify together while <= %d decode, %d draft(s) "
            "each, %d after a full accept, %d from %d requests up "
            "(%d spare recurrent slots%s)",
            cap_seq, depth, streak_depth, bv_.wide_depth, kMtpBoundVerifyWideRows,
            batch_verify_spare_slots(runtime_config_, model_.get(), config_.max_batch_size),
            bv_.factor_row ? ", the drafts before the last also as factor rows" : "");
        return true;  // slot-swapping spares: a factored row would need every following step to apply it
    }
    return ensure_factored_rows(runtime_config_, model_->config(), ssm_state_.get(), vram_alloc_, bv_,
                                stream);
}

// speculative.factored_spare with its rows allocated; the bound verify (bv.factor_row) owns the same buffers.
bool factored_spare_active(const RuntimeConfig& cfg, const BatchVerifyState& bv) {
    return cfg.speculative.factored_spare && bv.d_fac != nullptr && !bv.factor_row;
}

void Engine::free_batch_verify_bufs_() {
    if (bv_.d_stage)
        vram_alloc_.free(bv_.d_stage);
    if (bv_.d_row_tables)
        vram_alloc_.free(bv_.d_row_tables);
    if (bv_.d_argmax)
        vram_alloc_.free(bv_.d_argmax);
    bv_.d_stage = bv_.d_row_tables = bv_.d_argmax = nullptr;
    free_factored_rows(vram_alloc_, bv_);
    bv_.h_stage = PinnedBuffer{};
    bv_.h_row_tables = PinnedBuffer{};
    bv_.h_argmax = PinnedBuffer{};
    bv_.cap_seq = bv_.cap_rows = bv_.table_cap = 0;
    bv_.depth = bv_.wide_depth = bv_.streak_depth = 1;
    bv_.streak = bv_.pin = 0;
    bv_.factor_row = false;
    bv_.fac_levels = 0;
}

// Spare slots are the reserved pool slots past the multi-candidate ones.
// Committed on first use (lazy pool), held for the request's lifetime.
int Engine::acquire_spare_slot_(int req_id, bool second) {
    auto& spare_of = second ? bv_.spare2_of : bv_.spare_of;
    auto it = spare_of.find(req_id);
    if (it != spare_of.end())
        return it->second;
    if (!ssm_state_)
        return -1;
    if (!bv_.initialized) {
        bv_.free.clear();
        const int mc = spec_mc_reserved_slots_();
        for (int i = ssm_state_->n_reserved() - 1; i >= mc; --i)
            bv_.free.push_back(ssm_state_->reserved_slot(i));
        bv_.initialized = true;
    }
    if (bv_.free.empty())
        return -1;
    const int slot = bv_.free.back();
    if (!ssm_state_->ensure_slot(slot))
        return -1;  // stays free; the step falls back to plain decode
    bv_.free.pop_back();
    spare_of[req_id] = slot;
    return slot;
}

// A verify left the state after 1 + matched rows in `slot` (role from bound_verify_state_slot, 0 = the live
// slot already): that slot becomes the live one and the old live slot takes its place among the spares.
void Engine::bound_verify_adopt_slot_(int req_id, int role, int live, int slot) {
    if (role == 0)
        return;
    recurrent_slot_of_[req_id] = slot;
    (role == 1 ? bv_.spare_of : bv_.spare2_of)[req_id] = live;
    if (async_parked_req_id_ == req_id) {  // its parked graph loop baked the old live slot
        async_graph_runner_.cleanup();
        release_async_block_tables_();
        async_parked_req_id_ = -1;
    }
}

// Up to depth draft tokens for the request's next positions: the pending
// MTP chain (bound request), else the suffix / n-gram draft. 0 = no draft
// (the request's draft rows are pads that are never accepted).
int Engine::batch_verify_draft_(Request& req, int depth, int32_t* out, bool& from_mtp) {
    from_mtp = false;
    auto take = [&](const std::vector<int32_t>& d) {
        const int n = std::min(depth, static_cast<int>(d.size()));
        std::copy_n(d.begin(), n, out);
        return n;
    };
    // The request's suffix index with every output token appended.
    auto synced_index = [&]() -> SuffixDraftIndex& {
        const int pred_end = static_cast<int>(req.input_tokens.size() + req.prediction_tokens.size());
        SuffixDraftIndex& idx = spec_suffix_index_(req);
        const int out_indexed = std::clamp(idx.size() - pred_end, 0,
                                           static_cast<int>(req.output_tokens.size()));
        idx.append(std::span(req.output_tokens).subspan(static_cast<size_t>(out_indexed)));
        return idx;
    };
    if (runtime_config_.diagnostics.spec_trace && runtime_config_.speculative.suffix) {
        int match = 0;
        NgramDraft sd = synced_index().draft(kSuffixShadowDrafts, kSuffixShadowDrafts, &match);
        bv_.sfx_shadow.trace_draft(req.id, req.output_tokens, std::move(sd.tokens), match);
    }
    if (mtp_spec_decode_enabled()) {
        auto d = mtp_take_draft_(req);
        // Plain batched steps since its last draft (speculative.mtp_batch_verify): the carried rows catch
        // the head up to the context end and draft the next tokens (the request's own mtp_k bounds the
        // chain).
        if (d.empty() && bv_.bound_only && mtp_bound(mtp_active_, mtp_pool_, req.id) &&
            mtp_active_.carry_n > 0) {
            mtp_carry_flush_(req, /*draft_k=*/std::clamp(mtp_request_k_(req), 1, depth));
            d = mtp_take_draft_(req);
        }
        if (!d.empty()) {
            from_mtp = true;
            return take(d);
        }
    }
    if (!spec_ngram_enabled_(req) || req.spec_ngram_given_up)
        return 0;
    const auto& scfg = runtime_config_.speculative;
    NgramDraft nd;
    if (scfg.suffix) {
        nd = synced_index().draft(depth, std::max(depth, cfg::kSpecSuffixKMax));
    } else {
        std::vector<int32_t> history;
        history.reserve(req.input_tokens.size() + req.prediction_tokens.size() + req.output_tokens.size());
        history.insert(history.end(), req.input_tokens.begin(), req.input_tokens.end());
        history.insert(history.end(), req.prediction_tokens.begin(), req.prediction_tokens.end());
        history.insert(history.end(), req.output_tokens.begin(), req.output_tokens.end());
        nd = ngram_draft(history, depth, std::max(1, scfg.min_match), scfg.max_match);
    }
    return take(nd.tokens);
}

// Which gate refuses the batched verify for this batch, or nullptr. Logged
// once per reason: a batch that never verifies must not look like one that
// does (the pipeline gates had the same blind spot, #1646).
const char* Engine::batch_verify_refusal_(const std::vector<std::shared_ptr<Request>>& batch) const {
    const bool full = batch_verify_on(runtime_config_, model_.get());
    const char* why = [&]() -> const char* {
        if (!ssm_state_)
            return "off";
        if (const char* mode = batch_mode_refusal(full, bv_, static_cast<int>(batch.size()),
                                                  bound_rows(batch, mtp_active_, mtp_pool_),
                                                  mtp_spec_decode_enabled(),
                                                  bound_solo(runtime_config_, model_.get(), bv_, mtp_active_,
                                                             mtp_pool_, batch)))
            return mode;
        // The factored form carries the drafted row as (g, k, delta) plus a
        // conv tap, so it reserves no slots by design; this gate must not
        // refuse every batch on that basis alone.
        if (!factored_spare_active(runtime_config_, bv_) && ssm_state_->n_reserved() - spec_mc_reserved_slots_() <= 0)
            return "no_spare_slots";
        if (swa_sizing_active_ || config_.streaming_kv_enabled)
            return "swa_or_streaming_kv";
        if (kv_manager_ && kv_manager_->residual_enabled())
            return "residual_kv";
        if (!runtime_config_.runtime.gdn_batched_decode || d_ssm_seq_slots_ == nullptr)
            return "gdn_batched_decode_off";
        // A think budget whose forcing is due needs the eager step (it forces
        // </think>); any such row sends the whole batch to the plain step.
        for (const auto& r : batch) {
            if (r->status != RequestStatus::DECODING)
                continue;
            if (!batch_verify_row_ok(*r))
                return "row_shaped";
            if (const char* sampled = spec_sampling_refusal_(*r))
                return sampled;
            if (think_closer_pending_(*r))
                return "think_forcing_due";
            if (r->think_budget > 0.0f && r->in_think_block &&
                think_logic::should_force_think_end(r->think_budget, think_end_id_, r->max_tokens,
                                                    r->output_tokens, think_start_id_, r->started_in_think,
                                                    runtime_config_.runtime.think_answer_reserve,
                                                    runtime_config_.runtime.think_max_tokens))
                return "think_forcing_due";
        }
        return nullptr;
    }();
    if (why != nullptr && batch.size() >= 2 && bv_.last_refusal != why && refusal_logged(full, bv_, why)) {
        bv_.last_refusal = why;
        IMP_LOG_INFO("spec-batch: batched verify refused by '%s' (n=%zu)", why, batch.size());
    }
    return why;
}

// A pipelined plain chain never reaches the per-step verify entries: #1003 yields it periodically for the
// round-robin spec verify (dense models), speculative.mtp_batch_verify when the bound rows could verify
// together (every yield_every-th continuation).
bool Engine::pipeline_yields_(const std::vector<std::shared_ptr<Request>>& rows) {
    if (bv_.bound_only && ++bv_.since_yield >= bv_.yield_every) {
        bv_.since_yield = 0;
        if (batch_verify_refusal_(rows) == nullptr)
            return true;
    }
    if (ssm_state_ || ++bd_pipe_.steps_since_spec_yield < spec_rr_yield_interval_)
        return false;
    bd_pipe_.steps_since_spec_yield = 0;
    return std::any_of(rows.begin(), rows.end(), [this](const auto& r) {
        return r->status == RequestStatus::DECODING && spec_any_drafter_enabled_(*r) &&
               !r->spec_ngram_given_up;
    });
}

bool Engine::step_spec_verify_batched_(std::vector<std::shared_ptr<Request>>& batch, cudaStream_t stream) {
    const auto verify_t0 = std::chrono::steady_clock::now();
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    if (!ensure_batch_verify_bufs_(stream))
        return false;
    // The resolved batch: runtime.max_batch_size 0 = auto (a default server) took no row at all.
    const int max_bs = std::min(config_.max_batch_size, bv_.cap_seq);
    std::vector<std::shared_ptr<Request>> reqs;
    for (auto& r : batch) {
        if (static_cast<int>(reqs.size()) >= max_bs)
            break;
        if (r->status != RequestStatus::DECODING || r->output_tokens.empty())
            continue;
        reqs.push_back(r);
    }
    const int N = static_cast<int>(reqs.size());
    auto decline = [&](const char* why) {
        bv_.yield_every = std::min(64, 2 * bv_.yield_every);
        if (bv_.last_refusal != why) {
            bv_.last_refusal = why;
            IMP_LOG_INFO("spec-batch: step declined by '%s' (n=%d)", why, N);
        }
        return false;
    };
    const bool solo = bound_solo(runtime_config_, model_.get(), bv_, mtp_active_, mtp_pool_, reqs);
    if (N + static_cast<int>(solo) < 2)
        return decline("fewer_than_two_decoding");

    // Slots: the live one must exist (acquired at prefill), the spare is
    // taken now and kept for the request's lifetime.
    // Rows per request: the last token plus up to D drafts (D >= 2 only on the bound verify).
    const int D = bound_verify_stage_depth(bv_, N);
    const int T = D + 1;
    std::vector<int> live(N), spare(N), spare2(N, -1);
    const bool second_spare = bound_verify_spares_per_row(D, bv_.factor_row) > 1;
    for (int g = 0; g < N; ++g) {
        live[g] = recurrent_slot_for_(reqs[g]->id);
        if (live[g] < 0)
            return decline("no_live_slot");
        // Factored: no spare slot exists, the drafted row leaves as factors.
        spare[g] = factored_spare_active(runtime_config_, bv_) ? live[g] : acquire_spare_slot_(reqs[g]->id);
        if (spare[g] < 0)
            return decline("spare_slot_unavailable");
        // Factor row: no second spare, h_snap[N + g] is not read (ssm_snap_rows 1).
        spare2[g] = second_spare ? acquire_spare_slot_(reqs[g]->id, /*second=*/true) : spare[g];
        if (spare2[g] < 0)
            return decline("spare_slot_unavailable");
    }

    // Drafts. A batch without a single draft is a plain decode step.
    std::vector<int32_t> draft(static_cast<size_t>(N) * D, -1);
    std::vector<int> n_draft(N, 0);
    std::vector<uint8_t> from_mtp(N, 0);
    int n_drafted = 0;
    for (int g = 0; g < N; ++g) {
        bool mtp = false;
        n_draft[g] = batch_verify_draft_(*reqs[g], D, draft.data() + (static_cast<size_t>(g) * D), mtp);
        from_mtp[g] = mtp ? 1 : 0;
        n_drafted += static_cast<int>(n_draft[g] > 0);
    }
    if (const char* why = draft_refusal(bv_, N, n_drafted, solo))
        return decline(why);

    // Per-row penalties use the sampler's device history slots (host
    // output_tokens is the truth). Row T*g + j penalizes the history (ends
    // with t0) plus drafts 0..j-1 (written at the slot's end below); emitted
    // tokens land there afterward, keeping the slot in sync.
    std::vector<const int32_t*> pen_hist(static_cast<size_t>(T) * N, nullptr);
    std::vector<int> pen_n(static_cast<size_t>(T) * N, 0), pen_slot(N, -1), pen_need(N, 0);
    std::vector<float> pen_v(static_cast<size_t>(3 * T) * N, 0.0f);
    bool any_pen = false;
    for (int g = 0; g < N; ++g) {
        const Request& req = *reqs[g];
        const bool needs = req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                           req.presence_penalty != 0.0f;
        if (!needs)
            continue;
        const int need = static_cast<int>(req.output_tokens.size());
        if (d_penalty_hist_ == nullptr || penalty_hist_slots_ <= 0 || need + T > penalty_hist_cap_)
            return decline("penalty_history_unavailable");
        const int slot = penalty_hist_slot_(req.id, reqs);
        if (slot < 0)
            return decline("penalty_history_unavailable");
        auto& hs = penalty_hist_state_[static_cast<size_t>(slot)];
        int32_t* base = d_penalty_hist_ + (static_cast<size_t>(slot) * penalty_hist_cap_);
        if (hs.synced != need) {
            // Full (re)sync: first batch entry of this request or a diverged
            // host history (pageable source, the same copy the plain step does).
            if (cudaMemcpyAsync(base, req.output_tokens.data(), static_cast<size_t>(need) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream) != cudaSuccess)
                return decline("penalty_history_h2d");
            hs.synced = need;
        }
        pen_slot[g] = slot;
        pen_need[g] = need;
        for (int j = 0; j < T; ++j) {
            const size_t r = (static_cast<size_t>(T) * g) + j;
            pen_hist[r] = base;
            pen_n[r] = need + j;
            pen_v[(r * 3) + 0] = req.repetition_penalty;
            pen_v[(r * 3) + 1] = req.frequency_penalty;
            pen_v[(r * 3) + 2] = req.presence_penalty;
        }
        any_pen = true;
    }

    // KV blocks for every row of every request (positions p0 .. p0 + D).
    std::vector<int> p0(N);
    int max_ctx = 0;
    auto rollback_all = [&](int upto) {
        for (int h = 0; h < upto; ++h)
            kv_manager_->rollback(reqs[h]->id, p0[h]);
    };
    for (int g = 0; g < N; ++g) {
        p0[g] = reqs[g]->context_len() - 1;
        const int blocks_needed = (p0[g] + T + kv_bs - 1) / kv_bs;
        while (static_cast<int>(kv_manager_->block_table(reqs[g]->id).size()) < blocks_needed) {
            int nb = kv_manager_->append_block(reqs[g]->id);
            if (nb < 0 && kv_cache_raw_ != nullptr) {
                const int have = kv_cache_raw_->total_blocks();
                if (kv_cache_raw_->ceiling_blocks() > have) {
                    kv_cache_raw_->try_grow_to_now(have + std::max(64, have / 4));
                    nb = kv_manager_->append_block(reqs[g]->id);
                }
            }
            if (nb < 0) {
                rollback_all(g + 1);
                return false;
            }
        }
        if (static_cast<int>(kv_manager_->block_table(reqs[g]->id).size()) > bv_.table_cap) {
            rollback_all(g + 1);
            return false;
        }
        max_ctx = std::max(max_ctx, p0[g] + T);
    }

    // Stage: [tokens | positions | row ctx lens | seq slots | out slots |
    // snap slots x2 | chunk_len, snap_n x2], one H2D; row tables, one H2D.
    // Second snapshot (D == 2, no factor rows): row 2's state into spare2, indexed N + g (the scan's n_seq).
    const int cap_rows = bv_.cap_rows, cap_seq = bv_.cap_seq, table_cap = bv_.table_cap;
    int32_t* h = bv_.h_stage.as<int32_t>();
    int32_t* h_tok = h;
    int32_t* h_pos = h + cap_rows;
    int32_t* h_ctx = h + (2ull * cap_rows);
    int32_t* h_seq = h + (3ull * cap_rows);
    int32_t* h_out = h_seq + cap_seq;
    int32_t* h_snap = h_out + cap_seq;
    int32_t* h_misc = h_snap + (2ull * cap_seq);
    int32_t* h_tab = bv_.h_row_tables.as<int32_t>();
    for (int g = 0; g < N; ++g) {
        stage_group_rows(h_tok, h_pos, h_ctx, T, g, reqs[g]->output_tokens.back(),
                         draft.data() + (static_cast<size_t>(g) * D), n_draft[g], p0[g]);
        h_seq[g] = live[g];
        h_out[g] = spare[g];
        h_snap[g] = live[g];
        h_snap[N + g] = spare2[g];  // read only with a second spare
        stage_group_tables(h_tab, table_cap, T, g, kv_manager_->block_table(reqs[g]->id));
    }
    h_misc[0] = T;  // real rows per group
    h_misc[1] = 1;  // snapshot row (live slot, in place)
    h_misc[2] = 2;  // second snapshot row (spare2)
    int32_t* d = bv_.d_stage;
    int32_t* d_tok = d;
    int32_t* d_pos = d + cap_rows;
    int32_t* d_ctx = d + (2ull * cap_rows);
    int32_t* d_seq = d + (3ull * cap_rows);
    int32_t* d_out = d_seq + cap_seq;
    int32_t* d_snap = d_out + cap_seq;
    int32_t* d_misc = d_snap + (2ull * cap_seq);
    auto check = [&](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("spec-batch: %s failed: %s", op, cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    const size_t stage_ints = batch_verify_stage_ints(cap_rows, cap_seq);
    if (!check(cudaMemcpyAsync(d, h, stage_ints * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
               "stage H2D") ||
        !check(cudaMemcpyAsync(bv_.d_row_tables, h_tab,
                               static_cast<size_t>(T) * N * table_cap * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream),
               "row tables H2D")) {
        rollback_all(N);
        return false;
    }
    // The draft tokens behind each penalized request's history (pinned
    // source: the staged chunk tokens).
    for (int g = 0; g < N && any_pen; ++g) {
        if (pen_slot[g] < 0)
            continue;
        int32_t* base = d_penalty_hist_ + (static_cast<size_t>(pen_slot[g]) * penalty_hist_cap_);
        if (!check(cudaMemcpyAsync(base + pen_need[g], h_tok + (static_cast<size_t>(T) * g) + 1,
                                   static_cast<size_t>(D) * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
                   "penalty draft H2D")) {
            rollback_all(N);
            return false;
        }
    }

    // Workspace pinned to the largest batch: every N's graph then bakes the
    // same activation carve, and growing N never reallocates (which would
    // drop every cached graph).
    if (!executor_->resize_workspace(bv_.cap_rows, stream)) {
        rollback_all(N);
        return false;
    }
    if (executor_->has_decode_workspace())
        executor_->use_workspace(0);
    // Graph replay (engine_spec_capture.cpp): one graph per (rows, ctx tier).
    // Everything that varies per step is device data the H2D above refreshed;
    // the split-K grid bakes the tier ceiling.
    const bool capture_on = spec_capture_ready_(max_ctx, T * N, /*paged_attn=*/true);

    InferenceState state;
    state.token_ids = d_tok;
    state.positions = d_pos;
    state.n_tokens = T * N;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = bv_.d_row_tables;
    state.context_lens = d_ctx;
    state.max_context_len = max_ctx;
    state.n_sequences = T * N;
    state.max_blocks_per_seq = table_cap;
    state.is_prefill = true;
    state.prefill_offset = 0;
    state.spec_verify_chunk = true;
    state.skip_lm_head = true;  // greedy_argmax_all below projects every row
    state.chunk_decode_attn = true;
    state.kv_manager = kv_manager_.get();
    state.ssm_state = ssm_state_.get();
    state.ssm_seq_id = live[0];
    state.ssm_seq_slots = d_seq;
    state.ssm_n_seq = N;
    state.ssm_group_verify = true;
    state.ssm_seq_tokens = T;
    // Factored spare: no second slot, so no commit destination. The drafted
    // row leaves as (g, k, delta) plus one conv tap per layer, both addressed
    // by the live slot; the snapshot at row 1 stays the in-place one.
    const bool factored = factored_spare_active(runtime_config_, bv_);
    state.ssm_out_slots = factored ? nullptr : d_out;
    state.ssm_snap_slots = factored ? d_seq : d_snap;
    state.ssm_snap_rows = factored ? 1 : bound_verify_spares_per_row(D, bv_.factor_row);
    // Factor rows: 1 row in place, all T rows into the spare, rows 2 .. D as factor rows + taps by group
    // and level. d_fac is null unless factored or factor_row.
    state.ssm_fac_out = bv_.d_fac;
    state.ssm_fac_stride = bv_.fac_stride;
    state.ssm_fac_layer_stride = bv_.fac_layer_stride;
    state.ssm_tap_out = bv_.d_tap;
    state.ssm_tap_layer_stride = bv_.tap_layer_stride;
    state.ssm_tap_levels = bv_.factor_row ? D - 1 : 0;
    if (factored) {
        // Rows the LAST verify accepted: this forward applies them, to the
        // conv window before the conv reads it and to the recurrent state
        // inside the scan's register load.
        if (!bv_.pending.empty()) {
            const int np = static_cast<int>(bv_.pending.size());
            auto* hp = static_cast<int32_t*>(bv_.h_pending.data());
            std::copy(bv_.pending.begin(), bv_.pending.end(), hp);
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(bv_.d_pending, hp, np * sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream));
            state.ssm_fac_in = bv_.d_fac;
            state.ssm_tap_in = bv_.d_tap;
            state.ssm_tap_slots = bv_.d_pending;
            state.ssm_tap_n = np;
        }
        bv_.pending.clear();
    }
    state.d_chunk_len = d_misc;
    state.d_snap_n = d_misc + 1;
    if (capture_on) {
        state.ctx_capacity = spec_capture_ctx_tier_(max_ctx, /*paged_attn=*/true);
        state.max_context_len = state.ctx_capacity;
    }

    Tensor logits_out;
    bool replayed = false;
    if (capture_on && !spec_capture_doomed_)
        replayed = spec_captured_forward_(state, logits_out, stream);
    if (!replayed) {
        if (spec_capture_doomed_) {
            state.ctx_capacity = 0;
            state.max_context_len = max_ctx;
        }
        executor_->forward_logits(state, logits_out, stream);
    }

    // Argmax over every row. Rows of a request inside a budgeted think block
    // take the stop mask, the others the plain banned list.
    const int n_rows = T * N;
    std::vector<uint8_t> row_alt(static_cast<size_t>(n_rows), 0);
    bool any_alt = false;
    for (int g = 0; g < N; ++g) {
        if (stop_mask_active(*reqs[g], think_end_id_)) {
            std::fill_n(row_alt.begin() + (static_cast<ptrdiff_t>(T) * g), T, 1);
            any_alt = true;
        }
    }
    const int32_t* d_ban = banned_tokens_device_(stream);
    const int n_ban = d_ban ? static_cast<int>(banned_token_ids_.size()) : 0;
    const int32_t* d_alt = (any_alt && stop_mask_.n() > 0) ? stop_mask_.device(vram_alloc_, stream) : nullptr;
    const int n_alt = d_alt ? stop_mask_.n() : 0;
    const std::vector<VerifySampling> row_draw =
        engine_internal::verify_row_draws(reqs, T, model_->config().vocab_size);
    executor_->set_verify_sampling_rows(row_draw.data(), static_cast<int>(row_draw.size()));
    executor_->greedy_argmax_all(n_rows, bv_.d_argmax, stream, nullptr, 0, nullptr, 1.0f, 0.0f, 0.0f, nullptr,
                                 0, d_ban, n_ban, /*allow_cutlass=*/true, d_alt, n_alt,
                                 any_alt ? row_alt.data() : nullptr, any_pen ? pen_hist.data() : nullptr,
                                 any_pen ? pen_n.data() : nullptr, any_pen ? pen_v.data() : nullptr);
    executor_->set_verify_sampling(nullptr);
    int32_t* h_am = bv_.h_argmax.as<int32_t>();
    if (!check(cudaMemcpyAsync(h_am, bv_.d_argmax, static_cast<size_t>(n_rows) * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream),
               "argmax D2H") ||
        !check(cudaStreamSynchronize(stream), "verify sync")) {
        // The forward ran: every live slot already holds the post-t0 state and
        // no token can be emitted for it. A failed D2H on this stream is a
        // dead context; say so rather than pretend the step never happened.
        IMP_LOG_ERROR("spec-batch: verify step lost after the forward; recurrent state and tokens diverge");
        rollback_all(N);
        return false;
    }

    long long acc_total = 0, emit_total = 0, drafted_total = 0;
    bool any_mtp = false;
    int full_nd = 0;
    std::vector<int> row0(N), emitted_of(N, 0), fold_of(N, 0), matched_of(N, 0), id_of(N, 0);
    for (int g = 0; g < N; ++g) {
        auto& req = reqs[g];
        const int nd = n_draft[g];
        const int32_t* dr = draft.data() + (static_cast<size_t>(g) * D);
        const auto emit = [&](int, int32_t tok) {
            req->output_tokens.push_back(tok);
            track_think_state(*req, tok);
            const bool hard_stop = should_stop(*req, tok) ||
                                   static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (req->constraints)
                req->constraints->update(tok);
            if (hard_stop)
                finish_request(req);
            return hard_stop;
        };
        const auto [matched, emitted] = spec_accept_rows(h_am + (static_cast<size_t>(T) * g), dr, nd, nd + 1,
                                                         req->think_budget > 0.0f, req->in_think_block,
                                                         think_tokens_left_(*req), emit,
                                                         [](int) { return true; });
        row0[g] = T * g;
        emitted_of[g] = emitted;
        matched_of[g] = matched;
        id_of[g] = req->id;
        // The binding's economics tally (judged at its lone verifies); a finished request has none.
        if (from_mtp[g] && nd > 0)
            if (MtpBind* b = mtp_find_bind(mtp_active_, mtp_pool_, req->id))
                mtp_econ_count(*b, emitted, nd, /*batched=*/true);
        // Device penalty history: the emitted tokens are rows T*g.. of the
        // pinned argmax buffer, in order.
        if (pen_slot[g] >= 0 && emitted > 0) {
            int32_t* base = d_penalty_hist_ + (static_cast<size_t>(pen_slot[g]) * penalty_hist_cap_);
            auto& hs = penalty_hist_state_[static_cast<size_t>(pen_slot[g])];
            if (cudaMemcpyAsync(base + pen_need[g], h_am + (static_cast<size_t>(T) * g),
                                static_cast<size_t>(emitted) * sizeof(int32_t), cudaMemcpyHostToDevice,
                                stream) == cudaSuccess)
                hs.synced = pen_need[g] + emitted;
            else
                hs.synced = -1;
        }
        kv_manager_->touch(req->id);
        kv_manager_->rollback(req->id, p0[g] + 1 + matched);
        // State after 1 + matched rows: the live slot holds row 1 (in-place snapshot), the spare all T rows,
        // spare2 row 2 (D == 2); taking one swaps it with the live slot. Factored (D == 1): no spare, the
        // scan wrote a row for EVERY group, so accept keeps it for the next step and anything else must
        // clear it (else state advances by a token nobody accepted).
        // Factor rows (D >= 2): no spare2, 1 .. D - 1 accepted drafts fold their rows into the live slot.
        const bool done = req->status == RequestStatus::FINISHED;
        const int role = done ? 0 : bound_verify_kept_slot(matched, D, bv_.factor_row);
        fold_of[g] = bound_verify_factor_row_folds(bv_.factor_row, done, matched, D);
        if (mtp_bound_verify_full_accept(nd, matched, done))
            full_nd = std::max(full_nd, nd);
        if (runtime_config_.diagnostics.spec_trace)
            bv_.sfx_shadow.trace_step(req->id, N, T, nd, matched, emitted, req->output_tokens, done);
        if (factored)
            (role != 0 ? bv_.pending : bv_.clear_slots).push_back(live[g]);
        else
            bound_verify_adopt_slot_(req->id, role, live[g], role == 1 ? spare[g] : spare2[g]);
        if (nd > 0) {
            req->spec_verifies++;
            req->spec_drafted += nd;
            req->spec_accepted += matched;
            req->spec_emitted += emitted;
            req->spec_consecutive_misses = matched == 0 ? req->spec_consecutive_misses + 1 : 0;
        }
        acc_total += matched;
        emit_total += emitted;
        drafted_total += nd;
        any_mtp = any_mtp || from_mtp[g];
    }
    // Factored: rows nobody may apply - this step's rejects, plus slots
    // released since the last verify - get the 0 sentinel before the next
    // forward can read them.
    if (factored && !bv_.clear_slots.empty()) {
        const int nc = static_cast<int>(bv_.clear_slots.size());
        auto* hc = static_cast<int32_t*>(bv_.h_clear.data());
        std::copy(bv_.clear_slots.begin(), bv_.clear_slots.end(), hc);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(bv_.d_clear, hc, nc * sizeof(int32_t), cudaMemcpyHostToDevice,
                                           stream));
        const auto& mc = model_->config();
        const int heads = mc.ssm_dt_rank;
        gdn_factor_clear(bv_.d_fac, bv_.fac_layer_stride, ssm_state_->n_ssm_layers(), bv_.d_clear, nc,
                         heads, bv_.fac_stride, stream);
        bv_.clear_slots.clear();
    }
    bound_verify_factor_row_fold(model_->config(), *ssm_state_, bv_, D - 1, live.data(), fold_of.data(), N,
                                 stream);

    // MTP: feed every bound request's emitted pairs and draft its next token
    // in one head pass (the chunk's hidden rows are still in place).
    const auto feed_t0 = std::chrono::steady_clock::now();
    bv_.streak = mtp_bound_verify_streak_next(full_nd, bv_.streak_depth);
    if (mtp_spec_decode_enabled())
        mtp_batched_feed_(reqs, row0, emitted_of, stream);
    const double fwd_ms = std::chrono::duration<double, std::milli>(feed_t0 - verify_t0).count();
    const double head_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feed_t0).count();
    log_step_price(N, T, fwd_ms, head_ms, emit_total);
    spec_trace_emit_batch_step(runtime_config_.diagnostics.spec_trace, T, fwd_ms, head_ms, id_of, n_draft,
                               matched_of, emitted_of);
    spec_stats_record_(
        any_mtp, drafted_total, acc_total, emit_total,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - verify_t0).count(),
        /*sampled=*/!row_draw.empty());
    static bool logged = false;
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("spec-batch: first batched verify n=%d rows=%d drafted=%lld accepted=%lld emitted=%lld",
                     N, n_rows, drafted_total, acc_total, emit_total);
    }
    bv_.yield_every = 1;
    return true;
}

}  // namespace imp
