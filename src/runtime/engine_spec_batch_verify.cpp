// engine_spec_batch_verify.cpp: batched speculative verify on the GDN hybrid.
//
// One forward per decode step for N requests, 2 rows each: the request's last
// emitted token and its draft. Group g runs the GDN layers on its live slot
// L_g, commits the 2-row state into the request's spare slot P_g and the
// 1-row state in place into L_g (compute/gdn.h out_slots / snap_slots).
// Accept swaps L_g and P_g on the host; reject keeps L_g. No state copies, no
// replay: verify depth is 1 (docs/plans/2026-09-11-batched-mtp-verify.md).
// Attention rows take the decode-attention route with per-row block tables
// and context lengths, the same shape as the multi-candidate chunk.

#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/engine.h"
#include "compute/gdn_factor.cuh"
#include "runtime/ngram_draft.h"
#include "runtime/request.h"
#include "runtime/stop_mask.h"
#include "runtime/suffix_draft.h"
#include "runtime/think_stop_logic.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <span>

namespace imp {

namespace {
// Per-row conditions the batched verify replicates: greedy argmax, rep /
// freq / presence penalties over the unbounded window (per-row device
// histories), no logit shaping, no constraint FSM, no logprobs.
bool batch_verify_row_ok(const Request& r) {
    const bool greedy = r.temperature <= 0.0f || r.top_k == 1;
    if (!greedy)
        return false;
    const bool penalties = r.repetition_penalty != 1.0f || r.frequency_penalty != 0.0f ||
                           r.presence_penalty != 0.0f;
    if (penalties && r.repeat_last_n != 0)
        return false;  // a bounded window slides per chunk row; not replicated
    if (r.dry_multiplier != 0.0f || r.mirostat != 0 || !r.logit_bias.empty())
        return false;
    if (r.logprobs || r.json_mode || !r.json_schema.empty() || !r.regex_pattern.empty() ||
        !r.grammar.empty() || !r.tool_constraint_tools.empty())
        return false;
    return r.status == RequestStatus::DECODING && !r.output_tokens.empty();
}
}  // namespace

bool batch_verify_on(const RuntimeConfig& cfg, const Model* model) {
    const auto& scfg = cfg.speculative;
    if (!scfg.batch_verify || !scfg.hybrid)
        return false;
    return model != nullptr && model->config().ssm_inner_size > 0;
}

bool batch_verify_wants_bufs(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    return batch_verify_on(cfg, model) && max_batch_size > 1;
}

int mtp_draft_kv_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    return batch_verify_wants_bufs(cfg, model, max_batch_size) ? max_batch_size : 1;
}

int batch_verify_spare_slots(const RuntimeConfig& cfg, const Model* model, int max_batch_size) {
    if (!batch_verify_on(cfg, model) || max_batch_size <= 1)
        return 0;
    // The whole point of the factored form: the drafted row is (g, k, delta)
    // plus a conv tap, so the pool stops carrying a second slot per batch slot
    // and the planner stops charging one.
    if (cfg.speculative.factored_spare)
        return 0;
    return max_batch_size;
}

bool ensure_factored_rows(RuntimeConfig& cfg, const ModelConfig& mc, SSMState* ssm, VRAMAllocator& alloc,
                          BatchVerifyState& bv);
void free_factored_rows(VRAMAllocator& alloc, BatchVerifyState& bv);

bool Engine::ensure_batch_verify_bufs_() {
    if (!batch_verify_wants_bufs(runtime_config_, model_.get(), config_.max_batch_size))
        return true;  // nothing to stage, and that is not a failure
    if (bv_.d_stage != nullptr)
        return true;
    const int cap_seq = config_.max_batch_size;
    const int table_cap = decode_batch_pool_.is_allocated() ? decode_batch_pool_.max_blocks_per_seq() : 0;
    if (cap_seq < 2 || table_cap <= 0)
        return false;
    const int cap_rows = 2 * cap_seq;
    const size_t stage_ints = 3ull * cap_rows + 3ull * cap_seq + 2;
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
    return ensure_factored_rows(runtime_config_, model_->config(), ssm_state_.get(), vram_alloc_, bv_);
}

// Factored spare pools (speculative.factored_spare), sized over the LIVE
// slots only: rows are addressed by the same slot ids the scheduler hands
// out. A failure here falls back to slot-swapping rather than failing the
// verify (the two are interchangeable by construction).
bool factored_spare_active(const RuntimeConfig& cfg, const BatchVerifyState& bv) {
    return cfg.speculative.factored_spare && bv.d_fac != nullptr;
}

bool ensure_factored_rows(RuntimeConfig& cfg, const ModelConfig& mc, SSMState* ssm,
                          VRAMAllocator& alloc, BatchVerifyState& bv) {
    if (!cfg.speculative.factored_spare || bv.d_fac != nullptr)
        return true;
    const int layers = ssm ? ssm->n_ssm_layers() : 0;
    const int slots = ssm ? ssm->max_sequences() : 0;
    const int heads = mc.ssm_dt_rank;
    const int hd = heads > 0 ? mc.ssm_inner_size / heads : 0;
    const int ss = mc.ssm_state_size;
    const int ch = mc.ssm_conv_channels();
    if (layers <= 0 || slots <= 0 || heads <= 0 || hd <= 0 || ss <= 0 || ch <= 0) {
        IMP_LOG_WARN("speculative.factored_spare ignored: no GDN state geometry");
        cfg.speculative.factored_spare = false;
        return true;
    }
    bv.fac_stride = gdn_factor_floats_per_head(ss, hd);
    bv.fac_layer_stride = static_cast<int64_t>(slots) * heads * bv.fac_stride;
    bv.tap_layer_stride = static_cast<int64_t>(slots) * ch;
    const size_t fac_bytes = static_cast<size_t>(layers) * bv.fac_layer_stride * sizeof(float);
    const size_t tap_bytes = static_cast<size_t>(layers) * bv.tap_layer_stride * sizeof(uint16_t);
    bv.d_fac = static_cast<float*>(alloc.allocate(fac_bytes, "spec_factored_rows"));
    bv.d_tap = alloc.allocate(tap_bytes, "spec_factored_taps");
    bv.d_pending = static_cast<int*>(alloc.allocate(static_cast<size_t>(slots) * sizeof(int),
                                                           "spec_factored_pending"));
    bv.d_clear = static_cast<int*>(alloc.allocate(static_cast<size_t>(slots) * sizeof(int),
                                                         "spec_factored_clear"));
    if (bv.d_fac == nullptr || bv.d_tap == nullptr || bv.d_pending == nullptr || bv.d_clear == nullptr) {
        IMP_LOG_WARN("speculative.factored_spare ignored: %.1f MiB of factored rows would not fit",
                     (fac_bytes + tap_bytes) / (1024.0 * 1024.0));
        free_factored_rows(alloc, bv);
        cfg.speculative.factored_spare = false;
        return true;
    }
    bv.h_pending = PinnedBuffer::acquire(cuda_host_pinned_allocator(), slots * sizeof(int32_t));
    bv.h_clear = PinnedBuffer::acquire(cuda_host_pinned_allocator(), slots * sizeof(int32_t));
    if (bv.h_pending.empty() || bv.h_clear.empty()) {
        IMP_LOG_WARN("speculative.factored_spare ignored: pinned slot staging unavailable");
        free_factored_rows(alloc, bv);
        cfg.speculative.factored_spare = false;
        return true;
    }
    IMP_CUDA_CHECK_LOG(cudaMemset(bv.d_fac, 0, fac_bytes));
    IMP_LOG_INFO("speculative.factored_spare: %d layers x %d slots, %.1f MiB of rows + %.1f MiB of conv "
                 "taps (a spare slot pool would be %.1f MiB)",
                 layers, slots, fac_bytes / (1024.0 * 1024.0), tap_bytes / (1024.0 * 1024.0),
                 static_cast<double>(ssm->h_bytes()) * layers * slots / (1024.0 * 1024.0));
    return true;
}

void free_factored_rows(VRAMAllocator& alloc, BatchVerifyState& bv) {
    if (bv.d_fac)
        alloc.free(bv.d_fac);
    if (bv.d_tap)
        alloc.free(bv.d_tap);
    if (bv.d_pending)
        alloc.free(bv.d_pending);
    if (bv.d_clear)
        alloc.free(bv.d_clear);
    bv.d_fac = nullptr;
    bv.d_tap = nullptr;
    bv.d_pending = nullptr;
    bv.d_clear = nullptr;
    bv.pending.clear();
    bv.clear_slots.clear();
    bv.h_pending = PinnedBuffer{};
    bv.h_clear = PinnedBuffer{};
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
}

// Spare slots are the reserved pool slots past the multi-candidate ones.
// Committed on first use (lazy pool), held for the request's lifetime.
int Engine::acquire_spare_slot_(int req_id) {
    auto it = bv_.spare_of.find(req_id);
    if (it != bv_.spare_of.end())
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
    bv_.spare_of[req_id] = slot;
    return slot;
}

// One draft token for the request's next position: the pending MTP chain
// (bound request), else the first token of the suffix / n-gram draft. -1 = no
// draft (the request's second row is a pad that is never accepted).
int32_t Engine::batch_verify_draft_(Request& req, bool& from_mtp) {
    from_mtp = false;
    if (mtp_spec_decode_enabled()) {
        auto d = mtp_take_draft_(req);
        if (!d.empty()) {
            from_mtp = true;
            return d[0];
        }
    }
    if (!spec_ngram_enabled_(req) || req.spec_ngram_given_up)
        return -1;
    const auto& scfg = runtime_config_.speculative;
    NgramDraft nd;
    if (scfg.suffix) {
        const int pred_end = static_cast<int>(req.input_tokens.size() + req.prediction_tokens.size());
        SuffixDraftIndex& idx = spec_suffix_index_(req);
        const int out_indexed = std::clamp(idx.size() - pred_end, 0,
                                           static_cast<int>(req.output_tokens.size()));
        idx.append(std::span(req.output_tokens).subspan(static_cast<size_t>(out_indexed)));
        nd = idx.draft(1, std::max(1, scfg.suffix_k_max));
    } else {
        std::vector<int32_t> history;
        history.reserve(req.input_tokens.size() + req.prediction_tokens.size() + req.output_tokens.size());
        history.insert(history.end(), req.input_tokens.begin(), req.input_tokens.end());
        history.insert(history.end(), req.prediction_tokens.begin(), req.prediction_tokens.end());
        history.insert(history.end(), req.output_tokens.begin(), req.output_tokens.end());
        nd = ngram_draft(history, 1, std::max(1, scfg.min_match), scfg.max_match);
    }
    return nd.tokens.empty() ? -1 : nd.tokens[0];
}

// Which gate refuses the batched verify for this batch, or nullptr. Logged
// once per reason: a batch that never verifies must not look like one that
// does (the pipeline gates had the same blind spot, #1646).
const char* Engine::batch_verify_refusal_(const std::vector<std::shared_ptr<Request>>& batch) const {
    const char* why = [&]() -> const char* {
        if (!ssm_state_ || !batch_verify_on(runtime_config_, model_.get()))
            return "off";
        if (batch.size() < 2)
            return "batch_of_one";
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
                return "row_not_greedy_or_shaped";
            if (r->think_budget > 0.0f && r->in_think_block &&
                think_logic::should_force_think_end(r->think_budget, think_end_id_, r->max_tokens,
                                                    r->output_tokens, think_start_id_, r->started_in_think,
                                                    runtime_config_.runtime.think_answer_reserve))
                return "think_forcing_due";
        }
        return nullptr;
    }();
    if (why != nullptr && batch.size() >= 2 && bv_.last_refusal != why &&
        batch_verify_on(runtime_config_, model_.get())) {
        bv_.last_refusal = why;
        IMP_LOG_INFO("spec-batch: batched verify refused by '%s' (n=%zu)", why, batch.size());
    }
    return why;
}

bool Engine::step_spec_verify_batched_(std::vector<std::shared_ptr<Request>>& batch, cudaStream_t stream) {
    const auto verify_t0 = std::chrono::steady_clock::now();
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    if (!ensure_batch_verify_bufs_())
        return false;
    const int max_bs = std::min(runtime_config_.runtime.max_batch_size, bv_.cap_seq);
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
        if (bv_.last_refusal != why) {
            bv_.last_refusal = why;
            IMP_LOG_INFO("spec-batch: step declined by '%s' (n=%d)", why, N);
        }
        return false;
    };
    if (N < 2)
        return decline("fewer_than_two_decoding");

    // Slots: the live one must exist (acquired at prefill), the spare is
    // taken now and kept for the request's lifetime.
    std::vector<int> live(N), spare(N);
    for (int g = 0; g < N; ++g) {
        live[g] = recurrent_slot_for_(reqs[g]->id);
        if (live[g] < 0)
            return decline("no_live_slot");
        // Factored: no spare slot exists, the drafted row leaves as factors.
        spare[g] = factored_spare_active(runtime_config_, bv_) ? live[g] : acquire_spare_slot_(reqs[g]->id);
        if (spare[g] < 0)
            return decline("spare_slot_unavailable");
    }

    // Drafts. A batch without a single draft is a plain decode step.
    std::vector<int32_t> draft(N, -1);
    std::vector<uint8_t> from_mtp(N, 0);
    int n_drafted = 0;
    for (int g = 0; g < N; ++g) {
        bool mtp = false;
        draft[g] = batch_verify_draft_(*reqs[g], mtp);
        from_mtp[g] = mtp ? 1 : 0;
        if (draft[g] >= 0)
            ++n_drafted;
    }
    if (n_drafted == 0)
        return decline("no_draft_in_batch");

    // Per-row penalties use the sampler's device history slots (host
    // output_tokens is the truth). Row 2g penalizes the history (ends with
    // t0), row 2g+1 the history plus the draft (written at the slot's end
    // below); emitted tokens land there afterward, keeping the slot in sync.
    std::vector<const int32_t*> pen_hist(static_cast<size_t>(2 * N), nullptr);
    std::vector<int> pen_n(static_cast<size_t>(2 * N), 0), pen_slot(N, -1), pen_need(N, 0);
    std::vector<float> pen_v(static_cast<size_t>(6 * N), 0.0f);
    bool any_pen = false;
    for (int g = 0; g < N; ++g) {
        const Request& req = *reqs[g];
        const bool needs = req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                           req.presence_penalty != 0.0f;
        if (!needs)
            continue;
        const int need = static_cast<int>(req.output_tokens.size());
        if (d_penalty_hist_ == nullptr || penalty_hist_slots_ <= 0 || need + 2 > penalty_hist_cap_)
            return decline("penalty_history_unavailable");
        const int slot = penalty_hist_slot_(req.id, reqs);
        if (slot < 0)
            return decline("penalty_history_unavailable");
        auto& hs = penalty_hist_state_[static_cast<size_t>(slot)];
        int32_t* base = d_penalty_hist_ + static_cast<size_t>(slot) * penalty_hist_cap_;
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
        pen_hist[2 * g] = base;
        pen_n[2 * g] = need;
        pen_hist[2 * g + 1] = base;
        pen_n[2 * g + 1] = need + 1;
        for (int j = 0; j < 2; ++j) {
            pen_v[static_cast<size_t>(2 * g + j) * 3 + 0] = req.repetition_penalty;
            pen_v[static_cast<size_t>(2 * g + j) * 3 + 1] = req.frequency_penalty;
            pen_v[static_cast<size_t>(2 * g + j) * 3 + 2] = req.presence_penalty;
        }
        any_pen = true;
    }

    // KV blocks for both rows of every request (positions p0, p0+1).
    std::vector<int> p0(N);
    int max_ctx = 0;
    auto rollback_all = [&](int upto) {
        for (int h = 0; h < upto; ++h)
            kv_manager_->rollback(reqs[h]->id, p0[h]);
    };
    for (int g = 0; g < N; ++g) {
        p0[g] = reqs[g]->context_len() - 1;
        const int blocks_needed = (p0[g] + 2 + kv_bs - 1) / kv_bs;
        while (static_cast<int>(kv_manager_->block_table(reqs[g]->id).size()) < blocks_needed) {
            int nb = kv_manager_->append_block(reqs[g]->id);
            if (nb < 0 && kv_cache_raw_ != nullptr) {
                const int have = kv_cache_raw_->total_blocks();
                if (kv_cache_raw_->ceiling_blocks() > have) {
                    kv_cache_raw_->try_grow_to(have + std::max(64, have / 4));
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
        max_ctx = std::max(max_ctx, p0[g] + 2);
    }

    // Stage: [tokens | positions | row ctx lens | seq slots | out slots |
    // snap slots | chunk_len, snap_n], one H2D; row tables, one H2D.
    const int cap_rows = bv_.cap_rows, cap_seq = bv_.cap_seq, table_cap = bv_.table_cap;
    int32_t* h = bv_.h_stage.as<int32_t>();
    int32_t* h_tok = h;
    int32_t* h_pos = h + cap_rows;
    int32_t* h_ctx = h + 2ull * cap_rows;
    int32_t* h_seq = h + 3ull * cap_rows;
    int32_t* h_out = h_seq + cap_seq;
    int32_t* h_snap = h_out + cap_seq;
    int32_t* h_misc = h_snap + cap_seq;
    int32_t* h_tab = bv_.h_row_tables.as<int32_t>();
    for (int g = 0; g < N; ++g) {
        const int32_t t0 = reqs[g]->output_tokens.back();
        const bool has = draft[g] >= 0;
        h_tok[2 * g] = t0;
        h_tok[2 * g + 1] = has ? draft[g] : t0;
        h_pos[2 * g] = p0[g];
        h_pos[2 * g + 1] = p0[g] + 1;
        h_ctx[2 * g] = p0[g] + 1;
        h_ctx[2 * g + 1] = has ? p0[g] + 2 : 1;  // a pad row attends one token
        h_seq[g] = live[g];
        h_out[g] = spare[g];
        h_snap[g] = live[g];
        const auto& bt = kv_manager_->block_table(reqs[g]->id);
        for (int j = 0; j < 2; ++j) {
            int32_t* row = h_tab + static_cast<size_t>(2 * g + j) * table_cap;
            std::copy(bt.begin(), bt.end(), row);
        }
    }
    h_misc[0] = 2;  // real rows per group
    h_misc[1] = 1;  // snapshot row
    int32_t* d = bv_.d_stage;
    int32_t* d_tok = d;
    int32_t* d_pos = d + cap_rows;
    int32_t* d_ctx = d + 2ull * cap_rows;
    int32_t* d_seq = d + 3ull * cap_rows;
    int32_t* d_out = d_seq + cap_seq;
    int32_t* d_snap = d_out + cap_seq;
    int32_t* d_misc = d_snap + cap_seq;
    auto check = [&](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("spec-batch: %s failed: %s", op, cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    const size_t stage_ints = 3ull * cap_rows + 3ull * cap_seq + 2;
    if (!check(cudaMemcpyAsync(d, h, stage_ints * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
               "stage H2D") ||
        !check(cudaMemcpyAsync(bv_.d_row_tables, h_tab,
                               static_cast<size_t>(2 * N) * table_cap * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream),
               "row tables H2D")) {
        rollback_all(N);
        return false;
    }
    // The draft token behind each penalized request's history (pinned
    // source: the staged chunk tokens).
    for (int g = 0; g < N && any_pen; ++g) {
        if (pen_slot[g] < 0)
            continue;
        int32_t* base = d_penalty_hist_ + static_cast<size_t>(pen_slot[g]) * penalty_hist_cap_;
        if (!check(cudaMemcpyAsync(base + pen_need[g], h_tok + 2 * g + 1, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream),
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
    const bool capture_on = spec_capture_ready_(max_ctx);

    InferenceState state;
    state.token_ids = d_tok;
    state.positions = d_pos;
    state.n_tokens = 2 * N;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = bv_.d_row_tables;
    state.context_lens = d_ctx;
    state.max_context_len = max_ctx;
    state.n_sequences = 2 * N;
    state.max_blocks_per_seq = table_cap;
    state.is_prefill = true;
    state.prefill_offset = 0;
    state.spec_verify_chunk = true;
    state.chunk_decode_attn = true;
    state.kv_manager = kv_manager_.get();
    state.ssm_state = ssm_state_.get();
    state.ssm_seq_id = live[0];
    state.ssm_seq_slots = d_seq;
    state.ssm_n_seq = N;
    state.ssm_seq_tokens = 2;
    // Factored spare: no second slot, so no commit destination. The drafted
    // row leaves as (g, k, delta) plus one conv tap per layer, both addressed
    // by the live slot; the snapshot at row 1 stays the in-place one.
    const bool factored = runtime_config_.speculative.factored_spare && bv_.d_fac != nullptr;
    state.ssm_out_slots = factored ? nullptr : d_out;
    state.ssm_snap_slots = factored ? d_seq : d_snap;
    if (factored) {
        state.ssm_fac_out = bv_.d_fac;
        state.ssm_fac_stride = bv_.fac_stride;
        state.ssm_fac_layer_stride = bv_.fac_layer_stride;
        state.ssm_tap_out = bv_.d_tap;
        state.ssm_tap_layer_stride = bv_.tap_layer_stride;
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
        state.ctx_capacity = spec_capture_ctx_tier_(max_ctx);
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
    std::vector<uint8_t> row_alt(static_cast<size_t>(2 * N), 0);
    bool any_alt = false;
    for (int g = 0; g < N; ++g) {
        if (stop_mask_active(*reqs[g], think_end_id_)) {
            row_alt[2 * g] = row_alt[2 * g + 1] = 1;
            any_alt = true;
        }
    }
    const int32_t* d_ban = banned_tokens_device_(stream);
    const int n_ban = d_ban ? static_cast<int>(banned_token_ids_.size()) : 0;
    const int32_t* d_alt = (any_alt && stop_mask_.n() > 0) ? stop_mask_.device(vram_alloc_, stream) : nullptr;
    const int n_alt = d_alt ? stop_mask_.n() : 0;
    executor_->greedy_argmax_all(2 * N, bv_.d_argmax, stream, nullptr, 0, nullptr, 1.0f, 0.0f, 0.0f, nullptr,
                                 0, d_ban, n_ban, /*allow_cutlass=*/true, d_alt, n_alt,
                                 any_alt ? row_alt.data() : nullptr, any_pen ? pen_hist.data() : nullptr,
                                 any_pen ? pen_n.data() : nullptr, any_pen ? pen_v.data() : nullptr);
    int32_t* h_am = bv_.h_argmax.as<int32_t>();
    if (!check(cudaMemcpyAsync(h_am, bv_.d_argmax, static_cast<size_t>(2 * N) * sizeof(int32_t),
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

    long long acc_total = 0, emit_total = 0;
    bool any_mtp = false;
    std::vector<int> row0(N), emitted_of(N, 0);
    for (int g = 0; g < N; ++g) {
        auto& req = reqs[g];
        const bool has = draft[g] >= 0;
        const bool think_at_start = req->in_think_block;
        int matched = 0, emitted = 0;
        for (int j = 0; j < 2; ++j) {
            if (j == 1 && !has)
                break;
            const int32_t tok = h_am[2 * g + j];
            req->output_tokens.push_back(tok);
            track_think_state(*req, tok);
            emitted++;
            const bool hard_stop = should_stop(*req, tok) ||
                                   static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (req->constraints)
                req->constraints->update(tok);
            if (hard_stop) {
                finish_request(req);
                break;
            }
            if (j == 1 || tok != draft[g])
                break;  // bonus reached, or the draft diverged
            matched++;
            if (req->think_budget > 0.0f && req->in_think_block != think_at_start)
                break;
        }
        row0[g] = 2 * g;
        emitted_of[g] = emitted;
        // Device penalty history: the emitted tokens are rows 2g.. of the
        // pinned argmax buffer, in order.
        if (pen_slot[g] >= 0 && emitted > 0) {
            int32_t* base = d_penalty_hist_ + static_cast<size_t>(pen_slot[g]) * penalty_hist_cap_;
            auto& hs = penalty_hist_state_[static_cast<size_t>(pen_slot[g])];
            if (cudaMemcpyAsync(base + pen_need[g], h_am + 2 * g,
                                static_cast<size_t>(emitted) * sizeof(int32_t), cudaMemcpyHostToDevice,
                                stream) == cudaSuccess)
                hs.synced = pen_need[g] + emitted;
            else
                hs.synced = -1;
        }
        kv_manager_->touch(req->id);
        kv_manager_->rollback(req->id, p0[g] + 1 + matched);
        // State: the spare holds [t0, draft], the live slot holds t0; accept
        // swaps the spare in. Factored: no spare, the scan wrote a row for
        // EVERY group, so accept keeps it for the next step and anything
        // else must clear it (else state advances by a token nobody accepted).
        const bool take = matched == 1 && req->status != RequestStatus::FINISHED;
        if (factored)
            (take ? bv_.pending : bv_.clear_slots).push_back(live[g]);
        else if (take) {
            recurrent_slot_of_[req->id] = spare[g];
            bv_.spare_of[req->id] = live[g];
        }
        if (has) {
            req->spec_verifies++;
            req->spec_drafted += 1;
            req->spec_accepted += matched;
            req->spec_emitted += emitted;
            req->spec_consecutive_misses = matched == 0 ? req->spec_consecutive_misses + 1 : 0;
        }
        acc_total += matched;
        emit_total += emitted;
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

    // MTP: feed every bound request's emitted pairs and draft its next token
    // in one head pass (the chunk's hidden rows are still in place).
    static double t_verify_ms = 0.0;
    static long long t_steps = 0;
    t_verify_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - verify_t0).count();
    if (++t_steps % 100 == 0)
        IMP_LOG_INFO("spec-batch: %lld steps, forward+argmax %.2f ms/step (before the MTP feed)", t_steps,
                     t_verify_ms / t_steps);
    if (mtp_spec_decode_enabled())
        mtp_batched_feed_(reqs, row0, emitted_of, stream);
    spec_stats_record_(
        any_mtp, n_drafted, acc_total, emit_total,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - verify_t0).count());
    static bool logged = false;
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("spec-batch: first batched verify n=%d drafted=%d accepted=%lld emitted=%lld", N,
                     n_drafted, acc_total, emit_total);
    }
    return true;
}

}  // namespace imp
