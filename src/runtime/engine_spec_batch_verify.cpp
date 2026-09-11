// =============================================================================
// engine_spec_batch_verify.cpp - batched speculative verify on the GDN hybrid
// =============================================================================
//
// One forward per decode step for N requests, 2 rows each: the request's last
// emitted token and its draft. Group g runs the GDN layers on its live slot
// L_g, commits the 2-row state into the request's spare slot P_g and the 1-row
// state in place into L_g (compute/gdn.h out_slots / snap_slots). Accept
// swaps L_g and P_g on the host; reject keeps L_g. No state copies, no
// replay: the verify depth is 1 (docs/plans/2026-09-11-batched-mtp-verify.md).
// Attention rows take the decode-attention route with per-row block tables
// and context lengths, the same shape as the multi-candidate chunk.
// =============================================================================

#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/engine.h"
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
// Per-row conditions the batched verify replicates: greedy argmax, no
// penalties, no logit shaping, no constraint FSM, no logprobs.
bool batch_verify_row_ok(const Request& r) {
    const bool greedy = r.temperature <= 0.0f || r.top_k == 1;
    if (!greedy)
        return false;
    if (r.repetition_penalty != 1.0f || r.frequency_penalty != 0.0f || r.presence_penalty != 0.0f)
        return false;
    if (r.dry_multiplier != 0.0f || r.mirostat != 0 || !r.logit_bias.empty())
        return false;
    if (r.logprobs || r.json_mode || !r.json_schema.empty() || !r.regex_pattern.empty() || !r.grammar.empty() ||
        !r.tool_constraint_tools.empty())
        return false;
    return r.status == RequestStatus::DECODING && !r.output_tokens.empty();
}
}  // namespace

bool Engine::batch_verify_on_() const {
    const auto& scfg = runtime_config_.speculative;
    if (!scfg.batch_verify || !scfg.hybrid)
        return false;
    if (!model_ || model_->config().ssm_inner_size <= 0)
        return false;
    return true;
}

int Engine::batch_verify_spare_slots_() const {
    if (!batch_verify_on_() || config_.max_batch_size <= 1)
        return 0;
    return config_.max_batch_size;
}

bool Engine::ensure_batch_verify_bufs_() {
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
    const bool ok = cudaMalloc(&bv_.d_stage, stage_ints * sizeof(int32_t)) == cudaSuccess &&
                    cudaMalloc(&bv_.d_row_tables, table_ints * sizeof(int32_t)) == cudaSuccess &&
                    cudaMalloc(&bv_.d_argmax, static_cast<size_t>(cap_rows) * sizeof(int32_t)) == cudaSuccess &&
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
    return true;
}

void Engine::free_batch_verify_bufs_() {
    if (bv_.d_stage)
        IMP_CUDA_CHECK_LOG(cudaFree(bv_.d_stage));
    if (bv_.d_row_tables)
        IMP_CUDA_CHECK_LOG(cudaFree(bv_.d_row_tables));
    if (bv_.d_argmax)
        IMP_CUDA_CHECK_LOG(cudaFree(bv_.d_argmax));
    bv_ = BatchVerifyBufs{};
}

// Spare slots are the reserved pool slots past the multi-candidate ones.
// Committed on first use (lazy pool), held for the request's lifetime.
int Engine::acquire_spare_slot_(int req_id) {
    auto it = spare_slot_of_.find(req_id);
    if (it != spare_slot_of_.end())
        return it->second;
    if (!ssm_state_)
        return -1;
    if (!spare_slots_initialized_) {
        free_spare_slots_.clear();
        const int mc = spec_mc_reserved_slots_();
        for (int i = ssm_state_->n_reserved() - 1; i >= mc; --i)
            free_spare_slots_.push_back(ssm_state_->reserved_slot(i));
        spare_slots_initialized_ = true;
    }
    if (free_spare_slots_.empty())
        return -1;
    const int slot = free_spare_slots_.back();
    if (!ssm_state_->ensure_slot(slot))
        return -1;  // stays free; the step falls back to plain decode
    free_spare_slots_.pop_back();
    spare_slot_of_[req_id] = slot;
    return slot;
}

void Engine::release_spare_slot_(int req_id) {
    auto it = spare_slot_of_.find(req_id);
    if (it == spare_slot_of_.end())
        return;
    free_spare_slots_.push_back(it->second);
    spare_slot_of_.erase(it);
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
        const int out_indexed =
            std::clamp(idx.size() - pred_end, 0, static_cast<int>(req.output_tokens.size()));
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
    if (!ssm_state_ || !batch_verify_on_())
        return "off";
    if (batch.size() < 2)
        return "batch_of_one";
    if (ssm_state_->n_reserved() - spec_mc_reserved_slots_() <= 0)
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
            think_logic::should_force_think_end(r->think_budget, think_end_id_, r->max_tokens, r->output_tokens,
                                                think_start_id_, r->started_in_think,
                                                runtime_config_.runtime.think_answer_reserve))
            return "think_forcing_due";
    }
    return nullptr;
}

bool Engine::batch_verify_eligible_(const std::vector<std::shared_ptr<Request>>& batch) const {
    const char* why = batch_verify_refusal_(batch);
    if (why == nullptr)
        return true;
    if (batch.size() >= 2 && batch_verify_on_() && bv_last_refusal_ != why) {
        bv_last_refusal_ = why;
        IMP_LOG_INFO("spec-batch: batched verify refused by '%s' (n=%zu)", why, batch.size());
    }
    return false;
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
        if (bv_last_refusal_ != why) {
            bv_last_refusal_ = why;
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
        spare[g] = acquire_spare_slot_(reqs[g]->id);
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
    if (!check(cudaMemcpyAsync(d, h, stage_ints * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "stage H2D") ||
        !check(cudaMemcpyAsync(bv_.d_row_tables, h_tab, static_cast<size_t>(2 * N) * table_cap * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream),
               "row tables H2D")) {
        rollback_all(N);
        return false;
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
    state.ssm_out_slots = d_out;
    state.ssm_snap_slots = d_snap;
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
    executor_->greedy_argmax_all(2 * N, bv_.d_argmax, stream, nullptr, 0, nullptr, 1.0f, 0.0f, 0.0f, nullptr, 0,
                                 d_ban, n_ban, /*allow_cutlass=*/true, d_alt, n_alt,
                                 any_alt ? row_alt.data() : nullptr);
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
            const bool hard_stop =
                should_stop(*req, tok) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
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
        if (from_mtp[g] && mtp_spec_decode_enabled())
            mtp_post_verify_update_(*req, emitted, 2 * g);
        kv_manager_->touch(req->id);
        kv_manager_->rollback(req->id, p0[g] + 1 + matched);
        // State: the spare holds the state after [t0, draft], the live slot
        // the state after t0. Accept = the spare becomes the live slot.
        if (matched == 1 && req->status != RequestStatus::FINISHED) {
            recurrent_slot_of_[req->id] = spare[g];
            spare_slot_of_[req->id] = live[g];
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
    spec_stats_record_(any_mtp, n_drafted, acc_total, emit_total,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - verify_t0)
                           .count());
    static bool logged = false;
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("spec-batch: first batched verify n=%d drafted=%d accepted=%lld emitted=%lld", N, n_drafted,
                     acc_total, emit_total);
    }
    return true;
}

}  // namespace imp
