// =============================================================================
// engine_spec_ngram.cpp — n-gram (prompt-lookup) speculative decoding
// =============================================================================
//
// Drafts come from suffix matches against the request's own prompt+output
// tokens — no draft model, no MTP head. Two matchers: the suffix index
// (SuffixDraftIndex, speculative.suffix, default) with frequency-voted
// continuations and adaptive draft length, or the legacy single-most-recent
// backward scan (ngram_draft). The verify step replays
// [t0, d1..dK] as a teacher-forced continuation chunk through the standard
// chunked-prefill forward (KV written in place), applies the tier-aware LM
// head to every chunk position (executor greedy_argmax_all) and accepts the
// longest prefix where the model's greedy token equals the draft. Rejected
// draft KV entries are dropped via KVCacheManager::rollback — safe because
// draft blocks were appended this step and are never content-hashed (prefix
// hashing only covers prompt prefill blocks).
//
// Phase 1 scope (gated in spec_ngram_gates_ok_): batch-1, greedy sampling,
// no penalties / logit_bias / DRY / mirostat, no logprobs, no json/schema
// constraints, no think budget, chunked-prefill-capable archs only. The
// verify loop runs eager — the async conditional graph loop stays off while
// speculation is enabled (the host must see every token to draft the next
// step).
//
// Hybrid (SSM/GDN) models (speculative.hybrid): the chunk forward advances
// recurrent state through rejected draft positions, so the committed
// per-sequence state slab is copied to scratch before the chunk; a full
// acceptance keeps the advanced state as-is (it covers exactly the forwarded
// tokens), a partial acceptance restores the slab and re-forwards the
// accepted prefix (~one extra chunk forward, amortized over the accepted
// tokens). Draft sources: the suffix/ngram matcher first; when it has no
// match and an MTP head is enabled (--mtp-spec-decode / speculative.mtp_k),
// the pending MTP chain fills the chunk (engine_spec_mtp.cpp).
// =============================================================================

#include "compute/json_constrain.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/engine.h"
#include "runtime/ngram_draft.h"
#include "runtime/request.h"
#include "runtime/suffix_draft.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace imp {

bool Engine::ensure_spec_buffers_(int chunk_cap, int max_blocks) {
    if (spec_chunk_cap_ >= chunk_cap && spec_block_table_cap_ >= max_blocks)
        return true;
    free_spec_buffers_();
    bool ok = cudaMalloc(&d_spec_tokens_, chunk_cap * sizeof(int32_t)) == cudaSuccess &&
              cudaMalloc(&d_spec_positions_, chunk_cap * sizeof(int)) == cudaSuccess &&
              cudaMalloc(&d_spec_block_table_, max_blocks * sizeof(int)) == cudaSuccess &&
              cudaMalloc(&d_spec_context_len_, sizeof(int)) == cudaSuccess &&
              cudaMalloc(&d_spec_past_len_, sizeof(int)) == cudaSuccess &&
              cudaMalloc(&d_spec_chunk_len_, sizeof(int)) == cudaSuccess &&
              cudaMalloc(&d_spec_argmax_, chunk_cap * sizeof(int32_t)) == cudaSuccess &&
              cudaMallocHost(&h_spec_argmax_, chunk_cap * sizeof(int32_t)) == cudaSuccess;
    if (!ok) {
        IMP_LOG_WARN("spec-ngram: buffer allocation failed — speculation disabled this step");
        free_spec_buffers_();
        return false;
    }
    spec_chunk_cap_ = chunk_cap;
    spec_block_table_cap_ = max_blocks;
    return true;
}

void Engine::log_spec_stats_() const {
    if (spec_stats_.verify_steps + spec_stats_.miss_steps == 0)
        return;
    IMP_LOG_INFO("[spec-ngram] verify_steps=%lld miss_steps=%lld drafted=%lld accepted=%lld "
                 "(%.1f%%) emitted=%lld (%.2f tok/verify)",
                 spec_stats_.verify_steps, spec_stats_.miss_steps, spec_stats_.drafted,
                 spec_stats_.accepted,
                 spec_stats_.drafted ? 100.0 * spec_stats_.accepted / spec_stats_.drafted : 0.0,
                 spec_stats_.emitted,
                 spec_stats_.verify_steps
                     ? static_cast<double>(spec_stats_.emitted) / spec_stats_.verify_steps
                     : 0.0);
}

// Hybrid verify: scratch slab holding the committed recurrent state across
// the speculative chunk forward. Sized once (per_seq_bytes is fixed after
// init); freed with the other spec buffers.
bool Engine::ensure_spec_state_scratch_() {
    if (!ssm_state_) return false;
    const size_t bytes = ssm_state_->per_seq_bytes();
    if (spec_state_scratch_ && spec_state_scratch_bytes_ >= bytes) return true;
    if (spec_state_scratch_) {
        IMP_CUDA_CHECK_LOG(cudaFree(spec_state_scratch_));
        spec_state_scratch_ = nullptr;
        spec_state_scratch_bytes_ = 0;
    }
    if (cudaMalloc(&spec_state_scratch_, bytes) != cudaSuccess) {
        IMP_LOG_WARN("spec-hybrid: state scratch alloc failed (%zu bytes) — "
                     "speculation disabled this step", bytes);
        return false;
    }
    spec_state_scratch_bytes_ = bytes;
    return true;
}

// Mirror of fill_recurrent_state's slot resolution (decode requests own a
// slot acquired at prefill; the modulo fallback matches its legacy path).
int Engine::recurrent_slot_for_(int req_id) const {
    auto it = recurrent_slot_of_.find(req_id);
    if (it != recurrent_slot_of_.end()) return it->second;
    const int cap = ssm_state_ ? ssm_state_->max_sequences() : 0;
    return cap > 0 ? req_id % cap : 0;
}

void Engine::free_spec_buffers_() {
    // Captured verify graphs bake these buffer pointers — drop them first.
    free_spec_graphs_();
    if (spec_state_scratch_) {
        IMP_CUDA_CHECK_LOG(cudaFree(spec_state_scratch_));
        spec_state_scratch_ = nullptr;
        spec_state_scratch_bytes_ = 0;
    }
    if (d_spec_tokens_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_tokens_));
    if (d_spec_positions_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_positions_));
    if (d_spec_block_table_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_block_table_));
    if (d_spec_context_len_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_context_len_));
    if (d_spec_past_len_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_past_len_));
    if (d_spec_chunk_len_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_chunk_len_));
    if (d_spec_argmax_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_argmax_));
    if (h_spec_argmax_) IMP_CUDA_CHECK_LOG(cudaFreeHost(h_spec_argmax_));
    d_spec_tokens_ = nullptr;
    d_spec_positions_ = nullptr;
    d_spec_block_table_ = nullptr;
    d_spec_context_len_ = nullptr;
    d_spec_past_len_ = nullptr;
    d_spec_chunk_len_ = nullptr;
    d_spec_argmax_ = nullptr;
    h_spec_argmax_ = nullptr;
    spec_chunk_cap_ = 0;
    spec_block_table_cap_ = 0;
}

// Burst-hybrid re-arm: a given-up request whose async-loop burst
// (speculative.burst tokens) has completed gets a short probe window — two
// draft attempts and a fresh acceptance sample. Think models produce their
// draft-rich region only after the reasoning prose; a sticky give-up would
// lock them out exactly there.
void Engine::spec_maybe_rearm_(Request& req) const {
    const auto& scfg = runtime_config_.speculative;
    if (!req.spec_ngram_given_up || req.spec_acceptance_doomed || scfg.burst <= 0 ||
        scfg.give_up_after <= 0)
        return;
    if (static_cast<int>(req.output_tokens.size()) - req.spec_last_giveup_pos < scfg.burst)
        return;
    req.spec_ngram_given_up = false;
    req.spec_consecutive_misses = std::max(0, scfg.give_up_after - 2);
    req.spec_verifies = 0;
    req.spec_drafted = 0;
    req.spec_accepted = 0;
}

// Adaptive miss burst: consecutive draft misses double the burst length
// (boundary overhead amortizes on draft-poor stretches), a hit resets it.
// Capped at speculative.burst so re-probing never stops entirely.
int Engine::spec_effective_miss_burst_(const Request& req) const {
    const auto& scfg = runtime_config_.speculative;
    int burst = scfg.miss_burst;
    if (burst <= 0)
        return 0;
    // Cap growth at 4x: longer bursts overshoot draft-rich regions right
    // after a transition (think prose -> code), losing more speculation
    // upside than the saved boundary overhead is worth.
    const int doublings = std::min(2, req.spec_consecutive_misses / 8);
    burst <<= doublings;
    const int cap = scfg.burst > 0 ? scfg.burst : 128;
    return std::min(burst, cap);
}

// Whether a bounded async-loop burst may be launched directly from the spec
// hook (mirrors the launch conditions in step_decode_process_outputs).
bool Engine::spec_burst_launch_ok_(const Request& req) const {
    if (!decode_graph_pool_[0].is_ready() || offload_mgr_ || !config_.use_cuda_graphs)
        return false;
    // A runner that is setup but NOT parked is in flight — blocked. Parked
    // for a DIFFERENT request is fine: try_launch_async_graph_loop tears the
    // stale park down and rebuilds (a parked warmup request would otherwise
    // lock every later server request out of the loop entirely).
    if (async_graph_runner_.is_setup() && async_parked_req_id_ < 0)
        return false;
    // Text-fallback think tracking needs host-side per-token matching.
    if (req.in_think_block && think_end_id_ < 0)
        return false;
    if (req.output_tokens.empty() || req.status != RequestStatus::DECODING)
        return false;
    return true;
}

bool Engine::spec_ngram_gates_ok_(const Request& req, bool ignore_think) const {
    if (req.spec_ngram_given_up) return false;
    // Greedy sampling only: verify compares argmax tokens.
    const bool greedy = (req.temperature <= 0.0f || req.top_k == 1);
    if (!greedy) return false;
    // rep/freq/presence penalties are replicated in the verify
    // (greedy_argmax_all) for the unbounded window; a bounded repeat_last_n
    // window slides per chunk row and is not replicated — stay eager there.
    const bool penalties = req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                           req.presence_penalty != 0.0f;
    if (penalties && req.repeat_last_n != 0) return false;
    // Logit-shaping the verify chunk does not replicate disqualifies.
    if (req.dry_multiplier != 0.0f || req.mirostat != 0 || !req.logit_bias.empty())
        return false;
    if (req.logprobs || req.json_mode || !req.json_schema.empty()) return false;
    // Think budget forces tokens INSIDE the think block (loop/host-side) —
    // verify only outside it; the think interior runs loop bursts, which
    // handle the budget device-side.
    if (!ignore_think && req.think_budget > 0.0f && req.in_think_block) return false;
    if (req.status != RequestStatus::DECODING || req.output_tokens.empty()) return false;
    // Recurrent state (SSM/GDN) advances on every forwarded token. The
    // hybrid verify path (speculative.hybrid) rides SSMState's contiguous
    // per-sequence slab for snapshot/restore.
    if (ssm_state_ && !runtime_config_.speculative.hybrid) return false;
    // MoE speculation engages only for native-NVFP4 experts: the batched
    // verify forward reads the NVFP4 expert cache directly and nets +49-81%
    // on draft-rich code-edit (Qwen3-Coder-30B-FP4, 2026-07-02) with a -3-7%
    // draft-poor floor (miss_burst hybrid). GGUF-MoE verify re-dequants every
    // activated expert per step and measured -22% — those stay on the async
    // conditional-graph loop (as does everything when speculative.moe=false).
    if (model_->profile().is_moe &&
        !(runtime_config_.speculative.moe && model_->profile().moe_experts_nvfp4))
        return false;
    if (!supports_chunked_prefill_()) return false;
    return true;
}

SuffixDraftIndex& Engine::spec_suffix_index_(const Request& req) {
    auto it = spec_suffix_idx_.find(req.id);
    if (it == spec_suffix_idx_.end()) {
        const auto& scfg = runtime_config_.speculative;
        it = spec_suffix_idx_
                 .emplace(req.id, SuffixDraftIndex(std::max(1, scfg.min_match), scfg.max_match))
                 .first;
        it->second.append(req.input_tokens.data(), static_cast<int>(req.input_tokens.size()));
        it->second.append(req.prediction_tokens.data(),
                          static_cast<int>(req.prediction_tokens.size()));
    }
    return it->second;
}

bool Engine::step_spec_verify_(std::shared_ptr<Request>& req, cudaStream_t stream) {
    const auto& scfg = runtime_config_.speculative;
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // Build the draft from the request's own token history. Predicted-output
    // tokens (OpenAI `prediction`, never forwarded through the model) sit
    // between prompt and output so the current output suffix stays the tail:
    // a completion that tracks the prediction finds max_match-length matches
    // in the prediction region every verify step.
    const int pred_begin = static_cast<int>(req->input_tokens.size());
    const int pred_end = pred_begin + static_cast<int>(req->prediction_tokens.size());

    const int k = std::max(1, scfg.k);
    int draft_start = -1;
    std::vector<int32_t> draft;
    if (scfg.suffix) {
        // Suffix index: input ++ prediction indexed at first use, output
        // tokens appended incrementally (every emit path lands in
        // output_tokens, so loop-burst tokens are picked up here too).
        SuffixDraftIndex& idx = spec_suffix_index_(*req);
        const int out_indexed = idx.size() - pred_end;
        idx.append(req->output_tokens.data() + out_indexed,
                   static_cast<int>(req->output_tokens.size()) - out_indexed);
        draft = idx.draft(k, std::max(k, scfg.suffix_k_max), &draft_start);
    } else {
        std::vector<int32_t> history;
        history.reserve(req->input_tokens.size() + req->prediction_tokens.size() +
                        req->output_tokens.size());
        history.insert(history.end(), req->input_tokens.begin(), req->input_tokens.end());
        history.insert(history.end(), req->prediction_tokens.begin(),
                       req->prediction_tokens.end());
        history.insert(history.end(), req->output_tokens.begin(), req->output_tokens.end());
        draft = ngram_draft(history.data(), static_cast<int>(history.size()), k,
                            std::max(1, scfg.min_match), scfg.max_match, &draft_start);
    }
    const bool draft_from_prediction = draft_start >= pred_begin && draft_start < pred_end;
    // MTP fallback: when the matcher has no draft, the pending MTP chain
    // (drafted at the end of the previous verify step / prefill tail) fills
    // the chunk — the trained head drafts where suffix matching cannot
    // (78-94% depth-1 accept on Qwen3.6, PR #804). Subject to the economics
    // guard below: high accept alone does not pay for the eager chunk +
    // chain lm_head GEMVs + hybrid replay.
    bool draft_from_mtp = false;
    if (draft.empty() && mtp_spec_decode_enabled()) {
        draft = mtp_take_draft_(*req);
        draft_from_mtp = !draft.empty();
    }
    if (draft.empty()) {
        spec_stats_.miss_steps++;
        if (++req->spec_consecutive_misses >= scfg.give_up_after && scfg.give_up_after > 0 &&
            !req->spec_ngram_given_up) {
            req->spec_ngram_given_up = true;
            req->spec_last_giveup_pos = static_cast<int>(req->output_tokens.size());
            IMP_LOG_INFO("spec-ngram: req %d gave up after %d consecutive draft misses — "
                         "re-enabling async graph loop",
                         req->id, req->spec_consecutive_misses);
        }
        // Skip the eager probe step entirely when a bounded loop burst can
        // take over right away — the eager path costs ~2x per token and the
        // burst forwards output.back() itself.
        if (scfg.miss_burst > 0 && !req->spec_ngram_given_up && spec_burst_launch_ok_(*req) &&
            try_launch_async_graph_loop(req, req->output_tokens.back(), stream,
                                        spec_effective_miss_burst_(*req))) {
            return true;  // step handled by the burst launch
        }
        return false;  // no usable draft — normal decode step
    }

    // Verify chunk = [t0, d1..dK]: t0 is the last emitted (not yet forwarded)
    // token; positions p0..p0+K with p0 = context_len-1.
    const int32_t t0 = req->output_tokens.back();
    int K = static_cast<int>(draft.size());
    const int p0 = req->context_len() - 1;

    // attn_scores_ capacity clamp (same rule as chunked prefill):
    // n_tokens × ctx_len must fit s_cap².
    if (executor_) {
        const int s_cap = executor_->attn_scores_cap();
        if (s_cap > 0) {
            const int64_t cap2 = static_cast<int64_t>(s_cap) * s_cap;
            const int64_t ctx_end = static_cast<int64_t>(p0) + 1 + K;
            while (K > 0 && static_cast<int64_t>(K + 1) * ctx_end > cap2) --K;
            if (K <= 0) {
                spec_stats_.miss_steps++;
                return false;
            }
        }
    }
    const int chunk_len = K + 1;
    // Graph-captured verify (#847): pad the chunk up to its bucket length so
    // one cached graph serves every draft length in the bucket. Pad rows
    // (copies of t0 at positions after every real row) are causally invisible
    // to the real rows; their KV entries fall to the same rollback that drops
    // rejected drafts, and the argmax window below stays [0, chunk_len).
    const bool capture_on = spec_capture_ready_(p0 + spec_capture_bucket_(chunk_len));
    const int chunk_pad = capture_on ? spec_capture_bucket_(chunk_len) : chunk_len;
    const int ctx_len = p0 + chunk_pad;  // context including the full (padded) chunk

    const int blocks_needed = (ctx_len + kv_bs - 1) / kv_bs;

    // KV blocks for all chunk positions (mirror step_decode's append loop).
    // On allocation failure, trim back to the pre-step valid length (p0 KV
    // entries: t0 has not been forwarded yet) and fall through to the normal
    // decode path.
    while (static_cast<int>(kv_manager_->block_table(req->id).size()) < blocks_needed) {
        int new_block = kv_manager_->append_block(req->id);
        if (new_block < 0) {
            // KV exhausted. The old evict_lru fallback freed a LIVE sequence (no
            // recompute path) → silent corruption. Just roll back the
            // speculative growth and fall through to the normal decode path.
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
    }
    const auto& block_table = kv_manager_->block_table(req->id);
    const int n_blocks = static_cast<int>(block_table.size());

    // Staging capacity follows the REAL table size — the async graph loop
    // pre-allocates blocks for the whole remaining generation, so the table
    // is usually much larger than this chunk needs. Captured graphs bake the
    // staging pointers, so size for the largest bucket up front (a later
    // realloc would silently invalidate every cached graph).
    const int chunk_cap =
        std::max({chunk_pad, scfg.k + 1, capture_on ? spec_capture_bucket_max_() : 0});
    const int table_cap = capture_on
                              ? std::max(n_blocks + 16,
                                         (spec_capture_ctx_cap_ + kv_bs - 1) / kv_bs + 16)
                              : n_blocks + 16;
    if (!ensure_spec_buffers_(chunk_cap, table_cap)) {
        spec_stats_.miss_steps++;
        return false;
    }

    // Upload chunk metadata.
    std::vector<int32_t> h_tokens;
    h_tokens.reserve(chunk_pad);
    h_tokens.push_back(t0);
    h_tokens.insert(h_tokens.end(), draft.begin(), draft.begin() + K);
    h_tokens.resize(chunk_pad, t0);  // bucket padding
    std::vector<int> h_positions(chunk_pad);
    for (int i = 0; i < chunk_pad; ++i) h_positions[i] = p0 + i;

    auto check = [&](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("spec-ngram: %s failed: %s", op, cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    if (!check(cudaMemcpyAsync(d_spec_tokens_, h_tokens.data(), chunk_pad * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream), "tokens H2D") ||
        !check(cudaMemcpyAsync(d_spec_positions_, h_positions.data(), chunk_pad * sizeof(int),
                               cudaMemcpyHostToDevice, stream), "positions H2D") ||
        !check(cudaMemcpyAsync(d_spec_block_table_, block_table.data(), n_blocks * sizeof(int),
                               cudaMemcpyHostToDevice, stream), "block table H2D") ||
        !check(cudaMemcpyAsync(d_spec_context_len_, &ctx_len, sizeof(int), cudaMemcpyHostToDevice,
                               stream), "context len H2D") ||
        (capture_on &&
         (!check(cudaMemcpyAsync(d_spec_past_len_, &p0, sizeof(int), cudaMemcpyHostToDevice,
                                 stream), "past len H2D") ||
          !check(cudaMemcpyAsync(d_spec_chunk_len_, &chunk_len, sizeof(int),
                                 cudaMemcpyHostToDevice, stream), "chunk len H2D"))))
        return false;

    // Forward the chunk through the standard continuation-prefill path.
    // Capture mode pins the workspace layout to the largest bucket so every
    // bucket's graph bakes the same activation-tensor carve.
    if (!executor_->resize_workspace(
            capture_on ? std::max(chunk_pad, spec_capture_bucket_max_()) : chunk_len, stream)) {
        spec_stats_.miss_steps++;
        return false;
    }
    if (executor_->has_decode_workspace()) executor_->use_workspace(0);

    InferenceState state;
    state.token_ids = d_spec_tokens_;
    state.positions = d_spec_positions_;
    state.n_tokens = chunk_pad;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_spec_block_table_;
    state.context_lens = d_spec_context_len_;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = 0;
    state.is_prefill = true;
    state.prefill_offset = p0;
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) state.kv_seq_id = req->id;
    if (capture_on) {
        // The tier (not the full capacity) sizes the baked gather grids; the
        // executor scratch covers the full capacity, so any tier fits it.
        state.ctx_capacity = spec_capture_ctx_tier_(ctx_len);
        state.d_past_len = d_spec_past_len_;
        state.d_chunk_len = d_spec_chunk_len_;
    }

    // Hybrid (SSM/GDN): bind the recurrent state and preserve the committed
    // slab — the chunk forward advances it through rejected draft positions.
    const bool hybrid = ssm_state_ != nullptr;
    int rec_slot = -1;
    if (hybrid) {
        if (!ensure_spec_state_scratch_()) {
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
        rec_slot = recurrent_slot_for_(req->id);
        if (!check(cudaMemcpyAsync(spec_state_scratch_, ssm_state_->seq_base(rec_slot),
                                   ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice, stream),
                   "state snapshot D2D")) {
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
        fill_recurrent_state(*req, state, /*reset=*/false, stream);
    }

    Tensor logits_out;
    if (runtime_config().diagnostics.spec_capture_probe) {
        spec_capture_probe_forward_(state, logits_out, stream);
    } else if (capture_on) {
        if (!spec_captured_forward_(state, logits_out, stream)) {
            // Warmup use, or capture/launch failed (nothing executed) — run
            // eagerly. A doomed capture means the capture-mode path itself
            // threw; strip the capture fields so the eager forward takes the
            // plain host-length chunk path (the padded chunk stays valid).
            if (spec_capture_doomed_) {
                state.ctx_capacity = 0;
                state.d_past_len = nullptr;
                // Keep d_chunk_len: the chunk stays padded, and the hybrid
                // recurrent-state updates must still stop at the real length.
            }
            executor_->forward_logits(state, logits_out, stream);
        }
    } else {
        executor_->forward_logits(state, logits_out, stream);
    }

    // Penalty history (production parity: output_tokens, unbounded window).
    const bool penalties = req->repetition_penalty != 1.0f ||
                           req->frequency_penalty != 0.0f || req->presence_penalty != 0.0f;
    const int32_t* d_hist = nullptr;
    int n_hist = 0;
    if (penalties) {
        InferenceState pstate;
        upload_penalties(*req, pstate, stream);
        if (pstate.penalty_tokens == nullptr) {
            // Upload failed — an unpenalized verify would diverge from the
            // eager path; fall back to the normal step. The chunk forward
            // already ran: restore the committed hybrid state.
            if (hybrid)
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                    ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                    ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice, stream));
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
        d_hist = pstate.penalty_tokens;
        n_hist = pstate.n_penalty_tokens;
    }

    // Greedy token for every chunk position, D2H, host compare.
    executor_->greedy_argmax_all(chunk_len, d_spec_argmax_, stream, d_hist, n_hist,
                                 d_spec_tokens_ + 1, req->repetition_penalty,
                                 req->frequency_penalty, req->presence_penalty);
    const bool argmax_ok =
        check(cudaMemcpyAsync(h_spec_argmax_, d_spec_argmax_, chunk_len * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream), "argmax D2H") &&
        check(cudaStreamSynchronize(stream), "verify sync");
    if (!argmax_ok) {
        // No tokens were emitted; leave the request exactly as before the
        // step (the chunk forward advanced hybrid state — restore it).
        if (hybrid)
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                                               ssm_state_->per_seq_bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
        kv_manager_->rollback(req->id, p0);
        return false;
    }

    if (getenv("IMP_SPEC_TRACE")) {
        std::string s = "[verify] p0=" + std::to_string(p0) + " t0=" + std::to_string(t0) +
                        " draft=[";
        for (int j = 0; j < K; ++j) s += std::to_string(draft[j]) + (j + 1 < K ? "," : "");
        s += "] argmax=[";
        for (int j = 0; j < chunk_len; ++j)
            s += std::to_string(h_spec_argmax_[j]) + (j + 1 < chunk_len ? "," : "");
        s += "]";
        IMP_LOG_INFO("%s", s.c_str());
    }

    // Accept the longest matching prefix and emit tokens through the same
    // per-token bookkeeping as the eager decode path.
    int matched = 0;  // accepted draft tokens (their KV entries are valid)
    int emitted = 0;
    for (int j = 0; j < chunk_len; ++j) {
        const int32_t tokj = h_spec_argmax_[j];
        req->output_tokens.push_back(tokj);
        track_think_state(*req, tokj);
        emitted++;
        const bool hard_stop = should_stop(*req, tokj) ||
                               static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
        // Per-request FSM; must advance before finish_request returns the
        // manager to the pool. (Spec gates exclude json/schema requests, so
        // this is normally null — kept for parity with the eager path.)
        if (req->constraints)
            req->constraints->update(tokj);
        if (hard_stop) {
            finish_request(req);
            break;
        }
        if (j >= K || tokj != draft[j]) break;  // bonus token reached or draft diverged
        matched++;
        // Entering a budgeted think block mid-chunk: the budget forcing lives
        // in the loop/eager path — stop extending; the accepted prefix stays.
        if (req->think_budget > 0.0f && req->in_think_block) break;
    }
    kv_manager_->touch(req->id);

    // MTP bookkeeping runs BEFORE the hybrid re-forward below — it consumes
    // this chunk's hidden rows, which the re-forward overwrites.
    if (mtp_spec_decode_enabled())
        mtp_post_verify_update_(*req, emitted);

    // Drop KV for rejected draft positions: keep t0 + matched drafts.
    kv_manager_->rollback(req->id, p0 + 1 + matched);

    // Hybrid, partial acceptance: the in-place state advanced through
    // rejected draft positions — restore the committed slab and re-advance
    // it over the accepted prefix ([t0, d1..d_matched] is still staged in
    // d_spec_tokens_). Full acceptance keeps the advanced state (it covers
    // exactly the forwarded tokens); finished requests skip (the slot is
    // released and nothing reads the state again). The replay rewrites the
    // kept KV rows with identical values.
    if (hybrid && matched < K && req->status != RequestStatus::FINISHED) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                                           ssm_state_->per_seq_bytes(),
                                           cudaMemcpyDeviceToDevice, stream));
        const int replay_ctx = p0 + 1 + matched;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_spec_context_len_, &replay_ctx, sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
        state.n_tokens = matched + 1;
        state.max_context_len = replay_ctx;
        // The replay is an unpadded eager forward of the accepted prefix —
        // run it on the plain host-length chunk path, not the capture path.
        state.ctx_capacity = 0;
        state.d_past_len = nullptr;
        state.d_chunk_len = nullptr;
        Tensor replay_logits;
        executor_->forward_logits(state, replay_logits, stream);
    }

    spec_stats_.verify_steps++;
    spec_stats_.drafted += K;
    spec_stats_.accepted += matched;
    spec_stats_.emitted += emitted;
    // (request-end stats logging lives in finish_request)

    // Acceptance economics: structured-but-mutating content (number tables,
    // counters) produces plenty of suffix matches whose continuations are
    // always wrong — pure miss counting never triggers there while every
    // step pays a full verify chunk for 1 emitted token. After a fair
    // sample, an acceptance rate below 15% can't amortize the verify cost;
    // hand the request back to the async loop.
    if (matched == 0) {
        ++req->spec_consecutive_misses;
    } else {
        req->spec_consecutive_misses = 0;
    }
    req->spec_verifies++;
    req->spec_drafted += K;
    req->spec_accepted += matched;
    if (draft_from_prediction) {
        req->pred_accepted += matched;
        req->pred_rejected += K - matched;
    }
    // MTP economics: an MTP-filled verify must emit enough tokens to beat
    // the async loop it displaces (eager chunk ≈ 2x a loop step, plus the
    // chain's full-vocab lm_head GEMVs, plus the hybrid partial-accept
    // replay). Below ~4 emitted/verify the step loses outright — doom MTP
    // drafting for this request; the suffix matcher and miss bursts carry on.
    if (draft_from_mtp) {
        mtp_econ_verifies_++;
        mtp_econ_emitted_ += emitted;
        constexpr int kMtpEconSample = 8;  // fair sample before judging
        // Break-even avg emitted/verify is configurable (0 disables): the
        // right value depends on the chain lm_head cost (NVFP4 vs FP16) and
        // the verify-chunk cost, both of which have moved since #852.
        const float min_emit = runtime_config_.speculative.mtp_econ_min_emit;
        if (min_emit > 0.0f && mtp_econ_verifies_ >= kMtpEconSample &&
            static_cast<float>(mtp_econ_emitted_) <
                static_cast<float>(mtp_econ_verifies_) * min_emit)
            mtp_unbind_("uneconomic: avg emitted/verify below break-even");
    }
    const bool acceptance_poor =
        req->spec_verifies >= 8 &&
        req->spec_accepted * 100 < req->spec_drafted * 15;
    if (!req->spec_ngram_given_up && scfg.give_up_after > 0 &&
        (acceptance_poor || req->spec_consecutive_misses >= scfg.give_up_after)) {
        req->spec_ngram_given_up = true;
        req->spec_last_giveup_pos = static_cast<int>(req->output_tokens.size());
        if (acceptance_poor)
            req->spec_acceptance_doomed = true;  // economics verdict is final
        IMP_LOG_INFO("spec-ngram: req %d gave up (%s: verifies=%d accepted=%lld/%lld misses=%d) — "
                     "re-enabling async graph loop",
                     req->id, acceptance_poor ? "acceptance-poor" : "draft-poor",
                     req->spec_verifies, req->spec_accepted, req->spec_drafted,
                     req->spec_consecutive_misses);
    }

    return true;
}

// spec_capture_probe_forward_ (diagnostics census) and the production
// graph-captured verify (#847) live in engine_spec_capture.cpp.

}  // namespace imp
