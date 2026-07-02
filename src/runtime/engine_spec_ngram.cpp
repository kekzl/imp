// =============================================================================
// engine_spec_ngram.cpp — n-gram (prompt-lookup) speculative decoding
// =============================================================================
//
// Drafts come from suffix matches against the request's own prompt+output
// tokens (ngram_draft) — no draft model, no MTP head. The verify step replays
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
// constraints, no think budget, no recurrent state (SSM/GDN cannot rewind),
// chunked-prefill-capable archs only. The verify loop runs eager — the async
// conditional graph loop stays off while speculation is enabled (the host
// must see every token to draft the next step).
// =============================================================================

#include "compute/json_constrain.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/engine.h"
#include "runtime/ngram_draft.h"
#include "runtime/request.h"

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

void Engine::free_spec_buffers_() {
    if (d_spec_tokens_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_tokens_));
    if (d_spec_positions_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_positions_));
    if (d_spec_block_table_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_block_table_));
    if (d_spec_context_len_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_context_len_));
    if (d_spec_argmax_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_argmax_));
    if (h_spec_argmax_) IMP_CUDA_CHECK_LOG(cudaFreeHost(h_spec_argmax_));
    d_spec_tokens_ = nullptr;
    d_spec_positions_ = nullptr;
    d_spec_block_table_ = nullptr;
    d_spec_context_len_ = nullptr;
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
    // Recurrent state (SSM/GDN) advances on every forwarded token and cannot
    // be rewound on draft rejection.
    if (ssm_state_ || gdn_state_) return false;
    // MoE speculation engages only for native-NVFP4 experts: the batched
    // verify forward reads the NVFP4 expert cache directly and nets +49-81%
    // on draft-rich code-edit (Qwen3-Coder-30B-FP4, 2026-07-02) with a -3-7%
    // draft-poor floor (miss_burst hybrid). GGUF-MoE verify re-dequants every
    // activated expert per step and measured -22% — those stay on the async
    // conditional-graph loop (as does everything when speculative.moe=false).
    if (model_->profile().is_moe &&
        !(runtime_config_.speculative.moe && model_->profile().moe_experts_nvfp4))
        return false;
    if (mtp_spec_decode_enabled()) return false;
    if (!supports_chunked_prefill_()) return false;
    return true;
}

bool Engine::step_spec_verify_(std::shared_ptr<Request>& req, cudaStream_t stream) {
    const auto& scfg = runtime_config_.speculative;
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // Build the draft from the request's own token history.
    std::vector<int32_t> history;
    history.reserve(req->input_tokens.size() + req->output_tokens.size());
    history.insert(history.end(), req->input_tokens.begin(), req->input_tokens.end());
    history.insert(history.end(), req->output_tokens.begin(), req->output_tokens.end());
    const int hist_n = static_cast<int>(history.size());

    const int k = std::max(1, scfg.k);
    std::vector<int32_t> draft =
        ngram_draft(history.data(), hist_n, k, std::max(1, scfg.min_match), scfg.max_match);
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
    const int ctx_len = p0 + chunk_len;  // context including the full chunk

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
    // is usually much larger than this chunk needs.
    if (!ensure_spec_buffers_(std::max(chunk_len, scfg.k + 1), n_blocks + 16)) {
        spec_stats_.miss_steps++;
        return false;
    }

    // Upload chunk metadata.
    std::vector<int32_t> h_tokens;
    h_tokens.reserve(chunk_len);
    h_tokens.push_back(t0);
    h_tokens.insert(h_tokens.end(), draft.begin(), draft.begin() + K);
    std::vector<int> h_positions(chunk_len);
    for (int i = 0; i < chunk_len; ++i) h_positions[i] = p0 + i;

    auto check = [&](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("spec-ngram: %s failed: %s", op, cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    if (!check(cudaMemcpyAsync(d_spec_tokens_, h_tokens.data(), chunk_len * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream), "tokens H2D") ||
        !check(cudaMemcpyAsync(d_spec_positions_, h_positions.data(), chunk_len * sizeof(int),
                               cudaMemcpyHostToDevice, stream), "positions H2D") ||
        !check(cudaMemcpyAsync(d_spec_block_table_, block_table.data(), n_blocks * sizeof(int),
                               cudaMemcpyHostToDevice, stream), "block table H2D") ||
        !check(cudaMemcpyAsync(d_spec_context_len_, &ctx_len, sizeof(int), cudaMemcpyHostToDevice,
                               stream), "context len H2D"))
        return false;

    // Forward the chunk through the standard continuation-prefill path.
    if (!executor_->resize_workspace(chunk_len, stream)) {
        spec_stats_.miss_steps++;
        return false;
    }
    if (executor_->has_decode_workspace()) executor_->use_workspace(0);

    InferenceState state;
    state.token_ids = d_spec_tokens_;
    state.positions = d_spec_positions_;
    state.n_tokens = chunk_len;
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

    Tensor logits_out;
    executor_->forward_logits(state, logits_out, stream);

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
            // eager path; fall back to the normal step.
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
    if (!check(cudaMemcpyAsync(h_spec_argmax_, d_spec_argmax_, chunk_len * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream), "argmax D2H"))
        return false;
    if (!check(cudaStreamSynchronize(stream), "verify sync")) return false;

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

    // Drop KV for rejected draft positions: keep t0 + matched drafts.
    kv_manager_->rollback(req->id, p0 + 1 + matched);

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

}  // namespace imp
