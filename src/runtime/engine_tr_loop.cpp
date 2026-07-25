// =============================================================================
// engine_tr_loop.cpp — verify-in-loop launch/resume (#1055 phase 2)
// =============================================================================
//
// Runs the token-recycling draft→verify→accept cycle inside a conditional
// CUDA-graph WHILE loop (TrVerifyLoopRunner): the device adjacency table
// drafts, the bucket-4 capture-mode chunk forward verifies, and the step
// kernel accepts + stages the next chunk — no host round-trip per verify
// step. The host drains accepted tokens from the mapped ring (with think
// tracking + stop handling, mirroring step_async_graph_resume) and only
// re-enters on miss / stop / budget / ceiling.
//
// Self-priming: chunk 0 is staged as [t0] alone (a decode-as-verify step),
// so a launch never needs a host-side draft; the loop exits with reason
// "miss" when the device table has no confirmed successor.
//
// Design + reuse map: docs/plans/2026-07-23-verify-in-loop.md.
// =============================================================================

#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "compute/rowwise_topm.h"
#include "runtime/think_stop_logic.h"

#include <cuda_runtime.h>
#include <chrono>
#include <climits>
#include <thread>
#include <vector>

namespace imp {

bool Engine::tr_loop_in_flight_() const { return tr_loop_runner_.launch_in_flight(); }

void Engine::tr_loop_teardown_() {
    if (tr_loop_runner_.launch_in_flight())
        tr_loop_runner_.finish(nullptr);
    tr_loop_runner_.cleanup();
    tr_loop_req_ = nullptr;
    if (spec_tr_dev_.succ)
        tr_device_free(spec_tr_dev_);
}

bool Engine::try_launch_tr_verify_loop_(std::shared_ptr<Request>& req, cudaStream_t stream) {
    const auto& scfg = runtime_config_.speculative;
    if (tr_loop_doomed_ || !scfg.recycle_loop || !scfg.token_recycling ||
        !config_.use_cuda_graphs || !scfg.capture || mtp_spec_decode_enabled())
        return false;
    // Native-ST-NVFP4 only: on GGUF-source models the bucket-4 verify chunk
    // forward rides the dequant prefill path — every verify pays source
    // dequant, and the loop regresses instead of winning (measured
    // 2026-07-24: -9.5% Qwen3-8B-Q8, -28.8% Qwen3-14B-Q6K vs spec-off; the
    // +38-97% wins are on the ST-NVFP4 small-M verify route, #1055).
    if (!model_->config().is_nvfp4_prequant) {
        tr_loop_doomed_ = true;
        IMP_LOG_INFO("recycle_loop: disabled — GGUF-source model pays source dequant per "
                     "verify chunk (see issue #1060, measured -9.5%%/-28.8%%)");
        return false;
    }
    if (tr_loop_runner_.launch_in_flight())
        return false;
    if (req->tr_loop_given_up)
        return false;  // economics verdict is final for this request
    // v1 gates: greedy, no penalties / logit shaping / constraints — the
    // in-loop argmax is penalty-free by construction.
    const bool greedy = req->temperature <= 0.0f || req->top_k == 1;
    if (!greedy || req->repetition_penalty != 1.0f || req->frequency_penalty != 0.0f ||
        req->presence_penalty != 0.0f || req->dry_multiplier != 0.0f || req->mirostat != 0 ||
        !req->logit_bias.empty() || req->logprobs || req->json_mode ||
        !req->json_schema.empty() || !req->tool_constraint_tools.empty() || req->constraints)
        return false;
    // Think budget (design: "no think-budget-IN-think"): forcing injects a
    // host-side think-end exactly at the budget boundary, which the
    // autonomous loop cannot do. Instead of gating the whole request, cap
    // the burst so the loop exits AT the boundary — the host recount then
    // forces on the drained tokens. Out of think (or budget spent →
    // decline), forcing cannot fire and the loop runs uncapped. The server
    // default think_budget=0.5 previously kept the loop off for every
    // server request (#1060 server story).
    int think_cap = INT_MAX;
    if (req->think_budget > 0.0f && think_end_id_ >= 0) {
        bool thinking = false;
        const int n_reason = think_logic::count_reasoning_tokens(
            req->output_tokens, think_start_id_, think_end_id_, req->started_in_think, thinking);
        if (thinking) {
            const int frac_limit = static_cast<int>(req->max_tokens * req->think_budget);
            const int reserve_limit =
                static_cast<int>(req->max_tokens) - think_logic::kMaxAnswerReserve;
            const int think_limit = frac_limit > reserve_limit ? frac_limit : reserve_limit;
            think_cap = think_limit - n_reason;
            if (think_cap <= 0)
                return false;  // forcing is due now — the host path handles it
        }
    }
    // Dense decode-attn route only (the per-row block tables are the
    // mask-free multi-row mechanism the chunk forward relies on).
    if (!scfg.verify_decode_attn || ssm_state_ != nullptr || model_->profile().is_moe ||
        model_->config().is_mla() || swa_sizing_active_ || offload_mgr_)
        return false;
    if (req->status != RequestStatus::DECODING || req->output_tokens.empty() ||
        !supports_chunked_prefill_())
        return false;

    // Miss-exit backoff: a cold adjacency table exits after ~1 token; give
    // the eager/burst paths miss_burst tokens to warm it before relaunching
    // (else every launch pays graph-launch overhead for one token).
    if (tr_loop_backoff_req_ == req->id &&
        req->output_tokens.size() < tr_loop_backoff_out_)
        return false;

    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    constexpr int kChunkPad = 4;  // bucket-4 body: one batched-GEMV sweep
    const int p0 = req->context_len() - 1;
    const int remaining = static_cast<int>(req->max_tokens) -
                          static_cast<int>(req->output_tokens.size());
    int token_limit = remaining;
    if (scfg.burst > 0 && token_limit > scfg.burst)
        token_limit = scfg.burst;
    if (token_limit > think_cap)
        token_limit = think_cap;
    if (token_limit < 8)
        return false;  // not worth a launch

    // KV blocks for the whole burst (prepare_graph_loop pattern) — the loop
    // never allocates; the step kernel's token budget keeps writes inside.
    const int final_ctx = p0 + token_limit + kChunkPad + 1;
    const int blocks_needed = (final_ctx + kv_bs - 1) / kv_bs;
    while (static_cast<int>(kv_manager_->block_table(req->id).size()) < blocks_needed) {
        if (kv_manager_->append_block(req->id) < 0) {
            kv_manager_->rollback(req->id, p0);
            return false;
        }
    }
    const auto& block_table = kv_manager_->block_table(req->id);
    const int n_blocks = static_cast<int>(block_table.size());

    // Bake the tier for the request's FULL remaining generation (not this
    // burst) so successive bursts reuse the captured graph instead of
    // re-instantiating per launch (first probe: 91 rebuilds / 256 tokens).
    const int tier = spec_capture_ctx_tier_(p0 + remaining + kChunkPad + 1);
    const int chunk_cap = std::max({kChunkPad, scfg.k + 1, spec_capture_bucket_max_()});
    const int table_cap =
        std::max(n_blocks + 16, (spec_capture_ctx_cap_ + kv_bs - 1) / kv_bs + 16);
    if (!ensure_spec_buffers_(chunk_cap, table_cap))
        return false;

    // Device adjacency table + one-time prompt seeding for this request.
    if (!spec_tr_dev_.succ) {
        if (!tr_device_init(spec_tr_dev_, model_->config_.vocab_size,
                            std::max(1, scfg.recycle_slots), stream)) {
            tr_loop_doomed_ = true;
            return false;
        }
    }
    if (!req->tr_dev_prompt_fed) {
        std::vector<int32_t> seq;
        seq.reserve(req->input_tokens.size() + req->output_tokens.size());
        seq.insert(seq.end(), req->input_tokens.begin(), req->input_tokens.end());
        seq.insert(seq.end(), req->output_tokens.begin(), req->output_tokens.end());
        if (seq.size() >= 2) {
            int32_t* d_tmp = nullptr;
            if (cudaMallocAsync(&d_tmp, seq.size() * sizeof(int32_t), stream) == cudaSuccess) {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_tmp, seq.data(),
                                                   seq.size() * sizeof(int32_t),
                                                   cudaMemcpyHostToDevice, stream));
                tr_observe_pairs(spec_tr_dev_, d_tmp, static_cast<int>(seq.size()), stream);
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_tmp, stream));
            }
        }
        req->tr_dev_prompt_fed = true;
    }

    // Stage chunk 0 = [t0] (self-priming) via the pinned stage block.
    const int32_t t0 = req->output_tokens.back();
    const int cap = spec_chunk_cap_;
    for (int i = 0; i < kChunkPad; ++i) {
        h_spec_stage_[i] = t0;
        h_spec_stage_[cap + i] = p0 + i;
        h_spec_stage_[2 * cap + i] = (i == 0) ? (p0 + 1) : 1;
    }
    h_spec_stage_[3 * cap] = p0 + kChunkPad;
    h_spec_stage_[3 * cap + 1] = p0;
    h_spec_stage_[3 * cap + 2] = 1;
    if (cudaMemcpyAsync(d_spec_stage_, h_spec_stage_, (3ull * cap + 3) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return false;
    // Row-replicated block tables covering the whole pre-allocated burst.
    for (int i = 0; i < kChunkPad; ++i)
        std::copy(block_table.begin(), block_table.end(),
                  h_spec_row_tables_pinned_ + static_cast<size_t>(i) * spec_block_table_cap_);
    if (cudaMemcpyAsync(d_spec_row_block_tables_, h_spec_row_tables_pinned_,
                        static_cast<size_t>(kChunkPad) * spec_block_table_cap_ *
                            sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return false;

    // Top-M scratch must exist BEFORE the body capture (lazy malloc would
    // abort it).
    rowwise_topm_reserve(kChunkPad, std::min({std::max(1, scfg.recycle_slots),
                                              static_cast<int>(kRowwiseTopMMax)}));

    // Workspace pinned to the largest bucket (same rule as the eager capture).
    if (!executor_->resize_workspace(std::max(kChunkPad, spec_capture_bucket_max_()), stream))
        return false;
    if (executor_->has_decode_workspace())
        executor_->use_workspace(0);

    // Runner setup (or rearm-style reuse when compatible).
    if (!tr_loop_runner_.compatible(executor_->workspace_generation(), tier)) {
        InferenceState state;
        state.token_ids = d_spec_tokens_;
        state.positions = d_spec_positions_;
        state.n_tokens = kChunkPad;
        state.kv_cache = kv_cache_raw_;
        state.block_tables = d_spec_row_block_tables_;
        state.context_lens = d_spec_row_ctx_lens_;
        state.max_blocks_per_seq = spec_block_table_cap_;
        state.n_sequences = kChunkPad;
        state.chunk_decode_attn = true;
        state.max_context_len = tier;
        state.is_prefill = true;
        state.prefill_offset = p0;
        state.spec_verify_chunk = true;
        state.kv_manager = kv_manager_.get();
        if (kv_manager_ && kv_manager_->residual_enabled())
            state.kv_seq_id = req->id;
        state.ctx_capacity = tier;
        state.d_past_len = d_spec_past_len_;
        state.d_chunk_len = d_spec_chunk_len_;

        TrVerifyLoopRunner::Config rcfg;
        rcfg.params.chunk_pad = kChunkPad;
        rcfg.params.depth = std::min(kChunkPad - 1, std::max(1, scfg.recycle_depth));
        rcfg.params.min_streak = std::max(0, scfg.recycle_min_streak);
        rcfg.params.topm =
            std::min({std::max(1, scfg.recycle_slots), static_cast<int>(kRowwiseTopMMax)});
        Tokenizer* tok = model_->tokenizer();
        rcfg.params.eos_id = (tok && !req->ignore_eos) ? tok->eos_id() : -1;
        rcfg.params.ctx_ceiling = tier;
        rcfg.stop_ids = chat_template_.stop_token_ids();
        for (int32_t bid : banned_token_ids_)
            rcfg.stop_ids.push_back(bid);

        TrLoopView bufs{};
        bufs.tab = spec_tr_dev_;
        bufs.tokens = d_spec_tokens_;
        bufs.positions = d_spec_positions_;
        bufs.row_ctx_lens = d_spec_row_ctx_lens_;
        bufs.ctx_len = d_spec_context_len_;
        bufs.past_len = d_spec_past_len_;
        bufs.chunk_len = d_spec_chunk_len_;
        bufs.argmax = d_spec_argmax_;
        bufs.topm = d_spec_topm_;

        if (!tr_loop_runner_.setup(executor_.get(), state, bufs, rcfg, stream)) {
            IMP_LOG_WARN("TrVerifyLoop: setup failed — eager verify path from here on");
            tr_loop_doomed_ = true;
            kv_manager_->rollback(req->id, p0);
            return false;
        }
    }
    if (!tr_loop_runner_.launch(token_limit, stream)) {
        kv_manager_->rollback(req->id, p0);
        return false;
    }
    tr_loop_req_ = req;
    tr_loop_initial_p0_ = p0;
    tr_loop_launch_out_ = req->output_tokens.size();
    IMP_LOG_DEBUG("TrVerifyLoop: launched (p0=%d limit=%d)", p0, token_limit);
    return true;
}

bool Engine::step_tr_loop_resume_(cudaStream_t stream) {
    if (!tr_loop_runner_.launch_in_flight() || !tr_loop_req_)
        return false;
    auto req = tr_loop_req_;

    // Micro-poll until at least one new token or the loop exits (the async
    // resume pattern: 200 us sleeps, 30 s safety deadline).
    std::vector<int32_t> toks;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
        tr_loop_runner_.poll_new_tokens(toks);
        if (!toks.empty() || tr_loop_runner_.exit_reason() != 0)
            break;
        if (std::chrono::steady_clock::now() > deadline) {
            IMP_LOG_ERROR("TrVerifyLoop: 30 s without progress — finishing burst");
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }

    bool done = false;
    for (int32_t tok : toks) {
        if (req->status != RequestStatus::DECODING)
            break;  // stop already hit — surplus ring tokens discarded
        req->output_tokens.push_back(tok);
        track_think_state(*req, tok);
        if (should_stop(*req, tok) ||
            static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
            done = true;
        }
    }

    const int exit_reason = tr_loop_runner_.exit_reason();
    if (exit_reason != 0) {
        // Drain any tokens published together with the exit.
        toks.clear();
        tr_loop_runner_.poll_new_tokens(toks);
        for (int32_t tok : toks) {
            if (req->status != RequestStatus::DECODING)
                break;
            req->output_tokens.push_back(tok);
            track_think_state(*req, tok);
            if (should_stop(*req, tok) ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
                finish_request(req);
                done = true;
            }
        }
        tr_loop_runner_.finish(stream);
        // KV reconcile: valid entries = context_len - 1 (the last emitted
        // token is not yet forwarded), exactly the eager-path invariant.
        kv_manager_->rollback(req->id, req->context_len() - 1);
        kv_manager_->touch(req->id);
        // Economics: only miss exits enter the relaunch cycle (stop/budget/
        // ceiling either end the request or are one-offs), so only they are
        // fair evidence. Below the break-even average the loop is handed back
        // to the async graph loop for the rest of the request.
        if (exit_reason == 1) {
            const int emitted = static_cast<int>(req->output_tokens.size()) -
                                static_cast<int>(tr_loop_launch_out_);
            req->tr_loop_miss_bursts++;
            req->tr_loop_miss_emitted += std::max(0, emitted);
            constexpr int kTrEconSample = 4;  // fair sample before judging
            const float min_emit = runtime_config_.speculative.recycle_loop_min_emit;
            if (min_emit > 0.0f && req->tr_loop_miss_bursts >= kTrEconSample &&
                static_cast<float>(req->tr_loop_miss_emitted) <
                    static_cast<float>(req->tr_loop_miss_bursts) * min_emit) {
                req->tr_loop_given_up = true;
                IMP_LOG_INFO(
                    "recycle_loop: req %d gave up (uneconomic: %d tokens over %d "
                    "miss-exit bursts, below %.1f/burst) — async graph loop from here on",
                    req->id, req->tr_loop_miss_emitted, req->tr_loop_miss_bursts,
                    static_cast<double>(min_emit));
            }
        }
        if (exit_reason == 1 && req->status == RequestStatus::DECODING) {
            // Draft miss on a (still) cold table — produce miss_burst tokens
            // via the async decode loop before the next launch, and hand it
            // the step DIRECTLY (leaving it to the eager fallthrough measured
            // 51 stray 10-ms eager verifies per 1024 tokens).
            const int back = std::max(1, runtime_config_.speculative.miss_burst);
            tr_loop_backoff_req_ = req->id;
            tr_loop_backoff_out_ = req->output_tokens.size() + back;
            if (spec_burst_launch_ok_(*req))
                try_launch_async_graph_loop(req, req->output_tokens.back(), stream, back);
        }
        IMP_LOG_DEBUG("TrVerifyLoop: burst end (reason=%d, output=%zu)", exit_reason,
                      req->output_tokens.size());
        tr_loop_req_ = nullptr;
        (void)done;
    }
    return true;
}

}  // namespace imp
