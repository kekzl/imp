#include "runtime/engine.h"
#include "runtime/cuda_graph.h"
#include "runtime/engine_internal.h"
#include "runtime/think_stop_logic.h"
#include "memory/kv_cache.h"
#include "model/chat_template.h"
#include "compute/sampling.h"
#include "core/logging.h"
#include "memory/vram_query.h"

#include <cstring>
#include <algorithm>
#include <vector>

namespace imp {

// CUDA graph decode helpers: async and conditional-graph variants for the
// production decode path.

const int32_t* Engine::banned_tokens_device_(cudaStream_t stream) {
    return upload_id_list_once(d_banned_tokens_, banned_token_ids_, vram_alloc_, "banned_tokens", stream);
}

void Engine::release_async_block_tables_() {
    if (!async_bt_pooled_) {
        if (async_d_block_tables_)
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
        if (async_d_block_tables_swa_)
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_swa_));
    }
    async_d_block_tables_ = nullptr;
    async_d_block_tables_swa_ = nullptr;
    async_bt_pooled_ = false;
}

bool Engine::graph_loop_blocked_(const Request& r, bool experts_matter) const {
    return model_->ngram_table() != nullptr || (experts_matter && experts_on_host_) || r.forcing();
}

int Engine::prepare_graph_loop(std::shared_ptr<Request>& req, int step_limit) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    int remaining = req->max_tokens - static_cast<int>(req->output_tokens.size());
    if (remaining <= 0)
        return 0;

    // Reserves for the burst about to run, not the whole generation (#1636): the
    // launch is clamped to kDecodeBurst (128), 16 while another request
    // waits, or kSpecMissBurst (8); the loop relaunches per burst anyway.
    // With plan_tokens set (max_tokens = remaining context) reserve up to the planning horizon only,
    // or one request would pull the whole pool.
    int reserve_tokens = std::max(1, planning_max_tokens(*req) - static_cast<int>(req->output_tokens.size()));
    if (step_limit > 0)
        reserve_tokens = std::min(reserve_tokens, step_limit);

    constexpr int kMaxLayersForConditionalGraph = 128;
    if (model_->config().n_layers > kMaxLayersForConditionalGraph)
        return 0;

    {
        size_t f = 0, t = 0;
        // Failure zeroes both outputs (vram_query.h): sized as no free VRAM, never over.
        (void)vram_budget_mem_get_info(&f, &t);
        if (f < 256ULL * 1024 * 1024)
            return 0;
    }

    // Pre-allocate KV blocks
    int ctx_len = req->context_len();
    int final_ctx = ctx_len + reserve_tokens;
    int blocks_needed = (final_ctx + kv_bs - 1) / kv_bs;
    int blocks_have = static_cast<int>(kv_manager_->block_table(req->id).size());

    for (int b = blocks_have; b < blocks_needed; b++) {
        if (kv_manager_->append_block(req->id) < 0)
            break;
    }

    int blocks_got = static_cast<int>(kv_manager_->block_table(req->id).size());
    int capped = (blocks_got * kv_bs) - ctx_len;
    if (capped <= 0)
        return 0;
    int steps = std::min(capped, remaining);

    // SWA-aware sizing: the loop runs on-device with no host trim mid-burst, so
    // the WHOLE burst span needs live SWA blocks. Clamp to the SWA group's sized
    // span (the loop relaunches afterwards) and prepare the range up front.
    if (swa_sizing_active_) {
        steps = std::min(steps, swa_burst_cap_tokens_);
        if (steps <= 0)
            return 0;
        kv_manager_->swa_trim(req->id, ctx_len);
        if (!kv_manager_->swa_prepare(req->id, ctx_len, ctx_len + steps))
            return 0;
    }
    return steps;
}

CudaGraphConditionalRunner::Config Engine::build_graph_config(const Request& req, int remaining) const {
    Tokenizer* tok = model_->tokenizer();
    CudaGraphConditionalRunner::Config gcfg;
    gcfg.max_steps = remaining;
    gcfg.initial_context_len = req.context_len();
    gcfg.initial_position = req.context_len() - 1;
    gcfg.eos_id = tok ? tok->eos_id() : -1;
    gcfg.stop_ids = chat_template_.stop_token_ids();
    // Banned tokens (<pad>, <unk>) also stop the graph loop: ban_logits_kernel
    // sets them to -1e30 before sampling, but this catches any that leak through
    // (FP32 precision edge cases with 262K vocab) as a safety net.
    for (int32_t bid : banned_token_ids_) {
        gcfg.stop_ids.push_back(bid);
    }
    gcfg.temperature = req.temperature;
    gcfg.top_p = req.top_p;
    gcfg.top_k = req.top_k;
    gcfg.seed = engine_internal::compute_step_seed(req);
    gcfg.ignore_eos = req.ignore_eos;
    // Penalty parameters for device-side application inside the graph loop
    gcfg.repetition_penalty = req.repetition_penalty;
    gcfg.frequency_penalty = req.frequency_penalty;
    gcfg.presence_penalty = req.presence_penalty;
    gcfg.repeat_last_n = req.repeat_last_n;
    if (req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f || req.presence_penalty != 0.0f) {
        if (!req.output_tokens.empty()) {
            gcfg.penalty_history = req.output_tokens;
        }
    }
    // Think tracking (post_decode_step_kernel, device-side): enabled whenever
    // think_end_id_ >= 0, so reasoning models use the conditional loop instead of
    // eager decode. Mirrors the eager should_stop grace window; budget is opt-in.
    if (think_end_id_ >= 0) {
        gcfg.think_start_id = think_start_id_;
        gcfg.think_end_id = think_end_id_;
        gcfg.initial_in_think = req.in_think_block;
        gcfg.think_grace_tokens = think_logic::kMinAnswerAfterThink;
        gcfg.token_is_whitespace = d_token_is_whitespace_;
        gcfg.vocab_size = static_cast<int>(token_is_whitespace_.size());
        if (req.think_budget > 0.0f) {
            // Device counter starts at 0 every launch; bounded bursts relaunch mid-think,
            // so the per-launch limit is the REMAINING budget (full grant per burst would
            // over-count). Same think_logic::think_limit the eager sampler enforces.
            const int full = think_logic::think_limit(req.max_tokens, req.think_budget,
                                                      runtime_config_.runtime.think_answer_reserve,
                                                      runtime_config_.runtime.think_max_tokens);
            bool thinking_now = false;
            const int used = think_logic::count_reasoning_tokens(
                req.output_tokens, think_start_id_, think_end_id_, req.started_in_think,
                thinking_now);
            gcfg.think_budget_limit = std::max(1, full - used);
        }
    }
    return gcfg;
}

bool Engine::try_launch_async_graph_loop(std::shared_ptr<Request> req, int32_t first_token,
                                         cudaStream_t stream, int step_limit) {
    // Models with per-step host work (PLE n-gram rows from a host table, device expert
    // cache take-over from the host LRU) decode on the per-step graph pool instead.
    if (graph_loop_blocked_(*req, true))
        return false;
    // Constrained requests (json_mode / json_schema / enforced tool call) can
    // NEVER run here: the loop samples device-side with no FSM mask. Guards the
    // spec-ngram burst hooks, which call this directly, bypassing step_decode (#1002).
    if (req->constraints || req->json_mode || !req->json_schema.empty() ||
        !req->regex_pattern.empty() || !req->grammar.empty() || !req->tool_constraint_tools.empty())
        return false;
    // A pending think-budget force belongs to the eager step: it emits "\n" then
    // "</think>" over two steps. Relaunching the loop between them let the device
    // sample a free token after the "\n", so the eager step forced "\n" again and the
    // block never closed (200-token single-stream reply, 663 chars of reasoning, no
    // answer, Qwen3.8-27B 2026-09-15). Covers every launch site, the burst hooks included.
    // A client end (#2420) and a Harmony opener mid-way belong to the eager step too: the device
    // loop samples freely and would run past them.
    if (think_closer_pending_(*req))
        return false;

    int remaining = prepare_graph_loop(req, step_limit);
    if (remaining <= 0)
        return false;
    step_limit = burst_launch_step_limit(step_limit, remaining);

    // Fast relaunch: a parked runner from a previous burst of the SAME request
    // keeps its captured graph, reseeding device state instead of recapturing
    // (full setup costs ~10-20 ms/launch). Rearm and fresh-capture share the same eager first-forward semantics (#683).
    if (async_graph_runner_.is_setup() && async_parked_req_id_ >= 0) {
        const auto& bt = kv_manager_->block_table(req->id);
        const int ctx = req->context_len();
        // SWA table must exist at matching length for a rearm (swa_prepare in
        // prepare_graph_loop already covered the new burst range).
        const auto& swa_bt = kv_manager_->swa_block_table(req->id);
        const bool swa_ok = !swa_sizing_active_ ||
                            (async_d_block_tables_swa_ != nullptr && swa_bt.size() == bt.size());
        if (async_parked_req_id_ == req->id && async_d_block_tables_ != nullptr && swa_ok &&
            static_cast<int>(bt.size()) <= async_bt_capacity_) {
            // Verify steps may have appended KV blocks since the park:
            // refresh the table contents (pointer/capacity baked in graph).
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(async_d_block_tables_, bt.data(),
                                               bt.size() * sizeof(int), cudaMemcpyHostToDevice,
                                               stream));
            if (swa_sizing_active_)
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(async_d_block_tables_swa_, swa_bt.data(),
                                                   swa_bt.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice, stream));
            // Remaining think budget for this launch (the device counter
            // restarts at 0 per launch; see build_graph_config).
            int think_limit = 0;
            if (req->think_budget > 0.0f && think_end_id_ >= 0) {
                bool thinking_now = false;
                const int used = think_logic::count_reasoning_tokens(
                    req->output_tokens, think_start_id_, think_end_id_, req->started_in_think,
                    thinking_now);
                think_limit = std::max(1,
                                       think_logic::think_limit(req->max_tokens, req->think_budget,
                                                                runtime_config_.runtime.think_answer_reserve,
                                                                runtime_config_.runtime.think_max_tokens) -
                                           used);
            }
            if (runtime_config_.diagnostics.spec_trace)
                IMP_LOG_INFO("[burst-launch] REARM seed=%d pos=%d ctx=%d limit=%d", (int)first_token,
                             ctx - 1, ctx, step_limit);
            if (async_graph_runner_.rearm(first_token, /*position=*/ctx - 1, /*context_len=*/ctx, step_limit,
                                          req->in_think_block, think_limit, req->output_tokens, stream) &&
                async_graph_runner_.launch(stream)) {
                async_graph_req_ = req;
                async_pending_tokens_.clear();
                async_pending_cursor_ = 0;
                IMP_LOG_DEBUG("AsyncGraphLoop: rearmed for %d-step burst (ctx=%d)", step_limit,
                              ctx);
                return true;
            }
        }
        // Rearm impossible (table outgrew capacity / context past ceiling /
        // upload failure): tear down and rebuild below.
        async_graph_runner_.cleanup();
        release_async_block_tables_();
        async_parked_req_id_ = -1;
    }

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    // Sized for everything this request can still reach, not today's table
    // (#1647): since #1636 the KV reservation is clamped per burst, so today's
    // table size is always outgrown by the next burst. One int/block: 2 KiB at an 8192-token ceiling.
    const int bt_kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    int bt_ctx_ceiling = req->context_len() +
                         std::max(0, req->max_tokens - static_cast<int>(req->output_tokens.size()));
    if (max_seq_len() > 0)
        bt_ctx_ceiling = std::min(bt_ctx_ceiling, max_seq_len());
    const int bt_capacity = std::max(max_blocks_per_seq, (bt_ctx_ceiling + bt_kv_bs - 1) / bt_kv_bs);

    // Table from the serving metadata pool, a stable address the captured
    // graph can bake; cudaMalloc stays as the fallback for a pool that failed
    // at init (invariant I2).
    const bool pooled = d_agl_block_tables_ != nullptr && bt_capacity <= pool_bt_cap_ &&
                        (!swa_sizing_active_ || d_agl_block_tables_swa_ != nullptr);
    int* d_bt = pooled ? d_agl_block_tables_ : nullptr;
    if (!pooled && cudaMalloc(&d_bt, static_cast<size_t>(bt_capacity) * sizeof(int)) != cudaSuccess)
        return false;
    // The tail past the live table is never read (kernels iterate over
    // max_context_len, stride by max_blocks_per_seq), but zeroing it keeps a
    // stray index pointing at block 0 rather than whatever the allocator returned.
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_bt, 0, static_cast<size_t>(bt_capacity) * sizeof(int), stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_bt, full_bt.data(), max_blocks_per_seq * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
    int* d_bt_swa = nullptr;
    if (swa_sizing_active_) {
        const auto& swa_bt = kv_manager_->swa_block_table(req->id);
        bool swa_ok = static_cast<int>(swa_bt.size()) == max_blocks_per_seq;
        if (swa_ok && pooled)
            d_bt_swa = d_agl_block_tables_swa_;
        else if (swa_ok)
            swa_ok = cudaMalloc(&d_bt_swa, static_cast<size_t>(bt_capacity) * sizeof(int)) == cudaSuccess;
        if (!swa_ok) {
            if (!pooled)
                IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
            return false;
        }
        IMP_CUDA_CHECK_LOG(
            cudaMemsetAsync(d_bt_swa, 0, static_cast<size_t>(bt_capacity) * sizeof(int), stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_bt_swa, swa_bt.data(), max_blocks_per_seq * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    }

    (void)executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_bt;
    state_template.block_tables_swa = d_bt_swa;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = bt_capacity;
    state_template.is_prefill = false;

    // Recurrent state for SSM/GDN layers: pointers are constant for
    // single-sequence decode, so they're safe to bake into the graph.
    (void)fill_recurrent_state(*req, state_template, /*reset=*/false, stream);  // reset=false cannot fail
    bind_mrope_single_(state_template, *req, stream);

    // Upload banned tokens to device for graph-captured logit masking
    if (const int32_t* d_banned = banned_tokens_device_(stream)) {
        state_template.d_banned_tokens = const_cast<int32_t*>(d_banned);
        state_template.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
    }
    // Stop mask id list; the runner supplies the per-step flag (cuda_graph.cu).
    if (const int32_t* d_mask = stop_mask_.device(vram_alloc_, stream)) {
        state_template.d_stop_mask_tokens = d_mask;
        state_template.n_d_stop_mask_tokens = stop_mask_.n();
    }

    if (runtime_config_.diagnostics.spec_trace)
        IMP_LOG_INFO("[burst-launch] FRESH seed=%d ctx=%d limit=%d remaining=%d", (int)first_token,
                     req->context_len(), step_limit, remaining);
    // Captures a ceiling for the WHOLE generation, not this burst (#1647):
    // rearm() refuses once context_len + step_limit exceeds initial_context_len
    // + max_steps. Only when step_limit > 0; with step_limit == 0 the KV blocks are reserved per burst already, so no extra ceiling is needed.
    int capture_steps = remaining;
    if (step_limit > 0) {
        int span = req->max_tokens - static_cast<int>(req->output_tokens.size());
        if (max_seq_len() > 0)
            span = std::min(span, max_seq_len() - req->context_len());
        capture_steps = std::max(remaining, span);
    }
    auto gcfg = build_graph_config(*req, capture_steps);
    gcfg.step_limit = step_limit;

    if (!async_graph_runner_.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (!pooled) {
            if (d_bt_swa)
                IMP_CUDA_CHECK_LOG(cudaFree(d_bt_swa));
            IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
        }
        return false;
    }
    if (!async_graph_runner_.launch(stream)) {
        async_graph_runner_.cleanup();
        if (!pooled) {
            if (d_bt_swa)
                IMP_CUDA_CHECK_LOG(cudaFree(d_bt_swa));
            IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
        }
        return false;
    }

    async_graph_req_ = req;
    async_d_block_tables_ = d_bt;
    async_d_block_tables_swa_ = d_bt_swa;
    async_bt_pooled_ = pooled;
    async_bt_capacity_ = bt_capacity;
    async_parked_req_id_ = -1;
    IMP_LOG_DEBUG("AsyncGraphLoop: launched with %d banned tokens", state_template.n_d_banned_tokens);
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;
    IMP_LOG_INFO("AsyncGraphLoop: launched for %d remaining tokens", remaining);
    return true;
}

// Pipelined constrained decode (json_mode / json_schema, single seq): the
// conditional graph loop cannot run constrained requests (host-side FSM), so
// this splits each tick, enqueuing the NEXT forward before waiting only for the sampled token, keeping the GPU busy during host FSM/mask turnaround.

bool Engine::try_launch_constrained_pipeline(std::shared_ptr<Request> req, cudaStream_t stream) {
    // SWA-aware sizing: the captured decode graph bakes the block-table pointer,
    // not wired for the per-step SWA table rewrite.
    // Falls back to eager constrained decode (step_decode threads the SWA table): correct, just not
    // pipelined.
    if (swa_sizing_active_)
        return false;
    int budget = prepare_graph_loop(req);
    if (budget <= 0)
        return false;

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    auto& p = cpipe_;
    // Device state from the serving metadata pool; cudaMalloc is the fallback
    // for a pool that failed at init (invariant I2). Pinned landing + event
    // outlive one pipeline: acquired on first launch, released by ~Engine.
    p.pooled = d_cp_block_tables_ != nullptr && max_blocks_per_seq <= pool_bt_cap_;
    bool ok = true;
    if (p.pooled) {
        p.d_bt = d_cp_block_tables_;
        p.d_token = d_cp_token_;
        p.d_pos = d_cp_pos_;
        p.d_ctx = d_cp_ctx_;
    } else {
        if (cudaMalloc(&p.d_bt, max_blocks_per_seq * sizeof(int)) != cudaSuccess)
            return false;
        ok = cudaMalloc(&p.d_token, SAMPLE_SCRATCH_BYTES) == cudaSuccess &&
             cudaMalloc(&p.d_pos, sizeof(int)) == cudaSuccess &&
             cudaMalloc(&p.d_ctx, sizeof(int)) == cudaSuccess;
    }
    if (ok && p.h_token.empty())
        p.h_token = PinnedBuffer::acquire(cuda_host_pinned_allocator(), sizeof(int32_t));
    ok = ok && !p.h_token.empty();
    if (ok && !p.ev)
        ok = p.ev.create(cudaEventDisableTiming);
    if (!ok) {
        teardown_constrained_pipeline(/*synchronize=*/false);
        return false;
    }

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_bt, full_bt.data(), max_blocks_per_seq * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
    int32_t first_token = req->output_tokens.back();
    int ctx = req->context_len();
    int pos = ctx - 1;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_token, &first_token, sizeof(int32_t), cudaMemcpyHostToDevice,
                                       stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice, stream));

    // Engine-owned copy, uploaded at init (build_banned_token_list).
    p.d_banned = const_cast<int32_t*>(banned_tokens_device_(stream));

    // Decode workspace (mirrors step_decode_forward).
    (void)executor_->resize_workspace(1, stream);

    InferenceState& st = p.state;
    st = InferenceState{};
    st.token_ids = p.d_token;
    st.positions = p.d_pos;
    st.n_tokens = 1;
    st.kv_cache = kv_cache_raw_;
    st.block_tables = p.d_bt;
    st.context_lens = p.d_ctx;
    st.max_context_len = ctx + budget;
    st.n_sequences = 1;
    st.max_blocks_per_seq = max_blocks_per_seq;
    st.is_prefill = false;
    (void)fill_recurrent_state(*req, st, /*reset=*/false, stream);  // reset=false cannot fail
    bind_mrope_single_(st, *req, stream);
    if (p.d_banned) {
        st.d_banned_tokens = p.d_banned;
        st.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
    }
    // Sampling params (seed refreshed per tick).
    st.temperature = req->temperature;
    st.top_p = req->top_p;
    st.top_k = req->top_k;
    st.repetition_penalty = req->repetition_penalty;
    st.frequency_penalty = req->frequency_penalty;
    st.presence_penalty = req->presence_penalty;
    st.repeat_last_n = req->repeat_last_n;
    // Constraint hooks: the request's manager was prepared at admission.
    st.schema_constrainer = req->constraints ? req->constraints->schema_constrainer() : nullptr;
    st.json_constrainer = req->constraints ? req->constraints->json_constrainer() : nullptr;
    st.regex_constrainer = req->constraints ? req->constraints->regex_constrainer() : nullptr;
    st.grammar_constrainer = req->constraints ? req->constraints->grammar_constrainer() : nullptr;

    p.runner.set_decode_fn([this](cudaStream_t s) { executor_->forward_logits(cpipe_.state, cpipe_.logits, s); });

    // Prime the pipeline: forward for the NEXT token starts now; the mask for
    // it is computed on the next tick (the FSM already absorbed first_token in
    // step_decode_process_outputs).
    if (!p.runner.execute(stream)) {
        teardown_constrained_pipeline(/*synchronize=*/true);
        return false;
    }
    p.forward_in_flight = true;
    p.budget = budget;
    p.produced = 0;
    p.req = req;
    p.active = true;
    IMP_LOG_INFO("ConstrainedPipeline: launched for %d budgeted tokens (%s)", budget,
                 st.grammar_constrainer  ? "grammar"
                 : st.regex_constrainer  ? "regex"
                 : st.schema_constrainer ? "schema"
                                         : "json");
    return true;
}

int Engine::step_constrained_pipeline() {
    auto& p = cpipe_;
    if (!p.active)
        return 0;
    auto req = p.req;
    if (!req || req->status != RequestStatus::DECODING) {
        teardown_constrained_pipeline(/*synchronize=*/true);
        return 0;
    }

    cudaStream_t stream = decode_stream();

    // 1. Mask + sample this tick's logits from the in-flight forward. Mask is
    //    host-computed from FSM state, uploaded stream-ordered behind the logits' producer.
    p.state.seed = engine_internal::compute_step_seed(*req);
    // Remaining output allowance for the force-close mask (#1104): the
    // pipeline keeps its own InferenceState, so it must be refreshed per tick.
    p.state.constrain_remaining_tokens = req->max_tokens - static_cast<int>(req->output_tokens.size());
    // Think-budget enforcement (mirrors fill_sampling_params): when the
    // reasoning budget is exhausted mid-think, force </think> this step.
    p.state.force_token = -1;
    if (think_end_due_(*req))
        p.state.force_token = think_end_id_;
    // Stop mask (mirrors fill_sampling_params): the host drives this sampler
    // per tick, so the list swap is enough, no device flag.
    p.state.d_banned_tokens = p.d_banned;
    p.state.n_d_banned_tokens = p.d_banned ? static_cast<int>(banned_token_ids_.size()) : 0;
    if (p.state.force_token < 0 && stop_mask_active(*req, think_end_id_)) {
        if (const int32_t* d_mask = stop_mask_.device(vram_alloc_, stream)) {
            p.state.d_banned_tokens = d_mask;
            p.state.n_d_banned_tokens = stop_mask_.n();
        }
    }
    // Token-history penalties: per-tick upload, exactly like the eager path.
    p.state.penalty_tokens = nullptr;
    p.state.n_penalty_tokens = 0;
    upload_penalties(*req, p.state, stream);
    executor_->masked_sample_async(p.state, p.logits, p.d_token, p.h_token.as<int32_t>(), stream);
    launch_pipeline_advance(p.d_pos, p.d_ctx, stream);
    IMP_CUDA_CHECK_LOG(cudaEventRecord(p.ev, stream));

    // 2. Enqueue the NEXT forward before the host knows the token: it reads
    //    d_token (just written by the sampler) on the GPU timeline.
    bool more = (p.produced + 1 < p.budget) &&
                (static_cast<int>(req->output_tokens.size()) + 1 < req->max_tokens);
    if (more)
        // false only without a decode fn; capture/replay failures fall back to eager inside execute().
        (void)p.runner.execute(stream);
    p.forward_in_flight = more;

    // 3. Wait only for the sampled token (GPU continues in forward N+1).
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(p.ev));
    int32_t token = *p.h_token.as<int32_t>();

    // 4. Harvest: mirrors step_decode_process_outputs for one token.
    req->output_tokens.push_back(token);
    p.produced++;
    track_think_state(*req, token);
    if (req->constraints)
        req->constraints->update(token);
    kv_manager_->touch(req->id);

    bool done = should_stop(*req, token) ||
                static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
    if (done) {
        teardown_constrained_pipeline(/*synchronize=*/true);
        auto saved = req;
        finish_request(saved);
        return -1;
    }
    if (!more) {
        // KV budget exhausted: drain and let the eager path continue.
        teardown_constrained_pipeline(/*synchronize=*/true);
        return 0;
    }
    return 1;
}

void Engine::teardown_constrained_pipeline(bool synchronize) {
    auto& p = cpipe_;
    if (synchronize)
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(decode_stream()));
    p.runner.invalidate();
    if (!p.pooled) {
        if (p.d_bt)
            IMP_CUDA_CHECK_LOG(cudaFree(p.d_bt));
        if (p.d_token)
            IMP_CUDA_CHECK_LOG(cudaFree(p.d_token));
        if (p.d_pos)
            IMP_CUDA_CHECK_LOG(cudaFree(p.d_pos));
        if (p.d_ctx)
            IMP_CUDA_CHECK_LOG(cudaFree(p.d_ctx));
    }
    p.d_bt = nullptr;
    p.d_token = nullptr;
    p.d_pos = nullptr;
    p.d_ctx = nullptr;
    p.pooled = false;
    p.d_banned = nullptr;  // engine-owned (d_banned_tokens_)
    // h_token and ev stay for the next pipeline; ~Engine releases them.
    p.req = nullptr;
    p.active = false;
    p.forward_in_flight = false;
    p.produced = 0;
    p.budget = 0;
}

void Engine::check_moe_decode_cache_coverage_() {
    const auto& mcfg = model_->config();
    // Coverage check: for prequant MoE, the nvfp4_moe decode cache is
    // all-or-nothing (mirrors executor_forward_moe.cu nvfp4_covers_layer). One
    // uncovered layer falls to host-args legacy, which throws under graph capture.
    if (mcfg.is_nvfp4_prequant && mcfg.n_experts > 0) {
        int moe_layers = 0, covered = 0, uncovered_on_device = 0;
        for (int i = 0; i < mcfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            bool has_experts = L.expert_up_packed.data != nullptr ||
                               (!L.expert_w_up.empty() && L.expert_w_up[0].data != nullptr);
            if (!has_experts)
                continue;
            moe_layers++;
            bool ok = L.nvfp4_moe_up_ptr != nullptr && L.nvfp4_moe_down_ptr != nullptr;
            if (ok && L.expert_gate_packed.data != nullptr)
                ok = L.nvfp4_moe_gate_ptr != nullptr;
            if (ok)
                covered++;
            else if (!L.expert_w_up.empty() && L.expert_w_up[0].on_device)
                uncovered_on_device++;  // host-resident layers go to the device expert cache instead
        }
        // #2180: an uncovered device-resident layer decodes on the legacy path, whose replay faulted.
        if (uncovered_on_device > 0)
            demote_graphs_(GraphDemotionReason::MoeDecodeCacheIncomplete);
        // Host-resident uncovered layers are capture-safe when the device expert cache serves them.
        const bool host_rest_cached = uncovered_on_device == 0 &&
                                      executor_->device_expert_cache_covers_host_layers();
        if (moe_layers > 0 && covered == moe_layers) {
            IMP_LOG_INFO(
                "NVFP4 decode caches: FULL (%d/%d MoE layers) — decode graph "
                "capture eligible",
                covered, moe_layers);
        } else if (moe_layers > 0 && host_rest_cached) {
            IMP_LOG_INFO(
                "NVFP4 decode caches: %d/%d MoE layers, the other %d served by the device "
                "expert cache - decode graph capture eligible",
                covered, moe_layers, moe_layers - covered);
        } else if (moe_layers > 0) {
            IMP_LOG_WARN(
                "NVFP4 decode caches: PARTIAL (%d/%d MoE layers covered): decode graph "
                "capture aborts unless the device expert cache serves the host-resident "
                "layers. Remedies: lower runtime.max_seq_len / max_batch_size, [vram] knobs.",
                covered, moe_layers);
        }
    }
}

}  // namespace imp
