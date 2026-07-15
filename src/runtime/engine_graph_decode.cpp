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

// =====================================================================
// CUDA Graph decode helpers — async and conditional-graph variants
// for the production decode path.
// =====================================================================

int Engine::prepare_graph_loop(std::shared_ptr<Request>& req) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    int remaining = req->max_tokens - static_cast<int>(req->output_tokens.size());
    if (remaining <= 0)
        return 0;

    constexpr int kMaxLayersForConditionalGraph = 128;
    if (model_->config().n_layers > kMaxLayersForConditionalGraph)
        return 0;

    {
        size_t f = 0, t = 0;
        vram_budget_mem_get_info(&f, &t);
        if (f < 256ULL * 1024 * 1024)
            return 0;
    }

    // Pre-allocate KV blocks
    int ctx_len = req->context_len();
    int final_ctx = ctx_len + remaining;
    int blocks_needed = (final_ctx + kv_bs - 1) / kv_bs;
    int blocks_have = static_cast<int>(kv_manager_->block_table(req->id).size());

    for (int b = blocks_have; b < blocks_needed; b++) {
        if (kv_manager_->append_block(req->id) < 0)
            break;
    }

    int blocks_got = static_cast<int>(kv_manager_->block_table(req->id).size());
    int capped = blocks_got * kv_bs - ctx_len;
    if (capped <= 0)
        return 0;
    int steps = std::min(capped, remaining);

    // SWA-aware sizing: the loop runs on-device with no host trim mid-burst,
    // so the WHOLE burst span must have live SWA blocks. Clamp the burst to
    // the span the SWA group is sized for (the loop relaunches afterwards)
    // and prepare the range up front.
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
    // Banned tokens (e.g. <pad>, <unk>) should also stop the graph loop.
    // The ban_logits_kernel sets them to -1e30 before sampling, but if
    // they still leak through (FP32 precision edge cases with 262K vocab),
    // the stop check catches them as a safety net.
    for (int32_t bid : banned_token_ids_) {
        gcfg.stop_ids.push_back(bid);
    }
    gcfg.temperature = req.temperature;
    gcfg.top_p = req.top_p;
    gcfg.top_k = req.top_k;
    gcfg.seed = req.seed;
    gcfg.ignore_eos = req.ignore_eos;
    // Penalty parameters for device-side application inside the graph loop
    gcfg.repetition_penalty = req.repetition_penalty;
    gcfg.frequency_penalty = req.frequency_penalty;
    gcfg.presence_penalty = req.presence_penalty;
    gcfg.repeat_last_n = req.repeat_last_n;
    // Seed penalty history from existing output tokens
    if (req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f || req.presence_penalty != 0.0f) {
        if (!req.output_tokens.empty()) {
            gcfg.penalty_history = req.output_tokens;
        }
    }
    // Think tracking: device-side in post_decode_step_kernel. Enabled whenever a
    // single-token </think> id is known (think_end_id_ >= 0), so the conditional
    // loop runs for reasoning models instead of falling back to eager decode.
    // Provides EOS/stop suppression while inside the block + a post-</think>
    // grace window (matches the eager should_stop path). think_budget_limit stays
    // opt-in (only when the request set a budget).
    if (think_end_id_ >= 0) {
        gcfg.think_start_id = think_start_id_;
        gcfg.think_end_id = think_end_id_;
        gcfg.initial_in_think = req.in_think_block;
        gcfg.think_grace_tokens = think_logic::kMinAnswerAfterThink;
        gcfg.token_is_whitespace = d_token_is_whitespace_;
        gcfg.vocab_size = static_cast<int>(token_is_whitespace_.size());
        if (req.think_budget > 0.0f) {
            // The loop's device-side counter starts at 0 every launch; with
            // bounded bursts (n-gram speculation think-phase) the request
            // relaunches mid-think, so the per-launch limit must be the
            // REMAINING budget or every burst would re-grant the full one.
            const int full = static_cast<int>(req.max_tokens * req.think_budget);
            bool thinking_now = false;
            const int used = think_logic::count_reasoning_tokens(
                req.output_tokens, think_start_id_, think_end_id_, req.started_in_think,
                thinking_now);
            gcfg.think_budget_limit = std::max(1, full - used);
        }
    }
    return gcfg;
}

std::vector<int32_t> Engine::try_graph_loop_decode(std::shared_ptr<Request> req, int32_t first_token,
                                                   cudaStream_t stream) {
    int remaining = prepare_graph_loop(req);
    if (remaining <= 0)
        return {};

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    int* d_block_tables = nullptr;
    if (cudaMallocAsync(&d_block_tables, max_blocks_per_seq * sizeof(int), stream) != cudaSuccess)
        return {};
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_block_tables, full_bt.data(), max_blocks_per_seq * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
    int* d_block_tables_swa = nullptr;
    if (swa_sizing_active_) {
        const auto& swa_bt = kv_manager_->swa_block_table(req->id);
        if (static_cast<int>(swa_bt.size()) == max_blocks_per_seq &&
            cudaMallocAsync(&d_block_tables_swa, max_blocks_per_seq * sizeof(int), stream) ==
                cudaSuccess) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_block_tables_swa, swa_bt.data(),
                                               max_blocks_per_seq * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
        } else {
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
            return {};
        }
    }

    (void)executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_block_tables;
    state_template.block_tables_swa = d_block_tables_swa;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Recurrent state for SSM/GDN layers — pointers are constant for
    // single-sequence decode, so they're safe to bake into the graph.
    fill_recurrent_state(*req, state_template, /*reset=*/false, stream);

    // Upload banned tokens for graph-captured logit masking
    int32_t* d_banned = nullptr;
    if (!banned_token_ids_.empty()) {
        if (cudaMallocAsync(&d_banned, banned_token_ids_.size() * sizeof(int32_t), stream) == cudaSuccess) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_banned, banned_token_ids_.data(),
                                               banned_token_ids_.size() * sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream));
            state_template.d_banned_tokens = d_banned;
            state_template.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
        }
    }

    auto gcfg = build_graph_config(*req, remaining);

    CudaGraphConditionalRunner runner;
    if (!runner.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (d_banned)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
        if (d_block_tables_swa)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables_swa, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
        return {};
    }
    if (!runner.launch(stream)) {
        if (d_banned)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
        if (d_block_tables_swa)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables_swa, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
        return {};
    }

    auto tokens = runner.wait_and_get_tokens(stream);
    if (d_banned)
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
    if (d_block_tables_swa)
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables_swa, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
    IMP_LOG_INFO("ConditionalGraph: generated %zu tokens in graph loop", tokens.size());
    runner.cleanup();
    return tokens;
}

bool Engine::try_launch_async_graph_loop(std::shared_ptr<Request> req, int32_t first_token,
                                         cudaStream_t stream, int step_limit) {
    // Constrained requests (json_mode / json_schema / enforced tool call) can
    // NEVER run here: the loop samples device-side and applies no FSM mask.
    // The step_decode flags block them, but the spec-ngram burst hooks call
    // this directly — without this guard a tool-enforced request decoded a
    // full unmasked generation (#1002).
    if (req->constraints || req->json_mode || !req->json_schema.empty() ||
        !req->tool_constraint_tools.empty())
        return false;

    int remaining = prepare_graph_loop(req);
    if (remaining <= 0)
        return false;

    // Fast relaunch: a parked runner from a previous burst of the SAME
    // request keeps its captured graph — reseed device state instead of
    // recapturing (the burst-hybrid n-gram speculation path relaunches every
    // few tokens; a full setup costs ~10-20 ms per launch).
    // #683 postscript: the "rearm emits a wrong token" artifact was the
    // fresh-captured loop writing KV one slot too high (setup() position
    // off-by-one) — the correctly-positioned rearm then collided with the
    // shifted layout. Both paths share the eager first-forward semantics now.
    if (runtime_config_.speculative.burst_rearm && async_graph_runner_.is_setup() &&
        async_parked_req_id_ >= 0) {
        const auto& bt = kv_manager_->block_table(req->id);
        const int ctx = req->context_len();
        // SWA table must exist at matching length for a rearm (swa_prepare in
        // prepare_graph_loop already covered the new burst range).
        const auto& swa_bt = kv_manager_->swa_block_table(req->id);
        const bool swa_ok = !swa_sizing_active_ ||
                            (async_d_block_tables_swa_ != nullptr && swa_bt.size() == bt.size());
        if (async_parked_req_id_ == req->id && async_d_block_tables_ != nullptr && swa_ok &&
            static_cast<int>(bt.size()) <= async_bt_capacity_) {
            // Verify steps may have appended KV blocks since the park —
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
                think_limit = std::max(
                    1, static_cast<int>(req->max_tokens * req->think_budget) - used);
            }
            if (getenv("IMP_SPEC_TRACE"))
                IMP_LOG_INFO("[burst-launch] REARM seed=%d pos=%d ctx=%d limit=%d", (int)first_token,
                             ctx - 1, ctx, step_limit);
            if (async_graph_runner_.rearm(first_token, /*position=*/ctx - 1, /*context_len=*/ctx,
                                          step_limit, req->in_think_block, think_limit, stream) &&
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
        // upload failure) — tear down and rebuild below.
        async_graph_runner_.cleanup();
        if (async_d_block_tables_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
            async_d_block_tables_ = nullptr;
        }
        if (async_d_block_tables_swa_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_swa_));
            async_d_block_tables_swa_ = nullptr;
        }
        if (async_d_banned_tokens_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
            async_d_banned_tokens_ = nullptr;
        }
        async_parked_req_id_ = -1;
    }

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    int* d_bt = nullptr;
    if (cudaMalloc(&d_bt, max_blocks_per_seq * sizeof(int)) != cudaSuccess)
        return false;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_bt, full_bt.data(), max_blocks_per_seq * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
    int* d_bt_swa = nullptr;
    if (swa_sizing_active_) {
        const auto& swa_bt = kv_manager_->swa_block_table(req->id);
        if (static_cast<int>(swa_bt.size()) != max_blocks_per_seq ||
            cudaMalloc(&d_bt_swa, max_blocks_per_seq * sizeof(int)) != cudaSuccess) {
            IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
            return false;
        }
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_bt_swa, swa_bt.data(), max_blocks_per_seq * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    }

    (void)executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_bt;
    state_template.block_tables_swa = d_bt_swa;
    state_template.n_sequences = 1;
    state_template.max_blocks_per_seq = max_blocks_per_seq;
    state_template.is_prefill = false;

    // Recurrent state for SSM/GDN layers — pointers are constant for
    // single-sequence decode, so they're safe to bake into the graph.
    fill_recurrent_state(*req, state_template, /*reset=*/false, stream);

    // Upload banned tokens to device for graph-captured logit masking
    int32_t* d_banned = nullptr;
    if (!banned_token_ids_.empty()) {
        if (cudaMalloc(&d_banned, banned_token_ids_.size() * sizeof(int32_t)) == cudaSuccess) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_banned, banned_token_ids_.data(),
                                               banned_token_ids_.size() * sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream));
            state_template.d_banned_tokens = d_banned;
            state_template.n_d_banned_tokens = static_cast<int>(banned_token_ids_.size());
        }
    }

    if (getenv("IMP_SPEC_TRACE"))
        IMP_LOG_INFO("[burst-launch] FRESH seed=%d ctx=%d limit=%d remaining=%d", (int)first_token,
                     req->context_len(), step_limit, remaining);
    auto gcfg = build_graph_config(*req, remaining);
    gcfg.step_limit = step_limit;

    if (!async_graph_runner_.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (d_banned)
            IMP_CUDA_CHECK_LOG(cudaFree(d_banned));
        if (d_bt_swa)
            IMP_CUDA_CHECK_LOG(cudaFree(d_bt_swa));
        IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
        return false;
    }
    if (!async_graph_runner_.launch(stream)) {
        async_graph_runner_.cleanup();
        if (d_banned)
            IMP_CUDA_CHECK_LOG(cudaFree(d_banned));
        if (d_bt_swa)
            IMP_CUDA_CHECK_LOG(cudaFree(d_bt_swa));
        IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
        return false;
    }

    async_graph_req_ = req;
    async_d_block_tables_ = d_bt;
    async_d_block_tables_swa_ = d_bt_swa;
    async_d_banned_tokens_ = d_banned;
    async_bt_capacity_ = max_blocks_per_seq;
    async_parked_req_id_ = -1;
    IMP_LOG_DEBUG("AsyncGraphLoop: launched with %d banned tokens", state_template.n_d_banned_tokens);
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;
    IMP_LOG_INFO("AsyncGraphLoop: launched for %d remaining tokens", remaining);
    return true;
}

// =====================================================================
// Pipelined constrained decode (json_mode / json_schema, single seq).
//
// The conditional graph loop can't run constrained requests (the grammar
// FSM is host-side), and the eager path leaves the GPU idle during every
// host turnaround. This mode splits the step: per tick the host enqueues
// [banned+mask+sample+advance] for the forward already in flight AND the
// NEXT forward (a CudaGraphRunner replay reading the freshly sampled
// token from device memory), then waits only for the sampled token. The
// GPU is already deep in forward N+1 while the host updates the FSM for
// token N and computes the next mask.
// =====================================================================

bool Engine::try_launch_constrained_pipeline(std::shared_ptr<Request> req, cudaStream_t stream) {
    // SWA-aware sizing: the pipeline captures a decode graph with a baked
    // block-table pointer + jump-ahead chunk, neither wired for the per-step
    // SWA table rewrite. Fall back to eager constrained decode (step_decode,
    // which threads the SWA table) — correct, just not pipelined.
    if (swa_sizing_active_)
        return false;
    int budget = prepare_graph_loop(req);
    if (budget <= 0)
        return false;

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    auto& p = cpipe_;
    if (cudaMalloc(&p.d_bt, max_blocks_per_seq * sizeof(int)) != cudaSuccess)
        return false;
    bool ok = cudaMalloc(&p.d_token, SAMPLE_SCRATCH_BYTES) == cudaSuccess &&
              cudaMalloc(&p.d_pos, sizeof(int)) == cudaSuccess &&
              cudaMalloc(&p.d_ctx, sizeof(int)) == cudaSuccess &&
              cudaHostAlloc(&p.h_token, sizeof(int32_t), cudaHostAllocDefault) == cudaSuccess &&
              cudaEventCreateWithFlags(&p.ev, cudaEventDisableTiming) == cudaSuccess;
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

    if (!banned_token_ids_.empty()) {
        if (cudaMalloc(&p.d_banned, banned_token_ids_.size() * sizeof(int32_t)) == cudaSuccess) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_banned, banned_token_ids_.data(),
                                               banned_token_ids_.size() * sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream));
        }
    }

    // Decode workspace (mirrors step_decode_forward's single-seq path).
    // Without a dedicated decode workspace (no green contexts on sm_120),
    // jump-ahead chunks (#844) share THIS workspace with the captured
    // graph — pre-size it for the largest chunk NOW, before capture bakes
    // the buffer pointer, so no later resize can reallocate under the graph.
    if (executor_->has_decode_workspace())
        executor_->use_workspace(1);
    const int ws_tokens = (!executor_->has_decode_workspace() &&
                           runtime_config_.constrained.jump_ahead)
                              ? kJumpRowsCap
                              : 1;
    (void)executor_->resize_workspace(ws_tokens, stream);

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
    fill_recurrent_state(*req, st, /*reset=*/false, stream);
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
    // Constraint hooks — the request's manager was prepared at admission.
    st.schema_constrainer = req->constraints ? req->constraints->schema_constrainer() : nullptr;
    st.json_constrainer = req->constraints ? req->constraints->json_constrainer() : nullptr;

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
                 st.schema_constrainer ? "schema" : "json");
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
    // Jump-ahead span consumption (#844): while fnext > 0, the tick's
    // logits come from the drafted chunk's precomputed rows — no forward
    // runs, no device pos advance (repointed absolutely at span exit).
    const bool consuming = p.fnext > 0;

    // 1. Mask + sample this tick's logits: the in-flight forward's (normal
    //    tick, and the first span tick — the forward already in flight
    //    predicts the draft's first position), or a precomputed draft row.
    //    The constraint mask is host-computed from the FSM state after the
    //    last harvested token and uploaded stream-ordered behind the
    //    producer of those logits.
    p.state.seed = engine_internal::compute_step_seed(*req);
    // Think-budget enforcement (mirrors fill_sampling_params): when the
    // reasoning budget is exhausted mid-think, force </think> this step.
    p.state.force_token = -1;
    if (think_logic::should_force_think_end(req->think_budget, think_end_id_, req->max_tokens,
                                            req->output_tokens, think_start_id_, req->started_in_think)) {
        p.state.force_token = think_end_id_;
    }
    // Token-history penalties — per-tick upload, exactly like the eager path.
    p.state.penalty_tokens = nullptr;
    p.state.n_penalty_tokens = 0;
    upload_penalties(*req, p.state, stream);
    Tensor row_logits;
    const Tensor* tick_logits = &p.logits;
    if (consuming && p.fnext >= 2) {
        const int64_t vocab = model_->config().vocab_size;
        int64_t shape[2] = {1, vocab};
        row_logits = Tensor(p.d_frows + static_cast<size_t>(p.fnext - 2) * vocab, QType::F32, 2,
                            shape, /*borrowed=*/true);
        tick_logits = &row_logits;
    }
    executor_->masked_sample_async(p.state, *tick_logits, p.d_token, p.h_token, stream);
    if (!consuming)
        launch_pipeline_advance(p.d_pos, p.d_ctx, stream);
    IMP_CUDA_CHECK_LOG(cudaEventRecord(p.ev, stream));

    // 2. Enqueue the NEXT forward before the host knows the token — it reads
    //    d_token (just written by the sampler) on the GPU timeline. While
    //    consuming a drafted span, the chunk already covers the positions
    //    ahead — no forward until the span exits.
    bool more = (p.produced + 1 < p.budget) &&
                (static_cast<int>(req->output_tokens.size()) + 1 < req->max_tokens);
    if (more && !consuming)
        p.runner.execute(stream);
    p.forward_in_flight = more && !consuming;

    // 3. Wait only for the sampled token (GPU continues in forward N+1).
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(p.ev));
    int32_t token = *p.h_token;

    // 4. Harvest — mirrors step_decode_process_outputs for one token.
    req->output_tokens.push_back(token);
    p.produced++;
    if (consuming && p.fnext >= 2)
        p.jumped_tokens++;  // sampled from a precomputed row — no forward ran
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
        // KV budget exhausted — drain and let the eager path continue.
        teardown_constrained_pipeline(/*synchronize=*/true);
        return 0;
    }

    if (consuming) {
        if (p.fnext <= p.frows && token == p.fdraft[p.fnext - 1]) {
            // On draft: this token's KV came from the chunk and the row
            // predicting the next position is already materialized — the
            // next tick stays forward-free.
            p.fnext++;
            return 1;
        }
        if (getenv("IMP_JUMP_TRACE")) {
            Tokenizer* tk = model_->tokenizer();
            const int32_t want = p.fnext <= p.frows ? p.fdraft[p.fnext - 1] : -1;
            IMP_LOG_INFO("[jump] exit at %d/%d: sampled %d [%s] draft %d [%s]", p.fnext, p.frows,
                         token, tk ? tk->decode_token(token).c_str() : "?", want,
                         want >= 0 && tk ? tk->decode_token(want).c_str() : "<bonus>");
        }
        // Span exit: the sampled token diverged from the draft (its KV slot
        // holds the drafted token's KV) or it is the free token after a
        // fully-matched span (no KV yet either way). Repoint the pipeline
        // and replay — the forward rewrites the correct KV and produces the
        // next tick's logits. Stale draft KV beyond d_ctx is never read.
        const int pos = p.fbase_pos + p.fnext - 1;
        const int ctx = pos + 1;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_token, &token, sizeof(int32_t),
                                           cudaMemcpyHostToDevice, stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice,
                                           stream));
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(p.d_ctx, &ctx, sizeof(int), cudaMemcpyHostToDevice,
                                           stream));
        p.runner.execute(stream);
        p.forward_in_flight = true;
        p.fdraft.clear();
        p.fnext = 0;
        p.frows = 0;
        return 1;
    }

    // Jump-ahead (#844), two-step: a pending draft is confirmed token by
    // token for free (these ticks' forwards run regardless); after
    // kJumpFreeVerify matches the speculative chunk over the remainder is
    // committed. A diverging token drops the draft at zero GPU cost. Then
    // probe the FSM for the next forced span.
    if (!p.fpending.empty()) {
        if (token == p.fpending[p.fpend_cursor]) {
            if (++p.fpend_cursor >= kJumpFreeVerify) {
                constrained_jump_commit_(req, stream);
                p.fpending.clear();
                p.fpend_cursor = 0;
            }
            return 1;
        }
        if (getenv("IMP_JUMP_TRACE")) {
            Tokenizer* tk = model_->tokenizer();
            IMP_LOG_INFO("[jump] pending dropped at %d: sampled %d [%s] draft %d [%s]",
                         p.fpend_cursor, token,
                         tk ? tk->decode_token(token).c_str() : "?", p.fpending[p.fpend_cursor],
                         tk ? tk->decode_token(p.fpending[p.fpend_cursor]).c_str() : "?");
        }
        p.fpending.clear();
        p.fpend_cursor = 0;
    }
    constrained_jump_probe_(req);
    return 1;
}

// Jump-ahead (#844) probe: derive the forced continuation TEXT from the
// schema FSM (chars every legal completion must spell — token-level forcing
// is vacuous on BPE vocabs, where ':' / ':"' / ':{"' all spell the same
// skeleton) and pend its canonical tokenization. Host-only; the chunk is
// only committed after the next tick confirms the draft's first token for
// free (see step_constrained_pipeline).
void Engine::constrained_jump_probe_(std::shared_ptr<Request>& req) {
    auto& p = cpipe_;
    const auto& ccfg = runtime_config_.constrained;
    if (!ccfg.jump_ahead || !req->constraints || !req->constraints->has_schema())
        return;
    // v1 gates: the chunk needs continuation prefill; recurrent state
    // (SSM/GDN) advances per forwarded token and has no chunk-continuation
    // wiring here (spec-ngram gates it out for the same reason).
    if (ssm_state_ || !supports_chunked_prefill_())
        return;
    // Think-budget forcing owns the next step — don't compete with it.
    if (p.state.force_token >= 0)
        return;

    // Probe the forced continuation (pure FSM walk). ~96 chars cover any
    // realistic skeleton span; longer ones re-probe after this span.
    std::string text;
    if (req->constraints->forced_text(text, 96) <= 0)
        return;
    Tokenizer* tok = model_->tokenizer();
    if (!tok)
        return;
    std::vector<int32_t> draft = tok->encode(text, /*no_prefix=*/true);
    // The first kJumpFreeVerify tokens are confirmed for free; the chunk
    // needs >=2 more tokens to pay for itself, and jump_min_run is the
    // quality knob on top.
    if (static_cast<int>(draft.size()) > kJumpRowsCap + kJumpFreeVerify)
        draft.resize(kJumpRowsCap + kJumpFreeVerify);
    if (static_cast<int>(draft.size()) <
        std::max(kJumpFreeVerify + 2, ccfg.jump_min_run))
        return;
    p.fpending = std::move(draft);
    p.fpend_cursor = 0;
    if (getenv("IMP_JUMP_TRACE"))
        IMP_LOG_INFO("[jump] pended %d tokens for forced span \"%s\"",
                     static_cast<int>(p.fpending.size()), text.c_str());
}

// Jump-ahead (#844) commit: the pipeline just confirmed the draft's first
// kJumpFreeVerify tokens for free, and the in-flight forward for the last
// of them will produce the logits predicting the next draft position. Run
// ONE speculative teacher-forced chunk over the rest of the pending draft:
// KV for every draft position plus
// a materialized logits row predicting each following position. Later ticks
// masked-sample from those rows (see step_constrained_pipeline) instead of
// running forwards. Stream-ordered behind the in-flight forward; pure on
// any failure — nothing emitted, no FSM advance.
void Engine::constrained_jump_commit_(std::shared_ptr<Request>& req, cudaStream_t stream) {
    auto& p = cpipe_;

    std::vector<int32_t> draft(p.fpending.begin() + kJumpFreeVerify, p.fpending.end());

    // Bounds: row-buffer cap, KV budget (>=1 token stays for the pipeline
    // after the span), max_tokens, workspace token limit (resize_workspace
    // silently clamps there), attn-scores capacity.
    const int out_size = static_cast<int>(req->output_tokens.size());
    int max_k = std::min({kJumpRowsCap, p.budget - p.produced - 1,
                          req->max_tokens - out_size - 1, executor_->max_tokens()});
    const int pos1 = req->context_len() - 1;  // position of the last confirmed token
    const int s_cap = executor_->attn_scores_cap();
    if (s_cap > 0) {
        // Same rule as chunked prefill: n_tokens x ctx_len must fit s_cap^2.
        const int64_t cap2 = static_cast<int64_t>(s_cap) * s_cap;
        while (max_k > 0 && static_cast<int64_t>(max_k) * (pos1 + max_k + 1) > cap2)
            --max_k;
    }
    if (static_cast<int>(draft.size()) > max_k)
        draft.resize(std::max(max_k, 0));
    const int K = static_cast<int>(draft.size());
    if (K < 1)
        return;

    if (!ensure_spec_buffers_(K, 1))
        return;
    if (!p.d_frows) {
        const size_t bytes = static_cast<size_t>(kJumpRowsCap) *
                             model_->config().vocab_size * sizeof(float);
        if (cudaMalloc(&p.d_frows, bytes) != cudaSuccess) {
            p.d_frows = nullptr;
            return;
        }
    }
    // With a dedicated decode workspace the chunk runs on the prefill slot
    // (the captured graph's buffers are untouched); without one it shares
    // the graph's workspace — pre-sized to kJumpRowsCap at launch, so this
    // resize can only shrink bookkeeping, never reallocate under the graph.
    const bool dual_ws = executor_->has_decode_workspace();
    if (dual_ws)
        executor_->use_workspace(0);
    if (!executor_->resize_workspace(K, stream)) {
        if (dual_ws)
            executor_->use_workspace(1);
        return;
    }

    // Teacher-forced chunk over the draft at positions pos1+1..pos1+K,
    // stream-ordered behind the in-flight forward (which contributes the
    // harvested token's KV at pos1).
    std::vector<int> h_positions(K);
    for (int i = 0; i < K; ++i)
        h_positions[i] = pos1 + 1 + i;
    const int chunk_ctx = pos1 + K + 1;  // KV valid through the chunk's last position
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_spec_tokens_, draft.data(), K * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_spec_positions_, h_positions.data(), K * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_spec_context_len_, &chunk_ctx, sizeof(int),
                                       cudaMemcpyHostToDevice, stream));

    InferenceState cs;
    cs.token_ids = d_spec_tokens_;
    cs.positions = d_spec_positions_;
    cs.n_tokens = K;
    cs.kv_cache = kv_cache_raw_;
    cs.block_tables = p.d_bt;  // full table uploaded at launch, covers the whole budget
    cs.context_lens = d_spec_context_len_;
    cs.max_context_len = chunk_ctx;
    cs.n_sequences = 1;
    cs.max_blocks_per_seq = 0;
    cs.is_prefill = true;
    cs.prefill_offset = pos1 + 1;
    cs.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled())
        cs.kv_seq_id = req->id;

    Tensor chunk_logits;
    executor_->forward_logits(cs, chunk_logits, stream);
    // Materialize the per-position logits rows while the workspace that ran
    // the chunk (hidden_) is still active; row j predicts position pos1+2+j.
    executor_->project_logits_all(K, p.d_frows, stream);
    if (dual_ws)
        executor_->use_workspace(1);

    p.fdraft = std::move(draft);
    p.fnext = 1;  // the in-flight forward's logits predict fdraft[0]'s position
    p.fbase_pos = pos1 + 1;
    p.frows = K;
    p.jumps++;
    if (getenv("IMP_JUMP_TRACE")) {
        std::string ids;
        for (int32_t t : p.fdraft) ids += std::to_string(t) + " ";
        IMP_LOG_INFO("[jump] committed %d-token chunk after free first-token match (ids: %s)", K,
                     ids.c_str());
    }
    IMP_LOG_DEBUG("ConstrainedPipeline: jump-ahead committed a %d-token draft chunk", K);
}

void Engine::teardown_constrained_pipeline(bool synchronize) {
    auto& p = cpipe_;
    if (p.jumps > 0) {
        IMP_LOG_INFO("ConstrainedPipeline: jump-ahead saved %lld forwards across %d drafted "
                     "spans (%d tokens produced)",
                     p.jumped_tokens, p.jumps, p.produced);
    }
    if (synchronize)
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(decode_stream()));
    p.runner.invalidate();
    if (p.d_bt) { IMP_CUDA_CHECK_LOG(cudaFree(p.d_bt)); p.d_bt = nullptr; }
    if (p.d_token) { IMP_CUDA_CHECK_LOG(cudaFree(p.d_token)); p.d_token = nullptr; }
    if (p.d_pos) { IMP_CUDA_CHECK_LOG(cudaFree(p.d_pos)); p.d_pos = nullptr; }
    if (p.d_ctx) { IMP_CUDA_CHECK_LOG(cudaFree(p.d_ctx)); p.d_ctx = nullptr; }
    if (p.d_banned) { IMP_CUDA_CHECK_LOG(cudaFree(p.d_banned)); p.d_banned = nullptr; }
    if (p.h_token) { IMP_CUDA_CHECK_LOG(cudaFreeHost(p.h_token)); p.h_token = nullptr; }
    if (p.ev) { IMP_CUDA_CHECK_LOG(cudaEventDestroy(p.ev)); p.ev = nullptr; }
    if (p.d_frows) { IMP_CUDA_CHECK_LOG(cudaFree(p.d_frows)); p.d_frows = nullptr; }
    p.req = nullptr;
    p.active = false;
    p.forward_in_flight = false;
    p.produced = 0;
    p.budget = 0;
    p.fdraft.clear();
    p.fpending.clear();
    p.fpend_cursor = 0;
    p.fnext = 0;
    p.fbase_pos = 0;
    p.frows = 0;
    p.jumps = 0;
    p.jumped_tokens = 0;
}

}  // namespace imp
