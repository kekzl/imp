#include "runtime/engine.h"
#include "runtime/cuda_graph.h"
#include "runtime/engine_internal.h"
#include "runtime/think_stop_logic.h"
#include "memory/kv_cache.h"
#include "model/chat_template.h"
#include "compute/sampling.h"
#include "core/logging.h"

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

    // Burst-hybrid n-gram speculation: a given-up request runs the loop in
    // bounded bursts so the host can re-probe for drafts in between (see
    // Engine::spec_maybe_rearm_).
    if (runtime_config_.speculative.ngram && req->spec_ngram_given_up &&
        runtime_config_.speculative.burst > 0)
        remaining = std::min(remaining, runtime_config_.speculative.burst);

    constexpr int kMaxLayersForConditionalGraph = 128;
    if (model_->config().n_layers > kMaxLayersForConditionalGraph)
        return 0;

    {
        size_t f = 0, t = 0;
        cudaMemGetInfo(&f, &t);
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
    return std::min(capped, remaining);
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
        if (req.think_budget > 0.0f)
            gcfg.think_budget_limit = static_cast<int>(req.max_tokens * req.think_budget);
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

    (void)executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_block_tables;
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
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
        return {};
    }
    if (!runner.launch(stream)) {
        if (d_banned)
            IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
        return {};
    }

    auto tokens = runner.wait_and_get_tokens(stream);
    if (d_banned)
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_banned, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
    IMP_LOG_INFO("ConditionalGraph: generated %zu tokens in graph loop", tokens.size());
    runner.cleanup();
    return tokens;
}

bool Engine::try_launch_async_graph_loop(std::shared_ptr<Request> req, int32_t first_token,
                                         cudaStream_t stream) {
    int remaining = prepare_graph_loop(req);
    if (remaining <= 0)
        return false;

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    int* d_bt = nullptr;
    if (cudaMalloc(&d_bt, max_blocks_per_seq * sizeof(int)) != cudaSuccess)
        return false;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_bt, full_bt.data(), max_blocks_per_seq * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));

    (void)executor_->resize_workspace(1, stream);

    InferenceState state_template;
    state_template.kv_cache = kv_cache_raw_;
    state_template.block_tables = d_bt;
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

    auto gcfg = build_graph_config(*req, remaining);

    if (!async_graph_runner_.setup(executor_.get(), state_template, first_token, gcfg, stream)) {
        if (d_banned)
            IMP_CUDA_CHECK_LOG(cudaFree(d_banned));
        IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
        return false;
    }
    if (!async_graph_runner_.launch(stream)) {
        async_graph_runner_.cleanup();
        if (d_banned)
            IMP_CUDA_CHECK_LOG(cudaFree(d_banned));
        IMP_CUDA_CHECK_LOG(cudaFree(d_bt));
        return false;
    }

    async_graph_req_ = req;
    async_d_block_tables_ = d_bt;
    async_d_banned_tokens_ = d_banned;
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
    int budget = prepare_graph_loop(req);
    if (budget <= 0)
        return false;

    const auto& full_bt = kv_manager_->block_table(req->id);
    int max_blocks_per_seq = static_cast<int>(full_bt.size());

    auto& p = cpipe_;
    if (cudaMalloc(&p.d_bt, max_blocks_per_seq * sizeof(int)) != cudaSuccess)
        return false;
    bool ok = cudaMalloc(&p.d_token, ARGMAX_SCRATCH_BYTES) == cudaSuccess &&
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
    if (executor_->has_decode_workspace())
        executor_->use_workspace(1);
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
    // Constraint hooks — the engine-level manager was prepared at admission.
    st.schema_constrainer = constraints_.schema_constrainer();
    st.json_constrainer = constraints_.json_constrainer();

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

    // 1. Mask + sample the in-flight forward's logits. The constraint mask is
    //    host-computed from the FSM state after the last harvested token and
    //    uploaded stream-ordered behind the forward.
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
    executor_->masked_sample_async(p.state, p.logits, p.d_token, p.h_token, stream);
    launch_pipeline_advance(p.d_pos, p.d_ctx, stream);
    IMP_CUDA_CHECK_LOG(cudaEventRecord(p.ev, stream));

    // 2. Enqueue the NEXT forward before the host knows the token — it reads
    //    d_token (just written by the sampler) on the GPU timeline.
    bool more = (p.produced + 1 < p.budget) &&
                (static_cast<int>(req->output_tokens.size()) + 1 < req->max_tokens);
    if (more)
        p.runner.execute(stream);
    p.forward_in_flight = more;

    // 3. Wait only for the sampled token (GPU continues in forward N+1).
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(p.ev));
    int32_t token = *p.h_token;

    // 4. Harvest — mirrors step_decode_process_outputs for one token.
    req->output_tokens.push_back(token);
    p.produced++;
    track_think_state(*req, token);
    constraints_.update(token);
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
    return 1;
}

void Engine::teardown_constrained_pipeline(bool synchronize) {
    auto& p = cpipe_;
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
    p.req = nullptr;
    p.active = false;
    p.forward_in_flight = false;
    p.produced = 0;
    p.budget = 0;
}

}  // namespace imp
