#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/vram_budget.h"
#include "runtime/batch.h"
#include "runtime/mtp_forward.h"
#include "memory/kv_cache.h"
#include "model/gguf_loader.h"
#include "model/chat_template.h"
#include "compute/ffn_sparsity_probe.h"
#include "compute/gemm.h"
#include "compute/gemm_capture_fp16_sm120.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/attention_cublas.h"
#include "compute/gemm_grouped.h"
#include "compute/sampling.h"
#include "compute/attention.h"
#include "compute/layernorm.h"
#include "core/logging.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <functional>
#include <vector>

namespace imp {

// =====================================================================
// File-local helpers (pure refactoring — no behavior changes)
// =====================================================================
namespace {

// Free prefill metadata buffers when not using the pre-allocated pool.
void free_prefill_buffers(int32_t* d_token_ids, int* d_positions, int* d_block_tables, int* d_context_lens,
                          cudaStream_t stream) {
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, stream));
}

// Compute a deterministic-but-varying seed for each decode step.
// Mixes the request seed (or a hash of the request ID + clock) with
// the current output token count so each step gets a unique RNG draw.
int compute_step_seed(const Request& req) {
    int base_seed = req.seed >= 0
                        ? req.seed
                        : static_cast<int>(std::hash<int>{}(req.id) ^
                                           std::chrono::steady_clock::now().time_since_epoch().count());
    int step = static_cast<int>(req.output_tokens.size());
    return base_seed + step;
}

// Build a TokenLogprobInfo from raw logits on the host.
TokenLogprobInfo build_logprob_info(const float* h_logits, int vocab_size, int32_t sampled_token,
                                    int top_logprobs, Tokenizer* tok) {
    LogprobResult lp_result;
    compute_logprobs_cpu(h_logits, vocab_size, sampled_token, top_logprobs, &lp_result);

    TokenLogprobInfo info;
    info.logprob = lp_result.sampled_logprob;
    info.text = tok->decode_token(sampled_token);
    info.top.reserve(lp_result.top.size());
    for (const auto& [tid, tlp] : lp_result.top) {
        info.top.push_back({tid, tlp, tok->decode_token(tid)});
    }
    return info;
}

// Ensure workspace 0 is active (used before prefill and after decode).
void ensure_prefill_workspace(GraphExecutor* executor) {
    if (executor->has_decode_workspace() && executor->active_workspace() != 0) {
        executor->use_workspace(0);
    }
}

}  // anonymous namespace

Engine::~Engine() {
    // Save prefix cache to disk before shutdown
    if (kv_manager_ && !config_.prefix_cache_path.empty() && kv_manager_->prefix_caching_enabled()) {
        kv_manager_->save_prefix_cache(config_.prefix_cache_path, stream_);
    }

    // FFN sparsity probe (Vector 1 research instrumentation): drain per-layer
    // counters to stderr if any decode steps ran with the probe enabled.
    flush_ffn_sparsity_probe_log();

    gemm_cleanup();
    gemm_grouped_cleanup();
    sampling_cleanup();
    if (async_graph_runner_.is_setup()) {
        async_graph_runner_.cleanup();
    }
    if (async_d_block_tables_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
        async_d_block_tables_ = nullptr;
    }
    if (async_d_banned_tokens_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
        async_d_banned_tokens_ = nullptr;
    }
    if (d_penalty_tokens_) {
        vram_alloc_.free(d_penalty_tokens_);
        d_penalty_tokens_ = nullptr;
    }
    if (d_kv_slot_buf_) {
        cudaFree(d_kv_slot_buf_);
        d_kv_slot_buf_ = nullptr;
    }
    if (h_sample_pinned_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_sample_pinned_));
        h_sample_pinned_ = nullptr;
    }
    if (prefill_pool_) {
        vram_alloc_.free(prefill_pool_);
        prefill_pool_ = nullptr;
    }
    if (h_pf_positions_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_pf_positions_));
        h_pf_positions_ = nullptr;
    }
    if (h_pf_token_ids_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_pf_token_ids_));
        h_pf_token_ids_ = nullptr;
    }
    // MTP spec-decode workspace cleanup
    if (mtp_ws_storage_) {
        auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
        imp::mtp_workspace_free(*ws);
        delete ws;
        mtp_ws_storage_ = nullptr;
        mtp_spec_k_ = 0;
    }
    // stream_, prefill_done_, decode_done_ cleaned up by CudaStream/CudaEvent RAII
    // vision_ cleaned up by VisionPipeline RAII
}

// ── MTP spec-decode API (Phase 3 scaffolding) ─────────────────────────
bool Engine::enable_mtp_spec_decode(int k) {
    if (k <= 0) {
        IMP_LOG_ERROR("enable_mtp_spec_decode: k must be > 0 (got %d)", k);
        return false;
    }
    if (!model_) {
        IMP_LOG_ERROR("enable_mtp_spec_decode: no model loaded");
        return false;
    }
    if (!model_->mtp_.has_value() || !model_->mtp_->loaded) {
        IMP_LOG_ERROR("enable_mtp_spec_decode: model has no MTP head loaded");
        return false;
    }
    if (mtp_ws_storage_ != nullptr) {
        IMP_LOG_WARN("enable_mtp_spec_decode: already enabled, k=%d -> %d", mtp_spec_k_, k);
        mtp_spec_k_ = k;
        return true;
    }
    const int hidden_dim   = model_->config_.d_model;
    const int vocab_size   = model_->config_.vocab_size;
    const int n_experts    = model_->config_.n_experts;
    const int top_k        = model_->config_.n_experts_active;
    const int expert_d_ff  = model_->config_.expert_d_ff;
    const int shared_d_ff  = model_->config_.expert_shared_d_ff;

    // MTP attention dims: derived from the MTP head's q_proj / v_proj shapes
    // because the MTP attention head config differs from the main model
    // (Qwen3.6 MTP doubles Q output per-head for attn_output_gate).
    // q_proj shape [2 * num_heads * head_dim, hidden_dim]; v_proj shape
    // [num_kv_heads * head_dim, hidden_dim]. We use main model's head_dim
    // as the per-head attention dim and back-compute the MTP head counts.
    int mtp_num_heads = 0, mtp_num_kv_heads = 0, mtp_head_dim = 0;
    if (model_->mtp_.has_value() && model_->mtp_->loaded &&
        model_->mtp_->q_proj.data != nullptr && model_->mtp_->v_proj.data != nullptr) {
        const int q_out = static_cast<int>(model_->mtp_->q_proj.shape[0]);
        const int v_out = static_cast<int>(model_->mtp_->v_proj.shape[0]);
        mtp_head_dim     = model_->config_.head_dim;
        if (mtp_head_dim > 0) {
            // q_proj outputs 2 × num_heads × head_dim (attn_output_gate=True).
            mtp_num_heads    = q_out / (2 * mtp_head_dim);
            mtp_num_kv_heads = v_out / mtp_head_dim;
        }
    }

    // MTP KV-cache capacity: cap at the smaller of model's max_seq_len and 16K
    // (Phase 2.2.Attn+KV budget — ~16 MiB each for K and V at Qwen3.6 dims).
    constexpr int kMtpKvCap = 16384;
    int mtp_kv_max = std::min(model_->config_.max_seq_len, kMtpKvCap);
    if (mtp_kv_max <= 0) mtp_kv_max = kMtpKvCap;

    auto* ws = new imp::MtpDraftWorkspace();
    if (!imp::mtp_workspace_allocate(*ws, hidden_dim, vocab_size,
                                      n_experts, top_k, expert_d_ff, shared_d_ff,
                                      mtp_num_heads, mtp_num_kv_heads, mtp_head_dim,
                                      mtp_kv_max)) {
        delete ws;
        IMP_LOG_ERROR("enable_mtp_spec_decode: workspace alloc failed");
        return false;
    }
    // Configure RoPE for the MTP attention (Phase 2.2.Attn+RoPE).
    // Qwen3.5/3.6 uses partial rope (factor 0.25 → rope_dim=64 of head_dim=256),
    // theta from config (10M for long-context), NeoX-style.
    ws->rope_theta       = model_->config_.rope_theta;
    ws->rope_neox        = model_->config_.rope_neox;
    ws->rms_norm_eps     = model_->config_.rms_norm_eps;
    ws->rope_dim         = (model_->config_.rope_dim > 0) ? model_->config_.rope_dim : mtp_head_dim;
    // mrope section split. Qwen3.6 ships mrope_section = [11, 11, 10]
    // (half-counts; full rope_dim = 64 = 2*(11+11+10)). imp doesn't load
    // this from config yet — hardcoded here based on the on-disk spec.
    // For text-only generation all 3 positions are equal, so this is
    // mathematically equivalent to standard partial-rope; the section
    // split matters only for true multimodal tokens.
    if (ws->rope_dim == 64) {
        ws->mrope_sec0 = 11;
        ws->mrope_sec1 = 11;
        ws->mrope_sec2 = 10;
    } else {
        // Fall back to even-split: all of rope_dim/2 in section 0.
        ws->mrope_sec0 = ws->rope_dim / 2;
        ws->mrope_sec1 = 0;
        ws->mrope_sec2 = 0;
    }
    // Diagnostic: generation.mtp_no_rope (legacy IMP_MTP_NO_ROPE=1) disables RoPE entirely.
    if (RuntimeConfig::current().generation.mtp_no_rope) {
        ws->rope_dim = 0;
    }
    // Runtime weight_offset matches what the main model's rmsnorm calls pass:
    // norm_weight_offset from ModelConfig. For Qwen3.5/3.6 this is 0.0 because
    // the +1 (gamma = 1 + W) was already baked in during weight upload (see
    // upload_mtp_weights in weight_upload.cu). For Gemma-3 it's 1.0. Don't
    // double-apply.
    ws->arch_norm_offset = model_->config_.norm_weight_offset;

    mtp_ws_storage_ = ws;
    mtp_spec_k_ = k;
    IMP_LOG_INFO("MTP spec-decode enabled (k=%d, hidden=%d, vocab=%d, experts=%d/top%d, d_ff_e=%d, "
                 "d_ff_shared=%d, num_heads=%d/%d, head_dim=%d, kv_cap=%d, rope=%g/%d/%s, "
                 "mrope=[%d,%d,%d])",
                 k, hidden_dim, vocab_size, n_experts, top_k, expert_d_ff, shared_d_ff,
                 mtp_num_heads, mtp_num_kv_heads, mtp_head_dim, mtp_kv_max,
                 ws->rope_theta, ws->rope_dim, ws->rope_neox ? "neox" : "interleaved",
                 ws->mrope_sec0, ws->mrope_sec1, ws->mrope_sec2);
    return true;
}

bool Engine::mtp_prefill_prompt(const int32_t* prompt_tokens, const void* d_hidden, int n) {
    if (mtp_ws_storage_ == nullptr) return false;
    if (!model_ || !model_->mtp_.has_value() || !model_->mtp_->loaded) return false;
    if (n <= 0 || prompt_tokens == nullptr || d_hidden == nullptr) return false;

    auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
    if (ws->mtp_pos > 0) {
        IMP_LOG_WARN("mtp_prefill_prompt: cache already populated (pos=%d); skipping", ws->mtp_pos);
        return false;
    }

    const int hidden_dim = model_->config_.d_model;
    const int vocab_size = model_->config_.vocab_size;
    const __half* hidden = static_cast<const __half*>(d_hidden);

    // For each prompt position i in [0, n): run MTP forward with prev_token =
    // prompt_tokens[i] and d_h_prev = hidden[i]. Each call appends one row to
    // the MTP KV cache and advances mtp_pos. The last position's prediction
    // (i.e., what MTP thinks comes AFTER the prompt) becomes the pending
    // prediction for accuracy measurement on the first decode step.
    int last_prediction = -1;
    for (int i = 0; i < n; ++i) {
        const void* h_i = hidden + static_cast<int64_t>(i) * hidden_dim;
        int prediction = -1;
        bool ok = imp::mtp_draft_step(
            prompt_tokens[i],
            h_i,
            *model_->mtp_,
            model_->tok_emb_,
            model_->out_proj_,
            *ws,
            hidden_dim, vocab_size,
            &prediction,
            decode_stream());
        if (!ok) {
            IMP_LOG_WARN("mtp_prefill_prompt: forward failed at position %d/%d — abandoning", i, n);
            return false;
        }
        last_prediction = prediction;
    }

    mtp_pending_prediction_ = last_prediction;
    IMP_LOG_INFO("MTP prefill: %d prompt positions cached (mtp_pos=%d), pending prediction=%d",
                 n, ws->mtp_pos, last_prediction);
    return true;
}

void Engine::mtp_accuracy_reset() noexcept {
    mtp_accuracy_ = {};
    mtp_pending_prediction_ = -1;
    mtp_pending_chain_.clear();
    mtp_chain_accept_.clear();
    if (mtp_ws_storage_) {
        auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
        imp::mtp_kv_reset(*ws);
    }
}

bool Engine::mtp_draft_one(int prev_token_id, const void* d_h_prev,
                           int hidden_dim, int vocab_size, int* out_token_id) {
    if (mtp_ws_storage_ == nullptr) {
        IMP_LOG_ERROR("mtp_draft_one: spec-decode not enabled");
        return false;
    }
    if (!model_ || !model_->mtp_.has_value() || !model_->mtp_->loaded) {
        IMP_LOG_ERROR("mtp_draft_one: MTP head not loaded");
        return false;
    }
    auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
    return imp::mtp_draft_step(prev_token_id, d_h_prev, *model_->mtp_,
                                model_->tok_emb_, model_->out_proj_,
                                *ws, hidden_dim, vocab_size, out_token_id,
                                decode_stream());
}

// =====================================================================
// Helper methods
// =====================================================================

cudaStream_t Engine::prefill_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available()) ? green_ctx_.prefill_stream() : stream_;
}

cudaStream_t Engine::decode_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available()) ? green_ctx_.decode_stream() : stream_;
}

void Engine::reset_ssm_state(int seq_id) {
    if (ssm_state_) {
        ssm_state_->reset_sequence(seq_id % ssm_state_->max_sequences(), stream_);
    }
}

void Engine::reset_batch_pool_cache() { decode_batch_pool_.reset_upload_cache(); }

void Engine::invalidate_graphs() {
    // Preserve decode_graph_pool_ across context resets — the decode step
    // topology (forward_logits) doesn't change between requests. Inputs
    // (token IDs, positions, block tables) are uploaded fresh each step via
    // the batch pool. Per-entry invalidation already handles max_blocks_per_seq
    // changes in step_decode_forward(). Re-capturing on every benchmark rep
    // adds ~100ms overhead per reset.
    //
    // The conditional graph runner MUST be invalidated: it captures the full
    // decode loop including token feedback, stop conditions, and request-specific
    // KV block pointers.
    if (async_graph_runner_.is_setup()) {
        async_graph_runner_.cleanup();
    }
    async_graph_req_ = nullptr;
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;
}

size_t Engine::effective_free_vram() const {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return 0;
    }
    if (config_.vram_budget_mb > 0) {
        size_t budget = config_.vram_budget_mb * 1024ULL * 1024;
        size_t used = total_mem - free_mem;
        free_mem = (budget > used) ? (budget - used) : 0;
    }
    return free_mem;
}

bool Engine::is_stop_token(int32_t token) const {
    Tokenizer* tok = model_->tokenizer();
    if (tok && tok->is_eos(token))
        return true;
    for (int32_t stop_id : chat_template_.stop_token_ids()) {
        if (token == stop_id)
            return true;
    }
    // Banned tokens (e.g. <pad>) should also trigger stop — they indicate
    // the model has degenerated and continuing would produce garbage.
    for (int32_t bid : banned_token_ids_) {
        if (token == bid)
            return true;
    }
    return false;
}

void Engine::track_think_state(Request& req, int32_t token) const {
    // Fast path: single-token control IDs (GGUF metadata, or tokenizers that
    // promote <think>/</think> to special tokens).
    if (token == think_start_id_) {
        req.in_think_block = true;
        return;
    }
    if (token == think_end_id_) {
        req.in_think_block = false;
        req.think_exit_idx = static_cast<int>(req.output_tokens.size());
        return;
    }

    // Text-based fallback: NVFP4 SafeTensors loaders (Qwen3.6, Qwen3-Coder)
    // ship <think>/</think> as added_tokens with `special=False`. think_*_id_
    // stay -1 in that case, and the model emits </think> as a 3-token BPE
    // sequence ['</', 'think', '>'] which the single-id compare above can
    // never see. Append the decoded piece to a sliding window and match the
    // literal string. Without this, a model that has been chat-template-
    // primed with `<think>\n` (Qwen3.6 add_generation_prompt default) closes
    // its empty thinking block and the next sampled token (typically im_end)
    // hits should_stop with in_think_block=false → 0-content completion.
    Tokenizer* ptok = model_ ? model_->tokenizer() : nullptr;
    if (!ptok)
        return;
    const std::string piece = ptok->decode_token(token);
    if (piece.empty())
        return;
    req.think_text_tail += piece;
    constexpr size_t kThinkTailWindow = 32;
    if (req.think_text_tail.size() > kThinkTailWindow) {
        req.think_text_tail.erase(0, req.think_text_tail.size() - kThinkTailWindow);
    }
    if (req.in_think_block) {
        if (req.think_text_tail.find("</think>") != std::string::npos) {
            req.in_think_block = false;
            req.think_exit_idx = static_cast<int>(req.output_tokens.size());
            req.think_text_tail.clear();
        }
    } else {
        if (req.think_text_tail.find("<think>") != std::string::npos &&
            req.think_text_tail.find("</think>") == std::string::npos) {
            req.in_think_block = true;
            req.think_text_tail.clear();
        }
    }
}

bool Engine::should_stop(Request& req, int32_t token) const {
    if (req.ignore_eos)
        return false;
    // Inside <think>...</think>: suppress stop tokens so reasoning can complete.
    // The model may generate <|im_end|> during reasoning as part of its internal
    // monologue — stopping here produces empty content (llama.cpp ignores this).
    if (req.in_think_block) {
        // If the model emits a stop token while still inside thinking, treat
        // it as an implicit </think>: NVFP4 quants on Qwen3.6 occasionally
        // skip the explicit close marker and jump straight to <|im_end|>.
        // Without this, generation freezes inside the suppressed-stop branch
        // forever (in_think never flips, every EOS is masked). Flipping the
        // flag here lets the next stop honour normal semantics so the
        // request can actually finish.
        if (is_stop_token(token)) {
            req.in_think_block = false;
            req.think_exit_idx = static_cast<int>(req.output_tokens.size());
            req.think_text_tail.clear();
        }
        return false;
    }
    // After </think>: enforce a minimum answer budget when the model
    // wants to stop. NVFP4 quantization noise on Qwen3.6 lets the model
    // close an empty thinking block in ~3 tokens and then immediately
    // emit <|im_end|>; even after surviving that, the post-</think>
    // logits sometimes tilt toward stop again on the very first content
    // token (observed: model writes "Ger" — start of "Gerne, ..." —
    // then EOS). Counting content tokens AND stop tokens against the
    // grace budget would mean a model that wrote real content past the
    // budget then stopped naturally still hit the trap. Track stop
    // tokens separately: if the last N consecutive emissions since
    // </think> are all stops, accept the finish; if any content token
    // appeared in between, reset the counter.
    if (req.think_exit_idx >= 0 && is_stop_token(token)) {
        int tokens_since_exit = static_cast<int>(req.output_tokens.size()) - req.think_exit_idx;
        constexpr int kMinAnswerAfterThink = 16;
        if (tokens_since_exit < kMinAnswerAfterThink)
            return false;
    }
    return is_stop_token(token);
}

void Engine::fill_sampling_params(const Request& req, InferenceState& state) const {
    state.temperature = req.temperature;
    state.top_p = req.top_p;
    state.top_k = req.top_k;
    state.seed = req.seed;
    state.min_p = req.min_p;
    state.typical_p = req.typical_p;
    state.repetition_penalty = req.repetition_penalty;
    state.frequency_penalty = req.frequency_penalty;
    state.presence_penalty = req.presence_penalty;
    state.repeat_last_n = req.repeat_last_n;
    state.dry_multiplier = req.dry_multiplier;
    state.dry_base = req.dry_base;
    state.dry_allowed_length = req.dry_allowed_length;
    state.dry_penalty_last_n = req.dry_penalty_last_n;
    if (req.dry_multiplier > 0.0f && !req.output_tokens.empty())
        state.host_penalty_tokens = req.output_tokens.data();
    state.mirostat = req.mirostat;
    state.mirostat_tau = req.mirostat_tau;
    state.mirostat_eta = req.mirostat_eta;
    state.mirostat_mu = req.mirostat_mu;

    // Logit bias
    if (!req.logit_bias.empty()) {
        state.logit_bias = req.logit_bias.data();
        state.n_logit_bias = static_cast<int>(req.logit_bias.size());
    }

    // Banned tokens (chat template special tokens that must not be generated)
    if (!banned_token_ids_.empty()) {
        state.banned_tokens = banned_token_ids_.data();
        state.n_banned_tokens = static_cast<int>(banned_token_ids_.size());
    }

    // Think budget: force </think> token via logit manipulation when budget exceeded.
    // Count reasoning tokens (between <think> and </think>) from output history.
    // The model generates </think> itself so it lands in the KV cache correctly.
    // Think budget: force </think> via logit manipulation when budget exceeded.
    // Scan output_tokens directly (no dependency on in_think_block tracking).
    state.force_token = -1;
    if (req.think_budget > 0.0f && think_end_id_ >= 0 && !req.output_tokens.empty()) {
        int think_limit = static_cast<int>(req.max_tokens * req.think_budget);
        int n_reasoning = 0;
        bool currently_thinking = false;
        for (int32_t t : req.output_tokens) {
            if (t == think_start_id_)
                currently_thinking = true;
            else if (t == think_end_id_)
                currently_thinking = false;
            else if (currently_thinking)
                n_reasoning++;
        }
        if (currently_thinking && n_reasoning >= think_limit) {
            state.force_token = think_end_id_;
        }
    }
}

void Engine::upload_penalties(const Request& req, InferenceState& state, cudaStream_t stream) {
    bool needs_penalties = (req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                            req.presence_penalty != 0.0f);
    if (!needs_penalties || req.output_tokens.empty())
        return;

    size_t n = req.output_tokens.size();
    if (n > d_penalty_tokens_capacity_) {
        if (d_penalty_tokens_)
            vram_alloc_.free(d_penalty_tokens_);
        d_penalty_tokens_capacity_ = std::max(n, (size_t)256);
        d_penalty_tokens_ = static_cast<int32_t*>(
            vram_alloc_.allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
        if (!d_penalty_tokens_) {
            IMP_LOG_ERROR("VRAMAllocator failed for penalty tokens (%zu)", d_penalty_tokens_capacity_);
            d_penalty_tokens_capacity_ = 0;
            return;
        }
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_penalty_tokens_, req.output_tokens.data(), n * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
    state.penalty_tokens = d_penalty_tokens_;
    state.n_penalty_tokens = static_cast<int>(n);
}

void Engine::fill_recurrent_state(const Request& req, InferenceState& state, bool reset,
                                  cudaStream_t stream) {
    if (ssm_state_) {
        state.ssm_state = ssm_state_.get();
        state.ssm_seq_id = req.id % ssm_state_->max_sequences();
        if (reset)
            ssm_state_->reset_sequence(state.ssm_seq_id, stream);
    }
    if (gdn_state_) {
        state.gdn_state = gdn_state_.get();
        state.gdn_seq_id = req.id % gdn_state_->max_sequences();
        if (reset)
            gdn_state_->reset_sequence(state.gdn_seq_id, stream);
    }
}

void Engine::finish_request(std::shared_ptr<Request>& req) {
    req->status = RequestStatus::FINISHED;
    if (kv_manager_->prefix_caching_enabled()) {
        kv_manager_->register_block_hashes(req->id, req->input_tokens);
    }
    kv_manager_->free_sequence(req->id);
    constraints_.reset();
}

// =====================================================================
// Vision delegation
// =====================================================================

bool Engine::set_image(const std::string& path) { return vision_.set_image(path, stream_); }

bool Engine::set_image_from_memory(const uint8_t* data, size_t len) {
    return vision_.set_image_from_memory(data, len, stream_);
}

void Engine::clear_image() { vision_.clear_image(); }

// =====================================================================
// Initialization — decomposed into sub-phases
// =====================================================================

bool Engine::init(std::shared_ptr<Model> model, const EngineConfig& config) {
    if (!model)
        return false;

    model_ = std::move(model);
    config_ = config;

    const auto& mcfg = model_->config();

    // --- IMP_DEBUG_RAW: disable all optimization/cache/approximation paths ---
    // Meta-flag for debugging: forces the engine into a "naked" FP16 forward pass
    // for reproducible byte-level comparison against a reference implementation
    // (e.g. llama.cpp). Forces downstream paths off (FP8/NVFP4/warmup/graphs)
    // and cuBLAS to deterministic. Triggered via [runtime] debug_raw = true.
    const bool debug_raw_ = RuntimeConfig::current().runtime.debug_raw;
    if (debug_raw_) {
        IMP_LOG_INFO(
            "[runtime] debug_raw=true: naked FP16 path (FP8/NVFP4/graphs/warmup/FP8-KV off; deterministic "
            "cuBLAS)");
        // Weight storage: keep FP16 (skip the lossy cache paths)
        config_.use_fp8_prefill = 0;
        config_.use_nvfp4_decode = 0;
        config_.dual_path_quant = false;
        // CUDA graphs off (graph capture can mask state bugs)
        config_.use_cuda_graphs = 0;
        setenv("IMP_NO_CUDA_GRAPH", "1", 0);
        // No warmup (warmup can leak state into first request)
        setenv("IMP_NO_WARMUP", "1", 0);
        // Deterministic cuBLAS (bit-exact across runs, no algo jitter)
        setenv("IMP_DETERMINISTIC_GEMM", "1", 0);
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
        // MoE: no expert LRU cache (state-carrying)
        setenv("IMP_NO_EXPERT_CACHE", "1", 0);
        // GDN: use reference unfused scan (no register-state reordering)
        setenv("IMP_GDN_REF", "1", 0);
        // NOTE: intentionally NOT forcing IMP_FORCE_CUBLAS_DECODE / IMP_NO_FMHA_SM120 /
        // IMP_NO_MMVQ — those trigger incompatible kernel paths that produce IMAs on
        // some combinations. The RAW flag is about disabling *caches and approximations*,
        // not about swapping kernel variants.
    }

    // --- KV cache dtype policy ---
    // Default: FP16 (safe). FP8 E4M3 is opt-in via --kv-fp8 / IMP_KV_FP8=1.
    //
    // Rationale: the auto-upgrade to FP8 was found (2026-04-24) to produce
    // NaN logits on several model families (Mistral-Small-3.1, DeepSeek-R1,
    // Qwen3.5-4B/9B GDN, Gemma-4) due to a KV-write stride bug that has not
    // been root-caused yet. Correctness-first: users who want the 50% KV VRAM
    // savings explicitly ask for FP8 via the existing flag.
    //
    // Legacy escape hatches kept for compatibility:
    // - [kv_cache] dtype = "fp16" forces FP16 (no-op under the new default).
    // - [kv_cache] fp8_auto_legacy = true restores the old opt-out auto-
    //              upgrade behavior for users who rely on it for batch-
    //              serving VRAM budgets.
    const bool force_kv_fp16 = (RuntimeConfig::current().kv_cache.dtype == "fp16");
    const bool fp8_auto_legacy = RuntimeConfig::current().kv_cache.fp8_auto_legacy;
    if (fp8_auto_legacy && config_.kv_cache_dtype == QType::F16 && !debug_raw_ && !force_kv_fp16) {
        config_.kv_cache_dtype = QType::FP8_E4M3;
        IMP_LOG_INFO("KV cache dtype: IMP_KV_FP8_AUTO=1 → FP8_E4M3 (legacy opt-out)");
    } else if (config_.kv_cache_dtype == QType::F16) {
        IMP_LOG_INFO("KV cache dtype: FP16 (default — pass --kv-fp8 for FP8 E4M3 memory savings)");
    } else if (config_.kv_cache_dtype == QType::NVFP4) {
        // NVFP4 KV: ~3.6× compression vs FP16 (4 bits + UE4M3 per-16 + ~3% scale overhead).
        // Klasse-A unlock for long-ctx dense models (Gemma-4-26B, Gemma-3-27B, Qwen3-32B).
        IMP_LOG_INFO("KV cache dtype: NVFP4 (FP4 E2M1 + UE4M3 per-16-elem scales, ~3.6× compression)");
        // FP8 prefill cache stacks another lossy layer; disable to avoid compound drift.
        if (config_.use_fp8_prefill) {
            IMP_LOG_INFO("NVFP4 KV: disabling FP8 prefill cache (avoid stacked low-precision drift)");
            config_.use_fp8_prefill = 0;
        }
    } else if (config_.kv_cache_dtype == QType::MXFP4_KV) {
        // MXFP4-KV: same FP4 E2M1 byte layout + per-16-element grouping as NVFP4,
        // but UE8M0 scale bytes (pure-exponent, ~3.6× compression identical to NVFP4).
        // This is the Path A retirement target per design memo §3.1.2.
        IMP_LOG_INFO("KV cache dtype: MXFP4_KV (FP4 E2M1 + UE8M0 per-16-elem scales, ~3.6× compression)");
        if (config_.use_fp8_prefill) {
            IMP_LOG_INFO("MXFP4_KV: disabling FP8 prefill cache (avoid stacked low-precision drift)");
            config_.use_fp8_prefill = 0;
        }
    }

    // ROOT CAUSE of FP8-KV NaN bug (found 2026-04-24): non-deterministic
    // cuBLAS GEMM algo selection produces run-to-run numerical noise in
    // Q/K/V projections. When the KV cache is FP8-E4M3, the quantize-
    // dequantize round-trip amplifies that noise enough to push softmax
    // inputs into NaN after the first 1-3 decode tokens. Reproduced on
    // Mistral-Small-3.1 Q6_K, DeepSeek-R1-Distill-14B Q6_K, Qwen3.5-4B/9B
    // GDN Q8_0, Gemma-4 (the existing hard-coded arch override at
    // line ~492 was working around the same root cause).
    //
    // Fix: when FP8 KV cache is active, force deterministic cuBLAS. Near-
    // zero perf cost on sm_120 (deterministic mode only pins the algo
    // choice, does not disable tensor cores). Users who've verified their
    // model is fine with non-deterministic FP8 KV can opt out via
    // IMP_ALLOW_NONDETERMINISTIC_FP8_KV=1.
    if (config_.kv_cache_dtype == QType::FP8_E4M3 &&
        !RuntimeConfig::current().kv_cache.allow_nondeterministic_fp8 &&
        !RuntimeConfig::current().runtime.deterministic_gemm) {
        // Promote the runtime config in-place so downstream readers see it.
        RuntimeConfig promoted = RuntimeConfig::current();
        promoted.runtime.deterministic_gemm = true;
        RuntimeConfig::install(promoted);
        setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 0);
        IMP_LOG_INFO(
            "FP8 KV cache: forcing runtime.deterministic_gemm=true "
            "(non-deterministic cuBLAS + FP8 round-trip → NaN). "
            "Set kv_cache.allow_nondeterministic_fp8=true to opt out.");
    }
    // NOTE: compound FP8 precision loss (stacking FP8 KV on top of FP8
    // prefill + NVFP4 decode) is model-dependent. Mistral-Small-3.1 and
    // DeepSeek-R1-Distill need the secondary caches OFF (--no-fp8-prefill
    // --no-nvfp4) for coherent output, but Llama-3.2-3B and others break
    // DIFFERENTLY when NVFP4 is forced off on --kv-fp8. No one-size-fits-all
    // auto-toggle — users with affected models use the explicit flags.

    if (config_.max_batch_size <= 0) {
        // Estimate model weight size from config to determine batch capacity.
        // Rough heuristic: 2 bytes/param for FP16. d_model * d_model * n_layers * ~12 gives
        // approximate total weight bytes for a dense transformer.
        size_t approx_weight_bytes = static_cast<size_t>(mcfg.d_model) * mcfg.d_model * mcfg.n_layers *
                                     12;  // ~12 matrices per layer
        if (mcfg.n_experts > 0) {
            // MoE: expert weights dominate
            approx_weight_bytes += static_cast<size_t>(mcfg.n_experts) * mcfg.expert_d_ff * mcfg.d_model *
                                   mcfg.n_layers * 2;
        }
        if (approx_weight_bytes > 20ULL * 1024 * 1024 * 1024)
            config_.max_batch_size = 1;  // >20GB models
        else if (approx_weight_bytes > 10ULL * 1024 * 1024 * 1024)
            config_.max_batch_size = 4;  // 10-20GB
        else if (approx_weight_bytes > 5ULL * 1024 * 1024 * 1024)
            config_.max_batch_size = 8;  // 5-10GB
        else
            config_.max_batch_size = 16;  // <5GB
        IMP_LOG_INFO("max_batch_size: auto → %d (approx_weights=%.1f GB)", config_.max_batch_size,
                     approx_weight_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    // --- Auto-detect SSM state dtype for hybrid models ---
    // Nemotron-H and similar Mamba models: use FP16 for SSM h_state (~50% VRAM savings).
    // GDN models (Qwen3.5 / Qwen3.6) MUST keep FP32: the delta-rule scan kernel
    // writes FP32 (float) into h_state and assumes 4 bytes/element. FP16 allocation
    // would be half the size, so each layer's scan overflows into the next layer's
    // state region — shipped bug that corrupted L1+ GDN state on every Qwen 3.6
    // forward, producing 37% scan-output divergence vs llama.cpp.
    bool has_gdn_for_dtype = false;
    if (mcfg.ssm_state_size > 0) {
        for (int i = 0; i < mcfg.n_layers; i++) {
            if (model_->layer(i).gdn_gate.data != nullptr) {
                has_gdn_for_dtype = true;
                break;
            }
        }
    }
    if (config_.ssm_state_dtype == QType::F32 && mcfg.ssm_state_size > 0 && !has_gdn_for_dtype) {
        config_.ssm_state_dtype = QType::F16;
        IMP_LOG_INFO("SSM state dtype: auto → FP16 (hybrid SSM model, state_size=%d)", mcfg.ssm_state_size);
    }

    // --- Auto-detect FP8 prefill ---
    // Under runtime.debug_raw or [attention] fp8_prefill = "never", keep
    // disabled. The "never" escape hatch is for models (e.g. DeepSeek-R1-
    // Distill-Qwen-14B Q6_K) that produce garbage decode with FP8 weight
    // cache active — accumulated dequant error through deep narrow-GQA
    // stacks.
    const bool no_fp8_prefill = (RuntimeConfig::current().attention.fp8_prefill == "never");
    if (!config_.use_fp8_prefill && !debug_raw_ && !no_fp8_prefill) {
        config_.use_fp8_prefill = true;
        IMP_LOG_INFO("FP8 prefill: auto → enabled");
    } else if (no_fp8_prefill) {
        IMP_LOG_INFO("FP8 prefill: disabled (IMP_NO_FP8_PREFILL=1)");
    }

    // --- Resolve auto-detection flags ---
    // NVFP4 decode mode
    int n_gdn_auto = 0;
    for (int i = 0; i < mcfg.n_layers; i++)
        if (model_->layer(i).gdn_gate.data != nullptr)
            n_gdn_auto++;

    if (config_.use_nvfp4_decode < 0) {
        if (n_gdn_auto > 0) {
            // GDN models with large d_model: enable NVFP4 for attention + FFN weights,
            // but SSM/GDN projections (ssm_in/ssm_out) will be excluded in
            // pre_dequant_weights to preserve recurrent state precision.
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO(
                "NVFP4 decode: auto → mode 2 (GDN model, %d recurrent layers — "
                "ssm_in/ssm_out excluded for precision)",
                n_gdn_auto);
        } else {
            config_.use_nvfp4_decode = 2;
            IMP_LOG_INFO("NVFP4 decode: auto → mode 2");
        }
    }

    // FP8 prefill auto-disable for sub-8-bit models
    if (config_.use_fp8_prefill) {
        auto qtype = model_->layer(0).wq.qtype;
        bool sub_8bit = (qtype == QType::Q4_0 || qtype == QType::Q4_K || qtype == QType::Q5_0 ||
                         qtype == QType::Q5_K || qtype == QType::Q3_K || qtype == QType::Q2_K ||
                         qtype == QType::Q4_1 || qtype == QType::Q5_1);
        if (sub_8bit) {
            config_.use_fp8_prefill = 0;
            IMP_LOG_INFO("FP8 prefill cache: auto-disabled (sub-8-bit weights)");
        }
    }

    // Dual-path quant validation: requires NVFP4 decode + FP8 prefill.
    // If either is missing, auto-enable or warn.
    if (config_.dual_path_quant) {
        if (config_.use_nvfp4_decode <= 0) {
            IMP_LOG_WARN("Dual-path quant requires NVFP4 decode — enabling mode 2 (NVFP4 only)");
            config_.use_nvfp4_decode = 2;
        }
        if (!config_.use_fp8_prefill) {
            IMP_LOG_INFO("Dual-path quant: auto-enabling FP8 prefill for attention weight quality");
            config_.use_fp8_prefill = true;
        }
    }

    // Gemma 4: FP8 prefill, NVFP4 prefill, CUTLASS paths, and CUDA graphs all have
    // incompatibilities with the per-layer head_dim + split MoE tensor layout.
    // Force plain FP16 paths for Gemma 4 until proper kernels are added.
    // GDN models can't use FP8 prefill: recurrent state accumulates precision
    // error per token, FP8 E4M3 (3-bit mantissa) amplifies it through the delta
    // rule scan and degenerates output after ~50 multi-turn special tokens.
    // Decide this BEFORE executor_->init() so the fp8_activation scratch
    // buffer + d_act_scale / d_fp8_block_maxes / d_fp8_absmax aren't allocated
    // and then never used (was happening when the disable lived inside
    // init_kv_cache, ~3 MiB pure waste). Dual-path quant keeps the FP8 path
    // for FFN even on GDN — only attention drops to FP16.
    if (config_.use_fp8_prefill && !config_.dual_path_quant && n_gdn_auto > 0) {
        IMP_LOG_INFO("GDN model: disabling FP8 prefill (recurrent state needs FP16 precision)");
        config_.use_fp8_prefill = 0;
    }
    if (model_->config().arch == ModelArch::GEMMA4) {
        // CUDA graphs: enabled for Gemma-4 decode. The MoE decode fast path is fully
        // device-side (dp4a GEMV, no D2H memcpy), so graph capture works.
        // Only the MoE prefill path uses D2H sync, but prefill is never graph-captured.
        // FP8 prefill carve-out removed 2026-05-15. The 2026-05-09 measurement
        // showed -5..-19% prefill on Gemma-4 vs FP16; since then (PRs #177, #181)
        // the gap has closed. Re-measured 2026-05-15 on Q4_K_M:
        //   pp128:  +1.0%  pp512:  -0.9%  pp833:  -4.2%  pp2048: +7.3%
        // Net effect is neutral with a long-context advantage. FP8 also halves
        // the activation cache, which helps VRAM at long context. Users wanting
        // max prefill at medium pp can opt out via [attention] fp8_prefill = "never".
        if (config_.use_nvfp4_decode) {
            // Prequant SafeTensors NVFP4 weights are already in NVFP4 layout on
            // disk. Phase 3a (Q*_K → NVFP4 conversion) and Phase 3b
            // (NVFP4 → CUTLASS sm_120) iterate `wcache_.nvfp4` which stays
            // empty for prequant, so they are no-ops. Phase 3-MoE (the
            // cache_moe_native_nvfp4 lambda in executor_pre_dequant.cu) IS
            // load-bearing — it builds the contiguous per-layer expert buffer
            // that lights up the M=1 decode fast path (gemv_nvfp4_*) and lets
            // CUDA Graphs capture decode without D2H expert_offsets sync.
            //
            // For Q*_K source weights the per-tensor convert→quantize loop in
            // executor_pre_dequant.cu builds wcache_.nvfp4 per tensor; the
            // per-layer head_dim (256 SWA / 512 global) is uniformly handled
            // since each entry carries its own (N, K) shape. Verified 2026-05-15
            // on Q4_K_M + UD-Q4_K_M: tg256 184 → 204 tok/s (+11%), pp512
            // 1795 → 2347 tok/s (+30%). Coherent on chat prompts; the
            // pre-existing Q4_K_M code-gen drift (see roadmap) is orthogonal.
            IMP_LOG_INFO("Gemma 4: NVFP4 decode cache enabled (use_nvfp4_decode=%d, prequant=%d)",
                         config_.use_nvfp4_decode,
                         (int)model_->config().is_nvfp4_prequant);
        }
        if (config_.dual_path_quant) {
            IMP_LOG_INFO("Gemma 4: disabling dual_path_quant");
            config_.dual_path_quant = false;
        }
        // (Gemma-4 force-FP16 KV carve-out removed 2026-05-01.) The original
        // bug was the FP8 KV calibration reading garbage beyond the per-layer
        // live K/V region — Gemma-4 has dual head_dim (256 SWA / 512 global)
        // and the workspace is allocated for max_head_dim, leaving a
        // tail-region of uninitialized memory on SWA layers. The fix in
        // src/graph/executor_kv_write.cu narrows the calibration view to
        // `nkv * hd` per layer; FP8 KV is now safe to opt into on Gemma-4.
        // Gemma 4 output_norm has extreme outliers (max=588). Small numeric jitter
        // from cuBLAS algo autotuning / split-K atomics amplifies into wildly
        // different top-1 picks (coherent " Paris" vs garbage "\n"). Force
        // deterministic GEMM paths so generation is stable run-to-run.
        if (!RuntimeConfig::current().runtime.deterministic_gemm) {
            RuntimeConfig promoted = RuntimeConfig::current();
            promoted.runtime.deterministic_gemm = true;
            RuntimeConfig::install(promoted);
            IMP_LOG_INFO(
                "Gemma 4: enabling runtime.deterministic_gemm (output_norm outliers amplify algo jitter)");
        }
        if (!getenv("CUBLAS_WORKSPACE_CONFIG")) {
            setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
            IMP_LOG_INFO("Gemma 4: setting CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic grouped GEMM");
        }
        // Gemma 4: CUDA graphs are fully enabled by default. The user can opt
        // out via [gemma4] no_graphs = true for bisecting regressions.
        if (RuntimeConfig::current().gemma4.no_graphs) {
            IMP_LOG_INFO("Gemma 4: disabling all CUDA graphs (gemma4.no_graphs=true)");
            config_.use_cuda_graphs = false;
        }
        // Enable MMVQ for all weight GEMMs — quantized matmul matching llama.cpp's
        // accumulation behavior, critical for 128-expert MoE precision.
        if (!RuntimeConfig::current().gemma4.force_mmvq) {
            RuntimeConfig promoted = RuntimeConfig::current();
            promoted.gemma4.force_mmvq = true;
            RuntimeConfig::install(promoted);
            IMP_LOG_INFO("Gemma 4: enabling MMVQ for all weight GEMMs (numerical parity with llama.cpp)");
        }
    }

    // --- Auto-detect max_seq_len ---
    // Runs AFTER model-specific overrides (Gemma-4 forces FP16 KV etc.) so the
    // per-token cost reflects the actual dtype that will be allocated.
    if (int v = RuntimeConfig::current().runtime.max_seq_len; v > 0) {
        config_.max_seq_len = v;
        IMP_LOG_INFO("max_seq_len: runtime.max_seq_len=%d", v);
    }
    if (config_.max_seq_len <= 0) {
        int model_ctx = mcfg.max_seq_len;  // from GGUF metadata
        size_t free_vram = 0, total_vram = 0;
        cudaMemGetInfo(&free_vram, &total_vram);
        int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
        // Hybrid models (Qwen3.5/3.6 GDN, Nemotron-H Mamba2) populate
        // n_kv_heads_per_layer with zeros for non-attention layers — those don't
        // contribute to the KV cache. Counting only nonzero entries avoids a
        // 4-9× per-token-bytes overestimate that clamped max_seq_len far below
        // VRAM-feasible (e.g. Qwen3.5-4B GDN: 32 total / 8 attention = 4×).
        int kv_layer_count = mcfg.n_layers;
        if (!mcfg.n_kv_heads_per_layer.empty()) {
            int populated = 0;
            for (int v : mcfg.n_kv_heads_per_layer)
                if (v > 0)
                    ++populated;
            if (populated > 0)
                kv_layer_count = populated;
        }
        // Per-token KV bytes for the real kv dtype (INT4/TQ pack 2 elems/byte).
        auto kv = config_.kv_cache_dtype;
        bool packed_int4 = (kv == QType::INT4);
        size_t per_tok_elems = static_cast<size_t>(mcfg.n_kv_heads) * head_dim * kv_layer_count *
                               2;  // K+V, per KV head, attention layers only
        size_t kv_bytes_per_token = packed_int4 ? (per_tok_elems / 2) : (per_tok_elems * dtype_size(kv));
        // The budget planner downstream uses 80% of free VRAM for KV. Cap the
        // auto-detect at ~60% so it doesn't undershoot what the planner can
        // afford. (Was 30%, calibrated when weight caches competed at FP16.)
        int max_by_vram = (kv_bytes_per_token > 0) ? static_cast<int>(free_vram * 0.6 / kv_bytes_per_token)
                                                   : 131072;
        // Conservative auto-default: cap at 16K even when model + VRAM both
        // allow more. Long contexts (Qwen3.6 256K, Qwen3 40K) consume a lot of
        // VRAM and most workloads don't need them. Users requesting longer
        // context pass --max-seq-len explicitly; the cap doesn't apply when
        // the user sets it.
        constexpr int kAutoMaxSeqLenCap = 16384;
        config_.max_seq_len = std::min({model_ctx, std::max(max_by_vram, 4096), kAutoMaxSeqLenCap});
        IMP_LOG_INFO(
            "max_seq_len: auto → %d (model=%d, vram_cap=%d, auto_cap=%d, kv=%zu B/tok, attn_layers=%d/%d)",
            config_.max_seq_len, model_ctx, max_by_vram, kAutoMaxSeqLenCap, kv_bytes_per_token,
            kv_layer_count, mcfg.n_layers);
    }

    // --- Core initialization ---
    // 5% headroom (was 10%) — MoE models (30B Q6_K) need every MiB on 32GB.
    // WSL2/WDDM has ~500 MiB driver overhead, 5% of 32GB = 1.6 GB covers it.
    if (!vram_alloc_.init(0.05f)) {
        IMP_LOG_ERROR("Failed to initialize VRAM allocator");
        return false;
    }
    gemm_init();
    attention_cublas_prewarm();
    gemm_grouped_3x_nvfp4_prewarm();
    scheduler_ = std::make_unique<Scheduler>(config_.max_batch_size);
    (void)stream_.create(cudaStreamNonBlocking);

    // --- Sub-phases ---
    if (!init_weights())
        return false;
    if (!init_kv_cache())
        return false;
    if (!init_features())
        return false;
    if (!RuntimeConfig::current().runtime.warmup) {
        IMP_LOG_INFO("Warmup SKIPPED (runtime.warmup=false)");
    } else {
        warmup();
    }

    return true;
}

bool Engine::init_weights() {
    const auto& mcfg = model_->config();

    // Initialize graph executor (Phase 1: compute sizes, no GPU allocation)
    executor_ = std::make_unique<GraphExecutor>();
    executor_->set_vram_allocator(&vram_alloc_);
    {
        int eff_batch = config_.max_batch_size;
        if (!executor_->init(*model_, config_.compute_dtype, config_.use_pdl, eff_batch, config_.max_seq_len,
                             config_.use_fp8_prefill, config_.use_nvfp4_decode, config_.use_mxfp4_prefill))
            return false;

        if (config_.dual_path_quant) {
            executor_->set_dual_path_quant(true);
            IMP_LOG_INFO("Dual-path quant: attention weights → FP8, FFN weights → NVFP4");
        }

        if (config_.streaming_kv_enabled) {
            // Streaming is only safe for the FP16 GQA decode kernel — the
            // quantized variants don't yet skip -1 sentinels in their block
            // tables. Refuse to enable streaming with non-FP16 KV caches so
            // we never call evict_middle_blocks for an unsupported path.
            if (config_.kv_cache_dtype != QType::F16) {
                IMP_LOG_WARN(
                    "StreamingLLM smart KV cache requires FP16 KV cache "
                    "(requested %d) — disabling streaming.",
                    static_cast<int>(config_.kv_cache_dtype));
                config_.streaming_kv_enabled = false;
            } else {
                int n_sinks = (config_.streaming_kv_n_sinks > 0) ? config_.streaming_kv_n_sinks : 4;
                int win = (config_.streaming_kv_window > 0) ? config_.streaming_kv_window
                                                            : model_->config().sliding_window;
                executor_->set_streaming_kv(n_sinks, win);
                if (n_sinks > 0 && win > 0) {
                    IMP_LOG_INFO("StreamingLLM smart KV cache enabled: %d sinks + %d-token window", n_sinks,
                                 win);
                    // Block-table contents change every step once eviction
                    // begins; a CUDA graph captured against an old table
                    // would replay stale pointers. Re-capturing per step
                    // negates the graph's win, so disable graphs entirely.
                    if (config_.use_cuda_graphs) {
                        IMP_LOG_INFO(
                            "Disabling CUDA Graphs while StreamingLLM is active "
                            "(block table mutates per decode step).");
                        config_.use_cuda_graphs = false;
                    }
                } else {
                    IMP_LOG_WARN(
                        "StreamingLLM enabled but no sliding window configured "
                        "(n_sinks=%d, window=%d) — disabling streaming.",
                        n_sinks, win);
                    config_.streaming_kv_enabled = false;
                }
            }
        }
    }

    // Reserve L2 persisting cache for decode GEMV
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t max_persist = prop.persistingL2CacheMaxSize;
        if (max_persist > 0) {
            size_t reserve = max_persist * 3 / 4;
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, reserve);
            IMP_LOG_INFO("L2 persisting cache: reserved %zu MB / %zu MB total", reserve >> 20,
                         max_persist >> 20);
        }
    }

    // Tune the default cudaMallocAsync pool so it retains freed memory instead
    // of returning it to the driver. Many paths (prefill metadata, MoE scratch,
    // spec decoder block tables, vision staging) use cudaMallocAsync with the
    // default pool; the default threshold is 0, which calls cuMemUnmap on every
    // free. Setting UINT64_MAX keeps allocations for re-use — the KV cache and
    // workspaces already own their memory through the DeviceAllocator, so the
    // default-pool footprint stays small.
    {
        cudaMemPool_t default_pool = nullptr;
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetDefaultMemPool(&default_pool, dev) == cudaSuccess && default_pool != nullptr) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(default_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
    }

    // Compute VRAM reserve for expert weight upload
    size_t expert_reserve = executor_->workspace_estimate();
    {
        int head_dim_est = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);
        size_t elem_sz = dtype_size(config_.kv_cache_dtype);
        int est_bs = config_.kv_block_size > 0 ? config_.kv_block_size : kKVBlockSize;
        int blocks_per_seq = (config_.max_seq_len + est_bs - 1) / est_bs;
        int n_attn = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).wq.data != nullptr)
                n_attn++;
        if (n_attn == 0)
            n_attn = mcfg.n_layers;
        size_t kv_block_bytes = static_cast<size_t>(est_bs) * mcfg.n_kv_heads * head_dim_est * elem_sz;
        size_t kv_est = static_cast<size_t>(blocks_per_seq * config_.max_batch_size) * 2 * n_attn *
                        kv_block_bytes;
        {
            size_t total_vram = 0, f = 0;
            cudaMemGetInfo(&f, &total_vram);
            // For large MoE models (128 experts), prefer fitting all experts on GPU
            // over reserving huge KV cache. All-GPU experts enable the decode fast
            // path (dp4a GEMV, no D2H sync) and CUDA graph capture.
            size_t vram_frac = (mcfg.n_experts > 16) ? 10 : 5;
            kv_est = std::min(kv_est, total_vram / vram_frac);
        }
        expert_reserve += kv_est;

        if (mcfg.ssm_inner_size > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int hd_ssm = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            int n_ssm = 0;
            for (int i = 0; i < mcfg.n_layers; i++)
                if (model_->layer(i).ssm_in.data != nullptr)
                    n_ssm++;
            expert_reserve += static_cast<size_t>(n_ssm) * config_.max_batch_size *
                              (conv_ch * std::max(mcfg.ssm_conv_kernel - 1, 0) * sizeof(float) +
                               n_heads * hd_ssm * mcfg.ssm_state_size * dtype_size(config_.ssm_state_dtype));
        }

        size_t safety = 256ULL * 1024 * 1024;  // base safety
        // Only add safety for features that will actually allocate VRAM.
        // On tight VRAM models (Nemotron-30B), every MiB matters for expert coverage.
        expert_reserve += safety;

        IMP_LOG_INFO("Expert upload reserve: %.2f MiB (workspace=%.2f, kv=%.2f, ssm+safety=rest)",
                     expert_reserve / (1024.0 * 1024.0), executor_->workspace_estimate() / (1024.0 * 1024.0),
                     kv_est / (1024.0 * 1024.0));
    }

    // Upload weights
    size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);
    IMP_LOG_INFO("GPU memory before weight upload: %zu MiB free / %zu MiB total", free_before / (1024 * 1024),
                 total_before / (1024 * 1024));

    cudaStream_t upload_stream = nullptr;
    IMP_CUDA_CHECK_LOG(cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking));

    if (!model_->upload_weights_gpu(config_.compute_dtype, upload_stream ? upload_stream : stream_,
                                    expert_reserve)) {
        IMP_LOG_ERROR("Weight upload failed. Try a smaller quantization.");
        if (upload_stream)
            IMP_CUDA_CHECK_LOG(cudaStreamDestroy(upload_stream));
        return false;
    }

    if (upload_stream) {
        cudaEvent_t upload_done;
        IMP_CUDA_CHECK_LOG(cudaEventCreate(&upload_done));
        IMP_CUDA_CHECK_LOG(cudaEventRecord(upload_done, upload_stream));
        IMP_CUDA_CHECK_LOG(cudaStreamWaitEvent(stream_, upload_done));
        IMP_CUDA_CHECK_LOG(cudaEventDestroy(upload_done));
        IMP_CUDA_CHECK_LOG(cudaStreamDestroy(upload_stream));
    }

    size_t free_after = 0, total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);
    IMP_LOG_INFO("GPU memory after weight upload: %zu MiB free / %zu MiB total (weights ~%zu MiB)",
                 free_after / (1024 * 1024), total_after / (1024 * 1024),
                 (free_before - free_after) / (1024 * 1024));

    // Check for host-resident expert weights
    if (mcfg.n_experts > 0) {
        for (int i = 0; i < mcfg.n_layers; i++) {
            if (model_->layer(i).expert_up_packed.data && !model_->layer(i).expert_up_packed.on_device) {
                experts_on_host_ = true;
                break;
            }
        }
        if (experts_on_host_ && config_.use_cuda_graphs) {
            // Phase 5: the opt-in `moe.allow_graphs_under_offload` flag skips
            // this guard so the user can experiment with captured decode
            // under host-offload. Correctness is conditional on prefetch
            // coverage matching router selection — captured cudaMemcpyAsync
            // nodes have fixed (src host ptr, dst slot) pairs that don't
            // adapt to per-token routing changes. See config.h doc for the
            // architectural caveat; Phase 5.1+ refactors dispatch kernels
            // to read the device mirror at runtime so the captured graph
            // adapts correctly.
            if (RuntimeConfig::current().moe.allow_graphs_under_offload) {
                IMP_LOG_WARN(
                    "CUDA graphs ENABLED under host-offload "
                    "(moe.allow_graphs_under_offload=true). EXPERIMENTAL: output "
                    "correctness depends on prefetch coverage matching router "
                    "selection. Set moe.prefetch_top_k high enough that the "
                    "captured top-K covers the router's hot set.");
            } else {
                IMP_LOG_INFO("Disabling CUDA graphs: expert weights on host");
                IMP_LOG_INFO(
                    "  Tip: if model+KV fits in VRAM, set IMP_EXPERT_OVERHEAD_PCT=10 "
                    "(default 30) to upload ALL experts and re-enable CUDA graphs "
                    "(+~180%% decode on Qwen 3.6 35B Q4_K_M).");
                IMP_LOG_INFO(
                    "  Or set moe.allow_graphs_under_offload=true (experimental) "
                    "to opt into captured decode under host-offload.");
                config_.use_cuda_graphs = false;
            }
        }
        if (RuntimeConfig::current().runtime.cuda_graphs == "never" && config_.use_cuda_graphs) {
            IMP_LOG_INFO("Disabling CUDA graphs: runtime.cuda_graphs=never");
            config_.use_cuda_graphs = false;
        }
        // MoE decode fast path is fully device-side (no D2H memcpy) — graph-safe.
        // Only MoE prefill paths use D2H sync for expert_offsets, but prefill is
        // never captured in CUDA graphs.
    }

    // Phase 2: allocate GPU workspace
    (void)executor_->allocate_workspaces(experts_on_host_);

    // Layer offloading
    if (config_.gpu_layers >= 0) {
        offload_mgr_ = std::make_unique<LayerOffloadManager>();
        if (!offload_mgr_->init(model_.get(), config_.gpu_layers)) {
            IMP_LOG_WARN("Layer offloading init failed, continuing without it");
            offload_mgr_.reset();
        }
    }

    return true;
}

bool Engine::init_kv_cache() {
    const auto& mcfg = model_->config();
    int head_dim = mcfg.head_dim > 0 ? mcfg.head_dim : (mcfg.d_model / mcfg.n_heads);

    // Build KV layer mapping for hybrid models
    int n_attn_layers = 0;
    std::vector<int> kv_layer_map(mcfg.n_layers, -1);
    for (int i = 0; i < mcfg.n_layers; i++) {
        if (model_->layer(i).wq.data != nullptr && model_->layer(i).gdn_gate.data == nullptr)
            kv_layer_map[i] = n_attn_layers++;
    }
    if (n_attn_layers == 0) {
        n_attn_layers = mcfg.n_layers;
        for (int i = 0; i < mcfg.n_layers; i++)
            kv_layer_map[i] = i;
    }
    int n_kv_layers = n_attn_layers;
    IMP_LOG_INFO("KV cache layers: %d attention out of %d total", n_kv_layers, mcfg.n_layers);

    // Auto-select block size
    if (config_.kv_block_size <= 0) {
        config_.kv_block_size = (mcfg.n_kv_heads <= 4 && mcfg.n_kv_heads > 0) ? 32 : kKVBlockSize;
        IMP_LOG_INFO("KV block size: auto → %d (n_kv_heads=%d)", config_.kv_block_size, mcfg.n_kv_heads);
    }
    const int kv_bs = config_.kv_block_size;
    int blocks_per_seq = (config_.max_seq_len + kv_bs - 1) / kv_bs;

    // VRAM budget
    auto vram_budget = compute_vram_budget(*model_, config_, n_kv_layers, head_dim, effective_free_vram());
    int max_blocks = config_.kv_cache_max_blocks > 0 ? config_.kv_cache_max_blocks
                                                     : vram_budget.kv_max_blocks;

    {
        QType kv_dtype = config_.kv_cache_dtype;
        size_t block_bytes = static_cast<size_t>(kv_bs) * mcfg.n_kv_heads * head_dim * dtype_size(kv_dtype);
        size_t total_kv = static_cast<size_t>(n_kv_layers) * max_blocks * 2 * block_bytes;
        IMP_LOG_INFO(
            "KV cache: %d blocks (%.0f tokens), %.2f MiB, dtype=%s "
            "(layers=%d/%d, kv_heads=%d, head_dim=%d, block_size=%d)",
            max_blocks, static_cast<double>(max_blocks) * kv_bs,
            static_cast<double>(total_kv) / (1024.0 * 1024.0), dtype_name(kv_dtype), n_kv_layers,
            mcfg.n_layers, mcfg.n_kv_heads, head_dim, kv_bs);
    }

    // Per-layer KV shape path (Gemma 4 dual attention geometry): build per-layer
    // nkv/hd arrays restricted to attention layers (hybrid models may have non-attn layers).
    std::unique_ptr<KVCache> kv_cache;
    if (!mcfg.head_dim_per_layer.empty() && config_.kv_cache_dtype != QType::INT8 &&
        config_.kv_cache_dtype != QType::INT4) {
        std::vector<int> per_layer_nkv(n_kv_layers, 0);
        std::vector<int> per_layer_hd(n_kv_layers, 0);
        for (int l = 0, k = 0; l < mcfg.n_layers && k < n_kv_layers; l++) {
            // Only attention layers get KV cache entries
            int attn_nkv = (l < (int)mcfg.n_kv_heads_per_layer.size()) ? mcfg.n_kv_heads_per_layer[l]
                                                                       : mcfg.n_kv_heads;
            if (attn_nkv <= 0)
                continue;  // non-attention layer (SSM/GDN)
            per_layer_nkv[k] = attn_nkv;
            per_layer_hd[k] = (l < (int)mcfg.head_dim_per_layer.size() && mcfg.head_dim_per_layer[l] > 0)
                                  ? mcfg.head_dim_per_layer[l]
                                  : head_dim;
            k++;
        }
        kv_cache = std::make_unique<KVCache>(n_kv_layers, per_layer_nkv, per_layer_hd, config_.kv_cache_dtype,
                                             max_blocks, kv_bs, &vram_alloc_);
    } else {
        kv_cache = std::make_unique<KVCache>(n_kv_layers, mcfg.n_kv_heads, head_dim, config_.kv_cache_dtype,
                                             max_blocks, kv_bs, &vram_alloc_);
    }
    kv_cache_raw_ = kv_cache.get();
    kv_manager_ = std::make_unique<KVCacheManager>(std::move(kv_cache));

    // BitDecoding Phase 3: residual FP16 cache (opt-in).
    //
    // Ring state (write_idx / fill_count per slot) lives in device memory
    // (kv_manager_->d_residual_widx_ptr / d_residual_fc_ptr). Updated by a
    // tiny advance_residual_state_kernel at the end of forward_logits; the
    // residual write/read kernels read the state at execution time. This
    // makes the whole path graph-capture-safe — graphs stay enabled.
    {
        const auto& rcfg = RuntimeConfig::current();
        int residual_n = rcfg.kv_cache.bitdecoding_residual_tokens;
        if (residual_n > 0 && config_.kv_cache_dtype == QType::NVFP4) {
            int max_seqs = config_.max_batch_size > 0 ? config_.max_batch_size : 1;
            if (kv_manager_->enable_residual_buffer(max_seqs, residual_n, &vram_alloc_)) {
                // Persistent batch→slot lookup buffer (graph-safe). [max_batch_size] ints.
                size_t slot_bytes = static_cast<size_t>(max_seqs) * sizeof(int);
                cudaMalloc(&d_kv_slot_buf_, slot_bytes);
                std::vector<int> init_slots(max_seqs, -1);
                cudaMemcpy(d_kv_slot_buf_, init_slots.data(), slot_bytes, cudaMemcpyHostToDevice);
                d_kv_slot_last_uploaded_.assign(max_seqs, -1);
            }
        } else if (residual_n > 0) {
            IMP_LOG_INFO("kv_cache.bitdecoding_residual_tokens=%d ignored (only active with kv_cache_dtype=NVFP4)",
                         residual_n);
        }
    }

    if (config_.use_prefix_caching) {
        if (mcfg.ssm_inner_size > 0) {
            IMP_LOG_WARN(
                "Prefix caching disabled for recurrent model — "
                "SSM/GDN state requires full sequential prefill");
        } else {
            kv_manager_->set_prefix_caching_enabled(true);
            IMP_LOG_INFO("Prefix caching enabled");
            if (!config_.prefix_cache_path.empty()) {
                int restored = kv_manager_->load_prefix_cache(config_.prefix_cache_path, stream_);
                if (restored > 0)
                    IMP_LOG_INFO("Restored %d prefix cache blocks from %s", restored,
                                 config_.prefix_cache_path.c_str());
            }
        }
    }

    executor_->set_kv_layer_map(std::move(kv_layer_map));

    if (offload_mgr_)
        executor_->set_offload_manager(offload_mgr_.get());
    scheduler_->set_kv_manager(kv_manager_.get());

    // SSM state
    if (mcfg.ssm_inner_size > 0) {
        int n_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).ssm_in.data != nullptr)
                n_ssm++;
        if (n_ssm > 0) {
            int conv_ch = mcfg.ssm_inner_size + 2 * mcfg.ssm_group_count * mcfg.ssm_state_size;
            int n_heads = mcfg.ssm_dt_rank;
            int hd = (n_heads > 0) ? mcfg.ssm_inner_size / n_heads : 0;
            ssm_state_ = std::make_unique<SSMState>();
            if (!ssm_state_->init(n_ssm, config_.max_batch_size, conv_ch, mcfg.ssm_conv_kernel, n_heads, hd,
                                  mcfg.ssm_state_size, config_.ssm_state_dtype, &vram_alloc_)) {
                IMP_LOG_WARN("Failed to init SSM state, continuing without it");
                ssm_state_.reset();
            }
        }
    }

    // GDN detection
    {
        int n_gdn = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).gdn_gate.data != nullptr)
                n_gdn++;
        if (n_gdn > 0) {
            if (config_.use_cuda_graphs) {
                IMP_LOG_INFO("GDN model: %d layers, CUDA graphs enabled (recurrent state in-place)", n_gdn);
            } else {
                IMP_LOG_INFO(
                    "GDN model: %d layers, CUDA graphs disabled (disabled earlier by caller or expert "
                    "offload)",
                    n_gdn);
            }
            // GDN recurrent state accumulates small precision errors per token.
            // FP8 E4M3 (3-bit mantissa) amplifies these through the delta rule
            // scan, causing degenerate output after ~50 special tokens in
            // multi-turn chat.  Force FP16 weights for GDN prefill.
            if (config_.use_fp8_prefill) {
                if (config_.dual_path_quant) {
                    IMP_LOG_WARN(
                        "GDN + dual-path: attention weights forced to FP16 (not FP8) — "
                        "recurrent state needs FP16 precision. FFN weights still use NVFP4.");
                } else {
                    IMP_LOG_INFO("GDN model: disabling FP8 prefill (recurrent state needs FP16 precision)");
                }
                config_.use_fp8_prefill = 0;
                executor_->disable_fp8_prefill();
            }
        }
    }

    // (Gemma 4 FP8 prefill disabled earlier, before executor init)

    // Detect pure Mamba2 SSM layers (layers with ssm_in but without gdn_gate).
    // GDN-only models (Qwen3.5) are graph-compatible; pure SSM (Nemotron-H) is not yet.
    {
        int n_pure_ssm = 0;
        for (int i = 0; i < mcfg.n_layers; i++)
            if (model_->layer(i).ssm_in.data != nullptr && model_->layer(i).gdn_gate.data == nullptr)
                n_pure_ssm++;
        has_pure_ssm_layers_ = (n_pure_ssm > 0);
    }

    // Dequant weights → FP16/FP8/NVFP4 caches
    executor_->pre_dequant_weights(stream_, vram_budget);
    dequant_done_ = true;
    cudaStreamSynchronize(stream_);

    // Pre-allocate the gemm_nvfp4 fallback dequant workspace. Sized from
    // wcache_.nvfp4 which is populated by pre_dequant_weights above, so this
    // call must come AFTER. Lets the M>1 fallback path (used by future
    // multi-token verify / spec-decode) run inside CUDA stream capture
    // without crashing on cudaMalloc.
    (void)executor_->allocate_nvfp4_dequant_workspace();
    if (config_.use_fp8_prefill)
        IMP_LOG_INFO("Weight cache: FP8 E4M3 (2x prefill throughput on sm_120)");

    // Pre-allocate decode batch pool + penalty buffer
    decode_batch_pool_.allocate(config_.max_batch_size, blocks_per_seq, &vram_alloc_);
    {
        d_penalty_tokens_capacity_ = static_cast<size_t>(config_.max_seq_len);
        d_penalty_tokens_ = static_cast<int32_t*>(
            vram_alloc_.allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
        if (!d_penalty_tokens_) {
            IMP_LOG_WARN("Failed to pre-allocate penalty token buffer");
            d_penalty_tokens_capacity_ = 0;
        }
    }

    // Pre-allocate prefill metadata pool (avoids per-request cudaMallocAsync)
    {
        size_t tok_bytes = config_.max_seq_len * sizeof(int32_t);
        size_t pos_bytes = config_.max_seq_len * sizeof(int);
        // A single request's block_table can grow to the entire KV cache
        // pool (max_blocks), not just max_seq_len/block_size. Size from
        // max_blocks so the H2D copy at the prefill metadata upload site
        // doesn't overflow on long-cumulative-KV requests.
        size_t bt_bytes = static_cast<size_t>(max_blocks) * sizeof(int);
        size_t cl_bytes = sizeof(int);
        prefill_pool_size_ = tok_bytes + pos_bytes + bt_bytes + cl_bytes;
        prefill_pool_ = vram_alloc_.allocate(prefill_pool_size_, "prefill_pool");
        if (prefill_pool_) {
            auto* base = static_cast<char*>(prefill_pool_);
            d_pf_token_ids_ = reinterpret_cast<int32_t*>(base);
            d_pf_positions_ = reinterpret_cast<int*>(base + tok_bytes);
            d_pf_block_tables_ = reinterpret_cast<int*>(base + tok_bytes + pos_bytes);
            d_pf_context_lens_ = reinterpret_cast<int*>(base + tok_bytes + pos_bytes + bt_bytes);
        } else {
            IMP_LOG_WARN("Failed to pre-allocate prefill pool, will use per-request malloc");
        }

        // Pinned host staging buffers for prefill
        if (cudaHostAlloc(&h_pf_positions_, config_.max_seq_len * sizeof(int), cudaHostAllocDefault) !=
            cudaSuccess)
            h_pf_positions_ = nullptr;
        if (cudaHostAlloc(&h_pf_token_ids_, config_.max_seq_len * sizeof(int32_t), cudaHostAllocDefault) !=
            cudaSuccess)
            h_pf_token_ids_ = nullptr;
    }

    // Report memory
    {
        size_t free_mem = 0, total_mem = 0;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            IMP_LOG_INFO("GPU memory: %.0f MiB used / %.0f MiB total (%.0f MiB free)",
                         (total_mem - free_mem) / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0),
                         free_mem / (1024.0 * 1024.0));
        vram_alloc_.report();
    }

    return true;
}

bool Engine::init_features() {
    const auto& mcfg = model_->config();

    // Green contexts
    if (config_.use_green_contexts) {
        if (!green_ctx_.init(0, config_.green_ctx_prefill_ratio)) {
            IMP_LOG_WARN("Green context init failed — falling back to regular streams");
            // Clear the CUDA error state so it doesn't corrupt subsequent operations.
            // Green context failure on sm_120 consumer GPUs is expected (requires
            // data-center features). Without clearing, the stale error causes
            // cublasLtMatmul to fail with CUBLAS_STATUS_INVALID_VALUE.
            cudaGetLastError();
        }
        if (green_ctx_.is_available() && resolve_prefill_chunk_size_() > 0)
            if (executor_->allocate_decode_workspace(stream_, config_.max_batch_size))
                IMP_LOG_INFO("Concurrent prefill/decode overlap enabled");
    }

    // Chat template
    if (Tokenizer* tok = model_->tokenizer()) {
        auto family = ChatTemplate::detect_family(tok->chat_template_str());
        if (family == ChatTemplateFamily::RAW) {
            family = ChatTemplate::default_family_for_arch(mcfg.arch);
            if (family != ChatTemplateFamily::RAW)
                IMP_LOG_INFO("No chat template in metadata, using %s default for %s",
                             chat_template_family_name(family), model_arch_name(mcfg.arch));
        }
        if (family != ChatTemplateFamily::RAW)
            chat_template_.init(family, *tok, tok->chat_template_str());
    }

    build_banned_token_list();

    // Cache think token IDs for stop-suppression during reasoning.
    // Only treat as think model if <think> is a CONTROL token (from GGUF metadata),
    // not a regular text piece. Nemotron has "<think>" at ID 12 as normal text.
    {
        Tokenizer* ptok = model_->tokenizer();
        if (ptok) {
            int32_t ts = ptok->find_token("<think>");
            int32_t te = ptok->find_token("</think>");
            int vocab = ptok->vocab_size();
            bool is_special = (ts >= 0) &&
                              (ptok->has_token_types() ? ptok->is_control_token(ts) : ts > vocab * 99 / 100);
            if (is_special) {
                think_start_id_ = ts;
                think_end_id_ = te;
            }
        }
    }

    // Vision
    if (!config_.mmproj_path.empty()) {
        if (!vision_.init(config_.mmproj_path, mcfg.d_model, model_.get(), vram_alloc_, stream_))
            return false;
    }

    // Pinned sample buffer for CUDA graphs
    if (!h_sample_pinned_) {
        cudaError_t err = cudaHostAlloc(&h_sample_pinned_, sizeof(int32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostAlloc for sample buffer failed: %s", cudaGetErrorString(err));
            if (config_.use_cuda_graphs)
                config_.use_cuda_graphs = false;
            h_sample_pinned_ = nullptr;
        }
    }
    if (!decode_done_)
        (void)decode_done_.create(cudaEventDisableTiming);

    // Pre-allocate DRY penalty buffers to avoid cudaStreamSynchronize on first
    // use during inference (the lazy-alloc path blocks the decode stream).
    sampling_preallocate_dry(config_.max_seq_len, decode_stream());

    return true;
}

void Engine::build_banned_token_list() {
    // Diagnostic bypass: generation.no_ban (legacy IMP_NO_BAN=1) disables the
    // ban list. Used to bisect Mistral-Small-3.2-NVFP4 long-form repetition
    // (ban vs weight quality).
    if (RuntimeConfig::current().generation.no_ban) {
        banned_token_ids_.clear();
        IMP_LOG_WARN("generation.no_ban=true: skipping banned-token list (debug)");
        return;
    }
    banned_token_ids_.clear();
    auto add_if_valid = [this](int32_t id) {
        if (id >= 0) banned_token_ids_.push_back(id);
    };

    // Collect IDs that must NOT be banned: stop tokens, EOS, think tokens,
    // and Gemma-4 channel markers (the model is trained to emit them).
    std::vector<int32_t> keep_ids;
    Tokenizer* tok = model_->tokenizer();
    if (tok) {
        for (int32_t eid : tok->eos_ids()) keep_ids.push_back(eid);
    }
    for (int32_t sid : chat_template_.stop_token_ids()) keep_ids.push_back(sid);
    if (tok) {
        for (const char* name : {"<think>", "</think>", "<|think|>", "<|/think|>",
                                  "<|channel>", "<channel|>"}) {
            int32_t tid = tok->find_token(name);
            if (tid >= 0) keep_ids.push_back(tid);
        }
    }
    auto is_kept = [&](int32_t id) {
        return std::find(keep_ids.begin(), keep_ids.end(), id) != keep_ids.end();
    };

    // Chat template start-of-turn delimiters (never valid in output)
    if (!is_kept(chat_template_.im_start_id()))
        add_if_valid(chat_template_.im_start_id());
    if (!is_kept(chat_template_.start_header_id()))
        add_if_valid(chat_template_.start_header_id());
    if (!is_kept(chat_template_.end_header_id()))
        add_if_valid(chat_template_.end_header_id());

    // Scan vocab for control tokens. Authoritative path uses GGUF token_type
    // metadata; fallback uses heuristic pattern matching on legacy GGUFs.
    if (tok) {
        int vocab_size = tok->vocab_size();
        if (tok->has_token_types()) {
            for (int i = 0; i < vocab_size; i++) {
                if (is_kept(static_cast<int32_t>(i))) continue;
                if (tok->is_control_token(i)) add_if_valid(static_cast<int32_t>(i));
            }
        } else {
            for (int i = 0; i < vocab_size; i++) {
                if (is_kept(static_cast<int32_t>(i))) continue;
                const std::string& t = tok->token_text(i);
                if (t.size() < 3 || t[0] != '<' || t.back() != '>') continue;
                if (t.size() >= 4 && t[1] == '|' && t[t.size() - 2] == '|') {
                    add_if_valid(static_cast<int32_t>(i));
                    continue;
                }
                if (t == "<pad>" || t == "<unk>" || t == "<mask>" || t == "<unused0>" ||
                    t == "<start_of_turn>" || t == "<end_of_turn>" ||
                    t == "<start_of_image>" || t == "<end_of_image>") {
                    add_if_valid(static_cast<int32_t>(i));
                }
            }
        }
    }

    // Deduplicate
    std::sort(banned_token_ids_.begin(), banned_token_ids_.end());
    banned_token_ids_.erase(std::unique(banned_token_ids_.begin(), banned_token_ids_.end()),
                            banned_token_ids_.end());

    if (!banned_token_ids_.empty()) {
        IMP_LOG_INFO("Banned %zu special tokens from generation", banned_token_ids_.size());
        if (tok) {
            std::string bl;
            for (int32_t bid : banned_token_ids_) {
                bl += std::to_string(bid) + "(" + tok->token_text(bid) + ") ";
            }
            IMP_LOG_INFO("  banned: %s", bl.c_str());
        }
    }
}

void Engine::warmup() {
    // Skip warmup for MXFP4 models — the warmup forward pass triggers
    // illegal memory access due to kernel paths that bypass the FP16 cache
    // and attempt to use raw MXFP4 data as FP16 weights.
    bool has_mxfp4_weights = false;
    for (int i = 0; i < model_->config().n_layers && !has_mxfp4_weights; i++) {
        if (model_->layer(i).wq.qtype == QType::MXFP4)
            has_mxfp4_weights = true;
    }
    if (has_mxfp4_weights) {
        IMP_LOG_INFO("Warmup skipped (MXFP4 model)");
        return;
    }

    // Gemma-4 has outlier-heavy output_norm activations that amplify cuBLAS
    // algo jitter — warming up with BOS-filled buffers pins an algo that
    // produces wrong logits under real inputs and drives decode into
    // backtick/markdown degeneration. IMP_NO_WARMUP=1 was the manual
    // mitigation; make it automatic for the arch.
    if (model_->config().arch == ModelArch::GEMMA4) {
        IMP_LOG_INFO("Warmup skipped (Gemma-4 algo-jitter protection)");
        return;
    }

    Tokenizer* tok = model_->tokenizer();
    int32_t warmup_id = tok ? tok->bos_id() : 1;
    if (warmup_id < 0)
        warmup_id = 1;

    for (int prompt_len : {16, 32}) {
        auto req = std::make_shared<Request>();
        req->id = next_request_id_++;
        req->input_tokens.resize(prompt_len, warmup_id);
        req->max_tokens = 2;
        req->temperature = 0.0f;
        req->ignore_eos = true;
        scheduler_->add_request(req);

        for (int i = 0; i < 8 && req->status != RequestStatus::FINISHED; i++)
            (void)step();

        kv_manager_->free_sequence(req->id);
        reset_ssm_state(req->id);
        while (kv_manager_->evict_cached_block()) {}
        req->status = RequestStatus::CANCELLED;
    }

    for (int i = 0; i < kMaxGraphPoolSize; i++)
        decode_graph_pool_[i].invalidate();
    decode_batch_pool_.reset_upload_cache();
    if (async_graph_runner_.is_setup())
        async_graph_runner_.cleanup();
    if (async_d_block_tables_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
        async_d_block_tables_ = nullptr;
    }
    if (async_d_banned_tokens_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
        async_d_banned_tokens_ = nullptr;
    }
    async_graph_req_ = nullptr;
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;
    cudaDeviceSynchronize();
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            IMP_LOG_ERROR("warmup CUDA error: %s", cudaGetErrorString(e));
    }
    // Clear any stale CUDA errors from warmup (e.g. green context reconfigure
    // failure on consumer GPUs — the error propagates to cuBLAS otherwise).
    cudaGetLastError();
    cudaDeviceSynchronize();  // ensure all weight upload/dequant kernels are done
    // Drop FP8 KV calibrated_ flags so the first real prefill re-runs absmax
    // and promotes the per-layer scale via high-water-mark. Warmup uses
    // synthetic BOS tokens whose K/V absmax is unrepresentative; without this
    // reset, Llama-3.2-3B with --kv-fp8 degenerated to " France, and, 2008,
    // 201, 201, …" within 30 tokens. The high-water-mark logic in
    // executor_kv_write.cu (FP8 path) keeps the scale monotonically
    // non-decreasing, so warmup's contribution survives if it was already
    // wider than real prefill (Qwen3 case), and real prefill widens it
    // further when needed (Llama case).
    if (executor_)
        executor_->reset_kv_calibration();
    IMP_LOG_INFO("Warmup complete");
}

// =====================================================================
// step() — main inference loop
// =====================================================================

bool Engine::step() {
    // Fast path: async conditional graph loop completed on GPU.
    int async_result = step_async_graph_resume();
    if (async_result == 1)
        return true;  // still running
    if (async_result == -1) {
        return scheduler_->has_pending() || scheduler_->active_count() > 0;
    }

    // Schedule prefill/decode batches and reconfigure green contexts.
    if (!step_schedule())
        return false;

    // Process prefill requests.
    if (!sched_prefill_batch_.empty()) {
        step_prefill(prefill_stream());
        ensure_prefill_workspace(executor_.get());
    }

    // Process decode requests (batched).
    if (!sched_decode_batch_.empty()) {
        step_decode(decode_stream());
        ensure_prefill_workspace(executor_.get());
    }

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
}

// =====================================================================
// step_async_graph_resume — handle async conditional graph loop
// Returns: 0 = no graph active, 1 = still running, -1 = generation done
// =====================================================================

int Engine::step_async_graph_resume() {
    if (async_graph_runner_.is_setup() && async_graph_req_) {
        auto& req = async_graph_req_;

        if (async_pending_tokens_.empty() && async_pending_cursor_ == 0) {
            cudaStream_t dec_stream = decode_stream();
            async_pending_tokens_ = async_graph_runner_.wait_and_get_tokens(dec_stream);
        }

        int32_t token = -1;
        if (async_pending_cursor_ < static_cast<int>(async_pending_tokens_.size())) {
            token = async_pending_tokens_[async_pending_cursor_++];
        }

        bool generation_done = false;
        if (token >= 0) {
            req->output_tokens.push_back(token);
            track_think_state(*req, token);
            bool is_stop = should_stop(*req, token);
            generation_done = is_stop || static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (!generation_done)
                return 1;
        }

        auto saved_req = async_graph_req_;

        async_graph_runner_.cleanup();
        if (async_d_block_tables_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
            async_d_block_tables_ = nullptr;
        }
        if (async_d_banned_tokens_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
            async_d_banned_tokens_ = nullptr;
        }
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;

        if (generation_done) {
            finish_request(saved_req);
            return -1;
        }

        IMP_LOG_DEBUG("AsyncGraphLoop: graph tokens exhausted, continuing with step decode");
    }

    // Clean up stale async graph state
    if (async_graph_req_ && !async_graph_runner_.is_setup()) {
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;
    }

    return 0;
}

// =====================================================================
// step_schedule — call scheduler, reconfigure green contexts
// Returns true if there is work to do.
// =====================================================================

bool Engine::step_schedule() {
    sched_prefill_batch_.clear();
    sched_decode_batch_.clear();
    scheduler_->schedule(sched_prefill_batch_, sched_decode_batch_);

    if (sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
        return false;
    }

    // Dynamic Green Context SM reconfiguration
    if (config_.use_green_contexts && green_ctx_.is_available() && green_ctx_.has_green_contexts()) {
        float target_ratio = config_.green_ctx_prefill_ratio;
        if (sched_prefill_batch_.empty() && !sched_decode_batch_.empty()) {
            target_ratio = 0.0f;
        } else if (!sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
            target_ratio = 1.0f;
        }
        if (std::abs(target_ratio - green_ctx_.prefill_ratio()) > 0.1f) {
            green_ctx_.reconfigure(target_ratio);
        }
    }

    return true;
}

// =====================================================================
// supports_chunked_prefill_ / resolve_prefill_chunk_size_
// Whether the model arch + KV dtype combination supports chunked prefill.
// Returns true for full-attention models (Qwen3, Llama, Mistral) and hybrid
// GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) with FP16, FP8, or
// NVFP4 KV cache. Returns false for SWA models (Gemma-3/4, Llama-4) and
// sub-byte KV dtypes (INT4, TurboQuant) lacking gather kernels.
// =====================================================================

bool Engine::supports_chunked_prefill_() const {
    if (!model_)
        return false;
    const auto& cfg = model_->config();
    // Out-of-scope archs. Hybrid GDN+MoE / Mamba2+MoE archs (QWEN35*,
    // QWEN36_MOE, NEMOTRON_H_MOE) ARE supported. Gemma-4 (SWA + dual
    // head_dim 256/512) is now supported via:
    //   - cuBLAS softmax sliding_window param (PR feat(attn): sliding_window),
    //   - per-layer dispatch through the rectangular cuBLAS prefill path
    //     (every layer call uses its own nh/nkv/hd from layer-local vars).
    if (cfg.arch == ModelArch::GEMMA3) return false;       // SWA, no test model
    if (cfg.arch == ModelArch::LLAMA4) return false;       // MoE + SWA, untested
    // Per-layer attention shape uniformity gate. Hybrid archs (QWEN35*, QWEN36_MOE,
    // NEMOTRON_H_MOE) populate n_kv_heads_per_layer with zeros for non-attention
    // layers — uniformity here means all *nonzero* values agree. Truly heterogeneous
    // shapes (Gemma-4 dual head_dim 256/512) are now allowed because the chunked
    // path dispatches per-layer with the correct nh/nkv/hd from layer-local vars.
    auto first_nonzero_int = [](const std::vector<int>& v) -> int {
        for (int x : v) if (x > 0) return x;
        return 0;
    };
    auto any_nonzero_differs = [](const std::vector<int>& v, int ref) -> bool {
        for (int x : v) if (x > 0 && x != ref) return true;
        return false;
    };
    if (!cfg.n_kv_heads_per_layer.empty()) {
        int ref = first_nonzero_int(cfg.n_kv_heads_per_layer);
        if (ref > 0 && any_nonzero_differs(cfg.n_kv_heads_per_layer, ref)) {
            // Allow Gemma-4: per-layer dispatch covers heterogeneous shapes.
            if (cfg.arch != ModelArch::GEMMA4) return false;
        }
    }
    if (!cfg.head_dim_per_layer.empty()) {
        int ref = first_nonzero_int(cfg.head_dim_per_layer);
        if (ref > 0 && any_nonzero_differs(cfg.head_dim_per_layer, ref)) {
            if (cfg.arch != ModelArch::GEMMA4) return false;
        }
    }
    // KV dtypes wired through paged_kv_gather: FP16, FP8_E4M3, NVFP4, MXFP4_KV, INT4.
    // INT8 and TurboQuant variants would need their own gather kernels.
    if (kv_cache_raw_) {
        QType kvt = kv_cache_raw_->qtype();
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3 &&
            kvt != QType::NVFP4 && kvt != QType::MXFP4_KV && kvt != QType::INT4)
            return false;
    }
    return true;
}

int Engine::resolve_prefill_chunk_size_() const {
    int explicit_val = config_.prefill_chunk_size;
    if (explicit_val < 0) {
        return supports_chunked_prefill_() ? 512 : 0;
    }
    if (explicit_val == 0)
        return 0;
    // explicit_val > 0
    if (!supports_chunked_prefill_()) {
        IMP_LOG_WARN(
            "prefill_chunk_size=%d ignored: arch=%d / kv_dtype=%d not in chunked-prefill scope; using 0",
            explicit_val, (int)model_->config().arch,
            kv_cache_raw_ ? (int)kv_cache_raw_->qtype() : -1);
        return 0;
    }
    return explicit_val;
}

// =====================================================================
// step_prefill — process all prefill requests
// =====================================================================

void Engine::step_prefill(cudaStream_t stream) {
    int resolved    = resolve_prefill_chunk_size_();
    int effective_chunk = (resolved > 0) ? resolved : executor_->max_tokens();
    // Hard cap: chunk size must never exceed the executor's max_tokens
    // (which is itself capped to 256 for SSM/GDN+MoE hybrids and 512 for
    // dense GDN to bound workspace VRAM). Without this clamp, a server-side
    // prefill_chunk_size default of 512 (handlers.cpp) overflows the
    // workspace and crashes with `n_tokens (X) exceeds max_tokens (Y)` →
    // `terminate: reshape: numel mismatch` on long prompts to e.g. Qwen3.6.
    if (effective_chunk > executor_->max_tokens()) {
        effective_chunk = executor_->max_tokens();
    }
    if (kv_manager_) {
        int bs = kv_manager_->kv_cache()->block_size();
        if (effective_chunk > bs)
            effective_chunk = (effective_chunk / bs) * bs;
    }

    for (auto& req : sched_prefill_batch_) {
        step_prefill_one(req, effective_chunk, stream);
        kv_manager_->touch(req->id);
    }
}

// =====================================================================
// step_prefill_one — process a single prefill request
// =====================================================================

void Engine::step_prefill_one(std::shared_ptr<Request>& req, int effective_chunk, cudaStream_t pf_stream) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    int total_input = static_cast<int>(req->input_tokens.size());
    int offset = req->prefill_offset;

    // Out-of-scope archs (Gemma-3/4 SWA, Llama-4, sub-byte KV) lack a paged
    // chunked-prefill path, so the chunked-prefill branch in
    // executor_attention.cu aborts on chunk 2+ (q_offset > 0 + per_layer
    // shapes). Reject prompts > effective_chunk gracefully here instead of
    // letting them hit std::abort. Real fix is the paged hybrid-prefill
    // kernel (roadmap).
    if (offset == 0 && total_input > effective_chunk && !supports_chunked_prefill_()) {
        IMP_LOG_ERROR(
            "Prompt has %d tokens but max_tokens=%d on hybrid/out-of-scope arch — "
            "chunked prefill not supported. Cancelling request %d.",
            total_input, effective_chunk, req->id);
        req->status = RequestStatus::CANCELLED;
        return;
    }

    // Clamp effective_chunk so n × ctx_len ≤ s_cap² where s_cap is the
    // attn_scores_ workspace dimension. Worst case is the final chunk where
    // ctx_len ≈ total_input. Without this, long prompts on hybrid models
    // (Qwen3.5/3.6 GDN, Nemotron-H) hit "attn_scores_ capacity too small"
    // abort in executor_attention.cu — see chunked_prefill_attn_scores_capacity_bug.
    if (executor_) {
        int s_cap = executor_->attn_scores_cap();
        if (s_cap > 0 && total_input > 0) {
            int64_t cap2 = static_cast<int64_t>(s_cap) * s_cap;
            int max_chunk_for_buf = static_cast<int>(cap2 / total_input);
            max_chunk_for_buf = (max_chunk_for_buf / kv_bs) * kv_bs;
            if (max_chunk_for_buf > 0 && effective_chunk > max_chunk_for_buf) {
                effective_chunk = max_chunk_for_buf;
            }
        }
    }

    // Determine chunk boundaries
    int chunk_len = total_input - offset;
    bool is_last_chunk = true;
    if (chunk_len > effective_chunk) {
        chunk_len = effective_chunk;
        is_last_chunk = false;
    }

    int ctx_len = offset + chunk_len;
    (void)executor_->resize_workspace(chunk_len, pf_stream);

    int num_blocks = (ctx_len + kv_bs - 1) / kv_bs;

    // Allocate KV cache blocks
    int prefix_reused = 0;
    int existing = static_cast<int>(kv_manager_->block_table(req->id).size());

    if (kv_manager_->prefix_caching_enabled() && existing == 0 && offset == 0) {
        int total_blocks_needed = (total_input + kv_bs - 1) / kv_bs;
        prefix_reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens);
        if (prefix_reused < 0) {
            while (kv_manager_->num_free_blocks() < total_blocks_needed) {
                int evicted = kv_manager_->evict_lru();
                if (evicted < 0)
                    break;
            }
            prefix_reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens);
            if (prefix_reused < 0) {
                req->status = RequestStatus::CANCELLED;
                return;
            }
        }

        if (prefix_reused > 0) {
            int effective_reused = (prefix_reused > 1) ? prefix_reused - 1 : 0;
            int skip_tokens = effective_reused * kv_bs;
            if (skip_tokens >= total_input) {
                skip_tokens = (total_input / kv_bs) * kv_bs;
                if (skip_tokens >= total_input) {
                    skip_tokens = total_input - 1;
                }
            }
            if (skip_tokens > offset) {
                IMP_LOG_INFO("PrefixCache: seq %d skipping %d/%d prefill tokens (%d blocks reused)", req->id,
                             skip_tokens, total_input, prefix_reused);
                req->cached_tokens = skip_tokens;
                offset = skip_tokens;
                req->prefill_offset = offset;
                chunk_len = total_input - offset;
                is_last_chunk = true;
                if (chunk_len > effective_chunk) {
                    chunk_len = effective_chunk;
                    is_last_chunk = false;
                }
                ctx_len = offset + chunk_len;
                (void)executor_->resize_workspace(chunk_len, pf_stream);
            }
        }
    } else {
        int additional = num_blocks - existing;
        if (additional > 0) {
            if (!kv_manager_->allocate_blocks(req->id, additional)) {
                while (kv_manager_->num_free_blocks() < additional) {
                    int evicted = kv_manager_->evict_lru();
                    if (evicted < 0)
                        break;
                }
                if (!kv_manager_->allocate_blocks(req->id, additional)) {
                    kv_manager_->free_sequence(req->id);
                    req->status = RequestStatus::CANCELLED;
                    return;
                }
            }
        }
    }

    const auto& block_table = kv_manager_->block_table(req->id);

    // Upload prefill metadata to device (pre-allocated pool or fallback malloc)
    int32_t* d_token_ids = nullptr;
    int* d_positions = nullptr;
    int* d_block_tables = nullptr;
    int* d_context_lens = nullptr;
    bool pf_pool_used = false;

    auto check = [&req](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Engine::step prefill %s failed: %s", op, cudaGetErrorString(err));
            req->status = RequestStatus::CANCELLED;
        }
        return err == cudaSuccess;
    };

    if (prefill_pool_ && chunk_len <= config_.max_seq_len) {
        d_token_ids = d_pf_token_ids_;
        d_positions = d_pf_positions_;
        d_block_tables = d_pf_block_tables_;
        d_context_lens = d_pf_context_lens_;
        pf_pool_used = true;
    } else {
        if (!check(cudaMallocAsync(&d_token_ids, chunk_len * sizeof(int32_t), pf_stream),
                   "malloc token_ids") ||
            !check(cudaMallocAsync(&d_positions, chunk_len * sizeof(int), pf_stream), "malloc positions") ||
            !check(cudaMallocAsync(&d_block_tables, block_table.size() * sizeof(int), pf_stream),
                   "malloc block_tables") ||
            !check(cudaMallocAsync(&d_context_lens, sizeof(int), pf_stream), "malloc context_lens")) {
            if (d_token_ids)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, pf_stream));
            if (d_positions)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, pf_stream));
            if (d_block_tables)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, pf_stream));
            if (d_context_lens)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, pf_stream));
            kv_manager_->free_sequence(req->id);
            return;
        }
    }

    // Use pinned staging buffers when available (avoids internal pageable->pinned copy)
    if (h_pf_token_ids_ && chunk_len <= config_.max_seq_len) {
        memcpy(h_pf_token_ids_, req->input_tokens.data() + offset, chunk_len * sizeof(int32_t));
        check(cudaMemcpyAsync(d_token_ids, h_pf_token_ids_, chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    } else {
        check(cudaMemcpyAsync(d_token_ids, req->input_tokens.data() + offset, chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    }

    if (h_pf_positions_ && chunk_len <= config_.max_seq_len) {
        for (int i = 0; i < chunk_len; i++)
            h_pf_positions_[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, h_pf_positions_, chunk_len * sizeof(int), cudaMemcpyHostToDevice,
                              pf_stream),
              "memcpy positions");
    } else {
        std::vector<int> positions(chunk_len);
        for (int i = 0; i < chunk_len; i++)
            positions[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, positions.data(), chunk_len * sizeof(int), cudaMemcpyHostToDevice,
                              pf_stream),
              "memcpy positions");
    }

    check(cudaMemcpyAsync(d_block_tables, block_table.data(), block_table.size() * sizeof(int),
                          cudaMemcpyHostToDevice, pf_stream),
          "memcpy block_tables");
    check(cudaMemcpyAsync(d_context_lens, &ctx_len, sizeof(int), cudaMemcpyHostToDevice, pf_stream),
          "memcpy context_lens");

    // Build InferenceState
    InferenceState state;
    state.token_ids = d_token_ids;
    state.positions = d_positions;
    state.n_tokens = chunk_len;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_block_tables;
    state.context_lens = d_context_lens;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = 0;
    state.is_prefill = true;
    state.prefill_offset = offset;  // absolute pos of state.positions[0]
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) {
        // Slot lookup happens inside KVCacheManager::residual_k_ptr; if no
        // slot is allocated yet (prefill before first decode), the residual
        // pointers return nullptr and the kernel skips the residual pass.
        state.kv_seq_id = req->id;
    }
    fill_sampling_params(*req, state);

    // Constraints via ConstraintManager
    constraints_.prepare(req->json_mode, req->json_schema, model_->tokenizer(), req->has_tools,
                         req->tpl_family);
    state.json_constrainer = constraints_.json_constrainer();
    state.schema_constrainer = constraints_.schema_constrainer();

    // Penalties
    upload_penalties(*req, state, pf_stream);

    // Recurrent state (SSM/GDN)
    // Reset on the first chunk of a new request so previous-request state
    // doesn't leak in.  Subsequent chunks must NOT reset — the recurrent
    // state built during earlier chunks must carry forward.
    fill_recurrent_state(*req, state, /*reset=*/(offset == 0), pf_stream);

    // Vision embeddings on first chunk
    if (vision_.has_input() && vision_.is_available() && offset == 0) {
        state.vision_embeddings = vision_.embeddings();
        state.vision_token_id = vision_.soft_token_id();
        state.n_vision_tokens = vision_.num_image_tokens();
    }

    if (!is_last_chunk) {
        if (executor_->has_decode_workspace()) {
            executor_->use_workspace(0);
        }
        Tensor logits_out;

        // Prefill graph capture (opt-in, Phase 4 of MoE-prefill-graphs work).
        // Conditions: env-gated, pool path (stable device buffers), and
        // chunk shape stable (in practice all non-last chunks share chunk_len
        // = prefill_chunk_size). H2D upload happened above on pf_stream
        // *before* this wrapper — captured region is forward_logits only,
        // analogous to the decode graph pattern.
        const bool prefill_graph_enabled = RuntimeConfig::current().runtime.prefill_graph;
        const bool can_capture = prefill_graph_enabled && pf_pool_used && config_.use_cuda_graphs;
        if (can_capture) {
            const int block_count = static_cast<int>(block_table.size());
            if (chunk_len != last_prefill_chunk_len_ || block_count != last_prefill_block_count_) {
                prefill_graph_runner_.invalidate_for_update();
                last_prefill_chunk_len_ = chunk_len;
                last_prefill_block_count_ = block_count;
            }
            prefill_graph_runner_.set_decode_fn([this, &state, &logits_out](cudaStream_t s) {
                executor_->forward_logits(state, logits_out, s);
            });
            prefill_graph_runner_.execute(pf_stream);
            if (logits_out.data == nullptr) {
                logits_out = executor_->get_logits_view(/*n_sequences=*/1);
            }
        } else {
            executor_->forward_logits(state, logits_out, pf_stream);
        }

        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
        }

        req->prefill_offset = offset + chunk_len;
        IMP_LOG_DEBUG("Chunked prefill: req %d chunk [%d, %d) of %d", req->id, offset, offset + chunk_len,
                      total_input);
    } else {
        // Last chunk: forward + sample
        int32_t next_token;
        bool use_event_sync = (h_sample_pinned_ != nullptr && executor_->d_sample_result() != nullptr &&
                               (state.temperature <= 0.0f || state.top_k == 1) && !req->logprobs &&
                               !state.json_constrainer && !state.schema_constrainer);

        Tensor prefill_logits_out;

        if (use_event_sync) {
            Tensor logits_out;
            executor_->forward_logits(state, logits_out, pf_stream);
            Tensor last_logits = logits_out.slice(0, 1);
            int64_t vocab_shape[1] = {last_logits.shape[1]};
            last_logits = last_logits.reshape(1, vocab_shape);

            // Ban special tokens (e.g. Gemma-4 <|channel>) before greedy
            // argmax — otherwise the natural-argmax channel marker triggers
            // is_stop_token and the request finishes with 0 completion
            // tokens. Same logic as GraphExecutor::forward (executor.cu:88)
            // and apply_pre_sample (executor.cu) but inline here because
            // sample_greedy_device runs on raw logits without going through
            // either of those wrappers.
            if (state.banned_tokens != nullptr && state.n_banned_tokens > 0) {
                float* lp = static_cast<float*>(last_logits.data);
                int vocab = static_cast<int>(last_logits.shape[0]);
                float neg_inf = -1e30f;
                for (int bi = 0; bi < state.n_banned_tokens; bi++) {
                    int32_t tid = state.banned_tokens[bi];
                    if (tid >= 0 && tid < vocab) {
                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(lp + tid, &neg_inf, sizeof(float),
                                                           cudaMemcpyHostToDevice, pf_stream));
                    }
                }
            }

            sample_greedy_device(last_logits, executor_->d_sample_result(), h_sample_pinned_, pf_stream);

            if (!prefill_done_)
                (void)prefill_done_.create();
            cudaEventRecord(prefill_done_, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }

            cudaEventSynchronize(prefill_done_);
            next_token = *h_sample_pinned_;
        } else if (req->logprobs) {
            executor_->forward_logits(state, prefill_logits_out, pf_stream);
            auto sampled = executor_->sample_from_logits(prefill_logits_out, state, pf_stream);
            next_token = sampled[0];

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }
        } else {
            next_token = executor_->forward(state, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }
        }

        if (req->mirostat == 2)
            req->mirostat_mu = state.mirostat_mu;

        // Extract logprobs
        if (req->logprobs && prefill_logits_out.data != nullptr) {
            int vocab_size = static_cast<int>(prefill_logits_out.shape[prefill_logits_out.ndim - 1]);
            executor_->ensure_logits_pinned(vocab_size);

            const float* d_logits = static_cast<const float*>(prefill_logits_out.data);
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(executor_->h_logits_pinned(), d_logits,
                                               vocab_size * sizeof(float), cudaMemcpyDeviceToHost,
                                               pf_stream));
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(pf_stream));

            req->output_logprobs.push_back(build_logprob_info(executor_->h_logits_pinned(), vocab_size,
                                                              next_token, req->top_logprobs,
                                                              model_->tokenizer()));
        }

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        Tokenizer* tok = model_->tokenizer();
        IMP_LOG_DEBUG("Prefill -> token %d (ctx=%d): id=%d [%s]", (int)req->output_tokens.size(),
                      req->context_len(), next_token, tok->decode_token(next_token).c_str());

        // MTP prefill: populate MTP KV cache with all prompt position hidden
        // states so the head enters decode with the same context as the main
        // model. Only the LAST chunk (where we have the complete tail of the
        // prompt in executor's hidden_ buffer) and only the non-chunked path
        // for now (chunked prefill would need per-chunk capture). Cost: ~n
        // extra MTP forwards, one-time per session.
        if (mtp_spec_decode_enabled() && offset == 0 && req->prefill_offset == 0) {
            // executor's hidden_ buffer holds [chunk_len, d_model] FP16 right
            // after forward_logits; that matches the whole prompt when
            // offset==0 and is_last_chunk==true.
            const int n_prompt = chunk_len;
            imp::Tensor h_view = executor_->view_hidden(n_prompt);
            if (h_view.data != nullptr) {
                mtp_prefill_prompt(req->input_tokens.data(), h_view.data, n_prompt);
            }
        }

        // Update constraint FSM
        constraints_.update(next_token);

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        } else {
            req->status = RequestStatus::DECODING;
            if (kv_manager_->prefix_caching_enabled()) {
                kv_manager_->register_block_hashes(req->id, req->input_tokens);
            }
        }
    }
}

// =====================================================================
// step_decode — process all decode requests (batched)
// =====================================================================

void Engine::step_decode(cudaStream_t dec_stream) {
    auto& decode_batch = sched_decode_batch_;
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // SSM/GDN: limit decode batch to 1 sequence
    if ((ssm_state_ || gdn_state_) && decode_batch.size() > 1) {
        decode_batch.resize(1);
    }

    // Allocate new KV blocks where needed
    valid_decode_.clear();
    auto& valid_decode = valid_decode_;

    for (auto& req : decode_batch) {
        int ctx_len = req->context_len();
        int blocks_needed = (ctx_len + kv_bs - 1) / kv_bs;
        const auto& block_table = kv_manager_->block_table(req->id);
        int blocks_have = static_cast<int>(block_table.size());

        if (blocks_needed > blocks_have) {
            int new_block = kv_manager_->append_block(req->id);
            if (new_block < 0) {
                int evicted = kv_manager_->evict_lru();
                if (evicted >= 0) {
                    new_block = kv_manager_->append_block(req->id);
                }
                if (new_block < 0) {
                    kv_manager_->free_sequence(req->id);
                    req->status = RequestStatus::CANCELLED;
                    continue;
                }
            }
        }

        // StreamingLLM smart KV cache: once context exceeds the threshold,
        // free middle blocks while keeping sinks + window. The decode kernel
        // skips the freed (-1 sentinel) slots via its own n_sinks logic.
        if (config_.streaming_kv_enabled) {
            int n_sinks = (config_.streaming_kv_n_sinks > 0) ? config_.streaming_kv_n_sinks : 4;
            int win = (config_.streaming_kv_window > 0) ? config_.streaming_kv_window
                                                        : model_->config().sliding_window;
            if (n_sinks > 0 && win > 0) {
                int threshold = (config_.streaming_kv_threshold > 0) ? config_.streaming_kv_threshold
                                                                     : (n_sinks + win + 2 * kv_bs);
                if (req->context_len() > threshold) {
                    kv_manager_->evict_middle_blocks(req->id, n_sinks, win);
                }
            }
        }
        valid_decode.push_back(req);
    }

    if (!valid_decode.empty()) {
        step_decode_forward(valid_decode, dec_stream);
    }
}

// =====================================================================
// step_decode_forward — build batch, run forward pass, sample, process
// =====================================================================

void Engine::step_decode_forward(std::vector<std::shared_ptr<Request>>& valid_decode,
                                 cudaStream_t dec_stream) {
    // Switch workspace for decode
    if (executor_->has_decode_workspace() && valid_decode.size() == 1) {
        executor_->use_workspace(1);
    } else {
        if (executor_->active_workspace() == 1)
            executor_->use_workspace(0);
        (void)executor_->resize_workspace(static_cast<int>(valid_decode.size()), dec_stream);
    }

    // Build batched decode
    decode_builder_.reset();

    int max_ctx = 0;
    for (auto& req : valid_decode) {
        int ctx_len = req->context_len();
        max_ctx = std::max(max_ctx, ctx_len);

        int32_t last_token = req->output_tokens.empty() ? req->input_tokens.back()
                                                        : req->output_tokens.back();
        int position = ctx_len - 1;

        const auto& bt = kv_manager_->block_table(req->id);
        decode_builder_.add_decode_sequence(last_token, position, bt.data(), static_cast<int>(bt.size()),
                                            ctx_len);
    }

    Batch batch = decode_builder_.build();

    // Upload to GPU using pre-allocated pool
    GPUBatch gpu_batch;
    if (decode_batch_pool_.is_allocated()) {
        int pool_max = decode_batch_pool_.max_blocks_per_seq();
        if (batch.max_blocks_per_seq < pool_max) {
            int n_seq = batch.n_sequences;
            int old_stride = batch.max_blocks_per_seq;
            size_t needed = static_cast<size_t>(n_seq) * pool_max;
            padded_block_table_.resize(needed);
            std::memset(padded_block_table_.data(), 0, needed * sizeof(int));
            for (int s = 0; s < n_seq; s++) {
                for (int b = 0; b < old_stride; b++) {
                    padded_block_table_[s * pool_max + b] = batch.block_tables[s * old_stride + b];
                }
            }
            batch.block_tables.swap(padded_block_table_);
            batch.max_blocks_per_seq = pool_max;
        }
        gpu_batch = decode_batch_pool_.upload_into_pool(batch, dec_stream);
    } else {
        gpu_batch.upload(batch, dec_stream);
    }

    // Build InferenceState
    InferenceState state;
    state.token_ids = gpu_batch.d_token_ids;
    state.positions = gpu_batch.d_positions;
    state.n_tokens = gpu_batch.total_tokens;
    state.n_sequences = gpu_batch.n_sequences;
    state.max_blocks_per_seq = gpu_batch.max_blocks_per_seq;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = gpu_batch.d_block_tables;
    state.context_lens = gpu_batch.d_context_lens;
    state.max_context_len = max_ctx;
    state.is_prefill = false;
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) {
        // Allocate / refresh per-seq residual metadata for this decode step.
        // Slot allocation is idempotent (returns existing slot on re-call).
        const int N = gpu_batch.n_sequences;
        residual_meta_h_seq_ids_.resize(N);
        for (int i = 0; i < N; i++) {
            int sid = valid_decode[i]->id;
            residual_meta_h_seq_ids_[i] = sid;
            kv_manager_->allocate_residual_slot(sid);
        }
        if (N == 1) {
            // Single-seq path: kernel reads ring state from kv_manager's
            // persistent device buffers. Slot is a constant per-request
            // value uploaded into d_kv_slot_buf_[0]; only re-uploads when
            // it changes (i.e. when the active request rotates).
            state.kv_seq_id = valid_decode[0]->id;
            state.h_residual_seq_ids = residual_meta_h_seq_ids_.data();
            int slot_for_req = kv_manager_->residual_slot_of(valid_decode[0]->id);
            if (d_kv_slot_buf_ != nullptr) {
                if (d_kv_slot_last_uploaded_.empty() ||
                    d_kv_slot_last_uploaded_[0] != slot_for_req) {
                    cudaMemcpyAsync(d_kv_slot_buf_, &slot_for_req, sizeof(int),
                                    cudaMemcpyHostToDevice, dec_stream);
                    if (d_kv_slot_last_uploaded_.empty()) d_kv_slot_last_uploaded_.assign(1, -1);
                    d_kv_slot_last_uploaded_[0] = slot_for_req;
                }
                state.d_residual_seq_slots = d_kv_slot_buf_;
            }
        } else {
            // Multi-seq path: build per-batch metadata arrays + upload to
            // a per-step device buffer.
            residual_meta_h_slots_.resize(N);
            residual_meta_h_counts_.resize(N);
            residual_meta_h_widxes_.resize(N);
            for (int i = 0; i < N; i++) {
                int sid = residual_meta_h_seq_ids_[i];
                residual_meta_h_slots_[i] = kv_manager_->residual_slot_of(sid);
                auto rs = kv_manager_->residual_state(sid);
                residual_meta_h_counts_[i] = rs.fill_count;
                residual_meta_h_widxes_[i] = rs.write_idx;
            }
            const size_t meta_bytes = static_cast<size_t>(3) * N * sizeof(int);
            if (cudaMallocAsync(&residual_meta_d_buf_, meta_bytes, dec_stream) == cudaSuccess) {
                int* base = residual_meta_d_buf_;
                cudaMemcpyAsync(base + 0 * N, residual_meta_h_slots_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + 1 * N, residual_meta_h_counts_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + 2 * N, residual_meta_h_widxes_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                state.d_residual_seq_slots = base + 0 * N;
                state.d_residual_counts = base + 1 * N;
                state.d_residual_write_idxes = base + 2 * N;
                state.h_residual_seq_ids = residual_meta_h_seq_ids_.data();
            }
        }
    }
    fill_sampling_params(*valid_decode[0], state);

    // Derive per-step seed: mix request seed with output count so each
    // decode step gets a different random draw.  Without this, seed=-1
    // falls back to a fixed 42 on every step, producing identical RNG
    // values and causing repetition loops on long structured outputs.
    state.seed = compute_step_seed(*valid_decode[0]);

    // Penalties (single-sequence only)
    if (gpu_batch.n_sequences == 1) {
        upload_penalties(*valid_decode[0], state, dec_stream);
    }

    // Recurrent state
    fill_recurrent_state(*valid_decode[0], state, false, dec_stream);

    // Check if any request needs logprobs or constrained mode
    bool needs_logprobs = false;
    bool needs_json_mode = false;
    bool needs_schema_mode = false;
    for (const auto& r : valid_decode) {
        if (r->logprobs)
            needs_logprobs = true;
        if (r->json_mode && r->json_schema.empty())
            needs_json_mode = true;
        if (!r->json_schema.empty())
            needs_schema_mode = true;
    }

    // Schema/JSON constraints for decode (reuse state from prefill)
    if (needs_schema_mode && valid_decode.size() == 1 && !valid_decode[0]->json_schema.empty()) {
        if (constraints_.has_schema()) {
            state.schema_constrainer = constraints_.schema_constrainer();
        }
    }
    if (needs_json_mode && valid_decode.size() == 1 && valid_decode[0]->json_mode) {
        // Lazily init if needed (decode might be first step with json_mode)
        if (!constraints_.has_json() && !constraints_.has_schema()) {
            constraints_.prepare(true, "", model_->tokenizer());
        }
        state.json_constrainer = constraints_.json_constrainer();
    }

    // Per-request sampling lambda
    auto sample_per_request = [&](const Tensor& logits) -> std::vector<int32_t> {
        int n = static_cast<int>(valid_decode.size());

        if (n == 1) {
            auto& req = valid_decode[0];
            int32_t tok = executor_->sample_single_from_logits(logits, state, dec_stream);
            if (state.mirostat == 2)
                req->mirostat_mu = state.mirostat_mu;
            return {tok};
        }

        std::vector<int32_t> result(n);
        for (int i = 0; i < n; i++) {
            auto& req = valid_decode[i];
            InferenceState per_state = state;
            fill_sampling_params(*req, per_state);
            // Per-step seed (same fix as single-sequence path)
            per_state.seed = compute_step_seed(*req);
            per_state.penalty_tokens = nullptr;
            per_state.n_penalty_tokens = 0;
            bool req_needs_pen = (req->repetition_penalty != 1.0f || req->frequency_penalty != 0.0f ||
                                  req->presence_penalty != 0.0f);
            if (req_needs_pen && !req->output_tokens.empty() && d_penalty_tokens_) {
                size_t rn = req->output_tokens.size();
                if (rn <= d_penalty_tokens_capacity_) {
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_penalty_tokens_, req->output_tokens.data(),
                                                       rn * sizeof(int32_t), cudaMemcpyHostToDevice,
                                                       dec_stream));
                    per_state.penalty_tokens = d_penalty_tokens_;
                    per_state.n_penalty_tokens = static_cast<int>(rn);
                }
            }
            per_state.n_sequences = 1;
            Tensor seq_logits = logits.slice(i, i + 1);
            result[i] = executor_->sample_single_from_logits(seq_logits, per_state, dec_stream);
            if (per_state.mirostat == 2)
                req->mirostat_mu = per_state.mirostat_mu;
        }
        return result;
    };

    // Execute forward pass (piecewise CUDA Graph: forward in graph,
    // sampling always eager — per-batch-size graph pool avoids
    // re-capture when continuous batching changes batch size)
    std::vector<int32_t> tokens;
    Tensor decode_logits_out;

    const bool profiling = RuntimeConfig::current().diagnostics.profile;
    int graph_idx = gpu_batch.n_sequences - 1;
    if (config_.use_cuda_graphs && !profiling && gpu_batch.n_sequences > 0 && graph_idx < kMaxGraphPoolSize &&
        decode_batch_pool_.is_allocated()) {
        auto& graph_runner = decode_graph_pool_[graph_idx];

        // P5 §2.2 M4: pow-2 bucket the max_blocks_per_seq before comparing.
        // The decode_batch_pool_ padding (engine.cpp:2456-2468) already lifts
        // batch.max_blocks_per_seq to pool_max, so in practice this comparison
        // never trips during steady-state decode — but if a future path
        // bypasses the pool (e.g. spec-decode verify), pow-2 bucketing caps
        // the re-capture frequency to log2(max_blocks) events per decode
        // instead of one per 16-token boundary. Re-capture cost is
        // dominated by the eager forward that builds the new graph
        // (~5-10 ms on Qwen3-8B Q8_0) so the bucket pays back across long
        // contexts.
        auto bucket_pow2 = [](int x) -> int {
            if (x <= 1) return 1;
            int b = 1;
            while (b < x) b <<= 1;
            return b;
        };
        const int bucketed_max_blocks = bucket_pow2(gpu_batch.max_blocks_per_seq);
        if (bucketed_max_blocks != last_decode_max_blocks_per_graph_[graph_idx]) {
            // Topology stable across max_blocks growth (same kernels, only
            // grid dims / params differ) — cudaGraphExecUpdate handles this
            // without tearing down the exec + graph mem pool.
            graph_runner.invalidate_for_update();
            last_decode_max_blocks_per_graph_[graph_idx] = bucketed_max_blocks;
        }
        // Graph captures ONLY forward_logits — sampling runs eager after
        Tensor logits_out;
        graph_runner.set_decode_fn(
            [this, &state, &logits_out](cudaStream_t s) { executor_->forward_logits(state, logits_out, s); });
        graph_runner.execute(dec_stream);

        if (logits_out.data == nullptr) {
            logits_out = executor_->get_logits_view(gpu_batch.n_sequences);
        }
        // Eager sampling (handles all modes: greedy, top-k/p, penalties,
        // force_token, constraints, logprobs, mirostat)
        tokens = sample_per_request(logits_out);
        if (needs_logprobs)
            decode_logits_out = logits_out;
    } else {
        executor_->forward_logits(state, decode_logits_out, dec_stream);
        tokens = sample_per_request(decode_logits_out);
    }

    if (!decode_batch_pool_.is_allocated()) {
        gpu_batch.free();
    }

    // Free per-step residual metadata buffer (allocated in step_decode_forward
    // when residual is enabled). cudaFreeAsync orders behind the just-issued
    // forward + sample on dec_stream.
    if (residual_meta_d_buf_ != nullptr) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(residual_meta_d_buf_, dec_stream));
        residual_meta_d_buf_ = nullptr;
    }

    // Phase 3.5 telemetry: measure MTP-draft prediction accuracy without
    // changing generation. Single-sequence only (batch=1 simplifies hidden-
    // state addressing). Skipped when MTP is disabled or the workspace was
    // allocated without attention dims (older callers).
    if (mtp_spec_decode_enabled() && model_ && model_->mtp_.has_value() &&
        model_->mtp_->loaded && gpu_batch.n_sequences == 1 && !tokens.empty()) {
        const int32_t next_token = tokens[0];
        if (mtp_pending_prediction_ >= 0) {
            mtp_accuracy_.total++;
            bool match = (mtp_pending_prediction_ == next_token);
            if (match) mtp_accuracy_.matches++;
            // Optional verbose log: prints (predicted, actual, match) with
            // decoded strings so accept patterns can be analyzed offline.
            const bool s_pattern_log = RuntimeConfig::current().diagnostics.mtp_pattern_log;
            if (s_pattern_log) {
                Tokenizer* tok = model_->tokenizer();
                std::string ps = tok ? tok->decode_token(mtp_pending_prediction_) : std::string();
                std::string as = tok ? tok->decode_token(next_token) : std::string();
                IMP_LOG_INFO("MTP-PAT %s pred=%d '%s' actual=%d '%s'",
                             match ? "+" : "-",
                             mtp_pending_prediction_, ps.c_str(),
                             next_token, as.c_str());
            }
        }
        // K-chain measurement: verify pending predictions, then draft fresh.
        // Current position is the index of the token we just emitted (=
        // length of input + already-emitted output minus 1).
        const int cur_pos = valid_decode[0]->context_len() - 1;
        // Verify any pending predictions whose intended position is cur_pos.
        if (!mtp_pending_chain_.empty()) {
            // Drop stale (intended < cur_pos) — shouldn't happen in K=1 path
            // but possible if the engine skips/restarts mid-chain.
            mtp_pending_chain_.erase(
                std::remove_if(mtp_pending_chain_.begin(), mtp_pending_chain_.end(),
                               [cur_pos](const MtpChainEntry& e) {
                                   return e.intended_position < cur_pos;
                               }),
                mtp_pending_chain_.end());
            for (auto it = mtp_pending_chain_.begin(); it != mtp_pending_chain_.end();) {
                if (it->intended_position == cur_pos) {
                    if (static_cast<int>(mtp_chain_accept_.size()) <= it->lookahead) {
                        mtp_chain_accept_.resize(it->lookahead + 1);
                    }
                    mtp_chain_accept_[it->lookahead].total++;
                    if (it->prediction == next_token) {
                        mtp_chain_accept_[it->lookahead].matches++;
                    }
                    it = mtp_pending_chain_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        Tensor h_view = executor_->view_hidden(1);  // [1, d_model] FP16
        if (h_view.data != nullptr) {
            const int hidden_dim = model_->config_.d_model;
            const int vocab_size = model_->config_.vocab_size;

            // Optional: apply the main model's output norm before passing
            // h_prev to MTP. Upstream vllm passes post-RMSNorm hidden states
            // in some MTP variants; gate by env so we can A/B.
            const bool s_pre_norm_h = RuntimeConfig::current().diagnostics.mtp_prenorm_h;
            const void* h_for_mtp = h_view.data;
            // Scratch buffer for the normalized variant (allocated once).
            static void* s_h_normed = nullptr;
            if (s_pre_norm_h) {
                if (s_h_normed == nullptr) {
                    cudaMalloc(&s_h_normed, hidden_dim * sizeof(__half));
                }
                int64_t hd_shape[2] = {1, hidden_dim};
                Tensor in_view (h_view.data, QType::F16, 2, hd_shape, true);
                Tensor out_view(s_h_normed,  QType::F16, 2, hd_shape, true);
                imp::rmsnorm(in_view, model_->output_norm(), out_view,
                             model_->config_.rms_norm_eps, decode_stream(),
                             model_->config_.norm_weight_offset);
                h_for_mtp = s_h_normed;
            }

            // K-chain draft. K=mtp_spec_k_. For each step k=0..K-1:
            //   - input: (prev_token_k, h_prev_k)
            //   - output: prediction_k, ws.d_h_final updated for next iter
            //   - chain: prev_token_{k+1} = prediction_k, h_prev_{k+1} = d_h_final
            //
            // Cache roll-back: each chained call appends to MTP KV cache. Only
            // the first chain step's append should persist (it represents what
            // the main model actually does next). Roll back to mtp_pos_saved
            // after K-1 speculative steps so the real cache stays aligned.
            auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
            const int K = std::max(1, mtp_spec_k_);
            const int mtp_pos_before = ws->mtp_pos;
            int chain_prev_tok = next_token;
            const void* chain_h_prev = h_for_mtp;
            for (int k = 0; k < K; ++k) {
                int prediction = -1;
                if (!mtp_draft_one(chain_prev_tok, chain_h_prev, hidden_dim, vocab_size,
                                    &prediction)) {
                    break;
                }
                mtp_pending_chain_.push_back({prediction, k, cur_pos + 1 + k});
                // For k=0 only, also feed pending_prediction_ (legacy 1-step
                // accuracy counter remains in sync with chain_accept_[0]).
                if (k == 0) mtp_pending_prediction_ = prediction;
                // Chain: next iter uses this prediction + the MTP's own h_final.
                chain_prev_tok = prediction;
                chain_h_prev   = ws->d_h_final;
            }
            // Roll back the speculative cache writes from K-1 chained steps.
            // The first chained step (k=0) IS the real "next step" prediction
            // and matches what would have been drafted in K=1 mode — keep it.
            ws->mtp_pos = std::min(ws->mtp_pos, mtp_pos_before + 1);
        } else {
            mtp_pending_prediction_ = -1;
        }
    } else {
        mtp_pending_prediction_ = -1;  // batch>1 or MTP off → clear pending
    }

    // Process outputs: logprobs extraction + token distribution
    step_decode_process_outputs(valid_decode, tokens, decode_logits_out, needs_logprobs, needs_json_mode,
                                needs_schema_mode, dec_stream);
}

// =====================================================================
// step_decode_process_outputs — extract logprobs, distribute tokens,
//                                try async graph loop
// =====================================================================

void Engine::step_decode_process_outputs(std::vector<std::shared_ptr<Request>>& valid_decode,
                                         const std::vector<int32_t>& tokens, const Tensor& decode_logits_out,
                                         bool needs_logprobs, bool needs_json_mode, bool needs_schema_mode,
                                         cudaStream_t dec_stream) {
    Tokenizer* tok = model_->tokenizer();

    // Extract logprobs
    if (needs_logprobs && decode_logits_out.data != nullptr) {
        int vocab_size = static_cast<int>(decode_logits_out.shape[decode_logits_out.ndim - 1]);
        int n_lp = 0;
        for (const auto& r : valid_decode)
            if (r->logprobs)
                n_lp++;
        executor_->ensure_logits_pinned(vocab_size * n_lp);
        float* h_base = executor_->h_logits_pinned();

        int slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            if (!valid_decode[i]->logprobs)
                continue;
            const float* d_logits = static_cast<const float*>(decode_logits_out.data) +
                                    static_cast<size_t>(i) * vocab_size;
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_base + static_cast<size_t>(slot) * vocab_size, d_logits,
                                               vocab_size * sizeof(float), cudaMemcpyDeviceToHost,
                                               dec_stream));
            slot++;
        }
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(dec_stream));

        slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            auto& req = valid_decode[i];
            if (!req->logprobs)
                continue;

            float* h_logits = h_base + static_cast<size_t>(slot) * vocab_size;
            req->output_logprobs.push_back(
                build_logprob_info(h_logits, vocab_size, tokens[i], req->top_logprobs, tok));
            slot++;
        }
    }

    // Distribute sampled tokens back to requests
    for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
        auto& req = valid_decode[i];
        int32_t next_token = tokens[i];

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        IMP_LOG_DEBUG("Decode step %d (ctx=%d, pos=%d): id=%d [%s]", (int)req->output_tokens.size(),
                      req->context_len(), req->context_len() - 1, next_token,
                      tok->decode_token(next_token).c_str());

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        }

        constraints_.update(next_token);
        kv_manager_->touch(req->id);
    }

    // Try async graph loop after first decode step.
    // Think budget is now handled device-side in post_decode_step_kernel.
    if (decode_graph_pool_[0].is_ready() && valid_decode.size() == 1 && !offload_mgr_ &&
        config_.use_cuda_graphs && !async_graph_runner_.is_setup() && !needs_logprobs && !needs_json_mode &&
        !needs_schema_mode) {
        auto& dreq = valid_decode[0];
        // forward_decode_async only implements banned_tokens + rep/freq/presence
        // penalties device-side. Any sampling feature that requires host-side
        // logic (logit_bias, mirostat, typical_p, min_p, DRY) would be silently
        // skipped inside the captured graph — stay on the eager path instead.
        const bool async_compatible = dreq->logit_bias.empty() && dreq->mirostat == 0 &&
                                      dreq->dry_multiplier == 0.0f && dreq->min_p == 0.0f &&
                                      dreq->typical_p >= 1.0f &&
                                      // Phase 3.5 MTP telemetry hooks the per-step path; the async
                                      // conditional-graph loop bypasses it. Stay eager when MTP is on
                                      // so accuracy measurement covers the whole generation.
                                      !mtp_spec_decode_enabled();
        // Text-fallback think tracking: when <think>/</think> are not single
        // control-token IDs (NVFP4 SafeTensors for Qwen3 / Qwen3.5 / Qwen3.6
        // ship them as added_tokens with special=False, leaving think_end_id_
        // at -1), the graph kernel's device-side `in_think` predicate is
        // permanently false. eos_id then terminates the loop the moment the
        // model emits <|endoftext|> after an empty </think>, returning a
        // 3-token completion ("<", "answer", ">") with the actual answer
        // never sampled. Host-side track_think_state runs literal-string
        // matching per token but only fires in the eager step_decode path,
        // so route through it whenever the request started inside a think
        // block on a model that lacks single-token think markers.
        const bool needs_eager_for_text_think = dreq->in_think_block && think_end_id_ < 0;
        if (async_compatible && !needs_eager_for_text_think && dreq->status == RequestStatus::DECODING &&
            !dreq->output_tokens.empty()) {
            int32_t last_token = dreq->output_tokens.back();
            try_launch_async_graph_loop(dreq, last_token, dec_stream);
        }
    }
}

// =====================================================================
// generate()
// =====================================================================

std::string Engine::generate(const std::string& prompt, int max_tokens, float temperature, float top_p,
                             int top_k, int seed, bool apply_chat_template, float min_p,
                             float repetition_penalty, float frequency_penalty, float presence_penalty) {
    Tokenizer* tok = model_->tokenizer();
    if (!tok) {
        return "";
    }

    std::vector<int32_t> tokens;

    if (apply_chat_template && !chat_template_.is_raw()) {
        std::vector<ChatMessage> messages = {{"user", prompt}};
        if (vision_.has_input() && vision_.is_available()) {
            tokens = chat_template_.apply_with_image(*tok, messages, vision_.num_image_tokens());
        } else {
            tokens = chat_template_.apply(*tok, messages);
        }
        IMP_LOG_INFO("Applied %s chat template (%zu tokens%s)",
                     chat_template_family_name(chat_template_.family()), tokens.size(),
                     vision_.has_input() ? ", with image" : "");
    } else {
        tokens = tok->encode(prompt);
        if (tok->add_bos() && (tokens.empty() || tokens[0] != tok->bos_id())) {
            tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
        }
    }

    IMP_LOG_INFO("Encoded %zu tokens", tokens.size());
    {
        std::string dump;
        for (size_t i = 0; i < tokens.size() && i < 64; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%d", tokens[i]);
            dump += buf;
            if (i + 1 < tokens.size())
                dump += ", ";
        }
        if (tokens.size() > 64)
            dump += "...";
        IMP_LOG_INFO("Token IDs: [%s]", dump.c_str());
    }

    auto req = std::make_shared<Request>();
    req->id = next_request_id_++;
    req->input_tokens = std::move(tokens);
    req->max_tokens = max_tokens;
    req->temperature = temperature;
    req->top_p = top_p;
    req->top_k = top_k;
    req->seed = seed;
    req->min_p = min_p;
    req->repetition_penalty = repetition_penalty;
    req->frequency_penalty = frequency_penalty;
    req->presence_penalty = presence_penalty;
    req->status = RequestStatus::PENDING;

    scheduler_->add_request(req);

    // Prefill
    while (req->status == RequestStatus::PENDING || req->status == RequestStatus::PREFILLING) {
        bool has_work = step();
        if (!has_work)
            break;
    }

    // Decode — try conditional graph loop, fall back to step()
    // Think budget is now enforced device-side in post_decode_step_kernel.
    // Penalties are applied device-side via apply_penalties_device_count in the graph loop.
    if (req->status == RequestStatus::DECODING && !req->output_tokens.empty() && config_.use_cuda_graphs &&
        !offload_mgr_) {
        int32_t first_token = req->output_tokens.back();
        auto graph_tokens = try_graph_loop_decode(req, first_token, decode_stream());
        if (!graph_tokens.empty()) {
            int32_t last = graph_tokens.back();
            // Track think state through all graph tokens
            for (int32_t t : graph_tokens)
                track_think_state(*req, t);
            bool hit_stop = should_stop(*req, last);
            if (hit_stop)
                graph_tokens.pop_back();

            for (int32_t t : graph_tokens) {
                req->output_tokens.push_back(t);
            }

            bool done = hit_stop || static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (done) {
                req->status = RequestStatus::FINISHED;
                kv_manager_->free_sequence(req->id);
            }
        }
    }

    // Fallback — per-step decode
    while (req->status != RequestStatus::FINISHED && req->status != RequestStatus::CANCELLED) {
        bool has_work = step();
        if (!has_work && req->status != RequestStatus::FINISHED && req->status != RequestStatus::CANCELLED) {
            break;
        }
    }

    if (req->output_tokens.empty()) {
        return "";
    }

    vision_.clear_image();

    std::string result = tok->decode(req->output_tokens);
    return result;
}

// prepare_graph_loop: moved to engine_speculative.cpp
// build_graph_config: moved to engine_speculative.cpp
// try_graph_loop_decode: moved to engine_speculative.cpp
// try_launch_async_graph_loop: moved to engine_speculative.cpp

void Engine::add_request(std::shared_ptr<Request> req) {
    if (scheduler_) {
        req->id = next_request_id_++;
        // Initialize in_think_block from the prompt tail. Chat templates for
        // Qwen3 / Qwen3.5 / Qwen3.6 / DeepSeek-R1 inject `<think>\n` via
        // add_generation_prompt by default — without seeding the flag here,
        // a model that promptly closes its empty thinking block will hit
        // should_stop with in_think_block=false on the trailing im_end and
        // produce a 0-content completion. We scan the decoded text of the
        // last few input tokens (covers both single-id and BPE multi-token
        // forms) and look for whichever marker appears last.
        Tokenizer* ptok = model_ ? model_->tokenizer() : nullptr;
        if (ptok && !req->input_tokens.empty()) {
            constexpr int kTailScan = 16;  // covers worst case BPE split + slack
            int n = static_cast<int>(req->input_tokens.size());
            int start = std::max(0, n - kTailScan);
            std::string tail_text;
            for (int i = start; i < n; ++i) {
                tail_text += ptok->decode_token(req->input_tokens[i]);
            }
            size_t open_pos = tail_text.rfind("<think>");
            size_t close_pos = tail_text.rfind("</think>");
            // </think> shares a suffix with <think>, so resolve precedence:
            // open is "later" only if it appears AFTER any close.
            bool open_is_last = (open_pos != std::string::npos) &&
                                (close_pos == std::string::npos || open_pos > close_pos + 1);
            if (open_is_last) {
                req->in_think_block = true;
            }
        }
        scheduler_->add_request(std::move(req));
    }
}

}  // namespace imp
