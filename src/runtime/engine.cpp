#include "runtime/engine.h"
#include "runtime/engine_internal.h"
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

// File-local helpers were lifted to runtime/engine_internal.h so that
// the per-subsystem engine_*.cpp translation units (Phase 4 Tasks 2-7)
// can share them. The using-declarations below preserve the unqualified
// call sites in this file.
using engine_internal::build_logprob_info;
using engine_internal::compute_step_seed;
using engine_internal::ensure_prefill_workspace;
using engine_internal::free_prefill_buffers;

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
//
// The init_apply_debug_raw_overrides_ / init_resolve_kv_dtype_policy_ /
// init_resolve_ssm_dtype_ / init_resolve_fp8_prefill_ /
// init_resolve_quant_flags_ / init_compute_max_seq_len_ methods (~320 LOC
// across 6 methods) live in runtime/engine_init_resolver.cpp.

bool Engine::init(std::shared_ptr<Model> model, const EngineConfig& config) {
    if (!model)
        return false;

    model_ = std::move(model);
    config_ = config;

    const auto& mcfg = model_->config();

    init_apply_debug_raw_overrides_();
    init_resolve_kv_dtype_policy_();
    init_resolve_ssm_dtype_();
    init_resolve_fp8_prefill_();
    init_resolve_quant_flags_();

    init_compute_max_seq_len_();

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

// init_features, build_banned_token_list, warmup → engine_workspace_warmup.cpp
// step / step_async_graph_resume / step_schedule / supports_chunked_prefill_ /
// resolve_prefill_chunk_size_ / step_prefill / prefill_allocate_kv_blocks_ /
// prefill_upload_metadata_ / step_prefill_one / step_decode /
// decode_build_inference_state_ / step_decode_forward /
// step_decode_process_outputs → engine_scheduler.cpp

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
