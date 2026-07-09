#include "runtime/engine.h"
#include "core/buffer.h"
#include "vision/image_processor.h"
#include "runtime/request.h"
#include "lora/lora_adapter.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "runtime/process_diag.h"
#include "runtime/vram_budget.h"
#include "runtime/batch.h"
#include "compute/mtp_forward.h"
#include "compute/rope.h"           // rope_yarn_corr_dims (MTP rope-scaling parity)
#include "compute/encoder_forward.h"
#include "memory/kv_cache.h"
#include "memory/mem_account.h"
#include "memory/vram_query.h"
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
// can share them.

Engine::~Engine() {
    // Cross-model CUDA-error-leak guard. The CUDA error state is per primary
    // context, NOT per Engine, so a pending ("sticky") error left by this
    // model's workload survives imp_context_free / imp_model_free and is then
    // observed by the NEXT model loaded in the same process. That next model's
    // first cudaGetLastError()-guarded kernel — notably the NVFP4 CUTLASS GEMM
    // (gemm_cutlass_sm120.cu: "Flush any prior async errors") — bails to a
    // silent false return, producing degenerate garbage instead of running.
    // Observed repro: a GDN/SSM model (Qwen3.5) loaded before a Gemma-4 NVFP4
    // model garbled Gemma-4 ("own own else else"), while a dense or MoE
    // predecessor did not. Drain it here so it cannot cross the model boundary.
    if (cudaError_t leaked = cudaGetLastError(); leaked != cudaSuccess) {
        IMP_LOG_WARN("Engine teardown: cleared a leaked CUDA error (%s) so it cannot "
                     "corrupt the next model loaded in this process",
                     cudaGetErrorString(leaked));
    }

    // Phase-0 VRAM audit: stop the peak sampler and emit the final table
    // (captures the device-used peak reached during the workload).
    MemAccount::instance().sampler_stop();
    MemAccount::instance().report("shutdown");

    // Defensive: the mandatory-cache balloon is normally released before
    // pre_dequant_weights; don't leak it if init aborted in between.
    release_native_cache_balloon_("teardown");

    // Save prefix cache to disk before shutdown (dense only — hybrid reuse
    // needs recurrent snapshots, which are device-resident and not persisted)
    if (kv_manager_ && !config_.prefix_cache_path.empty() && kv_manager_->prefix_caching_enabled() &&
        !ssm_state_) {
        kv_manager_->save_prefix_cache(config_.prefix_cache_path, model_fingerprint_(), stream_);
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
    if (async_d_block_tables_swa_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_swa_));
        async_d_block_tables_swa_ = nullptr;
    }
    if (async_d_banned_tokens_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
        async_d_banned_tokens_ = nullptr;
    }
    if (d_penalty_tokens_) {
        vram_alloc_.free(d_penalty_tokens_);
        d_penalty_tokens_ = nullptr;
    }
    if (d_token_is_whitespace_) {
        vram_alloc_.free(d_token_is_whitespace_);
        d_token_is_whitespace_ = nullptr;
    }
    if (d_kv_slot_buf_) {
        cudaFree(d_kv_slot_buf_);
        d_kv_slot_buf_ = nullptr;
    }
    if (h_sample_pinned_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_sample_pinned_));
        h_sample_pinned_ = nullptr;
    }
    log_spec_stats_();
    free_spec_buffers_();
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
    if (pf_staging_evt_) {
        IMP_CUDA_CHECK_LOG(cudaEventDestroy(pf_staging_evt_));
        pf_staging_evt_ = nullptr;
    }
    // Encoder embedder workspace cleanup (#836)
    if (encoder_ws_storage_) {
        auto* ews = static_cast<imp::EncoderWorkspace*>(encoder_ws_storage_);
        imp::encoder_workspace_free(*ews);
        delete ews;
        encoder_ws_storage_ = nullptr;
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
    // MLP dims come from the HEAD tensors, not the main-model config: the
    // dense 27B checkpoint pairs a MoE-free MTP head with a plain SwiGLU MLP
    // (mapped onto the shared_expert fields), and the 35B head's expert d_ff
    // differs from the main model's.
    const auto& head       = *model_->mtp_;
    const bool head_moe    = head.router.data != nullptr &&
                             head.experts_gate_up_packed.data != nullptr;
    const int n_experts    = head_moe ? static_cast<int>(head.router.shape[0]) : 0;
    const int top_k        = head_moe ? model_->config_.n_experts_active : 0;
    const int expert_d_ff  = head_moe
                                 ? static_cast<int>(head.experts_gate_up_packed.shape[1]) / 2
                                 : 0;
    const int shared_d_ff  = head.shared_expert_gate_proj.data != nullptr
                                 ? static_cast<int>(head.shared_expert_gate_proj.shape[0])
                                 : 0;

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
    // RoPE scaling — mirror the main forward so the drafter rotates Q/K the
    // same way as the verifier at extended positions (issue #897). Without
    // this a rope-scaled model's draft head diverges and acceptance silently
    // degrades with position.
    ws->rope_freq_scale  = model_->config_.rope_freq_scale;
    ws->yarn_ext_factor  = model_->config_.yarn_ext_factor;
    ws->yarn_attn_factor = model_->config_.yarn_attn_factor;
    if (model_->config_.yarn_ext_factor > 0.0f) {
        int hd = model_->config_.head_dim > 0 ? model_->config_.head_dim
                                              : (model_->config_.d_model / model_->config_.n_heads);
        int n_dims = (model_->config_.rope_dim > 0) ? model_->config_.rope_dim : hd;
        int n_ctx_orig = model_->config_.rope_n_ctx_orig > 0 ? model_->config_.rope_n_ctx_orig
                                                             : model_->config_.max_seq_len;
        float corr[2] = {0.0f, 0.0f};
        imp::rope_yarn_corr_dims(n_dims, n_ctx_orig, model_->config_.rope_theta,
                                 model_->config_.yarn_beta_fast, model_->config_.yarn_beta_slow, corr);
        ws->yarn_corr_dim_0 = corr[0];
        ws->yarn_corr_dim_1 = corr[1];
        IMP_LOG_INFO("MTP YaRN: ext=%.2f attn=%.3f freq_scale=%.4f corr_dims=[%.1f, %.1f]",
                     ws->yarn_ext_factor, ws->yarn_attn_factor, ws->rope_freq_scale,
                     ws->yarn_corr_dim_0, ws->yarn_corr_dim_1);
    }
    // LongRoPE (Phi-family) isn't plumbed into the single-token MTP kernel — no
    // MTP model ships it today (Qwen uses YaRN/linear). Warn rather than silently
    // diverge if that ever changes.
    if (!model_->config_.rope_short_factor.empty() || !model_->config_.rope_long_factor.empty()) {
        IMP_LOG_WARN("MTP spec-decode: model uses LongRoPE scaling, which the draft head does not apply "
                     "— draft rope will diverge from the verifier; expect degraded acceptance");
    }

    // Diagnostic: generation.mtp_no_rope disables RoPE entirely.
    if (runtime_config_.generation.mtp_no_rope) {
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

// (mtp_prefill_prompt was replaced by the per-chunk mtp_prefill_feed_chunk
// in engine_spec_mtp.cpp — chunked-prefill capable, DeepSeek-aligned pairing,
// feed-only forwards without the lm_head GEMV.)

void Engine::mtp_accuracy_reset() noexcept {
    mtp_accuracy_ = {};
    mtp_pending_prediction_ = -1;
    mtp_pending_chain_.clear();
    mtp_chain_accept_.clear();
    mtp_chain_accept_w_.clear();
    mtp_bound_req_ = -1;
    mtp_history_.clear();
    mtp_pending_draft_.clear();
    mtp_draft_ctx_ = -1;
    mtp_econ_verifies_ = 0;
    mtp_econ_emitted_ = 0;
    if (mtp_ws_storage_) {
        auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
        imp::mtp_kv_reset(*ws);
    }
}

bool Engine::encoder_embed(const int32_t* tokens, int n, std::vector<float>& out) {
    if (encoder_ws_storage_ == nullptr || !model_) {
        IMP_LOG_ERROR("encoder_embed: no encoder workspace (not an encoder model?)");
        return false;
    }
    auto* ews = static_cast<imp::EncoderWorkspace*>(encoder_ws_storage_);
    out.resize(model_->config_.d_model);
    return imp::encoder_embed(*model_, *ews, tokens, n, out.data(), stream_);
}

bool Engine::mtp_draft_one(int prev_token_id, const void* d_h_prev,
                           int hidden_dim, int vocab_size, int* out_token_id,
                           int* out_topk_ids, int top_w,
                           const int32_t* d_prev_token, int32_t* d_out_token) {
    if (mtp_ws_storage_ == nullptr) {
        IMP_LOG_ERROR("mtp_draft_one: spec-decode not enabled");
        return false;
    }
    if (!model_ || !model_->mtp_.has_value() || !model_->mtp_->loaded) {
        IMP_LOG_ERROR("mtp_draft_one: MTP head not loaded");
        return false;
    }
    auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
    // Chain lm_head via the NVFP4 decode cache when available: the full-vocab
    // FP16 GEMV is the dominant per-draft cost (#847 lever 3). Draft-only
    // precision — verify stays lossless. Falls back to the FP16 GEMV when no
    // cache entry exists (nvfp4_lm_head/_gdn off, or FP8 LM head).
    imp::NvFP4QuantResult lm_nvfp4;
    const imp::NvFP4QuantResult* lm_nvfp4_p = nullptr;
    if (runtime_config_.speculative.mtp_nvfp4_head && executor_ &&
        executor_->lm_head_nvfp4_view(lm_nvfp4)) {
        lm_nvfp4_p = &lm_nvfp4;
        static bool logged = false;  // once-per-process path attribution
        if (!logged) {
            logged = true;
            IMP_LOG_INFO("MTP chain lm_head: NVFP4 decode-cache view engaged (N=%lld K=%lld)",
                         static_cast<long long>(lm_nvfp4.N), static_cast<long long>(lm_nvfp4.K));
        }
    }
    return imp::mtp_draft_step(prev_token_id, d_h_prev, *model_->mtp_,
                                model_->tok_emb_, model_->out_proj_,
                                *ws, hidden_dim, vocab_size, out_token_id,
                                decode_stream(), out_topk_ids, top_w, lm_nvfp4_p,
                                d_prev_token, d_out_token);
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
    // Public teardown entry point (API re-prefill / context reset / server
    // cancellation). Reset the sequence's ACTUAL allocated recurrent slot, then
    // return it to the free list — otherwise the slot leaks (these paths bypass
    // finish_request) and the pool exhausts, forcing every later request onto
    // the legacy id%cap aliasing fallback.
    auto it = recurrent_slot_of_.find(seq_id);
    if (ssm_state_) {
        const int cap = ssm_state_->max_sequences();
        int slot = (it != recurrent_slot_of_.end()) ? it->second : (cap > 0 ? seq_id % cap : 0);
        ssm_state_->reset_sequence(slot, stream_);
    }
    release_recurrent_slot_(seq_id);
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

    // The pipelined constrained decode holds a captured forward graph plus
    // request-specific device state — same invalidation requirement as the
    // conditional runner.
    if (cpipe_.active)
        teardown_constrained_pipeline(/*synchronize=*/true);

    // Captured verify-chunk graphs (#847) baked the forward as well (incl.
    // any active LoRA kernels/pointers) — recapture costs two verify steps.
    free_spec_graphs_();

    // #874 safety net: if an exception unwound past an active prefill-chunk
    // capture, the prefill stream is still in capture state and every later
    // op on it fails ("previous error during capture") — permanently wedging
    // the server. Close any stray capture and drop the prefill runner so the
    // next request starts from a clean stream.
    prefill_graph_runner_.invalidate();
    last_prefill_chunk_len_ = -1;
    last_prefill_block_count_ = -1;
    abort_stream_capture(prefill_stream());
}

int Engine::lora_load(const std::string& path) {
    auto a = std::make_unique<LoraAdapter>();
    if (!a->load(path, model_ ? model_->n_layers() : 0))
        return 0;
    lora_adapters_.push_back(std::move(a));
    return static_cast<int>(lora_adapters_.size());  // 1-based id
}

bool Engine::lora_set(int id) {
    if (id < 0 || id > static_cast<int>(lora_adapters_.size()))
        return false;
    if (id == active_lora_)
        return true;
    active_lora_ = id;
    executor_->set_lora(id == 0 ? nullptr : lora_adapters_[id - 1].get());
    // Decode graphs captured the previous forward (with/without the LoRA
    // kernels and with the old adapter's pointers) — drop everything,
    // including the per-batch pool that invalidate_graphs() preserves.
    invalidate_graphs();
    for (auto& g : decode_graph_pool_)
        g.invalidate();
    IMP_LOG_INFO("LoRA: active adapter -> %d%s", id, id == 0 ? " (base)" : "");
    return true;
}

size_t Engine::effective_free_vram() const {
    // Budget-aware view (installed in init from config_.vram_budget_mb).
    // The old inline formula counted GLOBAL device usage against the budget,
    // which mis-charged a co-tenant server's memory to this process;
    // vram_query uses the process baseline delta instead.
    size_t free_mem = 0;
    if (!vram_budget_mem_get_info(&free_mem, nullptr))
        return 0;
    return free_mem;
}

void Engine::finish_request(std::shared_ptr<Request>& req) {
    req->status = RequestStatus::FINISHED;
    if (kv_manager_->prefix_caching_enabled()) {
        // Register input AND generated tokens — minus the final sampled
        // token, which was never forwarded (its KV entry does not exist; the
        // spec-verify bonus token has the same property). The next agent turn
        // re-sends the assistant reply verbatim (tool-call JSON, code edits),
        // and its KV is live in the block table right now: hashing it turns
        // the whole previous turn into a prefix-cache hit instead of
        // re-prefilling the reply from scratch.
        if (req->output_tokens.size() > 1) {
            std::vector<int32_t> forwarded;
            forwarded.reserve(req->input_tokens.size() + req->output_tokens.size() - 1);
            forwarded.insert(forwarded.end(), req->input_tokens.begin(), req->input_tokens.end());
            forwarded.insert(forwarded.end(), req->output_tokens.begin(),
                             req->output_tokens.end() - 1);
            kv_manager_->register_block_hashes(req->id, forwarded);
        } else {
            kv_manager_->register_block_hashes(req->id, req->input_tokens);
        }
        // cache_control / cache_prompt: protect the prompt's full blocks
        // from eviction (must happen before free_sequence — pinning needs
        // the live block table).
        if (req->pin_kv_prefix) {
            int full_blocks =
                static_cast<int>(req->input_tokens.size()) / kv_manager_->kv_cache()->block_size();
            if (full_blocks > 0)
                kv_manager_->pin_prefix(req->id, full_blocks);
        }
    }
    kv_manager_->free_sequence(req->id);
    release_recurrent_slot_(req->id);
    req->recurrent_restore.reset();  // release the snapshot buffer for recycling
    spec_suffix_idx_.erase(req->id);
    if (req->constraints)
        constraints_return_(std::move(req->constraints));
    // Server visibility: the engine outlives requests, so cumulative
    // speculation telemetry is logged per request end (no-op when idle).
    if (spec_ngram_enabled_(*req))
        log_spec_stats_();
}

std::shared_ptr<ConstraintManager> Engine::constraints_checkout_(const std::string& json_schema) {
    // Prefer a pooled manager that already classified this schema.
    if (!json_schema.empty()) {
        for (auto it = constraint_pool_.begin(); it != constraint_pool_.end(); ++it) {
            if ((*it)->cached_schema() == json_schema) {
                auto cm = std::move(*it);
                constraint_pool_.erase(it);
                return cm;
            }
        }
    }
    if (!constraint_pool_.empty()) {
        auto cm = std::move(constraint_pool_.back());
        constraint_pool_.pop_back();
        return cm;
    }
    return std::make_shared<ConstraintManager>();
}

void Engine::constraints_return_(std::shared_ptr<ConstraintManager> cm) {
    if (!cm)
        return;
    cm->reset();
    constexpr size_t kMaxConstraintPool = 8;
    if (constraint_pool_.size() < kMaxConstraintPool)
        constraint_pool_.push_back(std::move(cm));
}

// =====================================================================
// Vision delegation
// =====================================================================

bool Engine::set_image(const std::string& path) { return vision_.set_image(path, stream_); }

bool Engine::set_image_from_memory(const uint8_t* data, size_t len) {
    return vision_.set_image_from_memory(data, len, stream_);
}

void Engine::clear_image() { vision_.clear_image(); }

bool Engine::preprocess_image(const uint8_t* data, size_t len, ImageData& out) {
    return vision_.preprocess(data, len, out);
}

bool Engine::encode_image_for(Request& req) {
    if (!req.image || !vision_.is_available())
        return false;
    auto buf = std::make_shared<Buffer>(Buffer::device(vision_.embeddings_bytes()));
    if (!*buf)
        return false;
    if (!vision_.encode_to(*req.image, buf->as<half>(), stream_))
        return false;
    req.vision_emb = std::move(buf);
    req.vision_token_id = vision_.soft_token_id();
    req.n_vision_tokens = vision_.num_image_tokens();
    req.image.reset();  // host pixels no longer needed after encode
    return true;
}

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

    // A previous engine on this same model handle freed the model's source
    // weight tensors to reclaim VRAM (Phase-4b drop); their .data pointers are
    // dangling now, so rebuilding this engine's weight caches would read freed
    // memory and poison the CUDA context with an illegal access (#830). Reject
    // up front with a clear error. Reload the model for a second engine. (Dense
    // models that never drop sources are unaffected — create/free/create works.)
    if (model->sources_consumed()) {
        IMP_LOG_ERROR(
            "Engine::init: this model handle was already bound to an engine whose "
            "weight caches consumed (freed) the model's source tensors — a second "
            "engine cannot be built on it. Reload the model (imp_model_load) to "
            "create another engine.");
        return false;
    }

    model_ = std::move(model);
    config_ = config;

    // Phase 5 Track D (follow-up): take the pending RuntimeConfig stashed
    // by tool main (imp-cli / imp-server) via set_pending_runtime_config().
    // If no pending config was set (library/test embeddings), this returns
    // a freshly loaded env-seeded default. Either way, every Engine::*
    // method reads runtime_config_ directly from here on; engine_init_
    // resolver_ helpers mutate this snapshot in place for arch-specific
    // defaults.
    runtime_config_ = take_pending_runtime_config();

    // Bridge the documented imp.conf [server]/[paths] keys into the live
    // EngineConfig. These keys are user-facing (imp.conf.example) but were
    // parsed into RuntimeConfig and never read — the live path flowed only
    // through the C-API/CLI, so setting them in imp.conf was silently inert
    // (the wiring PR #541 intended for [server] prefix_cache regressed in a
    // later refactor). imp.conf is the user's persistent preference; a CLI
    // flag / C-API value can additionally ENABLE a knob (OR), so a library
    // embedder's explicit choice is never clobbered. --mmproj (explicit
    // one-shot) overrides imp.conf. RuntimeConfig defaults for these match the
    // EngineConfig defaults (off), so no-imp.conf embedders are unaffected.
    config_.use_prefix_caching = config_.use_prefix_caching || runtime_config_.server.prefix_cache;
    config_.use_green_contexts = config_.use_green_contexts || runtime_config_.server.green_contexts;
    if (config_.prefix_pin_budget_pct == 25)  // EngineConfig default untouched → take imp.conf
        config_.prefix_pin_budget_pct = runtime_config_.server.prefix_pin_budget_pct;
    if (config_.mmproj_path.empty())
        config_.mmproj_path = runtime_config_.paths.mmproj;
    if (config_.vram_budget_mb == 0 && runtime_config_.runtime.vram_budget_mb > 0)
        config_.vram_budget_mb = static_cast<size_t>(runtime_config_.runtime.vram_budget_mb);
    if (config_.kv_fraction == 0.8f)  // EngineConfig default untouched → take imp.conf
        config_.kv_fraction = runtime_config_.vram.kv_fraction;
    if (config_.vram_reserve_floor_pct == 10)
        config_.vram_reserve_floor_pct = runtime_config_.vram.reserve_floor_pct;

    // Install the process-wide VRAM budget view BEFORE any sizing runs —
    // every cudaMemGetInfo-based decision below (weight upload gates, cache
    // budgets, KV clamp, workspaces) reads through vram_budget_mem_get_info.
    vram_budget_install(config_.vram_budget_mb);

    // The deterministic kernel gate lives in process_diag (compute kernels
    // read process_diag_deterministic_gemm()), but process_diag_install()
    // only runs in tool mains. Promote the gate here so library/test
    // embeddings (C API without a tool main) honor [runtime] deterministic /
    // IMP_DETERMINISTIC too. True-promotion only — arch resolvers and tool
    // installs may already have set it.
    if (runtime_config_.runtime.deterministic || runtime_config_.runtime.deterministic_gemm)
        process_diag_set_deterministic_gemm(true);

    // D1: derive the architecture profile ONCE, before the resolvers below that
    // currently re-derive GDN/SSM/MoE classification inline. The layers are
    // loaded by now (the resolvers read layer().gdn_gate.data directly).
    model_->build_profile();
    {
        const auto& mp = model_->profile();
        const char* av = nullptr;
        switch (mp.attn_variant) {
            case ModelProfile::AttnVariant::GEMMA4_SWA: av = "gemma4_swa"; break;
            case ModelProfile::AttnVariant::GPTOSS_SWA: av = "gptoss_swa"; break;
            case ModelProfile::AttnVariant::NOPE:       av = "nope";       break;
            case ModelProfile::AttnVariant::MLA:        av = "mla";        break;
            case ModelProfile::AttnVariant::STANDARD:   av = "standard";   break;
        }
        IMP_LOG_INFO("ModelProfile: moe=%d gdn=%d ssm=%d hybrid=%d dense=%d attn=%s",
                     mp.is_moe, mp.is_gdn, mp.is_ssm, mp.is_hybrid, mp.is_dense, av);
    }

    init_apply_debug_raw_overrides_();
    init_apply_rope_override_();
    init_resolve_kv_dtype_policy_();
    init_resolve_ssm_dtype_();
    init_resolve_fp8_prefill_();
    init_resolve_quant_flags_();

    init_compute_max_seq_len_();

    // --- Core initialization ---
    // Phase-0 VRAM audit harness: lifecycle checkpoints bracket each init
    // sub-phase so the device free-VRAM delta measures that phase's cost with
    // full coverage (raw cudaMalloc included). Gated, default off.
    if (runtime_config_.diagnostics.vram_audit) {
        MemAccount::instance().set_enabled(true);
        if (!runtime_config_.diagnostics.vram_audit_dump.empty())
            MemAccount::instance().set_dump_path(runtime_config_.diagnostics.vram_audit_dump);
    }
    MemAccount::instance().checkpoint("00_pre_init");

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
    MemAccount::instance().checkpoint("01_prewarm_gemm");

    // --- Encoder-only embedder (#836): no executor, KV cache, decoder
    // features, or warmup. Upload weights, dequant them into the dedicated
    // encoder workspace, done — /v1/embeddings drives encoder_embed().
    if (model_->profile().is_encoder) {
        if (!model_->upload_weights_gpu(config_.compute_dtype, stream_, 1ULL << 30)) {
            IMP_LOG_ERROR("encoder: weight upload failed");
            return false;
        }
        auto* ews = new imp::EncoderWorkspace();
        int cap = model_->config_.max_seq_len > 0 ? model_->config_.max_seq_len : 2048;
        if (config_.max_seq_len > 0)
            cap = std::min(cap, config_.max_seq_len);
        if (!imp::encoder_workspace_init(*ews, *model_, cap, stream_)) {
            delete ews;
            return false;
        }
        encoder_ws_storage_ = ews;
        IMP_LOG_INFO("Encoder embedder ready (arch=%s, max_tokens=%d, d=%d)",
                     model_arch_name(model_->config_.arch), cap, model_->config_.d_model);
        return true;
    }

    // --- Sub-phases ---
    if (!init_weights()) {
        release_native_cache_balloon_("init_weights failed");
        return false;
    }
    MemAccount::instance().checkpoint("02_weights+decode_cache");
    if (!init_kv_cache()) {
        release_native_cache_balloon_("init_kv_cache failed");
        return false;
    }
    MemAccount::instance().checkpoint("03_kv_cache");
    if (!init_features())
        return false;
    MemAccount::instance().checkpoint("04_features");
    if (!runtime_config_.runtime.warmup) {
        IMP_LOG_INFO("Warmup SKIPPED (runtime.warmup=false)");
    } else {
        warmup();
    }
    MemAccount::instance().checkpoint("05_post_warmup");
    // Start the device-used peak sampler so the prefill activation / score
    // matrix spike during the workload is captured, then dump the init table.
    MemAccount::instance().sampler_start(2000);
    MemAccount::instance().report("init_complete");

    return true;
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
        // gpt-oss Harmony generation starts in the analysis (reasoning) channel
        // — the model emits <|channel|>analysis<|message|> as its first output,
        // there is no <think> opener for the scan above to find. Seed the think
        // state so the answer-headroom budget counts reasoning from the start
        // and force-closes the analysis channel (<|end|>) before max_tokens is
        // exhausted, instead of returning an empty final channel.
        if (harmony_reasoning_) {
            req->started_in_think = true;
            req->in_think_block = true;
        }
        scheduler_->add_request(std::move(req));
    }
}

}  // namespace imp
