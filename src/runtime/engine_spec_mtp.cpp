// =============================================================================
// engine_spec_mtp.cpp — MTP head as a draft source for the verify loop (#847)
// =============================================================================
//
// The trained MTP head (model_mtp.safetensors, DeepSeek-V3-family — e.g.
// Qwen3.6) chain-drafts the next mtp_spec_k_ tokens; step_spec_verify_
// consumes the chain whenever the suffix/ngram matcher has no draft
// (draft-poor prose is where the trained head shines: 78-94% depth-1 accept
// on Qwen3.6-35B-A3B, PR #804).
//
// Pairing convention (DeepSeek MTP): cache pair i is (emb(t_{i+1}), h_i) —
// after P pairs the head is ready to draft the token following t_P. The
// per-step decode telemetry (engine_scheduler.cpp) always fed exactly this;
// the prefill feed here aligns with it (the pre-#847 prefill fed the
// off-by-one (t_i, h_i) pairing).
//
// The single MTP workspace tracks ONE request at a time (verify is batch-1).
// mtp_history_ records the covered tokens t_0..t_P so a follow-up request
// (multi-turn agent loop) resumes the cache over the shared prefix instead
// of restarting. Any gap the cache cannot cover — async-loop burst tokens
// (no host hidden states), a prefix-cache hit beyond the fed history —
// unbinds drafting for the request; the suffix matcher keeps working.
// =============================================================================

#include "compute/mtp_forward.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "model/model.h"
#include "runtime/engine.h"
#include "runtime/request.h"

#include <cuda_fp16.h>
#include <algorithm>

namespace imp {

void Engine::mtp_unbind_(const char* why) {
    if (mtp_bound_req_ >= 0 && !mtp_stale_logged_) {
        IMP_LOG_INFO("mtp-spec: drafting off for req %d (%s)", mtp_bound_req_, why);
        mtp_stale_logged_ = true;
    }
    mtp_bound_req_ = -1;
    mtp_pending_draft_.clear();
    mtp_draft_ctx_ = -1;
}

std::vector<int32_t> Engine::mtp_take_draft_(const Request& req) {
    if (mtp_bound_req_ != req.id || mtp_pending_draft_.empty())
        return {};
    // Drafted for a different context (an async-loop burst or eager steps
    // advanced the request without refreshing the chain) — not usable.
    if (mtp_draft_ctx_ != req.context_len())
        return {};
    return mtp_pending_draft_;
}

// Feed n_pairs (token, hidden-row) pairs into the MTP KV cache, then
// (optionally) chain-draft mtp_spec_k_ tokens for the next verify step.
// Only the final pair of a chaining feed needs logits/argmax — all other
// forwards skip the lm_head GEMV entirely (it dominates per-pair cost:
// ~1 GiB read on Qwen3.6's 248k vocab).
bool Engine::mtp_feed_pairs_(const int32_t* tokens, const void* d_hidden_rows, int n_pairs,
                             bool chain_after) {
    auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
    if (ws == nullptr || n_pairs <= 0)
        return false;
    const int hidden_dim = model_->config_.d_model;
    const int vocab_size = model_->config_.vocab_size;
    const int K = std::max(1, mtp_spec_k_);
    // MTP KV capacity (kMtpKvCap clamp at enable time): past it the cache
    // addressing is undefined — stop drafting instead.
    if (ws->max_seq_len > 0 && ws->mtp_pos + n_pairs + (chain_after ? K : 0) > ws->max_seq_len)
        return false;

    // Device-side chain: dense MTP heads have no host dependency between
    // chain steps (the MoE head's expert routing needs a per-step D2H) —
    // each step's argmax lands in ws->d_chain_tokens[i] and feeds step i+1's
    // embedding lookup on device. One D2H + sync drains the whole chain,
    // replacing K host round-trips (each of which stalls the GPU pipeline).
    const bool device_chain = chain_after && ws->n_experts == 0 &&
                              ws->d_chain_tokens != nullptr && K <= imp::kMtpMaxChainK;

    const char* base = static_cast<const char*>(d_hidden_rows);
    int pred = -1;
    for (int j = 0; j < n_pairs; ++j) {
        const void* h_j = base + static_cast<size_t>(j) * hidden_dim * sizeof(__half);
        const bool last = (j == n_pairs - 1);
        if (chain_after && last && device_chain) {
            if (!mtp_draft_one(tokens[j], h_j, hidden_dim, vocab_size, nullptr,
                               nullptr, 0, nullptr, /*d_out_token=*/ws->d_chain_tokens))
                return false;
        } else {
            int* out = (chain_after && last) ? &pred : nullptr;
            if (!mtp_draft_one(tokens[j], h_j, hidden_dim, vocab_size, out))
                return false;
        }
        mtp_history_.push_back(tokens[j]);
    }
    if (!chain_after)
        return true;

    // Chain: continue on the head's own h_final. The chained KV appends are
    // speculative — roll the cache back so only the real pairs persist.
    const int pos_after = ws->mtp_pos;
    std::vector<int32_t> chain;
    chain.reserve(K);
    if (device_chain) {
        int launched = 1;  // d_chain_tokens[0] came from the last feed pair
        for (int k = 1; k < K; ++k) {
            if (!mtp_draft_one(-1, ws->d_h_final, hidden_dim, vocab_size, nullptr,
                               nullptr, 0, /*d_prev_token=*/ws->d_chain_tokens + k - 1,
                               /*d_out_token=*/ws->d_chain_tokens + k))
                break;
            launched++;
        }
        int32_t h_chain[imp::kMtpMaxChainK];
        cudaStream_t stream = decode_stream();
        if (cudaMemcpyAsync(h_chain, ws->d_chain_tokens,
                            static_cast<size_t>(launched) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            ws->mtp_pos = pos_after;
            return false;
        }
        cudaStreamSynchronize(stream);
        for (int k = 0; k < launched; ++k) {
            if (h_chain[k] < 0 || h_chain[k] >= vocab_size)
                break;  // NaN-logits guard — keep the valid prefix
            chain.push_back(h_chain[k]);
        }
        if (chain.empty()) {
            ws->mtp_pos = pos_after;
            return false;
        }
    } else {
        if (pred < 0 || pred >= vocab_size) {
            ws->mtp_pos = pos_after;
            return false;
        }
        chain.push_back(pred);
        int prev = pred;
        for (int k = 1; k < K; ++k) {
            int p = -1;
            if (!mtp_draft_one(prev, ws->d_h_final, hidden_dim, vocab_size, &p) || p < 0 ||
                p >= vocab_size)
                break;
            chain.push_back(p);
            prev = p;
        }
    }
    ws->mtp_pos = pos_after;
    mtp_pending_draft_ = std::move(chain);
    mtp_draft_ctx_ = static_cast<int>(mtp_history_.size());
    mtp_pending_prediction_ = mtp_pending_draft_[0];  // legacy accuracy counter
    return true;
}

void Engine::mtp_post_verify_update_(const Request& req, int emitted) {
    if (mtp_bound_req_ != req.id || emitted <= 0)
        return;
    auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
    if (ws == nullptr)
        return;
    // Sync check: the cache must cover exactly the pre-step context
    // (pos == p0 == context_len - emitted - 1).
    const int p0 = req.context_len() - emitted - 1;
    if (ws->mtp_pos != p0) {
        mtp_unbind_("desync before verify feed");
        return;
    }
    // Row j of this verify chunk produced emitted token j — the (token,
    // hidden) pairs are exactly (emitted_j, h_row_j). Must run before the
    // hybrid partial-accept re-forward overwrites the hidden buffer.
    Tensor h = executor_->view_hidden(emitted);
    if (h.data == nullptr) {
        mtp_unbind_("no hidden view after verify");
        return;
    }
    const int32_t* toks = req.output_tokens.data() + req.output_tokens.size() - emitted;
    if (!mtp_feed_pairs_(toks, h.data, emitted, /*chain_after=*/req.status == RequestStatus::DECODING))
        mtp_unbind_("verify feed failed (kv cap or forward error)");
}

void Engine::mtp_prefill_feed_chunk(const Request& req, int offset, int chunk_len,
                                    int next_token) {
    if (!mtp_spec_decode_enabled() || mtp_ws_storage_ == nullptr)
        return;
    if (!model_ || !model_->mtp_.has_value() || !model_->mtp_->loaded)
        return;
    if (chunk_len <= 0)
        return;
    auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
    const auto& in = req.input_tokens;

    // (Re)bind on this request's first chunk: resume over the longest common
    // prefix of the previously fed history and this prompt (multi-turn agent
    // loops re-send the prior turn verbatim — the cache carries over).
    if (mtp_bound_req_ != req.id) {
        size_t L = 0;
        while (L < mtp_history_.size() && L < in.size() && mtp_history_[L] == in[L])
            ++L;
        const int keep_pairs = std::max(0, static_cast<int>(L) - 1);
        ws->mtp_pos = std::min(ws->mtp_pos, keep_pairs);
        if (mtp_history_.size() > static_cast<size_t>(ws->mtp_pos) + 1)
            mtp_history_.resize(static_cast<size_t>(ws->mtp_pos) + 1);
        mtp_bound_req_ = req.id;
        mtp_stale_logged_ = false;
        mtp_pending_draft_.clear();
        mtp_draft_ctx_ = -1;
        mtp_econ_verifies_ = 0;
        mtp_econ_emitted_ = 0;
        if (mtp_history_.empty() || ws->mtp_pos == 0) {
            // Nothing usable carried over — restart the cache. Pair 0 needs
            // h_0, so the prompt must actually be forwarded from position 0.
            ws->mtp_pos = 0;
            mtp_history_.clear();
            if (offset == 0 && !in.empty()) {
                mtp_history_.push_back(in[0]);
            } else {
                mtp_unbind_("prefix-cache gap without matching MTP history");
                return;
            }
        }
    }

    // Pair i consumes h_i; this chunk holds h_i for i in [offset,
    // offset+chunk_len). Resume at the cache position.
    const int start = ws->mtp_pos;
    const int end = offset + chunk_len;
    if (start < offset) {
        mtp_unbind_("hidden gap before chunk");
        return;
    }
    if (start >= end)
        return;  // chunk fully covered by the resumed cache

    std::vector<int32_t> toks;
    toks.reserve(end - start);
    for (int i = start; i < end; ++i) {
        if (i + 1 < static_cast<int>(in.size()))
            toks.push_back(in[i + 1]);
        else if (next_token >= 0)
            toks.push_back(next_token);  // final pair of the last chunk
        else
            break;  // unreachable for non-last chunks (end < input size)
    }
    if (toks.empty())
        return;

    Tensor h = executor_->view_hidden(chunk_len);
    if (h.data == nullptr) {
        mtp_unbind_("no hidden view during prefill");
        return;
    }
    const int hidden_dim = model_->config_.d_model;
    const char* rows = static_cast<const char*>(h.data) +
                       static_cast<size_t>(start - offset) * hidden_dim * sizeof(__half);
    const bool last_chunk = next_token >= 0;
    if (!mtp_feed_pairs_(toks.data(), rows, static_cast<int>(toks.size()),
                         /*chain_after=*/last_chunk))
        mtp_unbind_("prefill feed failed (kv cap or forward error)");
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
    const int hidden_dim = model_->config_.d_model;
    const int vocab_size = model_->config_.vocab_size;
    // MLP dims come from the HEAD tensors, not the main-model config: the
    // dense 27B checkpoint pairs a MoE-free MTP head with a plain SwiGLU MLP
    // (mapped onto the shared_expert fields), and the 35B head's expert d_ff
    // differs from the main model's.
    const auto& head = *model_->mtp_;
    const bool head_moe = head.router.data != nullptr && head.experts_gate_up_packed.data != nullptr;
    const int n_experts = head_moe ? static_cast<int>(head.router.shape[0]) : 0;
    const int top_k = head_moe ? model_->config_.n_experts_active : 0;
    const int expert_d_ff = head_moe ? static_cast<int>(head.experts_gate_up_packed.shape[1]) / 2 : 0;
    const int shared_d_ff = head.shared_expert_gate_proj.data != nullptr
                                ? static_cast<int>(head.shared_expert_gate_proj.shape[0])
                                : 0;

    // MTP attention dims: derived from the MTP head's q_proj / v_proj shapes
    // because the MTP attention head config differs from the main model
    // (Qwen3.6 MTP doubles Q output per-head for attn_output_gate).
    // q_proj shape [2 * num_heads * head_dim, hidden_dim]; v_proj shape
    // [num_kv_heads * head_dim, hidden_dim]. We use main model's head_dim
    // as the per-head attention dim and back-compute the MTP head counts.
    int mtp_num_heads = 0, mtp_num_kv_heads = 0, mtp_head_dim = 0;
    if (model_->mtp_.has_value() && model_->mtp_->loaded && model_->mtp_->q_proj.data != nullptr &&
        model_->mtp_->v_proj.data != nullptr) {
        const int q_out = static_cast<int>(model_->mtp_->q_proj.shape[0]);
        const int v_out = static_cast<int>(model_->mtp_->v_proj.shape[0]);
        mtp_head_dim = model_->config_.head_dim;
        if (mtp_head_dim > 0) {
            // q_proj outputs 2 × num_heads × head_dim (attn_output_gate=True).
            mtp_num_heads = q_out / (2 * mtp_head_dim);
            mtp_num_kv_heads = v_out / mtp_head_dim;
        }
    }

    // MTP KV-cache capacity: cap at the smaller of model's max_seq_len and 16K
    // (Phase 2.2.Attn+KV budget — ~16 MiB each for K and V at Qwen3.6 dims).
    constexpr int kMtpKvCap = 16384;
    int mtp_kv_max = std::min(model_->config_.max_seq_len, kMtpKvCap);
    if (mtp_kv_max <= 0)
        mtp_kv_max = kMtpKvCap;

    auto* ws = new imp::MtpDraftWorkspace();
    if (!imp::mtp_workspace_allocate(*ws, hidden_dim, vocab_size, n_experts, top_k, expert_d_ff, shared_d_ff,
                                     mtp_num_heads, mtp_num_kv_heads, mtp_head_dim, mtp_kv_max)) {
        delete ws;
        IMP_LOG_ERROR("enable_mtp_spec_decode: workspace alloc failed");
        return false;
    }
    // Configure RoPE for the MTP attention (Phase 2.2.Attn+RoPE).
    // Qwen3.5/3.6 uses partial rope (factor 0.25 → rope_dim=64 of head_dim=256),
    // theta from config (10M for long-context), NeoX-style.
    ws->rope_theta = model_->config_.rope_theta;
    ws->rope_neox = model_->config_.rope_neox;
    ws->rms_norm_eps = model_->config_.rms_norm_eps;
    ws->rope_dim = (model_->config_.rope_dim > 0) ? model_->config_.rope_dim : mtp_head_dim;
    // mrope section split. Read from config when the checkpoint carries one
    // (Qwen3-VL: [24, 20, 20]); the Qwen3.6 constant below stays as the
    // fallback for checkpoints that ship the split only in their spec.
    // For text-only generation all 3 positions are equal, so this is
    // mathematically equivalent to standard partial-rope; the section
    // split matters only for true multimodal tokens.
    const ModelConfig& mc = model_->config_;
    if (mc.has_mrope() &&
        mc.mrope_section[0] + mc.mrope_section[1] + mc.mrope_section[2] == ws->rope_dim / 2) {
        ws->mrope_sec0 = mc.mrope_section[0];
        ws->mrope_sec1 = mc.mrope_section[1];
        ws->mrope_sec2 = mc.mrope_section[2];
    } else if (ws->rope_dim == 64) {
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
    ws->rope_freq_scale = model_->config_.rope_freq_scale;
    ws->yarn_ext_factor = model_->config_.yarn_ext_factor;
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
                     ws->yarn_ext_factor, ws->yarn_attn_factor, ws->rope_freq_scale, ws->yarn_corr_dim_0,
                     ws->yarn_corr_dim_1);
    }
    // LongRoPE (Phi-family) isn't plumbed into the single-token MTP kernel — no
    // MTP model ships it today (Qwen uses YaRN/linear). Warn rather than silently
    // diverge if that ever changes.
    if (!model_->config_.rope_short_factor.empty() || !model_->config_.rope_long_factor.empty()) {
        IMP_LOG_WARN(
            "MTP spec-decode: model uses LongRoPE scaling, which the draft head does not apply "
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
    IMP_LOG_INFO(
        "MTP spec-decode enabled (k=%d, hidden=%d, vocab=%d, experts=%d/top%d, d_ff_e=%d, "
        "d_ff_shared=%d, num_heads=%d/%d, head_dim=%d, kv_cap=%d, rope=%g/%d/%s, "
        "mrope=[%d,%d,%d])",
        k, hidden_dim, vocab_size, n_experts, top_k, expert_d_ff, shared_d_ff, mtp_num_heads,
        mtp_num_kv_heads, mtp_head_dim, mtp_kv_max, ws->rope_theta, ws->rope_dim,
        ws->rope_neox ? "neox" : "interleaved", ws->mrope_sec0, ws->mrope_sec1, ws->mrope_sec2);
    return true;
}
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
bool Engine::mtp_draft_one(int prev_token_id, const void* d_h_prev, int hidden_dim, int vocab_size,
                           int* out_token_id, int* out_topk_ids, int top_w, const int32_t* d_prev_token,
                           int32_t* d_out_token) {
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
    if (runtime_config_.speculative.mtp_nvfp4_head && executor_ && executor_->lm_head_nvfp4_view(lm_nvfp4)) {
        lm_nvfp4_p = &lm_nvfp4;
        static bool logged = false;  // once-per-process path attribution
        if (!logged) {
            logged = true;
            IMP_LOG_INFO("MTP chain lm_head: NVFP4 decode-cache view engaged (N=%lld K=%lld)",
                         static_cast<long long>(lm_nvfp4.N), static_cast<long long>(lm_nvfp4.K));
        }
    }
    return imp::mtp_draft_step(prev_token_id, d_h_prev, *model_->mtp_, model_->tok_emb_, model_->out_proj_,
                               *ws, hidden_dim, vocab_size, out_token_id, decode_stream(), out_topk_ids,
                               top_w, lm_nvfp4_p, d_prev_token, d_out_token);
}

}  // namespace imp
