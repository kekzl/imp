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

}  // namespace imp
