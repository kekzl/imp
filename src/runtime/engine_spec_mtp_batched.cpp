// Batched feeds of the MTP head: the batched verify's ragged pass over every bound request's emitted pairs
// (mtp_batched_feed_), and speculative.mtp_carry, which keeps a bound request's head cache in step through
// batched decode by carrying its hidden rows (mtp_carry_rows_) until their next token is known
// (mtp_carry_flush_). Pairing convention and the binding invariant: engine_spec_mtp.cpp.

#include "compute/mtp_forward.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "model/model.h"
#include "quant/nvfp4_quant.h"
#include "runtime/engine.h"
#include "runtime/mtp_head_graph.h"
#include "runtime/request.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>

namespace imp {

static_assert(kMtpCarryRows <= kMtpFeedRows, "a full carry feeds the head in one mtp_feed_batch pass");

void Engine::mtp_batched_feed_(const std::vector<std::shared_ptr<Request>>& reqs,
                               const std::vector<int>& rows, const std::vector<int>& emitted,
                               cudaStream_t stream) {
    auto* ws = mtp_ws_storage_.get();
    if (ws == nullptr || !model_ || !model_->mtp_ || !model_->mtp_->loaded)
        return;
    const int N = static_cast<int>(reqs.size());
    if (ws->feed_rows_cap <= 0 || ws->d_b_h_final == nullptr) {
        for (int g = 0; g < N; ++g)
            if (mtp_bound(mtp_active_, mtp_pool_, reqs[g]->id))
                mtp_post_verify_update_(*reqs[g], emitted[g], rows[g]);
        return;
    }
    int n_rows_total = 0;
    for (int g = 0; g < N; ++g)
        n_rows_total = std::max(n_rows_total, rows[g] + emitted[g]);
    Tensor h = executor_->view_hidden(std::max(1, n_rows_total));
    if (h.data == nullptr)
        return;
    std::vector<int32_t> toks;
    std::vector<int> src, slots, pos, last_row, bound_g;
    // Second draft (bv_.depth 2): one chained head row per bound request at its next position.
    std::vector<int> chain_slot, chain_pos;
    for (int g = 0; g < N; ++g) {
        Request& req = *reqs[g];
        // Plain batched steps before this verify (speculative.mtp_batch_verify); a failed feed unbinds.
        mtp_carry_flush_(req);
        if (emitted[g] <= 0 || req.status != RequestStatus::DECODING ||
            !mtp_bound(mtp_active_, mtp_pool_, req.id))
            continue;
        mtp_activate_(req.id);
        const int p0 = req.context_len() - emitted[g] - 1;
        if (ws->mtp_pos != p0) {
            mtp_unbind_("desync before batched feed");
            continue;
        }
        if ((ws->max_seq_len > 0 && ws->mtp_pos + emitted[g] > ws->max_seq_len) ||
            static_cast<int>(toks.size()) + emitted[g] > ws->feed_rows_cap) {
            mtp_unbind_("kv cap reached");
            continue;
        }
        const int32_t* t = req.output_tokens.data() + req.output_tokens.size() -
                           static_cast<size_t>(emitted[g]);
        for (int j = 0; j < emitted[g]; ++j) {
            toks.push_back(t[j]);
            src.push_back(rows[g] + j);
            slots.push_back(ws->cur_slot);
            pos.push_back(ws->mtp_pos + j);
        }
        last_row.push_back(static_cast<int>(toks.size()) - 1);
        bound_g.push_back(g);
        chain_slot.push_back(ws->cur_slot);
        chain_pos.push_back(ws->mtp_pos + emitted[g]);
    }
    if (bound_g.empty())
        return;
    const int n = static_cast<int>(toks.size());
    const int m = static_cast<int>(bound_g.size());
    auto fail_all = [&](const char* why) {
        for (int g : bound_g) {
            mtp_activate_(reqs[g]->id);
            mtp_unbind_(why);
        }
    };
    // Timing (host wall through the draft D2H sync below), logged every 100
    // calls: the one number that says what the head costs per verify step.
    static double t_head_ms = 0.0;
    static long long t_calls = 0, t_rows = 0;
    const auto t0 = std::chrono::steady_clock::now();
    // The next draft per request lands in ws->d_chain_tokens: m rows, m more with the chain.
    if (2 * m > imp::kMtpMaxTopW * imp::kMtpMaxChainK || ws->d_chain_tokens == nullptr) {
        fail_all("batched draft: too many rows");
        return;
    }
    // Every table of the pass (feed rows, draft gather rows, chain rows) in one pinned H2D.
    imp::MtpPassTables tab;
    if (!imp::mtp_head_pass_upload(*ws, toks.data(), src.data(), slots.data(), pos.data(), n, last_row.data(),
                                   chain_slot.data(), chain_pos.data(), m, stream, tab)) {
        fail_all("batched feed failed");
        return;
    }
    // The lone chain's draft head: 682 MiB per pass instead of the 1213 MiB FP8 head.
    imp::NvFP4QuantResult head_view;
    const imp::NvFP4QuantResult* head = mtp_draft_lm_head_nvfp4_(head_view);
    // The chained KV row is speculative: the next feed starts at that position and overwrites it.
    const bool chain = bv_.depth > 1 && (ws->max_seq_len <= 0 || tab.chain_max_pos + 1 <= ws->max_seq_len);
    const int n_out = mtp_head_pass_(tab, h.data, head, chain, stream);
    if (n_out < 0) {
        fail_all(n_out == -1 ? "batched feed failed" : "batched draft: LM head unavailable");
        return;
    }
    // Pinned destination when the graph cache holds one: the D2H is a DMA, not a staged pageable copy.
    std::vector<int32_t> h_out_pageable;
    int32_t* h_out = mtp_head_graphs_ != nullptr ? mtp_head_graphs_->h_out.as<int32_t>() : nullptr;
    if (h_out == nullptr) {
        h_out_pageable.resize(static_cast<size_t>(n_out));
        h_out = h_out_pageable.data();
    }
    if (cudaMemcpyAsync(h_out, ws->d_chain_tokens, static_cast<size_t>(n_out) * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        fail_all("batched draft: D2H");
        return;
    }
    {
        t_head_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        t_rows += n;
        if (++t_calls % 100 == 0)
            IMP_LOG_INFO("mtp-batched-feed: %lld calls, %.2f rows/call, %.2f ms/call (feed + LM head + sync)",
                         t_calls, static_cast<double>(t_rows) / t_calls, t_head_ms / t_calls);
    }
    const bool two = n_out == 2 * m;
    for (int i = 0; i < m; ++i) {
        const int g = bound_g[i];
        const int32_t tok2 = two ? h_out[static_cast<size_t>(m) + i] : -1;
        mtp_batched_set_draft_(*reqs[g], emitted[g], h_out[static_cast<size_t>(i)], tok2);
    }
}

int Engine::mtp_head_pass_run_(const imp::MtpPassTables& tab, const void* hidden,
                               const imp::NvFP4QuantResult* head, bool chain, int feed_max, int chain_max,
                               cudaStream_t stream) {
    auto* ws = mtp_ws_storage_.get();
    const int hidden_dim = model_->config_.d_model;
    const int m = tab.m;
    // diagnostics.mtp_prenorm_h (default on): the head takes the target's
    // post-norm hidden, applied to the gathered rows inside the feed.
    const Tensor* post_norm = runtime_config_.diagnostics.mtp_prenorm_h ? &model_->output_norm() : nullptr;
    if (!imp::mtp_feed_rows_multislot_dev(tab.tokens, hidden, tab.src, tab.n, tab.slots, tab.pos, feed_max,
                                          *model_->mtp_, model_->tok_emb_, *ws, hidden_dim, stream, post_norm,
                                          model_->config_.rms_norm_eps, model_->config_.norm_weight_offset))
        return -1;
    // The next draft per request: LM head over the last fed row of each.
    imp::mtp_gather_rows(ws->d_b_h_final, tab.last, ws->d_b_gather, hidden_dim, m, stream);
    if (!executor_->lm_head_rows_argmax(ws->d_b_gather, m, ws->d_chain_tokens, stream, head))
        return -2;
    if (!chain)
        return m;
    // Second draft per bound request: (draft 1, the head's own final-norm row) at the request's next position
    // in one more ragged pass, the device chain of mtp_feed_pairs_ batched over requests (draft 1 never leaves
    // the device).
    const bool two = imp::mtp_feed_rows_multislot_dev(ws->d_chain_tokens, ws->d_b_h_final, tab.last, m,
                                                      tab.chain_slots, tab.chain_pos, chain_max, *model_->mtp_,
                                                      model_->tok_emb_, *ws, hidden_dim, stream) &&
                     executor_->lm_head_rows_argmax(ws->d_b_h_final, m, ws->d_chain_tokens + m, stream, head);
    return two ? 2 * m : m;
}

// A bound request after the batched feed: its emitted tokens join the head history, the drafts (tok2 < 0 =
// one draft; the request's own mtp_k bounds the chain) become its pending chain for the next verify.
void Engine::mtp_batched_set_draft_(const Request& req, int emitted, int32_t tok, int32_t tok2) {
    auto* ws = mtp_ws_storage_.get();
    const int vocab_size = model_->config_.vocab_size;
    mtp_activate_(req.id);
    const int32_t* t = req.output_tokens.data() + req.output_tokens.size() - static_cast<size_t>(emitted);
    mtp_active_.history.insert(mtp_active_.history.end(), t, t + emitted);
    ws->mtp_pos += emitted;
    mtp_active_.chains.clear();
    if (tok < 0 || tok >= vocab_size) {
        mtp_active_.pending.clear();
        mtp_active_.draft_ctx = -1;
        mtp_pool_.pending_prediction = -1;
        return;
    }
    mtp_active_.pending.assign(1, tok);
    if (tok2 >= 0 && tok2 < vocab_size && mtp_request_k_(req) >= 2)
        mtp_active_.pending.push_back(tok2);
    mtp_active_.draft_ctx = static_cast<int>(mtp_active_.history.size());
    mtp_pool_.pending_prediction = tok;
}

void Engine::mtp_carry_rows_(const std::vector<std::shared_ptr<Request>>& rows, int pos_bias,
                             cudaStream_t stream) {
    auto* ws = mtp_ws_storage_.get();
    if (ws == nullptr || ws->d_carry == nullptr || rows.size() < 2 ||
        (mtp_active_.req < 0 && mtp_pool_.binds.empty()))
        return;
    const int n = static_cast<int>(rows.size());
    const size_t row_bytes = static_cast<size_t>(imp::mtp_h_prev_cols(*ws)) * sizeof(__half);
    Tensor h = executor_->view_mtp_hidden(n);
    if (h.data == nullptr)
        return;
    for (int g = 0; g < n; ++g) {
        const Request& req = *rows[g];
        if (req.status != RequestStatus::DECODING || !mtp_bound(mtp_active_, mtp_pool_, req.id))
            continue;
        mtp_activate_(req.id);
        const int pos = req.context_len() - 1 + pos_bias;
        int idx = mtp_carry_index(ws->mtp_pos, mtp_active_.carry_n, pos);
        if (idx >= ws->carry_rows_cap) {
            mtp_carry_flush_(req);
            if (mtp_active_.req != req.id)
                continue;  // the feed failed and unbound it
            idx = mtp_carry_index(ws->mtp_pos, mtp_active_.carry_n, pos);
        }
        if (idx < 0 || idx >= ws->carry_rows_cap) {
            mtp_active_.carry_n = -1;  // a step without a row: the lone step unbinds it
            continue;
        }
        char* dst = static_cast<char*>(ws->d_carry) +
                    ((static_cast<size_t>(ws->cur_slot) * ws->carry_rows_cap + idx) * row_bytes);
        if (cudaMemcpyAsync(dst, static_cast<const char*>(h.data) + (static_cast<size_t>(g) * row_bytes),
                            row_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            mtp_active_.carry_n = -1;
            continue;
        }
        mtp_active_.carry_n = idx + 1;
    }
}

void Engine::mtp_carry_flush_(const Request& req, int draft_k) {
    auto* ws = mtp_ws_storage_.get();
    if (ws == nullptr || ws->d_carry == nullptr || !mtp_bound(mtp_active_, mtp_pool_, req.id))
        return;
    mtp_activate_(req.id);
    const int n = mtp_carry_feedable(ws->mtp_pos, mtp_active_.carry_n, req.context_len());
    if (n <= 0)
        return;
    const int left = mtp_active_.carry_n - n;
    if (left > n) {
        mtp_active_.carry_n = -1;  // the kept rows would overlap their new place; never with a one-step lag
        return;
    }
    const auto& in = req.input_tokens;
    const auto& out = req.output_tokens;
    const int n_in = static_cast<int>(in.size());
    std::vector<int32_t> toks(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        const int p = ws->mtp_pos + j + 1;
        toks[static_cast<size_t>(j)] = p < n_in ? in[static_cast<size_t>(p)]
                                                : out[static_cast<size_t>(p - n_in)];
    }
    const size_t row_bytes = static_cast<size_t>(imp::mtp_h_prev_cols(*ws)) * sizeof(__half);
    char* base = static_cast<char*>(ws->d_carry) +
                 (static_cast<size_t>(ws->cur_slot) * ws->carry_rows_cap * row_bytes);
    // The last carried row is h_{P-1} of context end P, so pair (t_P, h_{P-1}) drafts the token at P + 1.
    const bool draft = draft_k > 0 && left == 0 && req.status == RequestStatus::DECODING;
    const int k_live = mtp_active_.k_live;  // the chain depth the request's next lone verify resumes at
    mtp_active_.k_live = draft ? draft_k : k_live;
    const bool fed = mtp_feed_pairs_(toks.data(), base, n, /*chain_after=*/draft);
    mtp_active_.k_live = k_live;
    if (!fed) {
        mtp_unbind_("carried feed failed (kv cap or forward error)");
        return;
    }
    if (left > 0 && cudaMemcpyAsync(base, base + (static_cast<size_t>(n) * row_bytes),
                                    static_cast<size_t>(left) * row_bytes, cudaMemcpyDeviceToDevice,
                                    decode_stream()) != cudaSuccess) {
        mtp_active_.carry_n = -1;
        return;
    }
    mtp_active_.carry_n = left;
}

}  // namespace imp
