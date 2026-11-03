// Batched feeds of the MTP head: the batched verify's ragged pass over every bound request's emitted pairs
// (mtp_batched_feed_), and speculative.mtp_carry, which keeps a bound request's head cache in step through
// batched decode by carrying its hidden rows (mtp_carry_rows_) until their next token is known
// (mtp_carry_flush_). Pairing convention and the binding invariant: engine_spec_mtp.cpp.

#include "compute/mtp_forward.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "model/model.h"
#include "runtime/engine.h"
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
    for (int g = 0; g < N; ++g) {
        Request& req = *reqs[g];
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
    }
    if (bound_g.empty())
        return;
    const int n = static_cast<int>(toks.size());
    const int m = static_cast<int>(bound_g.size());
    const int hidden_dim = model_->config_.d_model;
    const int vocab_size = model_->config_.vocab_size;
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
    // diagnostics.mtp_prenorm_h (default on): the head takes the target's
    // post-norm hidden, applied to the gathered rows inside the feed.
    const Tensor* post_norm = runtime_config_.diagnostics.mtp_prenorm_h ? &model_->output_norm() : nullptr;
    if (!imp::mtp_feed_rows_multislot(toks.data(), h.data, src.data(), n, slots.data(), pos.data(),
                                      *model_->mtp_, model_->tok_emb_, *ws, hidden_dim, stream, post_norm,
                                      model_->config_.rms_norm_eps, model_->config_.norm_weight_offset)) {
        fail_all("batched feed failed");
        return;
    }
    // The next draft per request: LM head over the last fed row of each.
    if (m > imp::kMtpMaxTopW * imp::kMtpMaxChainK || ws->d_chain_tokens == nullptr) {
        fail_all("batched draft: too many rows");
        return;
    }
    if (cudaMemcpyAsync(ws->d_row_src, last_row.data(), static_cast<size_t>(m) * sizeof(int),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        fail_all("batched draft: index H2D");
        return;
    }
    imp::mtp_gather_rows(ws->d_b_h_final, ws->d_row_src, ws->d_b_gather, hidden_dim, m, stream);
    if (!executor_->lm_head_rows_argmax(ws->d_b_gather, m, ws->d_chain_tokens, stream)) {
        fail_all("batched draft: LM head unavailable");
        return;
    }
    std::vector<int32_t> h_out(static_cast<size_t>(m));
    if (cudaMemcpyAsync(h_out.data(), ws->d_chain_tokens, static_cast<size_t>(m) * sizeof(int32_t),
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
    for (int i = 0; i < m; ++i) {
        const int g = bound_g[i];
        Request& req = *reqs[g];
        mtp_activate_(req.id);
        const int32_t* t = req.output_tokens.data() + req.output_tokens.size() -
                           static_cast<size_t>(emitted[g]);
        for (int j = 0; j < emitted[g]; ++j)
            mtp_active_.history.push_back(t[j]);
        ws->mtp_pos += emitted[g];
        mtp_active_.chains.clear();
        const int32_t tok = h_out[static_cast<size_t>(i)];
        if (tok < 0 || tok >= vocab_size) {
            mtp_active_.pending.clear();
            mtp_active_.draft_ctx = -1;
            mtp_pool_.pending_prediction = -1;
        } else {
            mtp_active_.pending.assign(1, tok);
            mtp_active_.draft_ctx = static_cast<int>(mtp_active_.history.size());
            mtp_pool_.pending_prediction = tok;
        }
    }
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

void Engine::mtp_carry_flush_(const Request& req) {
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
    if (!mtp_feed_pairs_(toks.data(), base, n, /*chain_after=*/false)) {
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
