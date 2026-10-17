// Banned phrases (roadmap row 118): every append re-decodes the new token into the request's text;
// a completed phrase rewinds output_tokens and the KV to the token the phrase starts in and bans
// that token at that position. Position 0 resamples from the prefill's saved logits instead of a
// second forward over the prompt's last token. Dense attention only: a recurrent state has no
// history to rewind to.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/phrase_ban.h"
#include "runtime/request.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "model/model.h"
#include "model/tokenizer.h"

#include <algorithm>

namespace imp {

namespace {

// logit_bias for output position pos: the request's own entries plus the bans recorded there.
void phrase_bias_at(Request& req, size_t pos) {
    const PhraseBanState& s = *req.phrase;
    req.logit_bias = s.user_bias;
    for (const auto& [p, id] : s.bans)
        if (p == static_cast<int>(pos))
            req.logit_bias.emplace_back(id, -1e30f);
}

}  // namespace

size_t Request::deliverable_tokens() const {
    if (!phrase || status == RequestStatus::FINISHED || status == RequestStatus::CANCELLED)
        return output_tokens.size();
    return std::min(phrase->safe, output_tokens.size());
}

bool Engine::phrase_ban_supported() const {
    return model_ && !ssm_state_ && !(kv_manager_ && kv_manager_->residual_enabled());
}

void Engine::phrase_ban_init_(Request& req) const {
    if (!req.phrase)
        return;
    if (!phrase_ban_supported()) {  // the server answers 400 first; direct API users get no ban
        IMP_LOG_WARN("phrase ban: req %d dropped, the model keeps a recurrent state or a residual KV ring",
                     req.id);
        req.phrase.reset();
        return;
    }
    PhraseBanState& s = *req.phrase;
    s.user_bias = req.logit_bias;
    s.max_len = 0;
    for (const auto& p : s.phrases)
        s.max_len = std::max(s.max_len, p.size());
    s.in_think0 = req.in_think_block;
    s.think_exit0 = req.think_exit_idx;
    s.content_after0 = req.content_after_think;
    s.think_tail0 = req.think_text_tail;
}

void Engine::phrase_save_logits_(Request& req, const Tensor& logits, cudaStream_t stream) {
    if (!req.phrase || logits.data == nullptr)
        return;
    const auto v = static_cast<size_t>(logits.shape[logits.ndim - 1]);
    req.phrase->first_logits.resize(v);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(req.phrase->first_logits.data(), logits.data, v * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
}

int32_t Engine::phrase_first_token_(Request& req, int32_t token, cudaStream_t stream, bool resample) {
    if (!req.phrase)
        return token;
    PhraseBanState& s = *req.phrase;
    Tokenizer* tok = model_->tokenizer();
    const auto v = static_cast<int>(s.first_logits.size());
    std::string text = tok->decode_token(token);
    // resample: the caller already banned `token` at position 0 (a phrase that starts there).
    for (bool redo = resample;
         v > 0 && (redo || (s.rewinds < kMaxPhraseRewinds && phrase_match_start(text, 0, s.phrases)));
         redo = false) {
        if (!redo) {
            s.bans.emplace_back(0, token);
            s.rewinds++;
        }
        // Resample position 0 from the saved prefill logits with this position's bans.
        if (phrase_scratch_.count() < s.first_logits.size())
            phrase_scratch_ = VramOwned<float>(vram_alloc_, s.first_logits.size(), "phrase_ban_logits");
        if (!phrase_scratch_)
            break;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(phrase_scratch_.get(), s.first_logits.data(), v * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
        phrase_bias_at(req, 0);
        InferenceState st;
        fill_sampling_params(req, st);
        upload_penalties(req, st, stream);
        const int64_t shape[2] = {1, v};
        const Tensor lg(phrase_scratch_.get(), QType::F32, 2, shape, true);
        token = executor_->sample_from_logits(lg, st, stream)[0];
        text = tok->decode_token(token);
        const size_t pos = req.output_tokens.size();  // 0: the token is pushed by the caller
        if (req.logprobs) {
            executor_->ensure_logits_pinned(v);
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(executor_->h_logits_pinned(), phrase_scratch_.get(),
                                               v * sizeof(float), cudaMemcpyDeviceToHost, stream));
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            auto info = engine_internal::build_logprob_info(executor_->h_logits_pinned(), v, token,
                                                            req.top_logprobs, tok);
            if (req.output_logprobs.size() > pos)
                req.output_logprobs[pos] = std::move(info);
            else
                req.output_logprobs.push_back(std::move(info));
        }
        if (req.logit_lens > 0) {  // the lens read the forward of the token it replaced
            if (req.output_lens.size() > pos)
                req.output_lens[pos].clear();
            else
                req.output_lens.emplace_back();
        }
    }
    s.text = text;
    s.tok_end.assign(1, static_cast<uint32_t>(text.size()));
    s.safe = std::max(s.safe, phrase_safe_tokens(s.tok_end, s.text.size(), s.max_len));
    phrase_bias_at(req, 1);
    return token;
}

bool Engine::phrase_after_append_(Request& req, cudaStream_t stream) {
    if (!req.phrase)
        return false;
    PhraseBanState& s = *req.phrase;
    const size_t prev = s.text.size();
    s.text += model_->tokenizer()->decode_token(req.output_tokens.back());
    s.tok_end.push_back(static_cast<uint32_t>(s.text.size()));
    const auto hit = s.rewinds < kMaxPhraseRewinds ? phrase_match_start(s.text, prev, s.phrases)
                                                   : std::optional<size_t>{};
    const size_t pos = hit ? phrase_token_at(s.tok_end, *hit) : 0;
    if (!hit || pos < s.safe) {  // a delivered token is never taken back
        s.safe = std::max(s.safe, phrase_safe_tokens(s.tok_end, s.text.size(), s.max_len));
        phrase_bias_at(req, req.output_tokens.size());
        return false;
    }
    IMP_LOG_DEBUG("phrase ban: req %d rewinds %zu -> %zu tokens (ban id %d at %zu)", req.id,
                  req.output_tokens.size(), pos, req.output_tokens[pos], pos);
    s.bans.emplace_back(static_cast<int>(pos), req.output_tokens[pos]);
    s.rewinds++;
    phrase_rewind_(req, pos, stream);
    return pos > 0;  // position 0 was resampled in place: the caller goes on with the new token
}

void Engine::phrase_rewind_(Request& req, size_t pos, cudaStream_t stream) {
    PhraseBanState& s = *req.phrase;
    drain_decode_pipeline();
    req.output_tokens.resize(pos);
    if (req.output_logprobs.size() > pos)
        req.output_logprobs.resize(pos);
    if (req.output_lens.size() > pos)
        req.output_lens.resize(pos);
    s.text.resize(pos > 0 ? s.tok_end[pos - 1] : 0);
    s.tok_end.resize(pos);
    for (auto& hs : penalty_hist_state_)  // the device history holds the dropped tokens
        if (hs.req_id == req.id)
            hs.synced = -1;
    // Think state: back to its add_request seed, replayed over the kept tokens.
    req.in_think_block = s.in_think0;
    req.think_exit_idx = s.think_exit0;
    req.content_after_think = s.content_after0;
    req.think_text_tail = s.think_tail0;
    req.harmony_force_idx = -1;
    req.pipe_inflight_force = -1;
    std::vector<int32_t> kept = std::move(req.output_tokens);
    req.output_tokens.clear();
    for (const int32_t t : kept) {
        req.output_tokens.push_back(t);
        track_think_state(req, t);
        (void)should_stop(req, t);  // its think side effects; a kept token never stopped
    }
    const int n_in = static_cast<int>(req.input_tokens.size());
    if (pos > 0) {
        // The next step re-forwards output[pos - 1] at its position and samples pos with the ban.
        kv_manager_->rollback(req.id, n_in + static_cast<int>(pos) - 1);
        phrase_bias_at(req, pos);
        return;
    }
    kv_manager_->rollback(req.id, n_in);  // the prompt stays; token 0 comes from the saved logits
    req.output_tokens.push_back(phrase_first_token_(req, s.bans.back().second, stream, /*resample=*/true));
}

}  // namespace imp
