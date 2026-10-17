// Banned phrases (roadmap row 118): every append re-decodes the new token into the request's text;
// a completed phrase rewinds output_tokens and the KV to the token the phrase starts in and bans
// that token at that position. Position 0 resamples from the prefill's saved logits instead of a
// second forward over the prompt's last token. Hybrids restore a recurrent slab saved after the
// prefill or every kPhraseSnapEvery tokens and re-forward the kept tokens after it.

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

// Hybrids: per request, the recurrent slabs of PhraseBanState::snap_consumed and the logprobs and
// lens of the tokens a rewind replays.
struct Engine::PhraseRecurrent {
    std::array<VramOwned<uint8_t>, kPhraseSnapSlots> slab;
    std::vector<TokenLogprobInfo> replay_lp;
    std::vector<std::vector<LogitLensLayer>> replay_lens;
};

void Engine::PhraseRecurrentFree::operator()(PhraseRecurrent* p) const { delete p; }

Engine::PhraseRecurrent& Engine::phrase_rec_of_(int req_id) {
    auto& p = phrase_rec_[req_id];
    if (!p)
        p.reset(new PhraseRecurrent());
    return *p;
}

void Engine::phrase_release_(int req_id) {
    if (!phrase_rec_.count(req_id))
        return;
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(decode_stream()));  // a snapshot copy may still be queued
    phrase_rec_.erase(req_id);
}

size_t Request::deliverable_tokens() const {
    if (!phrase || status == RequestStatus::FINISHED || status == RequestStatus::CANCELLED)
        return output_tokens.size();
    return std::min(phrase->safe, output_tokens.size());
}

bool Engine::phrase_ban_supported() const {
    return model_ && !(kv_manager_ && kv_manager_->residual_enabled());
}

void Engine::phrase_ban_init_(Request& req) const {
    if (!req.phrase)
        return;
    if (!phrase_ban_supported()) {  // the server answers 400 first; direct API users get no ban
        IMP_LOG_WARN("phrase ban: req %d dropped, the model keeps a residual KV ring", req.id);
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
    if (ssm_state_)
        phrase_snapshot_(req, 0, 0, stream);  // the prefill's final state, no output token in it yet
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
}

// Copies the request's recurrent slab into slab `slot`, recorded as holding `consumed` output tokens.
// Slot 0 failing to allocate drops the ban: no rewind could restore a state.
void Engine::phrase_snapshot_(Request& req, int slot, int consumed, cudaStream_t stream) {
    PhraseRecurrent& rec = phrase_rec_of_(req.id);
    auto& slab = rec.slab[static_cast<size_t>(slot)];
    const size_t bytes = ssm_state_->per_seq_bytes();
    if (slab.count() < bytes)
        slab = VramOwned<uint8_t>(vram_alloc_, bytes, "phrase_ban_recurrent");
    if (!slab) {
        IMP_LOG_WARN("phrase ban: req %d, no VRAM for a %.1f MiB recurrent snapshot%s", req.id,
                     static_cast<double>(bytes) / (1024.0 * 1024.0), slot == 0 ? ", ban dropped" : "");
        req.phrase->snap_consumed[static_cast<size_t>(slot)] = -1;
        if (slot == 0) {
            phrase_rec_.erase(req.id);
            req.phrase.reset();
        }
        return;
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(slab.get(), ssm_state_->seq_base(recurrent_slot_for_(req.id)), bytes,
                                       cudaMemcpyDeviceToDevice, stream));
    req.phrase->snap_consumed[static_cast<size_t>(slot)] = consumed;
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
    const size_t n_out = req.output_tokens.size();
    if (ssm_state_ && n_out > 1 &&
        (n_out - 1) % kPhraseSnapEvery == 0)  // the forward just run read n_out - 1
        phrase_snapshot_(req, phrase_snapshot_slot(s.snap_consumed), static_cast<int>(n_out - 1), stream);
    if (s.replaying()) {  // a kept token re-forwarded: its original logprobs and lens, no new match
        auto& rec = phrase_rec_of_(req.id);
        if (s.replay_at < rec.replay_lp.size() && req.output_logprobs.size() == n_out)
            req.output_logprobs.back() = std::move(rec.replay_lp[s.replay_at]);
        if (s.replay_at < rec.replay_lens.size() && req.output_lens.size() == n_out)
            req.output_lens.back() = std::move(rec.replay_lens[s.replay_at]);
        if (++s.replay_at == s.replay.size()) {
            s.replay.clear();
            s.replay_at = 0;
            rec.replay_lp.clear();
            rec.replay_lens.clear();
        }
        phrase_bias_at(req, n_out);
        return false;
    }
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
    // Hybrids: the recurrent state cannot step back, so restore the latest slab at or before
    // pos - 1 and re-forward the kept tokens after it (replay) before sampling pos.
    size_t keep = pos;
    if (ssm_state_) {
        const int slot = pos > 0 ? std::max(phrase_pick_snapshot(s.snap_consumed, pos), 0) : 0;  // 0 holds 0
        const int consumed = s.snap_consumed[static_cast<size_t>(slot)];
        keep = pos > 0 ? static_cast<size_t>(consumed) + 1 : 0;
        s.replay.assign(req.output_tokens.begin() + static_cast<std::ptrdiff_t>(keep),
                        req.output_tokens.begin() + static_cast<std::ptrdiff_t>(pos));
        s.replay_at = 0;
        auto& rec = phrase_rec_of_(req.id);
        rec.replay_lp.clear();
        rec.replay_lens.clear();
        for (size_t i = keep; i < pos && i < req.output_logprobs.size(); i++)
            rec.replay_lp.push_back(std::move(req.output_logprobs[i]));
        for (size_t i = keep; i < pos && i < req.output_lens.size(); i++)
            rec.replay_lens.push_back(std::move(req.output_lens[i]));
        for (auto& c : s.snap_consumed)  // a slab past pos folded tokens that are gone
            if (c > static_cast<int>(pos))
                c = -1;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(recurrent_slot_for_(req.id)),
                                           rec.slab[static_cast<size_t>(slot)].get(),
                                           ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice, stream));
        IMP_LOG_DEBUG("phrase ban: req %d restores the slab after %d tokens, replays %zu", req.id, consumed,
                      s.replay.size());
    }
    req.output_tokens.resize(keep);
    if (req.output_logprobs.size() > keep)
        req.output_logprobs.resize(keep);
    if (req.output_lens.size() > keep)
        req.output_lens.resize(keep);
    s.text.resize(keep > 0 ? s.tok_end[keep - 1] : 0);
    s.tok_end.resize(keep);
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
        // The next step re-forwards output[keep - 1] at its position; after the replay it samples
        // pos with the ban.
        kv_manager_->rollback(req.id, n_in + static_cast<int>(keep) - 1);
        phrase_bias_at(req, keep);
        return;
    }
    kv_manager_->rollback(req.id, n_in);  // the prompt stays; token 0 comes from the saved logits
    req.output_tokens.push_back(phrase_first_token_(req, s.bans.back().second, stream, /*resample=*/true));
}

}  // namespace imp
