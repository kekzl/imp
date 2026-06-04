// Engine sampling helpers + stop-token detection.
//
// fill_sampling_params:    pull per-request sampling config into InferenceState
// upload_penalties:        copy penalty buffers (repeat/freq/presence/DRY) to device
// fill_recurrent_state:    SSM/GDN per-request state setup
// is_stop_token:           single-token stop check (EOS variants)
// track_think_state:       update <think>...</think> blockcount budget
// should_stop:             aggregate stop check (EOS, max_tokens, stop_strings)
//
// Two related concern clusters colocated because they share the
// per-request state-passing pattern (Request& + InferenceState&) and
// run in the decode loop's tail.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. This is the final per-subsystem extraction.

#include "runtime/engine.h"
#include "runtime/batch.h"
#include "runtime/think_stop_logic.h"
#include "model/chat_template.h"
#include "core/logging.h"
#include <cstdlib>

#include <algorithm>
#include <string>

namespace imp {

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
    // Drive the pure text-tail state machine (see think_stop_logic.h) on the
    // Request's mirrored fields, then sync back. On a block EXIT, record the
    // output index for the post-</think> grace period in should_stop().
    think_logic::TextThinkState ts;
    ts.in_think_block = req.in_think_block;
    ts.think_text_tail = std::move(req.think_text_tail);
    bool was_in_think = ts.in_think_block;
    bool transitioned = ts.feed_piece(piece);
    req.in_think_block = ts.in_think_block;
    req.think_text_tail = std::move(ts.think_text_tail);
    if (transitioned && was_in_think && !req.in_think_block) {
        req.think_exit_idx = static_cast<int>(req.output_tokens.size());
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
        if (think_logic::grace_blocks_stop(req.think_exit_idx,
                                           static_cast<int>(req.output_tokens.size())))
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
    // Injected <think> prefixes live in the PROMPT — the output then has no
    // opener, so the recount must start in-think (req.started_in_think) or the
    // budget never fires (model thinks until max_tokens, content stays empty).
    // See think_stop_logic.h for the pure recount logic.
    state.force_token = -1;
    if (think_logic::should_force_think_end(req.think_budget, think_end_id_, req.max_tokens,
                                            req.output_tokens, think_start_id_,
                                            req.started_in_think)) {
        state.force_token = think_end_id_;
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
            memory_manager_.vram_allocator().free(d_penalty_tokens_);
        d_penalty_tokens_capacity_ = std::max(n, (size_t)256);
        d_penalty_tokens_ = static_cast<int32_t*>(
            memory_manager_.vram_allocator().allocate(d_penalty_tokens_capacity_ * sizeof(int32_t), "penalty_tokens"));
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

// Recurrent state-slot allocator. One slot per concurrent sequence; slots must
// be UNIQUE among live sequences (the recurrent state is the sequence memory).
// The free list is sized lazily from the state capacity (== max_batch_size).
int Engine::acquire_recurrent_slot_(int req_id) {
    const int cap = ssm_state_ ? ssm_state_->max_sequences()
                               : (gdn_state_ ? gdn_state_->max_sequences() : 0);
    if (cap <= 0)
        return 0;
    auto it = recurrent_slot_of_.find(req_id);
    if (it != recurrent_slot_of_.end())
        return it->second;  // already holds a slot (multi-chunk prefill)
    if (!recurrent_slots_initialized_) {
        free_recurrent_slots_.clear();
        for (int s = cap - 1; s >= 0; --s)
            free_recurrent_slots_.push_back(s);
        recurrent_slots_initialized_ = true;
    }
    int slot;
    if (!free_recurrent_slots_.empty()) {
        slot = free_recurrent_slots_.back();
        free_recurrent_slots_.pop_back();
    } else {
        // Should not happen: the scheduler caps concurrency at capacity. Fall
        // back to the legacy aliasing scheme rather than crash.
        slot = req_id % cap;
        IMP_LOG_WARN("recurrent slot pool exhausted (cap=%d) — falling back to id%%cap for req %d", cap,
                     req_id);
    }
    recurrent_slot_of_[req_id] = slot;
    return slot;
}

void Engine::release_recurrent_slot_(int req_id) {
    auto it = recurrent_slot_of_.find(req_id);
    if (it == recurrent_slot_of_.end())
        return;  // idempotent: request never acquired a slot (dense model / pre-prefill cancel)
    free_recurrent_slots_.push_back(it->second);
    recurrent_slot_of_.erase(it);
}

void Engine::fill_recurrent_state(const Request& req, InferenceState& state, bool reset,
                                  cudaStream_t stream) {
    if (!ssm_state_ && !gdn_state_)
        return;
    int slot;
    if (reset) {
        slot = acquire_recurrent_slot_(req.id);  // fresh slot for a new sequence
    } else {
        auto it = recurrent_slot_of_.find(req.id);
        // Decode / later prefill chunks reuse the slot acquired at offset==0.
        const int cap = ssm_state_ ? ssm_state_->max_sequences() : gdn_state_->max_sequences();
        slot = (it != recurrent_slot_of_.end()) ? it->second : (cap > 0 ? req.id % cap : 0);
    }
    if (ssm_state_) {
        state.ssm_state = ssm_state_.get();
        state.ssm_seq_id = slot;
        if (reset)
            ssm_state_->reset_sequence(slot, stream);
    }
    if (gdn_state_) {
        state.gdn_state = gdn_state_.get();
        state.gdn_seq_id = slot;
        if (reset)
            gdn_state_->reset_sequence(slot, stream);
    }
}


}  // namespace imp
