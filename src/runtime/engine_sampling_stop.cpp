// Engine sampling helpers + stop-token detection: fill_sampling_params
// (per-request config -> InferenceState), upload_penalties, fill_recurrent_state
// (SSM/GDN), is_stop_token, track_think_state (<think> budget), should_stop (aggregate EOS/max_tokens/stop_strings).

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/batch.h"
#include "runtime/snapshot_boundary.h"
#include "runtime/think_stop_logic.h"
#include "model/chat_template.h"
#include "core/logging.h"
#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <span>
#include <stdexcept>
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
    // Banned tokens (e.g. <pad>) should also trigger stop: they indicate
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
        req.content_after_think = false;  // fresh post-think grace window
        return;
    }

    // Text-based fallback: NVFP4 SafeTensors loaders (Qwen3.6, Qwen3-Coder)
    // ship <think>/</think> as added_tokens with special=False; think_*_id_
    // stay -1, and </think> arrives as a 3-token BPE sequence the single-id
    // compare above can never see. Without this, a chat-template-primed empty think block closes and the next token hits should_stop with in_think_block=false: 0-content completion.
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
        req.content_after_think = false;  // fresh post-think grace window
    }
}

bool Engine::should_stop(Request& req, int32_t token) const {
    if (req.ignore_eos)
        return false;
    // Inside <think>...</think>: suppress stop tokens so reasoning can complete.
    // The model may generate <|im_end|> during reasoning as part of its internal
    // monologue: stopping here produces empty content (llama.cpp ignores this).
    if (req.in_think_block) {
        // With a think budget the sampler masks stop ids here (stop_mask_active_)
        // and this branch is not reached. Without one, a stop token inside
        // thinking is an implicit </think> (NVFP4 quants on Qwen3.6 skip the
        // explicit close); flipping the flag lets the next stop finish the request instead of freezing forever with every EOS masked.
        if (is_stop_token(token)) {
            req.in_think_block = false;
            req.think_exit_idx = static_cast<int>(req.output_tokens.size());
            req.content_after_think = false;  // fresh post-think grace window
            req.think_text_tail.clear();
        }
        return false;
    }
    // After </think>: suppress a too-eager stop ONLY while no real answer
    // content has been emitted yet (content_after_think). NVFP4 quantization
    // noise on Qwen3.6 can close an empty thinking block in ~3 tokens and emit
    // <|im_end|> to a zero-content completion. Once real content appears, honour the stop instantly; the budget stays a HARD CAP for the no-content case.
    if (req.think_exit_idx >= 0 && is_stop_token(token)) {
        if (think_logic::grace_blocks_stop(req.think_exit_idx,
                                           static_cast<int>(req.output_tokens.size()),
                                           req.content_after_think))
            return false;
    } else if (req.think_exit_idx >= 0 &&
               static_cast<int>(req.output_tokens.size()) > req.think_exit_idx &&
               !token_is_whitespace(token)) {
        // A non-stop, non-whitespace token after </think> is real answer
        // content: release the grace so the next stop is honoured immediately.
        // Whitespace/newline right after the close must NOT release it, or a stop right after yields a 0-content completion (#798).
        req.content_after_think = true;
    }
    return is_stop_token(token);
}

void Engine::fill_sampling_params(Request& req, InferenceState& state) const {
    state.temperature = req.temperature;
    state.top_p = req.top_p;
    state.top_k = req.top_k;
    state.seed = engine_internal::compute_step_seed(req);
    state.min_p = req.min_p;
    state.typical_p = req.typical_p;
    state.repetition_penalty = req.repetition_penalty;
    state.frequency_penalty = req.frequency_penalty;
    state.presence_penalty = req.presence_penalty;
    state.repeat_last_n = req.repeat_last_n;
    // Remaining output allowance: the JSON constrainer uses it to force the
    // document closed before max_tokens truncates it mid-structure (#1104).
    state.constrain_remaining_tokens = req.max_tokens - static_cast<int>(req.output_tokens.size());
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

    // Logit bias (unconditional for the same reason as the banned list below).
    state.logit_bias = req.logit_bias.empty() ? nullptr : req.logit_bias.data();
    state.n_logit_bias = static_cast<int>(req.logit_bias.size());

    // Banned tokens (chat template special tokens that must not be generated).
    // Assigned unconditionally: batched decode fills every row's state from a
    // copy of the batch state, and an empty list left row 0's stop mask
    // (below) on every other row, so a row that had closed its think block
    // could not stop while row 0 was still thinking (17 x "</think> + answer"
    // measured on Qwen3.8-27B at 8 streams).
    state.banned_tokens = banned_token_ids_.empty() ? nullptr : banned_token_ids_.data();
    state.n_banned_tokens = static_cast<int>(banned_token_ids_.size());
    // Device twin of the same list: the eager sampler bans with one kernel instead of one
    // 4-byte H2D copy per id (gemma-3: ~6250 ids, 50 % GPU idle in pp512).
    state.d_banned_tokens = d_banned_tokens_.get();
    state.n_d_banned_tokens = state.d_banned_tokens ? state.n_banned_tokens : 0;

    // Force </think> via logit manipulation when the budget is exceeded: the
    // model generates it itself so it lands in the KV cache correctly. Scans
    // output_tokens directly; injected <think> prefixes need started_in_think as the recount seed (think_stop_logic.h has the pure logic).
    state.force_token = -1;
    if (req.harmony_force_idx >= 0) {
        // Mid-opener: keep forcing the Harmony final-channel sequence until it
        // is fully emitted, then hand control back to the model (now committed
        // to the answer channel).
        if (req.harmony_force_idx < static_cast<int>(harmony_force_seq_.size())) {
            state.force_token = harmony_force_seq_[req.harmony_force_idx];
            req.harmony_force_idx++;
        }
        if (req.harmony_force_idx >= static_cast<int>(harmony_force_seq_.size()))
            req.harmony_force_idx = -1;  // opener complete
    } else if (think_logic::should_force_think_end(req.think_budget, think_end_id_, req.max_tokens,
                                                   req.output_tokens, think_start_id_, req.started_in_think,
                                                   runtime_config_.runtime.think_answer_reserve)) {
        if (harmony_reasoning_ && !harmony_force_seq_.empty()) {
            // Start forcing the full <|end|>…<|message|> opener (see above).
            state.force_token = harmony_force_seq_[0];
            req.harmony_force_idx = 1;
        } else {
            // <think> models: "\n" then </think>, the way the model ends reasoning on its own
            // and the template renders a prior turn (reasoning + "\n</think>"). A bare forced
            // </think> re-tokenizes one token off and the reply's KV is never reusable.
            const bool nl_done = think_newline_id_ < 0 || req.output_tokens.back() == think_newline_id_;
            state.force_token = nl_done ? think_end_id_ : think_newline_id_;
        }
    }

    // Sampler-side stop mask (think_logic::stop_mask_active): on steps where
    // should_stop would suppress a stop token, take it out of the logits
    // instead, so it never enters the context. A forced token is the whole distribution already.
    if (state.force_token < 0 && stop_mask_.n() > 0 && stop_mask_active(req, think_end_id_)) {
        state.banned_tokens = stop_mask_.ids.data();
        state.n_banned_tokens = stop_mask_.n();
        state.d_banned_tokens = stop_mask_.d_ids.get();
        state.n_d_banned_tokens = state.d_banned_tokens ? state.n_banned_tokens : 0;
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

// Recurrent state-slot allocator. One slot per concurrent sequence; slots must
// be UNIQUE among live sequences (the recurrent state is the sequence memory).
// The free list is sized lazily from the state capacity (== max_batch_size).
int Engine::acquire_recurrent_slot_(int req_id) {
    const int cap = ssm_state_ ? ssm_state_->max_sequences() : 0;
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
    // The admission gate committed this slot already on the scheduled path;
    // this is the catch for every other path (imp-cli, the aliasing fallback
    // above). A slab that is not there cannot be handed to a kernel.
    if (!ssm_state_->ensure_slot(slot)) {
        if (recurrent_slot_of_.find(req_id) == recurrent_slot_of_.end())
            free_recurrent_slots_.push_back(slot);
        throw std::runtime_error("recurrent state slot " + std::to_string(slot) +
                                 " could not be committed: the card cannot spare it above the allocator "
                                 "headroom (vram.lazy_commit=false to allocate every slot at load)");
    }
    recurrent_slot_of_[req_id] = slot;
    return slot;
}

void Engine::trim_recurrent_slots_after_warmup_() {
    if (!ssm_state_ || !ssm_state_->lazy())
        return;
    int trimmed = 0, held = 0;
    for (int s = 0; s < ssm_state_->max_sequences(); ++s) {
        bool free_slot = true;
        for (const auto& [req, slot] : recurrent_slot_of_)
            if (slot == s) {
                free_slot = false;
                break;
            }
        if (!free_slot) {
            ++held;
            continue;
        }
        if (ssm_state_->slot_committed(s) && ssm_state_->decommit_slot(s))
            ++trimmed;
    }
    IMP_LOG_INFO("SSM state: %d slot(s) decommitted after warmup (%d still held), %.0f MiB committed",
                 trimmed, held, ssm_state_->committed_bytes() / (1024.0 * 1024.0));
}

bool Engine::recurrent_slot_admissible_() {
    const int cap = ssm_state_ ? ssm_state_->max_sequences() : 0;
    if (cap <= 0 || !ssm_state_->lazy())
        return true;
    if (!recurrent_slots_initialized_) {
        free_recurrent_slots_.clear();
        for (int s = cap - 1; s >= 0; --s)
            free_recurrent_slots_.push_back(s);
        recurrent_slots_initialized_ = true;
    }
    // Admitted requests take their slot later, at prefill: one scheduling round admits several
    // before any pops. Committing back() for each passed them all on the one committed slot and
    // the second acquire threw inside step(), failing the whole batch (Qwen3.6-35B-A3B, 60 of 68
    // concurrent requests internal_error). The request asking now pops after those already waiting.
    int waiting = 0;
    for (const int id : scheduler_->active_ids())
        if (recurrent_slot_of_.find(id) == recurrent_slot_of_.end())
            ++waiting;
    const int pos = static_cast<int>(free_recurrent_slots_.size()) - 1 - waiting;
    if (pos < 0)
        return true;  // max_batch_size holds the line, not this gate
    // acquire_recurrent_slot_ pops from the back: commit the slot this request will get.
    return ssm_state_->ensure_slot(free_recurrent_slots_[static_cast<size_t>(pos)]);
}

int Engine::recurrent_slot(int req_id) const {
    auto it = recurrent_slot_of_.find(req_id);
    return (it != recurrent_slot_of_.end()) ? it->second : -1;
}

void Engine::release_recurrent_slot_(int req_id) {
    // Batched verify spare, if the request held one.
    if (auto sp = bv_.spare_of.find(req_id); sp != bv_.spare_of.end()) {
        bv_.free.push_back(sp->second);
        bv_.spare_of.erase(sp);
    }
    auto it = recurrent_slot_of_.find(req_id);
    if (it == recurrent_slot_of_.end())
        return;  // idempotent: request never acquired a slot (dense model / pre-prefill cancel)
    // Factored spare: a request that finished right after an accept leaves a
    // row behind, and the next tenant of this slot must not inherit it.
    if (factored_spare_active(runtime_config_, bv_))
        bv_.clear_slots.push_back(it->second);
    free_recurrent_slots_.push_back(it->second);
    recurrent_slot_of_.erase(it);
}

bool Engine::fill_recurrent_state(const Request& req, InferenceState& state, bool reset,
                                  cudaStream_t stream) {
    if (!ssm_state_)
        return true;
    int slot;
    if (reset) {
        slot = acquire_recurrent_slot_(req.id);  // fresh slot for a new sequence
    } else {
        auto it = recurrent_slot_of_.find(req.id);
        // Decode / later prefill chunks reuse the slot acquired at offset==0.
        const int cap = ssm_state_->max_sequences();
        slot = (it != recurrent_slot_of_.end()) ? it->second : (cap > 0 ? req.id % cap : 0);
    }
    if (ssm_state_) {
        state.ssm_state = ssm_state_.get();
        state.ssm_seq_id = slot;
        if (reset) {
            // Prefix-cache hit with a matching recurrent snapshot: restore the
            // state at exactly req.cached_tokens instead of zeroing, so prefill
            // continues from the snapshot boundary. All snapshot copies run on the prefill stream, so save/restore/recycle ordering is the stream order.
            if (req.recurrent_restore && req.recurrent_restore->data &&
                recurrent_snapshots_ &&
                recurrent_snapshots_->entry_bytes() == ssm_state_->per_seq_bytes()) {
                // Transcript snapshot (unaligned): its partial tail block comes with the
                // state, cloned into this request's own block before the chunk appends to it.
                const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
                const int n = req.recurrent_restore->n_tokens;
                if (n % bs != 0 && !kv_manager_->clone_held_block(req.id, n / bs, stream)) {
                    IMP_LOG_ERROR("RecurrentSnapshot: tail block of the %d-token transcript restore for req %d is gone",
                                  n, req.id);
                    return false;
                }
                // cudaMemcpyDefault: the entry is a device slab or, from the
                // store's host tier, pinned host memory (H2D on the stream).
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                    ssm_state_->seq_base(slot), req.recurrent_restore->data,
                    ssm_state_->per_seq_bytes(), cudaMemcpyDefault, stream));
                IMP_LOG_DEBUG("RecurrentSnapshot: restored %d-token state for req %d (slot %d, %s)",
                              req.recurrent_restore->n_tokens, req.id, slot,
                              req.recurrent_restore->on_host ? "host tier" : "device");
            } else {
                ssm_state_->reset_sequence(slot, stream);
            }
        }
    }
    return true;
}

// ─── Recurrent-state snapshots (hybrid prefix caching) ──────────────────
// Dense models reuse prefix KV at block granularity; recurrent (SSM/GDN)
// state is cumulative, so prefill can only skip up to a snapshotted position.
// One snapshot is saved per prefill at the largest block-aligned position
// (step_prefill_one), keyed by the chained KV block hash; the scheduler caps KV block reuse to the longest restorable prefix so continuation prefill never rewrites blocks shared with other sequences.

int Engine::hybrid_prefix_reuse_limit_(Request& req) {
    req.recurrent_restore.reset();
    if (!recurrent_snapshots_ || !recurrent_snapshots_->enabled() || !ssm_state_)
        return 0;
    // Restoring means starting prefill at offset > 0: a chunked continuation.
    if (!supports_chunked_prefill_())
        return 0;
    // Vision prompts: image content is not represented in the token ids the
    // hash covers, so never match snapshots across them.
    if (req.vision_emb || req.image || vision_.has_input())
        return 0;
    const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    const int total = static_cast<int>(req.input_tokens.size());
    std::vector<size_t> hashes;
    int cached = kv_manager_->longest_cached_prefix_blocks(req.input_tokens, hashes);
    // At least one token must remain to forward (the model needs logits).
    int max_b = std::min(cached, (total - 1) / bs);
    kv_manager_->release_held_block(req.id);
    for (int b = max_b; b >= 1; --b) {
        // Transcript snapshot (unaligned, saved at finish) before the aligned entry: it
        // reaches further into the same block. Its full blocks are the LAST cached ones (the
        // tail block is cached under the transcript key, never as a full-block hash), so only
        // b == max_b can carry one.
        if (b == max_b && transcript_snapshot_active_()) {
            const int base = b * bs;
            const auto toks = std::span<const int32_t>(req.input_tokens);
            for (int m = std::min(bs - 1, total - 1 - base); m >= 1; --m) {
                const size_t key = transcript_tail_key(hashes[b - 1], toks.subspan(static_cast<size_t>(base), m));
                auto entry = recurrent_snapshots_->find(key);
                if (!entry || entry->n_tokens != base + m)
                    continue;
                if (kv_manager_->hold_cached_block(key, req.id) < 0)
                    continue;  // the tail block was reclaimed; the state alone cannot restore
                req.recurrent_restore = std::move(entry);
                return b;
            }
        }
        auto entry = recurrent_snapshots_->find(hashes[b - 1]);
        if (entry && entry->n_tokens == b * bs) {
            req.recurrent_restore = std::move(entry);
            return b;
        }
    }
    IMP_LOG_DEBUG("RecurrentSnapshot: no restorable prefix for req %d (%d/%d blocks cached, %d tokens)", req.id,
                  cached, total / bs, total);
    return 0;
}

bool Engine::transcript_snapshot_active_() const {
    if (!recurrent_snapshots_ || !recurrent_snapshots_->enabled() || !ssm_state_)
        return false;
    if (!runtime_config_.server.transcript_snapshot || !supports_chunked_prefill_())
        return false;
    // copy_blocks_device leaves the key min/max metadata behind; a cloned tail block
    // would score with stale pages.
    if (kv_cache_raw_ && kv_cache_raw_->key_minmax_enabled())
        return false;
    // Factored spare: the accepted draft row sits in the factor buffer, not the slab, so
    // the slab at finish is one token short of the transcript.
    if (factored_spare_active(runtime_config_, bv_))
        return false;
    return true;
}

void Engine::maybe_save_transcript_snapshot_(const Request& req, std::span<const int32_t> forwarded,
                                             cudaStream_t stream) {
    if (!transcript_snapshot_active_())
        return;
    if (req.vision_emb || req.image || req.n_vision_tokens > 0 || vision_.has_input())
        return;
    const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    const int n = static_cast<int>(forwarded.size());
    // No snapshot_min_prompt_tokens floor here: that floor prices the prefill SPLIT a
    // prompt-boundary snapshot costs, and a finish-time save splits nothing (0.1-2 ms).
    // Short chats are the common case: a 73-token prompt + 174-token reply re-prefilled
    // its whole 247-token transcript on turn 2 (cached 0) under the 256 floor.
    if (n < bs)
        return;
    const size_t key = transcript_snapshot_key(forwarded, bs);
    if (key == 0 || recurrent_snapshots_->contains(key))
        return;  // same transcript already stored (e.g. restored at exactly this position)
    auto it = recurrent_slot_of_.find(req.id);
    if (it == recurrent_slot_of_.end())
        return;
    const auto t0 = std::chrono::steady_clock::now();
    if (!recurrent_snapshots_->save(key, n, ssm_state_->seq_base(it->second), stream))
        return;
    // The tail block must survive free_sequence: hash it now, under the same key.
    kv_manager_->register_partial_block(req.id, forwarded, key);
    // release_recurrent_slot_ runs right after and the next tenant's prefill writes the
    // slot on another stream: the copy must be complete before this returns.
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    IMP_LOG_DEBUG("RecurrentSnapshot: saved %d-token transcript state for req %d (%d/%d slots) in %.1f ms", n,
                  req.id, recurrent_snapshots_->size(), recurrent_snapshots_->capacity(),
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Engine::maybe_save_recurrent_snapshot_(const Request& req, int snap_end, cudaStream_t stream) {
    if (!recurrent_snapshots_ || !recurrent_snapshots_->enabled() || !ssm_state_ || snap_end <= 0)
        return;
    const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    size_t key = 0;
    for (int b = 0; b < snap_end / bs; ++b) {
        key = KVCacheManager::compute_block_hash(
            std::span<const int32_t>(req.input_tokens).subspan(static_cast<size_t>(b) * bs, bs), key);
    }
    if (recurrent_snapshots_->contains(key))
        return;  // identical prefix already snapshotted (e.g. it was just restored)
    auto it = recurrent_slot_of_.find(req.id);
    if (it == recurrent_slot_of_.end())
        return;
    const auto t0 = std::chrono::steady_clock::now();
    if (recurrent_snapshots_->save(key, snap_end, ssm_state_->seq_base(it->second), stream)) {
        // The copy must complete before anything else mutates the slot: later
        // prefill chunks are ordered on this stream, but the first DECODE step
        // may run on a different stream (green contexts). One sync per prefill.
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        // The wall time sits inside the TTFT (between the prefill chunks):
        // the D2D save plus everything queued ahead of it on the stream.
        IMP_LOG_DEBUG(
            "RecurrentSnapshot: saved %d-token state for req %d (%d/%d slots) in %.1f ms", snap_end, req.id,
            recurrent_snapshots_->size(), recurrent_snapshots_->capacity(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
}

// ─── SWA window snapshots (prefix caching under SWA sizing) ─────────────
// Same admission/save pattern as the hybrid pair above, but the restorable
// state is the packed windowed-layer KV (trailing window at the reuse
// boundary) instead of a recurrent slab. Hybrid models are excluded from SWA sizing, so at most one of the two snapshot stores is live.

int Engine::swa_prefix_reuse_limit_(Request& req) {
    req.swa_restore.reset();
    if (!swa_snapshots_ || !swa_snapshots_->enabled())
        return 0;
    // Restoring means starting prefill at offset > 0: a chunked continuation.
    if (!supports_chunked_prefill_())
        return 0;
    if (req.vision_emb || req.image || vision_.has_input())
        return 0;
    const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    const int total = static_cast<int>(req.input_tokens.size());
    std::vector<size_t> hashes;
    int cached = kv_manager_->longest_cached_prefix_blocks(req.input_tokens, hashes);
    // At least one token must remain to forward (the model needs logits).
    int max_b = std::min(cached, (total - 1) / bs);
    for (int b = max_b; b >= 1; --b) {
        auto entry = swa_snapshots_->find(hashes[b - 1]);
        if (entry && entry->n_tokens == b * bs) {
            req.swa_restore = std::move(entry);
            return b;
        }
    }
    return 0;
}

int Engine::snapshot_end_(const Request& req) const {
    // Hybrid: the recurrent store must be live; otherwise the SWA store.
    if (ssm_state_ ? !(recurrent_snapshots_ && recurrent_snapshots_->enabled())
                   : !(swa_snapshots_ && swa_snapshots_->enabled()))
        return 0;
    if (!supports_chunked_prefill_())
        return 0;
    if (req.vision_emb || req.image || vision_.has_input())
        return 0;
    const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    return snapshot_boundary(static_cast<int>(req.input_tokens.size()), bs,
                             runtime_config_.server.snapshot_min_prompt_tokens);
}

// Core save: snapshots the seq's live window at the block-floor of `tokens`.
// Called at the prefill snapshot boundary (tokens = prompt) and at request
// finish (prompt + generated-minus-final, same span finish_request_release_ uses): the finish save lets the NEXT agent turn reuse the whole transcript.
void Engine::maybe_save_swa_snapshot_span_(int seq_id, std::span<const int32_t> tokens,
                                           cudaStream_t stream, bool hard_sync) {
    if (!swa_snapshots_ || !swa_snapshots_->enabled() || !swa_snap_slab_)
        return;
    const int bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    // One block short of the length, not the plain block floor: the restore
    // caps at (total-1)/bs blocks (a restore must leave a token to forward)
    // AND requires entry->n_tokens == b*bs, so a snapshot at the full aligned length could never match (same defect the hybrid path had, snapshot_boundary.h).
    const int snap_end = snapshot_boundary(static_cast<int>(tokens.size()), bs, /*min_tokens=*/0);
    if (snap_end <= 0)
        return;
    size_t key = 0;
    for (int b = 0; b < snap_end / bs; ++b) {
        key = KVCacheManager::compute_block_hash(tokens.subspan(static_cast<size_t>(b) * bs, bs),
                                                 key);
    }
    if (swa_snapshots_->contains(key))
        return;  // identical prefix already snapshotted (e.g. it was just restored)
    const auto t0 = std::chrono::steady_clock::now();
    if (!kv_manager_->swa_snapshot_pack(seq_id, snap_end, swa_snap_slab_, stream))
        return;  // read-relevant window not fully resident
    if (swa_snapshots_->save(key, snap_end, swa_snap_slab_, stream)) {
        const auto t1 = std::chrono::steady_clock::now();
        // Sync policy:
        //  - prefill-boundary save (hard_sync=false): NO sync needed. SWA blocks
        //    are private to this sequence; later writers (this request's own
        //    chunks/decode) are stream-ordered or gated by host causality (the
        //    host reads the first sampled token via a D2H enqueued after the pack).
        //  - finish save (hard_sync=true): free_sequence recycles the window
        //    blocks right after return, and another request already in decode
        //    can write them on ANOTHER stream with no ordering against the pack, so this needs a sync.
        if (hard_sync)
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        const auto t2 = std::chrono::steady_clock::now();
        // Permanent telemetry (ms/verify precedent): one line per save, rare
        // (<= 2 per request) and the only way to attribute warm-TTFT cost
        // between the enqueue half and the sync half.
        IMP_LOG_INFO("SwaSnapshot: saved %d-token window for seq %d (%d/%d slots, "
                     "enqueue %.2f ms + sync %.2f ms)",
                     snap_end, seq_id, swa_snapshots_->size(), swa_snapshots_->capacity(),
                     std::chrono::duration<double, std::milli>(t1 - t0).count(),
                     std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
}


}  // namespace imp
