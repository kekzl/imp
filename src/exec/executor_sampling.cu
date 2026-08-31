// executor_sampling.cu — the per-row and row-batched sampling family of
// GraphExecutor, moved VERBATIM out of executor.cu on 2026-08-27 (the batched
// greedy/penalty stash pushed that TU over the 600-LOC kernel hard threshold;
// this is the compile-time-isolation split the filesize gate asks for:
// sampling edits no longer re-ptxas the forward paths). Only mechanical
// changes: the shared apply_constraint_mask helper now comes from
// executor_sampling_internal.h, and the one in-family ban_logits_kernel call
// goes through launch_ban_logits (the kernel stays in executor.cu).

#include "exec/executor.h"
#include "exec/executor_sampling_internal.h"
#include "compute/sampling.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

namespace imp {

std::vector<int32_t> GraphExecutor::sample_from_logits(const Tensor& logits, const InferenceState& state,
                                                       cudaStream_t stream) {
    int n_seq = state.n_sequences;
    std::vector<int32_t> tokens(n_seq);

    // Helper: flatten [1, V] to [V] for sampling
    auto flatten_logits = [](Tensor t) -> Tensor {
        int64_t vocab_shape[1] = {t.shape[t.ndim - 1]};
        return t.reshape(1, vocab_shape);
    };

    // Helper: apply penalties + filters to logits before sampling
    auto apply_pre_sample = [&](Tensor& seq_logits, const InferenceState& st) {
        float* lp = static_cast<float*>(seq_logits.data);
        int vocab = static_cast<int>(seq_logits.shape[0]);

        if (st.penalty_tokens != nullptr && st.n_penalty_tokens > 0) {
            const int32_t* pen_ptr = st.penalty_tokens;
            int pen_n = st.n_penalty_tokens;
            if (st.repeat_last_n > 0 && pen_n > st.repeat_last_n) {
                pen_ptr += (pen_n - st.repeat_last_n);
                pen_n = st.repeat_last_n;
            }
            apply_penalties(lp, vocab, pen_ptr, pen_n, st.repetition_penalty, st.frequency_penalty,
                            st.presence_penalty, stream);
        }
        if (st.dry_multiplier > 0.0f && st.host_penalty_tokens != nullptr && st.n_penalty_tokens > 0) {
            apply_dry_penalty(lp, vocab, st.host_penalty_tokens, st.n_penalty_tokens, st.dry_multiplier,
                              st.dry_base, st.dry_allowed_length, st.dry_penalty_last_n, stream);
        }
        if (st.n_logit_bias > 0 && st.logit_bias != nullptr)
            apply_logit_bias(lp, vocab, st.logit_bias, st.n_logit_bias, stream);
        apply_constraint_mask(st, lp, vocab, stream);
        // Ban special tokens (chat template delimiters etc.). MUST happen
        // before sampling — without this, greedy can pick a banned token
        // (e.g. Gemma-4 NVFP4 picks `<|channel>` as the natural argmax) and
        // the request finishes immediately because is_stop_token treats banned
        // tokens as stop tokens. forward() (line 88) already does this; the
        // sample_from_logits / sample_single_from_logits / use_event_sync
        // prefill paths historically forgot to. Match forward()'s impl.
        if (st.banned_tokens != nullptr && st.n_banned_tokens > 0) {
            float neg_inf = -1e30f;
            for (int bi = 0; bi < st.n_banned_tokens; bi++) {
                int32_t tid = st.banned_tokens[bi];
                if (tid >= 0 && tid < vocab) {
                    IMP_CUDA_CHECK_LOG(
                        cudaMemcpyAsync(lp + tid, &neg_inf, sizeof(float), cudaMemcpyHostToDevice, stream));
                }
            }
        }
        if (st.min_p > 0.0f) {
            apply_min_p(lp, vocab, st.min_p, stream);
        }
        if (st.typical_p > 0.0f && st.typical_p < 1.0f) {
            apply_typical_p(lp, vocab, st.typical_p, stream);
        }
    };

    if (state.is_prefill || n_seq <= 1) {
        // Single sequence or prefill: logits is [1, V] (forward_logits already sliced)
        Tensor last_logits = flatten_logits(logits.slice(0, 1));
        apply_pre_sample(last_logits, state);

        // Overlap prefill: the parity slots belong to the concurrent decode
        // batch — the engine points this path at its dedicated slot.
        int32_t* slot = sample_slot_override_ ? sample_slot_override_ : d_sample_result_;
        if (state.mirostat == 2) {
            unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
            tokens[0] = slot ? sample_mirostat_v2(last_logits, state.temperature, state.mirostat_tau,
                                                  state.mirostat_eta, &state.mirostat_mu, seed, slot,
                                                  stream)
                             : sample_mirostat_v2(last_logits, state.temperature, state.mirostat_tau,
                                                  state.mirostat_eta, &state.mirostat_mu, seed, stream);
        } else {
            tokens[0] =
                (state.temperature <= 0.0f || state.top_k == 1)
                    ? (slot ? sample_greedy(last_logits, slot, stream)
                            : sample_greedy(last_logits, stream))
                    : (slot ? sample_topk_topp(last_logits, state.top_k > 0 ? state.top_k : 50,
                                               state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                               state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                               slot, stream)
                            : sample_topk_topp(last_logits, state.top_k > 0 ? state.top_k : 50,
                                               state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                               state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                               stream));
        }
    } else {
        // Batched decode: n_tokens == n_sequences, each row is one sequence's
        // logits. Enqueue every sequence's penalty+sampler chain back-to-back
        // into per-sequence scratch slots, then gather ALL tokens with ONE
        // pinned D2H + ONE stream sync. The previous per-sequence readback
        // (pageable 4-byte D2H + cudaStreamSynchronize each) serialized the
        // batch against ~200 us host round-trips: at n=16 sustained load that
        // was ~3 ms host time per decode step = 29% GPU idle (nsys,
        // 2026-07-12). Kernels, parameter normalization, and per-sequence
        // seeds are identical to the fallback loop, so tokens are
        // bit-identical.
        const bool greedy = (state.temperature <= 0.0f || state.top_k == 1);
        const int top_k = state.top_k > 0 ? state.top_k : 50;
        const float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        // Eligibility is batch-uniform (depends only on shared sampling params
        // and vocab), so decide BEFORE applying any penalties — no sequence is
        // ever half-processed across the two paths.
        //
        // No top_k term any more (#1654): sample_topk_topp_async enqueues the
        // CUB regime too, so a top_k over SAMPLE_MAX_TOP_K no longer drops the
        // whole batch onto the per-sequence synchronous path. It used to, and
        // that cost 14.5% of aggregate throughput at six sequences.
        const bool can_batch = d_sample_result_ && h_sample_pinned_.as<int32_t>() && n_seq <= sample_slots_;
        if (can_batch) {
            for (int i = 0; i < n_seq; i++) {
                Tensor seq_logits = flatten_logits(logits.slice(i, i + 1));
                apply_pre_sample(seq_logits, state);
                auto* slot = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(d_sample_result_) +
                                                        static_cast<size_t>(i) * SAMPLE_SCRATCH_BYTES);
                if (greedy) {
                    sample_greedy_async(seq_logits, slot, stream);
                } else {
                    unsigned int seed =
                        state.seed >= 0 ? static_cast<unsigned int>(state.seed + i) : (42u + i);
                    bool ok = sample_topk_topp_async(seq_logits, top_k, top_p, state.temperature, seed,
                                                     slot, stream);
                    (void)ok;  // eligibility pre-checked above; cannot decline here
                }
            }
            // Slots are SAMPLE_SCRATCH_BYTES apart; the token is the first
            // int32 of each slot — one strided D2H gathers the whole batch.
            IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(h_sample_pinned_.as<int32_t>(), sizeof(int32_t), d_sample_result_,
                                                 SAMPLE_SCRATCH_BYTES, sizeof(int32_t), n_seq,
                                                 cudaMemcpyDeviceToHost, stream));
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            for (int i = 0; i < n_seq; i++)
                tokens[i] = h_sample_pinned_.as<int32_t>()[i];
        } else {
            // Fallback: per-sequence synchronous sampling (no slot buffers, or
            // top_k in the CUB regime).
            for (int i = 0; i < n_seq; i++) {
                Tensor seq_logits = flatten_logits(logits.slice(i, i + 1));
                apply_pre_sample(seq_logits, state);
                tokens[i] =
                    greedy ? (d_sample_result_ ? sample_greedy(seq_logits, d_sample_result_, stream)
                                               : sample_greedy(seq_logits, stream))
                           : (d_sample_result_
                                  ? sample_topk_topp(seq_logits, top_k, top_p, state.temperature,
                                                     state.seed >= 0
                                                         ? static_cast<unsigned int>(state.seed + i)
                                                         : (42u + i),
                                                     d_sample_result_, stream)
                                  : sample_topk_topp(seq_logits, top_k, top_p, state.temperature,
                                                     state.seed >= 0
                                                         ? static_cast<unsigned int>(state.seed + i)
                                                         : (42u + i),
                                                     stream));
            }
        }
    }

    return tokens;
}

// Device copy of the engine-static banned-token list (same host array every
// step): uploaded once per list identity, then served from the cache for
// every row of every step. nullptr when the list is empty or the device copy
// could not be taken.
const int32_t* GraphExecutor::banned_cache_(const InferenceState& state, cudaStream_t stream) {
    if (state.banned_tokens == nullptr || state.n_banned_tokens <= 0)
        return nullptr;
    const bool cache_hit = d_banned_cache_ != nullptr && banned_cache_src_ == state.banned_tokens &&
                           banned_cache_n_ == state.n_banned_tokens;
    if (!cache_hit) {
        size_t ban_bytes = static_cast<size_t>(state.n_banned_tokens) * sizeof(int32_t);
        if (banned_cache_capacity_ < static_cast<size_t>(state.n_banned_tokens)) {
            if (d_banned_cache_)
                IMP_CUDA_CHECK_LOG(cudaFree(d_banned_cache_));
            d_banned_cache_ = nullptr;
            if (cudaMalloc(&d_banned_cache_, ban_bytes) != cudaSuccess) {
                d_banned_cache_ = nullptr;
                banned_cache_capacity_ = 0;
            }
            banned_cache_capacity_ = d_banned_cache_ ? state.n_banned_tokens : 0;
        }
        if (d_banned_cache_) {
            cudaMemcpyAsync(d_banned_cache_, state.banned_tokens, ban_bytes, cudaMemcpyHostToDevice,
                            stream);
            banned_cache_src_ = state.banned_tokens;
            banned_cache_n_ = state.n_banned_tokens;
        }
    }
    return d_banned_cache_;
}

// Shared per-row logits filter chain (penalties, DRY, token bans, logit
// bias, forced token, schema/json masks, min-p, typical-p) — used by both
// the synchronous sample_single_from_logits and the enqueue-only
// sample_single_from_logits_async (which declines the host-blocking
// logit-bias mode before calling this).
void GraphExecutor::apply_row_filters_(float* lp, int vocab, const InferenceState& state,
                                       cudaStream_t stream) {
    if (state.penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        const int32_t* pen_ptr = state.penalty_tokens;
        int pen_n = state.n_penalty_tokens;
        if (state.repeat_last_n > 0 && pen_n > state.repeat_last_n) {
            pen_ptr += (pen_n - state.repeat_last_n);
            pen_n = state.repeat_last_n;
        }
        apply_penalties(lp, vocab, pen_ptr, pen_n, state.repetition_penalty, state.frequency_penalty,
                        state.presence_penalty, stream);
    }
    if (state.dry_multiplier > 0.0f && state.host_penalty_tokens != nullptr && state.n_penalty_tokens > 0) {
        apply_dry_penalty(lp, vocab, state.host_penalty_tokens, state.n_penalty_tokens, state.dry_multiplier,
                          state.dry_base, state.dry_allowed_length, state.dry_penalty_last_n, stream);
    }
    // Ban special tokens. The list is engine-static (same host array every
    // step), so cache the device copy instead of re-allocating + re-uploading
    // it per row per step — at n=16 that was 16 cudaMallocAsync/H2D/FreeAsync
    // triplets per decode step for identical bytes.
    if (state.banned_tokens != nullptr && state.n_banned_tokens > 0) {
        if (const int32_t* d_ban = banned_cache_(state, stream))
            launch_ban_logits(lp, d_ban, state.n_banned_tokens, vocab, stream);
    }
    // Apply logit bias
    if (state.n_logit_bias > 0 && state.logit_bias != nullptr)
        apply_logit_bias(lp, vocab, state.logit_bias, state.n_logit_bias, stream);
    // Force token (think-budget)
    if (state.force_token >= 0 && state.force_token < vocab) {
        force_single_token(lp, vocab, state.force_token, stream);
    }
    if (state.force_token < 0) {
        apply_constraint_mask(state, lp, vocab, stream);
    }
    if (state.min_p > 0.0f)
        apply_min_p(lp, vocab, state.min_p, stream);
    if (state.typical_p > 0.0f && state.typical_p < 1.0f)
        apply_typical_p(lp, vocab, state.typical_p, stream);
}

int32_t GraphExecutor::sample_single_from_logits(const Tensor& logits, const InferenceState& state,
                                                 cudaStream_t stream) {
    // Flatten [1, V] to [V]
    int64_t vocab_shape[1] = {logits.shape[logits.ndim - 1]};
    Tensor flat = logits.slice(0, 1).reshape(1, vocab_shape);

    float* lp = static_cast<float*>(flat.data);
    int vocab = static_cast<int>(flat.shape[0]);
    apply_row_filters_(lp, vocab, state, stream);

    // Sample
    if (state.mirostat == 2) {
        unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
        return d_sample_result_
                   ? sample_mirostat_v2(flat, state.temperature, state.mirostat_tau, state.mirostat_eta,
                                        &state.mirostat_mu, seed, d_sample_result_, stream)
                   : sample_mirostat_v2(flat, state.temperature, state.mirostat_tau, state.mirostat_eta,
                                        &state.mirostat_mu, seed, stream);
    }
    return (state.temperature <= 0.0f || state.top_k == 1)
               ? (d_sample_result_ ? sample_greedy(flat, d_sample_result_, stream)
                                   : sample_greedy(flat, stream))
               : (d_sample_result_
                      ? sample_topk_topp(flat, state.top_k > 0 ? state.top_k : 50,
                                         state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                         state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                         d_sample_result_, stream)
                      : sample_topk_topp(flat, state.top_k > 0 ? state.top_k : 50,
                                         state.top_p > 0.0f ? state.top_p : 1.0f, state.temperature,
                                         state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                                         stream));
}

bool GraphExecutor::sample_single_from_logits_async(const Tensor& logits, const InferenceState& state,
                                                    int slot_idx, cudaStream_t stream) {
    // Enqueue-only per-row sampling for the batched decode loop: filters +
    // sampler land on the stream writing into scratch slot `slot_idx`; the
    // caller gathers ALL rows' tokens with collect_sampled_tokens (one pinned
    // D2H + one sync). The synchronous per-row readback cost ~850 us of
    // blocked host time per sequence per step at n=16 sustained serving
    // (pageable 4-byte D2H + stream sync each, nsys 2026-07-12).
    if (!d_sample_result_ || !h_sample_pinned_.as<int32_t>() || slot_idx < 0 || slot_idx >= sample_slots_)
        return false;
    // Parity offset: the pipelined decode enqueues into the half selected by
    // set_sample_parity while the other half's gather is still in flight.
    const int abs_slot = sample_parity_ * sample_slots_ + slot_idx;
    // Sync-only sampling modes decline BEFORE any filter is applied, so the
    // caller can re-run this row through sample_single_from_logits untouched:
    // mirostat mutates host-side mu every step; logit_bias does per-entry
    // host read-modify-write on the logits.
    if (state.mirostat == 2)
        return false;
    if (state.n_logit_bias > 0 && state.logit_bias != nullptr)
        return false;

    int64_t vocab_shape[1] = {logits.shape[logits.ndim - 1]};
    Tensor flat = logits.slice(0, 1).reshape(1, vocab_shape);
    float* lp = static_cast<float*>(flat.data);
    const int vocab = static_cast<int>(flat.shape[0]);

    const bool greedy = (state.temperature <= 0.0f || state.top_k == 1);
    const int top_k = state.top_k > 0 ? state.top_k : 50;
    const int eff_top_k = (top_k <= 0 || top_k > vocab) ? vocab : top_k;
    // Penalty stash: when the row's filter chain is EMPTY past the penalties
    // and the engine-static ban list (no DRY / bias / forced token /
    // constrainer / min-p / typical-p), the penalty launch is order-free
    // against every other stashed row and joins one batched vocab sweep at
    // flush; the ban rides in the same sweep (PenaltyRowArgs::banned - order
    // against the penalties is immaterial, -1e30 stays -1e30). Any active
    // later stage keeps the whole chain inline — the per-row order (penalties
    // first) must hold, and the flush runs after this call returns. Measured
    // reason (2026-08-31, 32-stream serving on Qwen3.8-27B-NVFP4, nsys
    // node-trace): the server's default repetition_penalty 1.05 plus the
    // 19 banned special tokens put every row on the inline chain - 2
    // launches per row per step with ~4 us gaps, ~0.45 ms of a 17 ms step.
    const bool ban_present = state.banned_tokens != nullptr && state.n_banned_tokens > 0;
    const int32_t* d_ban = ban_present ? banned_cache_(state, stream) : nullptr;
    const bool tail_empty =
        !(state.dry_multiplier > 0.0f && state.host_penalty_tokens != nullptr &&
          state.n_penalty_tokens > 0) &&
        (!ban_present || d_ban != nullptr) && state.force_token < 0 &&
        state.grammar_constrainer == nullptr && state.regex_constrainer == nullptr &&
        state.schema_constrainer == nullptr && state.json_constrainer == nullptr &&
        state.min_p <= 0.0f && !(state.typical_p > 0.0f && state.typical_p < 1.0f);
    const bool pen_active = state.penalty_tokens != nullptr && state.n_penalty_tokens > 0 &&
                            (state.repetition_penalty != 1.0f || state.frequency_penalty != 0.0f ||
                             state.presence_penalty != 0.0f);
    // The penalty stash is only sound when this row's SAMPLER is also batched
    // (flush order: penalties -> greedy -> top-k). A sampler that launches
    // immediately (top_k > SAMPLE_MAX_TOP_K) would read logits before the
    // stashed penalty sweep — keep that row's chain inline.
    const bool sampler_batches =
        greedy ? (d_sample_args_ != nullptr)
               : (d_sample_args_ != nullptr && eff_top_k <= SAMPLE_MAX_TOP_K);
    bool stashed_filters = false;
    if (tail_empty && sampler_batches && d_sample_args_ != nullptr) {
        if (pen_active || d_ban != nullptr) {
            const int32_t* pen_ptr = state.penalty_tokens;
            int pen_n = pen_active ? state.n_penalty_tokens : 0;
            if (pen_active && state.repeat_last_n > 0 && pen_n > state.repeat_last_n) {
                pen_ptr += (pen_n - state.repeat_last_n);
                pen_n = state.repeat_last_n;
            }
            PenaltyRowArgs& p = h_pen_rows_(sample_parity_)[n_pending_pen_rows_++];
            p.logits = lp;
            p.token_ids = pen_ptr;
            p.n_tokens = pen_n;
            p.repetition_penalty = state.repetition_penalty;
            p.frequency_penalty = state.frequency_penalty;
            p.presence_penalty = state.presence_penalty;
            p.banned = d_ban;
            p.n_banned = d_ban ? state.n_banned_tokens : 0;
        }
        stashed_filters = true;  // chain past penalties (+ban) is empty: nothing to enqueue
    }
    // No blanket CUB-regime refusal any more (#1654): sample_topk_topp_async
    // enqueues that regime now. Only the ROW-PARALLEL stash below is still
    // limited to SAMPLE_MAX_TOP_K - launch_topk_topp_rows takes top_k in
    // [1, SAMPLE_MAX_TOP_K] by contract - so a larger k skips the stash and
    // enqueues per row instead of dropping the caller onto a synchronous path.
    if (!stashed_filters)
        apply_row_filters_(lp, vocab, state, stream);
    pending_sample_vocab_ = vocab;

    auto* slot = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(d_sample_result_) +
                                            static_cast<size_t>(abs_slot) * SAMPLE_SCRATCH_BYTES);
    if (greedy) {
        // Greedy stash: the argmax always runs LAST for its row, so it is
        // batchable whether the filters were stashed or ran inline (single
        // stream, and the flush happens after every row is enqueued).
        if (d_sample_args_ != nullptr) {
            GreedyRowArgs& g = h_greedy_rows_(sample_parity_)[n_pending_greedy_rows_++];
            g.logits = lp;
            g.d_result = slot;
            return true;
        }
        sample_greedy_async(flat, slot, stream);
        return true;
    }
    unsigned int seed = state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u;
    float temperature = state.temperature <= 0.0f ? 1.0f : state.temperature;
    if (d_sample_args_ != nullptr && eff_top_k <= SAMPLE_MAX_TOP_K) {
        // STASH the row for the row-parallel batched launch in
        // collect_sampled_tokens — n serialized <<<64>>>+<<<1>>> launch pairs
        // become ONE partial + ONE finalize launch for the whole batch.
        TopkRowArgs& a = h_topk_rows_(sample_parity_)[n_pending_topk_rows_++];
        a.logits = lp;
        a.d_result = slot;
        a.inv_temperature = 1.0f / temperature;
        a.top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        a.seed = seed;
        a.top_k = eff_top_k;
        pending_topk_max_k_ = std::max(pending_topk_max_k_, eff_top_k);
        pending_topk_vocab_ = vocab;
        return true;
    }
    bool ok = sample_topk_topp_async(flat, top_k, state.top_p > 0.0f ? state.top_p : 1.0f,
                                     state.temperature, seed, slot, stream);
    (void)ok;  // eligibility pre-checked above
    return true;
}

// The three flushes below launch from the device slab; the ONE H2D that
// carries all three arrays' active-parity regions is flush_sample_args_()
// (must run first - and the penalty sweep before the greedy/top-k flushes,
// each row's penalties precede its sampler on the stream).
void GraphExecutor::flush_sample_args_(cudaStream_t stream) {
    const size_t used_pen = sizeof(PenaltyRowArgs) * n_pending_pen_rows_;
    const size_t used_greedy = sizeof(GreedyRowArgs) * n_pending_greedy_rows_;
    const size_t used_topk = sizeof(TopkRowArgs) * n_pending_topk_rows_;
    if (used_pen + used_greedy + used_topk == 0 || d_sample_args_ == nullptr)
        return;
    // One copy spanning from the parity block's start through the last used
    // array (the arrays are laid out pen | greedy | topk, so the span is
    // contiguous; unused gaps in between are dead bytes, cheaper than a
    // second launch).
    size_t end = used_pen;
    if (used_greedy)
        end = sample_args_off_greedy_ + used_greedy;
    if (used_topk)
        end = sample_args_off_topk_ + used_topk;
    const size_t off = static_cast<size_t>(sample_parity_) * sample_args_parity_bytes_;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_sample_args_ + off,
                                       static_cast<char*>(h_sample_args_.data()) + off, end,
                                       cudaMemcpyHostToDevice, stream));
}

void GraphExecutor::flush_pending_penalty_rows_(cudaStream_t stream) {
    if (n_pending_pen_rows_ <= 0)
        return;
    launch_penalties_rows(d_pen_rows_(sample_parity_), n_pending_pen_rows_, pending_sample_vocab_, stream);
    n_pending_pen_rows_ = 0;
}

// Flush the stashed greedy rows: one pinned H2D, one batched partial + one
// batched reduce covering every row (bit-identical per row to
// sample_greedy_async — same geometry, same slot scratch).
void GraphExecutor::flush_pending_greedy_rows_(cudaStream_t stream) {
    if (n_pending_greedy_rows_ <= 0)
        return;
    launch_greedy_rows(d_greedy_rows_(sample_parity_), n_pending_greedy_rows_, pending_sample_vocab_,
                       stream);
    n_pending_greedy_rows_ = 0;
}

// Flush the stashed top-k rows of the ACTIVE parity half: one pinned H2D of
// the args, one partial + one finalize launch covering every row.
void GraphExecutor::flush_pending_topk_rows_(cudaStream_t stream) {
    if (n_pending_topk_rows_ <= 0)
        return;
    launch_topk_topp_rows(d_topk_rows_(sample_parity_), n_pending_topk_rows_, pending_topk_max_k_,
                          pending_topk_vocab_, stream);
    n_pending_topk_rows_ = 0;
    pending_topk_max_k_ = 0;
}

bool GraphExecutor::append_sampled_history(const PenaltyAppendArgs& args, int32_t* d_hist,
                                           cudaStream_t stream) {
    if (!d_sample_result_ || args.n <= 0 || args.n > sample_slots_ || d_hist == nullptr)
        return false;
    const char* base = reinterpret_cast<const char*>(d_sample_result_) +
                       static_cast<size_t>(sample_parity_) * sample_slots_ * SAMPLE_SCRATCH_BYTES;
    penalty_hist_append(base, SAMPLE_SCRATCH_BYTES, args, d_hist, stream);
    return true;
}

const int32_t* GraphExecutor::collect_sampled_tokens(int n_slots, cudaStream_t stream) {
    if (!d_sample_result_ || !h_sample_pinned_.as<int32_t>() || n_slots <= 0 || n_slots > sample_slots_) {
        n_pending_topk_rows_ = 0;
        pending_topk_max_k_ = 0;
        n_pending_greedy_rows_ = 0;
        n_pending_pen_rows_ = 0;
        return nullptr;
    }
    flush_sample_args_(stream);
    flush_pending_penalty_rows_(stream);
    flush_pending_greedy_rows_(stream);
    flush_pending_topk_rows_(stream);
    // Slots are SAMPLE_SCRATCH_BYTES apart; the token is the first int32 of
    // each slot — one strided D2H gathers the whole batch.
    const size_t base = static_cast<size_t>(sample_parity_) * sample_slots_;
    IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(h_sample_pinned_.as<int32_t>() + base, sizeof(int32_t),
                                         reinterpret_cast<char*>(d_sample_result_) +
                                             base * SAMPLE_SCRATCH_BYTES,
                                         SAMPLE_SCRATCH_BYTES, sizeof(int32_t), n_slots,
                                         cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    return h_sample_pinned_.as<int32_t>() + base;
}

// Event-based split of collect_sampled_tokens for the pipelined decode:
// flush + strided D2H + event record, NO stream sync. The engine enqueues
// the NEXT step's work after this and only waits on the event — so the wait
// covers exactly this gather, not the freshly enqueued step.
bool GraphExecutor::gather_sampled_tokens_async(int n_slots, cudaStream_t stream) {
    if (!sample_pipeline_ready() || n_slots <= 0 || n_slots > sample_slots_) {
        n_pending_topk_rows_ = 0;
        pending_topk_max_k_ = 0;
        n_pending_greedy_rows_ = 0;
        n_pending_pen_rows_ = 0;
        return false;
    }
    flush_sample_args_(stream);
    flush_pending_penalty_rows_(stream);
    flush_pending_greedy_rows_(stream);
    flush_pending_topk_rows_(stream);
    const size_t base = static_cast<size_t>(sample_parity_) * sample_slots_;
    IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(h_sample_pinned_.as<int32_t>() + base, sizeof(int32_t),
                                         reinterpret_cast<char*>(d_sample_result_) +
                                             base * SAMPLE_SCRATCH_BYTES,
                                         SAMPLE_SCRATCH_BYTES, sizeof(int32_t), n_slots,
                                         cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaEventRecord(sample_gather_evt_[sample_parity_], stream));
    return true;
}

const int32_t* GraphExecutor::wait_gathered_tokens(int parity) {
    parity &= 1;
    if (!h_sample_pinned_.as<int32_t>() || !sample_gather_evt_[parity])
        return nullptr;
    IMP_CUDA_CHECK_LOG(cudaEventSynchronize(sample_gather_evt_[parity]));
    return h_sample_pinned_.as<int32_t>() + static_cast<size_t>(parity) * sample_slots_;
}

const int32_t* GraphExecutor::sample_slot_base(int parity) const {
    if (!d_sample_result_)
        return nullptr;
    return reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(d_sample_result_) +
                                            static_cast<size_t>(parity & 1) * sample_slots_ *
                                                SAMPLE_SCRATCH_BYTES);
}

}  // namespace imp
