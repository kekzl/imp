// =============================================================================
// engine_spec_ngram.cpp — n-gram (prompt-lookup) speculative decoding
// =============================================================================
//
// Drafts come from suffix matches against the request's own prompt+output
// tokens — no draft model, no MTP head. Two matchers: the suffix index
// (SuffixDraftIndex, speculative.suffix, default) with frequency-voted
// continuations and adaptive draft length, or the legacy single-most-recent
// backward scan (ngram_draft). The verify step replays
// [t0, d1..dK] as a teacher-forced continuation chunk through the standard
// chunked-prefill forward (KV written in place), applies the tier-aware LM
// head to every chunk position (executor greedy_argmax_all) and accepts the
// longest prefix where the model's greedy token equals the draft. Rejected
// draft KV entries are dropped via KVCacheManager::rollback — safe because
// draft blocks were appended this step and are never content-hashed (prefix
// hashing only covers prompt prefill blocks).
//
// Phase 1 scope (gated in spec_verify_gates_ok_): batch-1, greedy sampling,
// no penalties / logit_bias / DRY / mirostat, no logprobs, no json/schema
// constraints, no think budget, chunked-prefill-capable archs only. The
// verify loop runs eager — the async conditional graph loop stays off while
// speculation is enabled (the host must see every token to draft the next
// step).
//
// Hybrid (SSM/GDN) models (speculative.hybrid): the chunk forward advances
// recurrent state through rejected draft positions, so the committed
// per-sequence state slab is copied to scratch before the chunk; a full
// acceptance keeps the advanced state as-is (it covers exactly the forwarded
// tokens), a partial acceptance restores the slab and re-forwards the
// accepted prefix (~one extra chunk forward, amortized over the accepted
// tokens). Draft sources: the suffix/ngram matcher first; when it has no
// match and an MTP head is enabled (--mtp-spec-decode / speculative.mtp_k),
// the pending MTP chain fills the chunk (engine_spec_mtp.cpp).
// =============================================================================

#include "compute/json_constrain.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/engine.h"
#include "runtime/spec_trace.h"
#include "runtime/ngram_draft.h"
#include "runtime/request.h"
#include "runtime/suffix_draft.h"
#include "runtime/think_stop_logic.h"
#include "compute/mtp_forward.h"
#include "compute/rowwise_topm.h"

#include <cuda_runtime.h>

#include <span>
#include <algorithm>
#include <chrono>
#include <vector>

namespace imp {


void Engine::log_spec_stats_() const {
    if (spec_stats_.verify_steps + spec_stats_.miss_steps == 0)
        return;
    IMP_LOG_INFO("[spec-ngram] verify_steps=%lld miss_steps=%lld drafted=%lld accepted=%lld "
                 "(%.1f%%) emitted=%lld (%.2f tok/verify, %.2f ms/verify)",
                 spec_stats_.verify_steps, spec_stats_.miss_steps, spec_stats_.drafted,
                 spec_stats_.accepted,
                 spec_stats_.drafted ? 100.0 * spec_stats_.accepted / spec_stats_.drafted : 0.0,
                 spec_stats_.emitted,
                 spec_stats_.verify_steps
                     ? static_cast<double>(spec_stats_.emitted) / spec_stats_.verify_steps
                     : 0.0,
                 spec_stats_.verify_steps ? spec_stats_.verify_wall_ms / spec_stats_.verify_steps
                                          : 0.0);
}




// Burst-hybrid re-arm: a given-up request whose async-loop burst
// (speculative.burst tokens) has completed gets a short probe window — two
// draft attempts and a fresh acceptance sample. Think models produce their
// draft-rich region only after the reasoning prose; a sticky give-up would
// lock them out exactly there.
void Engine::spec_maybe_rearm_(Request& req) const {
    const auto& scfg = runtime_config_.speculative;
    if (!req.spec_ngram_given_up || req.spec_acceptance_doomed || scfg.burst <= 0 ||
        scfg.give_up_after <= 0)
        return;
    if (static_cast<int>(req.output_tokens.size()) - req.spec_last_giveup_pos < scfg.burst)
        return;
    req.spec_ngram_given_up = false;
    req.spec_consecutive_misses = std::max(0, scfg.give_up_after - 2);
    req.spec_verifies = 0;
    req.spec_drafted = 0;
    req.spec_accepted = 0;
}

// Adaptive miss burst: consecutive draft misses double the burst length
// (boundary overhead amortizes on draft-poor stretches), a hit resets it.
// Capped at speculative.burst so re-probing never stops entirely.
int Engine::spec_effective_miss_burst_(const Request& req) const {
    const auto& scfg = runtime_config_.speculative;
    int burst = scfg.miss_burst;
    if (burst <= 0)
        return 0;
    // Cap growth at 4x: longer bursts overshoot draft-rich regions right
    // after a transition (think prose -> code), losing more speculation
    // upside than the saved boundary overhead is worth.
    const int doublings = std::min(2, req.spec_consecutive_misses / 8);
    burst <<= doublings;
    const int cap = scfg.burst > 0 ? scfg.burst : 128;
    return std::min(burst, cap);
}

// Whether a bounded async-loop burst may be launched directly from the spec
// hook (mirrors the launch conditions in step_decode_process_outputs).
bool Engine::spec_burst_launch_ok_(const Request& req) const {
    // IMP_SPEC_TRACE: dump every term — the launch decision decides WHICH
    // kernel mix (loop vs pooled/eager) serves the next tokens, so an
    // asymmetric term here shows up as a greedy flip between requests.
    if (runtime_config_.diagnostics.spec_trace) {
        IMP_LOG_INFO("[burst-ok?] req=%d out=%zu pool_avail=%d offload=%d graphs=%d setup=%d parked=%d "
                     "think=%d think_end=%d status=%d",
                     req.id, req.output_tokens.size(),
                     (int)decode_graph_pool_[0].graph_path_available(), (int)(offload_mgr_ != nullptr),
                     (int)config_.use_cuda_graphs, (int)async_graph_runner_.is_setup(),
                     (int)async_parked_req_id_, (int)req.in_think_block, (int)think_end_id_,
                     (int)req.status);
    }
    if (!decode_graph_pool_[0].graph_path_available() || offload_mgr_ || !config_.use_cuda_graphs)
        return false;
    // A runner that is setup but NOT parked is in flight — blocked. Parked
    // for a DIFFERENT request is fine: try_launch_async_graph_loop tears the
    // stale park down and rebuilds (a parked warmup request would otherwise
    // lock every later server request out of the loop entirely).
    if (async_graph_runner_.is_setup() && async_parked_req_id_ < 0)
        return false;
    // Text-fallback think tracking needs host-side per-token matching.
    if (req.in_think_block && think_end_id_ < 0)
        return false;
    if (req.output_tokens.empty() || req.status != RequestStatus::DECODING)
        return false;
    return true;
}

// Which gate refuses this request, or nullptr when none does.
//
// Split out of spec_verify_gates_ok_ because the one-time diagnosis in
// engine_scheduler.cpp printed eighteen request fields and named none of them
// as the cause, so "why is speculation off" was a manual re-derivation of this
// function against a log line. #1538 and #1539 were both filed against that
// line; answering either took reading this file. The strings are the gate
// names, stable enough to grep for.
const char* Engine::spec_verify_gate_refusal_(const Request& req, bool ignore_think) const {
    if (req.spec_ngram_given_up) return "given_up";
    // Greedy sampling only: verify compares argmax tokens.
    const bool greedy = (req.temperature <= 0.0f || req.top_k == 1);
    if (!greedy) return "sampling_not_greedy";
    // rep/freq/presence penalties are replicated in the verify
    // (greedy_argmax_all) for the unbounded window; a bounded repeat_last_n
    // window slides per chunk row and is not replicated — stay eager there.
    const bool penalties = req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                           req.presence_penalty != 0.0f;
    if (penalties && req.repeat_last_n != 0) return "bounded_repeat_window";
    // Logit-shaping the verify chunk does not replicate disqualifies.
    if (req.dry_multiplier != 0.0f || req.mirostat != 0 || !req.logit_bias.empty())
        return "logit_shaping";
    if (req.logprobs || req.json_mode || !req.json_schema.empty() ||
        !req.regex_pattern.empty() || !req.grammar.empty() || !req.tool_constraint_tools.empty())
        return "constrained_decode";  // verify replicates no FSM masks (#1002)
    // Think budget forces tokens INSIDE the think block (loop/host-side) —
    // verify only outside it; the think interior runs loop bursts, which
    // handle the budget device-side. Exception: an MTP-bound request already
    // runs the interior eager (a loop burst would desync the MTP cache for
    // the rest of the generation, #847), and the eager step owns the budget
    // forcing - verifies between forcing points are legal and overshoot the
    // budget by at most k tokens per chunk. Without this the interior pays
    // the eager tax with no speculation, which on think-heavy chat eats the
    // MTP win.
    if (!ignore_think && req.think_budget > 0.0f && req.in_think_block) {
        if (!(mtp_spec_decode_enabled() && mtp_bound_req_ == req.id))
            return "think_budget_in_block";
        // Forcing due: the eager step must run NOW to force </think>; a
        // verify would emit past the exhausted budget instead.
        if (think_logic::should_force_think_end(req.think_budget, think_end_id_, req.max_tokens,
                                                req.output_tokens, think_start_id_,
                                                req.started_in_think))
            return "think_budget_in_block";
    }
    if (req.status != RequestStatus::DECODING || req.output_tokens.empty()) return "not_decoding";
    if (spec_history_too_short_(req)) return "cold_start";  // see speculative.min_history
    // Long-context economics on the DENSE path (#964): the captured chunk
    // verify runs the FA2 tile + paged-KV gather over the ctx TIER (pow2,
    // floor 4096, clamped to max_seq_len) — its cost follows the tier, not
    // the live context. Measured 2026-07-11 (Qwen3-8B Q8_0): a verify step
    // costs ~2.1× a decode step at 2k ctx and ~5.2× at 16k, so with dense
    // ngram's ~2 tok/verify payout speculation turns net-negative past ~2k
    // (−62% at 16k). Gate drafting once the request's context crosses the
    // cap, checked per step. MoE-NVFP4 and GDN-hybrid requests are exempt:
    // their drafts run much deeper (Coder-30B code-edit 15.9 tok/verify,
    // MTP chains), which pays for the verify at any measured context.
    {
        const int cap = runtime_config_.speculative.draft_ctx_cap;
        const bool moe_nvfp4_path = model_->profile().is_moe &&
                                    runtime_config_.speculative.moe &&
                                    model_->profile().moe_experts_nvfp4;
        const bool hybrid_path = ssm_state_ != nullptr;  // only reachable with speculative.hybrid
        if (!moe_nvfp4_path && !hybrid_path && cap > 0 && req.context_len() > cap)
            return "draft_ctx_cap";
    }
    // Recurrent state (SSM/GDN) advances on every forwarded token. The
    // hybrid verify path (speculative.hybrid) rides SSMState's contiguous
    // per-sequence slab for snapshot/restore.
    if (ssm_state_ && !runtime_config_.speculative.hybrid) return "recurrent_without_hybrid";
    // MoE speculation engages only for native-NVFP4 experts: the batched
    // verify forward reads the NVFP4 expert cache directly and nets +49-81%
    // on draft-rich code-edit (Qwen3-Coder-30B-FP4, 2026-07-02) with a -3-7%
    // draft-poor floor (miss_burst hybrid). GGUF-MoE verify re-dequants every
    // activated expert per step and measured -22% — those stay on the async
    // conditional-graph loop (as does everything when speculative.moe=false).
    if (!spec_ngram_model_capable_())
        return "model_not_capable";
    return nullptr;
}

bool Engine::spec_verify_gates_ok_(const Request& req, bool ignore_think) const {
    return spec_verify_gate_refusal_(req, ignore_think) == nullptr;
}

// Model-level gates, split out of spec_verify_gates_ok_ so spec_ngram_enabled_
// can ask the same question without the per-request checks. Nothing here
// depends on a request, so a false answer means "this model never speculates,
// whatever the flag says".
bool Engine::spec_ngram_model_capable_uncached_() const {
    if (ssm_state_ && !runtime_config_.speculative.hybrid)
        return false;
    // GGUF-MoE verify re-dequants every activated expert per step (-22%), so
    // these stay on the async conditional-graph loop — as does everything when
    // speculative.moe=false.
    if (model_->profile().is_moe &&
        !(runtime_config_.speculative.moe && model_->profile().moe_experts_nvfp4))
        return false;
    if (!supports_chunked_prefill_())
        return false;
    return true;
}

SuffixDraftIndex& Engine::spec_suffix_index_(const Request& req) {
    auto it = spec_suffix_idx_.find(req.id);
    if (it == spec_suffix_idx_.end()) {
        const auto& scfg = runtime_config_.speculative;
        it = spec_suffix_idx_
                 .emplace(req.id, SuffixDraftIndex(std::max(1, scfg.min_match), scfg.max_match))
                 .first;
        it->second.append(req.input_tokens);
        it->second.append(req.prediction_tokens);
    }
    return it->second;
}

TokenRecycleTable& Engine::spec_recycle_table_() {
    if (!spec_recycle_) {
        const auto& scfg = runtime_config_.speculative;
        spec_recycle_ = std::make_unique<TokenRecycleTable>(model_->config_.vocab_size,
                                                            std::max(1, scfg.recycle_slots));
    }
    return *spec_recycle_;
}

// Ingest this request's not-yet-seen tokens into the engine-scoped
// adjacency table: prompt bigrams once (spec_recycle_fed == 0), then the
// new output tokens (every emit path — verify accepts, loop bursts,
// plain steps — lands in output_tokens, so one cursor covers them all).
void Engine::spec_recycle_feed_(Request& req) {
    TokenRecycleTable& tr = spec_recycle_table_();
    const auto& out = req.output_tokens;
    if (req.spec_recycle_fed == 0) {
        const auto& in = req.input_tokens;
        for (size_t i = 1; i < in.size(); ++i)
            tr.observe_pair(in[i - 1], in[i]);
        if (!in.empty() && !out.empty())
            tr.observe_pair(in.back(), out.front());
    }
    for (int i = std::max(1, req.spec_recycle_fed); i < static_cast<int>(out.size()); ++i)
        tr.observe_pair(out[i - 1], out[i]);
    req.spec_recycle_fed = static_cast<int>(out.size());
}

bool Engine::step_spec_verify_(std::shared_ptr<Request>& req, cudaStream_t stream, int min_draft) {
    const auto& scfg = runtime_config_.speculative;
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // Build the draft from the request's own token history. Predicted-output
    // tokens (OpenAI `prediction`, never forwarded through the model) sit
    // between prompt and output so the current output suffix stays the tail:
    // a completion that tracks the prediction finds max_match-length matches
    // in the prediction region every verify step.
    const int pred_begin = static_cast<int>(req->input_tokens.size());
    const int pred_end = pred_begin + static_cast<int>(req->prediction_tokens.size());

    const int k = std::max(1, scfg.k);
    // One object, two names: the draft and the history index it came from are
    // produced together and were two locals only because the old signature
    // wrote the index through a pointer.
    NgramDraft nd;
    auto& [draft, draft_start] = nd;
    // The history matcher is the n-gram source, so it answers to the n-gram
    // flag alone. The step itself is entered whenever ANY drafter is enabled
    // (spec_any_drafter_enabled_), which is what lets MTP and token recycling
    // reach the verify with `speculative.ngram=false` — but they must not drag
    // the matcher in with them, or turning n-gram off would stop meaning
    // anything.
    const bool ngram_source_on = spec_ngram_enabled_(*req);
    if (!ngram_source_on) {
        // fall through to the MTP / recycling sources below
    } else if (scfg.suffix) {
        // Suffix index: input ++ prediction indexed at first use, output
        // tokens appended incrementally (every emit path lands in
        // output_tokens, so loop-burst tokens are picked up here too).
        SuffixDraftIndex& idx = spec_suffix_index_(*req);
        const int out_indexed = std::clamp(idx.size() - pred_end, 0,
                                           static_cast<int>(req->output_tokens.size()));
        idx.append(std::span(req->output_tokens).subspan(static_cast<size_t>(out_indexed)));
        nd = idx.draft(k, std::max(k, scfg.suffix_k_max));
    } else {
        std::vector<int32_t> history;
        history.reserve(req->input_tokens.size() + req->prediction_tokens.size() +
                        req->output_tokens.size());
        history.insert(history.end(), req->input_tokens.begin(), req->input_tokens.end());
        history.insert(history.end(), req->prediction_tokens.begin(),
                       req->prediction_tokens.end());
        history.insert(history.end(), req->output_tokens.begin(), req->output_tokens.end());
        nd = ngram_draft(history, k, std::max(1, scfg.min_match), scfg.max_match);
    }
    const bool draft_from_prediction = draft_start >= pred_begin && draft_start < pred_end;
    // #964 stage 2 — depth-aware long-context gate: a verify step costs
    // ~1.4x a decode step at 512 ctx rising to ~2.6x at 16k, so a 1-token
    // draft (2 emitted tokens) stops paying past ~14k while depth >= 2
    // keeps winning. Discard shallow drafts at long context and let the
    // miss/burst path serve the step at plain decode speed. MoE-NVFP4 and
    // hybrids are exempt (deep drafts pay for the verify at any context).
    if (!draft.empty() && static_cast<int>(draft.size()) < 2 &&
        scfg.shallow_draft_ctx > 0 && req->context_len() > scfg.shallow_draft_ctx &&
        ssm_state_ == nullptr &&
        !(model_->profile().is_moe && scfg.moe && model_->profile().moe_experts_nvfp4)) {
        nd = {};
    }
    // MTP fallback: when the matcher has no draft, the pending MTP chain
    // (drafted at the end of the previous verify step / prefill tail) fills
    // the chunk — the trained head drafts where suffix matching cannot
    // (78-94% depth-1 accept on Qwen3.6, PR #804). Subject to the economics
    // guard below: high accept alone does not pay for the eager chunk +
    // chain lm_head GEMVs + hybrid replay.
    std::vector<std::vector<int32_t>> mc;  // multi-candidate rows (route a)
    int mc_depth = 0;
    bool draft_from_mtp = false;
    // diagnostics.mtp_tree_probe is measurement-only: the chain is scored
    // against the eager decode path (engine_scheduler.cpp), so it must never
    // be consumed as a verify draft - a verified chain leaves no eager step
    // to score it on (measured 2026-08-31: 262 verifies, 0 eager steps, an
    // empty table).
    if (draft.empty() && mtp_spec_decode_enabled() && !runtime_config_.diagnostics.mtp_tree_probe) {
        // Multi-candidate MTP (speculative.mtp_tree_width > 1): the head's W
        // chains verify as one grouped chunk — the branch at position 1
        // hedges the chain's weakest link. v1 shares the TR route's
        // preconditions (the hybrid one falls in Stage 3 of
        // docs/plans/2026-08-31-mtp-multicandidate-hybrid.md, which is where
        // every MTP-bearing checkpoint actually lives); the old blanket
        // "!mtp_spec_decode_enabled()" mc exclusion was the row-consumer
        // offset, fixed by the winner harvest (mtp_post_verify_update_ row0).
        const bool mtp_pen = req->repetition_penalty != 1.0f || req->frequency_penalty != 0.0f ||
                             req->presence_penalty != 0.0f;
        // Hybrids take the route through the grouped recurrent chunk (Stage
        // 3): W candidates as W scan sequences on W slots, see the hybrid
        // block below and spec_mc_hybrid_ok_ for what it needs.
        const int mtp_w = runtime_config_.speculative.mtp_tree_width;
        const bool mtp_mc_ok = scfg.verify_decode_attn &&
                               (ssm_state_ == nullptr || spec_mc_hybrid_ok_(mtp_w)) &&
                               !model_->profile().is_moe && !model_->config().is_mla() &&
                               !swa_sizing_active_ && !mtp_pen;
        if (mtp_w > 1 && mtp_mc_ok) {
            auto chains = mtp_take_chains_(*req);
            if (chains.size() > 1) {
                mc = std::move(chains);
                for (const auto& c : mc)
                    mc_depth = std::max(mc_depth, static_cast<int>(c.size()));
                draft_from_mtp = true;
            }
        }
        if (mc.empty()) {
            draft = mtp_take_draft_(*req);
            draft_from_mtp = !draft.empty();
        }
    }
    // Token-Recycling fallback: adjacency draft from the last emitted token.
    // Fires on unigram context — exactly the fresh reasoning/agentic prose
    // where the suffix/n-gram sources measured 0 drafts (2026-07-22/23).
    // Preferred shape is the multi-candidate chunk (route (a), `mc` below):
    // `recycle_width` candidates verified at once lift the per-step accept
    // where a single linear chain measured below the verify break-even
    // (1.55-1.9 emitted/verify vs ~1.9x step cost, 2026-07-23). Falls back
    // to the linear chain when the decode-attn route (or its gates) is
    // unavailable. Same shallow-depth economics as above for the linear
    // form: at long context a depth-1 chain does not pay for the verify.
    if (runtime_config_.speculative.token_recycling && mc.empty()) {
        spec_recycle_feed_(*req);
        const bool penalties_active = req->repetition_penalty != 1.0f ||
                                      req->frequency_penalty != 0.0f ||
                                      req->presence_penalty != 0.0f;
        // Route (a) preconditions: per-row block tables exist only on the
        // decode-attn verify route; the grouped rows are incompatible with
        // the linear-draft penalty replication and the MTP row consumer.
        const bool mc_route_ok = scfg.verify_decode_attn && ssm_state_ == nullptr &&
                                 !model_->profile().is_moe && !model_->config().is_mla() &&
                                 !swa_sizing_active_ && !penalties_active &&
                                 !mtp_spec_decode_enabled();
        if (draft.empty()) {
            const int width = std::min(8, std::max(1, scfg.recycle_width));
            // Row budget: width * (1 + depth) <= 17 (the mid capture bucket)
            // keeps the per-row context walks at linear-chunk cost.
            const int depth_cap = std::max(1, 17 / std::max(2, width) - 1);
            const int depth = std::min({std::max(1, scfg.recycle_depth), depth_cap, k});
            if (mc_route_ok && width > 1) {
                auto cands = spec_recycle_table_().draft_candidates(
                    req->output_tokens.back(), width, depth);
                if (cands.size() > 1) {
                    mc = std::move(cands);
                    mc_depth = 0;
                    for (const auto& c : mc)
                        mc_depth = std::max(mc_depth, static_cast<int>(c.size()));
                } else if (!cands.empty()) {
                    draft = std::move(cands.front());
                }
            } else {
                draft = spec_recycle_table_().draft_linear(
                    req->output_tokens.back(), depth,
                    std::max(0, scfg.recycle_min_streak));
                if (static_cast<int>(draft.size()) < 2 && scfg.shallow_draft_ctx > 0 &&
                    req->context_len() > scfg.shallow_draft_ctx && ssm_state_ == nullptr &&
                    !(model_->profile().is_moe && scfg.moe &&
                      model_->profile().moe_experts_nvfp4))
                    draft.clear();
            }
        }
    }
    const bool mc_on = !mc.empty();
    // #1003 batch-RR economics: at batch > 1 the WHOLE batch waits for the
    // verify forward (~1.4-2.6x a decode step) while only this request
    // benefits — a shallow draft is net-negative (measured -10% aggregate at
    // batch 4 on shallow drafts). Soft-decline below the caller's depth
    // floor: no miss accounting, no give-up pressure — the request decodes
    // batched this step and gets its next turn with a hopefully deeper draft.
    if (min_draft > 0 && (mc_on ? mc_depth : static_cast<int>(draft.size())) < min_draft)
        return false;
    if (draft.empty() && !mc_on) {
        spec_stats_.miss_steps++;
        if (++req->spec_consecutive_misses >= scfg.give_up_after && scfg.give_up_after > 0 &&
            !req->spec_ngram_given_up) {
            req->spec_ngram_given_up = true;
            req->spec_last_giveup_pos = static_cast<int>(req->output_tokens.size());
            IMP_LOG_INFO("spec-ngram: req %d gave up after %d consecutive draft misses — "
                         "re-enabling async graph loop",
                         req->id, req->spec_consecutive_misses);
        }
        // Skip the eager probe step entirely when a bounded loop burst can
        // take over right away — the eager path costs ~2x per token and the
        // burst forwards output.back() itself. Not while MTP is bound to this
        // request: a stale chain resyncs on the very next eager step (the
        // per-step chain feed), while a burst desyncs the MTP cache for the
        // rest of the generation (#847 sync gate).
        if (scfg.miss_burst > 0 && !req->spec_ngram_given_up &&
            !(mtp_spec_decode_enabled() && mtp_bound_req_ == req->id) &&
            spec_burst_launch_ok_(*req) &&
            try_launch_async_graph_loop(req, req->output_tokens.back(), stream,
                                        spec_effective_miss_burst_(*req))) {
            return true;  // step handled by the burst launch
        }
        return false;  // no usable draft — normal decode step
    }

    // Verify chunk. Linear: [t0, d1..dK], positions p0..p0+K. Multi-candidate
    // (route a): mc.size() candidate groups of (1 + mc_depth) rows each —
    // every candidate re-forwards t0 itself (its KV lands in the candidate's
    // PRIVATE block copy, see the mc staging below), so no row is shared and
    // no token-level mask is needed. K doubles as the stats/economics depth
    // (winner-depth proxy in mc mode — the verify cost is ~flat in rows).
    const auto verify_t0 = std::chrono::steady_clock::now();
    const int32_t t0 = req->output_tokens.back();
    const int mc_rows_per_cand = mc_on ? (1 + mc_depth) : 0;
    int K = mc_on ? mc_depth : static_cast<int>(draft.size());
    const int p0 = req->context_len() - 1;

    // attn_scores_ capacity clamp (same rule as chunked prefill):
    // n_tokens × ctx_len must fit s_cap².
    if (executor_) {
        const int s_cap = executor_->attn_scores_cap();
        if (s_cap > 0) {
            const int64_t cap2 = static_cast<int64_t>(s_cap) * s_cap;
            const int64_t ctx_end = static_cast<int64_t>(p0) + 1 + K;
            if (mc_on) {
                // The grouped chunk doesn't shrink gracefully — decline the
                // step instead (only reachable at extreme context).
                const int64_t rows = static_cast<int64_t>(mc.size()) * mc_rows_per_cand;
                if (rows * ctx_end > cap2) {
                    spec_stats_.miss_steps++;
                    return false;
                }
            } else {
                while (K > 0 && static_cast<int64_t>(K + 1) * ctx_end > cap2) --K;
                if (K <= 0) {
                    spec_stats_.miss_steps++;
                    return false;
                }
            }
        }
    }
    const int chunk_len =
        mc_on ? static_cast<int>(mc.size()) * mc_rows_per_cand : K + 1;
    // Graph-captured verify (#847): pad the chunk up to its bucket length so
    // one cached graph serves every draft length in the bucket. Pad rows
    // (copies of t0 at positions after every real row) are causally invisible
    // to the real rows; their KV entries fall to the same rollback that drops
    // rejected drafts, and the argmax window below stays [0, chunk_len).
    // SWA-aware sizing: the verify chunk stays EAGER (captured verify bakes
    // full-context gather grids + a static block-table pointer; the SWA table
    // rewrites every step). Correctness still requires the SWA table below.
    // #964: dense verify chunks route their attention through the batched-
    // decode split-K paged kernels (see the route block below). Composes with
    // capture: the decode kernels pay per-ROW KV traffic, but the per-row
    // context lens are data — pad rows get ctx_len=1, so the capture bucket
    // padding (2 real rows -> 9) costs ~nothing in attention while the graph
    // still swallows the ~200 eager launches per verify step. MoE/hybrid keep
    // the FA2 chunk path (deep drafts amortize it; the hybrid scan needs the
    // chunk-forward semantics).
    // The multi-candidate chunk on a hybrid is the one hybrid case that takes
    // the decode-attention route: its recurrent layers run the grouped
    // geometry (W scan sequences), and its attention layers need the per-row
    // tables exactly like the dense mc chunk. mc_on implies the mc gate
    // above passed (spec_mc_hybrid_ok_ when ssm_state_ != nullptr).
    const bool hybrid_mc = mc_on && ssm_state_ != nullptr;
    const bool decode_attn_route = runtime_config_.speculative.verify_decode_attn &&
                                   (ssm_state_ == nullptr || hybrid_mc) && !model_->profile().is_moe &&
                                   !model_->config().is_mla() && !swa_sizing_active_;
    const bool capture_on =
        !swa_sizing_active_ && spec_capture_ready_(p0 + spec_capture_bucket_(chunk_len));
    const int chunk_pad = capture_on ? spec_capture_bucket_(chunk_len) : chunk_len;
    const int ctx_len = p0 + chunk_pad;  // context including the full (padded) chunk

    const int blocks_needed = (ctx_len + kv_bs - 1) / kv_bs;

    // mc: per-candidate PRIVATE blocks for the block indices the candidate
    // rows write (positions p0..p0+mc_depth). They are ordinary appended
    // table entries beyond blocks_needed — the per-row tables below alias
    // them in place of the canonical entries, and the post-accept rollback
    // frees them like any other rejected-draft block.
    const int mc_bp = mc_on ? p0 / kv_bs : 0;
    const int mc_n_priv = mc_on ? (p0 + mc_depth) / kv_bs - mc_bp + 1 : 0;
    const int mc_priv_base = blocks_needed;
    const int blocks_target =
        blocks_needed + (mc_on ? static_cast<int>(mc.size()) * mc_n_priv : 0);

    // KV blocks for all chunk positions (mirror step_decode's append loop).
    // On allocation failure, trim back to the pre-step valid length (p0 KV
    // entries: t0 has not been forwarded yet) and fall through to the normal
    // decode path.
    while (static_cast<int>(kv_manager_->block_table(req->id).size()) < blocks_target) {
        int new_block = kv_manager_->append_block(req->id);
        if (new_block < 0) {
            // KV exhausted. The old evict_lru fallback freed a LIVE sequence (no
            // recompute path) → silent corruption. Just roll back the
            // speculative growth and fall through to the normal decode path.
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
    }
    // SWA-aware sizing: keep the trailing window live across the whole
    // chunk's write+read span [p0, ctx_len). Runs after the global append
    // loop above so the SWA table can pad to the grown global length.
    if (swa_sizing_active_) {
        kv_manager_->swa_trim(req->id, p0);
        if (!kv_manager_->swa_prepare(req->id, p0, ctx_len)) {
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
    }

    const auto& block_table = kv_manager_->block_table(req->id);
    const int n_blocks = static_cast<int>(block_table.size());
    const auto& swa_block_table = kv_manager_->swa_block_table(req->id);

    // Staging capacity follows the REAL table size — the async graph loop
    // pre-allocates blocks for the whole remaining generation, so the table
    // is usually much larger than this chunk needs. Captured graphs bake the
    // staging pointers, so size for the largest bucket up front (a later
    // realloc would silently invalidate every cached graph).
    const int chunk_cap =
        std::max({chunk_pad, scfg.k + 1, capture_on ? spec_capture_bucket_max_() : 0});
    const int table_cap = capture_on
                              ? std::max(n_blocks + 16,
                                         (spec_capture_ctx_cap_ + kv_bs - 1) / kv_bs + 16)
                              : n_blocks + 16;
    if (!ensure_spec_buffers_(chunk_cap, table_cap)) {
        spec_stats_.miss_steps++;
        return false;
    }

    // Stage chunk metadata into the pinned block and upload with ONE H2D
    // (#1055): [tokens | positions | row_ctx_lens | ctx_len, past_len,
    // chunk_len]. mc: candidate c's rows are [t0, cand_c...] at positions
    // p0..p0+len_c; short candidates and the bucket padding fill with t0
    // rows whose argmax is never consulted (their KV writes land in dead
    // slots — private-block tails past the accepted length, or canonical
    // slots past the rollback point). Row ctx lens: real rows attend
    // p0+i+1 (per-row causality), pads attend 1 token — consumed only on
    // the decode-attn route, harmless otherwise.
    const int cap = spec_chunk_cap_;
    int32_t* h_stage = h_spec_stage_.as<int32_t>();
    int32_t* h_tokens = h_stage;
    int32_t* h_positions = h_stage + cap;
    int32_t* h_row_lens = h_stage + 2ull * cap;
    // Pad rows sit at positions AFTER every real row in both shapes: the
    // rollback drops them. (mc pads used to sit at p0 on the canonical
    // table, which the winner's block copy had to repair - and the hybrid
    // mc partial-accept replay re-forwards the chunk AFTER that copy, so a
    // pad writing canonical p0 would clobber the kept row.)
    for (int i = 0; i < chunk_pad; ++i) {
        h_tokens[i] = t0;
        h_positions[i] = p0 + i;
        h_row_lens[i] = 1;
    }
    if (mc_on) {
        for (size_t c = 0; c < mc.size(); ++c) {
            const int r0 = static_cast<int>(c) * mc_rows_per_cand;
            for (int j = 0; j < mc_rows_per_cand; ++j) {
                h_positions[r0 + j] = p0 + j;
                const bool real = j == 0 || j - 1 < static_cast<int>(mc[c].size());
                h_row_lens[r0 + j] = real ? (p0 + j + 1) : 1;
                if (j > 0 && j - 1 < static_cast<int>(mc[c].size()))
                    h_tokens[r0 + j] = mc[c][j - 1];
            }
        }
    } else {
        for (int i = 0; i < K; ++i) h_tokens[1 + i] = draft[i];
        for (int i = 0; i < chunk_len; ++i) h_row_lens[i] = p0 + i + 1;
    }
    h_stage[3ull * cap] = ctx_len;
    h_stage[3ull * cap + 1] = p0;
    // Hybrid mc: the recurrent kernels read the per-GROUP real row count here
    // (InferenceState::d_chunk_len under ssm_grouped_chunk), not the whole
    // chunk's - every candidate commits its state at its own last row.
    h_stage[3ull * cap + 2] = hybrid_mc ? mc_rows_per_cand : chunk_len;
    h_stage[3ull * cap + 3] = 1;  // snapshot row (refreshed below on hybrids)
    // Per-candidate recurrent slots (hybrid mc): candidate 0 on the request's
    // live slot, the rest on the reserved pool slots. Staged with the chunk
    // so the one H2D carries them; the values are what a partial-accept
    // replay swaps.
    int rec_slot = -1;
    if (hybrid_mc) {
        rec_slot = recurrent_slot_for_(req->id);
        h_spec_mc_slots_.assign(static_cast<size_t>(kMtpMaxTopW), 0);
        h_spec_mc_slots_[0] = rec_slot;
        for (size_t c = 1; c < mc.size(); ++c)
            h_spec_mc_slots_[c] = ssm_state_->reserved_slot(static_cast<int>(c) - 1);
        std::copy(h_spec_mc_slots_.begin(), h_spec_mc_slots_.end(), h_stage + 3ull * cap + 4);
    }

    auto check = [&](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("spec-ngram: %s failed: %s", op, cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    if (!check(cudaMemcpyAsync(d_spec_stage_, h_spec_stage_.as<int32_t>(),
                               (3ull * cap + 4 + kMtpMaxTopW) * sizeof(int32_t), cudaMemcpyHostToDevice,
                               stream), "stage H2D") ||
        !check(cudaMemcpyAsync(d_spec_block_table_, block_table.data(), n_blocks * sizeof(int),
                               cudaMemcpyHostToDevice, stream), "block table H2D") ||
        (swa_sizing_active_ && !swa_block_table.empty() &&
         !check(cudaMemcpyAsync(d_spec_block_table_swa_, swa_block_table.data(),
                                static_cast<int>(swa_block_table.size()) * sizeof(int),
                                cudaMemcpyHostToDevice, stream), "swa block table H2D")))
        return false;

    // Forward the chunk through the standard continuation-prefill path.
    // Capture mode pins the workspace layout to the largest bucket so every
    // bucket's graph bakes the same activation-tensor carve.
    if (!executor_->resize_workspace(
            capture_on ? std::max(chunk_pad, spec_capture_bucket_max_()) : chunk_len, stream)) {
        spec_stats_.miss_steps++;
        return false;
    }
    if (executor_->has_decode_workspace()) executor_->use_workspace(0);

    InferenceState state;
    state.token_ids = d_spec_tokens_;
    state.positions = d_spec_positions_;
    state.n_tokens = chunk_pad;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_spec_block_table_;
    state.block_tables_swa =
        (swa_sizing_active_ && !swa_block_table.empty()) ? d_spec_block_table_swa_ : nullptr;
    state.context_lens = d_spec_context_len_;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = 0;
    state.is_prefill = true;
    state.prefill_offset = p0;
    state.spec_verify_chunk = true;
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) state.kv_seq_id = req->id;
    if (capture_on) {
        // The tier (not the full capacity) sizes the baked gather grids; the
        // executor scratch covers the full capacity, so any tier fits it.
        state.ctx_capacity = spec_capture_ctx_tier_(ctx_len);
        state.d_past_len = d_spec_past_len_;
        state.d_chunk_len = d_spec_chunk_len_;
    }

    // #964 decode-attention route: present the chunk rows as n same-KV
    // "sequences" with per-row context lens (p0+1+i) and row-replicated block
    // tables — causality holds by construction, and run_attention takes the
    // batched-decode split-K path (quantized-KV direct reads, context split
    // across CTAs) instead of the small-M prefill FA2 tile + full-context
    // FP16 KV gather (557 vs ~44 us/layer at 16k, nsys 2026-07-12).
    if (decode_attn_route) {
        // Row ctx lens were staged with the chunk metadata above (pads and
        // short-candidate tails attend 1 token: their output is never read,
        // and a 1-token walk keeps the per-row KV traffic at real-chunk
        // cost). Only the row-replicated tables remain — pinned staging,
        // one H2D of the used region.
        for (int i = 0; i < chunk_pad; ++i)
            std::copy(block_table.begin(), block_table.end(),
                      h_spec_row_tables_pinned_.as<int32_t>() +
                          static_cast<size_t>(i) * spec_block_table_cap_);
        // mc: alias each candidate's written block indices [mc_bp ..
        // mc_bp + mc_n_priv) to its private appended blocks — reads of the
        // committed prefix stay canonical, writes are candidate-isolated.
        if (mc_on) {
            for (int i = 0; i < chunk_len; ++i) {
                const int c = i / mc_rows_per_cand;
                int32_t* row = h_spec_row_tables_pinned_.as<int32_t>() +
                               static_cast<size_t>(i) * spec_block_table_cap_;
                for (int t = 0; t < mc_n_priv; ++t)
                    row[mc_bp + t] = block_table[mc_priv_base + c * mc_n_priv + t];
            }
        }
        if (check(cudaMemcpyAsync(d_spec_row_block_tables_, h_spec_row_tables_pinned_.as<int32_t>(),
                                  static_cast<size_t>(chunk_pad) * spec_block_table_cap_ *
                                      sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream),
                  "row block tables H2D")) {
            state.chunk_decode_attn = true;
            state.n_sequences = chunk_pad;
            state.context_lens = d_spec_row_ctx_lens_;
            state.block_tables = d_spec_row_block_tables_;
            state.max_blocks_per_seq = spec_block_table_cap_;
            // Captured graphs bake the split-K grid from max_context_len at
            // capture time — bake the tier ceiling so later replays in the
            // same tier stay covered (per-row device lens bound the real
            // work; #948/#950 pow2-bucket class).
            if (capture_on)
                state.max_context_len = spec_capture_ctx_tier_(ctx_len);
        }
        // On H2D failure the state keeps the plain single-sequence chunk
        // fields and the FA2 prefill path serves the forward (correct, slow).
    }
    if (mc_on) {
        // The grouped chunk is only correct with per-row tables engaged.
        if (!state.chunk_decode_attn) {
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
        // Seed each candidate's private partial block with the committed
        // rows of the canonical one (positions [mc_bp*kv_bs, p0) — t0 is
        // forwarded by the candidate itself). A block-aligned p0 has no
        // committed rows in mc_bp; the private block starts fresh.
        if (p0 % kv_bs != 0) {
            int srcs[KVCache::kCopyMaxPairs];
            int dsts[KVCache::kCopyMaxPairs];
            const int n_pairs =
                std::min(static_cast<int>(mc.size()), KVCache::kCopyMaxPairs);
            for (int c = 0; c < n_pairs; ++c) {
                srcs[c] = block_table[mc_bp];
                dsts[c] = block_table[mc_priv_base + c * mc_n_priv];
            }
            kv_cache_raw_->copy_blocks_device(srcs, dsts, n_pairs, stream);
        }
    }

    // Hybrid (SSM/GDN): bind the recurrent state and preserve the committed
    // slab — the chunk forward advances it through rejected draft positions.
    const bool hybrid = ssm_state_ != nullptr;
    if (hybrid) {
        if (!ensure_spec_state_scratch_()) {
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
        rec_slot = recurrent_slot_for_(req->id);
        if (!check(cudaMemcpyAsync(spec_state_scratch_, ssm_state_->seq_base(rec_slot),
                                   ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice, stream),
                   "state snapshot D2D")) {
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
        fill_recurrent_state(*req, state, /*reset=*/false, stream);
        if (hybrid_mc) {
            // Grouped recurrent chunk: candidate c is scan sequence c on slot
            // h_spec_mc_slots_[c]. Seed the reserved slots from the committed
            // state (candidate 0 runs in place on the live slot, exactly like
            // the linear chunk). The pad rows past the last group belong to
            // no sequence (run_gdn zeroes their conv/scan rows).
            bool seeded = true;
            for (size_t c = 1; c < mc.size() && seeded; ++c)
                seeded = check(cudaMemcpyAsync(ssm_state_->seq_base(h_spec_mc_slots_[c]),
                                               spec_state_scratch_, ssm_state_->per_seq_bytes(),
                                               cudaMemcpyDeviceToDevice, stream),
                               "mc slot seed D2D");
            if (!seeded) {
                kv_manager_->rollback(req->id, p0);
                spec_stats_.miss_steps++;
                return false;
            }
            state.ssm_seq_slots = d_spec_mc_slots_;
            state.ssm_n_seq = static_cast<int>(mc.size());
            state.ssm_seq_tokens = mc_rows_per_cand;
            state.d_chunk_len = d_spec_chunk_len_;  // per-group real rows (staged above)
        }
        // Snapshot the state as of the chunk's FIRST row: that row is the last
        // real token, so it is the state a fully rejected draft needs.
        if (spec_state_snap_ != nullptr && d_spec_snap_n_ != nullptr) {
            const int snap_rows = 1;
            if (check(cudaMemcpyAsync(d_spec_snap_n_, &snap_rows, sizeof(int), cudaMemcpyHostToDevice,
                                      stream),
                      "snapshot row H2D")) {
                state.spec_snap_slab = spec_state_snap_;
                state.d_snap_n = d_spec_snap_n_;
                state.spec_prev_slab = spec_state_scratch_;  // the pre-chunk copy
            }
        }
    }

    Tensor logits_out;
    if (runtime_config().diagnostics.spec_capture_probe) {
        spec_capture_probe_forward_(state, logits_out, stream);
    } else if (capture_on) {
        if (!spec_captured_forward_(state, logits_out, stream)) {
            // Warmup use, or capture/launch failed (nothing executed) — run
            // eagerly. A doomed capture means the capture-mode path itself
            // threw; strip the capture fields so the eager forward takes the
            // plain host-length chunk path (the padded chunk stays valid).
            if (spec_capture_doomed_) {
                state.ctx_capacity = 0;
                state.d_past_len = nullptr;
                // Keep d_chunk_len: the chunk stays padded, and the hybrid
                // recurrent-state updates must still stop at the real length.
            }
            executor_->forward_logits(state, logits_out, stream);
        }
    } else {
        executor_->forward_logits(state, logits_out, stream);
    }

    // Penalty history (production parity: output_tokens, unbounded window).
    const bool penalties = req->repetition_penalty != 1.0f ||
                           req->frequency_penalty != 0.0f || req->presence_penalty != 0.0f;
    const int32_t* d_hist = nullptr;
    int n_hist = 0;
    if (penalties) {
        InferenceState pstate;
        upload_penalties(*req, pstate, stream);
        if (pstate.penalty_tokens == nullptr) {
            // Upload failed — an unpenalized verify would diverge from the
            // eager path; fall back to the normal step. The chunk forward
            // already ran: restore the committed hybrid state.
            if (hybrid)
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                    ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                    ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice, stream));
            kv_manager_->rollback(req->id, p0);
            spec_stats_.miss_steps++;
            return false;
        }
        d_hist = pstate.penalty_tokens;
        n_hist = pstate.n_penalty_tokens;
    }

    // Greedy token for every chunk position, D2H, host compare. With
    // token_recycling the same lm-head pass also harvests each row's top-M
    // logit ids for the adjacency table (the model's own successor
    // candidates — Token Recycling's feed signal).
    const int recycle_m =
        runtime_config_.speculative.token_recycling
            ? std::min({std::max(1, runtime_config_.speculative.recycle_slots),
                        kRowwiseTopMMax})
            : 0;
    executor_->greedy_argmax_all(chunk_len, d_spec_argmax_, stream, d_hist, n_hist,
                                 d_spec_tokens_ + 1, req->repetition_penalty,
                                 req->frequency_penalty, req->presence_penalty,
                                 recycle_m > 0 ? d_spec_topm_ : nullptr, recycle_m,
                                 banned_tokens_device_(stream),
                                 static_cast<int>(banned_token_ids_.size()));
    // One D2H covers [argmax | topm] (contiguous block, #1055); without the
    // harvest only the argmax prefix is copied.
    const size_t d2h_ints =
        recycle_m > 0
            ? static_cast<size_t>(spec_chunk_cap_) + static_cast<size_t>(chunk_len) * recycle_m
            : static_cast<size_t>(chunk_len);
    bool argmax_ok =
        check(cudaMemcpyAsync(h_spec_argmax_.as<int32_t>(), d_spec_argmax_, d2h_ints * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream), "argmax D2H") &&
        check(cudaStreamSynchronize(stream), "verify sync");
    if (!argmax_ok) {
        // No tokens were emitted; leave the request exactly as before the
        // step (the chunk forward advanced hybrid state — restore it).
        if (hybrid)
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                                               ssm_state_->per_seq_bytes(),
                                               cudaMemcpyDeviceToDevice, stream));
        kv_manager_->rollback(req->id, p0);
        return false;
    }

    if (runtime_config_.diagnostics.spec_trace) {
        // Whole block in runtime/spec_trace.cpp: the speculation loop should
        // not carry its own diagnostics, and this one reads full logits.
        spec_trace_emit_verify(p0, t0, mc_on ? &mc[0] : &draft, mc_on ? static_cast<int>(mc.size()) : 0,
                               h_spec_argmax_.as<int32_t>(), chunk_len, executor_.get(), d_spec_logits_.get(),
                               h_spec_logits_, model_->config().vocab_size, stream);
    }

    // Accept and emit through the same per-token bookkeeping as the eager
    // decode path. mc: pick the candidate with the longest matching prefix
    // first (ties -> higher adjacency rank), then run the shared emit loop
    // over that candidate's rows.
    int mc_row0 = 0;                     // row offset of the winning group
    const std::vector<int32_t>* acc = &draft;  // tokens the emit loop verifies against
    if (mc_on) {
        int best_c = 0, best_m = -1;
        for (size_t c = 0; c < mc.size(); ++c) {
            const int r0 = static_cast<int>(c) * mc_rows_per_cand;
            int m = 0;
            while (m < static_cast<int>(mc[c].size()) && h_spec_argmax_.as<int32_t>()[r0 + m] == mc[c][m])
                ++m;
            if (m > best_m) {
                best_m = m;
                best_c = static_cast<int>(c);
            }
        }
        mc_row0 = best_c * mc_rows_per_cand;
        acc = &mc[best_c];
    }
    // Linear: K (attn-scores clamp may have trimmed the staged chunk below
    // draft.size()). mc: the winning candidate's full length.
    const int acc_len = mc_on ? static_cast<int>(acc->size()) : K;
    int matched = 0;  // accepted draft tokens (their KV entries are valid)
    int emitted = 0;
    // Whether this chunk STARTED inside a budgeted think block. Verifies run
    // inside think for MTP-bound requests (gate exception above); the break
    // below must fire only on ENTERING the block mid-chunk, or every in-think
    // verify emits 1 token, reads as avg 1.0 emitted/verify and trips the
    // economics guard on a head that is accepting fine.
    const bool think_at_chunk_start = req->in_think_block;
    for (int j = 0; j + mc_row0 < chunk_len; ++j) {
        if (mc_on && j > acc_len) break;  // stay inside the winning group
        const int32_t tokj = h_spec_argmax_.as<int32_t>()[mc_row0 + j];
        req->output_tokens.push_back(tokj);
        track_think_state(*req, tokj);
        emitted++;
        const bool hard_stop = should_stop(*req, tokj) ||
                               static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
        // Per-request FSM; must advance before finish_request returns the
        // manager to the pool. (Spec gates exclude json/schema requests, so
        // this is normally null — kept for parity with the eager path.)
        if (req->constraints)
            req->constraints->update(tokj);
        if (hard_stop) {
            finish_request(req);
            break;
        }
        if (j >= acc_len || tokj != (*acc)[j]) break;  // bonus reached or draft diverged
        matched++;
        // Crossing a think boundary mid-chunk (either direction): stop
        // extending; the accepted prefix stays. Entering: the budget forcing
        // lives in the loop/eager path. Leaving: the post-</think> stop/grace
        // window is the boundary where chunk-argmax noise diverges from the
        // eager path (empty-content completions in degen_suite) - hand it to
        // the eager step, which is byte-identical to the pre-MTP behavior.
        if (req->think_budget > 0.0f && req->in_think_block != think_at_chunk_start) break;
    }
    kv_manager_->touch(req->id);

    // mc: materialize the winner's KV — copy its private block(s) covering
    // the accepted span [p0, p0+matched] back over the canonical entries.
    // The rollback below frees the private blocks; a freed block re-used by
    // work on ANOTHER stream (a concurrent prefill on pf_stream) must not
    // race the in-flight copy, so sync first — but only when such a stream
    // can exist. Single-stream steady state (batch=1, no prefill queued —
    // the CLI/agent case) skips the sync: every later consumer of the pool
    // enqueues on this same stream, which is ordered after the copy.
    // (WSL2 measures cudaStreamSynchronize at 1-4 ms — per verify step that
    // sync alone was a double-digit share of the TR verify cost.)
    if (mc_on && req->status != RequestStatus::FINISHED) {
        const int mc_best_c = mc_row0 / mc_rows_per_cand;
        int srcs[KVCache::kCopyMaxPairs];
        int dsts[KVCache::kCopyMaxPairs];
        int n_pairs = std::min((p0 + matched) / kv_bs - mc_bp + 1, KVCache::kCopyMaxPairs);
        for (int t = 0; t < n_pairs; ++t) {
            srcs[t] = block_table[mc_priv_base + mc_best_c * mc_n_priv + t];
            dsts[t] = block_table[mc_bp + t];
        }
        kv_cache_raw_->copy_blocks_device(srcs, dsts, n_pairs, stream);
        const bool other_stream_may_reuse =
            !sched_prefill_batch_.empty() || sched_decode_batch_.size() > 1;
        if (other_stream_may_reuse)
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    }

    // Token-Recycling: feed the model's own top-M successor candidates for
    // every real chunk row into the adjacency table — rejected rows
    // included (the prediction is the model's, valid regardless of
    // acceptance; that breadth is what makes the next draft fire).
    if (recycle_m > 0) {
        TokenRecycleTable& tr = spec_recycle_table_();
        for (int j = 0; j < chunk_len; ++j) {
            int32_t tok;
            if (mc_on) {
                const int c = j / mc_rows_per_cand;
                const int r = j % mc_rows_per_cand;
                if (r > static_cast<int>(mc[c].size()))
                    continue;  // pad row of a short candidate
                tok = (r == 0) ? t0 : mc[c][r - 1];
            } else {
                tok = (j == 0) ? t0 : draft[j - 1];
            }
            tr.observe_topk(tok, std::span(h_spec_topm_ + static_cast<size_t>(j) * recycle_m,
                                           static_cast<size_t>(recycle_m)));
        }
    }

    // MTP bookkeeping runs BEFORE the hybrid re-forward below — it consumes
    // this chunk's hidden rows, which the re-forward overwrites.
    if (mtp_spec_decode_enabled())
        mtp_post_verify_update_(*req, emitted, mc_on ? mc_row0 : 0);

    // Hybrid mc: land the winner's recurrent state on the live slot. Runs
    // BEFORE the rollback because the partial-accept replay below re-forwards
    // the chunk through the same grouped geometry, whose row tables alias
    // the candidates' private blocks - they must still be this request's.
    //   full accept, winner 0   : in place already (the dominant case).
    //   full accept, winner c>0 : one slab copy from candidate c's slot.
    //   nothing accepted        : adopt the row-0 snapshot (below, shared
    //                             with the linear path).
    //   partial                 : restore committed, swap the winner's slot
    //                             onto the live one, replay real_rows =
    //                             matched + 1 through the captured graph (or
    //                             eager): every group re-runs, only the
    //                             winner's group now advances the live slot.
    if (hybrid_mc && req->status != RequestStatus::FINISHED) {
        const int mc_best_c = mc_row0 / mc_rows_per_cand;
        if (matched >= K && mc_best_c != 0) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot),
                                               ssm_state_->seq_base(h_spec_mc_slots_[mc_best_c]),
                                               ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice,
                                               stream));
        } else if (matched < K && (matched > 0 || state.spec_snap_slab == nullptr)) {
            // (matched == 0 without a snapshot slab: replay one row rather
            // than leave the live slot at the primary group's last row.)
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                                               ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice,
                                               stream));
            std::swap(h_spec_mc_slots_[0], h_spec_mc_slots_[mc_best_c]);
            const int real_rows = matched + 1;
            const bool restaged =
                check(cudaMemcpyAsync(d_spec_mc_slots_, h_spec_mc_slots_.data(),
                                      static_cast<size_t>(kMtpMaxTopW) * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream),
                      "mc replay slots H2D") &&
                check(cudaMemcpyAsync(d_spec_chunk_len_, &real_rows, sizeof(int), cudaMemcpyHostToDevice,
                                      stream),
                      "mc replay rows H2D");
            // The graph reads the group's real row count and the slot table
            // from device, so the recording that verified the chunk serves
            // the replay unchanged (no re-capture per accepted length).
            Tensor replay_logits;
            bool replayed = false;
            if (restaged && capture_on && !spec_capture_doomed_)
                replayed = spec_captured_forward_(state, replay_logits, stream);
            if (restaged && !replayed) {
                if (spec_capture_doomed_) {
                    state.ctx_capacity = 0;
                    state.d_past_len = nullptr;
                }
                executor_->forward_logits(state, replay_logits, stream);
            }
            if (!restaged)
                IMP_LOG_WARN("spec-mc: hybrid replay staging failed - recurrent state left at the "
                             "chunk's first row (draft tail re-verified next step)");
            std::swap(h_spec_mc_slots_[0], h_spec_mc_slots_[mc_best_c]);
        }
    }

    // Drop KV for rejected draft positions: keep t0 + matched drafts.
    kv_manager_->rollback(req->id, p0 + 1 + matched);

    // Hybrid, partial acceptance: the in-place state advanced through
    // rejected draft positions — restore the committed slab and re-advance
    // it over the accepted prefix ([t0, d1..d_matched] is still staged in
    // d_spec_tokens_). Full acceptance keeps the advanced state (it covers
    // exactly the forwarded tokens); finished requests skip (the slot is
    // released and nothing reads the state again). The replay rewrites the
    // kept KV rows with identical values.
    if (hybrid && matched == 0 && state.spec_snap_slab != nullptr && req->status != RequestStatus::FINISHED) {
        // Nothing was accepted, so the committed state must be the one after
        // the chunk's first row — which the scan just wrote into the snapshot
        // slab. Adopt it: a 151 MiB device copy instead of a full model
        // forward. Everything below is the general path for a partial
        // acceptance that landed further along.
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot), state.spec_snap_slab,
                                           ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice, stream));
    } else if (hybrid && !hybrid_mc && matched < K && req->status != RequestStatus::FINISHED) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                                           ssm_state_->per_seq_bytes(),
                                           cudaMemcpyDeviceToDevice, stream));
        const int replay_ctx = p0 + 1 + matched;
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_spec_context_len_, &replay_ctx, sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
        state.max_context_len = replay_ctx;
        Tensor replay_logits;
        bool replayed = false;
        if (capture_on && !spec_capture_doomed_) {
            // Reuse the captured chunk graph rather than forwarding eagerly.
            // The graph reads its chunk length from device memory and the GDN
            // scan commits h_state at that row, so the same recording serves a
            // shorter prefix: keep the padded row count, tell the device the
            // real one. Eager here cost 25.1 ms for one or two rows against
            // 17.8 ms for the captured three-row chunk, measured over 300
            // verifies on Qwen3.8-27B — the difference is launch pacing, not
            // work.
            const int real_rows = matched + 1;
            if (check(cudaMemcpyAsync(d_spec_chunk_len_, &real_rows, sizeof(int), cudaMemcpyHostToDevice,
                                      stream),
                      "replay chunk len H2D")) {
                state.n_tokens = chunk_pad;
                replayed = spec_captured_forward_(state, replay_logits, stream);
            }
        }
        if (!replayed) {
            state.n_tokens = matched + 1;
            // Fallback: an unpadded eager forward of the accepted prefix on the
            // plain host-length chunk path.
            state.ctx_capacity = 0;
            state.d_past_len = nullptr;
            state.d_chunk_len = nullptr;
            executor_->forward_logits(state, replay_logits, stream);
        }
    }

    spec_stats_.verify_steps++;
    spec_stats_.drafted += K;
    spec_stats_.accepted += matched;
    spec_stats_.emitted += emitted;
    spec_stats_.verify_wall_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - verify_t0)
            .count();
    // (request-end stats logging lives in finish_request)

    // Acceptance economics: structured-but-mutating content (number tables,
    // counters) produces plenty of suffix matches whose continuations are
    // always wrong — pure miss counting never triggers there while every
    // step pays a full verify chunk for 1 emitted token. After a fair
    // sample, an acceptance rate below 15% can't amortize the verify cost;
    // hand the request back to the async loop.
    if (matched == 0) {
        ++req->spec_consecutive_misses;
    } else {
        req->spec_consecutive_misses = 0;
    }
    req->spec_verifies++;
    req->spec_drafted += K;
    req->spec_accepted += matched;
    if (draft_from_prediction) {
        req->pred_accepted += matched;
        req->pred_rejected += K - matched;
    }
    // MTP economics: an MTP-filled verify must emit enough tokens to beat
    // the async loop it displaces (eager chunk ≈ 2x a loop step, plus the
    // chain's full-vocab lm_head GEMVs, plus the hybrid partial-accept
    // replay). Below ~4 emitted/verify the step loses outright — doom MTP
    // drafting for this request; the suffix matcher and miss bursts carry on.
    if (draft_from_mtp) {
        mtp_econ_verifies_++;
        mtp_econ_emitted_ += emitted;
        mtp_econ_rows_ += K;
        // Adaptive chain depth (AIMD): a fully accepted chain earns one more
        // row next draft (up to the configured k), any rejection sheds one.
        // Draft-poor prompts converge to k=1 verifies instead of paying the
        // deep-chunk cost at low accept; draft-rich prompts climb back.
        if (runtime_config_.speculative.mtp_adaptive_k) {
            if (matched >= K)
                mtp_k_live_ = std::min(mtp_spec_decode_k(), mtp_chain_k_() + 1);
            else
                mtp_k_live_ = std::max(1, mtp_chain_k_() - 1);
        }
        constexpr int kMtpEconSample = 8;  // fair sample before judging
        // Break-even avg emitted/verify is configurable (0 disables): the
        // right value depends on the chain lm_head cost (NVFP4 vs FP16) and
        // the verify-chunk cost, both of which have moved since #852.
        // Negative selects the k-aware default; see dispatch_policy.h for the
        // measurement it comes from and for why an absolute value doomed every
        // chain length this engine runs.
        const float cfg_min = runtime_config_.speculative.mtp_econ_min_emit;
        // 1 + f*k IS a draft-acceptance threshold, because a verify emits
        // exactly 1 + accepted. f = 0.40 sits on the measured break-even
        // acceptance (39.0 / 37.5 / 47.7 % at k = 1 / 2 / 3); see
        // dispatch_policy.h for the derivation and for why permissive at k=3
        // is the deliberate direction.
        constexpr float kMtpEconAccept = 0.40f;
        // Price the k that RAN (avg rows/verify), not the configured ceiling:
        // with the adaptive depth above, a request parked at k_live=1 must be
        // judged against the k=1 break-even or the deeper config dooms it.
        const float k_ran = static_cast<float>(mtp_econ_rows_) /
                            static_cast<float>(mtp_econ_verifies_);
        const float min_emit = cfg_min < 0.0f ? 1.0f + kMtpEconAccept * k_ran : cfg_min;
        if (min_emit > 0.0f && mtp_econ_verifies_ >= kMtpEconSample &&
            static_cast<float>(mtp_econ_emitted_) <
                static_cast<float>(mtp_econ_verifies_) * min_emit) {
            IMP_LOG_INFO("[mtp-econ] verifies=%d emitted=%lld min_emit=%.2f this_emitted=%d matched=%d",
                         mtp_econ_verifies_, mtp_econ_emitted_, min_emit, emitted, matched);
            mtp_unbind_("uneconomic: avg emitted/verify below break-even");
        }
    }
    const bool acceptance_poor =
        req->spec_verifies >= 8 &&
        req->spec_accepted * 100 < req->spec_drafted * 15;
    if (!req->spec_ngram_given_up && scfg.give_up_after > 0 &&
        (acceptance_poor || req->spec_consecutive_misses >= scfg.give_up_after)) {
        req->spec_ngram_given_up = true;
        req->spec_last_giveup_pos = static_cast<int>(req->output_tokens.size());
        if (acceptance_poor)
            req->spec_acceptance_doomed = true;  // economics verdict is final
        IMP_LOG_INFO("spec-ngram: req %d gave up (%s: verifies=%d accepted=%lld/%lld misses=%d) — "
                     "re-enabling async graph loop",
                     req->id, acceptance_poor ? "acceptance-poor" : "draft-poor",
                     req->spec_verifies, req->spec_accepted, req->spec_drafted,
                     req->spec_consecutive_misses);
    }

    return true;
}

// spec_capture_probe_forward_ (diagnostics census) and the production
// graph-captured verify (#847) live in engine_spec_capture.cpp.

}  // namespace imp
