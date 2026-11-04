// MTP head (model_mtp.safetensors, DeepSeek-V3 family) chain-drafts the next
// mtp_spec_k_ tokens; step_spec_verify_ uses it when the suffix/ngram matcher
// has no draft (#847, #804).
// Pairing convention: cache pair i is (emb(t_{i+1}), h_i); after P pairs the
// head is ready to draft the token following t_P (matches the per-step decode
// telemetry in engine_scheduler.cpp).
// The single MTP workspace tracks ONE request at a time (verify is batch-1);
// mtp_active_.history covers t_0..t_P so a follow-up request resumes over the
// shared prefix instead of restarting. Any uncoverable gap unbinds drafting; the suffix matcher keeps working.

#include "memory/backend.h"
#include "compute/layernorm.h"
#include "compute/mtp_forward.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "model/model.h"
#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/request.h"

#include <cuda_fp16.h>
#include <algorithm>

namespace imp {

// The wire ceiling for a per-request `speculative.mtp_k` cannot exceed the
// device chain buffer. spec_request.h stays out of the CUDA include chain, so
// the two constants are tied here, where both are visible.
static_assert(kSpecRequestMaxMtpK <= kMtpMaxChainK,
              "per-request MTP depth ceiling must fit the device chain buffer");

void Engine::mtp_unbind_(const char* why) {
    if (mtp_active_.req >= 0 && !mtp_active_.stale_logged) {
        IMP_LOG_INFO("mtp-spec: drafting off for req %d (%s)", mtp_active_.req, why);
        mtp_active_.stale_logged = true;
    }
    // The slot goes back to the pool with the tokens its cache covers, so a
    // follow-up request can resume over the shared prefix.
    if (mtp_active_.req >= 0 && mtp_ws_storage_ != nullptr) {
        auto* ws = mtp_ws_storage_.get();
        const int slot = ws->cur_slot;
        if (slot >= 0 && slot < static_cast<int>(mtp_pool_.slot_history.size())) {
            mtp_pool_.slot_history[static_cast<size_t>(slot)] = mtp_active_.history;
            mtp_pool_.free_slots.push_back(slot);
        }
    }
    mtp_active_.req = -1;
    mtp_active_.pending.clear();
    mtp_active_.chains.clear();
    mtp_active_.draft_ctx = -1;
}

// ── Per-request bindings (batched verify) ────────────────────────────────
int Engine::mtp_chain_k_() const noexcept {
    return std::max(1, mtp_active_.k_live > 0 ? mtp_active_.k_live : mtp_spec_k_);
}

// Park the active binding (its slot stays held), then load req_id's parked
// one and select its KV slot. req_id < 0 = park only.
void Engine::mtp_activate_(int req_id) {
    if (mtp_ws_storage_ == nullptr || (req_id >= 0 && mtp_active_.req == req_id))
        return;
    auto it = req_id >= 0 ? mtp_pool_.binds.find(req_id) : mtp_pool_.binds.end();
    if (req_id >= 0 && it == mtp_pool_.binds.end())
        return;
    if (mtp_active_.req >= 0) {
        mtp_active_.slot = mtp_ws_storage_->cur_slot;
        const int req = mtp_active_.req;
        mtp_pool_.binds[req] = std::move(mtp_active_);
        mtp_active_ = MtpBind{};
        if (req_id < 0)
            return;
        it = mtp_pool_.binds.find(req_id);  // the insert above may have rehashed
    }
    if (req_id < 0)
        return;
    mtp_active_ = std::move(it->second);
    mtp_pool_.binds.erase(it);
    mtp_active_.req = req_id;
    imp::mtp_select_slot(*mtp_ws_storage_, mtp_active_.slot);
}

int Engine::mtp_acquire_slot_(const Request& req, int offset) {
    if (mtp_ws_storage_ == nullptr)
        return -1;
    auto* ws = mtp_ws_storage_.get();
    const int n_slots = std::max(1, ws->n_kv_slots);
    if (!mtp_pool_.initialized) {
        mtp_pool_.free_slots.clear();
        for (int s = n_slots - 1; s >= 0; --s)
            mtp_pool_.free_slots.push_back(s);
        mtp_pool_.slot_history.assign(static_cast<size_t>(n_slots), {});
        mtp_pool_.slot_used.assign(static_cast<size_t>(n_slots), 0);
        // The active binding (if any) holds its slot.
        if (mtp_active_.req >= 0)
            mtp_pool_.free_slots.erase(std::remove(mtp_pool_.free_slots.begin(), mtp_pool_.free_slots.end(), ws->cur_slot),
                                  mtp_pool_.free_slots.end());
        mtp_pool_.initialized = true;
    }
    const int held = (mtp_active_.req >= 0 ? 1 : 0) + static_cast<int>(mtp_pool_.binds.size());
    if (mtp_pool_.free_slots.empty() || held >= mtp_pool_.bind_cap) {
        // Every binding place is taken: evict a parked binding (with one slot
        // this is what the single workspace always did to the previous request).
        if (mtp_pool_.binds.empty())
            return -1;
        auto it = mtp_pool_.binds.begin();
        mtp_pool_.slot_history[static_cast<size_t>(it->second.slot)] = std::move(it->second.history);
        mtp_pool_.free_slots.push_back(it->second.slot);
        mtp_pool_.binds.erase(it);
    }
    std::vector<MtpSlotCand> cands;
    cands.reserve(mtp_pool_.free_slots.size());
    for (const int s : mtp_pool_.free_slots)
        cands.push_back({.pos = s == ws->cur_slot ? ws->mtp_pos : ws->slot_pos[static_cast<size_t>(s)],
                         .history = &mtp_pool_.slot_history[static_cast<size_t>(s)],
                         .last_use = mtp_pool_.slot_used[static_cast<size_t>(s)]});
    const int pick = mtp_pick_slot(cands, req.input_tokens, offset);
    if (pick < 0)
        return -1;
    const int slot = mtp_pool_.free_slots[static_cast<size_t>(pick)];
    mtp_pool_.free_slots.erase(mtp_pool_.free_slots.begin() + pick);
    mtp_pool_.slot_used[static_cast<size_t>(slot)] = ++mtp_pool_.use_clock;
    return slot;
}

void Engine::mtp_release_(int req_id) {
    if (req_id < 0 || mtp_ws_storage_ == nullptr)
        return;
    if (mtp_active_.req == req_id) {
        mtp_active_.stale_logged = true;  // a finished request is not "drafting off"
        mtp_unbind_("request finished");
        return;
    }
    auto it = mtp_pool_.binds.find(req_id);
    if (it == mtp_pool_.binds.end())
        return;
    if (it->second.slot >= 0 && it->second.slot < static_cast<int>(mtp_pool_.slot_history.size())) {
        mtp_pool_.slot_history[static_cast<size_t>(it->second.slot)] = std::move(it->second.history);
        mtp_pool_.free_slots.push_back(it->second.slot);
    }
    mtp_pool_.binds.erase(it);
}

bool Engine::spec_history_too_short_(const Request& req) const {
    const int n = runtime_config_.speculative.min_history;
    return n > 0 && static_cast<int>(req.output_tokens.size()) < n;
}

bool Engine::mtp_feed_wanted_(const Request& req) const {
    return mtp_binds_request({.mtp_armed = mtp_spec_decode_enabled() && mtp_ws_storage_ != nullptr,
                              .constrained = spec_constrained_refused_(req),
                              .sampling_refused = spec_sampling_refusal_(req) != nullptr});
}

bool Engine::mtp_holds_eager_(const Request& req) const {
    return mtp_spec_decode_enabled() && mtp_bound(mtp_active_, mtp_pool_, req.id);
}

bool Engine::mtp_takes_prompt_(const Request& req) const {
    if (mtp_ws_storage_ == nullptr || !mtp_feed_wanted_(req))
        return false;
    return mtp_prompt_takes_head(mtp_bound(mtp_active_, mtp_pool_, req.id),
                                 mtp_slot_available(mtp_active_, mtp_pool_, mtp_pool_.bind_cap, req.id),
                                 static_cast<int>(sched_prefill_batch_.size()),
                                 bv_.bound_only ? bv_.cap_seq : 1);
}

// A lone step whose head cache is behind can never draft: unbind, so the request leaves the eager path.
void Engine::mtp_unbind_if_behind_(int req_id, int cur_pos) {
    const auto* ws = mtp_ws_storage_.get();
    if (ws != nullptr && mtp_active_.req == req_id && mtp_cache_behind(ws->mtp_pos, cur_pos))
        mtp_unbind_("head cache behind the context after batched decode");
}

SpecDrafterState Engine::spec_drafter_state_(const Request& req) const {
    SpecDrafterState s;
    s.model_capable = spec_ngram_model_capable_();
    // "speculative": false means no speculation at all, not just "no
    // n-gram speculation" - it must also stop MTP and recycling (#1639).
    const bool forced_off = req.spec_override == 0;
    s.ngram_on = req.spec_override >= 0 ? req.spec_override == 1 : runtime_config_.speculative.ngram;
    // The head is a per-request contract, not a process-wide switch: a
    // request may lower its depth or turn it off, and asking for it on a
    // process that armed nothing is a DECLINE with a reason rather than a
    // silent plain decode (spec_request.h). The rule is shared with the
    // server so the two cannot disagree about what a request asked for.
    s.mtp_on = mtp_resolve_request(mtp_request_state_(req)).k > 0;
    s.recycling_on = !forced_off && runtime_config_.speculative.token_recycling;
    s.dspark_on = !forced_off && dspark_ != nullptr;
    return s;
}

MtpRequestState Engine::mtp_request_state_(const Request& req) const {
    MtpRequestState s;
    s.requested_k = req.spec_mtp_k;
    s.armed_k = mtp_spec_k_;
    // "present" covers the head this process declined to upload: that is what
    // the concurrency decline looks like from a request's side, and telling it
    // apart from a checkpoint with no head is the whole point of the reason.
    s.head_present = model_ && (model_->mtp_ || model_->mtp_head_available_unloaded_);
    s.head_loaded = model_ && model_->mtp_ && model_->mtp_->loaded;
    s.forced_off = req.spec_override == 0;
    return s;
}

int Engine::mtp_request_k_(const Request& req) const {
    return mtp_resolve_request(mtp_request_state_(req)).k;
}

int Engine::mtp_chain_k_(const Request& req) const {
    const int req_k = mtp_request_k_(req);
    return req_k <= 0 ? 0 : std::min(mtp_chain_k_(), req_k);
}

std::vector<int32_t> Engine::mtp_take_draft_(const Request& req) {
    if (mtp_bound(mtp_active_, mtp_pool_, req.id))
        mtp_activate_(req.id);
    if (mtp_active_.req != req.id || mtp_active_.pending.empty())
        return {};
    // Drafted for a different context (an async-loop burst or eager steps
    // advanced the request without refreshing the chain): not usable.
    if (mtp_active_.draft_ctx != req.context_len())
        return {};
    return mtp_active_.pending;
}

std::vector<std::vector<int32_t>> Engine::mtp_take_chains_(const Request& req) {
    if (mtp_bound(mtp_active_, mtp_pool_, req.id))
        mtp_activate_(req.id);
    if (mtp_active_.req != req.id || mtp_active_.chains.size() < 2)
        return {};
    if (mtp_active_.draft_ctx != req.context_len())
        return {};
    return mtp_active_.chains;
}

// Feed n_pairs (token, hidden-row) pairs into the MTP KV cache, then
// (optionally) chain-draft mtp_spec_k_ tokens. Only the final pair of a
// chaining feed needs logits/argmax: lm_head GEMV dominates per-pair cost (~1 GiB read on Qwen3.6's 248k vocab).
bool Engine::mtp_feed_pairs_(const int32_t* tokens, const void* d_hidden_rows, int n_pairs,
                             bool chain_after) {
    auto* ws = mtp_ws_storage_.get();
    if (ws == nullptr || n_pairs <= 0)
        return false;
    const int hidden_dim = model_->config_.d_model;
    const int vocab_size = model_->config_.vocab_size;
    const int K = mtp_chain_k_();
    // MTP KV capacity (kMtpKvCap clamp at enable time): past it the cache
    // addressing is undefined, so stop drafting instead.
    if (ws->max_seq_len > 0 && ws->mtp_pos + n_pairs + (chain_after ? K : 0) > ws->max_seq_len)
        return false;

    // Device-side chain: dense MTP heads have no host dependency between chain
    // steps (MoE head's routing needs a per-step D2H); each step's argmax feeds
    // the next step's embedding lookup on device. One D2H+sync drains the whole chain instead of K host round-trips.
    const bool device_chain = chain_after && ws->n_experts == 0 &&
                              ws->d_chain_tokens != nullptr && K <= imp::kMtpMaxChainK;

    // diagnostics.mtp_prenorm_h: feed the head the POST-final-norm hidden
    // (upstream vLLM/HF taps the target's post-norm hidden; imp's executor
    // hidden_ is pre-norm). Covers every fed pair, not just the chain input.
    // Not on an hc head (Qwen4Exp): its h_prev is the multi_hidden stream, fed un-normed (V:302-308).
    if (runtime_config_.diagnostics.mtp_prenorm_h && ws->hc_count == 0) {
        // ws->d_prenorm_rows is sized once at enable time (executor's token cap,
        // bounding every prefill feed); a feed past it fails, and the caller
        // unbinds drafting for the request.
        if (!ws->d_prenorm_rows || n_pairs > ws->prenorm_rows_cap) {
            IMP_LOG_WARN("mtp-spec: feed of %d rows exceeds the post-norm scratch (%d rows)", n_pairs,
                         ws->prenorm_rows_cap);
            return false;
        }
        int64_t shape[2] = {n_pairs, hidden_dim};
        Tensor in_v(const_cast<void*>(d_hidden_rows), QType::F16, 2, shape, true);
        Tensor out_v(ws->d_prenorm_rows, QType::F16, 2, shape, true);
        imp::rmsnorm(in_v, model_->output_norm(), out_v, model_->config_.rms_norm_eps,
                     decode_stream(), model_->config_.norm_weight_offset);
        d_hidden_rows = ws->d_prenorm_rows;
    }
    const char* base = static_cast<const char*>(d_hidden_rows);
    const int row_cols = imp::mtp_h_prev_cols(*ws);  // hc_count x d_model on an hc head
    int pred = -1;
    // Batched feed (dense heads): the per-pair loop reads the whole head's
    // weights once per token, costing prefill throughput badly at scale. The
    // last pair of a chaining feed always stays per-pair (it drafts); below 8 pairs, batch setup costs more launches than it saves.
    int j0 = 0;
    const int n_feed = chain_after ? n_pairs - 1 : n_pairs;
    if (n_feed >= 8 && ws->feed_rows_cap > 0) {
        while (j0 < n_feed) {
            const int m = std::min(ws->feed_rows_cap, n_feed - j0);
            const void* rows = base + (static_cast<size_t>(j0) * row_cols * sizeof(__half));
            if (!imp::mtp_feed_batch(tokens + j0, rows, m, *model_->mtp_, model_->tok_emb_,
                                     *ws, hidden_dim, decode_stream()))
                break;  // unsupported head or transient failure, per-pair takes over
            for (int j = j0; j < j0 + m; ++j)
                mtp_active_.history.push_back(tokens[j]);
            j0 += m;
        }
    }
    // Multi-candidate width (speculative.mtp_tree_width): W > 1 drafts W chains,
    // chain 0 linear, chains 1..W-1 branching at the first position on the head's
    // top-W ids. Device-chain only (host path would reintroduce a top-W sync); MoE heads silently keep W = 1.
    const int cfg_w = runtime_config_.speculative.mtp_tree_width;
    const int W = (chain_after && device_chain && cfg_w > 1 && ws->d_topk != nullptr &&
                   ws->d_h_final_snap != nullptr)
                      ? std::min(cfg_w, imp::kMtpMaxTopW)
                      : 1;
    for (int j = j0; j < n_pairs; ++j) {
        const void* h_j = base + (static_cast<size_t>(j) * row_cols * sizeof(__half));
        const bool last = (j == n_pairs - 1);
        if (chain_after && last && device_chain) {
            // W > 1: the same launch also fills ws->d_topk (fast top-W, no
            // sync); rank 0 lands in the chain-0 slot either way.
            if (!mtp_draft_one(tokens[j], h_j, hidden_dim, vocab_size, nullptr, nullptr, W > 1 ? W : 0,
                               nullptr,
                               /*d_out_token=*/ws->d_chain_tokens))
                return false;
        } else {
            int* out = (chain_after && last) ? &pred : nullptr;
            if (!mtp_draft_one(tokens[j], h_j, hidden_dim, vocab_size, out))
                return false;
        }
        mtp_active_.history.push_back(tokens[j]);
    }
    if (!chain_after)
        return true;

    // Chain: continue on the head's own h_final. The chained KV appends are
    // speculative, so roll the cache back after so only real pairs persist;
    // with W > 1 the rollback also runs BETWEEN chains (successive chains overwrite the same slots, stream order makes that safe).
    const int pos_after = ws->mtp_pos;
    std::vector<int32_t> chain;
    chain.reserve(K);
    if (device_chain) {
        cudaStream_t stream = decode_stream();
        constexpr int kStride = imp::kMtpMaxChainK;
        int launched[imp::kMtpMaxTopW] = {};
        launched[0] = 1;  // d_chain_tokens[0] came from the last feed pair
        // Chains 1..W-1 branch at position 1: their first continuation
        // forward consumes the last FEED pair's h_final, which chain 0's
        // first continuation is about to overwrite, so snapshot it.
        if (W > 1 && cudaMemcpyAsync(ws->d_h_final_snap, ws->d_h_final,
                                     static_cast<size_t>(hidden_dim) * sizeof(__half),
                                     cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            ws->mtp_pos = pos_after;
            return false;
        }
        for (int k = 1; k < K; ++k) {
            if (!mtp_draft_one(-1, ws->d_h_final, hidden_dim, vocab_size, nullptr, nullptr, 0,
                               /*d_prev_token=*/ws->d_chain_tokens + k - 1,
                               /*d_out_token=*/ws->d_chain_tokens + k))
                break;
            launched[0]++;
        }
        for (int c = 1; c < W; ++c) {
            ws->mtp_pos = pos_after;  // drop the previous chain's appends
            int32_t* slots = ws->d_chain_tokens + (static_cast<size_t>(c) * kStride);
            // Token 0 of chain c is the head's rank-c candidate, known, not
            // drafted.
            if (cudaMemcpyAsync(slots, ws->d_topk + c, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream) !=
                cudaSuccess)
                break;
            launched[c] = 1;
            for (int k = 1; k < K; ++k) {
                const bool first = (k == 1);
                if (!mtp_draft_one(-1, first ? ws->d_h_final_snap : ws->d_h_final, hidden_dim, vocab_size,
                                   nullptr, nullptr, 0,
                                   /*d_prev_token=*/slots + k - 1,
                                   /*d_out_token=*/slots + k))
                    break;
                launched[c]++;
            }
        }
        int32_t h_chain[imp::kMtpMaxTopW * imp::kMtpMaxChainK];
        const size_t drain = (W > 1 ? static_cast<size_t>(W) * kStride : static_cast<size_t>(launched[0]));
        // The head's top-1/top-2 logit margin rides the chain's sync (W floats)
        // - the branch gate below reads it.
        float h_topv[imp::kMtpMaxTopW] = {};
        bool have_topv = false;
        if (W > 1 && ws->d_topk_val != nullptr)
            have_topv = cudaMemcpyAsync(h_topv, ws->d_topk_val, static_cast<size_t>(W) * sizeof(float),
                                        cudaMemcpyDeviceToHost, stream) == cudaSuccess;
        if (engine_internal::copy_d2h_sync(h_chain, ws->d_chain_tokens, drain * sizeof(int32_t), stream) !=
            cudaSuccess) {
            ws->mtp_pos = pos_after;
            return false;
        }
        for (int k = 0; k < launched[0]; ++k) {
            if (h_chain[k] < 0 || h_chain[k] >= vocab_size)
                break;  // NaN-logits guard, keep the valid prefix
            chain.push_back(h_chain[k]);
        }
        if (chain.empty()) {
            ws->mtp_pos = pos_after;
            return false;
        }
        mtp_active_.chains.clear();
        // Margin gate (speculative.mtp_tree_margin): a confident head keeps
        // the linear chunk; the second candidate wins almost only where the
        // top-1/top-2 margin is small. Alternates are drafted regardless (cost paid either way); only the verify shape changes.
        const float margin_cfg = runtime_config_.speculative.mtp_tree_margin;
        const bool branch = W > 1 && !(margin_cfg > 0.0f && have_topv && (h_topv[0] - h_topv[1]) > margin_cfg);
        if (W > 1) {
            if (branch)
                mtp_pool_.tree_branched++;
            else
                mtp_pool_.tree_linear++;
        }
        if (branch) {
            mtp_active_.chains.push_back(chain);
            for (int c = 1; c < W; ++c) {
                std::vector<int32_t> alt;
                for (int k = 0; k < launched[c]; ++k) {
                    const int32_t t = h_chain[(static_cast<size_t>(c) * kStride) + k];
                    if (t < 0 || t >= vocab_size)
                        break;
                    alt.push_back(t);
                }
                // A branch whose seed duplicates the primary's first token is
                // a wasted candidate row, not a second path (top-W ids are
                // distinct by construction; this guards a degenerate drain).
                if (!alt.empty() && alt[0] != chain[0])
                    mtp_active_.chains.push_back(std::move(alt));
            }
            if (mtp_active_.chains.size() < 2)
                mtp_active_.chains.clear();
        }
    } else {
        mtp_active_.chains.clear();  // host chain drafts the linear path only
        if (pred < 0 || pred >= vocab_size) {
            ws->mtp_pos = pos_after;
            return false;
        }
        chain.push_back(pred);
        int prev = pred;
        for (int k = 1; k < K; ++k) {
            int p = -1;
            if (!mtp_draft_one(prev, imp::mtp_chain_hidden(*ws), hidden_dim, vocab_size, &p) || p < 0 ||
                p >= vocab_size)
                break;
            chain.push_back(p);
            prev = p;
        }
    }
    ws->mtp_pos = pos_after;
    mtp_active_.pending = std::move(chain);
    mtp_active_.draft_ctx = static_cast<int>(mtp_active_.history.size());
    mtp_pool_.pending_prediction = mtp_active_.pending[0];  // legacy accuracy counter
    return true;
}

void Engine::mtp_post_verify_update_(const Request& req, int emitted, int row0) {
    if (mtp_bound(mtp_active_, mtp_pool_, req.id))
        mtp_activate_(req.id);
    if (mtp_active_.req != req.id || emitted <= 0)
        return;
    auto* ws = mtp_ws_storage_.get();
    if (ws == nullptr)
        return;
    if (mtp_active_.carry_n > 0) {
        mtp_carry_flush_(req);  // batched steps before this verify (pairs below p0 only)
        if (mtp_active_.req != req.id)
            return;
    }
    // Sync check: the cache must cover exactly the pre-step context
    // (pos == p0 == context_len - emitted - 1).
    const int p0 = req.context_len() - emitted - 1;
    if (ws->mtp_pos != p0) {
        mtp_unbind_("desync before verify feed");
        return;
    }
    // Row row0+j of this verify chunk produced emitted token j (row0 = winning
    // candidate's first row in a multi-candidate chunk, 0 linear): pairs are
    // exactly (emitted_j, h_row_{row0+j}). Must run before the hybrid partial-accept re-forward overwrites the hidden buffer.
    Tensor h = executor_->view_mtp_hidden(row0 + emitted);
    if (h.data == nullptr) {
        mtp_unbind_("no hidden view after verify");
        return;
    }
    const void* rows = static_cast<const char*>(h.data) +
                       (static_cast<size_t>(row0) * imp::mtp_h_prev_cols(*ws) * sizeof(__half));
    const int32_t* toks = req.output_tokens.data() + req.output_tokens.size() - emitted;
    if (!mtp_feed_pairs_(toks, rows, emitted, /*chain_after=*/req.status == RequestStatus::DECODING))
        mtp_unbind_("verify feed failed (kv cap or forward error)");
}

// Bind req's first prefill chunk (at `offset`) to a head KV slot, resuming over the longest common
// prefix of the slot's fed history and this prompt (multi-turn agent loops re-send the prior turn
// verbatim, so the cache carries over). false = req drafts nothing; no slot's history is touched.
bool Engine::mtp_bind_prompt_(const Request& req, int offset) {
    auto* ws = mtp_ws_storage_.get();
    const auto& in = req.input_tokens;
    if (!mtp_slot_available(mtp_active_, mtp_pool_, mtp_pool_.bind_cap, req.id))
        return false;  // another request holds the head
    // Past the head KV capacity the feed fails anyway: keep every slot's history.
    const bool too_long = ws->max_seq_len > 0 && static_cast<int>(in.size()) + mtp_spec_k_ > ws->max_seq_len;
    const int slot = too_long ? -1 : mtp_acquire_slot_(req, offset);
    if (slot < 0) {
        // No slot covers the prefix-cache resume (or none is free): draft nothing, wipe no history.
        if (mtp_pool_.declined_req != req.id && (offset > 0 || too_long))
            IMP_LOG_INFO("mtp-spec: no head cache covers req %d's prefix-cache resume at %d (%s)", req.id,
                         offset,
                         too_long ? "prompt exceeds the head KV capacity" : "no slot history reaches it");
        mtp_pool_.declined_req = req.id;
        return false;
    }
    mtp_activate_(-1);
    imp::mtp_select_slot(*ws, slot);
    mtp_active_.history = mtp_pool_.slot_history[static_cast<size_t>(slot)];
    size_t L = 0;
    while (L < mtp_active_.history.size() && L < in.size() && mtp_active_.history[L] == in[L])
        ++L;
    const int keep_pairs = std::max(0, static_cast<int>(L) - 1);
    ws->mtp_pos = std::min(ws->mtp_pos, keep_pairs);
    if (mtp_active_.history.size() > static_cast<size_t>(ws->mtp_pos) + 1)
        mtp_active_.history.resize(static_cast<size_t>(ws->mtp_pos) + 1);
    mtp_active_.req = req.id;
    mtp_active_.stale_logged = false;
    mtp_active_.pending.clear();
    mtp_active_.chains.clear();
    mtp_active_.draft_ctx = -1;
    mtp_active_.econ_verifies = 0;
    mtp_active_.econ_emitted = 0;
    mtp_active_.econ_rows = 0;
    mtp_active_.k_live = mtp_spec_k_;
    mtp_active_.carry_n = 0;
    if (!mtp_active_.history.empty() && ws->mtp_pos > 0)
        return true;
    // Nothing usable carried over, restart the cache. Pair 0 needs
    // h_0, so the prompt must actually be forwarded from position 0.
    ws->mtp_pos = 0;
    mtp_active_.history.clear();
    if (offset == 0 && !in.empty()) {
        mtp_active_.history.push_back(in[0]);
        return true;
    }
    mtp_unbind_("prefix-cache gap without matching MTP history");
    return false;
}

void Engine::mtp_prefill_feed_chunk(const Request& req, int offset, int chunk_len,
                                    int next_token) {
    if (!mtp_feed_wanted_(req))
        return;
    if (!model_ || !model_->mtp_ || !model_->mtp_->loaded)
        return;
    if (chunk_len <= 0)
        return;
    auto* ws = mtp_ws_storage_.get();
    const auto& in = req.input_tokens;

    // A later chunk of a parked request only swaps its binding back in.
    if (mtp_active_.req != req.id && mtp_bound(mtp_active_, mtp_pool_, req.id))
        mtp_activate_(req.id);
    else if (mtp_active_.req != req.id && !mtp_bind_prompt_(req, offset))
        return;

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

    Tensor h = executor_->view_mtp_hidden(chunk_len);
    if (h.data == nullptr) {
        mtp_unbind_("no hidden view during prefill");
        return;
    }
    const char* rows = static_cast<const char*>(h.data) +
                       (static_cast<size_t>(start - offset) * imp::mtp_h_prev_cols(*ws) * sizeof(__half));
    const bool last_chunk = next_token >= 0;
    if (!mtp_feed_pairs_(toks.data(), rows, static_cast<int>(toks.size()),
                         /*chain_after=*/last_chunk))
        mtp_unbind_("prefill feed failed (kv cap or forward error)");
}

// MLP dims come from the HEAD tensors, not the main-model config: the dense 27B checkpoint pairs a
// MoE-free MTP head with a plain SwiGLU MLP (mapped onto shared_expert fields); the 35B head's expert
// d_ff differs from the main model's. false = qwen4_exp head without device FP8 experts or hc width.
static bool mtp_head_mlp_dims(const MtpHead& head, int n_experts_active, MtpWorkspaceShape& s) {
    // Two expert layouts. Qwen packs gate+up into one 3-D stack, so its per-expert
    // d_ff is half the packed row count; Nemotron keeps per-expert 2-D weights and
    // has no gate half, so d_ff is the up_proj row count as-is.
    const bool head_moe_packed = head.router.data != nullptr && head.experts_gate_up_packed.data != nullptr;
    const bool head_moe_per_expert = head.router.data != nullptr && !head.experts_up.empty() &&
                                     head.experts_up[0].data != nullptr;
    // Qwen4Exp: per-expert FP8 gate/up/down, uploaded behind device pointer tables.
    const bool head_moe_fp8 = head.router.data != nullptr && head.fp8_gate_up_tab != nullptr &&
                              !head.experts_fp8.empty();
    const bool head_moe = head_moe_packed || head_moe_per_expert || head_moe_fp8;
    s.n_experts = head_moe ? static_cast<int>(head.router.shape[0]) : 0;
    s.top_k = head_moe ? n_experts_active : 0;
    s.expert_d_ff = head_moe_packed       ? static_cast<int>(head.experts_gate_up_packed.shape[1]) / 2
                    : head_moe_per_expert ? static_cast<int>(head.experts_up[0].shape[0])
                    : head_moe_fp8        ? static_cast<int>(head.experts_fp8[0].gate_proj.shape[0])
                                          : 0;
    if (head.layout == MtpLayout::Qwen4Exp && (!head_moe_fp8 || head.hc_count <= 0))
        return false;
    s.hc_count = head.hc_count;
    s.hc_lowrank = head.hc_count > 0 ? static_cast<int>(head.attn_hc.mix_down.shape[0]) : 0;
    // The shared expert is sized off gate_proj on the Qwen layout; the Nemotron
    // one is non-gated, so up_proj is the only tensor that carries the width.
    s.shared_d_ff = head.shared_expert_gate_proj.data != nullptr
                        ? static_cast<int>(head.shared_expert_gate_proj.shape[0])
                    : head.shared_expert_up_proj.data != nullptr
                        ? static_cast<int>(head.shared_expert_up_proj.shape[0])
                        : 0;
    return true;
}

// MTP attention dims derive from the head's q_proj/v_proj shapes (the MTP
// attention config differs from the main model: Qwen3.6 doubles Q per head
// for attn_output_gate). q_proj shape [2*num_heads*head_dim, hidden_dim];
// v_proj shape [num_kv_heads*head_dim, hidden_dim]; head_dim comes from the main model, head counts are
// back-computed.
static void mtp_head_attn_dims(const MtpHead& head, const ModelConfig& cfg, MtpWorkspaceShape& s) {
    if (head.q_proj.data == nullptr || head.v_proj.data == nullptr || cfg.head_dim <= 0)
        return;
    s.head_dim = cfg.head_dim;
    // Qwen3.6 doubles Q per head for attn_output_gate; Nemotron does not.
    // Assuming the doubling unconditionally can halve the wrong model's
    // head count, which still runs and drafts from the wrong half of Q
    // (nothing ever accepted). Decide from the tensor: match the main model's head count first, fall back to
    // the gated reading only if that fails.
    const int plain = static_cast<int>(head.q_proj.shape[0]) / s.head_dim;
    const bool gated = (plain != cfg.n_heads) && (plain % 2 == 0) && (plain / 2 == cfg.n_heads);
    s.num_heads = gated ? plain / 2 : plain;
    s.num_kv_heads = static_cast<int>(head.v_proj.shape[0]) / s.head_dim;
}

// Every input the head's workspace is sized from; enable_mtp_spec_decode allocates it and the capacity
// plan prices it (mtp_workspace_plan_bytes_), so the two cannot drift. false = see mtp_head_mlp_dims.
bool Engine::mtp_workspace_shape_(MtpWorkspaceShape& s) const {
    s = MtpWorkspaceShape{};
    s.hidden_dim = model_->config_.d_model;
    s.vocab_size = model_->config_.vocab_size;
    const auto& head = *model_->mtp_;
    if (!mtp_head_mlp_dims(head, model_->config_.n_experts_active, s))
        return false;
    mtp_head_attn_dims(head, model_->config_, s);

    // MTP KV-cache capacity: cap at the smaller of model's max_seq_len and 16K
    // (Phase 2.2.Attn+KV budget, ~16 MiB each for K and V at Qwen3.6 dims).
    constexpr int kMtpKvCap = 16384;
    s.max_seq_len = std::min(model_->config_.max_seq_len, kMtpKvCap);
    if (s.max_seq_len <= 0)
        s.max_seq_len = kMtpKvCap;

    // Batched verify: one KV slot per batch slot, else the single one. With
    // many slots the per-slot capacity comes from a 1 GiB budget (floor 2048
    // rows); a request past its slot's capacity stops drafting, verify continues on its own rows.
    s.n_kv_slots = mtp_draft_kv_slots(runtime_config_, model_.get(), config_.max_batch_size);
    if (s.n_kv_slots > 1 && s.num_kv_heads > 0 && s.head_dim > 0) {
        const size_t row_bytes = 2ull * s.num_kv_heads * s.head_dim * sizeof(__half);
        const size_t budget = 1ull << 30;
        const int fit = static_cast<int>(budget / (row_bytes * static_cast<size_t>(s.n_kv_slots)));
        s.max_seq_len = std::clamp(fit, 2048, s.max_seq_len);
    }
    // speculative.mtp_carry scratch (2.5 MiB per slot on Qwen3.8).
    s.carry_rows = mtp_carry_rows(runtime_config_, model_.get(), config_.max_batch_size);
    // Post-norm feed scratch (diagnostics.mtp_prenorm_h): sized to the executor's token cap, the
    // widest feed a prefill chunk can hand mtp_feed_pairs_.
    if (runtime_config_.diagnostics.mtp_prenorm_h && head.hc_count == 0)
        s.prenorm_rows = std::max(1, executor_ ? executor_->max_tokens() : s.max_seq_len);
    return true;
}

size_t Engine::mtp_workspace_plan_bytes_() const {
    // The resolved depth (server and imp-cli auto re-publish it before Engine::init) and an uploaded
    // head: init() then takes the workspace itself (take_planned_mtp_workspace_).
    if (runtime_config_.speculative.mtp_k <= 0 || !model_ || !model_->mtp_ || !model_->mtp_->loaded)
        return 0;
    MtpWorkspaceShape s;
    return mtp_workspace_shape_(s) ? mtp_workspace_bytes(s) : 0;
}

void Engine::take_planned_mtp_workspace_() {
    if (mtp_workspace_planned_bytes_ == 0 || mtp_ws_storage_ != nullptr)
        return;
    if (!enable_mtp_spec_decode(runtime_config_.speculative.mtp_k))
        IMP_LOG_WARN("MTP workspace: the plan reserved %.1f MiB but enable failed; the server retries once",
                     static_cast<double>(mtp_workspace_planned_bytes_) / (1024.0 * 1024.0));
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
    if (!model_->mtp_ || !model_->mtp_->loaded) {
        IMP_LOG_ERROR("enable_mtp_spec_decode: model has no MTP head loaded");
        return false;
    }
    if (mtp_ws_storage_ != nullptr) {
        // init() took the planned workspace at this depth already; the server's call lands here.
        if (k != mtp_spec_k_)
            IMP_LOG_WARN("enable_mtp_spec_decode: already enabled, k=%d -> %d", mtp_spec_k_, k);
        mtp_spec_k_ = k;
        return true;
    }
    MtpWorkspaceShape shape;
    if (!mtp_workspace_shape_(shape)) {
        IMP_LOG_ERROR("enable_mtp_spec_decode: qwen4_exp head without device FP8 experts or hc width");
        return false;
    }
    const int hidden_dim = shape.hidden_dim;
    const int vocab_size = shape.vocab_size;
    const int n_experts = shape.n_experts;
    const int top_k = shape.top_k;
    const int expert_d_ff = shape.expert_d_ff;
    const int shared_d_ff = shape.shared_d_ff;
    const int mtp_num_heads = shape.num_heads;
    const int mtp_num_kv_heads = shape.num_kv_heads;
    const int mtp_head_dim = shape.head_dim;
    const int mtp_kv_max = shape.max_seq_len;
    auto ws = std::make_unique<imp::MtpDraftWorkspace>();
    // Taken inside Engine::init when the plan reserved it (take_planned_mtp_workspace_); an API caller
    // enabling later lands after set_alloc_phase(Serving): init traffic on the wrong side of the
    // boundary, labelled (not exempted) via AllocPhaseScope so `make check-alloc-interpose` sees it.
    AllocPhaseScope mtp_alloc_phase(AllocPhase::Planning, "mtp workspace");
    if (!imp::mtp_workspace_allocate(*ws, hidden_dim, vocab_size, n_experts, top_k, expert_d_ff, shared_d_ff,
                                     mtp_num_heads, mtp_num_kv_heads, mtp_head_dim, mtp_kv_max,
                                     shape.n_kv_slots, shape.hc_count, shape.hc_lowrank)) {
        IMP_LOG_ERROR("enable_mtp_spec_decode: workspace alloc failed");
        return false;
    }
    // speculative.mtp_carry scratch; refused = one binding, as without it.
    (void)imp::mtp_carry_allocate(*ws, shape.carry_rows);
    // Same labelled init scope: the hybrid verify scratch, unless init's prewarm already took it.
    prewarm_spec_state_scratch_();
    if (shape.prenorm_rows > 0 && !imp::mtp_prenorm_allocate(*ws, shape.prenorm_rows))
        IMP_LOG_WARN(
            "enable_mtp_spec_decode: post-norm feed scratch (%d rows, %.1f MiB) failed; "
            "feeds will refuse and drafting unbinds",
            shape.prenorm_rows,
            static_cast<double>(shape.prenorm_rows) * hidden_dim * sizeof(__half) / 1048576.0);
    {
        // Reconciliation: what the allocator took vs the plan's mirror of it.
        const size_t want = mtp_workspace_bytes(shape);
        const double mib = 1024.0 * 1024.0;
        if (ws->device_bytes != want)
            IMP_LOG_WARN(
                "MTP workspace: allocated %.2f MiB, mtp_workspace_bytes prices %.2f MiB (plan row %.2f MiB)",
                ws->device_bytes / mib, want / mib, mtp_workspace_planned_bytes_ / mib);
        else
            IMP_LOG_INFO("MTP workspace: %.2f MiB allocated = %s", ws->device_bytes / mib,
                         mtp_workspace_planned_bytes_ == want ? "the plan row"
                                                              : "mtp_workspace_bytes (not planned)");
    }
    // Configure RoPE for the MTP attention (Phase 2.2.Attn+RoPE).
    // Qwen3.5/3.6 uses partial rope (factor 0.25 -> rope_dim=64 of head_dim=256),
    // theta from config (10M for long-context), NeoX-style.
    ws->rope_theta = model_->config_.rope_theta;
    ws->rope_neox = model_->config_.rope_neox;
    ws->rms_norm_eps = model_->config_.rms_norm_eps;
    ws->rope_dim = (model_->config_.rope_dim > 0) ? model_->config_.rope_dim : mtp_head_dim;
    // mrope section split: read from config when the checkpoint carries one
    // (Qwen3-VL: [24,20,20]), else the Qwen3.6 constant below. For text-only
    // generation all 3 positions are equal (split matters only for multimodal tokens).
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
    // RoPE scaling mirrors the main forward so the drafter rotates Q/K the
    // same way as the verifier at extended positions (#897); without it a
    // rope-scaled model's draft head diverges and acceptance degrades with position.
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
    // LongRoPE (Phi-family) isn't plumbed into the single-token MTP kernel: no
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
    // weight_offset matches the main model's rmsnorm calls (norm_weight_offset
    // from ModelConfig): 0.0 for Qwen3.5/3.6 (the +1 gamma=1+W is already baked
    // in at upload, see upload_mtp_weights in weight_upload.cpp), 1.0 for Gemma-3. Don't double-apply.
    ws->arch_norm_offset = model_->config_.norm_weight_offset;

    mtp_ws_storage_.reset(ws.release());
    const imp::MtpDraftWorkspace& live = *mtp_ws_storage_;  // ws is null after release()
    mtp_spec_k_ = k;
    mtp_pool_.bind_cap = mtp_bind_cap(runtime_config_, model_.get(), config_.max_batch_size,
                                      live.carry_rows_cap > 0);
    mtp_pool_.binds.clear();
    mtp_pool_.free_slots.clear();
    mtp_pool_.slot_history.clear();
    mtp_pool_.initialized = false;
    IMP_LOG_INFO(
        "MTP spec-decode enabled (k=%d, hidden=%d, vocab=%d, experts=%d/top%d, d_ff_e=%d, "
        "d_ff_shared=%d, num_heads=%d/%d, head_dim=%d, kv_cap=%d x %d slot(s) = %.1f MiB K+V, "
        "rope=%g/%d/%s, mrope=[%d,%d,%d])",
        k, hidden_dim, vocab_size, n_experts, top_k, expert_d_ff, shared_d_ff, mtp_num_heads,
        mtp_num_kv_heads, mtp_head_dim, mtp_kv_max, live.n_kv_slots,
        2.0 * static_cast<double>(live.kv_slot_elems) * live.n_kv_slots * sizeof(__half) / (1024.0 * 1024.0),
        live.rope_theta, live.rope_dim, live.rope_neox ? "neox" : "interleaved", live.mrope_sec0,
        live.mrope_sec1, live.mrope_sec2);
    return true;
}
void Engine::mtp_accuracy_reset() noexcept {
    mtp_accuracy_ = {};
    mtp_pool_.pending_prediction = -1;
    mtp_pending_chain_.clear();
    mtp_chain_accept_.clear();
    mtp_chain_accept_w_.clear();
    mtp_active_.req = -1;
    mtp_active_.history.clear();
    mtp_active_.pending.clear();
    mtp_active_.draft_ctx = -1;
    mtp_active_.econ_verifies = 0;
    mtp_active_.econ_emitted = 0;
    mtp_active_.econ_rows = 0;
    mtp_active_.k_live = mtp_spec_k_;
    mtp_active_.carry_n = 0;
    mtp_pool_.binds.clear();
    mtp_pool_.free_slots.clear();
    mtp_pool_.slot_history.clear();
    mtp_pool_.initialized = false;
    if (mtp_ws_storage_) {
        auto* ws = mtp_ws_storage_.get();
        imp::mtp_kv_reset(*ws);
        imp::mtp_select_slot(*ws, 0);
    }
}
const imp::NvFP4QuantResult* Engine::mtp_draft_lm_head_nvfp4_(imp::NvFP4QuantResult& view) const {
    return runtime_config_.speculative.mtp_nvfp4_head && executor_ && executor_->lm_head_nvfp4_view(view)
               ? &view
               : nullptr;
}

bool Engine::mtp_draft_one(int prev_token_id, const void* d_h_prev, int hidden_dim, int vocab_size,
                           int* out_token_id, int* out_topk_ids, int top_w, const int32_t* d_prev_token,
                           int32_t* d_out_token) {
    if (mtp_ws_storage_ == nullptr) {
        IMP_LOG_ERROR("mtp_draft_one: spec-decode not enabled");
        return false;
    }
    if (!model_ || !model_->mtp_ || !model_->mtp_->loaded) {
        IMP_LOG_ERROR("mtp_draft_one: MTP head not loaded");
        return false;
    }
    auto* ws = mtp_ws_storage_.get();
    // Chain lm_head via the NVFP4 decode cache when available: full-vocab FP16
    // GEMV is the dominant per-draft cost (#847 lever 3). Draft-only precision
    // (verify stays lossless); without a cache entry: the FP8 head, a dp4a GEMV on a GGUF head, else FP16
    // GEMV.
    imp::NvFP4QuantResult lm_nvfp4;
    const imp::NvFP4QuantResult* lm_nvfp4_p = mtp_draft_lm_head_nvfp4_(lm_nvfp4);
    if (lm_nvfp4_p != nullptr) {
        static bool logged = false;  // once-per-process path attribution
        if (!logged) {
            logged = true;
            IMP_LOG_INFO("MTP chain lm_head: NVFP4 decode-cache view engaged (N=%lld K=%lld)",
                         static_cast<long long>(lm_nvfp4.N), static_cast<long long>(lm_nvfp4.K));
        }
    }
    const void* lm_fp8 = nullptr;
    const float* lm_fp8_scales = nullptr;
    // An NVFP4 view wins over the FP8 head for drafts (draft-only precision, verification reads FP8).
    if (executor_ && lm_nvfp4_p == nullptr)
        (void)executor_->lm_head_fp8_view(lm_fp8, lm_fp8_scales);  // false: no FP8 head, pointers stay null
    return imp::mtp_draft_step(prev_token_id, d_h_prev, *model_->mtp_, model_->tok_emb_, model_->out_proj_,
                               *ws, hidden_dim, vocab_size, out_token_id, decode_stream(), out_topk_ids,
                               top_w, lm_nvfp4_p, d_prev_token, d_out_token, lm_fp8, lm_fp8_scales);
}

}  // namespace imp
