// Cross-sequence ragged prefill (runtime.prefill_batch, roadmap 0(d)).
//
// A burst of K short prompts otherwise prefills ONE sequence per forward —
// launch-bound (64 layers x ~100 launches at M~120), measured at ~25% of a
// 32-stream burst wave's wall. This path concatenates the next chunk of each
// eligible request into ONE ragged forward: GEMMs/norms/elementwise/RoPE run
// over the concatenated rows, attention and the GDN conv loop per sequence
// inside the executor, the GDN scan batches sequences via the row-offset
// table (gdn.h ragged contract, shipped in #1779).
//
// Scope (phase 1): plain text generation requests only. Vision, embeddings,
// rerank scoring, logprobs and constrained decoding stay serial per request;
// Mamba2 and MLA models, MTP, perplexity capture, SWA sizing, residual KV and
// the fp32_scan/ref_kernel GDN routes disable the path entirely.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "exec/inference_state.h"
#include "memory/kv_cache.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

namespace imp {

bool Engine::prefill_ragged_enabled_() {
    if (!runtime_config_.runtime.prefill_batch)
        return false;
    if (mtp_spec_decode_enabled() || ppl_capture_.active || swa_sizing_active_)
        return false;
    if (kv_manager_ && kv_manager_->residual_enabled())
        return false;
    // The CLI's engine-global image has no per-request owner — serial only.
    if (vision_.has_input())
        return false;
    if (prefill_ragged_model_ok_ < 0) {
        bool ok = !model_->config_.is_mla();
        bool has_gdn = false;
        for (int i = 0; ok && i < model_->config_.n_layers; ++i) {
            const auto& ly = model_->layer(i);
            if (ly.gdn_gate.data != nullptr)
                has_gdn = true;
            else if (ly.ssm_in.data != nullptr)
                ok = false;  // Mamba2 (run_ssm) has no ragged path
        }
        // GDN ragged rides the fused batched scan; the fp32out/ref routes
        // have no ragged entry (executor throws as defense in depth).
        if (ok && has_gdn && (runtime_config_.gdn.fp32_scan || runtime_config_.gdn.ref_kernel))
            ok = false;
        prefill_ragged_model_ok_ = ok ? 1 : 0;
        if (!ok)
            IMP_LOG_INFO(
                "prefill_batch: model out of scope (Mamba2/MLA or GDN fp32_scan/ref_kernel) "
                "— serial prefill");
    }
    return prefill_ragged_model_ok_ == 1;
}

bool Engine::prefill_ragged_req_ok_(const Request& req) const {
    const bool has_vision = req.image || !req.qwen_patches.empty() || req.vision_emb ||
                            req.n_vision_tokens > 0;
    const bool wants_constraints = req.json_mode || !req.json_schema.empty() ||
                                   !req.tool_constraint_tools.empty() || !req.regex_pattern.empty() ||
                                   !req.grammar.empty();
    return !has_vision && !req.embedding_request && req.score_token_ids.empty() && !req.logprobs &&
           !wants_constraints;
}

void Engine::step_prefill_ragged_(std::vector<std::shared_ptr<Request>>& reqs, int effective_chunk,
                                  cudaStream_t stream) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    struct RaggedSeq {
        std::shared_ptr<Request> req;
        int offset = 0;
        int chunk_len = 0;
        int ctx_len = 0;
        bool is_last = false;
        int snap_end = 0;
        int slot = 0;
    };
    std::vector<RaggedSeq> geoms;
    geoms.reserve(reqs.size());

    InferenceState state;
    int rows_left = std::min(effective_chunk, executor_->max_tokens());
    size_t deferred = 0;

    for (auto& req : reqs) {
        if (rows_left <= 0) {
            deferred++;  // stays in sched_prefill_batch_ for the next step
            continue;
        }
        const int total_input = static_cast<int>(req->input_tokens.size());
        int offset = req->prefill_offset;

        // Same guards as step_prefill_one: out-of-scope archs cannot chunk,
        // and cuBLAS-only attention configs must fit the S-matrix.
        if (offset == 0 && total_input > effective_chunk && !supports_chunked_prefill_()) {
            IMP_LOG_ERROR(
                "Prompt has %d tokens but max_tokens=%d on hybrid/out-of-scope arch — "
                "chunked prefill not supported. Cancelling request %d.",
                total_input, effective_chunk, req->id);
            req->status = RequestStatus::CANCELLED;
            continue;
        }
        int eff = effective_chunk;
        if (executor_) {
            if (offset == 0 && total_input > kv_bs) {
                int last_off = ((total_input - 1) / kv_bs) * kv_bs;
                if (executor_->max_safe_prefill_chunk(last_off, kv_bs, kv_bs) < kv_bs) {
                    IMP_LOG_ERROR(
                        "Prompt (%d tokens) exceeds the chunked-attention workspace for this model "
                        "(S-matrix cap %d) — cancelling request %d.",
                        total_input, executor_->attn_scores_cap(), req->id);
                    req->status = RequestStatus::CANCELLED;
                    continue;
                }
            }
            int max_chunk = executor_->max_safe_prefill_chunk(offset, eff, kv_bs);
            if (max_chunk > 0 && eff > max_chunk)
                eff = max_chunk;
        }

        int chunk_len = total_input - offset;
        bool is_last = true;
        if (chunk_len > eff) {
            chunk_len = eff;
            is_last = false;
        }
        const int snap_end = snapshot_end_(*req);
        if (snap_end > offset && snap_end < offset + chunk_len) {
            chunk_len = snap_end - offset;
            is_last = false;
        }
        if (chunk_len > rows_left) {
            chunk_len = rows_left;
            is_last = false;
        }
        int ctx_len = offset + chunk_len;

        if (!prefill_allocate_kv_blocks_(req, kv_bs, total_input, eff, offset, chunk_len, is_last, ctx_len,
                                         stream)) {
            continue;  // cancelled; req->status already set
        }
        // A prefix-cache hit advanced offset and recomputed chunk_len — the
        // row budget still binds.
        if (chunk_len > rows_left) {
            chunk_len = rows_left;
            is_last = false;
            ctx_len = offset + chunk_len;
        }
        if (chunk_len <= 0) {
            deferred++;
            continue;
        }
        rows_left -= chunk_len;

        // Reset (or snapshot-restore) the recurrent state on the first chunk;
        // later chunks carry it forward. Captures the slot for the batched
        // scan / per-seq conv.
        fill_recurrent_state(*req, state, /*reset=*/(offset == req->cached_tokens), stream);

        geoms.push_back({req, offset, chunk_len, ctx_len, is_last, snap_end, state.ssm_seq_id});
    }

    if (geoms.empty())
        return;
    if (geoms.size() == 1) {
        // A single survivor gains nothing from the ragged plumbing — run it
        // through the serial path (the KV alloc and state reset above are
        // idempotent for the re-entry).
        step_prefill_one(geoms[0].req, effective_chunk, stream);
        return;
    }

    const int n_seq = static_cast<int>(geoms.size());
    int total = 0;
    for (const auto& g : geoms)
        total += g.chunk_len;

    // Host-side concatenated metadata. Pageable H2D sources are captured at
    // enqueue time by CUDA semantics, so plain vectors are safe here.
    std::vector<int32_t> h_tok(static_cast<size_t>(total));
    std::vector<int> h_pos(static_cast<size_t>(total));
    std::vector<int> h_soff(static_cast<size_t>(n_seq) + 1, 0);
    std::vector<int> h_qoff(static_cast<size_t>(n_seq));
    std::vector<int> h_slots(static_cast<size_t>(n_seq));
    std::vector<int> h_ctx(static_cast<size_t>(n_seq));
    size_t max_blocks = 0;
    for (const auto& g : geoms)
        max_blocks = std::max(max_blocks, kv_manager_->block_table(g.req->id).size());
    std::vector<int> h_bt(static_cast<size_t>(n_seq) * max_blocks, 0);

    {
        int col = 0;
        for (int s = 0; s < n_seq; ++s) {
            const auto& g = geoms[s];
            h_soff[s] = col;
            h_qoff[s] = g.offset;
            h_slots[s] = g.slot;
            h_ctx[s] = g.ctx_len;
            const auto& bt = kv_manager_->block_table(g.req->id);
            std::copy(bt.begin(), bt.end(), h_bt.begin() + static_cast<size_t>(s) * max_blocks);
            for (int i = 0; i < g.chunk_len; ++i) {
                h_tok[static_cast<size_t>(col) + i] = g.req->input_tokens[g.offset + i];
                h_pos[static_cast<size_t>(col) + i] = g.offset + i;
            }
            col += g.chunk_len;
        }
        h_soff[n_seq] = col;
    }

    // Device metadata: per-step stream-ordered allocs. Prefill is never
    // graph-captured, so the pool amortises these (same acknowledged
    // exception as the chunked-prefill gather scratch).
    int32_t* d_tok = nullptr;
    int* d_pos = nullptr;
    int* d_bt = nullptr;
    int* d_ctx = nullptr;
    int* d_soff = nullptr;
    int* d_slots = nullptr;
    auto cleanup = [&]() {
        if (d_tok)
            cudaFreeAsync(d_tok, stream);
        if (d_pos)
            cudaFreeAsync(d_pos, stream);
        if (d_bt)
            cudaFreeAsync(d_bt, stream);
        if (d_ctx)
            cudaFreeAsync(d_ctx, stream);
        if (d_soff)
            cudaFreeAsync(d_soff, stream);
        if (d_slots)
            cudaFreeAsync(d_slots, stream);
    };
    bool alloc_ok = cudaMallocAsync(&d_tok, h_tok.size() * sizeof(int32_t), stream) == cudaSuccess &&
                    cudaMallocAsync(&d_pos, h_pos.size() * sizeof(int), stream) == cudaSuccess &&
                    cudaMallocAsync(&d_bt, std::max<size_t>(h_bt.size(), 1) * sizeof(int), stream) ==
                        cudaSuccess &&
                    cudaMallocAsync(&d_ctx, h_ctx.size() * sizeof(int), stream) == cudaSuccess &&
                    cudaMallocAsync(&d_soff, h_soff.size() * sizeof(int), stream) == cudaSuccess &&
                    cudaMallocAsync(&d_slots, h_slots.size() * sizeof(int), stream) == cudaSuccess;
    if (!alloc_ok) {
        IMP_LOG_ERROR("prefill_batch: metadata allocation failed — cancelling %d requests", n_seq);
        cleanup();
        for (auto& g : geoms) {
            cancel_sequence_(g.req);
            g.req->status = RequestStatus::CANCELLED;
        }
        return;
    }
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_tok, h_tok.data(), h_tok.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_pos, h_pos.data(), h_pos.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    if (!h_bt.empty())
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(d_bt, h_bt.data(), h_bt.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_ctx, h_ctx.data(), h_ctx.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_soff, h_soff.data(), h_soff.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_slots, h_slots.data(), h_slots.size() * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));

    state.token_ids = d_tok;
    state.positions = d_pos;
    state.n_tokens = total;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_bt;
    state.context_lens = d_ctx;
    state.max_context_len = *std::max_element(h_ctx.begin(), h_ctx.end());
    state.n_sequences = n_seq;
    state.max_blocks_per_seq = static_cast<int>(max_blocks);
    state.is_prefill = true;
    state.prefill_offset = 0;  // per-seq offsets carry the truth (h_seq_q_offsets)
    state.kv_manager = kv_manager_.get();
    state.seq_offsets = d_soff;
    state.h_seq_offsets = h_soff.data();
    state.h_seq_q_offsets = h_qoff.data();
    if (ssm_state_) {
        state.h_ssm_slots = h_slots.data();
        state.ssm_seq_slots = d_slots;
        state.ssm_n_seq = n_seq;
        state.ssm_seq_id = h_slots[0];
    }
    // M-RoPE models bind the concatenated per-request axis rows; text-only
    // requests produce the plain ascending positions, bit-identical to the
    // single-axis path.
    {
        std::vector<std::shared_ptr<Request>> rr;
        std::vector<int> ro, rl;
        rr.reserve(geoms.size());
        for (const auto& g : geoms) {
            rr.push_back(g.req);
            ro.push_back(g.offset);
            rl.push_back(g.chunk_len);
        }
        bind_mrope_prefill_ragged_(state, rr, ro, rl, total, stream);
    }

    if (executor_->has_decode_workspace())
        executor_->use_workspace(0);
    (void)executor_->resize_workspace(total, stream);

    IMP_LOG_DEBUG("Ragged prefill: %d seqs, %d rows (chunk cap %d)", n_seq, total, effective_chunk);

    Tensor logits_out;
    executor_->forward_logits(state, logits_out, stream);

    // Per-request epilogue. Logits row s belongs to sequence s (forward_logits
    // compacted each sequence's last row).
    for (int s = 0; s < n_seq; ++s) {
        auto& g = geoms[s];
        auto& req = g.req;
        req->prefill_offset = g.offset + g.chunk_len;
        IMP_LOG_DEBUG("Ragged prefill: req %d chunk [%d, %d) of %d", req->id, g.offset, req->prefill_offset,
                      static_cast<int>(req->input_tokens.size()));
        if (g.snap_end > 0 && req->prefill_offset == g.snap_end) {
            maybe_save_recurrent_snapshot_(*req, g.snap_end, stream);
            maybe_save_swa_snapshot_(*req, g.snap_end, stream);
        }
        if (!g.is_last)
            continue;

        InferenceState sst;
        sst.is_prefill = true;
        sst.n_sequences = 1;
        fill_sampling_params(*req, sst);
        upload_penalties(*req, sst, stream);
        Tensor row = logits_out.slice(s, s + 1);
        auto sampled = executor_->sample_from_logits(row, sst, stream);
        const int32_t next_token = sampled[0];
        if (req->mirostat == 2)
            req->mirostat_mu = sst.mirostat_mu;

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        } else {
            req->status = RequestStatus::DECODING;
            if (kv_manager_->prefix_caching_enabled())
                kv_manager_->register_block_hashes(req->id, req->input_tokens, req->vision_content_hash);
        }
    }

    if (deferred > 0)
        IMP_LOG_DEBUG("Ragged prefill: %zu requests deferred to the next step (row cap)", deferred);
    cleanup();
}

}  // namespace imp
