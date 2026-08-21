// Device-buffer lifecycle for the speculative paths, split out of
// engine_spec_ngram.cpp on 2026-08-21.
//
// Split on RESPONSIBILITY, not size. That file held two things: the buffers the
// speculative paths allocate and release, and the policy and execution that use
// them. The trigger was the file reaching its hard-review ceiling exactly (800
// code LOC) so that a one-line feature could not be added without exceeding it -
// which is the size gate reporting a conflation rather than a length.
//
// Pure move: every function below is byte-identical to its previous form.

#include "compute/json_constrain.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "runtime/engine.h"
#include "runtime/spec_trace.h"
#include "runtime/ngram_draft.h"
#include "runtime/request.h"
#include "runtime/suffix_draft.h"
#include "compute/rowwise_topm.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <vector>

namespace imp {

bool Engine::ensure_spec_buffers_(int chunk_cap, int max_blocks) {
    if (spec_chunk_cap_ >= chunk_cap && spec_block_table_cap_ >= max_blocks)
        return true;
    free_spec_buffers_();
    // #1055 consolidated staging: tokens/positions/row-ctx-lens (chunk_cap
    // each) + {ctx_len, past_len, chunk_len} live in ONE device block with a
    // PINNED host twin — a single small H2D per verify step instead of six
    // (pageable-source async copies stage through a driver buffer on WSL2).
    // The captured graphs bake the sub-pointers; the block is allocated once
    // per capacity, so they stay stable. Same trick for argmax+topm D2H.
    // [tokens | positions | row_ctx_lens | ctx_len, past_len, chunk_len, snap_n]
    const size_t stage_ints = 3ull * chunk_cap + 4;
    const size_t out_ints = static_cast<size_t>(chunk_cap) * (1 + kRowwiseTopMMax);
    // T5b for the pinned twins (memory/host_pinned.h). Wrapped in a lambda so the
    // && chain still SHORT-CIRCUITS: acquiring them up front would allocate host
    // memory even when an earlier device allocation had already failed.
    auto pin = [](PinnedBuffer& b, size_t bytes) {
        b = PinnedBuffer::acquire(cuda_host_pinned_allocator(), bytes);
        return !b.empty();
    };
    bool ok = cudaMalloc(&d_spec_stage_, stage_ints * sizeof(int32_t)) == cudaSuccess &&
              pin(h_spec_stage_, stage_ints * sizeof(int32_t)) &&
              cudaMalloc(&d_spec_argmax_, out_ints * sizeof(int32_t)) == cudaSuccess &&
              pin(h_spec_argmax_, out_ints * sizeof(int32_t)) &&
              cudaMalloc(&d_spec_block_table_, max_blocks * sizeof(int)) == cudaSuccess &&
              // SWA-group mirror (kv_cache.swa_sizing): same capacity as the
              // main table. Allocated unconditionally so a mid-session gate
              // flip can't leave it null; tiny (max_blocks ints).
              cudaMalloc(&d_spec_block_table_swa_, max_blocks * sizeof(int)) == cudaSuccess &&
              // #964 decode-attention verify route staging (see engine.h).
              cudaMalloc(&d_spec_row_block_tables_,
                         static_cast<size_t>(chunk_cap) * max_blocks * sizeof(int)) ==
                  cudaSuccess &&
              pin(h_spec_row_tables_pinned_,
                  static_cast<size_t>(chunk_cap) * max_blocks * sizeof(int32_t));
    if (!ok) {
        IMP_LOG_WARN("spec-ngram: buffer allocation failed — speculation disabled this step");
        free_spec_buffers_();
        return false;
    }
    // Sub-pointer layout (all int32): [tokens | positions | row_ctx_lens |
    // ctx_len, past_len, chunk_len] and [argmax | topm].
    d_spec_tokens_ = d_spec_stage_;
    d_spec_positions_ = d_spec_stage_ + chunk_cap;
    d_spec_row_ctx_lens_ = d_spec_stage_ + 2ull * chunk_cap;
    d_spec_context_len_ = d_spec_stage_ + 3ull * chunk_cap;
    d_spec_past_len_ = d_spec_context_len_ + 1;
    d_spec_chunk_len_ = d_spec_context_len_ + 2;
    d_spec_snap_n_ = d_spec_context_len_ + 3;
    d_spec_topm_ = d_spec_argmax_ + chunk_cap;
    h_spec_topm_ = h_spec_argmax_.as<int32_t>() + chunk_cap;
    spec_chunk_cap_ = chunk_cap;
    spec_block_table_cap_ = max_blocks;

    // diagnostics.spec_trace only: room for the chunk's full logits, so the
    // trace can report the TOP-2 GAP per row and not just the argmax id.
    // Allocated here with the other spec buffers rather than lazily at the
    // trace site: a first-use allocation would be a serving-phase allocation
    // (docs/internals/MEMORY.md A3.2, and my own check_alloc_pairs gate would
    // see it). Off by default, so the memory is only taken when asked for.
    if (runtime_config_.diagnostics.spec_trace && d_spec_logits_ == nullptr) {
        const size_t v = static_cast<size_t>(model_->config().vocab_size);
        const size_t bytes = static_cast<size_t>(chunk_cap) * v * sizeof(float);
        // Through vram_alloc_, like spec_state_snap_ two functions down, rather
        // than a direct cudaMalloc: invariant I1 keeps the direct-allocation
        // allowlist shrinking, and a diagnostic is not a reason to grow it.
        d_spec_logits_ = static_cast<float*>(vram_alloc_.allocate(bytes, "spec_trace_logits"));
        if (d_spec_logits_ == nullptr) {
            IMP_LOG_WARN(
                "spec_trace: could not allocate %.1f MiB for the logit dump - the trace "
                "will report argmax only, without the top-2 gap",
                bytes / (1024.0 * 1024.0));
        } else {
            h_spec_logits_.assign(static_cast<size_t>(chunk_cap) * v, 0.0f);
        }
    }
    return true;
}

// Hybrid verify: scratch slab holding the committed recurrent state across
// the speculative chunk forward. Sized once (per_seq_bytes is fixed after
// init); freed with the other spec buffers.
bool Engine::ensure_spec_state_scratch_() {
    if (!ssm_state_) return false;
    const size_t bytes = ssm_state_->per_seq_bytes();
    if (spec_state_scratch_ && spec_state_scratch_bytes_ >= bytes) return true;
    if (spec_state_scratch_) {
        IMP_CUDA_CHECK_LOG(cudaFree(spec_state_scratch_));
        spec_state_scratch_ = nullptr;
        spec_state_scratch_bytes_ = 0;
    }
    if (cudaMalloc(&spec_state_scratch_, bytes) != cudaSuccess) {
        IMP_LOG_WARN("spec-hybrid: state scratch alloc failed (%zu bytes) — "
                     "speculation disabled this step", bytes);
        return false;
    }
    spec_state_scratch_bytes_ = bytes;
    // A second slab for the mid-chunk snapshot. The chunk writes the state as
    // of its first row here alongside the committed one at the last row, so a
    // draft that was rejected outright adopts it instead of restoring the
    // pre-chunk state and re-forwarding a full model pass to reach it — that
    // re-forward measures 17.2 ms against a 28.5 ms verify. Optional: without
    // it the replay path stands, so an allocation failure is a warning.
    // Through the VRAM allocator rather than cudaMalloc: invariant I1 keeps
    // direct driver calls inside src/memory/, and the allowlist this file sits
    // on only ever shrinks. #1459 added a raw pair here and pushed the file
    // from its budgeted 13 sites to 15, which failed the blocking Alloc-sites
    // gate on main from that commit onward.
    if (spec_state_snap_ == nullptr) {
        spec_state_snap_ = vram_alloc_.allocate(bytes, "spec_state_snapshot");
        if (spec_state_snap_ == nullptr)
            IMP_LOG_WARN(
                "spec-hybrid: snapshot slab alloc failed (%zu bytes) — partial acceptances "
                "will re-forward instead of adopting the snapshot",
                bytes);
    }
    return true;
}

// Mirror of fill_recurrent_state's slot resolution (decode requests own a
// slot acquired at prefill; the modulo fallback matches its legacy path).
int Engine::recurrent_slot_for_(int req_id) const {
    auto it = recurrent_slot_of_.find(req_id);
    if (it != recurrent_slot_of_.end()) return it->second;
    const int cap = ssm_state_ ? ssm_state_->max_sequences() : 0;
    return cap > 0 ? req_id % cap : 0;
}

void Engine::free_spec_buffers_() {
    // The spec_trace logit dump, freed through the API that allocated it.
    if (d_spec_logits_) {
        vram_alloc_.free(d_spec_logits_);
        d_spec_logits_ = nullptr;
    }
    h_spec_logits_.clear();
    h_spec_logits_.shrink_to_fit();
    if (spec_state_snap_) {
        vram_alloc_.free(spec_state_snap_);
        spec_state_snap_ = nullptr;
    }
    // Captured verify graphs bake these buffer pointers — drop them first.
    free_spec_graphs_();
    if (spec_state_scratch_) {
        IMP_CUDA_CHECK_LOG(cudaFree(spec_state_scratch_));
        spec_state_scratch_ = nullptr;
        spec_state_scratch_bytes_ = 0;
    }
    // d_spec_tokens_/positions_/row_ctx_lens_/context_len_/past_len_/
    // chunk_len_ are sub-pointers into d_spec_stage_; d_spec_topm_ into
    // d_spec_argmax_ — only the block heads are freed.
    if (d_spec_stage_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_stage_));
    if (d_spec_block_table_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_block_table_));
    if (d_spec_block_table_swa_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_block_table_swa_));
    if (d_spec_row_block_tables_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_row_block_tables_));
    if (d_spec_argmax_) IMP_CUDA_CHECK_LOG(cudaFree(d_spec_argmax_));
    d_spec_stage_ = nullptr;
    h_spec_stage_.reset();
    d_spec_tokens_ = nullptr;
    d_spec_positions_ = nullptr;
    d_spec_block_table_ = nullptr;
    d_spec_block_table_swa_ = nullptr;
    d_spec_row_ctx_lens_ = nullptr;
    d_spec_row_block_tables_ = nullptr;
    h_spec_row_tables_pinned_.reset();
    d_spec_context_len_ = nullptr;
    d_spec_past_len_ = nullptr;
    d_spec_chunk_len_ = nullptr;
    d_spec_argmax_ = nullptr;
    h_spec_argmax_.reset();
    d_spec_topm_ = nullptr;
    h_spec_topm_ = nullptr;
    spec_chunk_cap_ = 0;
    spec_block_table_cap_ = 0;
}

}  // namespace imp
