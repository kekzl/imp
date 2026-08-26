// Engine prefill execution: step_prefill driver, per-request chunked
// prefill (step_prefill_one), KV-block allocation and metadata upload.
//
// Split out of engine_scheduler.cpp on 2026-08-26 (the file had doubled to
// 2230 code LOC past its allowlist rationale). Pure move: the function bodies
// are byte-identical to their previous form in that file.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "core/buffer.h"
#include "compute/mtp_forward.h"
#include "compute/dispatch_record.h"
#include "model/image_placeholders.h"
#include "memory/kv_cache.h"
#include "compute/sampling.h"
#include "core/logging.h"

#include <climits>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>
#include <utility>

namespace imp {

using engine_internal::build_logprob_info;
using engine_internal::free_prefill_buffers;

// =====================================================================
// step_prefill — process all prefill requests
// =====================================================================

void Engine::step_prefill(cudaStream_t stream) {
    int resolved = resolve_prefill_chunk_size_();
    int effective_chunk = (resolved > 0) ? resolved : executor_->max_tokens();
    // Hard cap: chunk size must never exceed the executor's max_tokens
    // (which is itself capped to 2048 for SSM/GDN hybrids to bound workspace
    // VRAM — executor_workspace.cu:252). Without this clamp, a server-side
    // prefill_chunk_size default (handlers.cpp) would overflow the
    // workspace and crashes with `n_tokens (X) exceeds max_tokens (Y)` →
    // `terminate: reshape: numel mismatch` on long prompts to e.g. Qwen3.6.
    if (effective_chunk > executor_->max_tokens()) {
        effective_chunk = executor_->max_tokens();
    }
    if (kv_manager_) {
        int bs = kv_manager_->kv_cache()->block_size();
        if (effective_chunk > bs)
            effective_chunk = (effective_chunk / bs) * bs;
    }

    // Decode-aware chunking: prefill and decode share one CUDA stream, so
    // every chunk forward inserts its full latency (~40-80 ms at 2048)
    // between two decode steps of every concurrently DECODING session. Cap
    // the chunk while decoders are active so their inter-token latency stays
    // bounded during another session's ingest; the full chunk (and its
    // better weight-traffic amortization) returns as soon as nobody decodes.
    const int decode_cap = runtime_config_.runtime.prefill_chunk_decode_cap;
    if (decode_cap > 0 && !sched_decode_batch_.empty() && effective_chunk > decode_cap) {
        int capped = decode_cap;
        if (kv_manager_) {
            int bs = kv_manager_->kv_cache()->block_size();
            if (capped > bs)
                capped = (capped / bs) * bs;
        }
        effective_chunk = capped;
    }

    // Decode-aware batching: the size cap above bounds ONE chunk, this bounds
    // how much prefill runs per step before the decoders get their turn
    // (#1643). The budget is TOKEN-charged, not forward-counted: the old
    // count cap of 1 was measured on ~5.2k-token ingests, where one forward
    // IS the whole latency story - but it also serialised 31 concurrent
    // ~110-token prompts to one per engine step, which starved a burst
    // arrival for seconds (TTFT up to 8 s at 32 streams; the decoders being
    // protected were the burst's own first finishers). Each forward charges
    // at least kPrefillForwardFloorTokens, because a short forward's cost is
    // launch-bound (64 layers), not token-bound - so a 1024 budget admits at
    // most 4 small forwards (~100 ms stall, the same bound the size cap
    // targets) and exactly 1 full-sized chunk (the #1643 schedule,
    // unchanged). Starting index rotates so the ingests that do not run this
    // step are the ones that ran last step.
    const size_t n_prefill = sched_prefill_batch_.size();
    size_t budget = n_prefill;
    const int batch_cap = runtime_config_.runtime.prefill_batch_decode_cap;
    if (batch_cap > 0 && !sched_decode_batch_.empty() && n_prefill > static_cast<size_t>(batch_cap))
        budget = static_cast<size_t>(batch_cap);
    constexpr int kPrefillForwardFloorTokens = 256;
    const bool budgeted = decode_cap > 0 && !sched_decode_batch_.empty();
    int token_budget = budgeted ? std::max(decode_cap, kPrefillForwardFloorTokens) : 0;

    // Rotation is by request ID, not by index: requests LEAVE the batch as
    // their prefill completes, so an index-based rotor drifts over the
    // shrunken list and systematically jumps a moving cohort (measured as
    // 5 of 32 burst requests starved to wave-end TTFT while the budget had
    // room). The rotor remembers the last-served id and starts just past
    // it; ids are admission-ordered, so this is a clean round-robin under
    // membership churn.
    size_t start = 0;
    for (size_t i = 0; i < n_prefill; i++) {
        if (sched_prefill_batch_[i]->id > sched_prefill_last_id_) {
            start = i;
            break;
        }
    }
    size_t ran = 0;
    // Cross-sequence ragged prefill (runtime.prefill_batch): eligible requests
    // are collected and run as ONE ragged forward after the loop; ineligible
    // ones keep the serial path. Selection order, budget accounting and the
    // rotor are identical either way, with one pricing difference: the
    // 256-token launch-cost floor is charged once per FORWARD, so ragged
    // members charge their real chunk tokens (the group shares one launch
    // set; a 30-row continuation tail priced at 256 was measured burning a
    // whole engine step per 2-3 tails on a 32-stream burst). The step's
    // inserted latency still scales with total rows, which the token budget
    // continues to bound.
    const bool ragged_mode = prefill_ragged_enabled_();
    std::vector<std::shared_ptr<Request>> ragged_batch;
    bool ragged_floor_charged = false;
    for (size_t i = 0; i < budget; i++) {
        auto& req = sched_prefill_batch_[(start + i) % n_prefill];
        // Charge from the PRE-call remaining: step_prefill_one advances
        // prefill_offset, so reading it afterwards undercharges any chunk
        // that does not finish the prompt.
        const int remaining = static_cast<int>(req->input_tokens.size()) - req->prefill_offset;
        const int chunk_tokens = std::min(std::max(remaining, 0), effective_chunk);
        const bool rides_ragged = ragged_mode && prefill_ragged_req_ok_(*req);
        int charge;
        if (rides_ragged) {
            charge = (ragged_floor_charged || chunk_tokens >= kPrefillForwardFloorTokens)
                         ? chunk_tokens
                         : kPrefillForwardFloorTokens;
        } else {
            charge = std::max(kPrefillForwardFloorTokens, chunk_tokens);
        }
        // `budgeted`, not `token_budget > 0`: an exhausted budget must break,
        // not disarm the check (that bug ran the whole batch after charge 4).
        if (budgeted && ran > 0 && charge > token_budget)
            break;
        if (rides_ragged) {
            ragged_batch.push_back(req);
            ragged_floor_charged = true;
        } else {
            step_prefill_one(req, effective_chunk, stream);
        }
        kv_manager_->touch(req->id);
        ran++;
        sched_prefill_last_id_ = req->id;
        if (budgeted)
            token_budget -= charge;
    }
    if (ragged_batch.size() == 1)
        step_prefill_one(ragged_batch[0], effective_chunk, stream);
    else if (!ragged_batch.empty())
        step_prefill_ragged_(ragged_batch, effective_chunk, stream);
    if (ran >= n_prefill)
        sched_prefill_last_id_ = -1;
}

// =====================================================================
// step_prefill_one — process a single prefill request
// =====================================================================

// Allocate KV blocks for a prefill step. Two sub-paths:
//   - prefix caching: try allocate_blocks_with_prefix, evict + retry on
//     budget pressure, advance `offset` past the reused prefix.
//   - plain: allocate `additional` blocks, evict + retry, cancel on hard
//     failure.
// Returns false on unrecoverable failure (req->status already set to
// CANCELLED). On prefix-cache reuse, mutates offset / chunk_len /
// is_last_chunk / ctx_len in place.
bool Engine::prefill_allocate_kv_blocks_(std::shared_ptr<Request>& req, int kv_bs, int total_input,
                                         int effective_chunk, int& offset, int& chunk_len,
                                         bool& is_last_chunk, int& ctx_len, cudaStream_t pf_stream) {
    int num_blocks = (ctx_len + kv_bs - 1) / kv_bs;
    int prefix_reused = 0;
    int existing = static_cast<int>(kv_manager_->block_table(req->id).size());

    // Perplexity capture must forward EVERY position — a prefix-cache hit
    // skips the reused prefix's forward, leaving those NLL slots at 0.
    // Embedding requests mean-pool EVERY position's hidden state — a
    // prefix-cache hit would skip the reused prefix's forward and silently
    // bias the pooled vector (#1005). Same class as the ppl_capture guard.
    // An image request participates only through its content hash: the cache is
    // addressed by TOKEN IDS, every image token carries the SAME id, and two
    // different pictures would otherwise share a long prefix and the second one
    // would answer about the first one's picture. A request that carries an
    // image but reports no hash is excluded outright — a missed plumbing site
    // must degrade to "no reuse", never to "the previous picture".
    const bool has_image = req->image || !req->qwen_patches.empty() || req->vision_emb ||
                           req->n_vision_tokens > 0;
    const bool cacheable = !has_image || req->vision_content_hash != 0;
    if (kv_manager_->prefix_caching_enabled() && existing == 0 && offset == 0 && !ppl_capture_.active &&
        !req->embedding_request && cacheable) {
        prefix_reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens, -1,
                                                                 req->vision_content_hash);
        if (prefix_reused < 0) {
            // KV exhausted even after cached-block reclamation. The old fallback
            // evicted live sequences (every lru_order_ entry is live; no
            // recompute path) → silent corruption. Reject-newest instead.
            req->status = RequestStatus::CANCELLED;
            return false;
        }

        if (prefix_reused > 0) {
            int effective_reused = (prefix_reused > 1) ? prefix_reused - 1 : 0;
            int skip_tokens = effective_reused * kv_bs;
            if (skip_tokens >= total_input) {
                skip_tokens = (total_input / kv_bs) * kv_bs;
                if (skip_tokens >= total_input) {
                    skip_tokens = total_input - 1;
                }
            }
            if (skip_tokens > offset) {
                IMP_LOG_INFO("PrefixCache: seq %d skipping %d/%d prefill tokens (%d blocks reused)", req->id,
                             skip_tokens, total_input, prefix_reused);
                req->cached_tokens = skip_tokens;
                offset = skip_tokens;
                req->prefill_offset = offset;
                chunk_len = total_input - offset;
                is_last_chunk = true;
                // Re-apply the offset-aware S-matrix clamp: the caller computed
                // effective_chunk for the pre-skip offset, and a cuBLAS-served
                // chunk at the new (larger) offset may need to be smaller
                // (n × ctx_len ≤ s_cap²). The upfront servability check in
                // step_prefill_one guarantees ≥ kv_bs fits at any offset.
                int max_chunk = executor_->max_safe_prefill_chunk(offset, effective_chunk, kv_bs);
                if (max_chunk > 0 && max_chunk < effective_chunk)
                    effective_chunk = max_chunk;
                if (chunk_len > effective_chunk) {
                    chunk_len = effective_chunk;
                    is_last_chunk = false;
                }
                ctx_len = offset + chunk_len;
                (void)executor_->resize_workspace(chunk_len, pf_stream);
            }
        }
    } else {
        int additional = num_blocks - existing;
        if (additional > 0) {
            // allocate_blocks already reclaims cached blocks; if it still fails
            // the KV cache is genuinely exhausted. The old evict_lru fallback
            // freed a LIVE sequence (no recompute path) → silent corruption.
            // Reject-newest: cancel this request, leave in-flight ones intact.
            if (!kv_manager_->allocate_blocks(req->id, additional)) {
                kv_pressure_rejections_.fetch_add(1, std::memory_order_relaxed);
                cancel_sequence_(req);
                req->status = RequestStatus::CANCELLED;
                return false;
            }
        }
    }
    return true;
}

// Upload prefill metadata to device. Uses the prefill_pool_ pre-allocated
// buffers when chunk_len fits; otherwise falls back to cudaMallocAsync and
// frees on any allocation failure. Pinned staging buffers are used for the
// token_ids / positions H2D copies when available (avoids internal
// pageable→pinned copy inside cuMemcpy).
bool Engine::prefill_upload_metadata_(std::shared_ptr<Request>& req, const std::vector<int>& block_table,
                                      const std::vector<int>& swa_block_table, int chunk_len, int offset,
                                      int ctx_len, cudaStream_t pf_stream, int32_t*& d_token_ids,
                                      int*& d_positions, int*& d_block_tables, int*& d_block_tables_swa,
                                      int*& d_context_lens, bool& pf_pool_used) {
    d_token_ids = nullptr;
    d_positions = nullptr;
    d_block_tables = nullptr;
    d_block_tables_swa = nullptr;
    d_context_lens = nullptr;
    pf_pool_used = false;
    const bool want_swa = swa_sizing_active_ && !swa_block_table.empty();

    auto check = [&req](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Engine::step prefill %s failed: %s", op, cudaGetErrorString(err));
            req->status = RequestStatus::CANCELLED;
        }
        return err == cudaSuccess;
    };

    if (prefill_pool_ && chunk_len <= config_.max_seq_len) {
        d_token_ids = d_pf_token_ids_;
        d_positions = d_pf_positions_;
        d_block_tables = d_pf_block_tables_;
        if (want_swa)
            d_block_tables_swa = d_pf_block_tables_swa_;
        d_context_lens = d_pf_context_lens_;
        pf_pool_used = true;
    } else {
        if (!check(cudaMallocAsync(&d_token_ids, chunk_len * sizeof(int32_t), pf_stream),
                   "malloc token_ids") ||
            !check(cudaMallocAsync(&d_positions, chunk_len * sizeof(int), pf_stream), "malloc positions") ||
            !check(cudaMallocAsync(&d_block_tables, block_table.size() * sizeof(int), pf_stream),
                   "malloc block_tables") ||
            (want_swa &&
             !check(cudaMallocAsync(&d_block_tables_swa, swa_block_table.size() * sizeof(int), pf_stream),
                    "malloc block_tables_swa")) ||
            !check(cudaMallocAsync(&d_context_lens, sizeof(int), pf_stream), "malloc context_lens")) {
            if (d_token_ids)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, pf_stream));
            if (d_positions)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, pf_stream));
            if (d_block_tables)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, pf_stream));
            if (d_block_tables_swa)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables_swa, pf_stream));
            if (d_context_lens)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, pf_stream));
            // Not counted as KV pressure: this is a metadata cudaMallocAsync
            // failure, not the KV pool refusing blocks (#1641).
            cancel_sequence_(req);
            return false;
        }
    }

    // Use pinned staging buffers when available (avoids internal pageable->pinned copy).
    // PINNED sources are truly asynchronous: the H2D reads the buffer when the
    // copy EXECUTES (in stream order, behind all prior chunks' kernels), not
    // when it is enqueued. Before rewriting the staging for this chunk, wait
    // until the previous chunk's copies have actually run — otherwise a host
    // that runs several fully-async chunks ahead (FA2 attention path, no
    // implicit syncs) uploads chunk c+N's tokens/positions for chunk c
    // (#548: catastrophic chunked-prefill NLL, timing/arch-dependent).
    // Pageable sources below (block_table, ctx_len) are safe by CUDA
    // semantics (captured before cudaMemcpyAsync returns).
    if (pf_staging_evt_ && (h_pf_token_ids_.as<int32_t>() || h_pf_positions_.as<int>()) &&
        chunk_len <= config_.max_seq_len)
        IMP_CUDA_CHECK_LOG(cudaEventSynchronize(pf_staging_evt_));
    if (h_pf_token_ids_.as<int32_t>() && chunk_len <= config_.max_seq_len) {
        memcpy(h_pf_token_ids_.as<int32_t>(), req->input_tokens.data() + offset, chunk_len * sizeof(int32_t));
        check(cudaMemcpyAsync(d_token_ids, h_pf_token_ids_.as<int32_t>(), chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    } else {
        check(cudaMemcpyAsync(d_token_ids, req->input_tokens.data() + offset, chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    }

    if (int* h_pos = h_pf_positions_.as<int>(); h_pos && chunk_len <= config_.max_seq_len) {
        for (int i = 0; i < chunk_len; i++)
            h_pos[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, h_pos, chunk_len * sizeof(int), cudaMemcpyHostToDevice, pf_stream),
              "memcpy positions");
    } else {
        std::vector<int> positions(chunk_len);
        for (int i = 0; i < chunk_len; i++)
            positions[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, positions.data(), chunk_len * sizeof(int), cudaMemcpyHostToDevice,
                              pf_stream),
              "memcpy positions");
    }

    if (pf_staging_evt_ && (h_pf_token_ids_.as<int32_t>() || h_pf_positions_.as<int>()) &&
        chunk_len <= config_.max_seq_len)
        IMP_CUDA_CHECK_LOG(cudaEventRecord(pf_staging_evt_, pf_stream));

    check(cudaMemcpyAsync(d_block_tables, block_table.data(), block_table.size() * sizeof(int),
                          cudaMemcpyHostToDevice, pf_stream),
          "memcpy block_tables");
    if (want_swa && d_block_tables_swa) {
        check(cudaMemcpyAsync(d_block_tables_swa, swa_block_table.data(),
                              swa_block_table.size() * sizeof(int), cudaMemcpyHostToDevice, pf_stream),
              "memcpy block_tables_swa");
    }
    check(cudaMemcpyAsync(d_context_lens, &ctx_len, sizeof(int), cudaMemcpyHostToDevice, pf_stream),
          "memcpy context_lens");
    return true;
}

void Engine::step_prefill_one(std::shared_ptr<Request>& req, int effective_chunk, cudaStream_t pf_stream) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    int total_input = static_cast<int>(req->input_tokens.size());
    int offset = req->prefill_offset;

    // Out-of-scope archs (Gemma-3/4 SWA, Llama-4, sub-byte KV) lack a paged
    // chunked-prefill path, so the chunked-prefill branch in
    // executor_attention.cu aborts on chunk 2+ (q_offset > 0 + per_layer
    // shapes). Reject prompts > effective_chunk gracefully here instead of
    // letting them hit std::abort. Real fix is the paged hybrid-prefill
    // kernel (roadmap).
    if (offset == 0 && total_input > effective_chunk && !supports_chunked_prefill_()) {
        IMP_LOG_ERROR(
            "Prompt has %d tokens but max_tokens=%d on hybrid/out-of-scope arch — "
            "chunked prefill not supported. Cancelling request %d.",
            total_input, effective_chunk, req->id);
        req->status = RequestStatus::CANCELLED;
        return;
    }

    // Admission heads-up: the KV pool is often VRAM-clamped below the requested
    // max_seq_len. A prompt that fills the pool prefills fine and is then
    // cancelled on its first block append mid-decode (reject-newest) — flag it
    // here once, at submit time, where it is actionable. req->max_tokens is NOT
    // usable here (imp_prefill seeds a 4096 placeholder; decode_step drives the
    // real stop), so gate on the prompt leaving less than one block of headroom.
    if (offset == 0 && kv_cache_raw_) {
        int64_t pool_tokens = static_cast<int64_t>(kv_cache_raw_->total_blocks()) * kv_bs;
        if (pool_tokens > 0 && total_input > pool_tokens - kv_bs) {
            IMP_LOG_WARN(
                "Request %d: prompt (%d tokens) leaves <1 KV block of decode headroom in the "
                "%lld-token pool (VRAM-clamped) — decode will be cancelled almost immediately. "
                "Lower max_seq_len, shorten the prompt, or halve KV with kv_cache.dtype=fp8 "
                "(--kv-fp8).",
                req->id, total_input, (long long)pool_tokens);
        }
    }

    // Clamp effective_chunk so the chunked-attention S-matrix cannot overflow
    // (cuBLAS stores an [nh, n, ctx_len] score matrix; n × ctx_len ≤ s_cap²).
    // max_safe_prefill_chunk mirrors the executor dispatch and only clamps
    // chunks that will actually land on cuBLAS (learned sinks → gpt-oss,
    // heterogeneous shapes → Gemma-4); chunks served by the O(n) FA2/FMHA
    // family pass through unclamped. The clamp is offset-aware: early chunks
    // stay large and only late chunks shrink (previously EVERY chunk was
    // clamped to the final-chunk worst case cap²/total_input — e.g. 32-token
    // chunks across an entire 128k prompt on hd=256 hybrids).
    if (executor_) {
        if (offset == 0 && total_input > kv_bs) {
            // Upfront servability check: if even a kv_bs-sized final chunk
            // cannot fit the S-matrix, reject cleanly instead of letting the
            // kernel capacity guard abort the process mid-prefill.
            int last_off = ((total_input - 1) / kv_bs) * kv_bs;
            if (executor_->max_safe_prefill_chunk(last_off, kv_bs, kv_bs) < kv_bs) {
                IMP_LOG_ERROR(
                    "Prompt (%d tokens) exceeds the chunked-attention workspace for this model "
                    "(S-matrix cap %d; learned-sink/heterogeneous attention requires cuBLAS) — "
                    "cancelling request %d. Reduce the prompt or raise attention.attn_scores_mib.",
                    total_input, executor_->attn_scores_cap(), req->id);
                req->status = RequestStatus::CANCELLED;
                return;
            }
        }
        int max_chunk = executor_->max_safe_prefill_chunk(offset, effective_chunk, kv_bs);
        if (max_chunk > 0 && effective_chunk > max_chunk)
            effective_chunk = max_chunk;
    }

    // Determine chunk boundaries
    int chunk_len = total_input - offset;
    bool is_last_chunk = true;
    if (chunk_len > effective_chunk) {
        chunk_len = effective_chunk;
        is_last_chunk = false;
    }

    // Snapshot boundary (hybrid recurrent state / SWA window): end a chunk
    // exactly at the largest block-aligned prompt position so the state there
    // can be captured — the snapshot is only restorable where reused KV
    // blocks cover the whole prefix, and only full blocks are cacheable. The
    // extra tail chunk is at most block_size-1 tokens.
    const int snap_end = snapshot_end_(*req);
    if (snap_end > offset && snap_end < offset + chunk_len) {
        chunk_len = snap_end - offset;
        is_last_chunk = false;
    }

    int ctx_len = offset + chunk_len;
    (void)executor_->resize_workspace(chunk_len, pf_stream);

    if (!prefill_allocate_kv_blocks_(req, kv_bs, total_input, effective_chunk, offset, chunk_len,
                                     is_last_chunk, ctx_len, pf_stream)) {
        return;  // caller already set req->status = CANCELLED
    }

    // SWA-aware sizing: live window blocks for this chunk's write range plus
    // the window its queries/continuation-gathers read back into. Trimming of
    // blocks that fell out of the window happens after the chunk commits.
    if (swa_sizing_active_) {
        // SWA snapshot restore (prefix-cache hit): fill the window blocks at
        // exactly the reused-prefix boundary BEFORE the continuation chunk's
        // gathers read them. Without the restored window the reused global
        // prefix is unusable (windowed layers would attend holes) — treat a
        // failed restore like a swa_prepare failure below.
        if (req->swa_restore && offset > 0 && offset == req->cached_tokens) {
            const auto rt0 = std::chrono::steady_clock::now();
            const auto& entry = *req->swa_restore;
            const bool ok = entry.data && entry.n_tokens == offset && swa_snapshots_ &&
                            swa_snapshots_->entry_bytes() == kv_manager_->swa_snapshot_bytes() &&
                            kv_manager_->swa_snapshot_restore(req->id, offset, entry.data, pf_stream);
            req->swa_restore.reset();
            if (!ok) {
                IMP_LOG_WARN("SwaSnapshot: restore failed for req %d at %d tokens — cancelling", req->id,
                             offset);
                // Not KV pressure: a snapshot that does not match, not a pool
                // that cannot allocate (#1641).
                cancel_sequence_(req);
                req->status = RequestStatus::CANCELLED;
                return;
            }
            IMP_LOG_INFO(
                "SwaSnapshot: restored %d-token window for req %d (enqueue %.2f ms)", offset, req->id,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rt0).count());
        }
        kv_manager_->swa_trim(req->id, offset);
        if (!kv_manager_->swa_prepare(req->id, offset, ctx_len)) {
            kv_pressure_rejections_.fetch_add(1, std::memory_order_relaxed);
            cancel_sequence_(req);
            req->status = RequestStatus::CANCELLED;
            return;
        }
    }

    const auto& block_table = kv_manager_->block_table(req->id);
    const auto& swa_block_table = kv_manager_->swa_block_table(req->id);

    int32_t* d_token_ids = nullptr;
    int* d_positions = nullptr;
    int* d_block_tables = nullptr;
    int* d_block_tables_swa = nullptr;
    int* d_context_lens = nullptr;
    bool pf_pool_used = false;
    if (!prefill_upload_metadata_(req, block_table, swa_block_table, chunk_len, offset, ctx_len, pf_stream,
                                  d_token_ids, d_positions, d_block_tables, d_block_tables_swa,
                                  d_context_lens, pf_pool_used)) {
        return;  // caller already set req->status = CANCELLED
    }

    // First chunk of a request is where its image becomes device-resident: the
    // layout needs the prompt's final token sequence, which only exists now.
    // The CLI leaves its image pending on the engine; the server puts it on the
    // request itself. Both end up in `encode_qwen_image_for_`.
    if (offset == 0) {
        (void)attach_qwen_image_(*req);
        // The legacy global-image path (imp_set_image + the mmproj pipeline)
        // never puts anything on the request, so the prefix-cache guard below
        // cannot see it. Stamp the hash here, or an interactive session that
        // switches pictures would match the previous one's blocks.
        if (req->vision_content_hash == 0 && vision_.has_input() && vision_.is_available())
            req->vision_content_hash = pending_image_hash_;
    }
    // Not gated on `offset == 0`: `qwen_patches` is cleared by the encode, so
    // this runs exactly once regardless of how the prompt is chunked.
    {
        if (!req->qwen_patches.empty() && !encode_qwen_image_for_(*req, pf_stream)) {
            IMP_LOG_ERROR("Qwen3-VL: image encode failed — cancelling request %llu",
                          static_cast<unsigned long long>(req->id));
            req->status = RequestStatus::CANCELLED;
            return;
        }
    }

    // Build InferenceState
    InferenceState state;
    state.token_ids = d_token_ids;
    state.positions = d_positions;
    state.n_tokens = chunk_len;
    bind_mrope_prefill_(state, *req, offset, chunk_len, pf_stream);
    if (req->n_vision_tokens > 0 && req->vision_token_id >= 0 && req->vision_emb) {
        state.vision_embeddings = req->vision_emb->as<half>();
        state.vision_token_id = req->vision_token_id;
        state.n_vision_tokens = req->n_vision_tokens;
        // The kernels index placeholders within the chunk they are handed, so
        // they need to know how many this request already placed. A long enough
        // prompt puts an image across a chunk boundary (chunks default to 2048),
        // and without this the second chunk would re-use the image's FIRST
        // embeddings — the wrong region of the picture, with nothing failing.
        // Counted over the whole prompt so a prefix-cache hit, which starts at
        // `cached_tokens`, is covered by the same arithmetic.
        state.vision_emb_offset = image_tokens_before(req->input_tokens, req->vision_token_id, offset);
        state.n_deepstack = std::min<int>(static_cast<int>(req->deepstack_emb.size()),
                                          InferenceState::kMaxDeepStack);
        for (int d = 0; d < state.n_deepstack; ++d)
            state.deepstack_embeddings[d] = req->deepstack_emb[d]->as<half>();
    }
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_block_tables;
    state.block_tables_swa = d_block_tables_swa;
    state.context_lens = d_context_lens;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = 0;
    state.is_prefill = true;
    state.prefill_offset = offset;  // absolute pos of state.positions[0]
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) {
        // Slot lookup happens inside KVCacheManager::residual_k_ptr; if no
        // slot is allocated yet (prefill before first decode), the residual
        // pointers return nullptr and the kernel skips the residual pass.
        state.kv_seq_id = req->id;
    }
    fill_sampling_params(*req, state);

    // Constraints via the per-request ConstraintManager. The old engine-global
    // manager was re-prepared here for EVERY prefill (constrained or not),
    // which clobbered the FSM of any concurrently decoding constrained
    // request. Prepare once on first need; later chunks reuse the state.
    // thinking_open = req->in_think_block: if the prompt already closed the
    // <think> block (e.g. /no_think emits an empty <think></think> in the
    // prompt), no </think> is ever generated — the preamble gate must enforce
    // immediately instead of absorbing prose until the budget.
    ensure_constraints_(req);
    if (req->constraints) {
        state.json_constrainer = req->constraints->json_constrainer();
        state.regex_constrainer = req->constraints->regex_constrainer();
        state.grammar_constrainer = req->constraints->grammar_constrainer();
        state.schema_constrainer = req->constraints->schema_constrainer();
    }

    // Penalties
    upload_penalties(*req, state, pf_stream);

    // Recurrent state (SSM/GDN)
    // Reset on the first chunk of a new request so previous-request state
    // doesn't leak in.  Subsequent chunks must NOT reset — the recurrent
    // state built during earlier chunks must carry forward. The first chunk
    // starts at cached_tokens (> 0 on a prefix-cache hit, where "reset"
    // restores the matching recurrent snapshot instead of zeroing).
    fill_recurrent_state(*req, state, /*reset=*/(offset == req->cached_tokens), pf_stream);

    // Vision embeddings on first chunk.
    if (req->vision_emb && offset == 0) {
        // Per-request (server batched path): the worker encoded req->image into
        // req->vision_emb on admission, so vision batches with text.
        state.vision_embeddings = req->vision_emb->as<half>();
        state.vision_token_id = req->vision_token_id;
        state.n_vision_tokens = req->n_vision_tokens;
    } else if (vision_.has_input() && vision_.is_available() && offset == 0) {
        // Global path: the C-API (imp_set_image) / imp-cli set ONE image on the
        // engine for the next generation; its request carries no per-request
        // embeddings, so bind the global ones. (Restores the pre-per-request
        // binding the server no longer uses — imp_prefill_with_params builds a
        // bare request, so without this the CLI's image was silently ignored.)
        state.vision_embeddings = vision_.embeddings();
        state.vision_token_id = vision_.soft_token_id();
        state.n_vision_tokens = vision_.num_image_tokens();
    }

    if (!is_last_chunk) {
        if (executor_->has_decode_workspace()) {
            executor_->use_workspace(0);
        }
        Tensor logits_out;

        // Prefill graph capture (opt-in, Phase 4 of MoE-prefill-graphs work).
        // Conditions: env-gated, pool path (stable device buffers), and
        // chunk shape stable (in practice all non-last chunks share chunk_len
        // = prefill_chunk_size). H2D upload happened above on pf_stream
        // *before* this wrapper — captured region is forward_logits only,
        // analogous to the decode graph pattern.
        const bool prefill_graph_enabled = runtime_config_.runtime.prefill_graph;
        // The M>1 NVFP4 dequant fallback lazy-cudaMallocs when its workspace
        // couldn't be pre-allocated (largest weight > cap) — illegal under CUDA
        // graph capture (cublasLt status 14 → cascading "previous error during
        // capture"). Run prefill eager for those models (Qwen3.6-35B pp>=4096).
        // The recurrent-snapshot boundary chunk has a request-dependent odd
        // shape (prompt mod chunk size) — capturing it churns the prefill
        // graph every request AND the odd-M cuBLAS call can lazily allocate
        // workspace, which is illegal under capture (cublasLt status 14 →
        // cascading capture failure, observed on GGUF mxfp4 GDN). Run eager.
        const bool ends_at_snapshot = (snap_end > 0 && offset + chunk_len == snap_end);
        // moe_prefill_uncapturable: legacy host-args MoE prefill (GGUF Q*_K
        // MoE) reads routing on the host — its capture guard throws and the
        // aborted capture costs a wasted forward per chunk. Run eager (#874).
        // Quantized KV append runs a dynamic-scale reduction with a D2H
        // absmax sync per chunk — illegal under capture (the capture aborts
        // every chunk, spamming errors and wasting one forward per chunk).
        // F16 KV is the only append path that captures cleanly; run the rest
        // eager.
        const bool kv_append_capturable = (config_.kv_cache_dtype == QType::F16);
        // Continuation chunks (offset > 0) bake ctx_len/q_offset as host args
        // into the attention launches, so a replay only fits the exact same
        // offset — which never repeats within a request. Replaying chunk 1's
        // graph for chunk 2+ attended with chunk-1 geometry and silently
        // truncated long-context prefill (#981: teacher-forced PPL
        // 8.30 -> 15.35 past chunk 2). Capture only the offset-0 chunk, whose
        // geometry DOES repeat across requests; continuations run eager.
        const bool can_capture = prefill_graph_enabled && pf_pool_used && config_.use_cuda_graphs &&
                                 kv_append_capturable && offset == 0 && !ends_at_snapshot &&
                                 !executor_->nvfp4_dequant_uncapturable() &&
                                 !executor_->moe_prefill_uncapturable();
        if (can_capture) {
            const int block_count = static_cast<int>(block_table.size());
            if (chunk_len != last_prefill_chunk_len_ || block_count != last_prefill_block_count_) {
                prefill_graph_runner_.invalidate_for_update();
                last_prefill_chunk_len_ = chunk_len;
                last_prefill_block_count_ = block_count;
            }
            prefill_graph_runner_.set_decode_fn([this, &state, &logits_out](cudaStream_t s) {
                executor_->forward_logits(state, logits_out, s);
            });
            prefill_graph_runner_.execute(pf_stream);
            if (logits_out.data == nullptr) {
                logits_out = executor_->get_logits_view(/*n=*/1);
            }
        } else {
            // `runtime.prefill_graph` defaults to true, but seven conditions
            // gate the capture and none of them logged anything, so a model
            // that never captures looked exactly like one that does. Report
            // the failing condition once per process: measured on
            // Qwen3-8B-Q8_0 and Qwen3-Coder-30B-A3B-NVFP4, neither ever
            // captured a prefill chunk, and finding out which gate closed
            // took a source read plus three A/Bs.
            static bool logged_no_prefill_capture = false;
            if (prefill_graph_enabled && !logged_no_prefill_capture) {
                logged_no_prefill_capture = true;
                IMP_LOG_INFO(
                    "prefill graph: not capturing (runtime.prefill_graph=true) — "
                    "pf_pool=%d cuda_graphs=%d kv_append_f16=%d offset0=%d "
                    "not_snapshot_end=%d nvfp4_dequant_capturable=%d moe_capturable=%d",
                    (int)pf_pool_used, (int)config_.use_cuda_graphs, (int)kv_append_capturable,
                    (int)(offset == 0), (int)!ends_at_snapshot, (int)!executor_->nvfp4_dequant_uncapturable(),
                    (int)!executor_->moe_prefill_uncapturable());
            }
            executor_->forward_logits(state, logits_out, pf_stream);
        }

        // Teacher-forced NLL for this chunk's positions (imp_perplexity).
        // Runs eagerly after the (possibly graph-replayed) forward; hidden_
        // holds exactly this chunk and nothing reads logits_ afterwards.
        if (ppl_capture_.active) {
            executor_->perplexity_nll_partial(ppl_capture_.d_tokens, ppl_capture_.n, offset, chunk_len,
                                              ppl_capture_.d_nll, pf_stream, ppl_capture_.d_match);
        }

        // Embedding pooling for this chunk (#1005) — hidden_ still holds it.
        if (req->embedding_request)
            embed_accumulate_chunk_(*req, chunk_len, pf_stream);

        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_block_tables_swa, d_context_lens,
                                 pf_stream);
        }

        // MTP: feed this chunk's (token, hidden) pairs while the executor's
        // hidden_ buffer still holds it (feed-only forwards, no lm_head).
        if (mtp_spec_decode_enabled())
            mtp_prefill_feed_chunk(*req, offset, chunk_len, /*next_token=*/-1);

        req->prefill_offset = offset + chunk_len;
        IMP_LOG_DEBUG("Chunked prefill: req %d chunk [%d, %d) of %d", req->id, offset, offset + chunk_len,
                      total_input);
        // The chunk ended exactly at the snapshot boundary — capture the
        // recurrent state / SWA window now, before the next chunk advances it.
        if (snap_end > 0 && req->prefill_offset == snap_end) {
            maybe_save_recurrent_snapshot_(*req, snap_end, pf_stream);
            maybe_save_swa_snapshot_(*req, snap_end, pf_stream);
        }
    } else if (!req->score_token_ids.empty()) {
        // Rerank scoring (/v1/rerank): a cross-encoder reads its verdict from
        // the last position's logits and never samples. Same no-sampling shape
        // as the embedding branch above, but the pooling is a two-logit read.
        Tensor score_logits;
        executor_->forward_logits(state, score_logits, pf_stream);
        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_block_tables_swa, d_context_lens,
                                 pf_stream);
        }
        score_capture_(*req, score_logits, pf_stream);
        finish_request(req);
    } else if (req->embedding_request) {
        // Embedding request (#1005): last chunk — forward, pool, finish.
        // No sampling, no DECODING transition; the request rides the normal
        // finish path (KV free + prefix-hash registration, so re-embedding
        // the same document becomes a prefix-cache hit for OTHER requests).
        Tensor logits_unused;
        executor_->forward_logits(state, logits_unused, pf_stream);
        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_block_tables_swa, d_context_lens,
                                 pf_stream);
        }
        embed_accumulate_chunk_(*req, chunk_len, pf_stream);
        const size_t total = req->input_tokens.size();
        if (total > 0 && !req->embedding_out.empty()) {
            const float inv = 1.0f / static_cast<float>(total);
            for (float& v : req->embedding_out)
                v *= inv;
        }
        finish_request(req);
    } else {
        // Last chunk: forward + sample
        int32_t next_token;
        bool use_event_sync = (!h_sample_pinned_.empty() && executor_->d_sample_result() != nullptr &&
                               (state.temperature <= 0.0f || state.top_k == 1) && !req->logprobs &&
                               !state.json_constrainer && !state.schema_constrainer &&
                               !state.regex_constrainer && !state.grammar_constrainer);

        Tensor prefill_logits_out;

        if (use_event_sync) {
            Tensor logits_out;
            executor_->forward_logits(state, logits_out, pf_stream);
            Tensor last_logits = logits_out.slice(0, 1);
            int64_t vocab_shape[1] = {last_logits.shape[1]};
            last_logits = last_logits.reshape(1, vocab_shape);

            // Ban special tokens (e.g. Gemma-4 <|channel>) before greedy
            // argmax — otherwise the natural-argmax channel marker triggers
            // is_stop_token and the request finishes with 0 completion
            // tokens. Same logic as GraphExecutor::forward (executor.cu:88)
            // and apply_pre_sample (executor.cu) but inline here because
            // sample_greedy_device runs on raw logits without going through
            // either of those wrappers.
            if (state.banned_tokens != nullptr && state.n_banned_tokens > 0) {
                float* lp = static_cast<float*>(last_logits.data);
                int vocab = static_cast<int>(last_logits.shape[0]);
                float neg_inf = -1e30f;
                for (int bi = 0; bi < state.n_banned_tokens; bi++) {
                    int32_t tid = state.banned_tokens[bi];
                    if (tid >= 0 && tid < vocab) {
                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(lp + tid, &neg_inf, sizeof(float),
                                                           cudaMemcpyHostToDevice, pf_stream));
                    }
                }
            }

            sample_greedy_device(last_logits, executor_->d_sample_result(), h_sample_pinned_.as<int32_t>(),
                                 pf_stream);

            if (!prefill_done_)
                (void)prefill_done_.create();
            cudaEventRecord(prefill_done_, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_block_tables_swa,
                                     d_context_lens, pf_stream);
            }

            cudaEventSynchronize(prefill_done_);
            next_token = *h_sample_pinned_.as<int32_t>();
        } else if (req->logprobs) {
            executor_->forward_logits(state, prefill_logits_out, pf_stream);
            auto sampled = executor_->sample_from_logits(prefill_logits_out, state, pf_stream);
            next_token = sampled[0];

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_block_tables_swa,
                                     d_context_lens, pf_stream);
            }
        } else {
            next_token = executor_->forward(state, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_block_tables_swa,
                                     d_context_lens, pf_stream);
            }
        }

        // Block-aligned prompt: the snapshot boundary coincides with the last
        // chunk's end — capture after the forward (the sampling above only
        // reads logits, never the recurrent state; the SWA window blocks are
        // not mutated until the first decode step).
        if (snap_end == total_input) {
            maybe_save_recurrent_snapshot_(*req, snap_end, pf_stream);
            maybe_save_swa_snapshot_(*req, snap_end, pf_stream);
        }

        if (req->mirostat == 2)
            req->mirostat_mu = state.mirostat_mu;

        // Extract logprobs
        if (req->logprobs && prefill_logits_out.data != nullptr) {
            int vocab_size = static_cast<int>(prefill_logits_out.shape[prefill_logits_out.ndim - 1]);
            executor_->ensure_logits_pinned(vocab_size);

            const float* d_logits = static_cast<const float*>(prefill_logits_out.data);
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(executor_->h_logits_pinned(), d_logits,
                                               vocab_size * sizeof(float), cudaMemcpyDeviceToHost,
                                               pf_stream));
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(pf_stream));

            req->output_logprobs.push_back(build_logprob_info(executor_->h_logits_pinned(), vocab_size,
                                                              next_token, req->top_logprobs,
                                                              model_->tokenizer()));
        }

        // Teacher-forced NLL for the LAST chunk's positions (imp_perplexity).
        // After sampling + logprob extraction: the partial pass overwrites the
        // logits_ workspace, so it must run once nothing reads this chunk's
        // logits anymore. hidden_ still holds the chunk (forward_logits only
        // slices the last token for the production LM head).
        if (ppl_capture_.active) {
            executor_->perplexity_nll_partial(ppl_capture_.d_tokens, ppl_capture_.n, offset, chunk_len,
                                              ppl_capture_.d_nll, pf_stream, ppl_capture_.d_match);
        }

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        Tokenizer* tok = model_->tokenizer();
        IMP_LOG_DEBUG("Prefill -> token %d (ctx=%d): id=%d [%s]", (int)req->output_tokens.size(),
                      req->context_len(), next_token, tok->decode_token(next_token).c_str());

        // MTP: feed the last chunk's (token, hidden) pairs — earlier chunks
        // were fed inside the chunked-prefill branch above; the final pair
        // uses the just-sampled next_token and seeds the pending draft chain.
        if (mtp_spec_decode_enabled())
            mtp_prefill_feed_chunk(*req, offset, chunk_len, next_token);

        // Update constraint FSM
        if (req->constraints)
            req->constraints->update(next_token);

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        } else {
            req->status = RequestStatus::DECODING;
            // Publish under the same salt these blocks were looked up with, so
            // they can only ever be offered to a prompt with the same picture.
            const bool had_image = req->n_vision_tokens > 0 || req->image || req->vision_emb;
            if (kv_manager_->prefix_caching_enabled() && (!had_image || req->vision_content_hash != 0)) {
                kv_manager_->register_block_hashes(req->id, req->input_tokens, req->vision_content_hash);
            }
        }
    }
}

}  // namespace imp
