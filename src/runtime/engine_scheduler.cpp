// Engine scheduler: per-step prefill + decode driver.
// The bulk of engine.cpp at ~1250 LOC across 13 methods.
//
// Top-level flow:
//   step() → step_schedule() → step_prefill() OR step_decode()
//
// Prefill chain:
//   step_prefill → resolve_prefill_chunk_size_ → supports_chunked_prefill_
//                → prefill_allocate_kv_blocks_ → prefill_upload_metadata_
//                → step_prefill_one (per chunk)
//
// Decode chain:
//   step_decode → decode_build_inference_state_ → step_decode_forward
//               → step_decode_process_outputs
//
// Async graph resume:
//   step_async_graph_resume — for the CUDA-graph-captured decode path.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. This is the biggest single TU split in Phase 4.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "core/buffer.h"
#include "runtime/batch.h"
#include "runtime/think_stop_logic.h"
#include "compute/mtp_forward.h"
#include "memory/kv_cache.h"
#include "compute/sampling.h"
#include "compute/layernorm.h"
#include "core/logging.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace imp {

using engine_internal::build_logprob_info;
using engine_internal::compute_step_seed;
using engine_internal::ensure_prefill_workspace;
using engine_internal::free_prefill_buffers;

// =====================================================================
// step() — main inference loop
// =====================================================================


// =====================================================================
// step() — main inference loop
// =====================================================================

bool Engine::step() {
    // Fast path: async conditional graph loop completed on GPU.
    int async_result = step_async_graph_resume();
    if (async_result == 1)
        return true;  // still running
    if (async_result == -1) {
        return scheduler_->has_pending() || scheduler_->active_count() > 0;
    }

    // Fast path: pipelined constrained decode (json/schema) — one token/tick.
    int cp_result = step_constrained_pipeline();
    if (cp_result == 1)
        return true;
    if (cp_result == -1) {
        return scheduler_->has_pending() || scheduler_->active_count() > 0;
    }

    // Schedule prefill/decode batches and reconfigure green contexts.
    if (!step_schedule())
        return false;

    // Process prefill requests.
    if (!sched_prefill_batch_.empty()) {
        step_prefill(prefill_stream());
        ensure_prefill_workspace(executor_.get());
    }

    // Process decode requests (batched).
    if (!sched_decode_batch_.empty()) {
        step_decode(decode_stream());
        ensure_prefill_workspace(executor_.get());
    }

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
}

// =====================================================================
// step_async_graph_resume — handle async conditional graph loop
// Returns: 0 = no graph active, 1 = still running, -1 = generation done
// =====================================================================

int Engine::step_async_graph_resume() {
    if (async_graph_runner_.is_setup() && async_graph_req_) {
        auto& req = async_graph_req_;

        // Incremental harvest (#754): poll the mapped ring buffer instead of
        // blocking until the whole burst retires, so tokens reach the caller
        // (server SSE / imp_decode_step) as the device loop produces them.
        // The old cudaStreamSynchronize surfaced a burst's tokens only when
        // it finished — which is why streaming requests had to be excluded
        // from the loop entirely (every token arrived in burst-sized groups).
        // The micro-poll waits for AT LEAST one new token per step() so the
        // imp_decode_step zero-token retry bound (8) never trips; the device
        // loop runs ahead regardless — this only throttles the host to the
        // same per-token cadence as eager decode.
        if (async_pending_cursor_ >= static_cast<int>(async_pending_tokens_.size()) &&
            async_graph_runner_.launch_in_flight()) {
            cudaStream_t dec_stream = decode_stream();
            // Safety valve: a graph that errors out never reaches the stop
            // kernel and thus never publishes done — fall back to a blocking
            // sync (which surfaces the error) instead of polling forever.
            // 30 s dwarfs any legitimate inter-token gap (decode ≤ ~100 ms).
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (async_graph_runner_.poll_new_tokens(async_pending_tokens_) == 0) {
                if (async_graph_runner_.try_finish_burst(dec_stream)) {
                    // Burst retired between polls — drain the tail and stop.
                    async_graph_runner_.poll_new_tokens(async_pending_tokens_);
                    break;
                }
                if (std::chrono::steady_clock::now() > deadline) {
                    IMP_LOG_ERROR("AsyncGraphLoop: no token for 30 s and no done flag — "
                                  "forcing blocking burst finish");
                    async_graph_runner_.finish_burst_blocking(dec_stream);
                    async_graph_runner_.poll_new_tokens(async_pending_tokens_);
                    break;
                }
                // Burst still running, no new token yet (decode steps are
                // ~3-7 ms; polling every 200 µs keeps SSE latency negligible
                // without spinning a core).
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }

        int32_t token = -1;
        if (async_pending_cursor_ < static_cast<int>(async_pending_tokens_.size())) {
            token = async_pending_tokens_[async_pending_cursor_++];
        }

        bool generation_done = false;
        if (token >= 0) {
            req->output_tokens.push_back(token);
            track_think_state(*req, token);
            bool is_stop = should_stop(*req, token);
            generation_done = is_stop || static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (!generation_done)
                return 1;
        }

        auto saved_req = async_graph_req_;

        // Burst-hybrid speculation: keep the captured graph + block-table
        // buffer parked so the next burst of this request rearms instead of
        // recapturing (~10-20 ms per capture). Fully torn down on request
        // finish or when a different request launches.
        const bool park = saved_req && spec_ngram_enabled_(*saved_req) && !generation_done;
        if (park) {
            async_parked_req_id_ = saved_req->id;
        } else {
            async_graph_runner_.cleanup();
            if (async_d_block_tables_) {
                IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
                async_d_block_tables_ = nullptr;
            }
            if (async_d_banned_tokens_) {
                IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
                async_d_banned_tokens_ = nullptr;
            }
            async_parked_req_id_ = -1;
        }
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;

        if (generation_done) {
            finish_request(saved_req);
            return -1;
        }

        IMP_LOG_DEBUG("AsyncGraphLoop: graph tokens exhausted, continuing with step decode");
    }

    // Clean up stale async graph state
    if (async_graph_req_ && !async_graph_runner_.is_setup()) {
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;
    }

    return 0;
}

// =====================================================================
// step_schedule — call scheduler, reconfigure green contexts
// Returns true if there is work to do.
// =====================================================================

bool Engine::step_schedule() {
    sched_prefill_batch_.clear();
    sched_decode_batch_.clear();
    scheduler_->schedule(sched_prefill_batch_, sched_decode_batch_);

    if (sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
        return false;
    }

    // Dynamic Green Context SM reconfiguration
    if (config_.use_green_contexts && green_ctx_.is_available() && green_ctx_.has_green_contexts()) {
        float target_ratio = config_.green_ctx_prefill_ratio;
        if (sched_prefill_batch_.empty() && !sched_decode_batch_.empty()) {
            target_ratio = 0.0f;
        } else if (!sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
            target_ratio = 1.0f;
        }
        if (std::abs(target_ratio - green_ctx_.prefill_ratio()) > 0.1f) {
            green_ctx_.reconfigure(target_ratio);
        }
    }

    return true;
}

// =====================================================================
// supports_chunked_prefill_ / resolve_prefill_chunk_size_
// Whether the model arch + KV dtype combination supports chunked prefill.
// Returns true for full-attention models (Qwen3, Llama, Mistral) and hybrid
// GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) with FP16, FP8, or
// NVFP4 KV cache. Returns false for SWA models (Gemma-3/4, Llama-4) and
// sub-byte KV dtypes (INT4, TurboQuant) lacking gather kernels.
// =====================================================================

bool Engine::supports_chunked_prefill_() const {
    if (!model_)
        return false;
    const auto& cfg = model_->config();
    // Out-of-scope archs. Hybrid GDN+MoE / Mamba2+MoE archs (QWEN35*,
    // QWEN36_MOE, NEMOTRON_H_MOE) ARE supported. Gemma-4 (SWA + dual
    // head_dim 256/512) is now supported via:
    //   - cuBLAS softmax sliding_window param (PR feat(attn): sliding_window),
    //   - per-layer dispatch through the rectangular cuBLAS prefill path
    //     (every layer call uses its own nh/nkv/hd from layer-local vars).
    // GEMMA3 (SWA, uniform head_dim/kv_heads, sliding_window_pattern) reuses the
    // same per-layer cuBLAS sliding_window dispatch as GEMMA4 and passes the
    // uniformity gates below; verified coherent on gemma-3-12b-it-Q4_K_M.
    if (model_->profile().is_llama4) return false;       // MoE + SWA, untested
    // Per-layer attention shape uniformity gate. Hybrid archs (QWEN35*, QWEN36_MOE,
    // NEMOTRON_H_MOE) populate n_kv_heads_per_layer with zeros for non-attention
    // layers — uniformity here means all *nonzero* values agree. Truly heterogeneous
    // shapes (Gemma-4 dual head_dim 256/512) are now allowed because the chunked
    // path dispatches per-layer with the correct nh/nkv/hd from layer-local vars.
    auto first_nonzero_int = [](const std::vector<int>& v) -> int {
        for (int x : v) if (x > 0) return x;
        return 0;
    };
    auto any_nonzero_differs = [](const std::vector<int>& v, int ref) -> bool {
        for (int x : v) if (x > 0 && x != ref) return true;
        return false;
    };
    if (!cfg.n_kv_heads_per_layer.empty()) {
        int ref = first_nonzero_int(cfg.n_kv_heads_per_layer);
        if (ref > 0 && any_nonzero_differs(cfg.n_kv_heads_per_layer, ref)) {
            // Allow Gemma-4: per-layer dispatch covers heterogeneous shapes.
            if (cfg.arch != ModelArch::GEMMA4) return false;
        }
    }
    if (!cfg.head_dim_per_layer.empty()) {
        int ref = first_nonzero_int(cfg.head_dim_per_layer);
        if (ref > 0 && any_nonzero_differs(cfg.head_dim_per_layer, ref)) {
            if (cfg.arch != ModelArch::GEMMA4) return false;
        }
    }
    // KV dtypes wired through paged_kv_gather: FP16, FP8_E4M3, NVFP4, MXFP4_KV, INT4.
    // INT8 and TurboQuant variants would need their own gather kernels.
    if (kv_cache_raw_) {
        QType kvt = kv_cache_raw_->qtype();
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3 &&
            kvt != QType::NVFP4 && kvt != QType::MXFP4_KV && kvt != QType::INT4)
            return false;
    }
    return true;
}

int Engine::resolve_prefill_chunk_size_() const {
    int explicit_val = config_.prefill_chunk_size;
    if (explicit_val < 0) {
        // Default 2048 (was 512 until 2026-06-11). Chunked prefill re-reads
        // every weight once per chunk, so the memory-bound GEMMs pay the
        // weight traffic per chunk: at pp4096, 512-token chunks cost
        // NVFP4-MoE −43% prefill vs 2048 (14.8k -> 26.2k tok/s) and dense
        // −19% at pp2048 (17.6k -> 21.7k). chunk=0 measured equal to 2048,
        // so 2048 is the sweet spot that still bounds workspace VRAM and
        // multi-request interleaving latency. Hybrids stay safe: step_prefill
        // clamps to executor max_tokens (256/512), and step_prefill_one
        // clamps n × ctx_len into the attn-scores capacity.
        return supports_chunked_prefill_() ? 2048 : 0;
    }
    if (explicit_val == 0)
        return 0;
    // explicit_val > 0
    if (!supports_chunked_prefill_()) {
        IMP_LOG_WARN(
            "prefill_chunk_size=%d ignored: arch=%d / kv_dtype=%d not in chunked-prefill scope; using 0",
            explicit_val, (int)model_->config().arch,
            kv_cache_raw_ ? (int)kv_cache_raw_->qtype() : -1);
        return 0;
    }
    return explicit_val;
}

// =====================================================================
// begin/end_perplexity_capture — chunked-prefill-aware teacher forcing
// (imp_perplexity). See engine.h for the contract.
// =====================================================================

bool Engine::begin_perplexity_capture(const int32_t* tokens, int n) {
    if (ppl_capture_.active || !tokens || n < 2 || !executor_) {
        return false;
    }
    if (cudaMalloc(&ppl_capture_.d_tokens, static_cast<size_t>(n) * sizeof(int32_t)) != cudaSuccess) {
        ppl_capture_.d_tokens = nullptr;
        return false;
    }
    if (cudaMalloc(&ppl_capture_.d_nll, static_cast<size_t>(n) * sizeof(double)) != cudaSuccess) {
        cudaFree(ppl_capture_.d_tokens);
        ppl_capture_.d_tokens = nullptr;
        ppl_capture_.d_nll = nullptr;
        return false;
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpy(ppl_capture_.d_tokens, tokens,
                                  static_cast<size_t>(n) * sizeof(int32_t), cudaMemcpyHostToDevice));
    IMP_CUDA_CHECK_LOG(cudaMemset(ppl_capture_.d_nll, 0, static_cast<size_t>(n) * sizeof(double)));
    ppl_capture_.n = n;
    ppl_capture_.active = true;
    return true;
}

bool Engine::end_perplexity_capture(double* out_ppl) {
    if (!ppl_capture_.active) {
        return false;
    }
    const int n = ppl_capture_.n;
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());

    // Fixed-order host reduction over per-position NLLs (bit-reproducible —
    // same contract as GraphExecutor::perplexity_nll).
    std::vector<double> h_nll_pos(static_cast<size_t>(n), 0.0);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h_nll_pos.data(), ppl_capture_.d_nll,
                                  static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(ppl_capture_.d_tokens);
    cudaFree(ppl_capture_.d_nll);
    ppl_capture_ = PplCapture{};

    // IMP_PPL_DUMP=1: sparse per-position NLL (first 16, every 16th, tail).
    // IMP_PPL_DUMP=full: every position — needed for cross-mode forensics;
    // the sparse form hid that per-position values diverge between chunked
    // and single-shot runs (#655).
    if (const char* dump = getenv("IMP_PPL_DUMP")) {
        const bool full = (strcmp(dump, "full") == 0);
        fprintf(stderr, "[PPL-DUMP] per-pos nll:");
        for (int i = 0; i < n - 1; ++i) {
            if (full || i < 16 || i % 16 == 0 || i > n - 6)
                fprintf(stderr, " [%d]=%.3f", i, h_nll_pos[i]);
        }
        fprintf(stderr, "\n");
    }
    double h_nll = 0.0;
    for (int i = 0; i < n - 1; ++i)
        h_nll += h_nll_pos[i];
    double ppl = std::exp(h_nll / static_cast<double>(n - 1));
    IMP_LOG_INFO("perplexity_nll: n=%d  mean_nll=%.4f  PPL=%.4f", n, h_nll / (n - 1), ppl);
    if (out_ppl)
        *out_ppl = ppl;
    return true;
}

// =====================================================================
// step_prefill — process all prefill requests
// =====================================================================

void Engine::step_prefill(cudaStream_t stream) {
    int resolved    = resolve_prefill_chunk_size_();
    int effective_chunk = (resolved > 0) ? resolved : executor_->max_tokens();
    // Hard cap: chunk size must never exceed the executor's max_tokens
    // (which is itself capped to 256 for SSM/GDN+MoE hybrids and 512 for
    // dense GDN to bound workspace VRAM). Without this clamp, a server-side
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

    for (auto& req : sched_prefill_batch_) {
        step_prefill_one(req, effective_chunk, stream);
        kv_manager_->touch(req->id);
    }
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
bool Engine::prefill_allocate_kv_blocks_(std::shared_ptr<Request>& req, int kv_bs,
                                         int total_input, int effective_chunk,
                                         int& offset, int& chunk_len, bool& is_last_chunk,
                                         int& ctx_len, cudaStream_t pf_stream) {
    int num_blocks = (ctx_len + kv_bs - 1) / kv_bs;
    int prefix_reused = 0;
    int existing = static_cast<int>(kv_manager_->block_table(req->id).size());

    // Perplexity capture must forward EVERY position — a prefix-cache hit
    // skips the reused prefix's forward, leaving those NLL slots at 0.
    if (kv_manager_->prefix_caching_enabled() && existing == 0 && offset == 0 &&
        !ppl_capture_.active) {
        prefix_reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens);
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
                IMP_LOG_INFO("PrefixCache: seq %d skipping %d/%d prefill tokens (%d blocks reused)",
                             req->id, skip_tokens, total_input, prefix_reused);
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
                kv_manager_->free_sequence(req->id);
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
bool Engine::prefill_upload_metadata_(std::shared_ptr<Request>& req,
                                      const std::vector<int>& block_table,
                                      int chunk_len, int offset, int ctx_len,
                                      cudaStream_t pf_stream,
                                      int32_t*& d_token_ids, int*& d_positions,
                                      int*& d_block_tables, int*& d_context_lens,
                                      bool& pf_pool_used) {
    d_token_ids = nullptr;
    d_positions = nullptr;
    d_block_tables = nullptr;
    d_context_lens = nullptr;
    pf_pool_used = false;

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
        d_context_lens = d_pf_context_lens_;
        pf_pool_used = true;
    } else {
        if (!check(cudaMallocAsync(&d_token_ids, chunk_len * sizeof(int32_t), pf_stream),
                   "malloc token_ids") ||
            !check(cudaMallocAsync(&d_positions, chunk_len * sizeof(int), pf_stream), "malloc positions") ||
            !check(cudaMallocAsync(&d_block_tables, block_table.size() * sizeof(int), pf_stream),
                   "malloc block_tables") ||
            !check(cudaMallocAsync(&d_context_lens, sizeof(int), pf_stream), "malloc context_lens")) {
            if (d_token_ids)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, pf_stream));
            if (d_positions)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, pf_stream));
            if (d_block_tables)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, pf_stream));
            if (d_context_lens)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, pf_stream));
            kv_manager_->free_sequence(req->id);
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
    if (pf_staging_evt_ && (h_pf_token_ids_ || h_pf_positions_) && chunk_len <= config_.max_seq_len)
        IMP_CUDA_CHECK_LOG(cudaEventSynchronize(pf_staging_evt_));
    if (h_pf_token_ids_ && chunk_len <= config_.max_seq_len) {
        memcpy(h_pf_token_ids_, req->input_tokens.data() + offset, chunk_len * sizeof(int32_t));
        check(cudaMemcpyAsync(d_token_ids, h_pf_token_ids_, chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    } else {
        check(cudaMemcpyAsync(d_token_ids, req->input_tokens.data() + offset, chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    }

    if (h_pf_positions_ && chunk_len <= config_.max_seq_len) {
        for (int i = 0; i < chunk_len; i++)
            h_pf_positions_[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, h_pf_positions_, chunk_len * sizeof(int), cudaMemcpyHostToDevice,
                              pf_stream),
              "memcpy positions");
    } else {
        std::vector<int> positions(chunk_len);
        for (int i = 0; i < chunk_len; i++)
            positions[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, positions.data(), chunk_len * sizeof(int), cudaMemcpyHostToDevice,
                              pf_stream),
              "memcpy positions");
    }

    if (pf_staging_evt_ && (h_pf_token_ids_ || h_pf_positions_) && chunk_len <= config_.max_seq_len)
        IMP_CUDA_CHECK_LOG(cudaEventRecord(pf_staging_evt_, pf_stream));

    check(cudaMemcpyAsync(d_block_tables, block_table.data(), block_table.size() * sizeof(int),
                          cudaMemcpyHostToDevice, pf_stream),
          "memcpy block_tables");
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

    int ctx_len = offset + chunk_len;
    (void)executor_->resize_workspace(chunk_len, pf_stream);

    if (!prefill_allocate_kv_blocks_(req, kv_bs, total_input, effective_chunk, offset, chunk_len,
                                     is_last_chunk, ctx_len, pf_stream)) {
        return;  // caller already set req->status = CANCELLED
    }

    const auto& block_table = kv_manager_->block_table(req->id);

    int32_t* d_token_ids = nullptr;
    int* d_positions = nullptr;
    int* d_block_tables = nullptr;
    int* d_context_lens = nullptr;
    bool pf_pool_used = false;
    if (!prefill_upload_metadata_(req, block_table, chunk_len, offset, ctx_len, pf_stream,
                                  d_token_ids, d_positions, d_block_tables, d_context_lens,
                                  pf_pool_used)) {
        return;  // caller already set req->status = CANCELLED
    }

    // Build InferenceState
    InferenceState state;
    state.token_ids = d_token_ids;
    state.positions = d_positions;
    state.n_tokens = chunk_len;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_block_tables;
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
    if ((req->json_mode || !req->json_schema.empty()) && !req->constraints) {
        req->constraints = constraints_checkout_(req->json_schema);
        req->constraints->prepare(req->json_mode, req->json_schema, model_->tokenizer(),
                                  req->has_tools, req->tpl_family,
                                  /*thinking_open=*/req->in_think_block);
    }
    if (req->constraints) {
        state.json_constrainer = req->constraints->json_constrainer();
        state.schema_constrainer = req->constraints->schema_constrainer();
    }

    // Penalties
    upload_penalties(*req, state, pf_stream);

    // Recurrent state (SSM/GDN)
    // Reset on the first chunk of a new request so previous-request state
    // doesn't leak in.  Subsequent chunks must NOT reset — the recurrent
    // state built during earlier chunks must carry forward.
    fill_recurrent_state(*req, state, /*reset=*/(offset == 0), pf_stream);

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
        const bool can_capture = prefill_graph_enabled && pf_pool_used && config_.use_cuda_graphs &&
                                 !executor_->nvfp4_dequant_uncapturable();
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
            executor_->forward_logits(state, logits_out, pf_stream);
        }

        // Teacher-forced NLL for this chunk's positions (imp_perplexity).
        // Runs eagerly after the (possibly graph-replayed) forward; hidden_
        // holds exactly this chunk and nothing reads logits_ afterwards.
        if (ppl_capture_.active) {
            executor_->perplexity_nll_partial(ppl_capture_.d_tokens, ppl_capture_.n, offset,
                                              chunk_len, ppl_capture_.d_nll, pf_stream);
        }

        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
        }

        req->prefill_offset = offset + chunk_len;
        IMP_LOG_DEBUG("Chunked prefill: req %d chunk [%d, %d) of %d", req->id, offset, offset + chunk_len,
                      total_input);
    } else {
        // Last chunk: forward + sample
        int32_t next_token;
        bool use_event_sync = (h_sample_pinned_ != nullptr && executor_->d_sample_result() != nullptr &&
                               (state.temperature <= 0.0f || state.top_k == 1) && !req->logprobs &&
                               !state.json_constrainer && !state.schema_constrainer);

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

            sample_greedy_device(last_logits, executor_->d_sample_result(), h_sample_pinned_, pf_stream);

            if (!prefill_done_)
                (void)prefill_done_.create();
            cudaEventRecord(prefill_done_, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }

            cudaEventSynchronize(prefill_done_);
            next_token = *h_sample_pinned_;
        } else if (req->logprobs) {
            executor_->forward_logits(state, prefill_logits_out, pf_stream);
            auto sampled = executor_->sample_from_logits(prefill_logits_out, state, pf_stream);
            next_token = sampled[0];

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }
        } else {
            next_token = executor_->forward(state, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }
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
            executor_->perplexity_nll_partial(ppl_capture_.d_tokens, ppl_capture_.n, offset,
                                              chunk_len, ppl_capture_.d_nll, pf_stream);
        }

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        Tokenizer* tok = model_->tokenizer();
        IMP_LOG_DEBUG("Prefill -> token %d (ctx=%d): id=%d [%s]", (int)req->output_tokens.size(),
                      req->context_len(), next_token, tok->decode_token(next_token).c_str());

        // MTP prefill: populate MTP KV cache with all prompt position hidden
        // states so the head enters decode with the same context as the main
        // model. Only the LAST chunk (where we have the complete tail of the
        // prompt in executor's hidden_ buffer) and only the non-chunked path
        // for now (chunked prefill would need per-chunk capture). Cost: ~n
        // extra MTP forwards, one-time per session.
        if (mtp_spec_decode_enabled() && offset == 0 && req->prefill_offset == 0) {
            // executor's hidden_ buffer holds [chunk_len, d_model] FP16 right
            // after forward_logits; that matches the whole prompt when
            // offset==0 and is_last_chunk==true.
            const int n_prompt = chunk_len;
            imp::Tensor h_view = executor_->view_hidden(n_prompt);
            if (h_view.data != nullptr) {
                mtp_prefill_prompt(req->input_tokens.data(), h_view.data, n_prompt);
            }
        }

        // Update constraint FSM
        if (req->constraints)
            req->constraints->update(next_token);

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        } else {
            req->status = RequestStatus::DECODING;
            if (kv_manager_->prefix_caching_enabled()) {
                kv_manager_->register_block_hashes(req->id, req->input_tokens);
            }
        }
    }
}

// =====================================================================
// step_decode — process all decode requests (batched)
// =====================================================================

void Engine::step_decode(cudaStream_t dec_stream) {
    auto& decode_batch = sched_decode_batch_;
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // n-gram (prompt-lookup) speculation: when a draft is available, the
    // verify step replaces this decode step entirely (it allocates its own
    // KV blocks and emits accepted tokens). Falls through to the normal
    // path on a draft miss or when any gate fails.
    if (decode_batch.size() == 1 && spec_ngram_enabled_(*decode_batch[0])) {
        spec_maybe_rearm_(*decode_batch[0]);
        if (spec_ngram_gates_ok_(*decode_batch[0])) {
            if (step_spec_verify_(decode_batch[0], dec_stream))
                return;
        } else if (decode_batch[0]->spec_ngram_given_up &&
                   spec_burst_launch_ok_(*decode_batch[0])) {
            // Given-up request: hand it straight to the loop (no eager probe
            // step). Doomed = acceptance verdict final → run unbounded;
            // otherwise bounded so spec_maybe_rearm_ can re-probe later.
            auto& sreq = decode_batch[0];
            const int lim =
                sreq->spec_acceptance_doomed ? 0 : runtime_config_.speculative.burst;
            if (try_launch_async_graph_loop(sreq, sreq->output_tokens.back(), dec_stream, lim))
                return;
        } else if (decode_batch[0]->think_budget > 0.0f && decode_batch[0]->in_think_block &&
                   spec_ngram_gates_ok_(*decode_batch[0], /*ignore_think=*/true) &&
                   spec_burst_launch_ok_(*decode_batch[0]) &&
                   // Budget exhausted → the EAGER step must run: it forces
                   // the </think> token. Launching the loop here instead
                   // produced 1-token budget-stopped loops with a full
                   // recapture each, and </think> was never forced.
                   !think_logic::should_force_think_end(
                       decode_batch[0]->think_budget, think_end_id_, decode_batch[0]->max_tokens,
                       decode_batch[0]->output_tokens, think_start_id_,
                       decode_batch[0]->started_in_think)) {
            // Budgeted think interior: the loop handles the budget device-
            // side in bounded bursts so the host catches the think→answer
            // transition and resumes verification in the draft-rich answer
            // region. Fixed burst (the miss counter stays untouched — these
            // are not draft misses, and inflating it would instantly trip
            // give-up on the first real probe after </think>).
            auto& sreq = decode_batch[0];
            const int think_burst =
                std::min(32, runtime_config_.speculative.burst > 0
                                 ? runtime_config_.speculative.burst
                                 : 32);
            if (try_launch_async_graph_loop(sreq, sreq->output_tokens.back(), dec_stream,
                                            think_burst))
                return;
        } else {
            // One-time diagnosis: say WHY speculation is inactive (gates are
            // silent otherwise and a misconfigured request looks identical
            // to a draft-poor one).
            static bool s_gate_logged = false;
            if (!s_gate_logged && !decode_batch[0]->spec_ngram_given_up) {
                s_gate_logged = true;
                const auto& r = *decode_batch[0];
                IMP_LOG_INFO(
                    "spec-ngram: gates failed (temp=%.2f top_k=%d rep_pen=%.2f freq=%.2f "
                    "pres=%.2f dry=%.2f mirostat=%d bias=%zu logprobs=%d json=%d schema=%d "
                    "think_budget=%.2f ssm=%d gdn=%d moe=%d mtp=%d chunked_prefill=%d)",
                    r.temperature, r.top_k, r.repetition_penalty, r.frequency_penalty,
                    r.presence_penalty, r.dry_multiplier, r.mirostat, r.logit_bias.size(),
                    (int)r.logprobs, (int)r.json_mode, (int)!r.json_schema.empty(), r.think_budget,
                    (int)(ssm_state_ != nullptr), (int)(gdn_state_ != nullptr),
                    (int)model_->profile().is_moe, (int)mtp_spec_decode_enabled(),
                    (int)supports_chunked_prefill_());
            }
        }
    }

    // SSM/GDN: limit decode batch to 1 sequence
    if ((ssm_state_ || gdn_state_) && decode_batch.size() > 1) {
        decode_batch.resize(1);
    }

    // Cap at configured max batch size
    const int max_bs = runtime_config_.runtime.max_batch_size;
    if (max_bs > 0 && static_cast<int>(decode_batch.size()) > max_bs) {
        decode_batch.resize(max_bs);
    }

    // Allocate new KV blocks where needed
    valid_decode_.clear();
    auto& valid_decode = valid_decode_;

    for (auto& req : decode_batch) {
        int ctx_len = req->context_len();
        int blocks_needed = (ctx_len + kv_bs - 1) / kv_bs;
        const auto& block_table = kv_manager_->block_table(req->id);
        int blocks_have = static_cast<int>(block_table.size());

        if (blocks_needed > blocks_have) {
            int new_block = kv_manager_->append_block(req->id);
            if (new_block < 0) {
                // KV exhausted: append_block already reclaimed cached blocks, so
                // the free pool AND all reclaimable cached blocks are empty. The
                // old fallback evicted an LRU sequence — but every lru_order_
                // entry is LIVE and imp has no recompute path, so evicting one
                // (a current-batch member, or a still-active sequence beyond
                // max_batch_size) silently corrupted it (use-after-free once it
                // ran). Reject-newest instead: cancel THIS sequence and leave the
                // others' KV intact. StreamingLLM auto-enable (above) already
                // handles the graceful FP16 case before we reach here.
                kv_manager_->free_sequence(req->id);
                req->status = RequestStatus::CANCELLED;
                continue;
            }
        }

        // Auto-activate StreamingLLM when KV cache is nearly exhausted.
        // Only fires once (guards on !streaming_kv_enabled) and only for FP16
        // KV — quantized variants don't support sentinel-block skipping yet.
        if (!config_.streaming_kv_enabled && config_.streaming_kv_auto) {
            auto st = kv_manager_->stats();
            if (st.total_blocks > 0 && st.free_blocks < st.total_blocks / 10) {
                if (kv_cache_raw_ && kv_cache_raw_->qtype() == QType::F16) {
                    config_.streaming_kv_enabled = true;
                    int n_sinks = (config_.streaming_kv_n_sinks > 0) ? config_.streaming_kv_n_sinks : 4;
                    int win = (config_.streaming_kv_window > 0) ? config_.streaming_kv_window
                                                                : model_->config().sliding_window;
                    if (win <= 0) win = 4096;
                    config_.streaming_kv_window = win;
                    executor_->set_streaming_kv(n_sinks, win);
                    IMP_LOG_WARN(
                        "KV cache >90%% full (%d/%d blocks free) — auto-enabling "
                        "StreamingLLM (sinks=%d, window=%d)",
                        st.free_blocks, st.total_blocks, n_sinks, win);
                    if (config_.use_cuda_graphs) {
                        IMP_LOG_WARN(
                            "Disabling CUDA Graphs (block table mutates with StreamingLLM).");
                        config_.use_cuda_graphs = false;
                    }
                }
            }
        }

        // StreamingLLM smart KV cache: once context exceeds the threshold,
        // free middle blocks while keeping sinks + window. The decode kernel
        // skips the freed (-1 sentinel) slots via its own n_sinks logic.
        if (config_.streaming_kv_enabled) {
            int n_sinks = (config_.streaming_kv_n_sinks > 0) ? config_.streaming_kv_n_sinks : 4;
            int win = (config_.streaming_kv_window > 0) ? config_.streaming_kv_window
                                                        : model_->config().sliding_window;
            if (n_sinks > 0 && win > 0) {
                int threshold = (config_.streaming_kv_threshold > 0) ? config_.streaming_kv_threshold
                                                                     : (n_sinks + win + 2 * kv_bs);
                if (req->context_len() > threshold) {
                    kv_manager_->evict_middle_blocks(req->id, n_sinks, win);
                }
            }
        }
        valid_decode.push_back(req);
    }

    if (!valid_decode.empty()) {
        step_decode_forward(valid_decode, dec_stream);
    }
}

// =====================================================================
// step_decode_forward — build batch, run forward pass, sample, process
// =====================================================================

// Populate the InferenceState for a decode step from the uploaded GPU batch.
// Handles per-seq residual metadata (single-seq fast path vs multi-seq
// per-batch upload), sampling params, decode-step seed, penalties, recurrent
// state, and JSON/schema constrainer attach. Returns needs_logprobs so the
// caller knows whether to capture decode_logits_out for the logprobs pass.
void Engine::decode_build_inference_state_(GPUBatch& gpu_batch,
                                           std::vector<std::shared_ptr<Request>>& valid_decode,
                                           int max_ctx, cudaStream_t dec_stream,
                                           InferenceState& state, bool& needs_logprobs,
                                           bool& needs_json_mode, bool& needs_schema_mode) {
    state.token_ids = gpu_batch.d_token_ids;
    state.positions = gpu_batch.d_positions;
    state.n_tokens = gpu_batch.total_tokens;
    state.n_sequences = gpu_batch.n_sequences;
    state.max_blocks_per_seq = gpu_batch.max_blocks_per_seq;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = gpu_batch.d_block_tables;
    state.context_lens = gpu_batch.d_context_lens;
    state.max_context_len = max_ctx;
    state.is_prefill = false;
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) {
        // Allocate / refresh per-seq residual metadata for this decode step.
        // Slot allocation is idempotent (returns existing slot on re-call).
        const int N = gpu_batch.n_sequences;
        residual_meta_h_seq_ids_.resize(N);
        for (int i = 0; i < N; i++) {
            int sid = valid_decode[i]->id;
            residual_meta_h_seq_ids_[i] = sid;
            kv_manager_->allocate_residual_slot(sid);
        }
        if (N == 1) {
            // Single-seq path: kernel reads ring state from kv_manager's
            // persistent device buffers. Slot is a constant per-request
            // value uploaded into d_kv_slot_buf_[0]; only re-uploads when
            // it changes (i.e. when the active request rotates).
            state.kv_seq_id = valid_decode[0]->id;
            state.h_residual_seq_ids = residual_meta_h_seq_ids_.data();
            int slot_for_req = kv_manager_->residual_slot_of(valid_decode[0]->id);
            if (d_kv_slot_buf_ != nullptr) {
                if (d_kv_slot_last_uploaded_.empty() ||
                    d_kv_slot_last_uploaded_[0] != slot_for_req) {
                    cudaMemcpyAsync(d_kv_slot_buf_, &slot_for_req, sizeof(int),
                                    cudaMemcpyHostToDevice, dec_stream);
                    if (d_kv_slot_last_uploaded_.empty()) d_kv_slot_last_uploaded_.assign(1, -1);
                    d_kv_slot_last_uploaded_[0] = slot_for_req;
                }
                state.d_residual_seq_slots = d_kv_slot_buf_;
            }
        } else {
            // Multi-seq path: build per-batch metadata arrays + upload to
            // a per-step device buffer.
            residual_meta_h_slots_.resize(N);
            residual_meta_h_counts_.resize(N);
            residual_meta_h_widxes_.resize(N);
            for (int i = 0; i < N; i++) {
                int sid = residual_meta_h_seq_ids_[i];
                residual_meta_h_slots_[i] = kv_manager_->residual_slot_of(sid);
                auto rs = kv_manager_->residual_state(sid);
                residual_meta_h_counts_[i] = rs.fill_count;
                residual_meta_h_widxes_[i] = rs.write_idx;
            }
            const size_t meta_bytes = static_cast<size_t>(3) * N * sizeof(int);
            if (cudaMallocAsync(&residual_meta_d_buf_, meta_bytes, dec_stream) == cudaSuccess) {
                int* base = residual_meta_d_buf_;
                cudaMemcpyAsync(base + static_cast<ptrdiff_t>(0) * N, residual_meta_h_slots_.data(),
                                N * sizeof(int), cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + static_cast<ptrdiff_t>(1) * N, residual_meta_h_counts_.data(),
                                N * sizeof(int), cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + static_cast<ptrdiff_t>(2) * N, residual_meta_h_widxes_.data(),
                                N * sizeof(int), cudaMemcpyHostToDevice, dec_stream);
                state.d_residual_seq_slots = base + static_cast<ptrdiff_t>(0) * N;
                state.d_residual_counts = base + static_cast<ptrdiff_t>(1) * N;
                state.d_residual_write_idxes = base + static_cast<ptrdiff_t>(2) * N;
                state.h_residual_seq_ids = residual_meta_h_seq_ids_.data();
            }
        }
    }
    fill_sampling_params(*valid_decode[0], state);

    // Derive per-step seed: mix request seed with output count so each
    // decode step gets a different random draw.  Without this, seed=-1
    // falls back to a fixed 42 on every step, producing identical RNG
    // values and causing repetition loops on long structured outputs.
    state.seed = compute_step_seed(*valid_decode[0]);

    // Penalties (single-sequence only)
    if (gpu_batch.n_sequences == 1) {
        upload_penalties(*valid_decode[0], state, dec_stream);
    }

    // Recurrent state
    fill_recurrent_state(*valid_decode[0], state, false, dec_stream);

    // Check if any request needs logprobs or constrained mode
    needs_logprobs = false;
    needs_json_mode = false;
    needs_schema_mode = false;
    for (const auto& r : valid_decode) {
        if (r->logprobs)
            needs_logprobs = true;
        if (r->json_mode && r->json_schema.empty())
            needs_json_mode = true;
        if (!r->json_schema.empty())
            needs_schema_mode = true;
    }

    // Schema/JSON constraints for decode. Lazily create the per-request
    // manager if needed (decode might be the first step with json_mode).
    // The single-sequence state carries the request's constrainers (the
    // graph-loop / constrained-pipeline launch paths read them from state);
    // batched decode attaches them per row in sample_per_request, so
    // constraints stay enforced at batch>1 (previously they were silently
    // dropped whenever a constrained request shared a decode batch).
    for (auto& r : valid_decode) {
        if ((r->json_mode || !r->json_schema.empty()) && !r->constraints) {
            r->constraints = constraints_checkout_(r->json_schema);
            r->constraints->prepare(r->json_mode, r->json_schema, model_->tokenizer(), r->has_tools,
                                    r->tpl_family, /*thinking_open=*/r->in_think_block);
        }
    }
    if (valid_decode.size() == 1 && valid_decode[0]->constraints) {
        state.schema_constrainer = valid_decode[0]->constraints->schema_constrainer();
        state.json_constrainer = valid_decode[0]->constraints->json_constrainer();
    }
}

void Engine::step_decode_forward(std::vector<std::shared_ptr<Request>>& valid_decode,
                                 cudaStream_t dec_stream) {
    // Switch workspace for decode
    if (executor_->has_decode_workspace() && valid_decode.size() == 1) {
        executor_->use_workspace(1);
    } else {
        if (executor_->active_workspace() == 1)
            executor_->use_workspace(0);
        (void)executor_->resize_workspace(static_cast<int>(valid_decode.size()), dec_stream);
    }

    // Build batched decode
    decode_builder_.reset();

    int max_ctx = 0;
    for (auto& req : valid_decode) {
        int ctx_len = req->context_len();
        max_ctx = std::max(max_ctx, ctx_len);

        int32_t last_token = req->output_tokens.empty() ? req->input_tokens.back()
                                                        : req->output_tokens.back();
        int position = ctx_len - 1;

        const auto& bt = kv_manager_->block_table(req->id);
        decode_builder_.add_decode_sequence(last_token, position, bt.data(), static_cast<int>(bt.size()),
                                            ctx_len);
    }

    Batch batch = decode_builder_.build();

    // Upload to GPU using pre-allocated pool
    GPUBatch gpu_batch;
    if (decode_batch_pool_.is_allocated()) {
        int pool_max = decode_batch_pool_.max_blocks_per_seq();
        if (batch.max_blocks_per_seq < pool_max) {
            int n_seq = batch.n_sequences;
            int old_stride = batch.max_blocks_per_seq;
            size_t needed = static_cast<size_t>(n_seq) * pool_max;
            padded_block_table_.resize(needed);
            std::memset(padded_block_table_.data(), 0, needed * sizeof(int));
            for (int s = 0; s < n_seq; s++) {
                for (int b = 0; b < old_stride; b++) {
                    padded_block_table_[s * pool_max + b] = batch.block_tables[s * old_stride + b];
                }
            }
            batch.block_tables.swap(padded_block_table_);
            batch.max_blocks_per_seq = pool_max;
        }
        gpu_batch = decode_batch_pool_.upload_into_pool(batch, dec_stream);
    } else {
        gpu_batch.upload(batch, dec_stream);
    }

    InferenceState state;
    bool needs_logprobs = false;
    bool needs_json_mode = false;
    bool needs_schema_mode = false;
    decode_build_inference_state_(gpu_batch, valid_decode, max_ctx, dec_stream, state, needs_logprobs,
                                  needs_json_mode, needs_schema_mode);

    // Per-request sampling lambda
    auto sample_per_request = [&](const Tensor& logits) -> std::vector<int32_t> {
        int n = static_cast<int>(valid_decode.size());

        if (n == 1) {
            auto& req = valid_decode[0];
            int32_t tok = executor_->sample_single_from_logits(logits, state, dec_stream);
            if (state.mirostat == 2)
                req->mirostat_mu = state.mirostat_mu;
            return {tok};
        }

        std::vector<int32_t> result(n);
        for (int i = 0; i < n; i++) {
            auto& req = valid_decode[i];
            InferenceState per_state = state;
            fill_sampling_params(*req, per_state);
            // Per-step seed (same fix as single-sequence path)
            per_state.seed = compute_step_seed(*req);
            per_state.penalty_tokens = nullptr;
            per_state.n_penalty_tokens = 0;
            bool req_needs_pen = (req->repetition_penalty != 1.0f || req->frequency_penalty != 0.0f ||
                                  req->presence_penalty != 0.0f);
            if (req_needs_pen && !req->output_tokens.empty() && d_penalty_tokens_) {
                size_t rn = req->output_tokens.size();
                if (rn <= d_penalty_tokens_capacity_) {
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_penalty_tokens_, req->output_tokens.data(),
                                                       rn * sizeof(int32_t), cudaMemcpyHostToDevice,
                                                       dec_stream));
                    per_state.penalty_tokens = d_penalty_tokens_;
                    per_state.n_penalty_tokens = static_cast<int>(rn);
                }
            }
            // Per-row constraint masks: keeps json_schema/json_mode enforced
            // when the request shares a decode batch (the batch-level state
            // carries no constrainers at n>1; sample_single_from_logits
            // applies the mask to this row's logits before sampling).
            if (req->constraints) {
                per_state.schema_constrainer = req->constraints->schema_constrainer();
                per_state.json_constrainer = req->constraints->json_constrainer();
            } else {
                per_state.schema_constrainer = nullptr;
                per_state.json_constrainer = nullptr;
            }
            per_state.n_sequences = 1;
            Tensor seq_logits = logits.slice(i, i + 1);
            result[i] = executor_->sample_single_from_logits(seq_logits, per_state, dec_stream);
            if (per_state.mirostat == 2)
                req->mirostat_mu = per_state.mirostat_mu;
        }
        return result;
    };

    // Execute forward pass (piecewise CUDA Graph: forward in graph,
    // sampling always eager — per-batch-size graph pool avoids
    // re-capture when continuous batching changes batch size)
    std::vector<int32_t> tokens;
    Tensor decode_logits_out;

    const bool profiling = runtime_config_.diagnostics.profile;
    int graph_idx = gpu_batch.n_sequences - 1;
    if (config_.use_cuda_graphs && !profiling && gpu_batch.n_sequences > 0 && graph_idx < kMaxGraphPoolSize &&
        decode_batch_pool_.is_allocated()) {
        auto& graph_runner = decode_graph_pool_[graph_idx];

        // P5 §2.2 M4: pow-2 bucket the max_blocks_per_seq before comparing.
        // The decode_batch_pool_ padding (engine.cpp:2456-2468) already lifts
        // batch.max_blocks_per_seq to pool_max, so in practice this comparison
        // never trips during steady-state decode — but if a future path
        // bypasses the pool (e.g. spec-decode verify), pow-2 bucketing caps
        // the re-capture frequency to log2(max_blocks) events per decode
        // instead of one per 16-token boundary. Re-capture cost is
        // dominated by the eager forward that builds the new graph
        // (~5-10 ms on Qwen3-8B Q8_0) so the bucket pays back across long
        // contexts.
        auto bucket_pow2 = [](int x) -> int {
            if (x <= 1) return 1;
            int b = 1;
            while (b < x) b <<= 1;
            return b;
        };
        const int bucketed_max_blocks = bucket_pow2(gpu_batch.max_blocks_per_seq);
        if (bucketed_max_blocks != last_decode_max_blocks_per_graph_[graph_idx]) {
            // Topology stable across max_blocks growth (same kernels, only
            // grid dims / params differ) — cudaGraphExecUpdate handles this
            // without tearing down the exec + graph mem pool.
            graph_runner.invalidate_for_update();
            last_decode_max_blocks_per_graph_[graph_idx] = bucketed_max_blocks;
        }
        // Graph captures ONLY forward_logits — sampling runs eager after
        Tensor logits_out;
        graph_runner.set_decode_fn(
            [this, &state, &logits_out](cudaStream_t s) { executor_->forward_logits(state, logits_out, s); });
        graph_runner.execute(dec_stream);

        if (logits_out.data == nullptr) {
            logits_out = executor_->get_logits_view(gpu_batch.n_sequences);
        }
        // Eager sampling (handles all modes: greedy, top-k/p, penalties,
        // force_token, constraints, logprobs, mirostat)
        tokens = sample_per_request(logits_out);
        if (needs_logprobs)
            decode_logits_out = logits_out;
    } else {
        executor_->forward_logits(state, decode_logits_out, dec_stream);
        tokens = sample_per_request(decode_logits_out);
    }

    if (!decode_batch_pool_.is_allocated()) {
        gpu_batch.free();
    }

    // Free per-step residual metadata buffer (allocated in step_decode_forward
    // when residual is enabled). cudaFreeAsync orders behind the just-issued
    // forward + sample on dec_stream.
    if (residual_meta_d_buf_ != nullptr) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(residual_meta_d_buf_, dec_stream));
        residual_meta_d_buf_ = nullptr;
    }

    // Phase 3.5 telemetry: measure MTP-draft prediction accuracy without
    // changing generation. Single-sequence only (batch=1 simplifies hidden-
    // state addressing). Skipped when MTP is disabled or the workspace was
    // allocated without attention dims (older callers).
    if (mtp_spec_decode_enabled() && model_ && model_->mtp_.has_value() &&
        model_->mtp_->loaded && gpu_batch.n_sequences == 1 && !tokens.empty()) {
        const int32_t next_token = tokens[0];
        if (mtp_pending_prediction_ >= 0) {
            mtp_accuracy_.total++;
            bool match = (mtp_pending_prediction_ == next_token);
            if (match) mtp_accuracy_.matches++;
            // Optional verbose log: prints (predicted, actual, match) with
            // decoded strings so accept patterns can be analyzed offline.
            const bool s_pattern_log = runtime_config_.diagnostics.mtp_pattern_log;
            if (s_pattern_log) {
                Tokenizer* tok = model_->tokenizer();
                std::string ps = tok ? tok->decode_token(mtp_pending_prediction_) : std::string();
                std::string as = tok ? tok->decode_token(next_token) : std::string();
                IMP_LOG_INFO("MTP-PAT %s pred=%d '%s' actual=%d '%s'",
                             match ? "+" : "-",
                             mtp_pending_prediction_, ps.c_str(),
                             next_token, as.c_str());
            }
        }
        // K-chain measurement: verify pending predictions, then draft fresh.
        // Current position is the index of the token we just emitted (=
        // length of input + already-emitted output minus 1).
        const int cur_pos = valid_decode[0]->context_len() - 1;
        // Verify any pending predictions whose intended position is cur_pos.
        if (!mtp_pending_chain_.empty()) {
            // Drop stale (intended < cur_pos) — shouldn't happen in K=1 path
            // but possible if the engine skips/restarts mid-chain.
            mtp_pending_chain_.erase(
                std::remove_if(mtp_pending_chain_.begin(), mtp_pending_chain_.end(),
                               [cur_pos](const MtpChainEntry& e) {
                                   return e.intended_position < cur_pos;
                               }),
                mtp_pending_chain_.end());
            for (auto it = mtp_pending_chain_.begin(); it != mtp_pending_chain_.end();) {
                if (it->intended_position == cur_pos) {
                    if (static_cast<int>(mtp_chain_accept_.size()) <= it->lookahead) {
                        mtp_chain_accept_.resize(it->lookahead + 1);
                    }
                    mtp_chain_accept_[it->lookahead].total++;
                    if (it->prediction == next_token) {
                        mtp_chain_accept_[it->lookahead].matches++;
                    }
                    // Stage 0 per-width: was the true next token within top-(w+1)?
                    if (static_cast<int>(mtp_chain_accept_w_.size()) <= it->lookahead) {
                        mtp_chain_accept_w_.resize(it->lookahead + 1);
                    }
                    auto& wa = mtp_chain_accept_w_[it->lookahead];
                    wa.total++;
                    int found_rank = kMtpMeasureW;  // sentinel: not in top-W
                    for (int w = 0; w < kMtpMeasureW; ++w) {
                        if (it->topk[w] == next_token) { found_rank = w; break; }
                    }
                    // Cumulative: a hit at rank r counts for all widths ≥ r.
                    for (int w = found_rank; w < kMtpMeasureW; ++w) wa.matches[w]++;
                    it = mtp_pending_chain_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        Tensor h_view = executor_->view_hidden(1);  // [1, d_model] FP16
        if (h_view.data != nullptr) {
            const int hidden_dim = model_->config_.d_model;
            const int vocab_size = model_->config_.vocab_size;

            // Optional: apply the main model's output norm before passing
            // h_prev to MTP. Upstream vllm passes post-RMSNorm hidden states
            // in some MTP variants; gate by env so we can A/B.
            const bool s_pre_norm_h = runtime_config_.diagnostics.mtp_prenorm_h;
            const void* h_for_mtp = h_view.data;
            // Scratch buffer for the normalized variant (allocated once).
            static void* s_h_normed = nullptr;
            if (s_pre_norm_h) {
                if (s_h_normed == nullptr) {
                    cudaMalloc(&s_h_normed, hidden_dim * sizeof(__half));
                }
                int64_t hd_shape[2] = {1, hidden_dim};
                Tensor in_view (h_view.data, QType::F16, 2, hd_shape, true);
                Tensor out_view(s_h_normed,  QType::F16, 2, hd_shape, true);
                imp::rmsnorm(in_view, model_->output_norm(), out_view,
                             model_->config_.rms_norm_eps, decode_stream(),
                             model_->config_.norm_weight_offset);
                h_for_mtp = s_h_normed;
            }

            // K-chain draft. K=mtp_spec_k_. For each step k=0..K-1:
            //   - input: (prev_token_k, h_prev_k)
            //   - output: prediction_k, ws.d_h_final updated for next iter
            //   - chain: prev_token_{k+1} = prediction_k, h_prev_{k+1} = d_h_final
            //
            // Cache roll-back: each chained call appends to MTP KV cache. Only
            // the first chain step's append should persist (it represents what
            // the main model actually does next). Roll back to mtp_pos_saved
            // after K-1 speculative steps so the real cache stays aligned.
            auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
            const int K = std::max(1, mtp_spec_k_);
            const int mtp_pos_before = ws->mtp_pos;
            int chain_prev_tok = next_token;
            const void* chain_h_prev = h_for_mtp;
            for (int k = 0; k < K; ++k) {
                int prediction = -1;
                int topk[Engine::kMtpMeasureW] = {-1, -1, -1, -1};
                if (!mtp_draft_one(chain_prev_tok, chain_h_prev, hidden_dim, vocab_size,
                                    &prediction, topk, Engine::kMtpMeasureW)) {
                    break;
                }
                MtpChainEntry entry{prediction, k, cur_pos + 1 + k, {}};
                for (int w = 0; w < Engine::kMtpMeasureW; ++w) entry.topk[w] = topk[w];
                mtp_pending_chain_.push_back(entry);
                // For k=0 only, also feed pending_prediction_ (legacy 1-step
                // accuracy counter remains in sync with chain_accept_[0]).
                if (k == 0) mtp_pending_prediction_ = prediction;
                // Chain: next iter uses this prediction + the MTP's own h_final.
                chain_prev_tok = prediction;
                chain_h_prev   = ws->d_h_final;
            }
            // Roll back the speculative cache writes from K-1 chained steps.
            // The first chained step (k=0) IS the real "next step" prediction
            // and matches what would have been drafted in K=1 mode — keep it.
            ws->mtp_pos = std::min(ws->mtp_pos, mtp_pos_before + 1);
        } else {
            mtp_pending_prediction_ = -1;
        }
    } else {
        mtp_pending_prediction_ = -1;  // batch>1 or MTP off → clear pending
    }

    // Process outputs: logprobs extraction + token distribution
    step_decode_process_outputs(valid_decode, tokens, decode_logits_out, needs_logprobs, needs_json_mode,
                                needs_schema_mode, dec_stream);
}

// =====================================================================
// step_decode_process_outputs — extract logprobs, distribute tokens,
//                                try async graph loop
// =====================================================================

void Engine::step_decode_process_outputs(std::vector<std::shared_ptr<Request>>& valid_decode,
                                         const std::vector<int32_t>& tokens, const Tensor& decode_logits_out,
                                         bool needs_logprobs, bool needs_json_mode, bool needs_schema_mode,
                                         cudaStream_t dec_stream) {
    Tokenizer* tok = model_->tokenizer();

    // Extract logprobs
    if (needs_logprobs && decode_logits_out.data != nullptr) {
        int vocab_size = static_cast<int>(decode_logits_out.shape[decode_logits_out.ndim - 1]);
        int n_lp = 0;
        for (const auto& r : valid_decode)
            if (r->logprobs)
                n_lp++;
        executor_->ensure_logits_pinned(vocab_size * n_lp);
        float* h_base = executor_->h_logits_pinned();

        int slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            if (!valid_decode[i]->logprobs)
                continue;
            const float* d_logits = static_cast<const float*>(decode_logits_out.data) +
                                    static_cast<size_t>(i) * vocab_size;
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_base + static_cast<size_t>(slot) * vocab_size, d_logits,
                                               vocab_size * sizeof(float), cudaMemcpyDeviceToHost,
                                               dec_stream));
            slot++;
        }
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(dec_stream));

        slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            auto& req = valid_decode[i];
            if (!req->logprobs)
                continue;

            float* h_logits = h_base + static_cast<size_t>(slot) * vocab_size;
            req->output_logprobs.push_back(
                build_logprob_info(h_logits, vocab_size, tokens[i], req->top_logprobs, tok));
            slot++;
        }
    }

    // Distribute sampled tokens back to requests
    for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
        auto& req = valid_decode[i];
        int32_t next_token = tokens[i];

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        IMP_LOG_DEBUG("Decode step %d (ctx=%d, pos=%d): id=%d [%s]", (int)req->output_tokens.size(),
                      req->context_len(), req->context_len() - 1, next_token,
                      tok->decode_token(next_token).c_str());

        // Advance the FSM before finish_request — finishing returns the
        // request's ConstraintManager to the engine pool.
        if (req->constraints)
            req->constraints->update(next_token);

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        }

        kv_manager_->touch(req->id);
    }

    // Try async graph loop after first decode step.
    // Think budget is now handled device-side in post_decode_step_kernel.
    if (decode_graph_pool_[0].is_ready() && valid_decode.size() == 1 && !offload_mgr_ &&
        config_.use_cuda_graphs &&
        // A PARKED runner (burst-hybrid speculation) is setup but idle — it
        // must be allowed back in here, or bursts only ever fire once. A park
        // for a DIFFERENT request is torn down inside the launch.
        (!async_graph_runner_.is_setup() || async_parked_req_id_ >= 0) &&
        !needs_logprobs && !needs_json_mode && !needs_schema_mode) {
        auto& dreq = valid_decode[0];
        // forward_decode_async only implements banned_tokens + rep/freq/presence
        // penalties device-side. Any sampling feature that requires host-side
        // logic (logit_bias, mirostat, typical_p, min_p, DRY) would be silently
        // skipped inside the captured graph — stay on the eager path instead.
        const bool async_compatible = dreq->logit_bias.empty() && dreq->mirostat == 0 &&
                                      dreq->dry_multiplier == 0.0f && dreq->min_p == 0.0f &&
                                      dreq->typical_p >= 1.0f &&
                                      // Phase 3.5 MTP telemetry hooks the per-step path; the async
                                      // conditional-graph loop bypasses it. Stay eager when MTP is on
                                      // so accuracy measurement covers the whole generation.
                                      !mtp_spec_decode_enabled() &&
                                      // n-gram speculation: the loop runs in bounded bursts
                                      // (miss_burst) so the host can probe for drafts between
                                      // bursts; with miss_burst=0 the loop stays blocked while
                                      // speculation is engaged (legacy eager-miss behavior).
                                      (!spec_ngram_enabled_(*dreq) || dreq->spec_ngram_given_up ||
                                       runtime_config_.speculative.miss_burst > 0);
        // Text-fallback think tracking: when <think>/</think> are not single
        // control-token IDs (NVFP4 SafeTensors for Qwen3 / Qwen3.5 / Qwen3.6
        // ship them as added_tokens with special=False, leaving think_end_id_
        // at -1), the graph kernel's device-side `in_think` predicate is
        // permanently false. eos_id then terminates the loop the moment the
        // model emits <|endoftext|> after an empty </think>, returning a
        // 3-token completion ("<", "answer", ">") with the actual answer
        // never sampled. Host-side track_think_state runs literal-string
        // matching per token but only fires in the eager step_decode path,
        // so route through it whenever the request started inside a think
        // block on a model that lacks single-token think markers.
        const bool needs_eager_for_text_think = dreq->in_think_block && think_end_id_ < 0;
        if (async_compatible && !needs_eager_for_text_think && dreq->status == RequestStatus::DECODING &&
            !dreq->output_tokens.empty()) {
            int32_t last_token = dreq->output_tokens.back();
            // Burst-hybrid speculation: bound the loop so the host can probe
            // for drafts when it returns. Engaged requests use the short
            // miss_burst; given-up requests the long re-probe burst (0 =
            // run to completion).
            int spec_limit = 0;
            if (spec_ngram_enabled_(*dreq)) {
                spec_limit = dreq->spec_ngram_given_up ? runtime_config_.speculative.burst
                                                       : spec_effective_miss_burst_(*dreq);
                if (spec_limit < 0) spec_limit = 0;
            }
            // #754 RESOLVED: streaming requests run the loop too, since
            // step_async_graph_resume now polls the mapped ring buffer and
            // surfaces tokens per-step while the burst is still running —
            // SSE stays real per-token (previously the blocking per-burst
            // sync delivered tokens only in burst-sized groups, so streaming
            // had to stay on per-step decode and paid the +27-45% loop win).
            //
            // F-A2: a request with speculation off would run the UNBOUNDED
            // on-device loop (spec_limit == 0) all the way to max_tokens —
            // is_cancelled()/timeout are only polled between bursts, so a
            // client disconnect couldn't interrupt it and burned a full
            // generation. Bound it to runtime.decode_burst so the worker
            // regains control to re-poll cancellation; output is identical
            // (same decode, chunked + relaunched, exactly like the
            // speculation burst path). Speculation keeps its own miss_burst
            // limit; <=0 restores the old unbounded behavior.
            int launch_limit = spec_limit;
            // Bound only in the default (non-deterministic) serving mode.
            // The fully-on-device unbounded loop is the one decode path that
            // is greedy bit-reproducible run-to-run (no host re-entry);
            // chunking it would make non-streaming greedy output
            // non-reproducible, breaking the determinism.md eval guarantee.
            // Deterministic mode runs to completion and never needs
            // mid-burst cancellation, so keep it unbounded there. In
            // production the model is already non-deterministic, so bounding
            // costs no reproducibility and buys cancel responsiveness.
            if (launch_limit == 0 && runtime_config_.runtime.decode_burst > 0 &&
                !runtime_config_.runtime.deterministic)
                launch_limit = runtime_config_.runtime.decode_burst;
            // Admission-aware burst: while another request is waiting (still
            // pending admission or mid-prefill), Engine::step short-circuits
            // to the resume path for the whole burst — the waiting request's
            // prefill only advances between bursts, inflating its TTFT by
            // ~0.5-1 s per 128-token burst. Shorten the burst so scheduling
            // work interleaves every few tokens; the full burst returns when
            // nothing is waiting. Deterministic mode is exempt (same
            // reasoning as decode_burst above — reproducible evals are
            // single-stream, nothing ever waits).
            if (!runtime_config_.runtime.deterministic &&
                (scheduler_->has_pending() || !sched_prefill_batch_.empty())) {
                constexpr int kBusyBurst = 16;
                if (launch_limit == 0 || launch_limit > kBusyBurst)
                    launch_limit = kBusyBurst;
            }
            try_launch_async_graph_loop(dreq, last_token, dec_stream, launch_limit);
        }
    }

    // Constrained requests (json_mode / json_schema) can't run the conditional
    // loop (the grammar FSM is host-side) — launch the pipelined constrained
    // decode instead: per tick the host enqueues mask+sample AND the next
    // forward, hiding FSM/mask latency under GPU compute. masked_sample_async
    // covers banned tokens + greedy/top-k/top-p only — penalties or any
    // host-side sampling feature stays on the eager path.
    if (decode_graph_pool_[0].is_ready() && valid_decode.size() == 1 && !offload_mgr_ &&
        config_.use_cuda_graphs && !async_graph_runner_.is_setup() && !cpipe_.active && !needs_logprobs &&
        (needs_json_mode || needs_schema_mode)) {
        auto& dreq = valid_decode[0];
        // rep/freq/presence penalties and think_budget ARE supported (uploaded /
        // forced per tick like the eager path) — the server defaults
        // (repetition_penalty 1.05, think_budget 0.5) must not block the
        // pipeline. min_p/typical_p/mirostat/DRY/logit_bias stay eager.
        const bool pipeline_compatible =
            dreq->logit_bias.empty() && dreq->mirostat == 0 && dreq->dry_multiplier == 0.0f &&
            dreq->min_p == 0.0f && dreq->typical_p >= 1.0f && !mtp_spec_decode_enabled() &&
            dreq->constraints && dreq->constraints->is_active();
        if (pipeline_compatible && dreq->status == RequestStatus::DECODING && !dreq->output_tokens.empty()) {
            try_launch_constrained_pipeline(dreq, dec_stream);
        }
    }
}
}  // namespace imp
