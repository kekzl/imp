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

#include <climits>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <utility>

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
    if (!step_schedule()) {
        // No schedulable work — but an in-flight pipelined step may still
        // hold tokens + deferred KV (all rows finished last step). Drain so
        // nothing leaks while the engine idles.
        if (bd_pipe_.in_flight)
            drain_decode_pipeline();
        return false;
    }

    // A pipelined decode step in flight shares workspace 0 with prefill and
    // pins the batch composition — collect it before any prefill work or
    // when this step has no decode batch to continue it with.
    if (bd_pipe_.in_flight && (!sched_prefill_batch_.empty() || sched_decode_batch_.empty()))
        drain_decode_pipeline();

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
            if (async_d_block_tables_swa_) {
                IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_swa_));
                async_d_block_tables_swa_ = nullptr;
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
            explicit_val, std::to_underlying(model_->config().arch),
            kv_cache_raw_ ? std::to_underlying(kv_cache_raw_->qtype()) : -1);
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
    // Greedy-agreement probe rides along for free (fused into the NLL max
    // pass); allocation failure just disables it.
    if (cudaMalloc(&ppl_capture_.d_match, static_cast<size_t>(n) * sizeof(int32_t)) == cudaSuccess) {
        IMP_CUDA_CHECK_LOG(
            cudaMemset(ppl_capture_.d_match, 0, static_cast<size_t>(n) * sizeof(int32_t)));
    } else {
        ppl_capture_.d_match = nullptr;
    }
    ppl_capture_.n = n;
    ppl_capture_.active = true;
    return true;
}

bool Engine::end_perplexity_capture(double* out_ppl) {
    if (!ppl_capture_.active) {
        return false;
    }
    const int n = ppl_capture_.n;
    // diagnostics.ppl_first/ppl_last: llama-perplexity comparability (see
    // config.h). Row i predicts token i+1; count rows [first, last].
    const int first = std::clamp(runtime_config_.diagnostics.ppl_first, 0, n - 2);
    const int cfg_last = runtime_config_.diagnostics.ppl_last;
    const int last = (cfg_last < 0) ? n - 2 : std::clamp(cfg_last, first, n - 2);
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());

    // Fixed-order host reduction over per-position NLLs (bit-reproducible —
    // same contract as GraphExecutor::perplexity_nll).
    std::vector<double> h_nll_pos(static_cast<size_t>(n), 0.0);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h_nll_pos.data(), ppl_capture_.d_nll,
                                  static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
    long match_sum = -1;
    if (ppl_capture_.d_match) {
        std::vector<int32_t> h_match(static_cast<size_t>(n), 0);
        IMP_CUDA_CHECK_LOG(cudaMemcpy(h_match.data(), ppl_capture_.d_match,
                                      static_cast<size_t>(n) * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost));
        match_sum = 0;
        for (int i = first; i <= last; ++i)
            match_sum += h_match[i];
        cudaFree(ppl_capture_.d_match);
    }
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
    for (int i = first; i <= last; ++i)
        h_nll += h_nll_pos[i];
    const int counted = last - first + 1;
    double ppl = std::exp(h_nll / static_cast<double>(counted));
    IMP_LOG_INFO("perplexity_nll: n=%d first=%d last=%d counted=%d mean_nll=%.4f  PPL=%.4f", n,
                 first, last, counted, h_nll / counted, ppl);
    if (match_sum >= 0)
        IMP_LOG_INFO("greedy top1 match: %ld/%d (%.2f%%)", match_sum, counted,
                     100.0 * static_cast<double>(match_sum) / static_cast<double>(counted));
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
    // Embedding requests mean-pool EVERY position's hidden state — a
    // prefix-cache hit would skip the reused prefix's forward and silently
    // bias the pooled vector (#1005). Same class as the ppl_capture guard.
    if (kv_manager_->prefix_caching_enabled() && existing == 0 && offset == 0 &&
        !ppl_capture_.active && !req->embedding_request) {
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
                                      const std::vector<int>& swa_block_table,
                                      int chunk_len, int offset, int ctx_len,
                                      cudaStream_t pf_stream,
                                      int32_t*& d_token_ids, int*& d_positions,
                                      int*& d_block_tables, int*& d_block_tables_swa,
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
            const auto& entry = *req->swa_restore;
            const bool ok = entry.data && entry.n_tokens == offset && swa_snapshots_ &&
                            swa_snapshots_->entry_bytes() == kv_manager_->swa_snapshot_bytes() &&
                            kv_manager_->swa_snapshot_restore(req->id, offset, entry.data,
                                                              pf_stream);
            req->swa_restore.reset();
            if (!ok) {
                IMP_LOG_WARN("SwaSnapshot: restore failed for req %d at %d tokens — cancelling",
                             req->id, offset);
                kv_manager_->free_sequence(req->id);
                req->status = RequestStatus::CANCELLED;
                return;
            }
            IMP_LOG_DEBUG("SwaSnapshot: restored %d-token window for req %d", offset, req->id);
        }
        kv_manager_->swa_trim(req->id, offset);
        if (!kv_manager_->swa_prepare(req->id, offset, ctx_len)) {
            kv_manager_->free_sequence(req->id);
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
    if (!prefill_upload_metadata_(req, block_table, swa_block_table, chunk_len, offset, ctx_len,
                                  pf_stream, d_token_ids, d_positions, d_block_tables,
                                  d_block_tables_swa, d_context_lens, pf_pool_used)) {
        return;  // caller already set req->status = CANCELLED
    }

    // Build InferenceState
    InferenceState state;
    state.token_ids = d_token_ids;
    state.positions = d_positions;
    state.n_tokens = chunk_len;
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
            executor_->forward_logits(state, logits_out, pf_stream);
        }

        // Teacher-forced NLL for this chunk's positions (imp_perplexity).
        // Runs eagerly after the (possibly graph-replayed) forward; hidden_
        // holds exactly this chunk and nothing reads logits_ afterwards.
        if (ppl_capture_.active) {
            executor_->perplexity_nll_partial(ppl_capture_.d_tokens, ppl_capture_.n, offset,
                                              chunk_len, ppl_capture_.d_nll, pf_stream,
                                              ppl_capture_.d_match);
        }

        // Embedding pooling for this chunk (#1005) — hidden_ still holds it.
        if (req->embedding_request)
            embed_accumulate_chunk_(*req, chunk_len, pf_stream);

        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
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
    } else if (req->embedding_request) {
        // Embedding request (#1005): last chunk — forward, pool, finish.
        // No sampling, no DECODING transition; the request rides the normal
        // finish path (KV free + prefix-hash registration, so re-embedding
        // the same document becomes a prefix-cache hit for OTHER requests).
        Tensor logits_unused;
        executor_->forward_logits(state, logits_unused, pf_stream);
        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
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
            executor_->perplexity_nll_partial(ppl_capture_.d_tokens, ppl_capture_.n, offset,
                                              chunk_len, ppl_capture_.d_nll, pf_stream,
                                              ppl_capture_.d_match);
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

    // Pipelined batched decode: a chained step is in flight — enqueue its
    // successor (composition permitting) and collect it. This IS this
    // call's decode step; the spec/SSM/KV logic below belongs to the
    // per-step path and runs again once the pipeline drains.
    if (bd_pipe_.in_flight) {
        step_decode_pipeline_(dec_stream);
        return;
    }

    // Verify-in-loop (#1055): an in-flight conditional verify burst owns
    // the decode stream — drain it (think tracking + stop handling happen
    // in the resume, like the async loop's).
    if (tr_loop_in_flight_()) {
        if (step_tr_loop_resume_(dec_stream))
            return;
    }

    // n-gram (prompt-lookup) speculation: when a draft is available, the
    // verify step replaces this decode step entirely (it allocates its own
    // KV blocks and emits accepted tokens). Falls through to the normal
    // path on a draft miss or when any gate fails.
    if (decode_batch.size() == 1 && spec_ngram_enabled_(*decode_batch[0])) {
        spec_maybe_rearm_(*decode_batch[0]);
        // Verify-in-loop launch (#1055): preferred over the eager per-step
        // verify when eligible; any decline falls through unchanged.
        if (runtime_config_.speculative.recycle_loop &&
            try_launch_tr_verify_loop_(decode_batch[0], dec_stream))
            return;
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
                    "think_budget=%.2f ssm=%d moe=%d moe_nvfp4=%d spec_moe=%d mtp=%d "
                    "chunked_prefill=%d)",
                    r.temperature, r.top_k, r.repetition_penalty, r.frequency_penalty,
                    r.presence_penalty, r.dry_multiplier, r.mirostat, r.logit_bias.size(),
                    (int)r.logprobs, (int)r.json_mode, (int)!r.json_schema.empty(), r.think_budget,
                    (int)(ssm_state_ != nullptr),
                    (int)model_->profile().is_moe, (int)model_->profile().moe_experts_nvfp4,
                    (int)runtime_config_.speculative.moe, (int)mtp_spec_decode_enabled(),
                    (int)supports_chunked_prefill_());
            }
        }
    }

    // #1003 stage 1: at batch > 1 (dense, non-recurrent), ONE request per
    // step may run its spec verify while the rest decode batched below —
    // round-robin in cyclic id order (the hybrid-rotation pattern), so every
    // eligible request gets verify turns under sub-agent fan-out instead of
    // silently losing speculation to the batch-size-1 dispatch gate above.
    // The verify emits that request's tokens for this step; it is removed
    // from this step's batched decode and rejoins next step.
    if (decode_batch.size() > 1 && !ssm_state_ && runtime_config_.speculative.batch_rr &&
        runtime_config_.speculative.ngram) {
        int cand = -1;
        int best_id = INT_MAX, wrap_id = INT_MAX, wrap_idx = -1;
        for (size_t i = 0; i < decode_batch.size(); ++i) {
            auto& r = decode_batch[i];
            if (!spec_ngram_enabled_(*r))
                continue;
            spec_maybe_rearm_(*r);
            if (!spec_ngram_gates_ok_(*r))
                continue;
            const int id = r->id;
            if (id > spec_rr_last_id_ && id < best_id) {
                best_id = id;
                cand = static_cast<int>(i);
            }
            if (id < wrap_id) {
                wrap_id = id;
                wrap_idx = static_cast<int>(i);
            }
        }
        if (cand < 0)
            cand = wrap_idx;  // wrap around (or stay -1: none eligible)
        bool verified = false;
        if (cand >= 0) {
            auto holder = decode_batch[cand];
            // Depth floor ~2x batch: the verify must plausibly emit more
            // tokens than the batch-wide stall it costs (see min_draft).
            const int min_draft = 2 * static_cast<int>(decode_batch.size());
            if (step_spec_verify_(holder, dec_stream, min_draft)) {
                verified = true;
                spec_rr_last_id_ = holder->id;
                decode_batch.erase(decode_batch.begin() + cand);
            }
        }
        // Adaptive yield cadence: empty turns (shallow drafts / no candidate)
        // back the pipeline-break interval off exponentially — a draft-poor
        // batch measured -1.5..-2.7% from fruitless chain breaks alone.
        spec_rr_yield_interval_ = verified ? 8 : std::min(spec_rr_yield_interval_ * 2, 64);
        if (verified && decode_batch.empty())
            return;
        // fall through: the remaining rows decode batched this step
    }

    // SSM/GDN: the recurrent scan kernels are single-sequence, so decode one
    // sequence per step. The old resize(1) kept the OLDEST active request
    // every step — concurrent sessions serialized head-of-line (the first
    // request ran to completion before the second produced a token). Rotate
    // the slice holder round-robin every hybrid_decode_quantum tokens
    // instead; the decode graphs re-capture for the new sequence's state
    // slot on rotation (~10-20 ms), which the quantum amortizes.
    hybrid_decode_waiting_ = false;
    if (ssm_state_ && decode_batch.size() > 1) {
        const int quantum = runtime_config_.runtime.hybrid_decode_quantum;
        if (quantum > 0) {
            int cur = -1;
            for (size_t i = 0; i < decode_batch.size(); ++i) {
                if (decode_batch[i]->id == hybrid_active_req_) {
                    cur = static_cast<int>(i);
                    break;
                }
            }
            bool rotate = (cur < 0) || (static_cast<int>(decode_batch[cur]->output_tokens.size()) -
                                            hybrid_slice_start_ >=
                                        quantum);
            if (rotate) {
                // Next request in cyclic id order after the current holder —
                // ids are monotonically increasing, so this is admission
                // round-robin.
                int next = 0;
                int best_id = INT_MAX, best_wrap_id = INT_MAX;
                int next_wrap = 0;
                for (size_t i = 0; i < decode_batch.size(); ++i) {
                    int id = decode_batch[i]->id;
                    if (id > hybrid_active_req_ && id < best_id) {
                        best_id = id;
                        next = static_cast<int>(i);
                    }
                    if (id < best_wrap_id) {
                        best_wrap_id = id;
                        next_wrap = static_cast<int>(i);
                    }
                }
                cur = (best_id != INT_MAX) ? next : next_wrap;
                hybrid_active_req_ = decode_batch[cur]->id;
                hybrid_slice_start_ = static_cast<int>(decode_batch[cur]->output_tokens.size());
            }
            if (cur > 0)
                std::swap(decode_batch[0], decode_batch[cur]);
            // Others are waiting: bound the async graph-loop burst to the
            // slice remainder (step_decode_process_outputs) so rotation
            // actually gets a chance to run.
            hybrid_decode_waiting_ = true;
        }
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
                // Log loudly: this used to be a silent cancel that surfaced as a
                // bare "internal error" at the API (Gemma-4-12B at ctx 16384 on a
                // 1024-block FP16-KV pool cost a debugging session to attribute).
                int pool_blocks = kv_cache_raw_ ? kv_cache_raw_->total_blocks() : 0;
                IMP_LOG_ERROR(
                    "KV pool exhausted at decode: seq %d needs block %d but the %d-block pool has "
                    "0 free/reclaimable — cancelling this sequence (others keep their KV). The "
                    "pool was VRAM-clamped below the requested context; free VRAM, lower "
                    "max_seq_len, or halve KV with kv_cache.dtype=fp8 (--kv-fp8).",
                    req->id, blocks_needed, pool_blocks);
                kv_manager_->free_sequence(req->id);
                req->status = RequestStatus::CANCELLED;
                continue;
            }
        }

        // SWA-aware sizing: keep the trailing window live for this step's
        // write + reads; retire blocks that fell out of the window.
        if (swa_sizing_active_) {
            kv_manager_->swa_trim(req->id, ctx_len);
            if (!kv_manager_->swa_prepare(req->id, ctx_len)) {
                IMP_LOG_ERROR(
                    "SWA KV sizing failed at decode: seq %d could not prepare its window at "
                    "ctx_len=%d — cancelling this sequence (see kv_cache.swa_sizing).",
                    req->id, ctx_len);
                kv_manager_->free_sequence(req->id);
                req->status = RequestStatus::CANCELLED;
                continue;
            }
        }

        // Auto-activate StreamingLLM when KV cache is nearly exhausted.
        // Only fires once (guards on !streaming_kv_enabled) and only for FP16
        // KV — quantized variants don't support sentinel-block skipping yet.
        // Never under SWA sizing (streaming_kv_auto is cleared at init there).
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
    state.block_tables_swa = gpu_batch.d_block_tables_swa;
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
        if (r->json_mode && r->json_schema.empty() && r->tool_constraint_tools.empty())
            needs_json_mode = true;
        if (!r->json_schema.empty() || !r->tool_constraint_tools.empty())
            needs_schema_mode = true;
    }

    // Schema/JSON constraints for decode. Lazily create the per-request
    // manager if needed (decode might be the first step with json_mode).
    // The single-sequence state carries the request's constrainers (the
    // graph-loop / constrained-pipeline launch paths read them from state);
    // batched decode attaches them per row in sample_per_request, so
    // constraints stay enforced at batch>1 (previously they were silently
    // dropped whenever a constrained request shared a decode batch).
    for (auto& r : valid_decode)
        ensure_constraints_(r);
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
        const auto& sbt = kv_manager_->swa_block_table(req->id);
        decode_builder_.add_decode_sequence(last_token, position, bt.data(), static_cast<int>(bt.size()),
                                            ctx_len,
                                            swa_sizing_active_ && !sbt.empty() ? sbt.data() : nullptr,
                                            static_cast<int>(sbt.size()));
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
            // Re-pad the SWA tables at the same stride (-1 = hole, never 0).
            if (!batch.block_tables_swa.empty()) {
                padded_swa_block_table_.assign(needed, -1);
                for (int s = 0; s < n_seq; s++) {
                    for (int b = 0; b < old_stride; b++) {
                        padded_swa_block_table_[s * pool_max + b] =
                            batch.block_tables_swa[s * old_stride + b];
                    }
                }
                batch.block_tables_swa.swap(padded_swa_block_table_);
            }
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
        // Two-pass batched sampling: pass 1 ENQUEUES every row's filter+
        // sampler chain into its own scratch slot (stream-ordered, so the
        // shared d_penalty_tokens_ upload/consume pairs stay correct); pass 2
        // gathers all tokens with ONE pinned D2H + ONE stream sync. The
        // previous per-row synchronous readback blocked the engine thread
        // ~850 us per sequence per step (pageable 4-byte D2H + sync each) —
        // 29% GPU idle at n=16 sustained serving (nsys, 2026-07-12). Rows
        // with sync-only sampling modes (mirostat, logit_bias, CUB-regime
        // top_k) decline untouched in pass 1 and sample synchronously after
        // the gather.
        std::vector<int> sync_rows;
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
            // carries no constrainers at n>1; the row sampler applies the
            // mask to this row's logits before sampling).
            if (req->constraints) {
                per_state.schema_constrainer = req->constraints->schema_constrainer();
                per_state.json_constrainer = req->constraints->json_constrainer();
            } else {
                per_state.schema_constrainer = nullptr;
                per_state.json_constrainer = nullptr;
            }
            per_state.n_sequences = 1;
            Tensor seq_logits = logits.slice(i, i + 1);
            if (!executor_->sample_single_from_logits_async(seq_logits, per_state, i, dec_stream))
                sync_rows.push_back(i);
        }
        if (static_cast<int>(sync_rows.size()) < n) {
            const int32_t* toks = executor_->collect_sampled_tokens(n, dec_stream);
            if (toks) {
                for (int i = 0; i < n; i++)
                    result[i] = toks[i];
            } else {
                // Collector unavailable (no slot buffers) — every row falls
                // back to the synchronous path below.
                sync_rows.clear();
                for (int i = 0; i < n; i++)
                    sync_rows.push_back(i);
            }
        }
        for (int i : sync_rows) {
            auto& req = valid_decode[i];
            InferenceState per_state = state;
            fill_sampling_params(*req, per_state);
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
        // #948: the decode-attention LAUNCH topology derives from
        // max_context_len on the host (split-K num_splits / GQA-vs-split-K
        // kernel choice, scratch clamping) and is baked into the capture. The
        // max_blocks bucket above was meant to catch context growth but never
        // trips — the decode batch pool pads max_blocks_per_seq to the pool
        // stride. Re-derive the graph when the pow2 context bucket GROWS past
        // the captured one: a graph captured at ctx≈35 replayed at ctx≈2400
        // dereferences a stale-topology kernel and wedges the engine with an
        // illegal memory access. Growth-only (monotonic high-water mark,
        // ~log2(max ctx) captures per process — an on-any-change trigger
        // re-captured 1-2x per request under short/long ping-pong): replaying
        // a large-ctx capture for a SHORT request is safe, surplus split-K
        // splits write their empty-split sentinels and the reduce merges
        // them; the reverse direction is the crash. cudaGraphExecUpdate
        // absorbs same-topology re-captures; a kernel-choice change fails
        // the update and re-instantiates.
        const int bucketed_max_ctx = bucket_pow2(max_ctx);
        if (bucketed_max_ctx > last_decode_max_ctx_per_graph_[graph_idx]) {
            graph_runner.invalidate_for_update();
            last_decode_max_ctx_per_graph_[graph_idx] = bucketed_max_ctx;
        }
        // Recurrent (SSM/GDN) state pointers are baked into the capture as
        // kernel params. A replay for a request in a DIFFERENT state slot
        // would read/write the previous sequence's state — possible whenever
        // request lifetimes overlap (the next request acquired its slot while
        // the previous one was live) and every time the hybrid decode slice
        // rotates. Same-topology re-capture via graph-exec update.
        if (ssm_state_ && gpu_batch.n_sequences == 1) {
            auto slot_it = recurrent_slot_of_.find(valid_decode[0]->id);
            const int slot = (slot_it != recurrent_slot_of_.end()) ? slot_it->second : -1;
            if (slot != decode_graph_recurrent_slot_) {
                graph_runner.invalidate_for_update();
                decode_graph_recurrent_slot_ = slot;
            }
        }
        // Graph captures ONLY forward_logits — sampling runs eager after
        Tensor logits_out;
        graph_runner.set_decode_fn(
            [this, &state, &logits_out](cudaStream_t s) { executor_->forward_logits(state, logits_out, s); });
        graph_runner.execute(dec_stream);

        if (logits_out.data == nullptr) {
            logits_out = executor_->get_logits_view(gpu_batch.n_sequences);
        }
        // Pipelined entry: gather this step's tokens via the event-based
        // split path and, when everything is graph-replayable and every row
        // async-sampleable, enqueue step N+1 (device token chain) before
        // reading step N's tokens back. Falls through to the legacy
        // synchronous collect for every non-clean case.
        bool piped = false;
        if (!needs_logprobs && !needs_json_mode && !needs_schema_mode &&
            pipeline_batch_eligible_(valid_decode)) {
            piped = pipeline_enter_(valid_decode, gpu_batch, graph_idx, state, logits_out,
                                    dec_stream, tokens);
        }
        // Eager sampling (handles all modes: greedy, top-k/p, penalties,
        // force_token, constraints, logprobs, mirostat)
        if (!piped)
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

        // Verify-consumer sync gate (#847): the chain feed below appends a
        // (next_token, h) pair at ws->mtp_pos — valid only while the cache
        // exactly covers the pre-step context. After an async-loop burst
        // (device-side tokens, no host hiddens) the cache is stale — skip
        // feeding so it never desynchronizes silently.
        auto* ws_gate = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
        const bool mtp_synced = ws_gate != nullptr && mtp_bound_req_ == valid_decode[0]->id &&
                                ws_gate->mtp_pos == cur_pos &&
                                (ws_gate->max_seq_len <= 0 ||
                                 ws_gate->mtp_pos + std::max(1, mtp_spec_k_) <
                                     ws_gate->max_seq_len);
        Tensor h_view = executor_->view_hidden(1);  // [1, d_model] FP16
        if (h_view.data != nullptr && mtp_synced) {
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
            std::vector<int32_t> chain_preds;
            chain_preds.reserve(K);
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
                chain_preds.push_back(prediction);
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
            // Verify-consumer bookkeeping (#847): the kept pair covers
            // next_token; the chain IS the draft for the next verify step.
            mtp_history_.push_back(next_token);
            if (!chain_preds.empty()) {
                mtp_pending_draft_ = std::move(chain_preds);
                mtp_draft_ctx_ = static_cast<int>(mtp_history_.size());
            } else {
                mtp_pending_draft_.clear();
                mtp_draft_ctx_ = -1;
            }
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
            if (bd_pipe_.in_flight) {
                // A chained step is in flight and still WRITES this row's
                // next KV slot — mark FINISHED (stream delivery proceeds)
                // but defer the KV/slot release to the pipeline drain.
                req->status = RequestStatus::FINISHED;
                bd_pipe_.deferred_release.push_back(req);
            } else {
                finish_request(req);
            }
        }

        kv_manager_->touch(req->id);
    }

    // Try async graph loop after first decode step.
    // Think budget is now handled device-side in post_decode_step_kernel.
    if (decode_graph_pool_[0].graph_path_available() && valid_decode.size() == 1 && !offload_mgr_ &&
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
            // Hybrid decode fairness: other recurrent sequences are waiting
            // for the slice — bound the burst to the slice remainder so the
            // round-robin rotation in step_decode actually runs.
            if (!runtime_config_.runtime.deterministic && hybrid_decode_waiting_) {
                const int quantum = runtime_config_.runtime.hybrid_decode_quantum;
                int slice_left = quantum - (static_cast<int>(dreq->output_tokens.size()) -
                                            hybrid_slice_start_);
                slice_left = std::max(1, slice_left);
                if (launch_limit == 0 || launch_limit > slice_left)
                    launch_limit = slice_left;
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
    if (decode_graph_pool_[0].graph_path_available() && valid_decode.size() == 1 && !offload_mgr_ &&
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

// =====================================================================
// Pipelined batched decode (bd_pipe_) — one step in flight.
//
// Step N+1 is enqueued (device token chain feeding step N's sampled slot
// tokens as input ids + forward-graph replay + per-row sampler enqueue into
// the OTHER slot-parity half) BEFORE step N's tokens are read back; the
// host then event-waits on step N's gather only, so its bookkeeping + the
// server's SSE delivery overlap the GPU's work on N+1 instead of idling it
// (the ~15-20% step tail at n=16 sustained serving, nsys 2026-07-12).
//
// Scope (v1) is the clean serving case: every row async-sampleable (greedy
// or top-k<=SAMPLE_MAX_TOP_K/top-p; min_p/typical_p fine), no penalties/DRY
// (their token history grows per step host-side), no constraints/logprobs/
// mirostat/logit_bias, no SSM/SWA/StreamingLLM/residual-KV, CUDA graphs on.
// Anything else keeps the per-step path bit-for-bit unchanged.
//
// Known (accepted) semantics deltas vs the per-step path:
//  - host state lags the in-flight step by one token, so think-budget
//    forcing and stop-string detection fire one step later (one discarded
//    token; KV stays consistent because the row's release is deferred);
//  - a row that stops still has step N+1 computed for it — its token is
//    discarded and its KV blocks are freed only after that step completes
//    (deferred_release), so the in-flight forward never writes freed blocks.
// =====================================================================

namespace {
inline int pipeline_bucket_pow2(int x) {
    if (x <= 1) return 1;
    int b = 1;
    while (b < x) b <<= 1;
    return b;
}
}  // namespace

bool Engine::pipeline_row_eligible_(const Request& r) const {
    if (r.logprobs || r.json_mode || !r.json_schema.empty() || r.constraints)
        return false;
    if (r.mirostat != 0 || !r.logit_bias.empty())
        return false;
    // rep/freq/presence penalties ARE supported (device-side history — the
    // server defaults to repetition_penalty 1.05, so the common serving row
    // carries them). DRY needs the host-side token array — excluded.
    if (r.dry_multiplier > 0.0f)
        return false;
    const bool greedy = (r.temperature <= 0.0f || r.top_k == 1);
    if (!greedy) {
        const int vocab = model_ ? model_->config_.vocab_size : 0;
        const int top_k = r.top_k > 0 ? r.top_k : 50;
        const int eff_top_k = (top_k <= 0 || top_k > vocab) ? vocab : top_k;
        if (eff_top_k > SAMPLE_MAX_TOP_K)
            return false;  // CUB regime syncs internally — not enqueue-only
    }
    return true;
}

bool Engine::pipeline_batch_eligible_(const std::vector<std::shared_ptr<Request>>& rows) const {
    if (!runtime_config_.runtime.decode_pipeline)
        return false;
    const int n = static_cast<int>(rows.size());
    if (n < 2 || n > kMaxGraphPoolSize)
        return false;
    if (!config_.use_cuda_graphs || runtime_config_.diagnostics.profile)
        return false;
    if (!decode_batch_pool_.is_allocated())
        return false;
    if (ssm_state_ || offload_mgr_)
        return false;
    if (swa_sizing_active_ || config_.streaming_kv_enabled)
        return false;
    if (kv_manager_ && kv_manager_->residual_enabled())
        return false;
    if (!executor_->sample_pipeline_ready())
        return false;
    for (const auto& r : rows)
        if (!pipeline_row_eligible_(*r))
            return false;
    return true;
}

InferenceState Engine::pipeline_row_state_(Request& req, int row_idx) const {
    InferenceState per = bd_pipe_.base_state;
    fill_sampling_params(req, per);
    // The chained step samples one output AHEAD of the host-visible count —
    // match the seed the eager path would compute after processing the
    // in-flight step (compute_step_seed = base + output count).
    per.seed = compute_step_seed(req) + 1;
    per.penalty_tokens = nullptr;
    per.n_penalty_tokens = 0;
    per.host_penalty_tokens = nullptr;
    // Penalty rows read the per-row device history: host-known tokens were
    // uploaded at entry, and the advance kernel appended the in-flight
    // token at position `count` — the chained step penalizes count+1
    // tokens, exactly what the eager path would upload for that step.
    const bool needs_pen = (req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                            req.presence_penalty != 0.0f);
    if (needs_pen && d_pipe_hist_) {
        per.penalty_tokens =
            d_pipe_hist_ + static_cast<size_t>(row_idx) * pipe_hist_stride_;
        per.n_penalty_tokens = static_cast<int>(req.output_tokens.size()) + 1;
    }
    per.schema_constrainer = nullptr;
    per.json_constrainer = nullptr;
    per.n_sequences = 1;
    return per;
}

bool Engine::pipeline_staging_ensure_() {
    if (bt_patch_cap_ == 0) {
        // Lazy mapped-pinned staging: at most one appended block per row per
        // step, so kMaxGraphPoolSize entries per parity set cover any batch.
        // The per-row device output-token history (penalty rows) is sized
        // to the model context (outputs are KV-bounded) with a sanity cap.
        const int msl = model_ ? model_->config().max_seq_len : 0;
        pipe_hist_stride_ = std::max(1024, std::min(msl > 0 ? msl : 32768, 65536));
        bool ok = true;
        for (int p = 0; p < 2 && ok; ++p) {
            ok = cudaHostAlloc(&h_bt_patch_off_[p], kMaxGraphPoolSize * sizeof(int),
                               cudaHostAllocMapped) == cudaSuccess &&
                 cudaHostAlloc(&h_bt_patch_val_[p], kMaxGraphPoolSize * sizeof(int),
                               cudaHostAllocMapped) == cudaSuccess &&
                 cudaHostAlloc(&h_hist_pos_[p], kMaxGraphPoolSize * sizeof(int),
                               cudaHostAllocMapped) == cudaSuccess &&
                 cudaHostGetDevicePointer(&d_bt_patch_off_[p], h_bt_patch_off_[p], 0) ==
                     cudaSuccess &&
                 cudaHostGetDevicePointer(&d_bt_patch_val_[p], h_bt_patch_val_[p], 0) ==
                     cudaSuccess &&
                 cudaHostGetDevicePointer(&d_hist_pos_[p], h_hist_pos_[p], 0) == cudaSuccess;
        }
        if (ok)
            ok = cudaMalloc(&d_pipe_hist_, static_cast<size_t>(kMaxGraphPoolSize) *
                                               pipe_hist_stride_ * sizeof(int32_t)) == cudaSuccess;
        if (!ok) {
            for (int p = 0; p < 2; ++p) {
                if (h_bt_patch_off_[p]) cudaFreeHost(h_bt_patch_off_[p]);
                if (h_bt_patch_val_[p]) cudaFreeHost(h_bt_patch_val_[p]);
                if (h_hist_pos_[p]) cudaFreeHost(h_hist_pos_[p]);
                h_bt_patch_off_[p] = h_bt_patch_val_[p] = h_hist_pos_[p] = nullptr;
                d_bt_patch_off_[p] = d_bt_patch_val_[p] = d_hist_pos_[p] = nullptr;
            }
            if (d_pipe_hist_) {
                cudaFree(d_pipe_hist_);
                d_pipe_hist_ = nullptr;
            }
            bt_patch_cap_ = -1;  // don't retry every step
            return false;
        }
        bt_patch_cap_ = kMaxGraphPoolSize;
    }
    return bt_patch_cap_ > 0;
}

int Engine::pipeline_prebook_kv_(int parity) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    const int stride = decode_batch_pool_.max_blocks_per_seq();
    if (!pipeline_staging_ensure_())
        return -1;

    int n_patches = 0;
    for (int i = 0; i < bd_pipe_.n; ++i) {
        auto& req = bd_pipe_.rows[i];
        const int target_ctx = req->context_len() + 1;
        const int blocks_needed = (target_ctx + kv_bs - 1) / kv_bs;
        const auto& bt = kv_manager_->block_table(req->id);
        const int have = static_cast<int>(bt.size());
        if (blocks_needed > have) {
            if (blocks_needed > stride)
                return -1;  // batch-pool stride exhausted — per-step path re-pads
            const int nb = kv_manager_->append_block(req->id);
            if (nb < 0)
                return -1;  // KV exhausted — per-step path handles reject/streaming
            h_bt_patch_off_[parity][n_patches] = i * stride + have;
            h_bt_patch_val_[parity][n_patches] = nb;
            ++n_patches;
        }
    }
    return n_patches;
}

bool Engine::pipeline_enqueue_next_(int parity, cudaStream_t stream) {
    const int n = bd_pipe_.n;
    if (n <= 0 || bd_pipe_.graph_idx < 0)
        return false;
    auto& runner = decode_graph_pool_[bd_pipe_.graph_idx];
    if (!runner.is_ready())
        return false;
    // #948 guard: the captured decode-attention launch topology covers the
    // pow2 context bucket of the capture — never chain past it (the per-step
    // path re-derives the graph at the boundary).
    int next_max_ctx = 0;
    for (const auto& r : bd_pipe_.rows)
        next_max_ctx = std::max(next_max_ctx, r->context_len() + 1);
    if (pipeline_bucket_pow2(next_max_ctx) > last_decode_max_ctx_per_graph_[bd_pipe_.graph_idx])
        return false;

    const int n_patches = pipeline_prebook_kv_(parity);
    if (n_patches < 0)
        return false;

    // Per-row history append positions (= host-visible output count; the
    // in-flight token lands there). Capacity check: a row whose history
    // would outgrow the stride drains to the per-step path.
    for (int i = 0; i < n; ++i) {
        const int count = static_cast<int>(bd_pipe_.rows[i]->output_tokens.size());
        if (count + 2 > pipe_hist_stride_)
            return false;
        h_hist_pos_[parity][i] = count;
    }

    // Device-side chain: feed the IN-FLIGHT step's sampled tokens (other
    // parity's slots) as this step's input ids, bump positions/context lens,
    // append them to the per-row history, scatter freshly appended block
    // ids. Stream-ordered after the in-flight step's samplers, before this
    // step's forward.
    decode_pipeline_advance(n, executor_->sample_slot_base(parity ^ 1), SAMPLE_SCRATCH_BYTES,
                            bd_pipe_.gpu.d_token_ids, bd_pipe_.gpu.d_positions,
                            bd_pipe_.gpu.d_context_lens, bd_pipe_.gpu.d_block_tables, n_patches,
                            d_bt_patch_off_[parity], d_bt_patch_val_[parity], d_pipe_hist_,
                            pipe_hist_stride_, d_hist_pos_[parity], stream);
    if (!runner.replay_only(stream))
        return false;  // abandoned chained forward is safe: the per-step path
                       // re-uploads batch state and rewrites the same KV slot

    executor_->set_sample_parity(parity);
    Tensor logits = executor_->get_logits_view(n);
    for (int i = 0; i < n; ++i) {
        InferenceState per = pipeline_row_state_(*bd_pipe_.rows[i], i);
        Tensor seq_logits = logits.slice(i, i + 1);
        if (!executor_->sample_single_from_logits_async(seq_logits, per, i, stream)) {
            // Statically unreachable (eligibility excludes every decline
            // mode); abandoning the half-enqueued step is safe (see above).
            IMP_LOG_ERROR("decode-pipeline: unexpected sampler decline (row %d) — not chaining", i);
            return false;
        }
    }
    return executor_->gather_sampled_tokens_async(n, stream);
}

bool Engine::pipeline_enter_(std::vector<std::shared_ptr<Request>>& rows, const GPUBatch& gpu_batch,
                             int graph_idx, const InferenceState& state, const Tensor& logits,
                             cudaStream_t stream, std::vector<int32_t>& tokens_out) {
    const int n = static_cast<int>(rows.size());
    // Penalty rows need the device-side history BEFORE any sampler runs —
    // without it the entry pass would sample unpenalized. Bail to the
    // legacy path (which uploads d_penalty_tokens_ per row) if staging is
    // unavailable or a row's history would not fit.
    if (!pipeline_staging_ensure_())
        return false;
    for (const auto& req : rows) {
        if (static_cast<int>(req->output_tokens.size()) + 2 > pipe_hist_stride_)
            return false;
    }
    executor_->set_sample_parity(0);
    for (int i = 0; i < n; ++i) {
        auto& req = rows[i];
        InferenceState per = state;
        fill_sampling_params(*req, per);
        per.seed = compute_step_seed(*req);
        per.penalty_tokens = nullptr;
        per.n_penalty_tokens = 0;
        per.host_penalty_tokens = nullptr;
        const bool needs_pen = (req->repetition_penalty != 1.0f ||
                                req->frequency_penalty != 0.0f || req->presence_penalty != 0.0f);
        const int count = static_cast<int>(req->output_tokens.size());
        if (needs_pen && count > 0) {
            // Seed the per-row device history with the host-known tokens;
            // chained steps keep it current via the advance-kernel append.
            // The async H2D from the (pageable) vector is safe: the gather
            // event below is synced before process_outputs can grow it.
            int32_t* row_hist = d_pipe_hist_ + static_cast<size_t>(i) * pipe_hist_stride_;
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(row_hist, req->output_tokens.data(),
                                               count * sizeof(int32_t), cudaMemcpyHostToDevice,
                                               stream));
            per.penalty_tokens = row_hist;
            per.n_penalty_tokens = count;
        }
        per.schema_constrainer = nullptr;
        per.json_constrainer = nullptr;
        per.n_sequences = 1;
        Tensor seq_logits = logits.slice(i, i + 1);
        if (!executor_->sample_single_from_logits_async(seq_logits, per, i, stream)) {
            // Statically unreachable; the already-applied row filters are
            // idempotent under the eligibility gates, so the legacy collect
            // path can safely re-run every row.
            return false;
        }
    }
    if (!executor_->gather_sampled_tokens_async(n, stream)) {
        const int32_t* toks = executor_->collect_sampled_tokens(n, stream);
        if (!toks)
            return false;
        tokens_out.assign(toks, toks + n);
        return true;
    }

    // Chain step N+1 while step N's gather is in flight.
    bd_pipe_.rows = rows;
    bd_pipe_.n = n;
    bd_pipe_.graph_idx = graph_idx;
    bd_pipe_.gpu = gpu_batch;
    bd_pipe_.base_state = state;
    bd_pipe_.parity = 0;
    bd_pipe_.in_flight = false;
    bd_pipe_.steps_since_spec_yield = 0;
    if (pipeline_enqueue_next_(1, stream)) {
        bd_pipe_.in_flight = true;
        bd_pipe_.parity = 1;
    } else {
        // A partially attempted chain may have left the executor on the
        // other parity half — every non-pipelined caller expects parity 0.
        executor_->set_sample_parity(0);
        bd_pipe_.rows.clear();
        bd_pipe_.n = 0;
    }

    const int32_t* toks = executor_->wait_gathered_tokens(0);
    if (!toks) {
        // No event support — should not happen past sample_pipeline_ready.
        abandon_decode_pipeline();
        return false;
    }
    tokens_out.assign(toks, toks + n);
    return true;
}

void Engine::step_decode_pipeline_(cudaStream_t stream) {
    auto& db = sched_decode_batch_;
    // Continue only with the EXACT same composition in the same order (the
    // scheduler preserves admission order; any join/leave/cancel changes the
    // set) and only while the static gates still hold. Apply the same
    // max_batch_size cap the per-step path applies — at overload the rows
    // beyond the cap are not being stepped either way.
    size_t eff = db.size();
    const int max_bs_cap = runtime_config_.runtime.max_batch_size;
    if (max_bs_cap > 0 && eff > static_cast<size_t>(max_bs_cap))
        eff = static_cast<size_t>(max_bs_cap);
    bool cont = sched_prefill_batch_.empty() && eff == bd_pipe_.rows.size();
    if (cont) {
        for (size_t i = 0; i < eff; ++i) {
            if (db[i].get() != bd_pipe_.rows[i].get()) {
                cont = false;
                break;
            }
        }
    }
    if (cont) {
        for (const auto& r : bd_pipe_.rows) {
            if (r->status != RequestStatus::DECODING) {
                cont = false;
                break;
            }
        }
    }
    if (cont)
        cont = pipeline_batch_eligible_(bd_pipe_.rows);

    // #1003: yield the chain periodically so the plain path's round-robin
    // spec verify gets a turn — once in flight the pipeline otherwise chains
    // until the composition changes, and batch>1 speculation never fires
    // (observed verify_steps=1 per request). Fields-only candidate check;
    // the draft-depth economics (min_draft) run in the RR branch. The chain
    // re-engages on the following step via the normal plain-path entry.
    if (cont && runtime_config_.speculative.batch_rr && runtime_config_.speculative.ngram &&
        !ssm_state_ && ++bd_pipe_.steps_since_spec_yield >= spec_rr_yield_interval_) {
        bd_pipe_.steps_since_spec_yield = 0;
        for (const auto& r : bd_pipe_.rows) {
            if (r->status == RequestStatus::DECODING && spec_ngram_enabled_(*r) &&
                !r->spec_ngram_given_up) {
                cont = false;
                break;
            }
        }
    }

    const int next_parity = bd_pipe_.parity ^ 1;
    const bool chained = cont && pipeline_enqueue_next_(next_parity, stream);

    // Collect + process the in-flight step (event wait ≈ 0 in steady state —
    // it finished while the host was in the scheduler/server since last step).
    pipeline_collect_process_(stream);

    if (chained) {
        bd_pipe_.parity = next_parity;
        // If every row just finished, the scheduler goes idle and nothing
        // would ever collect the chained step — drain it now (its tokens are
        // all for FINISHED rows and get discarded).
        bool any_decoding = false;
        for (const auto& r : bd_pipe_.rows) {
            if (r->status == RequestStatus::DECODING) {
                any_decoding = true;
                break;
            }
        }
        if (!any_decoding)
            drain_decode_pipeline();
    } else {
        // Pipeline ends. If a chained enqueue was attempted and partially
        // enqueued (advance kernel / replay), that work may still reference
        // the deferred rows' KV — order the release behind it.
        if (cont)
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        bd_pipe_.in_flight = false;
        pipeline_run_deferred_releases_();
        executor_->set_sample_parity(0);
        bd_pipe_.rows.clear();
        bd_pipe_.n = 0;
    }
}

void Engine::pipeline_collect_process_(cudaStream_t stream) {
    (void)stream;
    const int32_t* toks = executor_->wait_gathered_tokens(bd_pipe_.parity);
    if (!toks)
        return;
    for (int i = 0; i < bd_pipe_.n; ++i) {
        auto& req = bd_pipe_.rows[i];
        if (req->status != RequestStatus::DECODING)
            continue;  // finished/cancelled while this step was in flight — discard
        const int32_t next_token = toks[i];
        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);
        if (should_stop(*req, next_token) ||
            static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            // The successor step (if chained) still writes this row's next
            // KV slot — defer the release until no step references it.
            req->status = RequestStatus::FINISHED;
            bd_pipe_.deferred_release.push_back(req);
        }
        kv_manager_->touch(req->id);
    }
}

void Engine::pipeline_run_deferred_releases_() {
    for (auto& req : bd_pipe_.deferred_release)
        finish_request_release_(req);
    bd_pipe_.deferred_release.clear();
}

void Engine::drain_decode_pipeline() {
    if (bd_pipe_.in_flight) {
        pipeline_collect_process_(decode_stream());
        bd_pipe_.in_flight = false;
    }
    pipeline_run_deferred_releases_();
    if (executor_)
        executor_->set_sample_parity(0);
    bd_pipe_.rows.clear();
    bd_pipe_.n = 0;
}

void Engine::abandon_decode_pipeline() {
    // Exception/teardown path: never wait on the device here.
    bd_pipe_.in_flight = false;
    pipeline_run_deferred_releases_();
    if (executor_)
        executor_->set_sample_parity(0);
    bd_pipe_.rows.clear();
    bd_pipe_.n = 0;
}
}  // namespace imp
