// Engine scheduler: per-step prefill + decode driver.
//
// Top-level flow:
//   step() → step_schedule() → step_prefill() OR step_decode()
//
// Prefill execution lives in engine_prefill.cpp (serial chunked path) and
// engine_prefill_ragged.cpp (cross-sequence ragged path); the pipelined
// batched decode (bd_pipe_) in engine_decode_pipeline.cpp — all split out
// 2026-08-26 when this TU hit 2230 code LOC.
//
// Decode chain:
//   step_decode → decode_build_inference_state_ → step_decode_forward
//               → step_decode_process_outputs
//
// Async graph resume:
//   step_async_graph_resume — for the CUDA-graph-captured decode path.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor roadmap.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "core/buffer.h"
#include "runtime/batch.h"
#include "runtime/think_stop_logic.h"
#include "compute/mtp_forward.h"
#include "compute/dispatch_record.h"  // resolved-path summary (#1205)
#include "model/image_placeholders.h"
#include "memory/kv_cache.h"
#include "compute/sampling.h"
#include "compute/layernorm.h"
#include "core/logging.h"

#include <climits>
#include <cstdio>
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

// The resolved-dispatch summary must survive EVERY exit from the step body,
// which is why the body is wrapped instead of ending with the call.
//
// #1205 put log_resolved_dispatch_once_() before the final `return` of step().
// That is the one exit the decode path almost never takes: once the async
// conditional graph loop is armed — the normal graphs-ON decode route — every
// subsequent step() returns from the `async_result == 1` fast path below, so the
// recorded decode tier was set and then never read. A full prefill+decode E2E run
// on Qwen3.6-35B-A3B-NVFP4 printed no summary line at all. The feature built to
// make silent routing visible was itself silent.
bool Engine::step() {
    const bool more = step_impl_();
    log_resolved_dispatch_once_();
    return more;
}

namespace {
struct OutsideTiming {
    double resume = 0, cp = 0, sched = 0, prefill = 0, decode_wrap = 0;
    int n = 0;
};
OutsideTiming g_ot;
}  // namespace

bool Engine::step_impl_() {
    const bool s_ot = runtime_config_.diagnostics.step_timing;
    std::chrono::steady_clock::time_point o0, o1, o2, o3, o4;
    if (s_ot)
        o0 = std::chrono::steady_clock::now();
    // Fast path: async conditional graph loop completed on GPU.
    int async_result = step_async_graph_resume();
    if (async_result == 1)
        return true;  // still running
    if (async_result == -1) {
        return scheduler_->has_pending() || scheduler_->active_count() > 0;
    }

    // Fast path: pipelined constrained decode (json/schema) — one token/tick.
    if (s_ot)
        o1 = std::chrono::steady_clock::now();
    int cp_result = step_constrained_pipeline();
    if (cp_result == 1)
        return true;
    if (cp_result == -1) {
        return scheduler_->has_pending() || scheduler_->active_count() > 0;
    }

    // Schedule prefill/decode batches and reconfigure green contexts.
    if (s_ot)
        o2 = std::chrono::steady_clock::now();
    if (!step_schedule()) {
        // No schedulable work — but an in-flight pipelined step may still
        // hold tokens + deferred KV (all rows finished last step). Drain so
        // nothing leaks while the engine idles.
        if (bd_pipe_.in_flight)
            drain_decode_pipeline();
        return false;
    }

    // Prefill/decode overlap engages only when the in-flight decode runs in
    // workspace slot 1 (>= 2 rows; the bs==1 regimes carry spec-verify
    // chunks, which share the CUTLASS activation scratch with prefill).
    const bool overlap_step = overlap_ready_ && sched_decode_batch_.size() >= 2;

    // A pipelined decode step in flight shares workspace 0 with prefill and
    // pins the batch composition — collect it before any prefill work or
    // when this step has no decode batch to continue it with. Under overlap
    // the decode step lives in slot 1 with its own quant scratches, so the
    // prefill no longer forces the drain: it enqueues on the low-priority
    // stream while the decode step is still running.
    if (bd_pipe_.in_flight &&
        ((!overlap_step && !sched_prefill_batch_.empty()) || sched_decode_batch_.empty()))
        drain_decode_pipeline();

    // Process prefill requests.
    if (s_ot)
        o3 = std::chrono::steady_clock::now();
    if (!sched_prefill_batch_.empty()) {
        if (overlap_step) {
            executor_->set_overlap_prefill_active(true);
            executor_->set_sample_slot_override(d_prefill_sample_);
        }
        step_prefill(prefill_stream());
        executor_->set_overlap_prefill_active(false);
        executor_->set_sample_slot_override(nullptr);
        ensure_prefill_workspace(executor_.get());
    }

    // Process decode requests (batched).
    if (s_ot)
        o4 = std::chrono::steady_clock::now();
    if (s_ot) {
        auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        g_ot.resume += us(o0, o1);
        g_ot.cp += us(o1, o2);
        g_ot.sched += us(o2, o3);
        g_ot.prefill += us(o3, o4);
        if (++g_ot.n >= 256) {
            const double inv = 1.0 / g_ot.n;
            IMP_LOG_INFO("outside-timing (n=%d): resume %.0f us, constrained %.0f, schedule %.0f, "
                         "prefill-block %.0f",
                         g_ot.n, g_ot.resume * inv, g_ot.cp * inv, g_ot.sched * inv, g_ot.prefill * inv);
            g_ot = {};
        }
    }
    if (!sched_decode_batch_.empty()) {
        step_decode(decode_stream());
        ensure_prefill_workspace(executor_.get());
    }

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
}

// The one answer site for graph eligibility (F-14).
//
// Eight sites across four TUs can turn graphs off. They stay where they are —
// two of them depend on state that does not exist until weight upload and
// warmup respectively — but they all route the *decision* through here, so the
// reason survives the call and prints in the resolved-dispatch summary instead
// of scrolling past in an init transcript.
void Engine::demote_graphs_(GraphDemotionReason reason) {
    if (!config_.use_cuda_graphs) {
        // Already off. Keep the first reason: it is the one that describes the
        // model, and a later demotion could not have taken effect anyway.
        return;
    }
    config_.use_cuda_graphs = false;
    graph_demotion_ = reason;
    IMP_LOG_INFO("CUDA graphs disabled: %s%s", graph_demotion_reason_name(reason),
                 graph_demotion_is_mid_run(reason) ? " (mid-run, one-way)" : "");
}

// One-shot summary of the kernels this model ACTUALLY resolved to (#1205).
//
// Emitted after the first step that has seen both a prefill and a decode, so
// every chain has had a chance to record. Everything here is read back from
// compute/dispatch_record.h — i.e. from the branches that ran, not from a
// prediction — so it cannot drift away from the dispatch the way a second copy
// of the routing rules would.
//
// Why this exists: the prefill chain has six tiers and the MoE chain five, and
// every one of them declines by returning `false` with no log. Before this,
// a model silently taking a slower or lower-quality path left no trace, which
// made every future routing regression invisible.
void Engine::log_resolved_dispatch_once_() {
    if (dispatch_dump_done_)
        return;
    const auto& r = dispatch_record::current();
    if (!r.has_prefill() || !r.has_decode())
        return;
    dispatch_dump_done_ = true;

    const bool is_moe = model_ && model_->profile().is_moe;

    char prefill[96];
    if (r.attn_prefill_outer == AttnPrefillOuter::FMHA_CHAIN && r.attn_prefill_tier_set) {
        snprintf(prefill, sizeof(prefill), "%s → %s", attn_prefill_outer_name(r.attn_prefill_outer),
                 attn_prefill_path_name(r.attn_prefill_tier));
    } else {
        snprintf(prefill, sizeof(prefill), "%s", attn_prefill_outer_name(r.attn_prefill_outer));
    }

    char moe[96];
    if (!is_moe) {
        snprintf(moe, sizeof(moe), "%s", moe_prefill_outer_name(MoePrefillOuter::NONE));
    } else if (r.moe_prefill_outer == MoePrefillOuter::CUTLASS3X && r.moe_prefill_tier_set) {
        snprintf(moe, sizeof(moe), "%s → %s", moe_prefill_outer_name(r.moe_prefill_outer),
                 moe_prefill_path_name(r.moe_prefill_tier));
    } else {
        snprintf(moe, sizeof(moe), "%s", moe_prefill_outer_name(r.moe_prefill_outer));
    }

    // graphs=0 alone made the interesting case unreadable — "off" and "off
    // because this model can never capture" are different facts (F-14).
    char graphs[64];
    if (config_.use_cuda_graphs)
        snprintf(graphs, sizeof(graphs), "1");
    else
        snprintf(graphs, sizeof(graphs), "0(%s)", graph_demotion_reason_name(graph_demotion_));

    IMP_LOG_INFO("Resolved dispatch: attn_prefill=%s attn_decode=%s moe_prefill=%s graphs=%s", prefill,
                 attn_decode_path_name(r.attn_decode), moe, graphs);
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

        // Keep the captured graph + block-table buffer parked so the next
        // burst of this request rearms instead of recapturing (~10-20 ms per
        // capture). Fully torn down on request finish or when a different
        // request launches (the FRESH path cleans a foreign park). This used
        // to be gated on spec_ngram_enabled_ — the burst-hybrid path was the
        // frequent relauncher — but the KV reservation clamps every burst to
        // ~128 steps (#1636), so a spec-OFF generation relaunched just as
        // often and paid a full recapture each time: 78 rebuilds x 27.8 ms in
        // a 116 s batch=1 bench window (nsys 2026-08-27), ~1.9% of wall.
        const bool park = saved_req && !generation_done;
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
// GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) with any KV dtype that
// has a paged_kv_gather kernel (FP16, FP8, NVFP4, MXFP4_KV, INT4, INT8).
// Returns false for Llama-4 and KV dtypes lacking a gather kernel.
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
    // KV dtypes wired through paged_kv_gather: FP16, FP8_E4M3, NVFP4, MXFP4_KV,
    // INT4, INT8. TurboQuant variants would need their own gather kernels.
    if (kv_cache_raw_) {
        QType kvt = kv_cache_raw_->qtype();
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3 && kvt != QType::NVFP4 && kvt != QType::MXFP4_KV &&
            kvt != QType::INT4 && kvt != QType::INT8)
            return false;
    }
    return true;
}

int Engine::resolve_prefill_chunk_size_() const {
    int explicit_val = config_.prefill_chunk_size;
    // #1645: the EngineConfig field is set by the CLI flag and the per-request
    // override; imp.conf reaches it through runtime.prefill_chunk_size, which
    // yields to an explicit CLI value.
    if (explicit_val < 0 && runtime_config_.runtime.prefill_chunk_size >= 0)
        explicit_val = runtime_config_.runtime.prefill_chunk_size;
    if (explicit_val < 0) {
        // Default 2048 since 2026-06-11. Chunked prefill re-reads
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

bool Engine::begin_perplexity_capture(std::span<const int32_t> tokens) {
    const int n = static_cast<int>(tokens.size());
    if (ppl_capture_.active || n < 2 || !executor_) {
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
    IMP_CUDA_CHECK_LOG(cudaMemcpy(ppl_capture_.d_tokens, tokens.data(),
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
    if (!runtime_config_.diagnostics.ppl_dump.empty()) {
        const char* dump = runtime_config_.diagnostics.ppl_dump.c_str();
        const bool full = (strcmp(dump, "full") == 0);
        std::string nll_line;
        for (int i = 0; i < n - 1; ++i) {
            if (full || i < 16 || i % 16 == 0 || i > n - 6) {
                char buf[32];
                snprintf(buf, sizeof(buf), " [%d]=%.3f", i, h_nll_pos[i]);
                nll_line += buf;
            }
        }
        IMP_LOG_DEBUG("[PPL-DUMP] per-pos nll:%s", nll_line.c_str());
    }
    double h_nll = 0.0;
    for (int i = first; i <= last; ++i)
        h_nll += h_nll_pos[i];
    const int counted = last - first + 1;
    double ppl = std::exp(h_nll / static_cast<double>(counted));
    // A run that produced no NLL at all reports PPL = exp(0) = 1.0000, which
    // reads as a perfect score. That is the worst direction for a defect to
    // fail in, and it happened: an aborted run ("perplexity failed:
    // insufficient KV capacity") logged `mean_nll=0.0000 PPL=1.0000` on the
    // next line, and anyone grepping for the PPL line reads the failure as a
    // result. Exactly zero summed NLL over a non-empty span is impossible for a
    // real forward (every position contributes a positive term), so it is the
    // signature of a buffer nobody filled. Say that instead of printing a
    // number. The CLI's own result line is already unreachable on failure
    // (imp-cli/main.cpp), so this closes the log-reader's path, not a gate's.
    if (counted > 0 && h_nll == 0.0) {
        IMP_LOG_WARN("perplexity_nll: n=%d counted=%d but the summed NLL is exactly 0 - "
                     "no forward filled this buffer. NOT a perplexity of 1.0; the run failed.",
                     n, counted);
        // false, not true-with-1.0: this function's contract is "did a
        // perplexity happen", and the caller already turns false into a stderr
        // line and exit 1. Returning true here handed a sentinel to anyone who
        // asked for a number. *out_ppl is deliberately left untouched.
        return false;
    }
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

    // Speculation: when a draft is available, the verify step replaces this
    // decode step entirely (it allocates its own KV blocks and emits accepted
    // tokens). Falls through to the normal path on a draft miss or when any
    // gate fails. The draft may come from the n-gram/suffix matcher, the MTP
    // head or token recycling, so the entry gate asks whether ANY drafter is
    // enabled — gating this on the n-gram flag alone left MTP unreachable.
    if (decode_batch.size() == 1 && spec_any_drafter_enabled_(*decode_batch[0])) {
        spec_maybe_rearm_(*decode_batch[0]);
        if (spec_verify_gates_ok_(*decode_batch[0])) {
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
                   // An MTP-bound request must stay on the eager path: the
                   // per-step chain feed below needs host hiddens for every
                   // token, and one device-side burst desyncs the MTP cache
                   // for the rest of the generation (the sync gate then skips
                   // feeding forever, #847). This think-burst was the one loop
                   // site without the MTP exclusion the process_outputs launch
                   // has - measured on Qwen3.8-27B-NVFP4: drafted_total 1 vs
                   // 436 over a 768-token essay, 84.7 vs 104.3 tok/s.
                   !(mtp_spec_decode_enabled() && mtp_bound_req_ == decode_batch[0]->id) &&
                   spec_verify_gates_ok_(*decode_batch[0], /*ignore_think=*/true) &&
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
                // Name the gate. The field dump below stays because a reader
                // usually wants the neighbouring values too, but it is no
                // longer the only thing to go on (#1538, #1539).
                const char* why = spec_verify_gate_refusal_(r);
                IMP_LOG_INFO(
                    "spec-ngram: gates failed, refused by '%s' (temp=%.2f top_k=%d rep_pen=%.2f freq=%.2f "
                    "pres=%.2f dry=%.2f mirostat=%d bias=%zu logprobs=%d json=%d schema=%d "
                    "think_budget=%.2f ssm=%d moe=%d moe_nvfp4=%d spec_moe=%d mtp=%d "
                    "chunked_prefill=%d)",
                    why ? why : "nothing (raced)", r.temperature, r.top_k, r.repetition_penalty,
                    r.frequency_penalty,
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
            if (!spec_verify_gates_ok_(*r))
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
    // Batched GDN decode runs all of them in one step; the rotation below is
    // the fallback for when it is off or unavailable (no slot table). See
    // runtime.gdn_batched_decode.
    const bool gdn_batch_ok = runtime_config_.runtime.gdn_batched_decode && d_ssm_seq_slots_ != nullptr;
    if (ssm_state_ && decode_batch.size() > 1 && !gdn_batch_ok) {
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
            if (new_block < 0 && kv_cache_raw_ != nullptr) {
                // A growable pool gets one chance to answer with memory before
                // the sequence is cancelled. Admission alone is not enough: a
                // generation that was admitted can still outgrow the pool
                // block by block, and without this a pool that started small
                // cancelled every long generation mid-decode — measured, a
                // synthetic 8192-token run produced ZERO tokens where a fixed
                // pool produced 354 tok/s.
                //
                // Grown in coarse steps rather than one block at a time: the
                // cost is per driver mapping call, not per byte.
                const int have = kv_cache_raw_->total_blocks();
                if (kv_cache_raw_->ceiling_blocks() > have) {
                    kv_cache_raw_->try_grow_to(have + std::max(64, have / 4));
                    new_block = kv_manager_->append_block(req->id);
                }
            }
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
                kv_pressure_rejections_.fetch_add(1, std::memory_order_relaxed);
                cancel_sequence_(req);
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
                kv_pressure_rejections_.fetch_add(1, std::memory_order_relaxed);
                cancel_sequence_(req);
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
                    demote_graphs_(GraphDemotionReason::StreamingKvKvPressure);
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
                    // Idempotent: returns 0 once this sequence is fully
                    // streamed, so accumulating gives the total context this
                    // request lost — which is what the caller is told.
                    const int freed = kv_manager_->evict_middle_blocks(req->id, n_sinks, win);
                    if (freed > 0)
                        req->evicted_kv_tokens += freed * kv_bs;
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

int Engine::penalty_hist_slot_(int req_id, const std::vector<std::shared_ptr<Request>>& batch) {
    if (d_penalty_hist_ == nullptr || penalty_hist_slots_ <= 0)
        return -1;
    int free_i = -1;
    for (int i = 0; i < penalty_hist_slots_; i++) {
        if (penalty_hist_state_[i].req_id == req_id)
            return i;
        if (penalty_hist_state_[i].req_id < 0 && free_i < 0)
            free_i = i;
    }
    if (free_i < 0) {
        // Evict a slot whose owner is not in the current batch. One exists:
        // the caller's row has no slot, so at most batch.size()-1 slots are
        // owned by current rows and penalty_hist_slots_ >= max_batch_size.
        for (int i = 0; i < penalty_hist_slots_ && free_i < 0; i++) {
            bool live = false;
            for (const auto& r : batch)
                if (r->id == penalty_hist_state_[i].req_id) {
                    live = true;
                    break;
                }
            if (!live)
                free_i = i;
        }
    }
    if (free_i < 0)
        return -1;
    penalty_hist_state_[free_i].req_id = req_id;
    penalty_hist_state_[free_i].synced = -1;
    return free_i;
}

// Populate the InferenceState for a decode step from the uploaded GPU batch.
// Handles per-seq residual metadata (single-seq fast path vs multi-seq
// per-batch upload), sampling params, decode-step seed, penalties, recurrent
// state, and JSON/schema constrainer attach. Returns needs_logprobs so the
// caller knows whether to capture decode_logits_out for the logprobs pass.
void Engine::decode_build_inference_state_(GPUBatch& gpu_batch,
                                           std::vector<std::shared_ptr<Request>>& valid_decode,
                                           int max_ctx, cudaStream_t dec_stream,
                                           InferenceState& state, bool& needs_logprobs,
                                           bool& needs_constrained) {
    state.token_ids = gpu_batch.d_token_ids;
    state.positions = gpu_batch.d_positions;
    state.n_tokens = gpu_batch.total_tokens;
    state.n_sequences = gpu_batch.n_sequences;

    // Batched GDN decode: hand the recurrent kernels one slot id per sequence.
    // Only for a genuine multi-sequence decode step — a single sequence keeps
    // the pointer null and takes the path it always took.
    state.ssm_seq_slots = nullptr;
    state.ssm_n_seq = 1;
    if (ssm_state_ && d_ssm_seq_slots_ && gpu_batch.n_sequences > 1 &&
        gpu_batch.total_tokens == gpu_batch.n_sequences &&
        static_cast<size_t>(gpu_batch.n_sequences) <= h_ssm_seq_slots_.size()) {
        bool all_known = true;
        for (int i = 0; i < gpu_batch.n_sequences; ++i) {
            auto it = recurrent_slot_of_.find(valid_decode[static_cast<size_t>(i)]->id);
            if (it == recurrent_slot_of_.end()) {
                all_known = false;
                break;
            }
            h_ssm_seq_slots_[static_cast<size_t>(i)] = it->second;
        }
        // A request without a slot would read another sequence's state, so the
        // whole step falls back rather than guessing one.
        if (all_known) {
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_ssm_seq_slots_, h_ssm_seq_slots_.data(),
                                               static_cast<size_t>(gpu_batch.n_sequences) * sizeof(int),
                                               cudaMemcpyHostToDevice, dec_stream));
            state.ssm_seq_slots = d_ssm_seq_slots_;
            state.ssm_n_seq = gpu_batch.n_sequences;
        }
    }
    state.max_blocks_per_seq = gpu_batch.max_blocks_per_seq;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = gpu_batch.d_block_tables;
    state.block_tables_swa = gpu_batch.d_block_tables_swa;
    state.context_lens = gpu_batch.d_context_lens;
    state.max_context_len = max_ctx;
    state.is_prefill = false;
    state.kv_manager = kv_manager_.get();
    bind_mrope_decode_(state, valid_decode, dec_stream);
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
            // The buffer is persistent and sized for max_batch_size (#1648):
            // a per-step cudaMallocAsync address was baked into a captured
            // graph that is replayed, and nothing invalidated the graph when
            // the allocator handed back a different one. Stride by the
            // CAPACITY, not by N, so the three sub-arrays keep fixed offsets
            // across steps whatever the batch width - a graph captured at one
            // N stays correct at another.
            if (residual_meta_d_buf_ != nullptr && N <= residual_meta_capacity_) {
                int* base = residual_meta_d_buf_;
                const ptrdiff_t stride = residual_meta_capacity_;
                cudaMemcpyAsync(base + 0 * stride, residual_meta_h_slots_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + 1 * stride, residual_meta_h_counts_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + 2 * stride, residual_meta_h_widxes_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                state.d_residual_seq_slots = base + 0 * stride;
                state.d_residual_counts = base + 1 * stride;
                state.d_residual_write_idxes = base + 2 * stride;
                state.h_residual_seq_ids = residual_meta_h_seq_ids_.data();
            } else {
                // Neither case is expected: the buffer is sized for
                // max_batch_size at init and admission is clamped to it. Say
                // so once rather than decoding with stale metadata.
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    IMP_LOG_WARN(
                        "residual metadata unavailable for a %d-sequence decode "
                        "(buffer=%p, capacity=%d); this step runs without it",
                        N, static_cast<const void*>(residual_meta_d_buf_), residual_meta_capacity_);
                }
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
    // ONE flag for "this batch is driven by a host-side constraint FSM".
    // It used to be two (json / schema) and every consumer only ever asked for
    // their disjunction — which meant a new constrainer had to be remembered in
    // both, and regex (#1091) was in neither, so a regex request could never
    // reach the constrained pipeline and silently stayed on the eager path.
    needs_logprobs = false;
    needs_constrained = false;
    for (const auto& r : valid_decode) {
        if (r->logprobs)
            needs_logprobs = true;
        if (r->json_mode || !r->json_schema.empty() || !r->tool_constraint_tools.empty() ||
            !r->regex_pattern.empty() || !r->grammar.empty())
            needs_constrained = true;
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
        state.regex_constrainer = valid_decode[0]->constraints->regex_constrainer();
        state.grammar_constrainer = valid_decode[0]->constraints->grammar_constrainer();
    }
}

namespace {
// diagnostics.step_timing: host-side phase attribution for a batched decode
// step. The GPU-side gap profile said 143 us/token of idle sits between the
// sampler tail and the next step's first kernel; this measures WHERE the
// host spends that time. Aggregated every 256 steps at n>1.
struct StepTiming {
    double build = 0, fwd = 0, sample = 0, dist = 0, outside = 0;
    int n = 0;
    std::chrono::steady_clock::time_point last_end{};
};
StepTiming g_step_timing;
}  // namespace

void Engine::step_decode_forward(std::vector<std::shared_ptr<Request>>& valid_decode,
                                 cudaStream_t dec_stream) {
    const bool s_timing = runtime_config_.diagnostics.step_timing && valid_decode.size() > 1;
    std::chrono::steady_clock::time_point tp0, tp1, tp2, tp3;
    if (s_timing) {
        tp0 = std::chrono::steady_clock::now();
        if (g_step_timing.last_end.time_since_epoch().count() != 0)
            g_step_timing.outside +=
                std::chrono::duration<double, std::micro>(tp0 - g_step_timing.last_end).count();
    }
    // Switch workspace for decode. The historical slot-1 path was bs==1 only
    // (its 2026-03 shape, pre-continuous-batching); under prefill overlap
    // the batched step runs there too — the warmup sized slot 1 and the
    // per-slot quant scratches for max_batch, and the workspace choice must
    // be a pure function of (flags, batch size) so graph captures stay
    // consistent with replays.
    const bool slot1 = executor_->has_decode_workspace() &&
                       (valid_decode.size() == 1 ||
                        (overlap_ready_ &&
                         static_cast<int>(valid_decode.size()) <= executor_->decode_max_batch()));
    if (slot1) {
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
        decode_builder_.add_decode_sequence(last_token, position, bt, ctx_len,
                                            swa_sizing_active_ ? std::span<const int>(sbt)
                                                               : std::span<const int>{});
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
    bool needs_constrained = false;
    if (s_timing)
        tp1 = std::chrono::steady_clock::now();
    decode_build_inference_state_(gpu_batch, valid_decode, max_ctx, dec_stream, state, needs_logprobs,
                                  needs_constrained);

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
        // Device-resident penalty histories (#1755): row i's history lives in
        // slot hist_slot[i]; after the gather ONE kernel appends this step's
        // sampled tokens (offs[i] < 0 skips a row). Replaces a pageable H2D
        // of the whole output history per row per step.
        imp::PenaltyAppendArgs pen_append;
        pen_append.n = n;
        pen_append.cap = penalty_hist_cap_;
        std::vector<int> pen_row_slot(n, -1);
        for (int i = 0; i < n; i++)
            pen_append.offs[i] = -1;
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
            if (req_needs_pen && !req->output_tokens.empty()) {
                const int need = static_cast<int>(req->output_tokens.size());
                int slot = penalty_hist_slot_(req->id, valid_decode);
                if (slot >= 0 && need <= penalty_hist_cap_) {
                    auto& hs = penalty_hist_state_[slot];
                    if (hs.synced != need) {
                        // Full (re)sync — first batch entry of this request, or
                        // the host history diverged (sync-row step, think strip).
                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(
                            d_penalty_hist_ + (size_t)slot * penalty_hist_cap_,
                            req->output_tokens.data(), (size_t)need * sizeof(int32_t),
                            cudaMemcpyHostToDevice, dec_stream));
                        hs.synced = need;
                    }
                    per_state.penalty_tokens = d_penalty_hist_ + (size_t)slot * penalty_hist_cap_;
                    per_state.n_penalty_tokens = need;
                    pen_row_slot[i] = slot;
                } else if (d_penalty_tokens_ &&
                           (size_t)need <= d_penalty_tokens_capacity_) {
                    // No slot pool — the old shared-buffer upload path.
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_penalty_tokens_, req->output_tokens.data(),
                                                       (size_t)need * sizeof(int32_t),
                                                       cudaMemcpyHostToDevice, dec_stream));
                    per_state.penalty_tokens = d_penalty_tokens_;
                    per_state.n_penalty_tokens = need;
                }
            }
            // Per-row constraint masks: keeps json_schema/json_mode enforced
            // when the request shares a decode batch (the batch-level state
            // carries no constrainers at n>1; the row sampler applies the
            // mask to this row's logits before sampling).
            if (req->constraints) {
                per_state.schema_constrainer = req->constraints->schema_constrainer();
                per_state.json_constrainer = req->constraints->json_constrainer();
                per_state.regex_constrainer = req->constraints->regex_constrainer();
                per_state.grammar_constrainer = req->constraints->grammar_constrainer();
            } else {
                per_state.schema_constrainer = nullptr;
                per_state.json_constrainer = nullptr;
                per_state.regex_constrainer = nullptr;
                per_state.grammar_constrainer = nullptr;
            }
            per_state.n_sequences = 1;
            Tensor seq_logits = logits.slice(i, i + 1);
            if (!executor_->sample_single_from_logits_async(seq_logits, per_state, i, dec_stream)) {
                sync_rows.push_back(i);
                // Sync-row token never lands in sample slot i — the device
                // history would diverge; force a resync next step.
                if (pen_row_slot[i] >= 0)
                    penalty_hist_state_[pen_row_slot[i]].synced = -1;
            } else if (pen_row_slot[i] >= 0) {
                pen_append.slots[i] = pen_row_slot[i];
                pen_append.offs[i] = static_cast<int>(req->output_tokens.size());
            }
        }
        if (static_cast<int>(sync_rows.size()) < n) {
            const int32_t* toks = executor_->collect_sampled_tokens(n, dec_stream);
            if (toks) {
                for (int i = 0; i < n; i++)
                    result[i] = toks[i];
                // Append this step's tokens to the device histories. AFTER
                // collect on purpose: the row-batched top-k stash only writes
                // its sample slots inside collect's flush, and the parity has
                // not flipped yet, so the slots still hold this step.
                bool any_append = false;
                for (int i = 0; i < n && !any_append; i++)
                    any_append = pen_append.offs[i] >= 0;
                if (any_append &&
                    executor_->append_sampled_history(pen_append, d_penalty_hist_, dec_stream)) {
                    for (int i = 0; i < n; i++)
                        if (pen_append.offs[i] >= 0)
                            penalty_hist_state_[pen_append.slots[i]].synced = pen_append.offs[i] + 1;
                }
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
                per_state.regex_constrainer = req->constraints->regex_constrainer();
                per_state.grammar_constrainer = req->constraints->grammar_constrainer();
            } else {
                per_state.schema_constrainer = nullptr;
                per_state.json_constrainer = nullptr;
                per_state.regex_constrainer = nullptr;
                per_state.grammar_constrainer = nullptr;
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
        if (!needs_logprobs && !needs_constrained && pipeline_batch_eligible_(valid_decode)) {
            piped = pipeline_enter_(valid_decode, gpu_batch, graph_idx, state, logits_out,
                                    dec_stream, tokens);
            static bool entered_logged = false;
            if (!entered_logged) {
                entered_logged = true;
                IMP_LOG_INFO("decode-pipeline: first eligible batch n=%d -> %s",
                             (int)valid_decode.size(), piped ? "ENTERED" : "enter DECLINED");
            }
        } else if (valid_decode.size() >= 2) {
            log_pipeline_gate_once_(valid_decode);
        }
        if (s_timing)
            tp2 = std::chrono::steady_clock::now();
        // Eager sampling (handles all modes: greedy, top-k/p, penalties,
        // force_token, constraints, logprobs, mirostat)
        if (!piped)
            tokens = sample_per_request(logits_out);
        if (s_timing)
            tp3 = std::chrono::steady_clock::now();
        if (needs_logprobs)
            decode_logits_out = logits_out;
    } else {
        executor_->forward_logits(state, decode_logits_out, dec_stream);
        tokens = sample_per_request(decode_logits_out);
    }

    if (!decode_batch_pool_.is_allocated()) {
        gpu_batch.free();
    }

    // (The residual metadata buffer is persistent since #1648 - allocated once
    // beside d_kv_slot_buf_ and freed with it in the destructor. Freeing it per
    // step is what made a replayed graph hold a dangling address.)

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
            // The top-w probe is opt-in: it costs a single-block scan of the
            // whole vocabulary per width plus a host sync, per drafted token,
            // and only imp-cli's tree-ceiling table reads it.
            const bool want_probe = runtime_config_.diagnostics.mtp_tree_probe;
            // Device-side chain, the same one mtp_feed_pairs_ uses: each step's
            // argmax lands in ws->d_chain_tokens[k] and feeds step k+1's
            // embedding lookup on device, so ONE drain replaces K host
            // round-trips. Without it every drafted token ends in a
            // cudaStreamSynchronize and the GPU idles while the host issues the
            // next step — measured on Qwen3.8-27B, ~6 ms of the 6.85 ms per
            // drafted token was that dead time rather than kernel work.
            //
            // The probe needs its top-w on the host per step, so it keeps the
            // round-trip path; it is a measurement mode, not a serving one.
            const bool device_chain = !want_probe && ws->d_chain_tokens != nullptr && ws->n_experts == 0 &&
                                      K <= imp::kMtpMaxChainK;
            if (device_chain) {
                int launched = 0;
                for (int k = 0; k < K; ++k) {
                    const bool first = (k == 0);
                    if (!mtp_draft_one(first ? chain_prev_tok : -1, first ? chain_h_prev : ws->d_h_final,
                                       hidden_dim, vocab_size, nullptr, nullptr, 0,
                                       first ? nullptr : ws->d_chain_tokens + k - 1,
                                       ws->d_chain_tokens + k)) {
                        break;
                    }
                    launched++;
                }
                int32_t h_chain[imp::kMtpMaxChainK];
                if (launched > 0 && cudaMemcpyAsync(h_chain, ws->d_chain_tokens,
                                                    static_cast<size_t>(launched) * sizeof(int32_t),
                                                    cudaMemcpyDeviceToHost, decode_stream()) == cudaSuccess) {
                    cudaStreamSynchronize(decode_stream());
                    for (int k = 0; k < launched; ++k) {
                        const int prediction = h_chain[k];
                        if (prediction < 0 || prediction >= vocab_size)
                            break;  // NaN-logits guard — keep the valid prefix
                        // topk stays unset: -1 is the "not a candidate"
                        // sentinel the width tally compares against, and a
                        // zero-filled array would count token id 0 as a hit.
                        MtpChainEntry entry{prediction, k, cur_pos + 1 + k, {}};
                        for (int w = 0; w < Engine::kMtpMeasureW; ++w)
                            entry.topk[w] = -1;
                        mtp_pending_chain_.push_back(entry);
                        chain_preds.push_back(prediction);
                        if (k == 0)
                            mtp_pending_prediction_ = prediction;
                    }
                }
            } else {
                for (int k = 0; k < K; ++k) {
                    int prediction = -1;
                    int topk[Engine::kMtpMeasureW] = {-1, -1, -1, -1};
                    if (!mtp_draft_one(chain_prev_tok, chain_h_prev, hidden_dim, vocab_size, &prediction,
                                       want_probe ? topk : nullptr, want_probe ? Engine::kMtpMeasureW : 0)) {
                        break;
                    }
                    MtpChainEntry entry{prediction, k, cur_pos + 1 + k, {}};
                    for (int w = 0; w < Engine::kMtpMeasureW; ++w)
                        entry.topk[w] = topk[w];
                    mtp_pending_chain_.push_back(entry);
                    chain_preds.push_back(prediction);
                    // For k=0 only, also feed pending_prediction_ (legacy 1-step
                    // accuracy counter remains in sync with chain_accept_[0]).
                    if (k == 0)
                        mtp_pending_prediction_ = prediction;
                    // Chain: next iter uses this prediction + the MTP's own h_final.
                    chain_prev_tok = prediction;
                    chain_h_prev = ws->d_h_final;
                }
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
    step_decode_process_outputs(valid_decode, tokens, decode_logits_out, needs_logprobs, needs_constrained,
                                dec_stream);
    if (s_timing) {
        auto tp4 = std::chrono::steady_clock::now();
        // Non-graph fallback path never sets tp2/tp3: fold everything after
        // build into "fwd" rather than reporting garbage.
        if (tp2.time_since_epoch().count() == 0)
            tp2 = tp4;
        if (tp3.time_since_epoch().count() == 0)
            tp3 = tp4;
        auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        g_step_timing.build += us(tp0, tp1);
        g_step_timing.fwd += us(tp1, tp2);
        g_step_timing.sample += us(tp2, tp3);
        g_step_timing.dist += us(tp3, tp4);
        g_step_timing.last_end = tp4;
        if (++g_step_timing.n >= 256) {
            const double inv = 1.0 / g_step_timing.n;
            IMP_LOG_INFO("step-timing (n=%d, batch=%zu): build %.0f us, fwd-enqueue+wait %.0f, "
                         "sample %.0f, distribute %.0f, outside-step %.0f  (per step)",
                         g_step_timing.n, valid_decode.size(), g_step_timing.build * inv,
                         g_step_timing.fwd * inv, g_step_timing.sample * inv, g_step_timing.dist * inv,
                         g_step_timing.outside * inv);
            g_step_timing = {};
        }
    }
}

// =====================================================================
// step_decode_process_outputs — extract logprobs, distribute tokens,
//                                try async graph loop
// =====================================================================

void Engine::step_decode_process_outputs(std::vector<std::shared_ptr<Request>>& valid_decode,
                                         const std::vector<int32_t>& tokens, const Tensor& decode_logits_out,
                                         bool needs_logprobs, bool needs_constrained,
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
        !needs_logprobs && !needs_constrained) {
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

    // Constrained requests (json_mode / json_schema / regex / GBNF grammar)
    // can't run the conditional loop (the FSM is host-side) — launch the
    // pipelined constrained decode instead: per tick the host enqueues
    // mask+sample AND the next forward, hiding FSM/mask latency under GPU
    // compute. This matters most for the two newest constrainers, whose cold
    // masks walk the whole vocabulary. masked_sample_async
    // covers banned tokens + greedy/top-k/top-p only — penalties or any
    // host-side sampling feature stays on the eager path.
    if (decode_graph_pool_[0].graph_path_available() && valid_decode.size() == 1 && !offload_mgr_ &&
        config_.use_cuda_graphs && !async_graph_runner_.is_setup() && !cpipe_.active && !needs_logprobs &&
        needs_constrained) {
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
