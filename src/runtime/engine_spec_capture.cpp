// engine_spec_capture.cpp: graph-captured verify chunk (#847).
//
// Capture mode (InferenceState::ctx_capacity > 0): kernels read real lengths
// from device (context_lens[0], d_past_len) instead of baking q_offset/ctx_len;
// grids and K/V scratch size once for ctx_capacity; per-step values refresh via
// H2D (d_spec_tokens_/positions_/block_table_/context_len_/past_len_).
//
// Bucketing pads drafts to {9, 17, 33, k_max+1} tokens with copies of t0;
// padded rows sit after every real row (invisible under causal masking) and
// are dropped by the same rollback as rejected drafts; verify reads argmax
// rows [0, real_chunk_len) only.
//
// Hybrids (SSMState): pad rows would corrupt conv/scan state in place, so
// chunk kernels read the real chunk length from device (d_chunk_len) and stop
// state updates at the last real row; slab pointers (seq_base(slot)) are
// baked in, so the graph cache key includes the recurrent slot.
//
// First use of a bucket runs eager through the same capture-mode path (algo
// warmup); second use captures. Capture/launch failure falls back to eager;
// repeated failures permanently disable capture for the process. Graphs bake
// pointers into the executor workspace and spec staging buffers; both
// invalidate the cache on move (workspace_generation; free_spec_buffers_ ->
// free_spec_graphs_).

#include "compute/gemm.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "runtime/engine.h"

#include <chrono>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

// Pre-sizes the speculative-verify allocations (chunk-capture K/V scratch,
// consolidated spec staging block, verify argmax/penalty scratch) that would
// otherwise be taken lazily during serving, avoiding a race against the KV
// pool for capacity (#1103).
//
// Capacity comes from the same expressions the runtime uses, evaluated at
// their maxima; a later call with a smaller request hits the >= guards inside
// ensure_* and allocates nothing.
void Engine::prewarm_spec_scratch_() {
    const auto& scfg = runtime_config_.speculative;
    const bool spec_on = scfg.ngram || scfg.suffix || scfg.capture || mtp_spec_decode_enabled();
    if (!spec_on || !executor_)
        return;

    // Goes through spec_capture_ready_ (not a direct resolve) so its
    // eligibility guards (host-offload, residual KV, census probe) are not
    // duplicated; ctx_padded=1 makes this a side-effect-only call.
    if (scfg.capture)
        (void)spec_capture_ready_(1);

    // Staging + block tables, at the caps engine_spec_ngram.cpp would reach:
    // the largest capture bucket, and a block table covering the whole context.
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    const int chunk_cap = std::max({spec_capture_bucket_max_(), scfg.k + 1, 33});
    const int ctx_for_table = std::max(config_.max_seq_len, spec_capture_ctx_cap_);
    const int table_cap = (ctx_for_table + kv_bs - 1) / kv_bs + 16;
    if (!ensure_spec_buffers_(chunk_cap, table_cap)) {
        IMP_LOG_WARN("[spec] scratch prewarm failed (chunk_cap=%d table_cap=%d) — "
                     "buffers will be taken on first use instead",
                     chunk_cap, table_cap);
        return;
    }
    executor_->prewarm_verify_scratch();
    IMP_LOG_INFO("[spec] scratch prewarmed: chunk_cap=%d table_cap=%d ctx_cap=%d", chunk_cap,
                 table_cap, spec_capture_ctx_cap_);
}

int Engine::spec_capture_bucket_max_() const {
    const auto& scfg = runtime_config_.speculative;
    int k_max = std::max(1, scfg.k);
    if (scfg.suffix)
        k_max = std::max(k_max, scfg.suffix_k_max);
    if (mtp_spec_decode_enabled())
        k_max = std::max(k_max, mtp_spec_decode_k());
    return k_max + 1;
}

// Context tier: power of two >= ctx (floor 4096), clamped to the resolved
// capacity. Sizes the baked gather grids close to the real context.
int Engine::spec_capture_ctx_tier_(int ctx_padded) const {
    int tier = 4096;
    while (tier < ctx_padded)
        tier <<= 1;
    return std::min(tier, std::max(spec_capture_ctx_cap_, ctx_padded));
}

int Engine::spec_capture_bucket_(int chunk_len) const {
    const int cap = std::max(chunk_len, spec_capture_bucket_max_());
    // 3/5 buckets (#964): split-K count derives from the PADDED row count at
    // capture time, so finer buckets keep it close to the real chunk (a 2-row
    // draft padded to 9 baked 5 splits instead of 21, KV walk grew 4x).
    // 4 = token-recycling depth-3 chunk (#1055): one batched-GEMV weight
    // sweep (MR=4); padding to 5 pays a second sweep.
    // 6/8 = W=2 multi-candidate chunk at depth 2/3 (W*(1+depth) rows);
    // rounding to 9 wastes a third of the rows.
    for (int b : {3, 4, 5, 6, 8, 9, 17, 33}) {
        if (chunk_len <= b && b <= cap)
            return b;
    }
    return cap;
}

bool Engine::spec_capture_ready_(int ctx_padded) {
    const auto& scfg = runtime_config_.speculative;
    if (!scfg.capture || spec_capture_doomed_)
        return false;
    // The census probe owns the forward when enabled (capture+destroy per
    // chunk, diagnostics only).
    if (runtime_config_.diagnostics.spec_capture_probe)
        return false;
    // SSMState hybrids ARE capture-eligible (pad rows read chunk length from
    // device, cache keyed on the recurrent slot); MoE host-offload is not: it
    // syncs on the host per layer.
    if (offload_mgr_)
        return false;
    // BitDecoding residual KV advances ring state on the host per forward.
    if (kv_manager_ && kv_manager_->residual_enabled())
        return false;
    if (!executor_)
        return false;
    if (spec_capture_ctx_cap_ < 0) {  // resolve once per engine
        spec_capture_ctx_cap_ = 0;
        if (executor_->chunk_capture_supported()) {
            int cap = scfg.capture_ctx_cap;
            if (config_.max_seq_len > 0)
                cap = std::min(cap, config_.max_seq_len);
            if (cap > 0 && executor_->ensure_chunk_capture_scratch(cap))
                spec_capture_ctx_cap_ = cap;
        }
        IMP_LOG_INFO("[spec-capture] %s (ctx_cap=%d)",
                     spec_capture_ctx_cap_ > 0 ? "enabled" : "not applicable for this model",
                     spec_capture_ctx_cap_);
    }
    return spec_capture_ctx_cap_ > 0 && ctx_padded <= spec_capture_ctx_cap_;
}

void Engine::free_spec_graphs_() {
    // SpecVerifyGraph::exec is a CudaGraphExec — clear() destroys the handles.
    spec_graphs_.clear();
}

bool Engine::spec_captured_forward_(InferenceState& state, Tensor& logits_out,
                                    cudaStream_t stream) {
    // The graphs bake workspace pointers — invalidate when the arena moved.
    const uint64_t ws_gen = executor_->workspace_generation();
    if (ws_gen != spec_capture_ws_gen_) {
        if (!spec_graphs_.empty()) {
            IMP_LOG_INFO("[spec-capture] workspace reallocated — dropping %zu cached graphs",
                         spec_graphs_.size());
            free_spec_graphs_();
        }
        spec_capture_ws_gen_ = ws_gen;
    }

    // Hybrids bake the recurrent slab pointer (seq_base(slot)) into the
    // graph, so slot is part of the cache key. Batched verify addresses
    // slots via device tables (no baked pointer), so it keys like a dense
    // model.
    const bool batched_verify = state.ssm_out_slots != nullptr && state.ssm_snap_slots != nullptr;
    const int rec_slot = (state.ssm_state && !batched_verify) ? state.ssm_seq_id : -1;
    const int grouped_rows = state.ssm_grouped_chunk() ? state.ssm_seq_tokens : 0;
    auto& slot = spec_graphs_[{state.n_tokens, state.ctx_capacity, rec_slot, grouped_rows}];
    if (slot.exec) {
        // diagnostics.spec_capture_fidelity: verifies a cached graph reproduces
        // an eager forward of the same state (restores the recurrent slab
        // pre-replay; non-hybrids need no restore) and diffs the row-0 logits.
        // Not used for the batched verify: its eager forward mutates live
        // slots in place, so a second forward of the same state cannot be
        // staged.
        if (runtime_config_.diagnostics.spec_capture_fidelity && model_ && !batched_verify) {
            const bool hybrid_restore = state.ssm_state != nullptr && spec_state_scratch_ != nullptr &&
                                        rec_slot >= 0;
            const size_t vocab = static_cast<size_t>(model_->config_.vocab_size);
            std::vector<float> eager(vocab), graph(vocab);
            executor_->forward_logits(state, logits_out, stream);
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            IMP_CUDA_CHECK_LOG(
                cudaMemcpy(eager.data(), logits_out.data, vocab * sizeof(float), cudaMemcpyDeviceToHost));
            if (hybrid_restore) {
                IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(ssm_state_->seq_base(rec_slot), spec_state_scratch_,
                                                   ssm_state_->per_seq_bytes(), cudaMemcpyDeviceToDevice,
                                                   stream));
            }
            if (cudaGraphLaunch(slot.exec, stream) == cudaSuccess) {
                IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
                IMP_CUDA_CHECK_LOG(
                    cudaMemcpy(graph.data(), logits_out.data, vocab * sizeof(float), cudaMemcpyDeviceToHost));
                size_t ie = 0, ig = 0;
                double mx = 0.0;
                for (size_t i = 0; i < vocab; i++) {
                    const double d = std::fabs(static_cast<double>(eager[i]) - static_cast<double>(graph[i]));
                    if (d > mx)
                        mx = d;
                    if (eager[i] > eager[ie])
                        ie = i;
                    if (graph[i] > graph[ig])
                        ig = i;
                }
                spec_fidelity_checked_++;
                if (mx > spec_fidelity_max_delta_)
                    spec_fidelity_max_delta_ = mx;
                if (ie != ig) {
                    spec_fidelity_differing_++;
                    IMP_LOG_WARN(
                        "[spec-capture] cached graph disagrees with an eager forward of the "
                        "same state: argmax %zu vs %zu, max|dlogit|=%.4f (checked=%lld "
                        "differing=%lld)",
                        ig, ie, mx, spec_fidelity_checked_, spec_fidelity_differing_);
                }
                return true;
            }
        }
        cudaError_t err = cudaGraphLaunch(slot.exec, stream);
        if (err == cudaSuccess)
            return true;
        IMP_LOG_WARN("[spec-capture] graph launch failed (%s) — dropping graph cache",
                     cudaGetErrorString(err));
        cudaGetLastError();
        free_spec_graphs_();
        return false;  // caller runs the eager forward
    }
    if (slot.eager_uses++ == 0)
        return false;  // warmup: caller runs eager through the capture-mode path

    auto doom_check = [this](const char* why) {
        if (++spec_capture_failures_ >= 2) {
            spec_capture_doomed_ = true;
            IMP_LOG_WARN("[spec-capture] disabled after repeated failures (%s)", why);
        }
    };

    // Permanent telemetry: the first-use gap of a bucket is the largest
    // inter-token gap a warm server still shows (39-93 ms vs 10.7 ms steady
    // steps on Qwen3.8-27B-NVFP4); only a server log prices it without CUPTI
    // inflation.
    const auto t_cap0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("[spec-capture] begin capture failed: %s", cudaGetErrorString(err));
        cudaGetLastError();
        doom_check("begin capture");
        return false;
    }
    // Let cuBLASLt record into the capture instead of the ~5x-slower WMMA
    // fallback (GGUF verify GEMMs) — every shape ran eagerly in the warmup
    // use, so Lt's algo cache and static workspace are warm. See gemm.cu.
    gemm_set_lt_capture_allowed(true);
    bool forward_threw = false;
    std::string what;
    try {
        executor_->forward_logits(state, logits_out, stream);
    } catch (const std::exception& e) {
        forward_threw = true;
        what = e.what();
    } catch (...) {
        forward_threw = true;
        what = "(non-std exception)";
    }
    gemm_set_lt_capture_allowed(false);
    cudaGraph_t raw_graph = nullptr;
    err = cudaStreamEndCapture(stream, &raw_graph);
    CudaGraph graph;
    graph.reset(raw_graph);
    if (forward_threw || err != cudaSuccess || !graph) {
        IMP_LOG_WARN("[spec-capture] capture failed: %s%s%s",
                     err != cudaSuccess ? cudaGetErrorString(err) : "(forward threw)",
                     forward_threw ? " — " : "", forward_threw ? what.c_str() : "");
        cudaGetLastError();
        doom_check("capture");
        return false;
    }
    const auto t_inst0 = std::chrono::steady_clock::now();
    cudaGraphExec_t raw_exec = nullptr;
    err = cudaGraphInstantiate(&raw_exec, graph, 0);
    CudaGraphExec exec;
    exec.reset(raw_exec);
    graph.reset();
    if (err != cudaSuccess) {
        IMP_LOG_WARN("[spec-capture] instantiate failed: %s", cudaGetErrorString(err));
        cudaGetLastError();
        doom_check("instantiate");
        return false;
    }
    const auto t_launch0 = std::chrono::steady_clock::now();
    err = cudaGraphLaunch(exec, stream);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("[spec-capture] first launch failed: %s", cudaGetErrorString(err));
        cudaGetLastError();
        doom_check("first launch");
        return false;
    }
    slot.exec = std::move(exec);
    spec_capture_failures_ = 0;
    const auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    IMP_LOG_INFO(
        "[spec-capture] verify chunk graph cached (n_tokens=%d, ctx_tier=%d, rec_slot=%d): "
        "capture %.1f ms, instantiate %.1f ms, launch %.1f ms",
        state.n_tokens, state.ctx_capacity, rec_slot, ms(t_cap0, t_inst0), ms(t_inst0, t_launch0),
        ms(t_launch0, std::chrono::steady_clock::now()));
    return true;
}

// #847 graph-captured-verify feasibility probe (diagnostics.spec_capture_probe).
// Capture/instantiate failure (e.g. the cuBLASLt status-14 class) falls back
// to eager, so the verify step always completes. NOT a perf path: answers
// only whether the forward is capturable, and from which chunk, per model
// class.
void Engine::spec_capture_probe_forward_(InferenceState& state, Tensor& logits_out,
                                         cudaStream_t stream) {
    static long probes = 0, launched = 0;
    probes++;
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("[spec-capture-probe] begin capture failed: %s", cudaGetErrorString(err));
        cudaGetLastError();
        executor_->forward_logits(state, logits_out, stream);
        return;
    }
    bool forward_threw = false;
    const char* what = "";
    try {
        executor_->forward_logits(state, logits_out, stream);
    } catch (const std::exception& e) {
        forward_threw = true;
        what = e.what();
    } catch (...) {
        forward_threw = true;
        what = "(non-std exception)";
    }
    cudaGraph_t raw_graph = nullptr;
    err = cudaStreamEndCapture(stream, &raw_graph);
    CudaGraph graph;
    graph.reset(raw_graph);
    bool ran = false;
    if (!forward_threw && err == cudaSuccess && graph) {
        cudaGraphExec_t raw_exec = nullptr;
        err = cudaGraphInstantiate(&raw_exec, graph, 0);
        CudaGraphExec exec;
        exec.reset(raw_exec);
        if (err == cudaSuccess) {
            err = cudaGraphLaunch(exec, stream);
            if (err == cudaSuccess) {
                ran = true;
                launched++;
            } else {
                IMP_LOG_WARN("[spec-capture-probe] graph launch failed: %s",
                             cudaGetErrorString(err));
            }
        } else {
            IMP_LOG_WARN("[spec-capture-probe] instantiate failed: %s", cudaGetErrorString(err));
        }
    } else {
        IMP_LOG_WARN("[spec-capture-probe] capture failed: %s%s%s",
                     err != cudaSuccess ? cudaGetErrorString(err) : "(forward threw)",
                     forward_threw ? " forward exception: " : "", forward_threw ? what : "");
    }
    cudaGetLastError();
    IMP_LOG_INFO("[spec-capture-probe] chunk n_tokens=%d %s (launched %ld/%ld)", state.n_tokens,
                 ran ? "CAPTURED+LAUNCHED" : "eager fallback", launched, probes);
    if (!ran) {
        // Nothing executed during a failed capture — run the forward for real.
        executor_->forward_logits(state, logits_out, stream);
    }
}

}  // namespace imp
