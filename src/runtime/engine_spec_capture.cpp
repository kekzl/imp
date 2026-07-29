// =============================================================================
// engine_spec_capture.cpp — graph-captured verify chunk (#847)
// =============================================================================
//
// The eager verify forward pays ~1800 kernel launches per cycle in host
// launch pacing (~8 ms on Qwen3-Coder-30B after the #854 LM-head batching).
// This module captures the chunk forward into one CUDA graph per padded
// chunk length ("bucket") and replays it every verify step.
//
// Replay across context growth is the hard part: the chunked-continuation
// attention path historically baked q_offset/ctx_len into gather grids,
// scratch sizes and FA2 kernel arguments. In capture mode (InferenceState::
// ctx_capacity > 0) those kernels read the REAL lengths from device instead
// (context_lens[0] and d_past_len), grids and the persistent K/V scratch are
// sized once for ctx_capacity, and everything that varies per step lives in
// device buffers the engine refreshes via H2D before each replay
// (d_spec_tokens_/positions_/block_table_/context_len_/past_len_).
//
// Bucketing: drafts are padded up to {9, 17, 33, k_max+1} tokens with copies
// of t0. Padded rows sit at positions AFTER every real row, so causal masking
// makes them invisible to the real rows; their KV entries are dropped by the
// same rollback that drops rejected drafts. The verify consumes argmax rows
// [0, real_chunk_len) only.
//
// Hybrids (SSMState): the recurrent state has no causal-masking escape — pad
// rows would advance the conv tail and scan state in place. The chunk
// kernels therefore read the real chunk length from device (InferenceState::
// d_chunk_len, refreshed per step like the other lengths) and stop the state
// updates at the real last row. The slab pointers (seq_base(slot)) are baked
// into the graph, so the cache key includes the recurrent slot.
//
// Safety: the first use of a bucket runs eager through the SAME capture-mode
// code path (cuBLASLt/CUTLASS algo warmup — census PR #855 showed only the
// first chunk per shape fails capture); the second use captures. Any capture
// or launch failure falls back to the eager forward, and repeated failures
// doom capture for the process (the census crash class — e.g. a forward that
// syncs on host — would otherwise retry forever). Graphs hold raw pointers
// into the executor workspace and the engine's spec staging buffers; both
// invalidate the graph cache when they move (workspace_generation, and
// free_spec_buffers_ → free_spec_graphs_).
// =============================================================================

#include "compute/gemm.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "runtime/engine.h"

#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

// Pre-size everything on the speculative verify path that would otherwise be
// allocated lazily during serving (A7 step 5.4). After the T2 slot pool these
// were the ONLY device allocations the --wrap interposer still saw — nine of
// them, each at "1 call" per process: the chunk-capture K/V scratch, the
// consolidated spec staging block, and the verify argmax/penalty scratch.
//
// None of them is per-request; they are one-shot capacity resolutions that
// happen at first use instead of at init. Doing them here turns a surprise
// mid-serving claim into a planned one, which is the difference that #1103 was
// about — the caches lost that race against the KV pool.
//
// Capacity comes from the same expressions the runtime uses, evaluated at
// their maxima; a later call with a smaller request hits the >= guards inside
// ensure_* and allocates nothing.
void Engine::prewarm_spec_scratch_() {
    const auto& scfg = runtime_config_.speculative;
    const bool spec_on = scfg.ngram || scfg.suffix || scfg.capture || mtp_spec_decode_enabled();
    if (!spec_on || !executor_)
        return;

    // Resolves spec_capture_ctx_cap_ and, through it, the chunk-capture K/V
    // scratch. Called through spec_capture_ready_ on purpose: it carries the
    // eligibility guards (host-offload, residual KV, the census probe), and
    // duplicating them here would be a second copy to keep in sync. ctx_padded
    // = 1 passes, so the call is a side-effect-only resolution.
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
    // 3/5 buckets (#964): the decode-attention verify route derives its
    // split-K count from the PADDED row count at capture time — a 2-row
    // draft padded to 9 rows baked 5 splits instead of 21 and the per-CTA
    // KV walk grew 4x (251 vs 65 us/layer at 16k). Finer buckets keep the
    // baked split geometry close to the real chunk; pad rows attend 1 token.
    // 4 = the token-recycling depth-3 chunk (#1055): exactly one batched-GEMV
    // weight sweep (MR=4); padding it into 5 pays a second sweep.
    for (int b : {3, 4, 5, 9, 17, 33}) {
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
    // SSMState hybrids are capture-eligible: the recurrent chunk kernels
    // read the real chunk length from device (InferenceState::d_chunk_len)
    // so pad rows never advance the committed state, and the graph cache is
    // keyed on the recurrent slot (the slab pointers are baked). MoE
    // host-offload syncs on the host per layer.
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

    // Hybrids bake the recurrent-slab pointers (seq_base(slot)) into the
    // graph — key it on the slot so a slot change gets its own graph.
    const int rec_slot = state.ssm_state ? state.ssm_seq_id : -1;
    auto& slot = spec_graphs_[{state.n_tokens, state.ctx_capacity, rec_slot}];
    if (slot.exec) {
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
    err = cudaGraphLaunch(exec, stream);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("[spec-capture] first launch failed: %s", cudaGetErrorString(err));
        cudaGetLastError();
        doom_check("first launch");
        return false;
    }
    slot.exec = std::move(exec);
    spec_capture_failures_ = 0;
    IMP_LOG_INFO("[spec-capture] verify chunk graph cached (n_tokens=%d, ctx_tier=%d, rec_slot=%d)",
                 state.n_tokens, state.ctx_capacity, rec_slot);
    return true;
}

// =============================================================================
// #847 graph-captured-verify feasibility probe (diagnostics.spec_capture_probe)
// =============================================================================
// Stream-captures the chunk forward, instantiates and launches the graph once,
// then destroys it. Any failure (capture-illegal call inside the forward,
// instantiate error — the cuBLASLt status-14 class) falls back to the eager
// forward, so the verify step always completes. Capture+instantiate every
// chunk is NOT a perf path; this only answers "is the verify forward
// capturable, and from which chunk on" per model class.
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
