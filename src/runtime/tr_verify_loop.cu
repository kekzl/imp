#include "runtime/tr_verify_loop.h"
#include "runtime/cuda_graph.h"
#include "runtime/pdl.h"
#include "exec/executor.h"
#include "core/logging.h"

#include <atomic>
#include <cstring>

namespace imp {

TrVerifyLoopRunner::~TrVerifyLoopRunner() { cleanup(); }

void TrVerifyLoopRunner::cleanup() {
    if (exec_) {
        IMP_CUDA_CHECK_LOG(cudaGraphExecDestroy(exec_));
        exec_ = nullptr;
    }
    if (graph_) {
        IMP_CUDA_CHECK_LOG(cudaGraphDestroy(graph_));
        graph_ = nullptr;
    }
    if (d_stop_ids_) IMP_CUDA_CHECK_LOG(cudaFree(d_stop_ids_));
    if (d_emit_count_) IMP_CUDA_CHECK_LOG(cudaFree(d_emit_count_));
    if (d_token_limit_) IMP_CUDA_CHECK_LOG(cudaFree(d_token_limit_));
    if (h_ring_) IMP_CUDA_CHECK_LOG(cudaFreeHost(h_ring_));
    if (h_ring_count_) IMP_CUDA_CHECK_LOG(cudaFreeHost(h_ring_count_));
    if (h_exit_) IMP_CUDA_CHECK_LOG(cudaFreeHost(h_exit_));
    d_stop_ids_ = nullptr;
    d_emit_count_ = nullptr;
    d_token_limit_ = nullptr;
    h_ring_ = d_ring_ = nullptr;
    h_ring_count_ = d_ring_count_ = nullptr;
    h_exit_ = d_exit_ = nullptr;
    launched_ = false;
    last_read_ = 0;
}

bool TrVerifyLoopRunner::setup(GraphExecutor* executor, const InferenceState& body_state,
                               const TrLoopView& engine_bufs, const Config& cfg,
                               cudaStream_t stream) {
    cleanup();
    cfg_ = cfg;
    workspace_generation_ = executor->workspace_generation();

    // Runner-owned state.
    const size_t ring_bytes = static_cast<size_t>(cfg.ring_capacity) * sizeof(int32_t);
    bool ok = cudaHostAlloc(&h_ring_, ring_bytes, cudaHostAllocMapped) == cudaSuccess &&
              cudaHostGetDevicePointer(&d_ring_, h_ring_, 0) == cudaSuccess &&
              cudaHostAlloc(&h_ring_count_, sizeof(int32_t), cudaHostAllocMapped) ==
                  cudaSuccess &&
              cudaHostGetDevicePointer(&d_ring_count_, h_ring_count_, 0) == cudaSuccess &&
              cudaHostAlloc(&h_exit_, sizeof(int32_t), cudaHostAllocMapped) == cudaSuccess &&
              cudaHostGetDevicePointer(&d_exit_, h_exit_, 0) == cudaSuccess &&
              cudaMalloc(&d_emit_count_, sizeof(int32_t)) == cudaSuccess &&
              cudaMalloc(&d_token_limit_, sizeof(int32_t)) == cudaSuccess;
    if (ok && !cfg.stop_ids.empty()) {
        ok = cudaMalloc(&d_stop_ids_, cfg.stop_ids.size() * sizeof(int32_t)) == cudaSuccess &&
             cudaMemcpy(d_stop_ids_, cfg.stop_ids.data(),
                        cfg.stop_ids.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) {
        IMP_LOG_WARN("TrVerifyLoop: state allocation failed — eager verify fallback");
        cleanup();
        return false;
    }

    // Assemble the full view the captured step kernel reads.
    TrLoopView view = engine_bufs;
    view.ring = d_ring_;
    view.ring_count_mapped = d_ring_count_;
    view.emit_count = d_emit_count_;
    view.exit_reason = d_exit_;
    view.token_limit = d_token_limit_;
    TrLoopParams params = cfg.params;
    params.stop_ids = d_stop_ids_;
    params.n_stop_ids = static_cast<int>(cfg.stop_ids.size());

    // ---- Conditional WHILE graph (async-decode-loop pattern) ----
    cudaError_t err = cudaGraphCreate(&graph_, 0);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("TrVerifyLoop: cudaGraphCreate failed: %s", cudaGetErrorString(err));
        cleanup();
        return false;
    }
    err = cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("TrVerifyLoop: handle create failed: %s (needs CUDA 12.4+)",
                     cudaGetErrorString(err));
        cleanup();
        return false;
    }
    cudaGraphNodeParams cond_params{};
    cond_params.type = cudaGraphNodeTypeConditional;
    cond_params.conditional.handle = handle_;
    cond_params.conditional.type = cudaGraphCondTypeWhile;
    cond_params.conditional.size = 1;
    cudaGraphNode_t cond_node = nullptr;
    err = cudaGraphAddNode(&cond_node, graph_, nullptr, nullptr, 0, &cond_params);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("TrVerifyLoop: add conditional node failed: %s", cudaGetErrorString(err));
        cleanup();
        return false;
    }
    cudaGraph_t body_graph = cond_params.conditional.phGraph_out[0];

    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    err = cudaStreamBeginCaptureToGraph(stream, body_graph, nullptr, nullptr, 0,
                                        get_capture_mode());
    if (err != cudaSuccess) {
        IMP_LOG_WARN("TrVerifyLoop: body capture begin failed: %s", cudaGetErrorString(err));
        cleanup();
        return false;
    }
    bool captured = true;
    try {
        // (a) capture-mode bucket verify forward — reads the staged chunk
        //     buffers (tokens/positions/row-lens/past_len/chunk_len) that the
        //     step kernel rewrites each iteration.
        Tensor logits_unused;
        InferenceState state = body_state;
        executor->forward_logits(state, logits_unused, stream);
        // (b) per-row greedy argmax + top-M harvest (device, penalties none —
        //     the launch gate excludes penalized requests in v1).
        executor->greedy_argmax_all(cfg.params.chunk_pad,
                                    const_cast<int32_t*>(engine_bufs.argmax), stream,
                                    /*d_hist=*/nullptr, 0, /*d_draft=*/nullptr, 1.0f, 0.0f,
                                    0.0f, const_cast<int32_t*>(engine_bufs.topm),
                                    cfg.params.topm);
        // (c) accept + adjacency feed + next draft + next-chunk staging;
        //     sets the WHILE handle to 0 on exit.
        tr_verify_step_conditional(view, params, handle_, stream);
    } catch (const std::exception& e) {
        IMP_LOG_WARN("TrVerifyLoop: forward threw during body capture (%s) — aborting",
                     e.what());
        captured = false;
    }
    cudaGraph_t captured_body = nullptr;
    err = cudaStreamEndCapture(stream, &captured_body);
    if (!captured || err != cudaSuccess) {
        if (err != cudaSuccess)
            IMP_LOG_WARN("TrVerifyLoop: body capture end failed: %s", cudaGetErrorString(err));
        abort_stream_capture(stream);
        cleanup();
        return false;
    }
    if (pdl::is_available()) {
        const int converted = apply_pdl_edges(body_graph);
        if (converted > 0)
            IMP_LOG_INFO("TrVerifyLoop: %d body edges converted to PDL", converted);
    }
    cudaGraphExec_t raw_exec = nullptr;
    err = cudaGraphInstantiate(&raw_exec, graph_, 0);
    if (err != cudaSuccess) {
        IMP_LOG_WARN("TrVerifyLoop: instantiate failed: %s", cudaGetErrorString(err));
        cleanup();
        return false;
    }
    exec_ = raw_exec;
    IMP_LOG_INFO("TrVerifyLoop: verify loop built (chunk_pad=%d depth=%d ceiling=%d)",
                 cfg.params.chunk_pad, cfg.params.depth, cfg.params.ctx_ceiling);
    return true;
}

bool TrVerifyLoopRunner::launch(int token_limit, cudaStream_t stream) {
    if (!exec_ || launched_)
        return false;
    if (token_limit > cfg_.ring_capacity)
        token_limit = cfg_.ring_capacity;
    const int32_t zero = 0;
    *h_ring_count_ = 0;
    *h_exit_ = 0;
    last_read_ = 0;
    if (cudaMemcpyAsync(d_emit_count_, &zero, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        cudaMemcpyAsync(d_token_limit_, &token_limit, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess)
        return false;
    if (cudaGraphLaunch(exec_, stream) != cudaSuccess) {
        IMP_LOG_WARN("TrVerifyLoop: graph launch failed");
        return false;
    }
    launched_ = true;
    return true;
}

int TrVerifyLoopRunner::poll_new_tokens(std::vector<int32_t>& out) {
    if (!launched_)
        return 0;
    const int32_t count = __atomic_load_n(h_ring_count_, __ATOMIC_ACQUIRE);
    int added = 0;
    for (; last_read_ < count && last_read_ < cfg_.ring_capacity; ++last_read_) {
        out.push_back(__atomic_load_n(&h_ring_[last_read_], __ATOMIC_ACQUIRE));
        ++added;
    }
    return added;
}

int TrVerifyLoopRunner::exit_reason() const {
    if (!launched_)
        return 0;
    return __atomic_load_n(h_exit_, __ATOMIC_ACQUIRE);
}

bool TrVerifyLoopRunner::finish(cudaStream_t stream) {
    if (!launched_)
        return true;
    const bool ok = cudaStreamSynchronize(stream) == cudaSuccess;
    launched_ = false;
    return ok;
}

}  // namespace imp
