#include "runtime/cuda_graph.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

// ---------------------------------------------------------------------------
// CudaGraphCapture
// ---------------------------------------------------------------------------

CudaGraphCapture::~CudaGraphCapture() {
    reset();
}

bool CudaGraphCapture::begin_capture(cudaStream_t stream) {
    if (!stream) {
        return false;
    }

    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: begin_capture failed: %s",
                      cudaGetErrorString(err));
        return false;
    }

    capture_stream_ = stream;
    return true;
}

bool CudaGraphCapture::end_capture() {
    if (!capture_stream_) {
        return false;
    }

    cudaError_t err = cudaStreamEndCapture(capture_stream_, &graph_);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: end_capture failed: %s",
                      cudaGetErrorString(err));
        capture_stream_ = nullptr;
        return false;
    }

    err = cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: instantiate failed: %s",
                      cudaGetErrorString(err));
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        capture_stream_ = nullptr;
        return false;
    }

    captured_ = true;
    capture_stream_ = nullptr;
    return true;
}

bool CudaGraphCapture::replay(cudaStream_t stream) {
    if (!captured_ || !graph_exec_) {
        return false;
    }

    cudaError_t err = cudaGraphLaunch(graph_exec_, stream);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: replay failed: %s",
                      cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CudaGraphCapture::try_update(cudaGraph_t new_graph) {
    if (!graph_exec_ || !new_graph) {
        return false;
    }

    cudaGraphExecUpdateResultInfo update_info;
    cudaError_t err = cudaGraphExecUpdate(graph_exec_, new_graph, &update_info);
    if (err != cudaSuccess || update_info.result != cudaGraphExecUpdateSuccess) {
        // Topology changed or update failed -- need full re-instantiation
        return false;
    }
    return true;
}

void CudaGraphCapture::reset() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    capture_stream_ = nullptr;
    captured_ = false;
}

// ---------------------------------------------------------------------------
// CudaGraphRunner
// ---------------------------------------------------------------------------

bool CudaGraphRunner::execute(cudaStream_t stream) {
    if (!decode_fn_) {
        IMP_LOG_ERROR("CudaGraphRunner: no decode function set");
        return false;
    }

    // Phase 1: Warmup - run without graph capture
    if (step_count_ < warmup_steps_) {
        decode_fn_(stream);
        step_count_++;
        return true;
    }

    // Phase 2: Capture - capture the graph on the first post-warmup step
    if (!graph_.is_captured()) {
        IMP_LOG_INFO("CudaGraphRunner: capturing CUDA graph (step %d)", step_count_);

        if (!graph_.begin_capture(stream)) {
            // Capture failed -- fall back to direct execution
            IMP_LOG_WARN("CudaGraphRunner: capture begin failed, running directly");
            decode_fn_(stream);
            step_count_++;
            return true;
        }

        decode_fn_(stream);

        if (!graph_.end_capture()) {
            IMP_LOG_WARN("CudaGraphRunner: capture end failed, running directly");
            graph_.reset();
            step_count_++;
            return true;
        }

        capture_count_++;
        step_count_++;
        // During graph capture the kernels are recorded but NOT executed.
        // Replay immediately so this step produces actual results.
        if (!graph_.replay(stream)) {
            IMP_LOG_WARN("CudaGraphRunner: first replay after capture failed");
            graph_.reset();
            return false;
        }
        replay_count_++;
        return true;
    }

    // Phase 3: Replay the captured graph
    if (!graph_.replay(stream)) {
        IMP_LOG_WARN("CudaGraphRunner: replay failed, invalidating graph");
        graph_.reset();
        step_count_ = 0;  // restart warmup
        // Fall back to direct execution
        decode_fn_(stream);
        return true;
    }

    replay_count_++;
    step_count_++;
    return true;
}

void CudaGraphRunner::invalidate() {
    graph_.reset();
    step_count_ = 0;
    last_batch_size_ = -1;
    last_max_blocks_ = -1;
}

} // namespace imp
