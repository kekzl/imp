#pragma once

#include <cuda_runtime.h>
#include <functional>

namespace imp {

// Low-level CUDA graph capture/replay wrapper.
class CudaGraphCapture {
public:
    CudaGraphCapture() = default;
    ~CudaGraphCapture();

    bool begin_capture(cudaStream_t stream);
    bool end_capture();
    bool replay(cudaStream_t stream);
    bool is_captured() const { return captured_; }
    void reset();

    // Try to update an existing graph exec with a new graph.
    // Returns true if the update succeeded (topology unchanged).
    bool try_update(cudaGraph_t new_graph);

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaStream_t capture_stream_ = nullptr;
    bool captured_ = false;
};

// High-level decode iteration graph runner.
// Manages capture/replay lifecycle for decode forward passes.
//
// Usage:
//   runner.set_decode_fn(decode_function);
//   runner.execute(stream);  // First call: runs normally (warmup)
//                             // Second call: captures graph
//                             // Subsequent calls: replays graph
//
// Call invalidate() when batch configuration changes.
class CudaGraphRunner {
public:
    CudaGraphRunner() = default;
    ~CudaGraphRunner() = default;

    // Set the decode function to capture. This function will be called
    // on the given stream and its kernel launches will be captured.
    using DecodeFn = std::function<void(cudaStream_t)>;
    void set_decode_fn(DecodeFn fn) { decode_fn_ = std::move(fn); }

    // Execute: runs the decode function, managing capture/replay.
    bool execute(cudaStream_t stream);

    // Mark the current graph as invalid (e.g., batch size changed).
    // Next execute() will re-capture.
    void invalidate();

    // Check if graph is ready for replay
    bool is_ready() const { return graph_.is_captured(); }

    // Get stats
    int replay_count() const { return replay_count_; }
    int capture_count() const { return capture_count_; }

    // Configuration
    void set_warmup_steps(int n) { warmup_steps_ = n; }

private:
    DecodeFn decode_fn_;
    CudaGraphCapture graph_;

    int step_count_ = 0;
    int warmup_steps_ = 1;    // Number of warmup steps before capture
    int replay_count_ = 0;
    int capture_count_ = 0;

    // Track batch config to detect changes
    int last_batch_size_ = -1;
    int last_max_blocks_ = -1;
};

} // namespace imp
