#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>
#include <cstdint>

namespace imp {

class GraphExecutor;
struct InferenceState;

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
    bool capture_failed_ = false;  // Set when capture fails; prevents infinite retry

    // Track batch config to detect changes
    int last_batch_size_ = -1;
    int last_max_blocks_ = -1;
};

// ---------------------------------------------------------------------------
// Conditional WHILE graph runner: GPU-autonomous multi-token decode loop.
//
// For single-sequence decode, captures the entire decode loop as a CUDA
// graph with a conditional WHILE node. The GPU generates N tokens without
// any host interaction. Tokens are streamed to the host via mapped pinned
// memory ring buffer.
//
// Requires CUDA 12.4+ (conditional graph nodes). Falls back gracefully
// if graph construction fails (e.g., layer offloading active).
// ---------------------------------------------------------------------------
class CudaGraphConditionalRunner {
public:
    CudaGraphConditionalRunner() = default;
    ~CudaGraphConditionalRunner();

    struct Config {
        int max_steps = 0;               // max tokens to generate
        int initial_context_len = 0;     // context length after prefill
        int initial_position = 0;        // position of last prefill token
        int eos_id = -1;                 // EOS token ID
        std::vector<int32_t> stop_ids;   // additional stop token IDs (chat template)
        float temperature = 1.0f;
        float top_p = 1.0f;
        int top_k = 0;
        int seed = -1;
    };

    // Build the conditional graph and all device state.
    // first_token: the first decode token (prefill output).
    // state_template: InferenceState with stable device pointers.
    //   - d_position[0] and d_context_len[0] will be set by setup.
    //   - block_tables must cover the full generation (pre-allocated).
    //   - max_context_len should be set to initial_ctx + max_steps.
    bool setup(GraphExecutor* executor, const InferenceState& state_template,
               int32_t first_token, Config config, cudaStream_t stream);

    // Launch the graph. Returns immediately.
    bool launch(cudaStream_t stream);

    // Synchronize and return all generated tokens.
    std::vector<int32_t> wait_and_get_tokens(cudaStream_t stream);

    // Poll for new tokens without blocking (for streaming).
    // Appends new tokens to out_tokens. Returns count of new tokens.
    int poll_new_tokens(std::vector<int32_t>& out_tokens);

    // Get number of steps completed so far (non-blocking).
    int steps_completed() const;

    void cleanup();

    bool is_setup() const { return exec_ != nullptr; }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaGraphConditionalHandle handle_{};

    // Device-side state (allocated by setup, freed by cleanup)
    int32_t* d_token_id_ = nullptr;       // [1] current token on device
    int* d_position_ = nullptr;            // [1] current position on device
    int* d_context_len_ = nullptr;         // [1] current context length on device
    int* d_step_counter_ = nullptr;        // [1] step counter on device
    int32_t* d_stop_ids_ = nullptr;        // [n_stop_ids] stop token IDs on device

    // Mapped pinned memory for zero-copy host readback
    int32_t* h_ring_buffer_ = nullptr;     // host pointer to ring buffer
    int32_t* d_ring_buffer_ = nullptr;     // device pointer to same ring buffer
    int* h_step_counter_ = nullptr;        // host pointer to step counter mirror
    int* d_step_counter_mapped_ = nullptr; // device pointer to mapped step counter

    Config config_;
    int last_read_step_ = 0;
    bool launched_ = false;
};

} // namespace imp
