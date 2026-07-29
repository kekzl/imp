#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <vector>
#include <cstdint>

#include "core/cuda_raii.h"
#include "memory/graph_slots.h"

namespace imp {

class GraphExecutor;
struct InferenceState;

// Low-level CUDA graph capture/replay wrapper.
// Close a stray open capture on `stream`, if any (no-op otherwise). Safety
// net for exception paths that unwound past an active capture (#874).
void abort_stream_capture(cudaStream_t stream);

// Shared graph-capture helpers:
// the process-wide capture mode (relaxed by default — global deadlocks the
// CUTLASS grouped GEMM), and the kernel→kernel PDL edge rewrite.
cudaStreamCaptureMode get_capture_mode();
int apply_pdl_edges(cudaGraph_t graph);

class CudaGraphCapture {
public:
    CudaGraphCapture() = default;
    ~CudaGraphCapture();

    bool begin_capture(cudaStream_t stream);
    bool end_capture();
    bool replay(cudaStream_t stream);
    bool is_captured() const { return captured_; }
    void reset();

    // End capture and update the existing exec in-place if possible.
    // Falls back to full re-instantiate if topology changed or no exec
    // exists. Skips cudaDeviceGraphMemTrim on fast path to avoid churn
    // during frequent re-captures (e.g. KV block table growth).
    bool end_capture_and_update();

    // Release graph_ only (keep graph_exec_ alive for in-place update).
    // Called between reusing an exec and the next capture.
    void drop_graph_keep_exec();

    // Mark captured_ as false (but keep exec / graph alive if held).
    // Used by CudaGraphRunner to force a re-capture pass while still
    // enabling cudaGraphExecUpdate against the retained exec.
    void mark_needs_recapture() { captured_ = false; }

    // Close an in-flight capture without keeping its graph (#874). Called
    // when the captured fn throws mid-capture: the stream must be taken out
    // of capture state or every later async op on it fails with
    // cudaErrorStreamCaptureInvalidated until process restart.
    void abort_capture();

private:
    CudaGraph graph_;
    CudaGraphExec graph_exec_;
    cudaStream_t capture_stream_ = nullptr;  // non-owning (engine's stream)
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
    // Next execute() will re-capture. Fully destroys exec_ and graph_.
    void invalidate();

    // Soft invalidate: keep graph_exec_ alive so the next capture can try
    // cudaGraphExecUpdate in-place. Use when topology is unchanged (e.g.
    // only kernel params / grid dims differ). Skips the warmup step on the
    // next execute() since cuBLAS algorithms are already tuned.
    void invalidate_for_update();

    // Check if graph is ready for replay
    bool is_ready() const { return graph_.is_captured(); }

    // Replay the captured graph WITHOUT the capture/warmup state machine and
    // WITHOUT needing a decode_fn. Used by the pipelined batched decode to
    // re-enqueue the forward for step N+1 after the device-side chain
    // advance — there is nothing new to capture, only a replay is valid.
    // Returns false (and resets, so the next execute() re-captures) when no
    // captured graph exists or the replay fails; the caller must then skip
    // the chained step and fall back to the per-step path.
    bool replay_only(cudaStream_t stream);
    // True when the next execute() runs graph kernels: either captured, or it
    // will capture immediately (process-warm via mark_process_warm, no eager
    // warmup steps pending, no prior capture failure). Scheduler gates that
    // pick the async loop / pipelines by pool readiness must use THIS, not
    // is_ready(): gating on is_captured() enters those paths one step later
    // on the process's FIRST request than on every later one — a numerically
    // different kernel mix for that step, and a greedy flip on near-ties
    // (the 30B-NVFP4-MoE temp=0 flipper).
    bool graph_path_available() const {
        return !capture_failed_ && (graph_.is_captured() || step_count_ >= warmup_steps_);
    }

    // Get stats
    int replay_count() const { return replay_count_; }
    int capture_count() const { return capture_count_; }

    // Configuration
    void set_warmup_steps(int n) { warmup_steps_ = n; }
    // Mark process-level lazy init (cuBLAS autotuning, workspaces) as already
    // done: the next execute() captures immediately instead of running the
    // once-per-runner eager warmup step. Engine::warmup() calls this after
    // tearing down the warmup graphs — otherwise that eager step lands in the
    // FIRST real request only, and its kernel mix differs numerically (FP
    // order) from the captured graph every later request replays: on near-tie
    // logits greedy output became request-order dependent (the documented
    // 30B-NVFP4-MoE temp=0 flipper).
    void mark_process_warm() {
        if (step_count_ < warmup_steps_)
            step_count_ = warmup_steps_;
    }

private:
    DecodeFn decode_fn_;
    CudaGraphCapture graph_;

    int step_count_ = 0;
    int warmup_steps_ = 1;  // Number of warmup steps before capture
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
        int max_steps = 0;              // max tokens to generate
        // Position/context of the loop's FIRST forward — identical semantics
        // to the eager decode step and rearm(): first_token is processed at
        // slot initial_position with initial_context_len covering it
        // (callers pass req.context_len()-1 / req.context_len()).
        int initial_context_len = 0;
        int initial_position = 0;
        int eos_id = -1;                // EOS token ID
        std::vector<int32_t> stop_ids;  // additional stop token IDs (chat template)
        float temperature = 1.0f;
        float top_p = 1.0f;
        int top_k = 0;
        int seed = -1;
        // Think budget: break loop when reasoning tokens exceed limit.
        // CPU then takes over with force_token for </think> injection.
        int think_budget_limit = 0;     // 0 = no limit
        int32_t think_start_id = -1;    // <think> token ID
        int32_t think_end_id = -1;      // </think> token ID
        bool initial_in_think = false;  // true if already inside <think> block
        // Post-</think> grace: suppress EOS/stop for this many tokens after the
        // think block closes, matching think_logic::kMinAnswerAfterThink on the
        // eager path. Guards against numerically-noisy NVFP4 quants that close an
        // empty think block in ~3 tokens then EOS to a 0-content completion.
        // 0 = no think tracking in the loop (set >0 whenever think_end_id >= 0).
        int think_grace_tokens = 0;
        // Device per-token "decodes to whitespace-only" mask (size vocab_size,
        // nullptr = treat nothing as whitespace). A whitespace/newline token
        // after </think> must not release the grace (post-#798 0-content fix).
        const uint8_t* token_is_whitespace = nullptr;
        int vocab_size = 0;
        bool ignore_eos = false;        // don't stop on EOS/stop tokens (benchmark mode)
        // Per-launch step cap read from device memory (0 = unbounded, i.e.
        // max_steps). Unlike max_steps it is NOT baked into the captured
        // graph: rearm() can change it between launches, which makes bounded
        // "burst" launches cheap (no recapture). max_steps stays the capacity
        // ceiling (ring buffer size, max_context_len).
        int step_limit = 0;
        // Penalty parameters (applied to logits before sampling each iteration)
        float repetition_penalty = 1.0f;
        float frequency_penalty = 0.0f;
        float presence_penalty = 0.0f;
        int repeat_last_n = 0;  // 0 = all generated tokens
        // Pre-existing output tokens to seed the penalty history (copied to ring buffer prefix)
        std::vector<int32_t> penalty_history;
    };

    // Build the conditional graph and all device state.
    // first_token: the first decode token (prefill output).
    // state_template: InferenceState with stable device pointers.
    //   - d_position[0] and d_context_len[0] will be set by setup.
    //   - block_tables must cover the full generation (pre-allocated).
    //   - max_context_len should be set to initial_ctx + max_steps.
    bool setup(GraphExecutor* executor, const InferenceState& state_template, int32_t first_token,
               Config config, cudaStream_t stream);

    // Launch the graph. Returns immediately.
    bool launch(cudaStream_t stream);

    // Re-seed device state for another bounded launch WITHOUT recapturing the
    // graph (the expensive part of setup). first_token is forwarded at
    // `position` with context length `context_len` (physical values, no +1
    // applied inside). step_limit bounds this launch (0 = max_steps);
    // think_limit is the REMAINING think budget for this launch (0 = no
    // budget; the device counter restarts at 0 every launch, so the caller
    // passes full_budget - tokens_already_thought). Returns false when no
    // graph is built, a launch is still in flight, or context_len would
    // exceed the captured ceiling — caller falls back to a full setup().
    bool rearm(int32_t first_token, int position, int context_len, int step_limit, bool in_think,
               int think_limit, cudaStream_t stream);

    // Context ceiling baked into the captured graph (attention workspace
    // sizing): initial_context_len + max_steps at setup time.
    int captured_context_ceiling() const {
        return config_.initial_context_len + config_.max_steps;
    }

    // Synchronize and return all generated tokens.
    std::vector<int32_t> wait_and_get_tokens(cudaStream_t stream);

    // Poll for new tokens without blocking (for streaming).
    // Appends new tokens to out_tokens. Returns count of new tokens.
    int poll_new_tokens(std::vector<int32_t>& out_tokens);

    // Non-blocking burst-completion check: returns true once the device loop
    // published its done flag (the kernel stop path is the only loop exit),
    // then drains the graph epilogue and clears the launched flag. A stream
    // error is surfaced like wait_and_get_tokens (F-A17) and still ends the
    // burst — tokens already read through poll_new_tokens stand.
    // Deliberately NOT cudaStreamQuery-based: the query reports the stream
    // idle while a conditional WHILE graph is still iterating.
    bool try_finish_burst(cudaStream_t stream);

    // Blocking fallback when the device loop stops making progress without
    // publishing done (graph error paths): sync the stream, end the burst.
    void finish_burst_blocking(cudaStream_t stream);

    // A conditional-loop burst is currently running on the device.
    bool launch_in_flight() const { return launched_; }

    // Get number of steps completed so far (non-blocking).
    int steps_completed() const;

    void cleanup();

    bool is_setup() const { return static_cast<bool>(exec_); }

private:
    CudaGraph graph_;
    CudaGraphExec exec_;
    cudaGraphConditionalHandle handle_{};

    // Device-side state (allocated by setup, freed by cleanup)
    int32_t* d_token_id_ = nullptr;  // [1] current token on device
    int* d_position_ = nullptr;      // [1] current position on device
    int* d_context_len_ = nullptr;   // [1] current context length on device
    int* d_step_counter_ = nullptr;  // [1] step counter on device
    int* d_step_limit_ = nullptr;    // [1] per-launch step cap (0 = max_steps)
    int* d_think_limit_ = nullptr;   // [1] per-launch remaining think budget (0 = off)
    int32_t* d_stop_ids_ = nullptr;  // [n_stop_ids] stop token IDs on device

    // Think budget tracking (device-side)
    int* d_think_count_ = nullptr;      // [1] reasoning token counter
    int* d_in_think_ = nullptr;         // [1] currently inside <think> block
    int* d_think_exit_step_ = nullptr;  // [1] step at which </think> last closed (-1 = never)
    int* d_content_after_think_ = nullptr;  // [1] real answer token seen since </think> (0/1)

    // Penalty token history: [prefix_len + max_steps] ring buffer for penalty computation.
    // prefix_len tokens are pre-populated from prior output; subsequent slots filled by
    // post_decode_step_kernel each iteration.
    int32_t* d_penalty_ring_ = nullptr;  // device penalty ring buffer
    int* d_penalty_count_ = nullptr;     // [1] device-side total penalty token count
    int penalty_prefix_len_ = 0;         // number of pre-populated history tokens

    // Mapped pinned memory for zero-copy host readback
    int32_t* h_ring_buffer_ = nullptr;      // host pointer to ring buffer
    int32_t* d_ring_buffer_ = nullptr;      // device pointer to same ring buffer
    int* h_step_counter_ = nullptr;         // host pointer to step counter mirror
    int* d_step_counter_mapped_ = nullptr;  // device pointer to mapped step counter
    // Burst-done flag, published by post_decode_step_kernel's stop path (the
    // only WHILE-loop exit). cudaStreamQuery reports the stream idle while a
    // conditional WHILE graph is still iterating on this platform, so the
    // host must not use it to decide teardown.
    int* h_burst_done_ = nullptr;           // host pointer to done flag
    int* d_burst_done_mapped_ = nullptr;    // device pointer to same flag
    // Dedicated scratch for forward_decode_async's per-iteration D2H token
    // copy. This used to alias h_step_counter_ — harmless when tokens were
    // only harvested after a full-burst sync, but a polling host could read
    // the transient token id as the step counter and over-read the ring.
    int32_t* h_decode_scratch_ = nullptr;

    // T2 slot backing every pointer above (A7 step 5.3). When the lease is
    // valid nothing here is individually owned and cleanup() only returns the
    // lease; when the pool declines, setup() allocates as it always did and
    // cleanup() frees. The pointers themselves are identical either way, so
    // nothing downstream can tell the difference.
    GraphSlotLease slot_;

    Config config_;
    int last_read_step_ = 0;
    bool launched_ = false;
};

// Advance the device-side decode cursor for the pipelined constrained loop:
// (*d_pos)++ and (*d_ctx)++ after a token was sampled, so the next (already
// enqueued) forward replay reads the new position without host involvement.
void launch_pipeline_advance(int* d_pos, int* d_ctx, cudaStream_t stream);

}  // namespace imp
