#include "runtime/cuda_graph.h"
#include "runtime/graph_diag.h"
#include "runtime/pdl.h"
#include "runtime/process_diag.h"
#include "exec/executor.h"
#include "compute/sampling.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace imp {

// runtime.graph_capture_mode = "global" | "relaxed" (default) | "thread_local"
// (legacy env: IMP_GRAPH_CAPTURE_MODE). Selects the cudaStreamCaptureMode
// used by CudaGraphCapture::begin_capture and the ConditionalRunner body-graph
// capture. Probed at first call and cached.
//
// Why "relaxed" became the default (M3-probe, 2026-05-16): CUTLASS 3.x grouped
// GEMM was observed to silently HANG under cudaStreamCaptureModeGlobal during
// NVFP4 MoE prefill capture (Blocker B in prefill_graph_blockers_2026_05_14
// memo). Relaxed drops the cross-thread synchronization constraint that the
// CUTLASS collective scheduler is suspected to deadlock on, and is a strict
// superset of capturable behaviors — the decode fast path that previously
// worked under "global" continues to work under "relaxed".
static cudaStreamCaptureMode get_capture_mode() {
    static cudaStreamCaptureMode cached = []() {
        const std::string& mode = process_diag_graph_capture_mode();
        const bool prefill_graph = process_diag_prefill_graph_enabled();
        if (mode == "global") {
            // Warn loudly when the user has both opted into prefill_graph AND
            // the known-deadlocking strict capture mode — most common cause of
            // a silent hang at first NVFP4 MoE prefill replay.
            if (prefill_graph) {
                IMP_LOG_WARN(
                    "CudaGraphCapture: runtime.graph_capture_mode=\"global\" + "
                    "runtime.prefill_graph=true is the known-hangy combination "
                    "(prefill_graph_blockers_2026_05_14 memo Blocker B). "
                    "Recommended: keep graph_capture_mode=\"relaxed\" (current "
                    "default) when prefill_graph is enabled.");
            }
            IMP_LOG_INFO("CudaGraphCapture: using cudaStreamCaptureModeGlobal");
            return cudaStreamCaptureModeGlobal;
        }
        if (mode == "thread_local") {
            IMP_LOG_INFO("CudaGraphCapture: using cudaStreamCaptureModeThreadLocal (runtime.graph_capture_mode=thread_local)");
            return cudaStreamCaptureModeThreadLocal;
        }
        // Default + "relaxed" branch:
        if (mode != "relaxed") {
            IMP_LOG_WARN(
                "CudaGraphCapture: unknown runtime.graph_capture_mode=\"%s\", "
                "falling back to \"relaxed\".",
                mode.c_str());
        }
        IMP_LOG_INFO("CudaGraphCapture: using cudaStreamCaptureModeRelaxed (default for prefill-safe capture)");
        return cudaStreamCaptureModeRelaxed;
    }();
    return cached;
}

// ---------------------------------------------------------------------------
// apply_pdl_edges — convert kernel→kernel edges to PDL edges in a graph
// ---------------------------------------------------------------------------
static int apply_pdl_edges(cudaGraph_t graph) {
    if (!graph)
        return 0;

    // 1. Enumerate all nodes
    size_t num_nodes = 0;
    cudaError_t err = cudaGraphGetNodes(graph, nullptr, &num_nodes);
    if (err != cudaSuccess || num_nodes == 0)
        return 0;

    std::vector<cudaGraphNode_t> nodes(num_nodes);
    err = cudaGraphGetNodes(graph, nodes.data(), &num_nodes);
    if (err != cudaSuccess)
        return 0;

    // 2. Build set of kernel nodes (use linear scan — node counts are small)
    std::vector<cudaGraphNode_t> kernel_nodes;
    kernel_nodes.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; i++) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) == cudaSuccess && type == cudaGraphNodeTypeKernel) {
            kernel_nodes.push_back(nodes[i]);
        }
    }
    if (kernel_nodes.size() < 2)
        return 0;

    // 3. Enumerate all edges with edge data
    size_t num_edges = 0;
    err = cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &num_edges);
    if (err != cudaSuccess || num_edges == 0)
        return 0;

    std::vector<cudaGraphNode_t> from(num_edges), to(num_edges);
    std::vector<cudaGraphEdgeData> edge_data(num_edges);
    err = cudaGraphGetEdges(graph, from.data(), to.data(), edge_data.data(), &num_edges);
    if (err != cudaSuccess)
        return 0;

    // Helper: check if a node is a kernel node
    auto is_kernel = [&](cudaGraphNode_t n) -> bool {
        for (const auto& kn : kernel_nodes) {
            if (kn == n)
                return true;
        }
        return false;
    };

    // 4. Replace default kernel→kernel edges with PDL edges, but ONLY when the
    //    source kernel has ProgrammaticStreamSerialization enabled.  Non-PDL
    //    kernels use the default port (programmatic == default for them), so
    //    converting their edges just adds driver bookkeeping overhead.
    cudaGraphEdgeData pdl_edge{};
    pdl_edge.from_port = cudaGraphKernelNodePortProgrammatic;
    pdl_edge.to_port = 0;
    pdl_edge.type = cudaGraphDependencyTypeProgrammatic;

    int converted = 0;
    int skipped_non_pdl = 0;
    for (size_t i = 0; i < num_edges; i++) {
        if (edge_data[i].type != cudaGraphDependencyTypeDefault)
            continue;
        if (!is_kernel(from[i]) || !is_kernel(to[i]))
            continue;

        // Check if the source kernel has PDL enabled.
        // NOTE: cudaGraphKernelNodeGetParams returns kparams.func = nullptr
        // for kernel nodes added via the driver-API form (CUkernel handle
        // rather than a host __global__ symbol pointer) AND sets the global
        // CUDA last-error to "invalid device function". That error is benign
        // in our flow — we just want to look up the host pointer in the PDL
        // registry, so a null func means "not in registry, skip it". We
        // clear the error immediately so it doesn't surface as a stale
        // error two function frames up at the start of the next forward
        // pass (which used to log every request as
        // "Cleared stale error before forward: invalid device function").
        cudaKernelNodeParams kparams{};
        cudaError_t kerr = cudaGraphKernelNodeGetParams(from[i], &kparams);
        if (kerr != cudaSuccess || !kparams.func || !pdl::is_enabled(kparams.func)) {
            (void)cudaGetLastError();  // swallow the per-edge "invalid device function"
            skipped_non_pdl++;
            continue;
        }

        // Remove old default edge
        err = cudaGraphRemoveDependencies(graph, &from[i], &to[i], &edge_data[i], 1);
        if (err != cudaSuccess)
            continue;

        // Add PDL edge
        err = cudaGraphAddDependencies(graph, &from[i], &to[i], &pdl_edge, 1);
        if (err != cudaSuccess) {
            // Rollback: re-add the default edge. If rollback fails the graph
            // has lost an edge entirely — surface it so the eventual
            // instantiate/launch failure can be traced back.
            cudaError_t rb_err = cudaGraphAddDependencies(graph, &from[i], &to[i], &edge_data[i], 1);
            if (rb_err != cudaSuccess) {
                IMP_LOG_ERROR(
                    "apply_pdl_edges: rollback failed after PDL add "
                    "error (%s); dropped edge — graph is corrupted: %s",
                    cudaGetErrorString(err), cudaGetErrorString(rb_err));
            }
            continue;
        }
        converted++;
    }

    if (skipped_non_pdl > 0)
        IMP_LOG_DEBUG("apply_pdl_edges: skipped %d edges (source kernel not PDL-enabled)", skipped_non_pdl);

    return converted;
}

// ---------------------------------------------------------------------------
// CudaGraphCapture
// ---------------------------------------------------------------------------

CudaGraphCapture::~CudaGraphCapture() { reset(); }

bool CudaGraphCapture::begin_capture(cudaStream_t stream) {
    if (!stream) {
        return false;
    }

    cudaError_t err = cudaStreamBeginCapture(stream, get_capture_mode());
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: begin_capture failed: %s", cudaGetErrorString(err));
        return false;
    }

    capture_stream_ = stream;
    graph_diag::g_phase = graph_diag::Phase::CAPTURE;
    return true;
}

bool CudaGraphCapture::end_capture() {
    if (!capture_stream_) {
        return false;
    }

    cudaError_t err = cudaStreamEndCapture(capture_stream_, &graph_);
    graph_diag::g_phase = graph_diag::Phase::NORMAL;
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: end_capture failed: %s", cudaGetErrorString(err));
        capture_stream_ = nullptr;
        return false;
    }

    // Convert kernel→kernel edges to PDL edges for tail/head overlap
    if (pdl::is_available()) {
        int converted = apply_pdl_edges(graph_);
        if (converted > 0)
            IMP_LOG_DEBUG("CudaGraphCapture: %d edges converted to PDL", converted);
    }

    graph_diag::log_kernel_nodes(graph_, "capture.plain");
    graph_diag::dump_graph(graph_, "capture.plain");

    err = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: instantiate failed: %s", cudaGetErrorString(err));
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

    graph_diag::PhaseScope scope(graph_diag::Phase::REPLAY);
    cudaError_t err = cudaGraphLaunch(graph_exec_, stream);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: replay failed: %s", cudaGetErrorString(err));
        return false;
    }
    graph_diag::check_post_launch(stream, "replay");
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

void CudaGraphCapture::drop_graph_keep_exec() {
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    capture_stream_ = nullptr;
    // graph_exec_ stays valid; captured_ stays true since the exec is usable.
}

bool CudaGraphCapture::end_capture_and_update() {
    if (!capture_stream_) {
        return false;
    }

    cudaGraph_t new_graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(capture_stream_, &new_graph);
    graph_diag::g_phase = graph_diag::Phase::NORMAL;
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: end_capture failed: %s", cudaGetErrorString(err));
        capture_stream_ = nullptr;
        return false;
    }

    // Log node composition (gated by diagnostics.graph_diag in imp.conf). Symmetric
    // with the end_capture() / ConditionalRunner paths so prefill/decode graphs are
    // comparable. Useful for diagnosing memcpy-density across models.
    graph_diag::log_kernel_nodes(new_graph, "capture.update");

    // PDL edge conversion before update/instantiate so both paths have it.
    if (pdl::is_available()) {
        int converted = apply_pdl_edges(new_graph);
        if (converted > 0)
            IMP_LOG_DEBUG("CudaGraphCapture: %d edges converted to PDL", converted);
    }

    // Fast path: try to update existing exec in place. Avoids destroying and
    // re-allocating the graph mem pool.
    if (graph_exec_ != nullptr) {
        cudaGraphExecUpdateResultInfo info;
        cudaError_t ue = cudaGraphExecUpdate(graph_exec_, new_graph, &info);
        if (ue == cudaSuccess && info.result == cudaGraphExecUpdateSuccess) {
            if (graph_)
                cudaGraphDestroy(graph_);
            graph_ = new_graph;
            captured_ = true;
            capture_stream_ = nullptr;
            return true;
        }
        // Update failed (topology changed) — fall through to reinstantiate.
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }

    if (graph_)
        cudaGraphDestroy(graph_);
    graph_ = new_graph;
    err = cudaGraphInstantiate(&graph_exec_, graph_, 0);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("CudaGraphCapture: instantiate failed: %s", cudaGetErrorString(err));
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        capture_stream_ = nullptr;
        return false;
    }

    captured_ = true;
    capture_stream_ = nullptr;
    return true;
}

void CudaGraphCapture::reset() {
    bool had_exec = (graph_exec_ != nullptr);
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
    // Release the per-device graph memory pool. Without this, instantiated
    // graphs (esp. for 128-expert MoE models) hold reserved VRAM until process
    // exit, which compounds across re-captures (config changes, batch size
    // changes). Trim is a no-op when the pool is already empty.
    if (had_exec) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGraphMemTrim(dev);
    }
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
        // If a previous capture attempt failed, skip further attempts
        if (capture_failed_) {
            decode_fn_(stream);
            step_count_++;
            return true;
        }

        IMP_LOG_INFO("CudaGraphRunner: capturing CUDA graph (step %d)", step_count_);

        if (!graph_.begin_capture(stream)) {
            // Capture failed -- fall back to direct execution permanently
            IMP_LOG_ERROR(
                "CudaGraphRunner: capture failed — falling back to per-step decode "
                "(up to 15x slower). Check for unsupported CUDA operations in the "
                "forward pass.");
            capture_failed_ = true;
            decode_fn_(stream);
            step_count_++;
            return true;
        }

        decode_fn_(stream);

        // Prefer the update path so re-captures (after invalidate_for_update)
        // reuse the existing exec via cudaGraphExecUpdate. On first capture
        // (no prior exec) this degrades cleanly to instantiate.
        if (!graph_.end_capture_and_update()) {
            IMP_LOG_ERROR(
                "CudaGraphRunner: capture failed — falling back to per-step decode "
                "(up to 15x slower). Check for unsupported CUDA operations in the "
                "forward pass.");
            graph_.reset();
            capture_failed_ = true;
            // end_capture consumed the stream work; must re-execute for actual results
            decode_fn_(stream);
            step_count_++;
            return true;
        }

        capture_count_++;
        step_count_++;
        // During graph capture the kernels are recorded but NOT executed.
        // Replay immediately so this step produces actual results.
        if (!graph_.replay(stream)) {
            IMP_LOG_ERROR(
                "CudaGraphRunner: first replay after capture failed — falling back "
                "to per-step decode (up to 15x slower).");
            graph_.reset();
            return false;
        }
        replay_count_++;
        return true;
    }

    // Phase 3: Replay the captured graph
    if (!graph_.replay(stream)) {
        IMP_LOG_ERROR(
            "CudaGraphRunner: replay failed — invalidating graph and falling back "
            "to per-step decode (up to 15x slower). Will attempt re-capture.");
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
    capture_failed_ = false;
    last_batch_size_ = -1;
    last_max_blocks_ = -1;
}

void CudaGraphRunner::invalidate_for_update() {
    // Keep graph_exec_ alive so the next capture can run cudaGraphExecUpdate
    // in-place. Skip warmup on the next execute() by leaving step_count_ at
    // warmup_steps_ — cuBLAS autotuning already ran during the prior capture.
    graph_.drop_graph_keep_exec();
    graph_.mark_needs_recapture();
    step_count_ = warmup_steps_;
    capture_failed_ = false;
}

// ---------------------------------------------------------------------------
// CudaGraphConditionalRunner — GPU-autonomous decode loop
// ---------------------------------------------------------------------------

// Device-side enable flag for post_decode_step_kernel tracing.
// Host sets to 1 via cudaMemcpyToSymbol when IMP_GRAPH_DIAG is enabled.
__constant__ int d_graph_diag_enabled = 0;

// Device kernel: post-decode-step bookkeeping.
// Copies sampled token to ring buffer, increments counters, checks stop
// conditions, and breaks the WHILE loop via cudaGraphSetConditional.
__global__ void post_decode_step_kernel(
    const int32_t* __restrict__ d_token_id,  // [1] sampled token
    int32_t* __restrict__ d_ring_buffer,     // [max_steps] output (mapped pinned)
    int* __restrict__ d_ring_step_counter,   // [1] mapped step counter (host-visible)
    int* __restrict__ d_position,            // [1] current position
    int* __restrict__ d_context_len,         // [1] current context length
    int* __restrict__ d_step_counter,        // [1] device-side step counter
    int max_steps, int eos_id, const int32_t* __restrict__ d_stop_ids, int n_stop_ids,
    // Think budget: remaining-budget limit read from device memory so
    // rearm() can shrink it across bounded burst launches (0 = disabled).
    const int* __restrict__ d_think_limit, int32_t think_start_id, int32_t think_end_id,
    int think_grace_tokens,
    int* __restrict__ d_think_count,      // [1] reasoning token counter
    int* __restrict__ d_in_think,         // [1] think block flag
    int* __restrict__ d_think_exit_step,  // [1] step </think> last closed (-1 = never)
    int* __restrict__ d_content_after_think,  // [1] real answer token seen since </think> (0/1)
    const uint8_t* __restrict__ d_token_is_whitespace,  // [vocab] 1 = decodes to whitespace-only
    int vocab_size,
    int ignore_eos,                   // 1 = don't stop on EOS/stop tokens
    // Penalty ring buffer: d_penalty_ring[prefix_len + step] = token
    int32_t* __restrict__ d_penalty_ring,  // may be null if no penalties
    int penalty_prefix_len,
    int* __restrict__ d_penalty_count,  // [1] total penalty token count
    const int* __restrict__ d_step_limit,  // [1] per-launch cap (0 = max_steps)
    int* __restrict__ d_burst_done,     // [1] mapped done flag (host-visible)
    cudaGraphConditionalHandle handle) {
    int step = *d_step_counter;
    int32_t token = *d_token_id;
    const int think_budget_limit = d_think_limit ? *d_think_limit : 0;

    // Per-launch step cap: read from device memory so rearm() can bound a
    // burst without recapturing the graph. max_steps stays the hard capacity
    // ceiling (ring buffer size).
    int eff_max = max_steps;
    if (d_step_limit) {
        const int lim = *d_step_limit;
        if (lim > 0 && lim < eff_max)
            eff_max = lim;
    }

    // Track think state on device. Active whenever a </think> id is known
    // (think_end_id >= 0), independent of the budget — it drives EOS/stop
    // suppression inside the block and the post-</think> grace window. The
    // reasoning-token counter only matters for the budget (budget_limit > 0).
    const bool track_think = (think_end_id >= 0);
    if (track_think) {
        if (token == think_start_id) {
            *d_in_think = 1;
        } else if (token == think_end_id) {
            *d_in_think = 0;
            *d_think_exit_step = step;  // open the post-think grace window
            if (d_content_after_think)
                *d_content_after_think = 0;  // fresh grace window: no answer yet
        } else if (*d_in_think && think_budget_limit > 0) {
            (*d_think_count)++;
        }
    }

    // Post-</think> content detection: the first real answer token (non-stop,
    // non-marker) emitted after the exit step releases the grace, so a complete
    // short answer stops on its own EOS instead of being padded/repeated. Must
    // run before the grace check below. step > exit_step skips the </think>
    // token itself (set this same invocation).
    if (track_think && d_content_after_think && d_think_exit_step && *d_think_exit_step >= 0 &&
        step > *d_think_exit_step && token != think_start_id && token != think_end_id) {
        bool tok_is_stop = (token == eos_id);
        for (int i = 0; i < n_stop_ids; i++) {
            if (token == d_stop_ids[i])
                tok_is_stop = true;
        }
        // Whitespace/newline tokens after </think> are not real answer content
        // and must not release the grace (post-#798 0-content fix).
        bool tok_is_ws = (d_token_is_whitespace && token >= 0 && token < vocab_size &&
                          d_token_is_whitespace[token]);
        if (!tok_is_stop && !tok_is_ws)
            *d_content_after_think = 1;
    }

    // Write token to ring buffer (visible to host via mapped memory)
    d_ring_buffer[step] = token;

    // Write token to penalty ring buffer (for next iteration's penalty application)
    if (d_penalty_ring) {
        d_penalty_ring[penalty_prefix_len + step] = token;
        *d_penalty_count = penalty_prefix_len + step + 1;
    }

    // Increment counters
    int new_pos = *d_position + 1;
    int new_ctx = *d_context_len + 1;
    *d_position = new_pos;
    *d_context_len = new_ctx;
    *d_step_counter = step + 1;

    // Flush the ring buffer write to system-scope memory before publishing the
    // host-visible counter. Without this, on WSL2 (and in principle any system
    // where mapped-pinned writes are not strongly ordered w.r.t. the host) the
    // host can observe the incremented counter while still reading the stale
    // previous token from the ring — producing corrupted streamed output.
    __threadfence_system();

    // Update mapped step counter (host-visible, for polling)
    *d_ring_step_counter = step + 1;

    // Check stop conditions.
    // Suppress stop tokens (EOS, <|im_end|>) while inside <think> block — the
    // model may emit them during reasoning, stopping prematurely — and for a
    // grace window after </think> closes (numerically-noisy NVFP4 quants can
    // close an empty block in ~3 tokens then EOS to a 0-content completion).
    bool in_think = (track_think && d_in_think && *d_in_think);
    // Grace blocks the stop only while no real answer content has appeared yet
    // (content_after_think == 0); the moment a genuine answer token is emitted,
    // the model's own stop is honoured. think_grace_tokens stays a hard cap for
    // the no-content case so generation is still bounded.
    bool grace = (think_grace_tokens > 0 && d_think_exit_step && *d_think_exit_step >= 0 &&
                  (step - *d_think_exit_step) < think_grace_tokens &&
                  (!d_content_after_think || *d_content_after_think == 0));
    bool should_stop = (step + 1 >= eff_max);
    if (!in_think && !grace && !ignore_eos) {
        if (token == eos_id)
            should_stop = true;
        for (int i = 0; i < n_stop_ids; i++) {
            if (token == d_stop_ids[i])
                should_stop = true;
        }
    }

    // Think budget: break loop to return to CPU for force_token injection.
    // Gated on budget_limit > 0 — without it, limit 0 would make every in-think
    // step satisfy (count >= 0) and break the loop on the first reasoning token.
    if (think_budget_limit > 0 && in_think && *d_think_count >= think_budget_limit) {
        should_stop = true;
    }

    if (should_stop) {
        if (d_graph_diag_enabled) {
            int stop_reason = 0;
            if (step + 1 >= eff_max)
                stop_reason = 1;  // max_steps / step_limit
            else if (!in_think && !grace && !ignore_eos) {
                if (token == eos_id)
                    stop_reason = 2;  // eos
                for (int i = 0; i < n_stop_ids; i++) {
                    if (token == d_stop_ids[i])
                        stop_reason = 3 + i;  // stop_ids[i]
                }
            }
            if (in_think && think_budget_limit > 0 && d_think_count && *d_think_count >= think_budget_limit)
                stop_reason = 100;
            printf(
                "[graph_diag:cond_stop] step=%d token=%d in_think=%d "
                "eos_id=%d n_stop_ids=%d reason=%d\n",
                step, token, in_think ? 1 : 0, eos_id, n_stop_ids, stop_reason);
        }
        cudaGraphSetConditional(handle, 0);  // break WHILE loop
        // Publish burst completion to the host. This is the ONLY loop exit,
        // so the polling host (try_finish_burst) keys teardown off this flag
        // — cudaStreamQuery reports the stream idle while the conditional
        // WHILE graph is still iterating on this platform, so it must not be
        // trusted. The fence orders the flag after this step's ring/counter
        // writes above.
        __threadfence_system();
        *d_burst_done = 1;
    }
}

CudaGraphConditionalRunner::~CudaGraphConditionalRunner() { cleanup(); }

bool CudaGraphConditionalRunner::setup(GraphExecutor* executor, const InferenceState& state_template,
                                       int32_t first_token, Config config, cudaStream_t stream) {
    cleanup();  // release any prior state
    config_ = std::move(config);

    // Propagate IMP_GRAPH_DIAG to device-side tracing in post_decode_step_kernel.
    // Done once per setup(); cheap enough to not bother caching. Symbol write
    // is skipped in the normal (non-diag) path to avoid perturbing CUDA error
    // state in pre-launch phases.
    if (graph_diag::enabled()) {
        int v = 1;
        cudaMemcpyToSymbol(d_graph_diag_enabled, &v, sizeof(int));
    }

    cudaError_t err;

    // ---- Allocate device state ----
    // Must be SAMPLE_SCRATCH_BYTES — both samplers write multi-block partial
    // reduction arrays after the result token (greedy argmax: ARGMAX scratch;
    // top-k/top-p: the larger SAMPLE scratch). SAMPLE_SCRATCH_BYTES covers both.
    err = cudaMalloc(&d_token_id_, SAMPLE_SCRATCH_BYTES);
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc(&d_position_, sizeof(int));
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc(&d_context_len_, sizeof(int));
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc(&d_step_counter_, sizeof(int));
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc(&d_step_limit_, sizeof(int));
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc(&d_think_limit_, sizeof(int));
    if (err != cudaSuccess)
        goto fail;

    // Stop token IDs
    if (!config_.stop_ids.empty()) {
        err = cudaMalloc(&d_stop_ids_, config_.stop_ids.size() * sizeof(int32_t));
        if (err != cudaSuccess)
            goto fail;
        err = cudaMemcpyAsync(d_stop_ids_, config_.stop_ids.data(), config_.stop_ids.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: stop_ids upload failed: %s", cudaGetErrorString(err));
            goto fail;
        }
    }

    // Think-state counters (device-side). Allocated whenever think tracking is
    // active (a </think> id is known), not only for budgets — the in_think flag
    // and exit-step drive EOS/stop suppression + the post-</think> grace window.
    if (config_.think_end_id >= 0) {
        err = cudaMalloc(&d_think_count_, sizeof(int));
        if (err != cudaSuccess)
            goto fail;
        err = cudaMalloc(&d_in_think_, sizeof(int));
        if (err != cudaSuccess)
            goto fail;
        err = cudaMalloc(&d_think_exit_step_, sizeof(int));
        if (err != cudaSuccess)
            goto fail;
        err = cudaMalloc(&d_content_after_think_, sizeof(int));
        if (err != cudaSuccess)
            goto fail;
        int zero = 0;
        int neg_one = -1;
        int init_think = config_.initial_in_think ? 1 : 0;
        err = cudaMemcpyAsync(d_think_count_, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: think_count init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_in_think_, &init_think, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: in_think init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_think_exit_step_, &neg_one, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: think_exit_step init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_content_after_think_, &zero, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: content_after_think init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
    }

    // ---- Allocate penalty ring buffer (prefix history + generated tokens) ----
    {
        bool has_penalties = (config_.repetition_penalty != 1.0f || config_.frequency_penalty != 0.0f ||
                              config_.presence_penalty != 0.0f);
        if (has_penalties) {
            penalty_prefix_len_ = static_cast<int>(config_.penalty_history.size());
            int total_penalty_slots = penalty_prefix_len_ + config_.max_steps;
            err = cudaMalloc(&d_penalty_ring_, total_penalty_slots * sizeof(int32_t));
            if (err != cudaSuccess)
                goto fail;
            err = cudaMalloc(&d_penalty_count_, sizeof(int));
            if (err != cudaSuccess)
                goto fail;
            // Copy prefix history to the beginning of the penalty ring
            if (penalty_prefix_len_ > 0) {
                err = cudaMemcpyAsync(d_penalty_ring_, config_.penalty_history.data(),
                                      penalty_prefix_len_ * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess) {
                    IMP_LOG_ERROR(
                        "ConditionalRunner: penalty prefix upload "
                        "failed: %s",
                        cudaGetErrorString(err));
                    goto fail;
                }
            }
            // Initialize penalty count to prefix length (before any generation)
            err = cudaMemcpyAsync(d_penalty_count_, &penalty_prefix_len_, sizeof(int), cudaMemcpyHostToDevice,
                                  stream);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("ConditionalRunner: penalty_count init failed: %s", cudaGetErrorString(err));
                goto fail;
            }
        }
    }

    // ---- Allocate mapped pinned memory for ring buffer ----
    {
        err = cudaHostAlloc(&h_ring_buffer_, config_.max_steps * sizeof(int32_t), cudaHostAllocMapped);
        if (err != cudaSuccess)
            goto fail;
        err = cudaHostGetDevicePointer(&d_ring_buffer_, h_ring_buffer_, 0);
        if (err != cudaSuccess)
            goto fail;

        err = cudaHostAlloc(&h_step_counter_, sizeof(int), cudaHostAllocMapped);
        if (err != cudaSuccess)
            goto fail;
        err = cudaHostGetDevicePointer(&d_step_counter_mapped_, h_step_counter_, 0);
        if (err != cudaSuccess)
            goto fail;

        err = cudaHostAlloc(&h_burst_done_, sizeof(int), cudaHostAllocMapped);
        if (err != cudaSuccess)
            goto fail;
        err = cudaHostGetDevicePointer(&d_burst_done_mapped_, h_burst_done_, 0);
        if (err != cudaSuccess)
            goto fail;

        err = cudaHostAlloc(&h_decode_scratch_, sizeof(int32_t), cudaHostAllocMapped);
        if (err != cudaSuccess)
            goto fail;
    }

    // ---- Initialize device state ----
    {
        // Same semantics as the eager decode step and rearm(): the loop's
        // first forward processes first_token at slot initial_position with
        // the context window covering it (#683 — the former +1 here shifted
        // every fresh-captured loop's KV writes and RoPE one slot too high;
        // the next correct-positioned writer, e.g. the spec-ngram verify
        // chunk, then overwrote the burst's last KV entry).
        int init_pos = config_.initial_position;
        int init_ctx = config_.initial_context_len;
        int init_step = 0;
        *h_step_counter_ = 0;
        *h_burst_done_ = 0;
        memset(h_ring_buffer_, 0, config_.max_steps * sizeof(int32_t));

        err = cudaMemcpyAsync(d_token_id_, &first_token, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: first_token upload failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_position_, &init_pos, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: position init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_context_len_, &init_ctx, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: context_len init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_step_counter_, &init_step, sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: step_counter init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_step_limit_, &config_.step_limit, sizeof(int),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: step_limit init failed: %s", cudaGetErrorString(err));
            goto fail;
        }
        err = cudaMemcpyAsync(d_think_limit_, &config_.think_budget_limit, sizeof(int),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: think_limit init failed: %s",
                          cudaGetErrorString(err));
            goto fail;
        }
    }

    // ---- Build InferenceState for graph body (uses our device pointers) ----
    // The state_template provides block_tables, kv_cache, etc.
    // We override positions, context_lens, token_ids with our device ptrs.
    {
        InferenceState body_state = state_template;
        body_state.token_ids = d_token_id_;
        body_state.positions = d_position_;
        body_state.context_lens = d_context_len_;
        body_state.n_tokens = 1;
        body_state.n_sequences = 1;
        body_state.is_prefill = false;
        body_state.temperature = config_.temperature;
        body_state.top_p = config_.top_p;
        body_state.top_k = config_.top_k;
        body_state.seed = config_.seed;
        // max_context_len drives kernel-path selection in paged_attention_decode.
        // The split-K pipeline kernel is broken when captured into a conditional
        // WHILE body (bisect 2026-04-16), but the dispatch now detects stream
        // capture and falls back to the non-pipeline split-K — safe to use the
        // real value here.
        body_state.max_context_len = config_.initial_context_len + config_.max_steps;

        // Penalty parameters for device-side application in forward_decode_async
        if (d_penalty_ring_) {
            body_state.penalty_tokens = d_penalty_ring_;
            body_state.d_n_penalty_tokens = d_penalty_count_;
            body_state.repetition_penalty = config_.repetition_penalty;
            body_state.frequency_penalty = config_.frequency_penalty;
            body_state.presence_penalty = config_.presence_penalty;
            body_state.repeat_last_n = config_.repeat_last_n;
        }

        // ---- Construct CUDA graph with conditional WHILE node ----
        // 1. Create top-level graph
        err = cudaGraphCreate(&graph_, 0);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR(
                "ConditionalRunner: cudaGraphCreate failed: %s — falling back to "
                "per-step decode (up to 15x slower). Check for unsupported CUDA "
                "operations in the forward pass.",
                cudaGetErrorString(err));
            goto fail;
        }

        // 2. Create conditional handle (default value = 1 = "continue looping")
        err = cudaGraphConditionalHandleCreate(&handle_, graph_, 1, cudaGraphCondAssignDefault);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR(
                "ConditionalRunner: handle create failed: %s — falling back to "
                "per-step decode (up to 15x slower). Requires CUDA 12.4+ with "
                "conditional graph support.",
                cudaGetErrorString(err));
            goto fail;
        }

        // 3. Add conditional WHILE node
        cudaGraphNodeParams cond_params{};
        cond_params.type = cudaGraphNodeTypeConditional;
        cond_params.conditional.handle = handle_;
        cond_params.conditional.type = cudaGraphCondTypeWhile;
        cond_params.conditional.size = 1;

        cudaGraphNode_t cond_node;
        err = cudaGraphAddNode(&cond_node, graph_, nullptr, nullptr, 0, &cond_params);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR(
                "ConditionalRunner: add conditional node failed: %s — falling back "
                "to per-step decode (up to 15x slower).",
                cudaGetErrorString(err));
            goto fail;
        }

        // 4. Get body graph
        cudaGraph_t body_graph = cond_params.conditional.phGraph_out[0];

        // 5. Capture decode body into body_graph via stream capture
        // Sync stream before capture to ensure all prior work is complete
        cudaStreamSynchronize(stream);

        err = cudaStreamBeginCaptureToGraph(stream, body_graph, nullptr, nullptr, 0,
                                            get_capture_mode());
        if (err != cudaSuccess) {
            IMP_LOG_ERROR(
                "ConditionalRunner: capture failed — falling back to per-step decode "
                "(up to 15x slower). Check for unsupported CUDA operations in the "
                "forward pass. Error: %s",
                cudaGetErrorString(err));
            goto fail;
        }

        graph_diag::g_phase = graph_diag::Phase::CAPTURE;

        // 5a. Forward decode step: embedding → layers → norm → LM head → sample
        //     Writes sampled token to d_token_id_. The h_mapped parameter receives
        //     a D2H copy each iteration (harmless scratch write; the real ring buffer
        //     write is in post_decode_step_kernel below).
        // NOTE: the h_mapped scratch must NOT alias h_step_counter_ — the
        // polling harvest (poll_new_tokens) reads the counter concurrently,
        // and this per-iteration D2H token copy would transiently overwrite
        // it with a token id, making the host over-read the ring.
        executor->forward_decode_async(body_state, d_token_id_, h_decode_scratch_, stream);

        // 5b. Post-decode-step kernel: ring buffer write, counter increment, EOS check, think budget
        post_decode_step_kernel<<<1, 1, 0, stream>>>(d_token_id_, d_ring_buffer_, d_step_counter_mapped_,
                                                     d_position_, d_context_len_, d_step_counter_,
                                                     config_.max_steps, config_.eos_id, d_stop_ids_,
                                                     static_cast<int>(config_.stop_ids.size()),
                                                     d_think_limit_, config_.think_start_id,
                                                     config_.think_end_id, config_.think_grace_tokens,
                                                     d_think_count_, d_in_think_, d_think_exit_step_,
                                                     d_content_after_think_,
                                                     config_.token_is_whitespace, config_.vocab_size,
                                                     config_.ignore_eos ? 1 : 0, d_penalty_ring_,
                                                     penalty_prefix_len_, d_penalty_count_,
                                                     d_step_limit_, d_burst_done_mapped_, handle_);

        // 5c. End capture
        cudaGraph_t captured_body = nullptr;
        err = cudaStreamEndCapture(stream, &captured_body);
        graph_diag::g_phase = graph_diag::Phase::NORMAL;
        if (err != cudaSuccess) {
            IMP_LOG_ERROR(
                "ConditionalRunner: capture failed — falling back to per-step decode "
                "(up to 15x slower). Check for unsupported CUDA operations in the "
                "forward pass. Error: %s",
                cudaGetErrorString(err));
            goto fail;
        }

        // 5d. Convert kernel→kernel edges to PDL in the body graph
        if (pdl::is_available()) {
            int converted = apply_pdl_edges(body_graph);
            if (converted > 0)
                IMP_LOG_INFO("ConditionalRunner: %d body graph edges converted to PDL", converted);
        }

        graph_diag::log_kernel_nodes(body_graph, "capture.cond_body");
        graph_diag::dump_graph(body_graph, "capture.cond_body");
        graph_diag::dump_graph(graph_, "capture.cond_top");

        // 6. Instantiate the top-level graph
        err = cudaGraphInstantiate(&exec_, graph_, 0);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR(
                "ConditionalRunner: graph instantiation failed: %s — falling back "
                "to per-step decode (up to 15x slower).",
                cudaGetErrorString(err));
            goto fail;
        }

        IMP_LOG_INFO("ConditionalRunner: graph built (max_steps=%d)", config_.max_steps);
    }

    last_read_step_ = 0;
    launched_ = false;
    return true;

fail:
    IMP_LOG_ERROR(
        "ConditionalRunner: setup failed — falling back to per-step decode "
        "(up to 15x slower). Check logs above for the specific failure.");
    cleanup();
    return false;
}

bool CudaGraphConditionalRunner::launch(cudaStream_t stream) {
    if (!exec_)
        return false;

    graph_diag::PhaseScope scope(graph_diag::Phase::REPLAY);
    cudaError_t err = cudaGraphLaunch(exec_, stream);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("ConditionalRunner: launch failed: %s", cudaGetErrorString(err));
        return false;
    }
    graph_diag::check_post_launch(stream, "cond_launch");
    launched_ = true;
    return true;
}

bool CudaGraphConditionalRunner::rearm(int32_t first_token, int position, int context_len,
                                       int step_limit, bool in_think, int think_limit,
                                       cudaStream_t stream) {
    if (!exec_ || launched_)
        return false;
    // The captured graph sized its attention workspace for
    // initial_context_len + max_steps tokens; never replay past that.
    const int steps_this_launch = (step_limit > 0) ? step_limit : config_.max_steps;
    if (context_len + steps_this_launch > captured_context_ceiling())
        return false;

    auto up = [&](void* dst, const void* src, size_t n, const char* what) {
        cudaError_t err = cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("ConditionalRunner: rearm %s upload failed: %s", what,
                          cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    const int zero = 0;
    const int neg_one = -1;
    const int think = in_think ? 1 : 0;
    if (!up(d_token_id_, &first_token, sizeof(int32_t), "token") ||
        !up(d_position_, &position, sizeof(int), "position") ||
        !up(d_context_len_, &context_len, sizeof(int), "context_len") ||
        !up(d_step_counter_, &zero, sizeof(int), "step_counter") ||
        !up(d_step_limit_, &step_limit, sizeof(int), "step_limit") ||
        !up(d_think_limit_, &think_limit, sizeof(int), "think_limit"))
        return false;
    if (d_in_think_) {
        if (!up(d_in_think_, &think, sizeof(int), "in_think") ||
            !up(d_think_count_, &zero, sizeof(int), "think_count") ||
            !up(d_think_exit_step_, &neg_one, sizeof(int), "think_exit") ||
            !up(d_content_after_think_, &zero, sizeof(int), "content_after_think"))
            return false;
    }
    *h_step_counter_ = 0;
    *h_burst_done_ = 0;
    last_read_step_ = 0;
    return true;
}

std::vector<int32_t> CudaGraphConditionalRunner::wait_and_get_tokens(cudaStream_t stream) {
    if (!launched_)
        return {};

    cudaError_t sync_err = cudaStreamSynchronize(stream);
    launched_ = false;
    // F-A17: a swallowed sync error here means the mapped step-counter / ring
    // buffer below hold stale or garbage data — surface it instead of emitting
    // bogus tokens. Returning empty lets the worker's step() error path handle
    // it (and the F-A3 poison check classify it) rather than misattributing the
    // failure to a later checked call.
    if (sync_err != cudaSuccess) {
        IMP_LOG_ERROR("AsyncGraphLoop: stream sync failed (%s) — discarding burst",
                      cudaGetErrorString(sync_err));
        return {};
    }

    int total_steps = *h_step_counter_;
    std::vector<int32_t> tokens(total_steps);
    for (int i = 0; i < total_steps; i++) {
        tokens[i] = h_ring_buffer_[i];
    }
    return tokens;
}

int CudaGraphConditionalRunner::poll_new_tokens(std::vector<int32_t>& out_tokens) {
    // Use atomic acquire load on step counter — ensures all prior GPU writes
    // to the ring buffer are visible before we read the counter value.
    // This is critical on WSL2 where mapped pinned memory writes from the GPU
    // may not be immediately visible without a memory barrier.
    int current_step = __atomic_load_n(h_step_counter_, __ATOMIC_ACQUIRE);
    int new_count = current_step - last_read_step_;
    for (int i = last_read_step_; i < current_step; i++) {
        out_tokens.push_back(__atomic_load_n(&h_ring_buffer_[i], __ATOMIC_ACQUIRE));
    }
    last_read_step_ = current_step;
    return new_count;
}

int CudaGraphConditionalRunner::steps_completed() const {
    return h_step_counter_ ? __atomic_load_n(h_step_counter_, __ATOMIC_ACQUIRE) : 0;
}

bool CudaGraphConditionalRunner::try_finish_burst(cudaStream_t stream) {
    if (!launched_)
        return true;
    // Key off the device-published done flag, NOT cudaStreamQuery — the query
    // reports the stream idle while the conditional WHILE graph is still
    // iterating (observed on WSL2/sm_120), and tearing the loop's buffers
    // down on that signal corrupts generation. The kernel's stop path is the
    // only loop exit and publishes the flag behind a system fence.
    if (!h_burst_done_ || __atomic_load_n(h_burst_done_, __ATOMIC_ACQUIRE) == 0)
        return false;
    // Loop broke — drain the graph epilogue (near-instant) and surface errors
    // with the same contract as wait_and_get_tokens (F-A17).
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    launched_ = false;
    if (sync_err != cudaSuccess) {
        IMP_LOG_ERROR("AsyncGraphLoop: burst failed (%s) — no further tokens from this burst",
                      cudaGetErrorString(sync_err));
    }
    return true;
}

void CudaGraphConditionalRunner::finish_burst_blocking(cudaStream_t stream) {
    if (!launched_)
        return;
    // Fallback for a device loop that stopped making progress without
    // publishing the done flag (graph error paths never reach the stop
    // kernel) — block until the stream drains and surface the error.
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    launched_ = false;
    if (sync_err != cudaSuccess) {
        IMP_LOG_ERROR("AsyncGraphLoop: blocking burst finish failed (%s)",
                      cudaGetErrorString(sync_err));
    }
}

void CudaGraphConditionalRunner::cleanup() {
    // Ensure all GPU work referencing these resources has completed before freeing.
    if (launched_) {
        cudaDeviceSynchronize();
        launched_ = false;
    }

    bool had_exec = (exec_ != nullptr);
    if (exec_) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }

    if (d_token_id_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_token_id_));
        d_token_id_ = nullptr;
    }
    if (d_position_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_position_));
        d_position_ = nullptr;
    }
    if (d_step_limit_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_step_limit_));
        d_step_limit_ = nullptr;
    }
    if (d_think_limit_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_think_limit_));
        d_think_limit_ = nullptr;
    }
    if (d_context_len_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_context_len_));
        d_context_len_ = nullptr;
    }
    if (d_step_counter_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_step_counter_));
        d_step_counter_ = nullptr;
    }
    if (d_stop_ids_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_stop_ids_));
        d_stop_ids_ = nullptr;
    }
    if (d_think_count_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_think_count_));
        d_think_count_ = nullptr;
    }
    if (d_in_think_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_in_think_));
        d_in_think_ = nullptr;
    }
    if (d_think_exit_step_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_think_exit_step_));
        d_think_exit_step_ = nullptr;
    }
    if (d_content_after_think_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_content_after_think_));
        d_content_after_think_ = nullptr;
    }
    if (d_penalty_ring_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_penalty_ring_));
        d_penalty_ring_ = nullptr;
    }
    if (d_penalty_count_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_penalty_count_));
        d_penalty_count_ = nullptr;
    }
    penalty_prefix_len_ = 0;

    if (h_ring_buffer_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_ring_buffer_));
        h_ring_buffer_ = nullptr;
    }
    d_ring_buffer_ = nullptr;
    if (h_step_counter_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_step_counter_));
        h_step_counter_ = nullptr;
    }
    d_step_counter_mapped_ = nullptr;
    if (h_burst_done_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_burst_done_));
        h_burst_done_ = nullptr;
    }
    d_burst_done_mapped_ = nullptr;
    if (h_decode_scratch_) {
        IMP_CUDA_CHECK_LOG(cudaFreeHost(h_decode_scratch_));
        h_decode_scratch_ = nullptr;
    }

    // Release the per-device graph memory pool (matches CudaGraphCapture::reset).
    // Keeps long-running sessions from holding stale graph reservations.
    if (had_exec) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGraphMemTrim(dev);
    }

    launched_ = false;
    last_read_step_ = 0;
}

// ---------------------------------------------------------------------------
// Pipelined constrained decode: device-side cursor advance
// ---------------------------------------------------------------------------

__global__ void pipeline_advance_kernel(int* __restrict__ d_pos, int* __restrict__ d_ctx) {
    (*d_pos)++;
    (*d_ctx)++;
}

void launch_pipeline_advance(int* d_pos, int* d_ctx, cudaStream_t stream) {
    pipeline_advance_kernel<<<1, 1, 0, stream>>>(d_pos, d_ctx);
}

}  // namespace imp
