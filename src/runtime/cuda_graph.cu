#include "runtime/cuda_graph.h"
#include "runtime/pdl.h"
#include "graph/executor.h"
#include "compute/sampling.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

namespace imp {

// ---------------------------------------------------------------------------
// apply_pdl_edges — convert kernel→kernel edges to PDL edges in a graph
// ---------------------------------------------------------------------------
#if IMP_CUDA_13_1
static int apply_pdl_edges(cudaGraph_t graph) {
    if (!graph) return 0;

    // 1. Enumerate all nodes
    size_t num_nodes = 0;
    cudaError_t err = cudaGraphGetNodes(graph, nullptr, &num_nodes);
    if (err != cudaSuccess || num_nodes == 0) return 0;

    std::vector<cudaGraphNode_t> nodes(num_nodes);
    err = cudaGraphGetNodes(graph, nodes.data(), &num_nodes);
    if (err != cudaSuccess) return 0;

    // 2. Build set of kernel nodes (use linear scan — node counts are small)
    std::vector<cudaGraphNode_t> kernel_nodes;
    kernel_nodes.reserve(num_nodes);
    for (size_t i = 0; i < num_nodes; i++) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(nodes[i], &type) == cudaSuccess &&
            type == cudaGraphNodeTypeKernel) {
            kernel_nodes.push_back(nodes[i]);
        }
    }
    if (kernel_nodes.size() < 2) return 0;

    // 3. Enumerate all edges with edge data
    size_t num_edges = 0;
    err = cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &num_edges);
    if (err != cudaSuccess || num_edges == 0) return 0;

    std::vector<cudaGraphNode_t> from(num_edges), to(num_edges);
    std::vector<cudaGraphEdgeData> edge_data(num_edges);
    err = cudaGraphGetEdges(graph, from.data(), to.data(), edge_data.data(), &num_edges);
    if (err != cudaSuccess) return 0;

    // Helper: check if a node is a kernel node
    auto is_kernel = [&](cudaGraphNode_t n) -> bool {
        for (const auto& kn : kernel_nodes) {
            if (kn == n) return true;
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
        if (edge_data[i].type != cudaGraphDependencyTypeDefault) continue;
        if (!is_kernel(from[i]) || !is_kernel(to[i])) continue;

        // Check if the source kernel has PDL enabled
        cudaKernelNodeParams kparams{};
        cudaError_t kerr = cudaGraphKernelNodeGetParams(from[i], &kparams);
        if (kerr != cudaSuccess || !pdl::is_enabled(kparams.func)) {
            skipped_non_pdl++;
            continue;
        }

        // Remove old default edge
        err = cudaGraphRemoveDependencies(graph, &from[i], &to[i], &edge_data[i], 1);
        if (err != cudaSuccess) continue;

        // Add PDL edge
        err = cudaGraphAddDependencies(graph, &from[i], &to[i], &pdl_edge, 1);
        if (err != cudaSuccess) {
            // Rollback: re-add the default edge
            cudaGraphAddDependencies(graph, &from[i], &to[i], &edge_data[i], 1);
            continue;
        }
        converted++;
    }

    if (skipped_non_pdl > 0)
        IMP_LOG_DEBUG("apply_pdl_edges: skipped %d edges (source kernel not PDL-enabled)",
                      skipped_non_pdl);

    return converted;
}
#endif // IMP_CUDA_13_1

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

    // Convert kernel→kernel edges to PDL edges for tail/head overlap
#if IMP_CUDA_13_1
    if (pdl::is_available()) {
        int converted = apply_pdl_edges(graph_);
        if (converted > 0)
            IMP_LOG_DEBUG("CudaGraphCapture: %d edges converted to PDL", converted);
    }
#endif

    err = cudaGraphInstantiate(&graph_exec_, graph_, 0);
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
        // If a previous capture attempt failed, skip further attempts
        if (capture_failed_) {
            decode_fn_(stream);
            step_count_++;
            return true;
        }

        IMP_LOG_INFO("CudaGraphRunner: capturing CUDA graph (step %d)", step_count_);

        if (!graph_.begin_capture(stream)) {
            // Capture failed -- fall back to direct execution permanently
            IMP_LOG_WARN("CudaGraphRunner: capture begin failed, disabling graph capture");
            capture_failed_ = true;
            decode_fn_(stream);
            step_count_++;
            return true;
        }

        decode_fn_(stream);

        if (!graph_.end_capture()) {
            IMP_LOG_WARN("CudaGraphRunner: capture end failed, disabling graph capture");
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
    capture_failed_ = false;
    last_batch_size_ = -1;
    last_max_blocks_ = -1;
}

// ---------------------------------------------------------------------------
// CudaGraphConditionalRunner — GPU-autonomous decode loop
// ---------------------------------------------------------------------------

// Device kernel: post-decode-step bookkeeping.
// Copies sampled token to ring buffer, increments counters, checks stop
// conditions, and breaks the WHILE loop via cudaGraphSetConditional.
__global__ void post_decode_step_kernel(
        const int32_t* __restrict__ d_token_id,     // [1] sampled token
        int32_t* __restrict__ d_ring_buffer,         // [max_steps] output (mapped pinned)
        int* __restrict__ d_ring_step_counter,       // [1] mapped step counter (host-visible)
        int* __restrict__ d_position,                // [1] current position
        int* __restrict__ d_context_len,             // [1] current context length
        int* __restrict__ d_step_counter,            // [1] device-side step counter
        int max_steps,
        int eos_id,
        const int32_t* __restrict__ d_stop_ids,
        int n_stop_ids,
        cudaGraphConditionalHandle handle) {
    int step = *d_step_counter;
    int32_t token = *d_token_id;

    // Write token to ring buffer (visible to host via mapped memory)
    d_ring_buffer[step] = token;

    // Increment counters
    int new_pos = *d_position + 1;
    int new_ctx = *d_context_len + 1;
    *d_position = new_pos;
    *d_context_len = new_ctx;
    *d_step_counter = step + 1;

    // Update mapped step counter (host-visible, for polling)
    *d_ring_step_counter = step + 1;

    // Check stop conditions
    bool should_stop = (step + 1 >= max_steps) || (token == eos_id);
    for (int i = 0; i < n_stop_ids; i++) {
        if (token == d_stop_ids[i]) should_stop = true;
    }

    if (should_stop) {
        cudaGraphSetConditional(handle, 0);  // break WHILE loop
    }
}

CudaGraphConditionalRunner::~CudaGraphConditionalRunner() {
    cleanup();
}

bool CudaGraphConditionalRunner::setup(
        GraphExecutor* executor,
        const InferenceState& state_template,
        int32_t first_token,
        Config config,
        cudaStream_t stream) {
    cleanup();  // release any prior state
    config_ = std::move(config);

    cudaError_t err;

    // ---- Allocate device state ----
    // Must be ARGMAX_SCRATCH_BYTES — sample_greedy_device uses multi-block
    // argmax that writes partial reduction arrays after the result token.
    err = cudaMalloc(&d_token_id_, ARGMAX_SCRATCH_BYTES);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_position_, sizeof(int));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_context_len_, sizeof(int));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_step_counter_, sizeof(int));
    if (err != cudaSuccess) goto fail;

    // Stop token IDs
    if (!config_.stop_ids.empty()) {
        err = cudaMalloc(&d_stop_ids_, config_.stop_ids.size() * sizeof(int32_t));
        if (err != cudaSuccess) goto fail;
        cudaMemcpyAsync(d_stop_ids_, config_.stop_ids.data(),
                         config_.stop_ids.size() * sizeof(int32_t),
                         cudaMemcpyHostToDevice, stream);
    }

    // ---- Allocate mapped pinned memory for ring buffer ----
    {
        err = cudaHostAlloc(&h_ring_buffer_,
                             config_.max_steps * sizeof(int32_t),
                             cudaHostAllocMapped);
        if (err != cudaSuccess) goto fail;
        err = cudaHostGetDevicePointer(&d_ring_buffer_, h_ring_buffer_, 0);
        if (err != cudaSuccess) goto fail;

        err = cudaHostAlloc(&h_step_counter_, sizeof(int), cudaHostAllocMapped);
        if (err != cudaSuccess) goto fail;
        err = cudaHostGetDevicePointer(&d_step_counter_mapped_, h_step_counter_, 0);
        if (err != cudaSuccess) goto fail;
    }

    // ---- Initialize device state ----
    {
        int init_pos = config_.initial_position + 1;  // next position after prefill
        int init_ctx = config_.initial_context_len + 1;  // context grows by 1
        int init_step = 0;
        *h_step_counter_ = 0;
        memset(h_ring_buffer_, 0, config_.max_steps * sizeof(int32_t));

        cudaMemcpyAsync(d_token_id_, &first_token, sizeof(int32_t),
                         cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_position_, &init_pos, sizeof(int),
                         cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_context_len_, &init_ctx, sizeof(int),
                         cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_step_counter_, &init_step, sizeof(int),
                         cudaMemcpyHostToDevice, stream);
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
        // max_context_len is set to cover the full generation
        body_state.max_context_len = config_.initial_context_len + config_.max_steps;

        // ---- Construct CUDA graph with conditional WHILE node ----
        // 1. Create top-level graph
        err = cudaGraphCreate(&graph_, 0);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("ConditionalRunner: cudaGraphCreate failed: %s",
                         cudaGetErrorString(err));
            goto fail;
        }

        // 2. Create conditional handle (default value = 1 = "continue looping")
        err = cudaGraphConditionalHandleCreate(&handle_, graph_, 1,
                                                cudaGraphCondAssignDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("ConditionalRunner: handle create failed: %s",
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
            IMP_LOG_WARN("ConditionalRunner: add conditional node failed: %s",
                         cudaGetErrorString(err));
            goto fail;
        }

        // 4. Get body graph
        cudaGraph_t body_graph = cond_params.conditional.phGraph_out[0];

        // 5. Capture decode body into body_graph via stream capture
        // Sync stream before capture to ensure all prior work is complete
        cudaStreamSynchronize(stream);

        err = cudaStreamBeginCaptureToGraph(stream, body_graph,
                                              nullptr, nullptr, 0,
                                              cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("ConditionalRunner: begin capture to graph failed: %s",
                         cudaGetErrorString(err));
            goto fail;
        }

        // 5a. Forward decode step: embedding → layers → norm → LM head → sample
        //     Writes sampled token to d_token_id_. The h_mapped parameter receives
        //     a D2H copy each iteration (harmless scratch write; the real ring buffer
        //     write is in post_decode_step_kernel below).
        executor->forward_decode_async(body_state, d_token_id_,
                                        reinterpret_cast<int32_t*>(h_step_counter_),
                                        stream);

        // 5b. Post-decode-step kernel: ring buffer write, counter increment, EOS check
        post_decode_step_kernel<<<1, 1, 0, stream>>>(
            d_token_id_, d_ring_buffer_, d_step_counter_mapped_,
            d_position_, d_context_len_, d_step_counter_,
            config_.max_steps, config_.eos_id,
            d_stop_ids_, static_cast<int>(config_.stop_ids.size()),
            handle_);

        // 5c. End capture
        cudaGraph_t captured_body = nullptr;
        err = cudaStreamEndCapture(stream, &captured_body);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("ConditionalRunner: end capture failed: %s",
                         cudaGetErrorString(err));
            goto fail;
        }

        // 5d. Convert kernel→kernel edges to PDL in the body graph
#if IMP_CUDA_13_1
        if (pdl::is_available()) {
            int converted = apply_pdl_edges(body_graph);
            if (converted > 0)
                IMP_LOG_INFO("ConditionalRunner: %d body graph edges converted to PDL", converted);
        }
#endif

        // 6. Instantiate the top-level graph
        err = cudaGraphInstantiate(&exec_, graph_, 0);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("ConditionalRunner: instantiate failed: %s",
                         cudaGetErrorString(err));
            goto fail;
        }

        IMP_LOG_INFO("ConditionalRunner: graph built (max_steps=%d)", config_.max_steps);
    }

    last_read_step_ = 0;
    launched_ = false;
    return true;

fail:
    IMP_LOG_WARN("ConditionalRunner: setup failed, will fall back to per-step decode");
    cleanup();
    return false;
}

bool CudaGraphConditionalRunner::launch(cudaStream_t stream) {
    if (!exec_) return false;

    cudaError_t err = cudaGraphLaunch(exec_, stream);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("ConditionalRunner: launch failed: %s",
                      cudaGetErrorString(err));
        return false;
    }
    launched_ = true;
    return true;
}

std::vector<int32_t> CudaGraphConditionalRunner::wait_and_get_tokens(
        cudaStream_t stream) {
    if (!launched_) return {};

    cudaStreamSynchronize(stream);
    launched_ = false;

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

void CudaGraphConditionalRunner::cleanup() {
    // Ensure all GPU work referencing these resources has completed before freeing.
    if (launched_) {
        cudaDeviceSynchronize();
        launched_ = false;
    }

    if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }

    if (d_token_id_) { cudaFree(d_token_id_); d_token_id_ = nullptr; }
    if (d_position_) { cudaFree(d_position_); d_position_ = nullptr; }
    if (d_context_len_) { cudaFree(d_context_len_); d_context_len_ = nullptr; }
    if (d_step_counter_) { cudaFree(d_step_counter_); d_step_counter_ = nullptr; }
    if (d_stop_ids_) { cudaFree(d_stop_ids_); d_stop_ids_ = nullptr; }

    if (h_ring_buffer_) { cudaFreeHost(h_ring_buffer_); h_ring_buffer_ = nullptr; }
    d_ring_buffer_ = nullptr;
    if (h_step_counter_) { cudaFreeHost(h_step_counter_); h_step_counter_ = nullptr; }
    d_step_counter_mapped_ = nullptr;

    launched_ = false;
    last_read_step_ = 0;
}

} // namespace imp
