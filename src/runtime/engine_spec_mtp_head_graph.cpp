// speculative.mtp_head_graph: the batched MTP head pass (mtp_head_pass_run_) replayed as one CUDA graph per
// (fed rows, requests, chain) instead of ~56 launches; cache and bound in mtp_head_graph.h.

#include "compute/gemm.h"
#include "compute/mtp_forward.h"
#include "core/logging.h"
#include "quant/nvfp4_quant.h"
#include "runtime/cuda_graph.h"
#include "runtime/engine.h"
#include "runtime/mtp_head_graph.h"

#include <algorithm>
#include <chrono>
#include <exception>

namespace imp {

void MtpHeadGraphsDeleter::operator()(MtpHeadGraphs* g) const noexcept { delete g; }

int Engine::mtp_head_pass_(const MtpPassTables& tab, const void* hidden, const NvFP4QuantResult* head,
                           bool chain, cudaStream_t stream) {
    if (mtp_head_graphs_ == nullptr) {
        mtp_head_graphs_.reset(new MtpHeadGraphs{});
        mtp_head_graphs_->h_out = PinnedBuffer::acquire(cuda_host_pinned_allocator(),
                                                        static_cast<size_t>(kMtpMaxTopW) * kMtpMaxChainK *
                                                            sizeof(int32_t));
    }
    auto eager = [&] {
        return mtp_head_pass_run_(tab, hidden, head, chain, tab.max_pos, tab.chain_max_pos, stream);
    };
    MtpHeadGraphs& G = *mtp_head_graphs_;
    if (!runtime_config_.speculative.mtp_head_graph || G.doomed)
        return eager();
    const auto* ws = mtp_ws_storage_.get();
    const int rows = mtp_scan_smem_rows(std::max(tab.max_pos, chain ? tab.chain_max_pos : 0), ws->max_seq_len);
    const void* head_data = head != nullptr ? head->packed_data : nullptr;
    const MtpHeadGraphKey key{tab.n, tab.m, chain};
    auto it = G.by_key.find(key);
    if (it != G.by_key.end() && it->second.exec && (it->second.hidden != hidden || it->second.head != head_data)) {
        G.by_key.erase(it);  // a captured pointer moved: capture again
        --G.n_graphs;
        it = G.by_key.end();
    }
    const bool cached = it != G.by_key.end() && it->second.exec;
    const int uses = it != G.by_key.end() ? it->second.eager_uses : 0;
    auto fail = [&](const char* what, cudaError_t err) {
        IMP_LOG_WARN("[mtp-head-graph] %s failed (n=%d m=%d chain=%d): %s", what, tab.n, tab.m, chain ? 1 : 0,
                     cudaGetErrorString(err));
        (void)cudaGetLastError();
        if (++G.failures >= kMtpHeadGraphMaxFailures) {
            G.doomed = true;
            IMP_LOG_WARN("[mtp-head-graph] disabled after %d failures, the head pass runs eager", G.failures);
        }
    };
    switch (mtp_head_graph_step(cached, uses, G.n_graphs, G.doomed)) {
        case MtpHeadGraphStep::Eager:
            if (G.n_graphs < kMtpHeadGraphMax)
                G.by_key[key].eager_uses++;
            return eager();
        case MtpHeadGraphStep::Replay: {
            MtpHeadGraph& g = it->second;
            if (!mtp_scan_nodes_set_rows(g.exec, g.scan, rows)) {
                fail("scan smem update", cudaGetLastError());
                return eager();
            }
            const cudaError_t err = cudaGraphLaunch(g.exec, stream);
            if (err != cudaSuccess) {
                fail("launch", err);
                return eager();
            }
            if (++G.replays % 1000 == 0)
                IMP_LOG_INFO("[mtp-head-graph] %lld replays, %zu graphs", G.replays, G.n_graphs);
            return g.n_out;
        }
        case MtpHeadGraphStep::Capture:
            break;
    }
    // Capture: nothing runs on the stream until the launch below; any failure runs the pass eager.
    const auto t0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaStreamBeginCapture(stream, get_capture_mode());
    if (err != cudaSuccess) {
        fail("begin capture", err);
        return eager();
    }
    // cuBLASLt records into the capture (the WMMA fallback otherwise): the eager use warmed every shape.
    gemm_set_lt_capture_allowed(true);
    int n_out = -1;
    try {
        n_out = mtp_head_pass_run_(tab, hidden, head, chain, rows - 1, rows - 1, stream);
    } catch (const std::exception& e) {
        IMP_LOG_WARN("[mtp-head-graph] pass threw inside the capture: %s", e.what());
        n_out = -1;
    }
    gemm_set_lt_capture_allowed(false);
    cudaGraph_t raw = nullptr;
    err = cudaStreamEndCapture(stream, &raw);
    CudaGraph graph;
    graph.reset(raw);
    if (err != cudaSuccess || !graph || n_out < 0) {
        fail("capture", err);
        return eager();
    }
    MtpScanNodes scan;
    if (!mtp_scan_nodes_find(graph, scan) || scan.nodes.size() != (n_out == 2 * tab.m ? 2u : 1u)) {
        fail("scan node lookup", cudaErrorInvalidValue);
        return eager();
    }
    cudaGraphExec_t raw_exec = nullptr;
    err = cudaGraphInstantiate(&raw_exec, graph, 0);
    CudaGraphExec exec;
    exec.reset(raw_exec);
    if (err != cudaSuccess) {
        fail("instantiate", err);
        return eager();
    }
    err = cudaGraphLaunch(exec, stream);
    if (err != cudaSuccess) {
        fail("first launch", err);
        return eager();
    }
    MtpHeadGraph& g = G.by_key[key];
    g.graph = std::move(graph);
    g.exec = std::move(exec);
    g.scan = std::move(scan);
    g.n_out = n_out;
    g.hidden = hidden;
    g.head = head_data;
    ++G.n_graphs;
    G.failures = 0;
    IMP_LOG_INFO("[mtp-head-graph] captured n=%d m=%d chain=%d (%zu scan nodes, %zu graphs): %.1f ms", tab.n,
                 tab.m, chain ? 1 : 0, g.scan.nodes.size(), G.n_graphs,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return n_out;
}

}  // namespace imp
