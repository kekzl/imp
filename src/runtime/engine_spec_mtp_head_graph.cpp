// The batched MTP head pass (Engine::mtp_head_pass_): feed, draft LM head, chain, drafts D2H. With
// speculative.mtp_head_graph one CUDA graph per (fed rows, requests, chain) replays it instead of ~56
// launches; cache and bound in mtp_head_graph.h.

#include "compute/gemm.h"
#include "compute/mtp_forward.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "model/model.h"
#include "quant/nvfp4_quant.h"
#include "runtime/cuda_graph.h"
#include "runtime/engine.h"
#include "runtime/mtp_head_graph.h"

#include <algorithm>
#include <chrono>
#include <exception>

namespace imp {

void MtpHeadGraphsDeleter::operator()(MtpHeadGraphs* g) const noexcept { delete g; }

namespace {

constexpr int kMtpHeadOutInts = kMtpMaxTopW * kMtpMaxChainK;
static_assert(kMtpBoundVerifyMaxDepth - 1 <= kMtpPassChainLevels, "one position table per chain level");

void note_failure(MtpHeadGraphs& G, const MtpHeadGraphKey& key, const char* what, cudaError_t err) {
    IMP_LOG_WARN("[mtp-head-graph] %s failed (n=%d m=%d chain=%d): %s", what, key.n, key.m, key.chain,
                 cudaGetErrorString(err));
    (void)cudaGetLastError();
    if (++G.failures >= kMtpHeadGraphMaxFailures && !G.doomed) {
        G.doomed = true;
        IMP_LOG_WARN("[mtp-head-graph] disabled after %d failures, the head pass runs eager", G.failures);
    }
}

// The cached graph of key, or null; one whose captured pointers moved is dropped (captured again later).
MtpHeadGraph* cached_graph(MtpHeadGraphs& G, const MtpHeadGraphKey& key, const void* hidden,
                           const void* head) {
    auto it = G.by_key.find(key);
    if (it == G.by_key.end() || !it->second.exec)
        return nullptr;
    if (it->second.hidden == hidden && it->second.head == head)
        return &it->second;
    G.by_key.erase(it);
    --G.n_graphs;
    return nullptr;
}

// Replays g with its scan smem set to rows; the drafts it writes, -1 = failed (nothing launched).
int replay(MtpHeadGraphs& G, const MtpHeadGraphKey& key, MtpHeadGraph& g, int rows, cudaStream_t stream) {
    if (!mtp_scan_nodes_set_rows(g.exec, g.scan, rows)) {
        note_failure(G, key, "scan smem update", cudaGetLastError());
        return -1;
    }
    const cudaError_t err = cudaGraphLaunch(g.exec, stream);
    if (err != cudaSuccess) {
        note_failure(G, key, "launch", err);
        return -1;
    }
    if (++G.replays % 1000 == 0)
        IMP_LOG_INFO("[mtp-head-graph] %lld replays, %zu graphs", G.replays, G.n_graphs);
    return g.n_out;
}

// Captures run(rows - 1) on stream, caches it under key and launches it once; the drafts it writes, -1 = not
// captured (nothing launched: the caller runs the pass eager).
template <class Run>
int capture(MtpHeadGraphs& G, const MtpHeadGraphKey& key, int rows, const void* hidden, const void* head,
            Run&& run, cudaStream_t stream) {
    const auto t0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaStreamBeginCapture(stream, get_capture_mode());
    if (err != cudaSuccess) {
        note_failure(G, key, "begin capture", err);
        return -1;
    }
    // cuBLASLt records into the capture (the WMMA fallback otherwise): the eager use warmed every shape.
    gemm_set_lt_capture_allowed(true);
    int n_out = -1;
    try {
        n_out = run(rows - 1);
    } catch (const std::exception& e) {
        IMP_LOG_WARN("[mtp-head-graph] pass threw inside the capture: %s", e.what());
    }
    gemm_set_lt_capture_allowed(false);
    cudaGraph_t raw = nullptr;
    err = cudaStreamEndCapture(stream, &raw);
    CudaGraph graph;
    graph.reset(raw);
    if (err != cudaSuccess || !graph || n_out < 0) {
        note_failure(G, key, "capture", err);
        return -1;
    }
    MtpScanNodes scan;
    if (!mtp_scan_nodes_find(graph, scan) || scan.nodes.size() != static_cast<size_t>(n_out / key.m)) {
        note_failure(G, key, "scan node lookup", cudaErrorInvalidValue);
        return -1;
    }
    cudaGraphExec_t raw_exec = nullptr;
    err = cudaGraphInstantiate(&raw_exec, graph, 0);
    CudaGraphExec exec;
    exec.reset(raw_exec);
    if (err == cudaSuccess)
        err = cudaGraphLaunch(exec, stream);
    if (err != cudaSuccess) {
        note_failure(G, key, "instantiate / first launch", err);
        return -1;
    }
    MtpHeadGraph& g = G.by_key[key];
    g.graph = std::move(graph);
    g.exec = std::move(exec);
    g.scan = std::move(scan);
    g.n_out = n_out;
    g.hidden = hidden;
    g.head = head;
    ++G.n_graphs;
    G.failures = 0;
    IMP_LOG_INFO("[mtp-head-graph] captured n=%d m=%d chain=%d draw=%d (%zu scan nodes, %zu graphs): %.1f ms",
                 key.n, key.m, key.chain, key.draw, g.scan.nodes.size(), G.n_graphs,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return n_out;
}

// What the head pass reads besides its tables.
struct HeadPassInputs {
    const Model& model;
    GraphExecutor& executor;
    MtpDraftWorkspace& ws;
    const void* hidden;            // the verify chunk's hidden rows (tab.src indexes them)
    const NvFP4QuantResult* head;  // NVFP4 draft LM head, or null
    const Tensor* post_norm;       // the target's final norm on the gathered rows, or null
    int draft_vocab;               // speculative.mtp_draft_vocab: FP8 head rows scored, 0 = all
    int draw_top_k;                // staged draft draws (stage_draft_draws), 0 = argmax drafts
};

// The pass after the table upload: feed, draft LM head, the chain when asked; drafts in ws.d_chain_tokens.
// Scans size their scores for feed_max / chain_max (+1 per further chain level). Returns m per finished draft
// level (m .. (1 + chain) * m), -1 = feed failed, -2 = LM head failed.
int run_head_pass(const HeadPassInputs& in, const MtpPassTables& tab, int chain, int feed_max, int chain_max,
                  cudaStream_t stream) {
    const Model& model = in.model;
    MtpDraftWorkspace& ws = in.ws;
    const int hidden_dim = model.config_.d_model;
    const int m = tab.m;
    if (!mtp_feed_rows_multislot_dev(tab.tokens, in.hidden, tab.src, tab.n, tab.slots, tab.pos, feed_max,
                                     *model.mtp_, model.tok_emb_, ws, hidden_dim, stream, in.post_norm,
                                     model.config_.rms_norm_eps, model.config_.norm_weight_offset))
        return -1;
    // The next draft per request: LM head over the last fed row of each.
    mtp_gather_rows(ws.d_b_h_final, tab.last, ws.d_b_gather, hidden_dim, m, stream);
    if (!in.executor.lm_head_rows_argmax(ws.d_b_gather, m, ws.d_chain_tokens, stream, in.head, in.draft_vocab,
                                         0, in.draw_top_k))
        return -2;
    // Draft lv + 1 per bound request: (draft lv, the head's own final-norm row) at the request's next
    // position in one more ragged pass, the device chain of mtp_feed_pairs_ batched over requests (drafts
    // never leave the device). Level 1 reads the last fed row, levels 2+ the chain's own rows 0..m-1.
    for (int lv = 1; lv <= chain; ++lv) {
        const bool ok = mtp_feed_rows_multislot_dev(ws.d_chain_tokens + (static_cast<ptrdiff_t>(lv - 1) * m),
                                                    ws.d_b_h_final, lv == 1 ? tab.last : tab.chain_rows, m,
                                                    tab.chain_slots, tab.chain_pos[lv - 1],
                                                    chain_max + lv - 1, *model.mtp_, model.tok_emb_, ws,
                                                    hidden_dim, stream) &&
                        in.executor.lm_head_rows_argmax(ws.d_b_h_final, m,
                                                        ws.d_chain_tokens + (static_cast<ptrdiff_t>(lv) * m),
                                                        stream, in.head, in.draft_vocab, lv * m,
                                                        in.draw_top_k);
        if (!ok)
            return lv * m;
    }
    return (1 + chain) * m;
}

}  // namespace

int Engine::mtp_head_pass_(const MtpPassTables& tab, const void* hidden, const NvFP4QuantResult* head,
                           cudaStream_t stream, const int32_t** drafts, const VerifySampling* draws,
                           int n_draws) {
    if (mtp_head_graphs_ == nullptr) {
        mtp_head_graphs_.reset(new MtpHeadGraphs{});
        mtp_head_graphs_->h_out.resize(kMtpHeadOutInts);
    }
    MtpHeadGraphs& G = *mtp_head_graphs_;
    // The chained KV row is speculative: the next feed starts at that position and overwrites it.
    const int max_seq = mtp_ws_storage_->max_seq_len;  // head KV ring rows: bounds the scan smem only
    // Chain levels: one per draft past the first that a step of these m requests verifies.
    const int chain = std::clamp(bound_verify_step_depth(bv_, tab.m) - 1, 0, kMtpPassChainLevels);
    const int draft_vocab = runtime_config_.speculative.mtp_draft_vocab;
    // Coupled drafts: every level's draw args go up before the pass (a replayed graph reads them too).
    const int draw_rows = (1 + chain) * tab.m;
    const int draw_top_k = draws != nullptr && n_draws >= draw_rows
                               ? executor_->stage_draft_draws(draws, draw_rows, tab.m, head, draft_vocab,
                                                              stream)
                               : 0;
    // diagnostics.mtp_prenorm_h (default on): the head takes the target's post-norm hidden.
    const HeadPassInputs in{
        *model_,     *executor_, *mtp_ws_storage_,
        hidden,      head,       runtime_config_.diagnostics.mtp_prenorm_h ? &model_->output_norm() : nullptr,
        draft_vocab, draw_top_k};
    auto run = [&](int feed_max, int chain_max) {
        return run_head_pass(in, tab, chain, feed_max, chain_max, stream);
    };
    const MtpHeadGraphKey key{tab.n, tab.m, chain, draw_top_k};
    const void* head_data = head != nullptr ? head->packed_data : nullptr;
    MtpHeadGraph* g = runtime_config_.speculative.mtp_head_graph && !G.doomed
                          ? cached_graph(G, key, hidden, head_data)
                          : nullptr;
    const int uses = G.by_key.contains(key) ? G.by_key[key].eager_uses : 0;
    const MtpHeadGraphStep step = runtime_config_.speculative.mtp_head_graph
                                      ? mtp_head_graph_step(g != nullptr, uses, G.n_graphs, G.doomed)
                                      : MtpHeadGraphStep::Eager;
    const int rows = mtp_scan_smem_rows(std::max(tab.max_pos, chain > 0 ? tab.chain_max_pos + chain - 1 : 0),
                                        max_seq);
    int n_out = -1;
    if (step == MtpHeadGraphStep::Replay)
        n_out = replay(G, key, *g, rows, stream);
    else if (step == MtpHeadGraphStep::Capture)
        n_out = capture(
            G, key, rows, hidden, head_data, [&](int max_pos) { return run(max_pos, max_pos); }, stream);
    else if (runtime_config_.speculative.mtp_head_graph && !G.doomed && G.n_graphs < kMtpHeadGraphMax)
        G.by_key[key].eager_uses++;
    if (n_out < 0)
        n_out = run(tab.max_pos, tab.chain_max_pos);
    if (n_out < 0)
        return n_out;
    // Pageable: a pinned destination priced no gain (nsys c2 host window 2561.8 -> 2687.2 us, n=1).
    int32_t* out = G.h_out.data();
    if (cudaMemcpyAsync(out, mtp_ws_storage_->d_chain_tokens, static_cast<size_t>(n_out) * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return -3;
    *drafts = out;
    return n_out;
}

}  // namespace imp
