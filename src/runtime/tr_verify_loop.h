#pragma once

#include "compute/token_recycle_device.h"
#include "exec/inference_state.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace imp {

class GraphExecutor;

// Conditional-graph verify loop (#1055 phase 2, docs/plans/2026-07-23-
// verify-in-loop.md): a WHILE graph whose body is
//   [capture-mode bucket-4 verify forward → greedy_argmax_all(+top-M) →
//    tr_verify_step_conditional]
// so the whole draft→verify→accept cycle runs device-side; the host drains
// accepted tokens from a mapped ring (async-decode-loop protocol) and only
// re-enters on a draft miss / stop / budget / context ceiling.
//
// Deliberately its OWN class — the decode ConditionalRunner's stability is
// hard-won; no surgery there. The engine owns the chunk staging buffers,
// the argmax/topm block, the row tables and the device adjacency table;
// this runner owns the ring, the counters and the graph.
class TrVerifyLoopRunner {
public:
    struct Config {
        TrLoopParams params;      // chunk_pad/depth/min_streak/topm/eos/ctx_ceiling
        std::vector<int32_t> stop_ids;
        int ring_capacity = 512;  // max tokens per burst
    };

    ~TrVerifyLoopRunner();

    // Build device state + capture the body. `body_state` must be the fully
    // staged capture-mode chunk state (the caller stages chunk 0 first).
    // `engine_bufs` carries the engine-owned pointers (tokens/positions/
    // row_ctx_lens/ctx_len/past_len/chunk_len/argmax/topm + adjacency
    // table); the runner fills in its own (ring/counters/exit/token_limit).
    // Returns false on ANY capture failure — the runner is left clean and
    // the caller falls back to the eager verify path.
    bool setup(GraphExecutor* executor, const InferenceState& body_state,
               const TrLoopView& engine_bufs, const Config& cfg, cudaStream_t stream);

    // Re-seed counters + token budget and launch. The caller must have
    // (re-)staged chunk 0 and refreshed block-table contents beforehand.
    bool launch(int token_limit, cudaStream_t stream);

    // Drain newly published ring tokens (non-blocking). Returns count added.
    int poll_new_tokens(std::vector<int32_t>& out);

    // 0 while running; 1=miss 2=stop 3=budget 4=ctx-ceiling once the loop
    // exited (mapped read; ring tokens are published before the exit).
    int exit_reason() const;

    bool is_setup() const { return exec_ != nullptr; }
    bool launch_in_flight() const { return launched_; }
    // Blocks until the in-flight graph fully completed (call after
    // exit_reason() != 0 before touching KV/stream state).
    bool finish(cudaStream_t stream);

    // Compatibility probe for rearm-style reuse: same executor workspace
    // generation and a ctx ceiling that still covers the request.
    bool compatible(uint64_t workspace_generation, int ctx_ceiling) const {
        return exec_ != nullptr && workspace_generation == workspace_generation_ &&
               ctx_ceiling <= cfg_.params.ctx_ceiling;
    }

    void cleanup();

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaGraphConditionalHandle handle_{};
    Config cfg_{};
    uint64_t workspace_generation_ = 0;
    bool launched_ = false;
    int last_read_ = 0;

    // Runner-owned device/mapped state.
    int32_t* d_stop_ids_ = nullptr;
    int32_t* d_emit_count_ = nullptr;
    int32_t* d_token_limit_ = nullptr;
    int32_t* h_ring_ = nullptr;        // mapped
    int32_t* d_ring_ = nullptr;
    int32_t* h_ring_count_ = nullptr;  // mapped
    int32_t* d_ring_count_ = nullptr;
    int32_t* h_exit_ = nullptr;        // mapped
    int32_t* d_exit_ = nullptr;
};

}  // namespace imp
