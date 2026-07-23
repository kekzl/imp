#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace imp {

// Device-side Token-Recycling adjacency table (#1055 verify-in-loop): the
// `token -> top-M successors` table lives in device memory so the draft walk
// and the observe updates can run INSIDE a conditional-graph verify loop
// with no host round-trip. Semantics are kept exactly equal to the host
// TokenRecycleTable (MRU promote, re-observation streak) — the loop path
// and the eager host path must draft identically from the same stream
// (pinned by tests/test_token_recycle_device.cu).
//
// The observe kernels run single-threaded on purpose: updates are a few
// slot shuffles per accepted token (microseconds), and MRU promotion is order-
// dependent — serial execution IS the specification.
struct TrDeviceTable {
    int32_t* succ = nullptr;   // vocab * slots, -1 = empty (MRU front-first)
    uint8_t* streak = nullptr; // vocab, re-observation count of the front slot
    int vocab = 0;
    int slots = 0;
};

// Allocate + zero-init (succ = -1, streak = 0). Returns false on OOM.
bool tr_device_init(TrDeviceTable& t, int vocab, int slots, cudaStream_t stream);
void tr_device_free(TrDeviceTable& t);

// Observe consecutive pairs of a token sequence: (toks[i-1] -> toks[i]).
void tr_observe_pairs(TrDeviceTable& t, const int32_t* d_toks, int n, cudaStream_t stream);

// Observe the model's top-M successor candidates per row (best first):
// row r records d_topm[r*m .. r*m+m) as successors of d_row_tokens[r].
void tr_observe_topk(TrDeviceTable& t, const int32_t* d_row_tokens, const int32_t* d_topm,
                     int rows, int m, cudaStream_t stream);

// Walk the front-slot chain from *d_last_token (device scalar) for up to
// `depth` hops, gated on streak >= min_streak per hop. Writes the chain to
// d_out_draft and its length to *d_out_len.
void tr_draft(TrDeviceTable& t, const int32_t* d_last_token, int depth, int min_streak,
              int32_t* d_out_draft, int32_t* d_out_len, cudaStream_t stream);

// ── verify-in-loop step kernel (#1055) ─────────────────────────────────
// The accept+draft+stage tail of the conditional verify-loop body (the
// post_decode_step analog). See docs/plans/2026-07-23-verify-in-loop.md.

// Device-pointer bundle (passed by value into the kernel launch).
struct TrLoopView {
    TrDeviceTable tab;
    // Chunk staging (the capture-mode verify forward reads these).
    int32_t* tokens = nullptr;        // [chunk_pad]
    int32_t* positions = nullptr;     // [chunk_pad]
    int32_t* row_ctx_lens = nullptr;  // [chunk_pad]
    int32_t* ctx_len = nullptr;       // scalar
    int32_t* past_len = nullptr;      // scalar (p0)
    int32_t* chunk_len = nullptr;     // scalar (real rows)
    // Verify outputs of the forward that just ran.
    const int32_t* argmax = nullptr;  // [chunk_pad]
    const int32_t* topm = nullptr;    // [chunk_pad * m]
    // Emission + loop state.
    int32_t* ring = nullptr;               // mapped device ptr, [token capacity]
    int32_t* ring_count_mapped = nullptr;  // mapped publish counter (host-visible)
    int32_t* emit_count = nullptr;         // device authoritative counter
    int32_t* exit_reason = nullptr;        // 0=continue 1=miss 2=stop 3=budget 4=ctx-ceiling
};

struct TrLoopParams {
    int chunk_pad = 4;
    int depth = 3;
    int min_streak = 1;
    int topm = 0;                       // 0 = no harvest
    int32_t eos_id = -1;
    const int32_t* stop_ids = nullptr;  // device, optional
    int n_stop_ids = 0;
    int token_limit = 0;                // total ring tokens allowed this burst
    int ctx_ceiling = 0;                // p0' + chunk_pad must stay <= this
};

// Standalone (non-graph) launcher used by tests and the eager fallback; the
// loop runner embeds the same kernel with a conditional handle.
void tr_verify_step(const TrLoopView& v, const TrLoopParams& p, bool no_handle,
                    cudaStream_t stream);
// Graph-embedded variant: sets the conditional handle to 0 on exit.
void tr_verify_step_conditional(const TrLoopView& v, const TrLoopParams& p,
                                cudaGraphConditionalHandle handle, cudaStream_t stream);

}  // namespace imp
