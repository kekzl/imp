#pragma once
// =============================================================================
// mtp_forward.h — Multi-Token-Predictor draft step (Phase 2 scaffolding)
// =============================================================================
//
// One draft-token forward pass through the MTP head. Phase 2 status:
//   - Phase 2.1: scaffolding + reduced forward (emb → pre_fc_norm → fc →
//                final_norm → lm_head → argmax). Skips the MTP transformer
//                block (attention + 256-expert MoE) — that's the bulk of the
//                work and is genuinely multi-week to implement from scratch.
//                The reduced path produces draft tokens but acceptance rate
//                will be far below trained-MTP optimum.
//   - Phase 2.2: full transformer block (attention + MoE). Future session.
//
// Design: docs/superpowers/specs/2026-05-14-mtp-wiring-design.md
// =============================================================================

#include "core/tensor.h"
#include "model/mtp_head.h"
#include <cuda_runtime.h>

namespace imp {

class Model;

// Workspace tensors needed for one draft step. Caller pre-allocates these so
// the draft step is graph-safe (no cudaMalloc inside captured graph).
struct MtpDraftWorkspace {
    // [hidden_dim] FP16 — normalized embedding input
    void* d_emb_norm = nullptr;
    // [hidden_dim] FP16 — normalized hidden-state input
    void* d_h_norm = nullptr;
    // [2*hidden_dim] FP16 — concat(emb_norm, h_norm) before fc
    void* d_fc_in = nullptr;
    // [hidden_dim] FP16 — fc output / transformer input
    void* d_fc_out = nullptr;
    // [hidden_dim] FP16 — final_norm output
    void* d_h_final = nullptr;
    // [vocab_size] FP16 — draft logits
    void* d_logits = nullptr;
};

// One MTP draft step. Returns the draft token id via host out_token_id.
//
// Inputs:
//   - prev_token_id   : last accepted token (host int, used to gather embedding)
//   - d_h_prev        : main-model final hidden state [hidden_dim] FP16 on GPU
//   - mtp             : loaded MTP head (.loaded must be true)
//   - main_tok_emb    : main model's token embedding [vocab, hidden] FP16
//   - main_lm_head    : main model's lm_head [vocab, hidden] FP16
//   - workspace       : pre-allocated scratch tensors
//   - hidden_dim      : 2048 for Qwen3.6
//   - vocab_size      : 248320 for Qwen3.6
//   - stream
//
// Output:
//   - *out_token_id   : drafted next token id (D2H copy of argmax)
//
// Returns false on any precondition violation (mtp not loaded, null buffers).
bool mtp_draft_step(int prev_token_id, const void* d_h_prev,
                    const MtpHead& mtp,
                    const Tensor& main_tok_emb,
                    const Tensor& main_lm_head,
                    const MtpDraftWorkspace& ws,
                    int hidden_dim, int vocab_size,
                    int* out_token_id,
                    cudaStream_t stream);

// Allocate the workspace from the VRAM allocator. Caller is responsible for
// keeping ws alive (typically owned by the Engine for the lifetime of a session).
// Phase 4 wires this into the engine; for now, callers manage manually.
bool mtp_workspace_allocate(MtpDraftWorkspace& ws, int hidden_dim, int vocab_size);
void mtp_workspace_free(MtpDraftWorkspace& ws);

}  // namespace imp
