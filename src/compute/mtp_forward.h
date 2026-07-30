#pragma once
// =============================================================================
// mtp_forward.h — Multi-Token-Predictor draft step
// =============================================================================
//
// One draft-token forward pass through the MTP head.
//
// Phase status:
//   - 2.1 (shipped PR #172): reduced forward (emb → pre_fc_norm → fc →
//          final_norm → lm_head → argmax). Skips the transformer block.
//          Acceptance rate will be far below trained-MTP optimum.
//   - 2.2.MoE (this file): MoE block plumbed via existing imp::gemm /
//          swiglu / moe_gate_topk_fused / shared_expert_gate_scale
//          primitives. Attention block still a passthrough (architectural
//          shape ambiguity — q_proj outputs 8192 but o_proj inputs 4096
//          on Qwen3.6 MTP, doesn't match standard GQA conventions; needs
//          upstream-reference investigation before correct implementation).
//   - 2.2.Attn (future): full attention block.
//
// Design: docs/superpowers/specs/2026-05-14-mtp-wiring-design.md
// =============================================================================

#include "compute/moe_routing.h"  // MoeRoutingBuffers
#include "memory/host_pinned.h"
#include "core/tensor.h"
#include "model/mtp_head.h"
#include <cuda_runtime.h>

namespace imp {

class Model;
struct NvFP4QuantResult;

// Max top-W width the MTP draft step can emit per position (tree-ceiling
// measurement, Stage 0). The draft argmax is top-0.
constexpr int kMtpMaxTopW = 8;

// Max device-side chain length (capacity of MtpDraftWorkspace::d_chain_tokens).
// Longer speculative.mtp_k values fall back to the host chain loop.
constexpr int kMtpMaxChainK = 16;

// Workspace tensors needed for one draft step. Caller pre-allocates these so
// the draft step is graph-safe (no cudaMalloc inside captured graph).
struct MtpDraftWorkspace {
    // ---- Phase 2.1 reduced-forward scratch ----
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
    // [vocab_size] FP32 — draft logits when the lm_head GEMV runs through the
    // NVFP4 decode cache (gemv_nvfp4_kpar_fp32 writes FP32). ~1 MiB.
    void* d_logits_f32 = nullptr;
    // [kMtpMaxTopW] int — top-W candidate ids (Stage 0 tree-ceiling probe).
    int*  d_topk = nullptr;
    // [kMtpMaxChainK] int — device-side chain slots: step i's argmax lands in
    // d_chain_tokens[i] and feeds step i+1's embedding lookup without a host
    // round-trip; one D2H of the whole chain at the end.
    int32_t* d_chain_tokens = nullptr;
    // [1] int — persistent argmax scratch for the host-path draft step
    // (replaces a per-draft cudaMallocAsync/cudaFreeAsync pair).
    int*  d_argmax = nullptr;

    // ---- Phase 2.2 MoE scratch ----
    // [hidden_dim] FP16 — post_attention_layernorm(fc_out)
    void* d_post_norm   = nullptr;
    // [2*expert_d_ff] FP16 — single-expert gate_up output (gate at 0..d_ff,
    // up at d_ff..2*d_ff)
    void* d_expert_gate_up = nullptr;
    // [expert_d_ff] FP16 — silu(gate)*up per chosen expert
    void* d_expert_act    = nullptr;
    // [top_k * hidden_dim] FP16 — per-chosen-expert down outputs, contiguous
    // along the top_k axis; consumed by moe_weighted_sum_residual.
    void* d_expert_outputs = nullptr;
    // [hidden_dim] FP16 — accumulator (moe weighted-sum + residual via
    // moe_weighted_sum_residual).
    void* d_moe_out       = nullptr;
    // Shared expert scratch
    void* d_shared_gate   = nullptr;  // [shared_d_ff] FP16
    void* d_shared_up     = nullptr;  // [shared_d_ff] FP16
    void* d_shared_act    = nullptr;  // [shared_d_ff] FP16 (silu(gate)*up)
    void* d_shared_out    = nullptr;  // [hidden_dim] FP16 (shared_down_proj @ act)

    // ---- Phase 2.2.Attn scratch ----
    void* d_input_norm    = nullptr;  // [hidden_dim] FP16 — input_layernorm output
    void* d_q_full        = nullptr;  // [2 * num_heads * head_dim] FP16 — q_proj (incl gate)
    void* d_q_attn        = nullptr;  // [num_heads * head_dim] FP16 — Q half extracted (post-qknorm+RoPE)
    void* d_k_proj        = nullptr;  // [num_kv_heads * head_dim] FP16 (current step's k, post-qknorm+RoPE)
    void* d_v_proj        = nullptr;  // [num_kv_heads * head_dim] FP16 (current step's v)
    void* d_attn_out      = nullptr;  // [num_heads * head_dim] FP16
    void* d_attn_residual = nullptr;  // [hidden_dim] FP16 — o_proj output (added to fc_out)
    int*  d_mtp_position  = nullptr;  // [1] int — current MTP cache position (for RoPE)

    // ---- Phase 2.2.Attn+KV — MTP-side KV cache (per-session, M=1 only) ----
    // K and V cache accumulate across MTP draft calls. Each call appends one
    // row at position `mtp_pos`, then runs softmax attention over positions
    // [0, mtp_pos+1). For Qwen3.6 max_seq=16K: 16384 × 2 × 256 × 2 bytes = 16 MiB
    // each = 32 MiB total. Reset on new sequence via mtp_kv_reset().
    void* d_k_cache       = nullptr;  // [max_seq_len, num_kv_heads, head_dim] FP16
    void* d_v_cache       = nullptr;  // [max_seq_len, num_kv_heads, head_dim] FP16
    int   mtp_pos         = 0;        // next slot to write (0..max_seq_len-1)
    int   max_seq_len     = 0;        // cache capacity

    // Routing buffer pool (n_experts, top_k both known at enable time)
    MoeRoutingBuffers routing_buf;
    // Per-step host-side copies of indices/weights for the M=1 host-side
    // per-expert GEMV loop. Allocated as cudaHostAlloc'd for pinned D2H.
    PinnedBuffer h_expert_indices;  // [top_k] (T5b, memory/host_pinned.h)
    PinnedBuffer h_expert_weights;  // [top_k]

    // Hyperparameters captured at workspace-allocate time so the draft step
    // doesn't need to re-derive them from the model.
    int hidden_dim   = 0;
    int n_experts    = 0;
    int top_k        = 0;
    int expert_d_ff  = 0;
    int shared_d_ff  = 0;

    // Attention dims (Phase 2.2.Attn). Set to 0 to disable the attention
    // block (current behavior); set to non-zero to engage the gated single-
    // token attention path.
    int num_heads    = 0;
    int num_kv_heads = 0;
    int head_dim     = 0;

    // RoPE config (Phase 2.2.Attn+RoPE). When rope_dim > 0, mrope-aware
    // Q/K rotation is applied BEFORE the attention scan. Both Q (extracted)
    // and K (this step's projection) get rotated; cached K's stay rotated
    // from their own insertion-time position.
    float rope_theta      = 0.0f;
    int   rope_dim        = 0;     // 0 = disable RoPE
    bool  rope_neox       = true;  // (currently mtp_mrope_kernel hardcodes neox)
    // mrope section half-counts (Qwen3-VL multimodal). Sum must equal
    // rope_dim/2. For Qwen3.6: {11, 11, 10}. For text-only tokens all 3
    // positions are equal so mrope reduces to standard partial-rope; sec*
    // fields stay relevant for future multimodal token handling.
    int   mrope_sec0      = 0;
    int   mrope_sec1      = 0;
    int   mrope_sec2      = 0;
    // RoPE scaling — must mirror the main forward's rope path or the drafter
    // rotates Q/K differently from the verifier at extended positions, silently
    // degrading acceptance with position (issue #897). rope_freq_scale is the
    // linear scaling (main uses inv_scaling = 1/freq_scale); yarn_ext_factor > 0
    // engages YaRN blending with yarn_corr_dim_0/1 (from rope_yarn_corr_dims())
    // and yarn_attn_factor (mscale). Defaults = no scaling (Qwen3.6 base).
    float rope_freq_scale = 1.0f;
    float yarn_ext_factor  = 0.0f;
    float yarn_attn_factor = 1.0f;
    float yarn_corr_dim_0  = 0.0f;
    float yarn_corr_dim_1  = 0.0f;
    float rms_norm_eps    = 1e-6f;
    float arch_norm_offset = 0.0f;  // for q_norm/k_norm (Qwen3.5/3.6 gamma=1+W)
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
//   - *out_token_id   : drafted next token id (D2H copy of argmax). Pass
//                       nullptr to skip the lm_head GEMV + argmax + stream
//                       sync entirely — a cache-feed-only step (prefill /
//                       verify catch-up positions whose prediction is never
//                       consumed). The lm_head read (~1 GiB FP16 on Qwen3.6's
//                       248k vocab) dominates per-step cost, so feed-only
//                       steps are ~an order of magnitude cheaper.
//   - out_topk_ids    : optional [top_w] host buffer; when non-null and
//                       top_w>0, receives the top-W candidate ids in
//                       descending-logit order (out_topk_ids[0] == *out_token_id).
//                       Used by the Stage 0 tree-ceiling measurement.
//   - lm_head_nvfp4   : optional NVFP4 decode-cache view of main_lm_head
//                       (GraphExecutor::lm_head_nvfp4_view). When set, the
//                       chain logits GEMV reads ~4x less HBM than the FP16
//                       weight — the dominant per-draft cost on large-vocab
//                       models. Draft-only: verification stays lossless
//                       regardless of the draft head's precision.
//   - d_prev_token    : device-chain input — read the previous token id from
//                       this device int instead of prev_token_id (which is
//                       then ignored, pass -1). No H2D upload, no host
//                       bounds check (validate the chain once after D2H).
//   - d_out_token     : device-chain output — write the argmax to this device
//                       int; NO D2H copy, NO stream sync (out_token_id and
//                       the top-W path are ignored). The caller drains the
//                       whole chain with a single D2H + sync.
//
// Returns false on any precondition violation (mtp not loaded, null buffers).
bool mtp_draft_step(int prev_token_id, const void* d_h_prev,
                    const MtpHead& mtp,
                    const Tensor& main_tok_emb,
                    const Tensor& main_lm_head,
                    MtpDraftWorkspace& ws,
                    int hidden_dim, int vocab_size,
                    int* out_token_id,
                    cudaStream_t stream,
                    int* out_topk_ids = nullptr, int top_w = 0,
                    const NvFP4QuantResult* lm_head_nvfp4 = nullptr,
                    const int32_t* d_prev_token = nullptr,
                    int32_t* d_out_token = nullptr);

// Apply the YaRN-aware mrope rotation to a single MTP step's Q [n_heads, head_dim]
// and K [n_kv_heads, head_dim] (FP16) in place at position `pos`. Mirrors the main
// forward's rope_forward math so the draft head and the verifier rotate Q/K
// identically on rope-scaled models (issue #897): inv_scaling = 1/rope_freq_scale,
// ext_factor > 0 engages YaRN blending via corr_dim_0/1 + attn_factor (mscale).
// Exposed for the rope-parity unit test. Pass null d_q or d_k to skip that side.
void mtp_apply_mrope(void* d_q, int n_heads, void* d_k, int n_kv_heads, int head_dim, int rope_dim,
                     float theta, int sec0, int sec1, int sec2, int pos, float inv_scaling,
                     float ext_factor, float attn_factor, float corr_dim_0, float corr_dim_1,
                     cudaStream_t stream);

// Allocate the workspace from the VRAM allocator. Caller is responsible for
// keeping ws alive (typically owned by the Engine for the lifetime of a session).
// The MoE-related buffers (post_norm, expert outputs, shared expert scratch,
// routing pool) are sized from `n_experts`, `top_k`, `expert_d_ff`,
// `shared_d_ff`. Pass 0 for any of those to disable the MoE block at runtime
// (back-compat — Phase 2.1 callers can keep using the 2-arg form below).
bool mtp_workspace_allocate(MtpDraftWorkspace& ws, int hidden_dim, int vocab_size,
                            int n_experts = 0, int top_k = 0,
                            int expert_d_ff = 0, int shared_d_ff = 0,
                            int num_heads = 0, int num_kv_heads = 0, int head_dim = 0,
                            int max_seq_len = 0);
void mtp_workspace_free(MtpDraftWorkspace& ws);

// Reset the MTP-side KV cache position (start of new sequence). The K/V
// buffers retain their allocation; only `mtp_pos` is zeroed.
inline void mtp_kv_reset(MtpDraftWorkspace& ws) { ws.mtp_pos = 0; }

}  // namespace imp
