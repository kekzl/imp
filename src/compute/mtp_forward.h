#pragma once
// MTP draft-token forward pass through the MTP head.
// Phase 2.1 (PR #172): reduced forward (emb->pre_fc_norm->fc->final_norm->lm_head->argmax),
// skips transformer block, acceptance rate below trained-MTP optimum.
// Phase 2.2.MoE (this file): MoE block via imp::gemm/swiglu/moe_gate_topk_fused/
// shared_expert_gate_scale; attention still passthrough (q_proj 8192 vs o_proj 4096
// mismatch vs standard GQA, needs upstream-reference investigation). 2.2.Attn: future.

#include "compute/moe_routing.h"  // MoeRoutingBuffers
#include "memory/host_pinned.h"
#include "core/tensor.h"
#include "model/mtp_head.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

class Model;
struct NvFP4QuantResult;

// Max top-W width the MTP draft step can emit per position (tree-ceiling
// measurement, Stage 0). The draft argmax is top-0.
constexpr int kMtpMaxTopW = 8;

// Max device-side chain length (per-chain stride of
// MtpDraftWorkspace::d_chain_tokens). Longer speculative.mtp_k values fall
// back to the host chain loop.
constexpr int kMtpMaxChainK = 16;

// Blocks in pass 1 of the two-pass serving top-W kernel.
constexpr int kMtpTopWBlocks = 64;

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
    // [kMtpMaxTopW] int — top-W candidate ids (Stage 0 tree-ceiling probe,
    // and the multi-candidate branch seed: chains 1..W-1 start from ranks
    // 1..W-1 here).
    int*  d_topk = nullptr;
    // [kMtpTopWBlocks*kMtpMaxTopW] float/int: partial per-block top-W (value,id) pairs for
    // the two-pass serving top-W kernel. Probe's single-CTA kernel scans the full vocab per
    // width (too slow for serving); pass 1 splits vocab across blocks, pass 2 merges.
    float* d_topk_part_val = nullptr;
    int* d_topk_part_idx = nullptr;
    // [kMtpMaxTopW] float: top-W logit values in rank order (serving kernel only).
    // d_topk_val[0]-d_topk_val[1] is the head's top-1/top-2 margin; gates
    // speculative.mtp_tree_margin (a confident head skips the second candidate).
    float* d_topk_val = nullptr;
    // [kMtpMaxTopW*kMtpMaxChainK] int: device-side chain slots; chain c at
    // [c*kMtpMaxChainK], step i's argmax lands in slot i, feeding step i+1's embedding
    // lookup without a host round-trip. One D2H drains all chains at the end.
    // Chain 0 is the linear path, byte-compatible with the old single-chain layout.
    int32_t* d_chain_tokens = nullptr;
    // [hidden_dim] FP16: h_final of the last FEED pair, snapshotted before chain 0's first
    // continuation overwrites d_h_final. Chains 1..W-1 branch at the first position and
    // need exactly this hidden for their first continuation.
    void* d_h_final_snap = nullptr;
    // [1] int — persistent argmax scratch for the host-path draft step
    // (replaces a per-draft cudaMallocAsync/cudaFreeAsync pair).
    int*  d_argmax = nullptr;
    // [1] int32: persistent token-id scratch for the host-chain draft step (input twin of
    // d_argmax). Host path used to cudaMalloc/cudaFreeAsync 4 bytes per draft step (AUDIT
    // B10, wrong allocator + a serving-phase allocation). Persistent here avoids both.
    int32_t* d_tok = nullptr;

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

    // Phase 2.2.Attn+KV: MTP-side KV cache (per-session, M=1 only). K/V accumulate across
    // draft calls; each call appends one row at mtp_pos, then attends [0,mtp_pos+1).
    // Qwen3.6 max_seq=16K: 16384*2*256*2 bytes = 16 MiB each, 32 MiB total. Reset via mtp_kv_reset().
    void* d_k_cache       = nullptr;  // [max_seq_len, num_kv_heads, head_dim] FP16
    void* d_v_cache       = nullptr;  // [max_seq_len, num_kv_heads, head_dim] FP16
    int   mtp_pos         = 0;        // next slot to write (0..max_seq_len-1)
    int   max_seq_len     = 0;        // cache capacity
    // Multi-slot KV (batched verify): n_kv_slots caches of max_seq_len rows in one
    // allocation. d_k_cache/d_v_cache/mtp_pos are the ACTIVE slot's view (mtp_select_slot);
    // slot_pos holds the other slots' positions.
    void* d_k_cache_base  = nullptr;
    void* d_v_cache_base  = nullptr;
    int   n_kv_slots      = 1;
    int   cur_slot        = 0;
    size_t kv_slot_elems  = 0;        // max_seq_len * num_kv_heads * head_dim
    std::vector<int> slot_pos;        // [n_kv_slots] next write position per slot
    // Ragged multi-slot feed scratch: per-row slot/position tables, gather indices into the
    // caller's hidden buffer, gathered rows, final_norm of every fed row.
    // Aliases: int tables carve d_feed_tokens (4x feed_rows_cap ints); d_b_gather is
    // d_b_h_norm; d_b_h_final is d_b_norm (freed once MLP consumes post-norm). Never freed alone.
    int*  d_row_slots     = nullptr;  // [feed_rows_cap]
    int*  d_row_pos       = nullptr;  // [feed_rows_cap]
    int*  d_row_src       = nullptr;  // [feed_rows_cap]
    void* d_b_gather      = nullptr;  // [feed_rows_cap, H]
    void* d_b_h_final     = nullptr;  // [feed_rows_cap, H]

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

    // RoPE config (Phase 2.2.Attn+RoPE): when rope_dim>0, mrope-aware Q/K rotation applies
    // before the attention scan. Both Q (extracted) and K (this step) rotate; cached K's
    // stay rotated from their own insertion-time position.
    float rope_theta      = 0.0f;
    int   rope_dim        = 0;     // 0 = disable RoPE
    bool  rope_neox       = true;  // (currently mtp_mrope_kernel hardcodes neox)
    // mrope section half-counts (Qwen3-VL). Sum must equal rope_dim/2. Qwen3.6: {11,11,10}.
    // Text-only tokens: all 3 positions equal, mrope reduces to partial-rope; sec* fields
    // stay for future multimodal handling.
    int   mrope_sec0      = 0;
    int   mrope_sec1      = 0;
    int   mrope_sec2      = 0;
    // RoPE scaling must mirror the main forward's rope path or the drafter rotates Q/K
    // differently from the verifier at extended positions, silently degrading acceptance
    // with position (#897). rope_freq_scale: linear scale (main uses inv_scaling=1/freq_scale).
    // yarn_ext_factor>0 engages YaRN blending via yarn_corr_dim_0/1 + yarn_attn_factor (mscale).
    float rope_freq_scale = 1.0f;
    float yarn_ext_factor  = 0.0f;
    float yarn_attn_factor = 1.0f;
    float yarn_corr_dim_0  = 0.0f;
    float yarn_corr_dim_1  = 0.0f;
    float rms_norm_eps    = 1e-6f;
    float arch_norm_offset = 0.0f;  // for q_norm/k_norm (Qwen3.5/3.6 gamma=1+W)

    // Batched prefill-feed scratch (dense attn+KV heads only). The per-pair loop reading the
    // whole head's weights once per token was prohibitively slow; mtp_feed_batch feeds up to
    // feed_rows_cap (token,hidden) pairs in one M=rows pass. feed_rows_cap==0: unsupported
    // head (MoE MLP, or no attention/KV cache), caller keeps the loop.
    int      feed_rows_cap = 0;
    int32_t* d_feed_tokens = nullptr;  // [feed_rows_cap]
    void* d_b_emb      = nullptr;  // [rows, H] emb rows, normed in place
    void* d_b_h_norm   = nullptr;  // [rows, H] normed hidden rows
    void* d_b_fc_in    = nullptr;  // [rows, 2H] concat(emb_n, h_n)
    void* d_b_fc_out   = nullptr;  // [rows, H] fc output / residual stream
    void* d_b_norm     = nullptr;  // [rows, H] input_layernorm, then post_norm
    void* d_b_q_full   = nullptr;  // [rows, q_out] q_proj (incl gate half)
    void* d_b_q_attn   = nullptr;  // [rows, nh*hd] extracted Q
    void* d_b_k        = nullptr;  // [rows, nkv*hd]
    void* d_b_v        = nullptr;  // [rows, nkv*hd]
    void* d_b_attn_out = nullptr;  // [rows, nh*hd]
    void* d_b_res      = nullptr;  // [rows, H] o_proj out, then down_proj out
    void* d_b_gate     = nullptr;  // [rows, d_ff]
    void* d_b_up       = nullptr;  // [rows, d_ff]
    void* d_b_act      = nullptr;  // [rows, d_ff]

    // Post-norm feed scratch (diagnostics.mtp_prenorm_h): [prenorm_rows_cap,H] FP16, the fed
    // hidden rows after the target's final norm. Sized once at enable time (engine_spec_mtp.cpp)
    // to the widest feed a prefill chunk can produce.
    void* d_prenorm_rows = nullptr;
    int prenorm_rows_cap = 0;
};

// Rows per batched prefill-feed pass (mtp_feed_batch). Bounds the batch
// scratch above: ~61 MiB at Qwen3.8-27B dims (H=5120, d_ff=17408, 24 heads).
constexpr int kMtpFeedRows = 256;

// One MTP draft step; returns the draft token id via out_token_id.
// d_h_prev: main-model final hidden [hidden_dim] FP16 GPU. mtp.loaded must be true.
// out_token_id=nullptr: skip lm_head GEMV+argmax+sync (feed-only step, ~10x cheaper;
// used for prefill/verify catch-up positions whose prediction is unused).
// out_topk_ids (optional, top_w>0): top-W candidate ids descending-logit order,
// out_topk_ids[0]==*out_token_id (Stage 0 tree-ceiling measurement).
// lm_head_nvfp4 (optional): NVFP4 decode-cache view of main_lm_head, ~4x less HBM read;
// draft-only precision; verification stays lossless.
// d_prev_token/d_out_token: device-chain I/O, no H2D/D2H/sync per step; with top_w>0 the
// fast top-W kernel fills ws.d_topk and rank 0 lands in d_out_token. Caller drains the
// whole chain with one D2H+sync at the end.
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

// Batched prefill feed: append n_rows (token,hidden) pairs to the MTP KV cache in one
// M=n_rows pass (embedding/norm/fc/attention/MLP batched, causal attention per query row
// over [0,mtp_pos+row+1)). Feed-only: no logits/argmax/sync. Advances ws.mtp_pos by n_rows.
// Requires ws.feed_rows_cap>=n_rows (dense-MLP head w/ attention+KV cache).
// h_tokens: host ptr (uploaded internally); d_hidden_rows: [n_rows,hidden_dim] FP16 device.
bool mtp_feed_batch(const int32_t* h_tokens, const void* d_hidden_rows, int n_rows,
                    const MtpHead& mtp, const Tensor& main_tok_emb,
                    MtpDraftWorkspace& ws, int hidden_dim, cudaStream_t stream);
// Ragged multi-slot feed (batched verify): row r = (h_tokens[r], d_hidden_all[h_src_rows[r]])
// appended to KV slot h_slots[r] at h_pos[r], attending [0,h_pos[r]+1) of that slot.
// Rows of one slot must be in ascending position order (append precedes the scan that reads
// it). Writes final_norm of every row to ws.d_b_h_final[n_rows,H]; caller advances
// mtp_pos/slot_pos. post_norm (optional): apply target's final norm before feed
// (diagnostics.mtp_prenorm_h, upstream convention).
bool mtp_feed_rows_multislot(const int32_t* h_tokens, const void* d_hidden_all, const int* h_src_rows,
                             int n_rows, const int* h_slots, const int* h_pos, const MtpHead& mtp,
                             const Tensor& main_tok_emb, MtpDraftWorkspace& ws, int hidden_dim,
                             cudaStream_t stream, const Tensor* post_norm = nullptr,
                             float post_norm_eps = 1e-6f, float post_norm_offset = 0.0f);
// Make `slot` the active KV slot: saves the current slot's position, loads
// the new one, re-points d_k_cache / d_v_cache. No-op for the active slot.
void mtp_select_slot(MtpDraftWorkspace& ws, int slot);
// dst[r] = src[d_idx[r]], rows of `cols` FP16 (device index array).
void mtp_gather_rows(const void* d_src, const int* d_idx, void* d_dst, int cols, int n_rows, cudaStream_t stream);

// Top-W over device logits into ws.d_topk (descending-logit order), no D2H/sync.
// `fast`: two-pass serving kernel (pass1 per-block top-W, pass2 single-block merge);
// `reference`: probe's single-CTA oracle for the GPU test to compare against.
// Both break exact-value ties by lowest index within a pass; fast kernel's pass structure
// can order EQUAL values differently across slice boundaries (test needs distinct values).
// logits FP32 when fp32_logits (NVFP4 lm_head cache path), FP16 otherwise.
bool mtp_topw_fast(const void* d_logits, bool fp32_logits, int vocab_size, int top_w, MtpDraftWorkspace& ws,
                   cudaStream_t stream);
bool mtp_topw_reference(const void* d_logits, bool fp32_logits, int vocab_size, int top_w,
                        MtpDraftWorkspace& ws, cudaStream_t stream);

// Applies YaRN-aware mrope rotation to one MTP step's Q[n_heads,head_dim]/K[n_kv_heads,
// head_dim] FP16 in place at `pos`; mirrors main forward's rope_forward math so draft
// and verifier rotate identically on rope-scaled models (#897).
// n_rows>1: Q/K hold n_rows consecutive steps; row r rotates at pos+r (or d_row_pos[r]
// if given). Pass null d_q or d_k to skip that side.
void mtp_apply_mrope(void* d_q, int n_heads, void* d_k, int n_kv_heads, int head_dim, int rope_dim,
                     float theta, int sec0, int sec1, int sec2, int pos, float inv_scaling,
                     float ext_factor, float attn_factor, float corr_dim_0, float corr_dim_1,
                     cudaStream_t stream, int n_rows = 1, const int* d_row_pos = nullptr);

// Allocates the workspace from the VRAM allocator; caller keeps ws alive (typically
// Engine, session lifetime). MoE buffers sized from n_experts/top_k/expert_d_ff/
// shared_d_ff; pass 0 for any to disable the MoE block (back-compat 2-arg form).
bool mtp_workspace_allocate(MtpDraftWorkspace& ws, int hidden_dim, int vocab_size,
                            int n_experts = 0, int top_k = 0,
                            int expert_d_ff = 0, int shared_d_ff = 0,
                            int num_heads = 0, int num_kv_heads = 0, int head_dim = 0,
                            int max_seq_len = 0, int n_kv_slots = 1);
void mtp_workspace_free(MtpDraftWorkspace& ws);

// Reset every MTP-side KV cache position (start of new sequences). The K/V
// buffers retain their allocation; only the positions are zeroed.
inline void mtp_kv_reset(MtpDraftWorkspace& ws) {
    ws.mtp_pos = 0;
    for (auto& p : ws.slot_pos) p = 0;
}

}  // namespace imp
