#pragma once
// MTP draft-token forward pass through the MTP head.
// Phase 2.1 (PR #172): reduced forward (emb->pre_fc_norm->fc->final_norm->lm_head->argmax),
// skips transformer block, acceptance rate below trained-MTP optimum.
// Phase 2.2.MoE (this file): MoE block via imp::gemm/swiglu/moe_gate_topk_fused/
// shared_expert_gate_scale; attention still passthrough (q_proj 8192 vs o_proj 4096
// mismatch vs standard GQA, needs upstream-reference investigation). 2.2.Attn: future.

#include "compute/moe_routing.h"  // MoeRoutingBuffers
#include "compute/mtp_workspace_bytes.h"  // kMtp* sizes, mtp_workspace_bytes (the plan's mirror)
#include "memory/host_pinned.h"
#include "core/cuda_raii.h"  // CudaEvent
#include "core/tensor.h"
#include "model/mtp_head.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

class Model;
struct NvFP4QuantResult;

// Workspace tensors needed for one draft step. Caller pre-allocates these so
// the draft step is graph-safe (no cudaMalloc inside captured graph).
struct MtpDraftWorkspace {
    // Device footprint of the mtp_dev_alloc sites (mtp_alloc_footprint each); equals mtp_workspace_bytes() of
    // the shape.
    size_t device_bytes = 0;
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
    // [hidden_dim / 32] block_q8_1 + d8: d_h_final quantized for a GGUF lm_head with neither an NVFP4 nor an
    // FP8 form (dp4a GEMV into d_logits_f32).
    void* d_q8_1 = nullptr;
    void* d_d8 = nullptr;
    // [kMtpChainSlots*kMtpMaxChainK] int: device-side chain slots; step i's argmax lands in
    // slot i, feeding step i+1's embedding lookup without a host round-trip; one D2H drains
    // the chain. The bound batched verify uses every slot (requests x depth).
    int32_t* d_chain_tokens = nullptr;
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
    float* d_scan_part = nullptr;     // split-K partials (mtp_scan_part_bytes), inside d_mtp_position

    // MTP-side KV cache, a ring of max_seq_len rows per slot: position p lives in row p % max_seq_len
    // and a query at p attends mtp_ring_window (the last max_seq_len positions, minus rows a later
    // position overwrote). Qwen3.8: 16384 rows x 4 KiB = 64 MiB per slot. Reset via mtp_kv_reset().
    void* d_k_cache       = nullptr;  // [max_seq_len, num_kv_heads, head_dim] FP16
    void* d_v_cache       = nullptr;  // [max_seq_len, num_kv_heads, head_dim] FP16
    int mtp_pos = 0;                  // next position to write (absolute, unbounded)
    int max_seq_len = 0;              // ring rows per slot
    // [n_kv_slots] device: per slot the largest position written + 1 (atomicMax in the appends, read by the
    // scans: rows a stale draft overwrote drop out of the window). Zeroed by mtp_kv_hi_reset.
    int* d_slot_hi = nullptr;
    // Multi-slot KV (batched verify): n_kv_slots caches of max_seq_len rows in one
    // allocation. d_k_cache/d_v_cache/mtp_pos are the ACTIVE slot's view (mtp_select_slot);
    // slot_pos holds the other slots' positions.
    void* d_k_cache_base  = nullptr;
    void* d_v_cache_base  = nullptr;
    int   n_kv_slots      = 1;
    int   cur_slot        = 0;
    size_t kv_slot_elems  = 0;        // max_seq_len * num_kv_heads * head_dim
    std::vector<int> slot_pos;        // [n_kv_slots] next write position per slot
    // [n_kv_slots] host upper bound of d_slot_hi (mtp_slot_hi_bound), raised at each bind; decides whether a
    // resume still finds a window (mtp_ring_resume_lost).
    std::vector<int> slot_hi_bound;
    // Ragged multi-slot feed scratch: per-row slot/position tables, gather indices into the
    // caller's hidden buffer, gathered rows, final_norm of every fed row.
    // Aliases: int tables carve d_feed_tokens (12x feed_rows_cap ints); d_b_gather is
    // d_b_h_norm; d_b_h_final is d_b_norm (freed once MLP consumes post-norm). Never freed alone.
    int*  d_row_slots     = nullptr;  // [feed_rows_cap]
    int*  d_row_pos       = nullptr;  // [feed_rows_cap]
    int*  d_row_src       = nullptr;  // [feed_rows_cap]
    // Batched head pass (mtp_head_pass_upload): its feed and chain tables packed for ONE pinned H2D,
    // h_pass_tables staged, pass_tables_copied guards reuse of the staging. [kMtpPassTableInts] each.
    int* d_pass_tables = nullptr;
    PinnedBuffer h_pass_tables;
    CudaEvent pass_tables_copied;
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
    // the MTP body FP8 copy: per-row E4M3 copies of the dense body (MtpBodyWeight order) the feed reads at
    // <= 32 rows; nullptr = the FP16 weights.
    void* d_fp8_body = nullptr;
    const void* fp8_w[8] = {};
    const float* fp8_s[8] = {};
    // the MTP carry: per KV slot [carry_rows_cap, h_prev cols] FP16, a bound request's target hidden
    // rows from batched decode steps until the head takes them (engine_spec_mtp_batched.cpp, 2.5 MiB per
    // slot on Qwen3.8).
    void* d_carry = nullptr;
    int carry_rows_cap = 0;

    // Qwen4Exp hyper-connection scratch (hc_count > 0): the head reads and writes an hc stream
    // [hc_count, hidden] instead of one hidden row. d_hc_x holds multi_hidden after a step.
    int hc_count = 0;
    int hc_lowrank = 0;
    void* d_hc_x = nullptr;       // [hc * hidden] stream state, next step's h_prev
    void* d_hc_normed = nullptr;  // [hc * hidden] grouped norm, also pre_fc_norm_hidden output
    void* d_hc_mixw = nullptr;    // [hc * hidden] mix gate logits
    void* d_hc_low = nullptr;     // [hc_lowrank]
    void* d_hc_inj = nullptr;     // [hc] inject weights of the pending combine
};

// Width of one h_prev row: hidden_dim, or hc_count x hidden_dim on a Qwen4Exp head.
inline int mtp_h_prev_cols(const MtpDraftWorkspace& ws) {
    return ws.hc_count > 0 ? ws.hc_count * ws.hidden_dim : ws.hidden_dim;
}
// h_prev of the next chained step: the head's h_final, or its multi_hidden hc stream (Qwen4Exp).
inline const void* mtp_chain_hidden(const MtpDraftWorkspace& ws) {
    return ws.hc_count > 0 ? ws.d_hc_x : ws.d_h_final;
}

// One MTP draft step; returns the draft token id via out_token_id.
// d_h_prev: main-model final hidden [hidden_dim] FP16 GPU. mtp.loaded must be true.
// out_token_id=nullptr: skip lm_head GEMV+argmax+sync (feed-only step, ~10x cheaper;
// used for prefill/verify catch-up positions whose prediction is unused).
// lm_head_nvfp4 (optional): NVFP4 decode-cache view of main_lm_head, ~4x less HBM read;
// draft-only precision; verification stays lossless.
// lm_head_fp8/_scales (optional): FP8 per-row head (gemm.nvfp4_lm_head=fp8), taken first.
// d_prev_token/d_out_token: device-chain I/O, no H2D/D2H/sync per step. Caller drains the
// whole chain with one D2H+sync at the end.
// Returns false on any precondition violation (mtp not loaded, null buffers).
[[nodiscard]] bool mtp_draft_step(int prev_token_id, const void* d_h_prev, const MtpHead& mtp,
                                  const Tensor& main_tok_emb, const Tensor& main_lm_head,
                                  MtpDraftWorkspace& ws, int hidden_dim, int vocab_size, int* out_token_id,
                                  cudaStream_t stream, const NvFP4QuantResult* lm_head_nvfp4 = nullptr,
                                  const int32_t* d_prev_token = nullptr, int32_t* d_out_token = nullptr,
                                  const void* lm_head_fp8 = nullptr,
                                  const float* lm_head_fp8_scales = nullptr);

// Batched prefill feed: append n_rows (token,hidden) pairs to the MTP KV cache in one
// M=n_rows pass (embedding/norm/fc/attention/MLP batched, causal attention per query row
// over [0,mtp_pos+row+1)). Feed-only: no logits/argmax/sync. Advances ws.mtp_pos by n_rows.
// Requires ws.feed_rows_cap>=n_rows (dense-MLP head w/ attention+KV cache).
// h_tokens: host ptr (uploaded internally); d_hidden_rows: [n_rows,hidden_dim] FP16 device.
[[nodiscard]] bool mtp_feed_batch(const int32_t* h_tokens, const void* d_hidden_rows, int n_rows,
                    const MtpHead& mtp, const Tensor& main_tok_emb,
                    MtpDraftWorkspace& ws, int hidden_dim, cudaStream_t stream);
// Ragged multi-slot feed (batched verify): row r = (h_tokens[r], d_hidden_all[h_src_rows[r]])
// appended to KV slot h_slots[r] at h_pos[r], attending [0,h_pos[r]+1) of that slot.
// Rows of one slot must be in ascending position order (append precedes the scan that reads
// it). Writes final_norm of every row to ws.d_b_h_final[n_rows,H]; caller advances
// mtp_pos/slot_pos. post_norm (optional): apply target's final norm before feed
// (diagnostics.mtp_prenorm_h, upstream convention). h_tokens may be a device pointer (cudaMemcpyDefault):
// the bound verify's chained second draft feeds draft 1 without a sync.
[[nodiscard]] bool mtp_feed_rows_multislot(const int32_t* h_tokens, const void* d_hidden_all, const int* h_src_rows,
                             int n_rows, const int* h_slots, const int* h_pos, const MtpHead& mtp,
                             const Tensor& main_tok_emb, MtpDraftWorkspace& ws, int hidden_dim,
                             cudaStream_t stream, const Tensor* post_norm = nullptr,
                             float post_norm_eps = 1e-6f, float post_norm_offset = 0.0f);
// Same feed with every table already on the device in stream order (d_tokens [n_rows] ids, d_src_rows,
// d_slots, d_pos); max_pos = the largest h_pos, checked by mtp_feed_tables_valid. No copy, no host read.
[[nodiscard]] bool mtp_feed_rows_multislot_dev(const int32_t* d_tokens, const void* d_hidden_all,
                                               const int* d_src_rows, int n_rows, const int* d_slots,
                                               const int* d_pos, int max_pos, const MtpHead& mtp,
                                               const Tensor& main_tok_emb, MtpDraftWorkspace& ws,
                                               int hidden_dim, cudaStream_t stream,
                                               const Tensor* post_norm = nullptr, float post_norm_eps = 1e-6f,
                                               float post_norm_offset = 0.0f);
// Every row's slot in [0, n_kv_slots) and position >= 0 (the head KV is a ring); *max_pos = the largest
// position.
[[nodiscard]] bool mtp_feed_tables_valid(const MtpDraftWorkspace& ws, const int* h_slots, const int* h_pos,
                                         int n_rows, int* max_pos);

// The batched head pass's tables on the device: the feed's n rows, then per bound request (m) the last fed
// row (draft gather and chain hidden source), the chain slot and the chain position.
struct MtpPassTables {
    const int32_t* tokens = nullptr;
    const int* src = nullptr;
    const int* slots = nullptr;
    const int* pos = nullptr;
    const int* last = nullptr;
    const int* chain_slots = nullptr;
    const int* chain_rows = nullptr;  // 0..m-1: each request's chain row, hidden source of chain level 2+
    const int* chain_pos[kMtpPassChainLevels] = {};  // [l]: the chain positions + l, chain level l + 1
    int n = 0, m = 0;
    int max_pos = 0;        // largest feed position (validated by the upload)
    int chain_max_pos = 0;  // largest chain position (not validated: the chain checks it)
};
// Validates the feed rows (mtp_feed_tables_valid), packs the 9 tables into ws.h_pass_tables and uploads
// them with one async H2D on `stream` (4n + 5m ints, m <= n <= feed_rows_cap). Waits for the previous pass's
// copy before reusing the staging.
[[nodiscard]] bool mtp_head_pass_upload(MtpDraftWorkspace& ws, const int32_t* toks, const int* src,
                                        const int* slots, const int* pos, int n, const int* last,
                                        const int* chain_slots, const int* chain_pos, int m,
                                        cudaStream_t stream, MtpPassTables& out);
// A captured head pass sizes its scan scores for the largest position + 1, rounded up to this many rows and
// clamped to max_seq_len: smem only, the kernel reads pos[r] + 1 scores per row (numerics unchanged).
inline constexpr int kMtpScanSmemRowStep = 256;
constexpr int mtp_scan_smem_rows(int max_pos, int max_seq_len) {
    const int rows = ((max_pos + kMtpScanSmemRowStep) / kMtpScanSmemRowStep) * kMtpScanSmemRowStep;
    return max_seq_len > 0 && rows > max_seq_len ? max_seq_len : rows;
}
// The scan kernel nodes (mtp_attn_scan_slots) of a captured head pass and their captured params; the params'
// kernelParams point into `graph`, which must outlive them.
struct MtpScanNodes {
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
};
// Every scan kernel node of `graph` into out; false = a graph API call failed.
[[nodiscard]] bool mtp_scan_nodes_find(cudaGraph_t graph, MtpScanNodes& out);
// Sets every scan node of `exec` to `rows` scores of dynamic smem (mtp_scan_smem_rows; a split node
// mtp_scan_split_chunk(rows)); no-op when unchanged.
[[nodiscard]] bool mtp_scan_nodes_set_rows(cudaGraphExec_t exec, MtpScanNodes& scan, int rows);
// Make `slot` the active KV slot: saves the current slot's position, loads
// the new one, re-points d_k_cache / d_v_cache. No-op for the active slot.
void mtp_select_slot(MtpDraftWorkspace& ws, int slot);
// dst[r] = src[d_idx[r]], rows of `cols` FP16 (device index array).
void mtp_gather_rows(const void* d_src, const int* d_idx, void* d_dst, int cols, int n_rows, cudaStream_t stream);


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
[[nodiscard]] bool mtp_workspace_allocate(MtpDraftWorkspace& ws, int hidden_dim, int vocab_size,
                            int n_experts = 0, int top_k = 0,
                            int expert_d_ff = 0, int shared_d_ff = 0,
                            int num_heads = 0, int num_kv_heads = 0, int head_dim = 0,
                            int max_seq_len = 0, int n_kv_slots = 1, int hc_count = 0, int hc_lowrank = 0);
// the MTP carry scratch after mtp_workspace_allocate: `rows` per KV slot, needs the batched feed
// (rows <= feed_rows_cap) and a plain head. false = none (rows <= 0 too); mtp_workspace_free releases it.
[[nodiscard]] bool mtp_carry_allocate(MtpDraftWorkspace& ws, int rows);
// diagnostics.mtp_prenorm_h feed scratch [rows, hidden] FP16 on a plain head; false = none.
[[nodiscard]] bool mtp_prenorm_allocate(MtpDraftWorkspace& ws, int rows);
// the MTP body FP8 copy after mtp_workspace_allocate: E4M3 rows + per-row scales of fc, q/k/v/o and the
// dense MLP of a plain FP16 head (Qwen3.8-27B: 405 MiB). false = none; mtp_workspace_free releases it.
[[nodiscard]] bool mtp_body_fp8_allocate(const MtpHead& mtp, MtpDraftWorkspace& ws, cudaStream_t stream);
// Bytes mtp_body_fp8_allocate takes (MtpWorkspaceShape::body_fp8_bytes); 0 = not a plain FP16 dense head.
[[nodiscard]] size_t mtp_body_fp8_bytes(const MtpHead& mtp);
void mtp_workspace_free(MtpDraftWorkspace& ws);

// Reset every MTP-side KV cache position (start of new sequences). The K/V
// buffers retain their allocation; only the positions are zeroed (d_slot_hi: mtp_kv_hi_reset).
inline void mtp_kv_reset(MtpDraftWorkspace& ws) {
    ws.mtp_pos = 0;
    for (auto& p : ws.slot_pos) p = 0;
}
// Zero d_slot_hi and slot_hi_bound of `slot` (< 0: every slot) on `stream`: the slot restarts at position 0.
[[nodiscard]] bool mtp_kv_hi_reset(MtpDraftWorkspace& ws, int slot, cudaStream_t stream);
// Start `slot` at position `floor` with no row below it: its window begins there (mtp_ring_floor_hi).
[[nodiscard]] bool mtp_kv_hi_floor(MtpDraftWorkspace& ws, int slot, int floor, cudaStream_t stream);
// The active slot's d_slot_hi entry (null without a head KV).
inline int* mtp_active_hi(const MtpDraftWorkspace& ws) {
    return ws.d_slot_hi != nullptr ? ws.d_slot_hi + ws.cur_slot : nullptr;
}
// Largest position + 1 a slot at `pos` can have written with chain depth k: the committed rows plus a
// rolled-back chain and the batched chain row past them.
constexpr int mtp_slot_hi_bound(int pos, int k) { return pos + std::max(k, kMtpMaxChainK) + 2; }

}  // namespace imp
