#pragma once
// Pieces of the MTP draft step shared between mtp_forward.cu and the Qwen4Exp layer
// (mtp_forward_qwen4exp.cu). Not part of the public draft API (mtp_forward.h).

#include "compute/mtp_forward.h"

namespace imp {

// Gated one-row attention over the MTP KV cache: ws.d_input_norm -> ws.d_attn_residual, appends
// this row's K/V at ws.mtp_pos and advances it.
[[nodiscard]] bool mtp_attention_row(const MtpHead& mtp, MtpDraftWorkspace& ws, int hidden_dim,
                                     cudaStream_t stream);  // false: a copy failed, draft abandoned

// MTP KV-cache attention (mtp_attn_scan.cu): one CTA of kMtpScanThreads per (query row, head).
constexpr int kMtpScanThreads = 512;
constexpr int kMtpScanMaxHeadDim = 512;  // head_dim % 8 == 0 and <= this (mtp_workspace_allocate)
constexpr int kMtpScanMaxSplits = 32;    // ws.d_scan_part: [num_heads, kMtpScanMaxSplits, head_dim + 2] FP32

// Both scan kernels opted into the device's largest dynamic smem, once per process (a per-workspace
// value is process-wide: a smaller cache would lower it under a larger one). False: attribute call failed.
[[nodiscard]] bool mtp_scan_smem_opt_in();

// head_dim % 8 == 0 and <= kMtpScanMaxHeadDim, smem opted in; false logs which (mtp_workspace_allocate).
[[nodiscard]] bool mtp_attn_scan_supported(int head_dim);

// FP16: query row r of q [n_rows, num_heads, head_dim] attends cache rows [0, base_ctx + r + 1) of
// k/v [*, num_kv_heads, head_dim] -> out [n_rows, num_heads, head_dim]. split_part (ws.d_scan_part):
// one row over > 512 cache rows runs split-K.
void mtp_attn_scan(const void* q, const void* k_cache, const void* v_cache, void* out, int base_ctx,
                   int n_rows, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream,
                   float* split_part = nullptr);

// Multi-slot rows: row r attends [0, d_pos[r] + 1) of slot d_slots[r] (k/v base + slot * slot_elems);
// max_pos = max over d_pos sizes the scores.
void mtp_attn_scan_slots(const void* q, const void* k_base, const void* v_base, void* out, const int* d_slots,
                         const int* d_pos, int64_t slot_elems, int max_pos, int n_rows, int num_heads,
                         int num_kv_heads, int head_dim, cudaStream_t stream);

// lm_head over ws.d_h_final, then argmax / top-W into the caller's slot (mtp_draft_step contract).
[[nodiscard]] bool mtp_emit_token(MtpDraftWorkspace& ws, const Tensor& main_lm_head, int hidden_dim, int vocab_size,
                    int* out_token_id, cudaStream_t stream, int* out_topk_ids, int top_w,
                    const NvFP4QuantResult* lm_head_nvfp4, int32_t* d_out_token, const void* lm_head_fp8,
                    const float* lm_head_fp8_scales);

// Qwen4Exp draft layer (docs/plans/2026-09-28-qwen4exp-mtp.md, steps 1-9). In: the token
// embedding in ws.d_fc_in, d_h_prev [hc_count * hidden]. Out: sample_hidden in ws.d_h_final,
// multi_hidden in ws.d_hc_x (the next step's h_prev). d_h_prev may alias ws.d_hc_x.
[[nodiscard]] bool mtp_qwen4exp_layer(const void* d_h_prev, const MtpHead& mtp, MtpDraftWorkspace& ws, int hidden_dim,
                        cudaStream_t stream);

}  // namespace imp
