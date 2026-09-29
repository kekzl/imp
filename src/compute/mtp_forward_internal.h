#pragma once
// Pieces of the MTP draft step shared between mtp_forward.cu and the Qwen4Exp layer
// (mtp_forward_qwen4exp.cu). Not part of the public draft API (mtp_forward.h).

#include "compute/mtp_forward.h"

namespace imp {

// Gated one-row attention over the MTP KV cache: ws.d_input_norm -> ws.d_attn_residual, appends
// this row's K/V at ws.mtp_pos and advances it.
void mtp_attention_row(const MtpHead& mtp, MtpDraftWorkspace& ws, int hidden_dim, cudaStream_t stream);

// lm_head over ws.d_h_final, then argmax / top-W into the caller's slot (mtp_draft_step contract).
bool mtp_emit_token(MtpDraftWorkspace& ws, const Tensor& main_lm_head, int hidden_dim, int vocab_size,
                    int* out_token_id, cudaStream_t stream, int* out_topk_ids, int top_w,
                    const NvFP4QuantResult* lm_head_nvfp4, int32_t* d_out_token, const void* lm_head_fp8,
                    const float* lm_head_fp8_scales);

// Qwen4Exp draft layer (docs/plans/2026-09-28-qwen4exp-mtp.md, steps 1-9). In: the token
// embedding in ws.d_fc_in, d_h_prev [hc_count * hidden]. Out: sample_hidden in ws.d_h_final,
// multi_hidden in ws.d_hc_x (the next step's h_prev). d_h_prev may alias ws.d_hc_x.
bool mtp_qwen4exp_layer(const void* d_h_prev, const MtpHead& mtp, MtpDraftWorkspace& ws, int hidden_dim,
                        cudaStream_t stream);

}  // namespace imp
