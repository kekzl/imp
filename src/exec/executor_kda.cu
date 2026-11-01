// Kimi Delta Attention layer (Ling-3.0 bailing_hybrid): the GDN slots with a per-key-channel decay.
// norm -> ssm_in (q|k|v) -> short conv + SiLU -> KDA scan (compute/kda.h) -> gated RMSNorm (sigmoid)
// -> ssm_out -> residual. Buffers: proj/conv_f32 in ssm_proj_buf_ (the GDN layout), f in ssm_xBC_buf_
// once the conv is done with it, beta in ssm_dt_buf_, gate in ssm_z_buf_, y in ssm_y_buf_.
// Speculative verify shapes (grouped chunks, state snapshots) are refused: Ling-3.0 ships no MTP head.

#include "compute/gdn.h"
#include "compute/kda.h"
#include "compute/layernorm.h"
#include "compute/ssm.h"
#include "core/cuda_errors.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_helpers.h"
#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace imp {

void GraphExecutor::run_recurrent_mixer_(int layer, const InferenceState& state, cudaStream_t stream) {
    if (model_->config().gdn_kda)
        run_kda(layer, state, stream);
    else
        run_gdn(layer, state, stream);
}

// conv1d + SiLU of the q|k|v projection into conv_f32 (FP32), committing the conv window.
void GraphExecutor::kda_conv_(int layer, const InferenceState& state, const Tensor& proj, float* conv_f32,
                              int ssm_idx, cudaStream_t stream) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    const int n = static_cast<int>(proj.shape[0]);
    const int conv_channels = cfg.ssm_conv_channels();
    const size_t es = dtype_size(compute_dtype_);
    Tensor xbc(ssm_xBC_buf_.data, compute_dtype_, 2, proj.shape, true);
    if (n > 1)  // conv_f32 overlaps proj from row 0: read a copy
        device_copy_async(xbc.data, proj.data, static_cast<size_t>(n) * conv_channels * es, stream);
    if (state.ragged_prefill()) {
        for (int s = 0; s < state.n_sequences; ++s) {
            const int rb = state.h_seq_offsets[s];
            const int ns = state.h_seq_offsets[s + 1] - rb;
            if (ns <= 0)
                continue;
            int64_t shape[2] = {ns, conv_channels};
            Tensor rows(static_cast<char*>(xbc.data) + (static_cast<size_t>(rb) * conv_channels * es),
                        compute_dtype_, 2, shape, true);
            ssm_conv1d_prefill_f32_silu(state.ssm_state->conv_state(state.h_ssm_slots[s], ssm_idx), rows,
                                        ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                                        conv_f32 + (static_cast<size_t>(rb) * conv_channels),
                                        cfg.ssm_conv_kernel, stream);
        }
        return;
    }
    void* conv_st = state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx);
    if (state.is_prefill) {
        ssm_conv1d_prefill_f32_silu(conv_st, xbc, ly.ssm_conv1d_w, ly.ssm_conv1d_b, conv_f32,
                                    cfg.ssm_conv_kernel, stream, state.d_chunk_len);
    } else if (state.ssm_seq_slots && state.ssm_n_seq > 1) {
        ssm_conv1d_decode_f32_silu_batched(state.ssm_state->conv_state(0, ssm_idx), state.ssm_seq_slots,
                                           static_cast<int64_t>(state.ssm_state->slot_stride_bytes() /
                                                                sizeof(float)),
                                           static_cast<const half*>(xbc.data), ly.ssm_conv1d_w,
                                           ly.ssm_conv1d_b, conv_f32, state.ssm_n_seq, conv_channels,
                                           cfg.ssm_conv_kernel, stream);
    } else {
        ssm_conv1d_decode_f32_silu(conv_st, n == 1 ? proj : xbc, ly.ssm_conv1d_w, ly.ssm_conv1d_b, conv_f32,
                                   cfg.ssm_conv_kernel, stream);
    }
}

// Scan route: ragged prefill and long single-sequence prefill take the chunked scan, decode,
// batched decode, short and padded chunks the recurrent one.
void GraphExecutor::kda_scan_(int layer, const InferenceState& state, KdaScanArgs a, int ssm_idx,
                              cudaStream_t stream) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    a.neg_a = static_cast<const float*>(ly.ssm_a.data);
    a.dt_bias = static_cast<const float*>(ly.ssm_dt_b.data);
    a.lower_bound = cfg.kda_lower_bound;
    a.conv_channels = cfg.ssm_conv_channels();
    a.n_heads = cfg.ssm_dt_rank;
    a.state_bf16 = state.ssm_state->h_dtype() == QType::BF16;
    if (state.ragged_prefill() || (state.ssm_seq_slots && state.ssm_n_seq > 1)) {
        a.h_state = state.ssm_state->h_state(0, ssm_idx);
        a.seq_slots = state.ssm_seq_slots;
        a.seq_stride = static_cast<int64_t>(state.ssm_state->slot_stride_bytes() /
                                            (a.state_bf16 ? sizeof(__nv_bfloat16) : sizeof(float)));
        a.n_seq = state.ssm_n_seq;
        a.seq_row_offsets = state.ragged_prefill() ? state.seq_offsets : nullptr;
        a.n_tokens = state.ragged_prefill() ? 0 : 1;
        if (state.ragged_prefill())
            kda_scan_chunk(a, stream);
        else
            kda_scan_recurrent(a, stream);
        return;
    }
    a.h_state = state.ssm_state->h_state(state.ssm_seq_id, ssm_idx);
    a.d_real_n = state.d_chunk_len;
    if (a.n_tokens >= kKdaChunkMinRows && a.d_real_n == nullptr)
        kda_scan_chunk(a, stream);
    else
        kda_scan_recurrent(a, stream);
}

void GraphExecutor::run_kda(int layer, const InferenceState& state, cudaStream_t stream) {
    configure_ssm_workspace(ws_.shared_max_tokens());
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    const int n = cur_n_tokens_;
    const int ssm_idx = ssm_layer_map_.empty() ? -1 : ssm_layer_map_[layer];
    if (!state.ssm_state || ssm_idx < 0)
        throw std::runtime_error("run_kda: no recurrent state slot for layer " + std::to_string(layer));
    if (state.ssm_grouped_chunk() || state.spec_snap_slab || state.ssm_tap_out || state.ssm_fac_in)
        throw std::runtime_error("run_kda: speculative verify chunks are not supported on KDA layers");
    const int inner = cfg.ssm_inner_size;
    const int conv_channels = cfg.ssm_conv_channels();
    const int n_heads = cfg.ssm_dt_rank;
    const size_t es = dtype_size(compute_dtype_);
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, dispatch_policy(), cur_force_fp16_,
                                 cfg.overrides.gemma4.force_mmvq, cur_spec_verify_);

    Tensor h = view_tokens(hidden_, n);
    Tensor r = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    device_copy_async(r.data, h.data, h.nbytes(), stream);
    rmsnorm_for_smallm_(h, ly.attn_norm, no, ly.ssm_in_id, n, cfg.rms_norm_eps, stream, norm_w_off_);

    int64_t proj_shape[2] = {n, conv_channels};
    Tensor proj(ssm_proj_buf_.data, compute_dtype_, 2, proj_shape, true);
    gemm_via_handle_(ly.ssm_in_id, no, proj, ctx);
    int64_t gate_shape[2] = {n, inner};
    Tensor gate(ssm_z_buf_.data, compute_dtype_, 2, gate_shape, true);
    gemm_via_handle_(ly.gdn_gate_id, no, gate, ctx);
    int64_t beta_shape[2] = {n, n_heads};
    Tensor beta(ssm_dt_buf_.data, compute_dtype_, 2, beta_shape, true);
    gemm_via_handle_(ly.gdn_beta_id, no, beta, ctx);

    // At n == 1 the conv reads proj in place and conv_f32 sits past its row (GDN layout).
    float* conv_f32 = static_cast<float*>(ssm_proj_buf_.data);
    if (n == 1)
        conv_f32 = reinterpret_cast<float*>(static_cast<char*>(ssm_proj_buf_.data) +
                                            align256(static_cast<size_t>(conv_channels) * es));
    kda_conv_(layer, state, proj, conv_f32, ssm_idx, stream);

    // f [n, H*K] reuses ssm_xBC_buf_ ([n, conv_channels] >= [n, inner]) now that the conv read it.
    int64_t f_shape[2] = {n, inner};
    Tensor f(ssm_xBC_buf_.data, compute_dtype_, 2, f_shape, true);
    gemm_via_handle_(ly.gdn_alpha_id, no, f, ctx);

    Tensor y = view_tokens(ssm_y_buf_, n);
    KdaScanArgs a;
    a.conv_f32 = conv_f32;
    a.f = static_cast<const half*>(f.data);
    a.beta = static_cast<const half*>(beta.data);
    a.y = static_cast<half*>(y.data);
    a.n_tokens = n;
    kda_scan_(layer, state, a, ssm_idx, stream);

    gdn_rmsnorm_gated_silu(static_cast<half*>(y.data), static_cast<const half*>(gate.data),
                           static_cast<const half*>(ly.ssm_norm_w.data), cfg.rms_norm_eps, n, n_heads,
                           inner / n_heads, stream, /*sigmoid_gate=*/true);
    Tensor out = view_tokens(ssm_out_buf_, n);
    gemm_via_handle_(ly.ssm_out_id, y, out, ctx);
    elementwise_add(out, r, stream);
    device_copy_async(h.data, out.data, h.nbytes(), stream);
}

}  // namespace imp
