// Kimi Delta Attention layer (Ling-3.0 bailing_hybrid): the GDN slots with a per-key-channel decay.
// norm -> ssm_in (q|k|v) -> short conv + SiLU -> KDA scan (compute/kda.h) -> gated RMSNorm (sigmoid)
// -> ssm_out -> residual. Buffers: proj and conv_f32 (offset 0 at every n) in ssm_proj_buf_, the conv
// reads a copy of proj in ssm_xBC_buf_, f in ssm_xBC_buf_ once the conv is done with it, beta in
// ssm_dt_buf_, gate in ssm_z_buf_, y in ssm_y_buf_.
// Kimi-Linear: softplus decay; NVFP4 q / k / v as three GEMMs copied into the ssm_xBC_buf_ rows
// (kda_split_qkv_); f and gate as rank-128 LoRA pairs, the [n, 128] intermediate in ssm_y_buf_.
// Speculative verify shapes (grouped chunks, state snapshots) are refused: neither model ships an MTP head.

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
#include "exec/workspace_sizes.h"

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
                              int ssm_idx, cudaStream_t stream, bool proj_in_xbc) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    const int n = static_cast<int>(proj.shape[0]);
    const int conv_channels = cfg.ssm_conv_channels();
    const size_t es = dtype_size(compute_dtype_);
    Tensor xbc(ssm_xBC_buf_.data, compute_dtype_, 2, proj.shape, true);
    // conv_f32 overlaps proj from row 0 at every n: read a copy (an n == 1 prefill chunk too).
    if (!proj_in_xbc)
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
        ssm_conv1d_decode_f32_silu(conv_st, xbc, ly.ssm_conv1d_w, ly.ssm_conv1d_b, conv_f32,
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
    a.softplus_gate = cfg.kda_softplus_gate;
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

// q / k / v GEMMs into back-to-back [n, inner] blocks of ssm_proj_buf_, then each block into its column
// range of the [n, conv_channels] rows in ssm_xBC_buf_ (the layout the fused in_proj_qkv GEMM writes).
void GraphExecutor::kda_split_qkv_(int layer, const Tensor& normed, const GemmContext& ctx,
                                   cudaStream_t stream) {
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    const int n = static_cast<int>(normed.shape[0]);
    const int inner = cfg.ssm_inner_size;
    const int conv_channels = cfg.ssm_conv_channels();
    const size_t es = dtype_size(compute_dtype_);
    if (exec_kda_split_qkv_bytes(n, inner, es) > ssm_proj_buf_.nbytes())
        throw std::runtime_error("run_kda: split q/k/v outputs exceed ssm_proj_buf_");
    const TensorID ids[3] = {ly.ssm_in_id, ly.kda_k_proj_id, ly.kda_v_proj_id};
    int64_t shape[2] = {n, inner};
    for (int i = 0; i < 3; ++i) {
        char* blk = static_cast<char*>(ssm_proj_buf_.data) + exec_kda_split_qkv_offset(n, inner, es, i);
        Tensor out(blk, compute_dtype_, 2, shape, true);
        gemm_via_handle_(ids[i], normed, out, ctx);
        IMP_CUDA_CHECK_LOG(
            cudaMemcpy2DAsync(static_cast<char*>(ssm_xBC_buf_.data) + (static_cast<size_t>(i) * inner * es),
                              static_cast<size_t>(conv_channels) * es, blk, static_cast<size_t>(inner) * es,
                              static_cast<size_t>(inner) * es, static_cast<size_t>(n),
                              cudaMemcpyDeviceToDevice, stream));
    }
}

void GraphExecutor::kda_lora_(TensorID a_id, TensorID b_id, const Tensor& normed, Tensor& out,
                              const GemmContext& ctx) {
    const int n = static_cast<int>(normed.shape[0]);
    const int rank = model_->config().kda_lora_rank;
    const size_t es = dtype_size(compute_dtype_);
    if (rank <= 0 || exec_kda_lora_bytes(n, rank, es) > ssm_y_buf_.nbytes())
        throw std::runtime_error("run_kda: LoRA intermediate does not fit ssm_y_buf_");
    int64_t shape[2] = {n, rank};
    Tensor lo(ssm_y_buf_.data, compute_dtype_, 2, shape, true);
    gemm_via_handle_(a_id, normed, lo, ctx);
    gemm_via_handle_(b_id, lo, out, ctx);
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

    // Kimi-Linear: NVFP4 q / k / v (three global scales) as three GEMMs, f and the gate as LoRA pairs.
    const bool split_qkv = ly.kda_k_proj_id != kInvalidTensorID;
    const bool lora = ly.kda_f_a_id != kInvalidTensorID;
    if (split_qkv != (ly.kda_v_proj_id != kInvalidTensorID) || lora != (ly.kda_g_a_id != kInvalidTensorID))
        throw std::runtime_error("run_kda: layer " + std::to_string(layer) +
                                 " has half a Kimi-Linear KDA layout");
    int64_t proj_shape[2] = {n, conv_channels};
    Tensor proj(ssm_proj_buf_.data, compute_dtype_, 2, proj_shape, true);
    if (split_qkv)
        kda_split_qkv_(layer, no, ctx, stream);
    else
        gemm_via_handle_(ly.ssm_in_id, no, proj, ctx);
    int64_t gate_shape[2] = {n, inner};
    Tensor gate(ssm_z_buf_.data, compute_dtype_, 2, gate_shape, true);
    if (lora)
        kda_lora_(ly.kda_g_a_id, ly.gdn_gate_id, no, gate, ctx);
    else
        gemm_via_handle_(ly.gdn_gate_id, no, gate, ctx);
    int64_t beta_shape[2] = {n, n_heads};
    Tensor beta(ssm_dt_buf_.data, compute_dtype_, 2, beta_shape, true);
    gemm_via_handle_(ly.gdn_beta_id, no, beta, ctx);

    float* conv_f32 = reinterpret_cast<float*>(static_cast<char*>(ssm_proj_buf_.data) +
                                               exec_ssm_conv_f32_offset(n, conv_channels, es, /*kda=*/true));
    kda_conv_(layer, state, proj, conv_f32, ssm_idx, stream, /*proj_in_xbc=*/split_qkv);

    // f [n, H*K] reuses ssm_xBC_buf_ ([n, conv_channels] >= [n, inner]) now that the conv read it.
    int64_t f_shape[2] = {n, inner};
    Tensor f(ssm_xBC_buf_.data, compute_dtype_, 2, f_shape, true);
    if (lora)
        kda_lora_(ly.kda_f_a_id, ly.gdn_alpha_id, no, f, ctx);  // its [n, rank] dies before the scan writes y
    else
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
