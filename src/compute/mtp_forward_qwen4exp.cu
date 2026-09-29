// Qwen4Exp MTP draft layer (Qwen3.8-Flash-Next). Math and reference line quotes:
// docs/plans/2026-09-28-qwen4exp-mtp.md. One token, FP16 buffers, FP32 arithmetic in the kernels.
//
//   e      = fc_embedding(norm(emb))                          [d]
//   x      = fc_hidden(norm_over_hc*d(h_prev)) per stream + e  [hc, d]   (first combine: unit weight)
//   x     += attn(hc_read_attn(x)) * inj_attn                  per stream
//   x     += moe(hc_read_mlp(x)) * inj_mlp                     per stream -> multi_hidden (next h_prev)
//   sample = hc_mix_final(x)                                   [d] -> lm_head (no final norm)
#include "compute/gated_residual.h"
#include "compute/activation.h"
#include "compute/gemm.h"
#include "compute/gemv_fp8_block_moe.h"
#include "compute/layernorm.h"
#include "compute/moe_routing.h"
#include "compute/mtp_forward_internal.h"
#include "core/logging.h"

#include <cuda_fp16.h>

namespace imp {

namespace {

Tensor row_view(void* p, int64_t rows, int64_t cols) {
    const int64_t shape[2] = {rows, cols};
    return Tensor(p, QType::F16, 2, shape, /*on_device=*/true);
}

// x[s * d + j] += e[j] for every stream s: the layer's first combine (prev_injection = None).
__global__ void mtp_hc_add_broadcast_kernel(__half* __restrict__ x, const __half* __restrict__ e, int hc, int d) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= hc * d)
        return;
    x[t] = __float2half(__half2float(x[t]) + __half2float(e[t % d]));
}

__global__ void mtp_add_rows_kernel(__half* __restrict__ a, const __half* __restrict__ b, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n)
        a[t] = __float2half(__half2float(a[t]) + __half2float(b[t]));
}

// One hyper-connection read of ws.d_hc_x: block input into `out` [d]; inject weights into
// ws.d_hc_inj when the module has a block_inject (final mixer: none).
void hc_read_row(const MtpHyperConnection& m, MtpDraftWorkspace& ws, int d, void* out, cudaStream_t stream) {
    const int hc = ws.hc_count;
    Tensor x = row_view(ws.d_hc_x, 1, static_cast<int64_t>(hc) * d);
    Tensor normed = row_view(ws.d_hc_normed, 1, static_cast<int64_t>(hc) * d);
    Tensor low = row_view(ws.d_hc_low, 1, ws.hc_lowrank);
    Tensor mixw = row_view(ws.d_hc_mixw, 1, static_cast<int64_t>(hc) * d);
    Tensor o = row_view(out, 1, d);
    hc_grouped_rmsnorm(x, m.norm, normed, hc, d, ws.rms_norm_eps, stream);
    gemm(normed, m.mix_down, low, 1.0f, 0.0f, stream);
    hc_silu_div(low, hc, stream);
    gemm(low, m.mix_up, mixw, 1.0f, 0.0f, stream);
    hc_mix(mixw, normed, o, hc, d, stream);
    if (m.block_inject.data != nullptr) {
        Tensor inj = row_view(ws.d_hc_inj, 1, hc);
        gemm(normed, m.block_inject, inj, 1.0f, 0.0f, stream);
        hc_inject_weights(inj, hc, stream);
    }
}

// x[s] += block_out * inj[s].
void hc_combine_row(MtpDraftWorkspace& ws, int d, void* block_out, cudaStream_t stream) {
    Tensor x = row_view(ws.d_hc_x, 1, static_cast<int64_t>(ws.hc_count) * d);
    Tensor o = row_view(block_out, 1, d);
    Tensor inj = row_view(ws.d_hc_inj, 1, ws.hc_count);
    hc_inject_add(x, o, inj, ws.hc_count, d, stream);
}

// Routed FP8 experts + gated shared expert on ws.d_post_norm -> ws.d_moe_out.
bool moe_row(const MtpHead& mtp, MtpDraftWorkspace& ws, int d, cudaStream_t stream) {
    const int top_k = ws.top_k;
    const int eff = ws.expert_d_ff;
    MoeRoutingResult routing{};
    moe_gate_topk_fused(mtp.router.data, ws.d_post_norm, ws.n_experts, d, top_k, ws.routing_buf, routing, stream,
                        /*use_sigmoid=*/false, /*normalize_weights=*/true, /*score_bias=*/nullptr);
    const int32_t* ids = ws.routing_buf.expert_indices;
    Fp8BlockMoeArgs gu;
    gu.w = mtp.fp8_gate_up_tab;
    gu.scales = mtp.fp8_gate_up_scales;
    gu.expert_ids = ids;
    gu.x = static_cast<const half*>(ws.d_post_norm);
    gu.y = static_cast<half*>(ws.d_expert_gate_up);
    gu.rows = eff;
    gu.K = d;
    gu.n_proj = 2;
    gu.top_k = top_k;
    Fp8BlockMoeArgs dn;
    dn.w = mtp.fp8_down_tab;
    dn.scales = mtp.fp8_down_scales;
    dn.expert_ids = ids;
    dn.x = static_cast<const half*>(ws.d_expert_act);
    dn.x_stride = eff;
    dn.y = static_cast<half*>(ws.d_expert_outputs);
    dn.rows = d;
    dn.K = eff;
    dn.top_k = top_k;
    if (!gemv_fp8_block_moe(gu, stream))
        return false;
    swiglu_packed_rows(static_cast<const half*>(ws.d_expert_gate_up), static_cast<half*>(ws.d_expert_act), eff,
                       top_k, stream);
    if (!gemv_fp8_block_moe(dn, stream))
        return false;
    moe_weighted_sum_residual(ws.d_expert_outputs, ws.routing_buf.expert_weights, /*residual=*/nullptr,
                              ws.d_moe_out, d, top_k, stream);

    const int sff = ws.shared_d_ff;
    Tensor in = row_view(ws.d_post_norm, 1, d);
    Tensor sg = row_view(ws.d_shared_gate, 1, sff);
    Tensor su = row_view(ws.d_shared_up, 1, sff);
    Tensor sa = row_view(ws.d_shared_act, 1, sff);
    Tensor so = row_view(ws.d_shared_out, 1, d);
    gemm(in, mtp.shared_expert_gate_proj, sg, 1.0f, 0.0f, stream);
    gemm(in, mtp.shared_expert_up_proj, su, 1.0f, 0.0f, stream);
    swiglu(sg, su, sa, stream);
    gemm(sa, mtp.shared_expert_down_proj, so, 1.0f, 0.0f, stream);
    shared_expert_gate_scale(ws.d_post_norm, mtp.shared_expert_gate.data, ws.d_shared_out, /*n=*/1, d, d, stream);
    mtp_add_rows_kernel<<<(d + 255) / 256, 256, 0, stream>>>(static_cast<__half*>(ws.d_moe_out),
                                                              static_cast<const __half*>(ws.d_shared_out), d);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool mtp_qwen4exp_layer(const void* d_h_prev, const MtpHead& mtp, MtpDraftWorkspace& ws, int hidden_dim,
                        cudaStream_t stream) {
    const int d = hidden_dim;
    const int hc = ws.hc_count;
    if (hc <= 0 || ws.d_hc_x == nullptr || ws.n_experts <= 0 || ws.shared_d_ff <= 0 || ws.num_heads <= 0 ||
        mtp.fp8_gate_up_tab == nullptr || mtp.fp8_down_tab == nullptr || mtp.hc_count != hc) {
        IMP_LOG_ERROR("mtp qwen4exp: workspace or head not set up (hc %d/%d, experts %d, fp8 tables %s)", hc,
                      mtp.hc_count, ws.n_experts, mtp.fp8_gate_up_tab ? "on device" : "missing");
        return false;
    }
    const float eps = ws.rms_norm_eps;
    const int64_t hcd = static_cast<int64_t>(hc) * d;

    // 1. Embedding branch: e = fc_embedding(pre_fc_norm_embedding(emb)) -> d_fc_out.
    Tensor emb = row_view(ws.d_fc_in, 1, d);
    Tensor emb_n = row_view(ws.d_emb_norm, 1, d);
    Tensor e = row_view(ws.d_fc_out, 1, d);
    rmsnorm(emb, mtp.pre_fc_norm_embedding, emb_n, eps, stream);
    gemm(emb_n, mtp.fc_embedding, e, 1.0f, 0.0f, stream);

    // 2. Hidden branch: ONE norm over all hc * d values, then fc_hidden per stream (shared weight).
    Tensor h = row_view(const_cast<void*>(d_h_prev), 1, hcd);
    Tensor hn = row_view(ws.d_hc_normed, 1, hcd);
    rmsnorm(h, mtp.pre_fc_norm_hidden, hn, eps, stream);
    Tensor hn_streams = row_view(ws.d_hc_normed, hc, d);
    Tensor x_streams = row_view(ws.d_hc_x, hc, d);
    gemm(hn_streams, mtp.fc_hidden, x_streams, 1.0f, 0.0f, stream);

    // 3. First combine: e onto every stream with unit weight.
    mtp_hc_add_broadcast_kernel<<<static_cast<int>((hcd + 255) / 256), 256, 0, stream>>>(
        static_cast<__half*>(ws.d_hc_x), static_cast<const __half*>(ws.d_fc_out), hc, d);
    IMP_CUDA_CHECK_LAUNCH();

    // 5-7. Attention block between the attn hc read and its inject.
    hc_read_row(mtp.attn_hc, ws, d, ws.d_input_norm, stream);
    if (!mtp_attention_row(mtp, ws, d, stream))
        return false;
    hc_combine_row(ws, d, ws.d_attn_residual, stream);

    // 7-8. MoE block between the mlp hc read and its inject: x becomes multi_hidden.
    hc_read_row(mtp.mlp_hc, ws, d, ws.d_post_norm, stream);
    if (!moe_row(mtp, ws, d, stream))
        return false;
    hc_combine_row(ws, d, ws.d_moe_out, stream);

    // 9. Final mixer (no inject, no final norm): sample_hidden for the LM head.
    hc_read_row(mtp.final_mixer, ws, d, ws.d_h_final, stream);
    return true;
}

}  // namespace imp
