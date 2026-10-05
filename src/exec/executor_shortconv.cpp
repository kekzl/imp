#include "compute/layernorm.h"
#include "compute/shortconv.h"
#include "core/cuda_errors.h"
#include "core/dispatch_policy.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/executor_kernels.h"
#include "exec/gemm_context.h"

#include <stdexcept>

namespace imp {

void GraphExecutor::run_ssm_layer_(int layer, const InferenceState& state, cudaStream_t stream) {
    if (model_->config().ssm_short_conv)
        run_shortconv(layer, state, stream);
    else
        run_ssm(layer, state, stream);
}

// LFM2 short-conv layer: h += out_proj(C * conv3(B * x)) with [B | C | x] = in_proj(rmsnorm(h)).
// The conv window lives in the SSM pool (inner = hidden, kernel = conv_L_cache).
void GraphExecutor::run_shortconv(int layer, const InferenceState& state, cudaStream_t stream) {
    if (state.ragged_prefill())
        throw std::runtime_error("run_shortconv: ragged cross-sequence prefill is not supported on LFM2");
    configure_ssm_workspace(ws_.shared_max_tokens());
    const auto& cfg = model_->config();
    const auto& ly = model_->layer(layer);
    const int n = state.n_tokens;
    const int hidden = cfg.ssm_inner_size;
    Tensor h = view_tokens(hidden_, n);
    Tensor r = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(r.data, h.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
    rmsnorm_for_smallm_(h, ly.attn_norm, no, ly.ssm_in_id, n, cfg.rms_norm_eps, stream, norm_w_off_);
    auto ctx = GemmContext::make(stream, wcache_, qscratch_, dispatch_policy(), cur_force_fp16_,
                                 cfg.overrides.gemma4.force_mmvq, cur_spec_verify_);
    Tensor bcx = view_tokens(ssm_proj_buf_, n);  // [n, 3 * hidden]
    gemm_via_handle_(ly.ssm_in_id, no, bcx, ctx);

    const int ssm_idx = ssm_layer_map_[layer];
    if (!state.ssm_state || ssm_idx < 0)
        throw std::runtime_error("run_shortconv: no conv window for this layer");
    Tensor y = view_tokens(ssm_y_buf_, n);  // [n, hidden]
    const bool batched = state.ssm_seq_slots != nullptr && !state.is_prefill;
    static bool logged_batched = false;
    if (batched && !logged_batched) {
        logged_batched = true;
        IMP_LOG_INFO("shortconv: batched decode over %d sequences (slot table)", n);
    }
    void* win = batched ? state.ssm_state->conv_state(0, ssm_idx)
                        : state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx);
    shortconv_forward(win, batched ? state.ssm_seq_slots : nullptr,
                      static_cast<int64_t>(state.ssm_state->slot_stride_bytes()),
                      static_cast<const half*>(bcx.data), static_cast<const half*>(ly.ssm_conv1d_w.data),
                      static_cast<half*>(y.data), batched ? n : 1, batched ? 1 : n, hidden,
                      cfg.ssm_conv_kernel, stream);

    if (residual_beta1_nvfp4_ok_(ly.ssm_out_id, n, h)) {
        gemm_via_handle_(ly.ssm_out_id, y, h, ctx.with_beta(1.0f));
        return;
    }
    Tensor out_buf = view_tokens(ssm_out_buf_, n);
    gemm_via_handle_(ly.ssm_out_id, y, out_buf, ctx);
    elementwise_add(out_buf, r, stream);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h.data, out_buf.data, h.nbytes(), cudaMemcpyDeviceToDevice, stream));
}

}  // namespace imp
