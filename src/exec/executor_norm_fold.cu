// RMSNorm fold across an M=1 NVFP4 residual GEMV and the next NVFP4 GEMV (quant/nvfp4_gemm.h).
// The residual producer arms the fold for the norm that follows it; that norm's site takes it
// when its GEMV is fold-aware and otherwise runs rmsnorm() as before (the producer's extra
// output is then overwritten). Slots are zeroed once per M=1 forward.
#include "exec/executor.h"
#include "quant/nvfp4_gemm.h"
#include "core/logging.h"
#include "memory/engine_arena.h"
#include <cuda_runtime.h>

namespace imp {

namespace {
// Two producers per layer; the slack covers a final norm and odd layer shapes.
int norm_fold_slots(int n_layers) { return 2 * n_layers + 4; }
// Kernel, not cudaMemsetAsync: a memset node in imp-server's AsyncGraphLoop conditional body
// faults with an illegal address (degen_suite 8/14 FAIL with memset, 50/50 with this kernel).
__global__ void norm_fold_zero_kernel(unsigned long long* p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = 0ull;
}
}  // namespace

void GraphExecutor::norm_fold_begin_(int n, cudaStream_t stream) {
    norm_fold_slot_ = 0;
    norm_fold_gamma_ = nullptr;
    norm_fold_out_ = nullptr;
    norm_fold_pending_ = nullptr;
    norm_fold_on_ = n == 1 && dispatch_policy().gemm.nvfp4_norm_fold && !calib_ && lora_ == nullptr &&
                    compute_dtype_ == QType::F16;
    if (!norm_fold_on_)
        return;
    const size_t bytes = sizeof(unsigned long long) *
                         static_cast<size_t>(norm_fold_slots(model_->config().n_layers));
    if (norm_fold_ssq_ == nullptr) {
        auto slab = engine_arena().take_bytes(bytes);
        if (slab.empty()) {
            IMP_LOG_WARN("norm fold: %zu B unavailable from the T2 arena, off", bytes);
            norm_fold_on_ = false;
            return;
        }
        norm_fold_ssq_ = reinterpret_cast<unsigned long long*>(slab.data());
    }
    const int n_slots = norm_fold_slots(model_->config().n_layers);
    norm_fold_zero_kernel<<<(n_slots + 255) / 256, 256, 0, stream>>>(norm_fold_ssq_, n_slots);
    IMP_CUDA_CHECK_LAUNCH();
}

NvFP4NormFoldOut GraphExecutor::norm_fold_arm_(int layer, bool after_ffn, const Tensor& no) {
    norm_fold_gamma_ = nullptr;
    norm_fold_out_ = nullptr;
    norm_fold_pending_ = nullptr;
    if (!norm_fold_on_ || norm_fold_slot_ >= norm_fold_slots(model_->config().n_layers))
        return {};
    const int n_layers = model_->config().n_layers;
    const Tensor* gamma = nullptr;
    if (!after_ffn)
        gamma = &ffn_norm_weight_(model_->layer(layer));
    else if (layer + 1 < n_layers)
        gamma = &model_->layer(layer + 1).attn_norm;
    else
        gamma = &model_->output_norm();
    if (gamma->data == nullptr || gamma->qtype != QType::F16 || no.qtype != QType::F16 ||
        gamma->numel() != no.shape[no.ndim - 1])
        return {};
    NvFP4NormFoldOut f;
    f.gamma = static_cast<const half*>(gamma->data);
    f.offset = norm_w_off_;
    f.out = static_cast<half*>(no.data);
    f.ssq = norm_fold_ssq_ + norm_fold_slot_++;
    norm_fold_gamma_ = gamma->data;
    norm_fold_out_ = no.data;
    norm_fold_pending_ = f.ssq;
    return f;
}

NvFP4NormFoldIn GraphExecutor::norm_fold_take_(const Tensor& norm_w, const Tensor& no, const Tensor& h,
                                               float eps) {
    NvFP4NormFoldIn f;
    if (norm_fold_pending_ != nullptr && norm_fold_gamma_ == norm_w.data && norm_fold_out_ == no.data &&
        h.shape[0] == 1) {
        f.ssq = norm_fold_pending_;
        f.d = static_cast<float>(h.shape[h.ndim - 1]);
        f.eps = eps;
    }
    norm_fold_gamma_ = nullptr;
    norm_fold_out_ = nullptr;
    norm_fold_pending_ = nullptr;
    return f;
}

NvFP4NormFoldIn GraphExecutor::norm_fold_or_norm_(bool allow, const Tensor& h, const Tensor& w, Tensor& no,
                                                  TensorID consumer_id, int n, float eps,
                                                  cudaStream_t stream) {
    NvFP4NormFoldIn f = norm_fold_take_(w, no, h, eps);
    if (!allow)
        f = {};
    if (!f.ssq)
        rmsnorm_for_smallm_(h, w, no, consumer_id, n, eps, stream, norm_w_off_);
    return f;
}

}  // namespace imp
