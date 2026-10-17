// Control vector add (roadmap row 114): h[r, c] += scale * v_l[c] after layer l, every row.
// One launch per layer that carries a direction; the FP32 residual shadow gets the same add.

#include "exec/executor.h"
#include "core/logging.h"
#include "lora/control_vector.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace imp {

__global__ void control_vector_add_fp16_kernel(half* __restrict__ h, const float* __restrict__ v, float scale,
                                               int d_model, int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    h[i] = __float2half(__half2float(h[i]) + scale * v[i % d_model]);
}

__global__ void control_vector_add_fp32_kernel(float* __restrict__ h, const float* __restrict__ v,
                                               float scale, int d_model, int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    h[i] += scale * v[i % d_model];
}

void GraphExecutor::set_control_vector(const ControlVectorDevice* cv, float scale) {
    cvec_ = cv;
    cvec_scale_ = scale;
}

void GraphExecutor::control_vector_add_(int layer, int n, cudaStream_t stream) {
    const int d = cvec_->d_model;
    const float* v = cvec_->dirs.get() + static_cast<int64_t>(layer) * d;
    const int64_t total = static_cast<int64_t>(n) * d;
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    control_vector_add_fp16_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<half*>(
                                                                        view_tokens(hidden_, n).data),
                                                                    v, cvec_scale_, d, total);
    IMP_CUDA_CHECK_LAUNCH();
    // Post-norm models (Gemma-3 too, not only fp32_residual_norms Gemma-4) carry the residual in
    // fp32_hidden_: an edit of the FP16 copy alone is overwritten by the next residual add.
    if (fp32_accum_buf_) {
        control_vector_add_fp32_kernel<<<blocks, kThreads, 0, stream>>>(
            static_cast<float*>(view_tokens(fp32_hidden_, n).data), v, cvec_scale_, d, total);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace imp
