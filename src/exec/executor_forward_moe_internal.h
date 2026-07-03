#pragma once

#include "core/tensor.h"
#include "compute/activation.h"
#include "compute/ssm.h"
#include "quant/dequant_gpu.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace imp {

// Several MoE prefill paths read routing metadata on the host (D2H + stream
// sync) to size the per-expert GEMMs. Under an active stream capture that
// sync fails SILENTLY (the error was unchecked) and the host reads
// uninitialized offsets — the recorded graph then launches expert GEMMs with
// garbage geometry and dies with `misaligned address` at graph launch (the
// #855 census crash class; root-caused on Nemotron-H NVFP4 in #847). Only
// the CUTLASS 3.x device-args path records cleanly; every host-args path
// must fail the capture loudly instead (same lesson as #858).
inline void moe_host_args_capture_guard(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone)
        throw std::runtime_error(
            "MoE host-args prefill path reads routing on the host — not graph-capturable");
}

__global__ void sanitize_fp16_kernel(__half* __restrict__ data, int64_t n);
__global__ void moe_apply_per_expert_scale_kernel(
    float* __restrict__ weights, const int32_t* __restrict__ indices,
    const __half* __restrict__ scales, int n_weights);

inline void sanitize_fp16(__half* data, int64_t n, cudaStream_t stream) {
    if (n <= 0)
        return;
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    sanitize_fp16_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

inline void apply_expert_activation(void* gate_data, void* up_data, void* swiglu_data, bool non_gated,
                                    int64_t rows, int64_t eff, QType compute_dtype, FFNActivation act_type,
                                    cudaStream_t stream) {
    int64_t act_shape[2] = {rows, eff};
    if (non_gated) {
        Tensor up_t(up_data, compute_dtype, 2, act_shape, true);
        relu_sqr_inplace(up_t, stream);
    } else {
        Tensor g(gate_data, compute_dtype, 2, act_shape, true);
        Tensor u(up_data, compute_dtype, 2, act_shape, true);
        Tensor a(swiglu_data, compute_dtype, 2, act_shape, true);
        if (act_type == FFNActivation::GEGLU)
            geglu(g, u, a, stream);
        else if (act_type == FFNActivation::GPT_OSS_GLU)
            gpt_oss_glu(g, u, a, stream);
        else
            swiglu(g, u, a, stream);
    }
}

inline size_t expert_stride(const Tensor& packed, QType qtype) {
    int64_t rows = packed.shape[1];
    int64_t cols = packed.shape[2];
    return static_cast<size_t>(rows) * qtype_row_bytes(qtype, cols);
}

}  // namespace imp
