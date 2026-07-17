#include "quant/fp8_quant.h"
#include "quant/fp8_utils.cuh"
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include <cfloat>
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// FP8 E4M3 quantization with per-tensor scale factor.
//
// Phase 2A of the imp quantization pipeline.  Builds on the unscaled FP8
// cast utilities in fp8_utils.{h,cu} by adding calibration-based scaling,
// which is essential for preserving accuracy on real model weights.
//
// Workflow:
//   1.  calibrate_fp8_scale()  -- find absmax, compute scale = absmax / 448
//   2.  quantize_fp16_to_fp8_e4m3_scaled()  -- val / scale -> E4M3
//   3.  dequantize_fp8_e4m3_to_fp16()       -- E4M3 * scale -> FP16
//
// E4M3 representable range: [-448, 448]  (max normal: e=14, m=7)
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;
static constexpr int kElemsPerThread = 4;
static constexpr float kFP8E4M3Max = 448.0f;

// float_to_fp8_e4m3() and fp8_e4m3_to_float() are provided by fp8_utils.cuh.
// Used in the #else software fallback paths below.

// ---------------------------------------------------------------------------
// Absmax reduction kernel
// ---------------------------------------------------------------------------

__global__ void absmax_reduce_kernel(const half* __restrict__ input, float* __restrict__ block_maxes, int n) {
    __shared__ float sdata[kBlockSize];

    const int tid = threadIdx.x;
    const int base = (blockIdx.x * blockDim.x + tid) * kElemsPerThread;

    float local_max = 0.0f;

// Vectorised load: process kElemsPerThread elements per thread.
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n) {
            float v = fabsf(__half2float(input[idx]));
            local_max = fmaxf(local_max, v);
        }
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Tree reduction within block.
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_maxes[blockIdx.x] = sdata[0];
    }
}

// Second-level reduction: reduce block_maxes -> single scalar.
__global__ void absmax_final_reduce_kernel(const float* __restrict__ block_maxes, float* __restrict__ result,
                                           int n_blocks) {
    __shared__ float sdata[kBlockSize];

    const int tid = threadIdx.x;
    float local_max = 0.0f;

    for (int i = tid; i < n_blocks; i += kBlockSize) {
        local_max = fmaxf(local_max, block_maxes[i]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// ---------------------------------------------------------------------------
// Fused calibrate+quantize: absmax → scale → quantize, all on device.
// Reads the absmax result from a device pointer, computes scale = absmax/448,
// writes scale to d_scale_out, and quantizes in a single kernel launch.
// ---------------------------------------------------------------------------

__global__ void calibrate_quantize_fp8_kernel(const half* __restrict__ input, uint8_t* __restrict__ output,
                                              const float* __restrict__ d_absmax,  // from absmax reduction
                                              float* __restrict__ d_scale_out,  // output: scale for dequant
                                              int n) {
    float absmax = d_absmax[0];
    float scale = (absmax > 0.0f) ? (absmax / 448.0f) : 1.0f;
    float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_scale_out[0] = scale;
    }

    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n)
        return;

#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n) {
            float val = __half2float(input[idx]) * inv_scale;
            val = fminf(fmaxf(val, -kFP8E4M3Max), kFP8E4M3Max);
            __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(val);
            memcpy(&output[idx], &fp8_val, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Quantize kernel: FP16 / scale -> FP8 E4M3
// ---------------------------------------------------------------------------

__global__ void quantize_fp16_to_fp8_scaled_kernel(const half* __restrict__ input,
                                                   uint8_t* __restrict__ output, int n,
                                                   float inv_scale)  // 1.0f / scale
{
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n)
        return;

#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n) {
            float val = __half2float(input[idx]) * inv_scale;
            // Clamp to E4M3 representable range before cast.
            val = fminf(fmaxf(val, -kFP8E4M3Max), kFP8E4M3Max);
            __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(val);
            memcpy(&output[idx], &fp8_val, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Dequantize kernel: FP8 E4M3 * scale -> FP16
// ---------------------------------------------------------------------------

__global__ void dequantize_fp8_to_fp16_scaled_kernel(const uint8_t* __restrict__ input,
                                                     half* __restrict__ output, int n, float scale) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n)
        return;

#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n) {
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &input[idx], 1);
            float fval = (float)fp8_val * scale;
            output[idx] = __float2half(fval);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

static inline int compute_grid(int n) {
    const int threads_needed = (n + kElemsPerThread - 1) / kElemsPerThread;
    return (threads_needed + kBlockSize - 1) / kBlockSize;
}

// ---- calibrate_fp8_scale --------------------------------------------------

float calibrate_fp8_scale(const Tensor& input, cudaStream_t stream) {
    if (!input.on_device || input.data == nullptr) {
        IMP_LOG_ERROR("calibrate_fp8_scale: input must be a non-null device tensor");
        return 1.0f;
    }

    const int n = (int)input.numel();
    if (n <= 0) {
        IMP_LOG_WARN("calibrate_fp8_scale: empty tensor, returning scale=1.0");
        return 1.0f;
    }

    const int grid = compute_grid(n);

    // Allocate temporary buffer for per-block absmax values + final scalar.
    float* d_block_maxes = nullptr;
    float* d_result = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_maxes, (size_t)grid * sizeof(float)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_result, sizeof(float)));

    // First-level reduction: per-block absmax.
    absmax_reduce_kernel<<<grid, kBlockSize, 0, stream>>>(static_cast<const half*>(input.data), d_block_maxes,
                                                          n);
    IMP_CUDA_CHECK_LAUNCH();

    // Second-level reduction: reduce block results to a single scalar.
    absmax_final_reduce_kernel<<<1, kBlockSize, 0, stream>>>(d_block_maxes, d_result, grid);
    IMP_CUDA_CHECK_LAUNCH();

    // Copy result back to host.
    float absmax = 0.0f;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(&absmax, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    IMP_CUDA_CHECK_LOG(cudaFree(d_block_maxes));
    IMP_CUDA_CHECK_LOG(cudaFree(d_result));

    // Avoid division by zero.
    if (absmax == 0.0f) {
        IMP_LOG_WARN("calibrate_fp8_scale: all-zero tensor, returning scale=1.0");
        return 1.0f;
    }

    float scale = absmax / kFP8E4M3Max;
    IMP_LOG_DEBUG("calibrate_fp8_scale: absmax=%.6f  scale=%.6f", absmax, scale);
    return scale;
}

// ---- calibrate_and_quantize_fp8_async -------------------------------------
// Fully asynchronous: calibrate + quantize with reusable temp buffers.
// No host sync — caller provides pre-allocated d_block_maxes and d_absmax.
// The scale is written to d_scale_out on device.

void calibrate_and_quantize_fp8_async(const void* input_fp16, void* output_fp8, int64_t n_elements,
                                      float* d_block_maxes, int max_grid, float* d_absmax, float* d_scale_out,
                                      cudaStream_t stream) {
    if (!input_fp16 || !output_fp8 || n_elements <= 0)
        return;
    // The reduction/quantize kernels index with int. No current model is close
    // (largest tensor ~778M elems), but guard the boundary loudly instead of
    // silently truncating size_t→int — a >2.1B-element tensor would otherwise
    // corrupt with a wrong grid + wrapped indices (F-A11). The callers pass the
    // full size_t now, so the truncation can only ever happen here.
    if (n_elements > static_cast<int64_t>(INT_MAX)) {
        IMP_LOG_ERROR("calibrate_and_quantize_fp8_async: n_elements=%lld exceeds int range — skipping",
                      static_cast<long long>(n_elements));
        return;
    }
    const int n = static_cast<int>(n_elements);

    const int grid = compute_grid(n);
    const int reduce_grid = (grid <= max_grid) ? grid : max_grid;

    // Pass 1: absmax reduction
    absmax_reduce_kernel<<<reduce_grid, kBlockSize, 0, stream>>>(static_cast<const half*>(input_fp16),
                                                                 d_block_maxes, n);
    IMP_CUDA_CHECK_LAUNCH();

    absmax_final_reduce_kernel<<<1, kBlockSize, 0, stream>>>(d_block_maxes, d_absmax, reduce_grid);
    IMP_CUDA_CHECK_LAUNCH();

    // Pass 2: fused scale computation + quantize (reads absmax from device)
    calibrate_quantize_fp8_kernel<<<grid, kBlockSize, 0, stream>>>(static_cast<const half*>(input_fp16),
                                                                   static_cast<uint8_t*>(output_fp8),
                                                                   d_absmax, d_scale_out, n);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---- quantize_fp8_rows_async ----------------------------------------------
// Per-ROW (per-output-channel) E4M3 quantization: one block per row reduces
// the row absmax, derives scale = absmax/448, records it in d_row_scales[row]
// and quantizes the row with it. Per-row scales avoid the range waste of one
// per-tensor scale across heterogeneous row blocks (e.g. the fused GDN input
// pack [conv | gate | alpha | beta]). Init-time only — the row is read twice
// from L2, which is irrelevant there.

__global__ void quantize_fp8_rows_kernel(const half* __restrict__ in, uint8_t* __restrict__ out,
                                         int K, float* __restrict__ d_row_scales) {
    const int row = blockIdx.x;
    const half* r = in + static_cast<int64_t>(row) * K;
    uint8_t* o = out + static_cast<int64_t>(row) * K;

    float m = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        m = fmaxf(m, fabsf(__half2float(r[i])));
    __shared__ float s_warp[32];
    for (int off = 16; off; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0)
        s_warp[threadIdx.x >> 5] = m;
    __syncthreads();
    const int n_warps = blockDim.x >> 5;
    if (threadIdx.x < 32) {
        m = (threadIdx.x < n_warps) ? s_warp[threadIdx.x] : 0.0f;
        for (int off = 16; off; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    }
    __shared__ float s_scale;
    if (threadIdx.x == 0) {
        float sc = (m > 0.0f) ? m / kFP8E4M3Max : 1.0f;
        s_scale = sc;
        d_row_scales[row] = sc;
    }
    __syncthreads();

    const float inv = 1.0f / s_scale;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        __nv_fp8_e4m3 q(__half2float(r[i]) * inv);
        o[i] = *reinterpret_cast<const uint8_t*>(&q);
    }
}

void quantize_fp8_rows_async(const void* input_fp16, void* output_fp8, int rows, int K,
                             float* d_row_scales, cudaStream_t stream) {
    if (!input_fp16 || !output_fp8 || !d_row_scales || rows <= 0 || K <= 0)
        return;
    quantize_fp8_rows_kernel<<<rows, 256, 0, stream>>>(static_cast<const half*>(input_fp16),
                                                       static_cast<uint8_t*>(output_fp8), K,
                                                       d_row_scales);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---- quantize_fp16_to_fp8_e4m3 (Tensor API) ------------------------------

void quantize_fp16_to_fp8_e4m3(const Tensor& input, Tensor& output, float* d_scale_out, cudaStream_t stream,
                               float* d_block_maxes_ext, float* d_absmax_ext, int max_grid_ext) {
    if (!input.on_device || input.data == nullptr) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: input must be a non-null device tensor");
        return;
    }
    if (!output.on_device || output.data == nullptr) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: output must be a pre-allocated device tensor");
        return;
    }
    if (input.qtype != QType::F16) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: input dtype must be FP16");
        return;
    }

    const int n = (int)input.numel();
    if (n <= 0)
        return;

    if (output.numel() != input.numel()) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: output numel (%lld) != input numel (%lld)",
                      (long long)output.numel(), (long long)input.numel());
        return;
    }

    // Fast path: pre-allocated reduction buffers → fully async (no malloc, no sync)
    if (d_block_maxes_ext && d_absmax_ext && max_grid_ext > 0) {
        calibrate_and_quantize_fp8_async(input.data, output.data, n, d_block_maxes_ext, max_grid_ext,
                                         d_absmax_ext, d_scale_out, stream);
        return;
    }

    // Slow path: allocate temp buffers + host sync (backward compat)
    const int grid = compute_grid(n);

    float* d_block_maxes = nullptr;
    float* d_scale_device = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_block_maxes, (size_t)grid * sizeof(float)));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_scale_device, sizeof(float)));

    absmax_reduce_kernel<<<grid, kBlockSize, 0, stream>>>(static_cast<const half*>(input.data), d_block_maxes,
                                                          n);
    IMP_CUDA_CHECK_LAUNCH();

    absmax_final_reduce_kernel<<<1, kBlockSize, 0, stream>>>(d_block_maxes, d_scale_device, grid);
    IMP_CUDA_CHECK_LAUNCH();

    float absmax = 0.0f;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(&absmax, d_scale_device, sizeof(float), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    float scale = (absmax > 0.0f) ? (absmax / kFP8E4M3Max) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (d_scale_out != nullptr) {
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(d_scale_out, &scale, sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    quantize_fp16_to_fp8_scaled_kernel<<<grid, kBlockSize, 0, stream>>>(static_cast<const half*>(input.data),
                                                                        static_cast<uint8_t*>(output.data), n,
                                                                        inv_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3 kernel launch failed: %s", cudaGetErrorString(err));
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_block_maxes));
    IMP_CUDA_CHECK_LOG(cudaFree(d_scale_device));

    IMP_LOG_DEBUG("quantize_fp16_to_fp8_e4m3: n=%d  scale=%.6f", n, scale);
}

// ---- quantize_fp16_to_fp8_e4m3_scaled (raw pointer API) ------------------

void quantize_fp16_to_fp8_e4m3_scaled(const void* input_fp16, void* output_fp8, int n_elements, float scale,
                                      cudaStream_t stream) {
    if (n_elements <= 0)
        return;
    if (input_fp16 == nullptr || output_fp8 == nullptr) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3_scaled: null pointer");
        return;
    }

    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;

    const int grid = compute_grid(n_elements);

    quantize_fp16_to_fp8_scaled_kernel<<<grid, kBlockSize, 0, stream>>>(static_cast<const half*>(input_fp16),
                                                                        static_cast<uint8_t*>(output_fp8),
                                                                        n_elements, inv_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3_scaled launch failed: %s", cudaGetErrorString(err));
    }
}

// ---- dequantize_fp8_e4m3_to_fp16 (raw pointer API) ----------------------

void dequantize_fp8_e4m3_to_fp16(const void* input_fp8, void* output_fp16, int n_elements, float scale,
                                 cudaStream_t stream) {
    if (n_elements <= 0)
        return;
    if (input_fp8 == nullptr || output_fp16 == nullptr) {
        IMP_LOG_ERROR("dequantize_fp8_e4m3_to_fp16: null pointer");
        return;
    }

    const int grid = compute_grid(n_elements);

    dequantize_fp8_to_fp16_scaled_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const uint8_t*>(input_fp8), static_cast<half*>(output_fp16), n_elements, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("dequantize_fp8_e4m3_to_fp16 launch failed: %s", cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// Per-expert FP8 scale calibration kernel for MoE.
// One block per expert: finds absmax within [offsets[e], offsets[e+1]) × K,
// writes scale = absmax / 448.0.
// ---------------------------------------------------------------------------

__global__ void calibrate_fp8_scales_per_expert_kernel(const half* __restrict__ input,
                                                       const int32_t* __restrict__ offsets,
                                                       float* __restrict__ scales_out, int K) {
    __shared__ float sdata[kBlockSize];

    const int expert = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = offsets[expert];
    const int end = offsets[expert + 1];
    const int n_elems = (end - start) * K;

    const half* expert_base = input + static_cast<int64_t>(start) * K;

    float local_max = 0.0f;
    for (int i = tid; i < n_elems; i += kBlockSize) {
        float v = fabsf(__half2float(expert_base[i]));
        local_max = fmaxf(local_max, v);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float absmax = sdata[0];
        scales_out[expert] = (absmax > 0.0f) ? (absmax / kFP8E4M3Max) : 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Per-expert FP8 quantization kernel for MoE.
// Each block handles one expert's activations with its own scale.
// ---------------------------------------------------------------------------

__global__ void quantize_fp16_to_fp8_per_expert_kernel(const half* __restrict__ input,
                                                       uint8_t* __restrict__ output,
                                                       const int32_t* __restrict__ offsets,
                                                       const float* __restrict__ scales, int K) {
    const int expert = blockIdx.y;
    const int start = offsets[expert];
    const int end = offsets[expert + 1];
    const int n_elems = (end - start) * K;

    if (n_elems == 0)
        return;

    float scale = scales[expert];
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;

    const half* expert_in = input + static_cast<int64_t>(start) * K;
    uint8_t* expert_out = output + static_cast<int64_t>(start) * K;

    int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n_elems)
        return;

#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n_elems) {
            float val = __half2float(expert_in[idx]) * inv_scale;
            val = fminf(fmaxf(val, -kFP8E4M3Max), kFP8E4M3Max);
            __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(val);
            memcpy(&expert_out[idx], &fp8_val, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers for per-expert FP8 operations
// ---------------------------------------------------------------------------

void calibrate_fp8_scales_per_expert(const void* input_fp16, int K, const int32_t* d_offsets, int n_experts,
                                     float* d_scales_out, cudaStream_t stream) {
    if (n_experts <= 0 || !input_fp16 || !d_offsets || !d_scales_out)
        return;

    calibrate_fp8_scales_per_expert_kernel<<<n_experts, kBlockSize, 0, stream>>>(static_cast<const half*>(
                                                                                     input_fp16),
                                                                                 d_offsets, d_scales_out, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("calibrate_fp8_scales_per_expert launch failed: %s", cudaGetErrorString(err));
    }
}

void quantize_fp16_to_fp8_e4m3_per_expert(const void* input_fp16, void* output_fp8, int K,
                                          const int32_t* d_offsets, int n_experts, const float* d_scales,
                                          cudaStream_t stream) {
    if (n_experts <= 0 || !input_fp16 || !output_fp8 || !d_offsets || !d_scales)
        return;

    // Launch with enough blocks per expert for the maximum possible token count.
    // We use a 2D grid: x = blocks within expert, y = expert index.
    // Conservative upper bound: use total token count for grid.x sizing.
    // Each expert's kernel skips work if base >= n_elems for that expert.
    //
    // For efficiency, we estimate max tokens per expert. In the worst case,
    // all tokens go to one expert. We read offsets[n_experts] via the last
    // cudaMemcpy that the caller already did, but here we don't have host
    // offsets. Use a generous grid.x that covers max_tokens * K.
    // A 128-expert model with 4096 tokens: max ~4096 tokens/expert × K.
    // With K=7168, that's 29M elements per expert. Grid.x = 29M/(256*4) = 28K blocks.
    // This is fine — excess blocks return immediately.
    constexpr int kMaxBlocksPerExpert = 32768;
    dim3 grid(kMaxBlocksPerExpert, n_experts);
    quantize_fp16_to_fp8_per_expert_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const half*>(input_fp16), static_cast<uint8_t*>(output_fp8), d_offsets, d_scales, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3_per_expert launch failed: %s", cudaGetErrorString(err));
    }
}

}  // namespace imp
