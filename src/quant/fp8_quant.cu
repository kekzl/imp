#include "quant/fp8_quant.h"
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// ---------------------------------------------------------------------------
// Software FP8 conversion helpers (reused from fp8_utils.cu patterns)
// ---------------------------------------------------------------------------

// FP32 -> FP8 E4M3 with saturation (no Inf in E4M3).
__device__ __forceinline__ uint8_t float_to_fp8_e4m3_sw(float val)
{
    const uint32_t sign = (val < 0.0f) ? 1u : 0u;
    float abs_val = fabsf(val);

    // Clamp to E4M3 max representable.
    if (abs_val > kFP8E4M3Max) abs_val = kFP8E4M3Max;

    // Smallest E4M3 subnormal: 2^(-9) = 1/512
    if (abs_val < (1.0f / 512.0f)) {
        return (uint8_t)(sign << 7);  // flush to zero
    }

    // Extract float fields.
    uint32_t fbits;
    memcpy(&fbits, &abs_val, sizeof(float));
    int f_exp = (int)((fbits >> 23) & 0xFF) - 127;  // unbiased
    uint32_t f_man = fbits & 0x7FFFFF;               // 23-bit mantissa

    int e4 = f_exp + 7;  // bias for E4M3

    uint8_t result;
    if (e4 <= 0) {
        // Subnormal in E4M3 range.
        int shift = 1 - e4;
        uint32_t full_man = (1u << 23) | f_man;
        int right_shift = 20 + shift;
        uint8_t m3;
        if (right_shift >= 32) {
            m3 = 0;
        } else {
            uint32_t shifted = full_man >> right_shift;
            uint32_t remainder = full_man & ((1u << right_shift) - 1);
            uint32_t half_point = 1u << (right_shift - 1);
            if (remainder > half_point ||
                (remainder == half_point && (shifted & 1))) {
                shifted += 1;
            }
            m3 = (uint8_t)(shifted & 0x07);
            if (shifted > 7) {
                // Rounded up into normal range.
                result = (uint8_t)((sign << 7) | (1 << 3) | 0);
                return result;
            }
        }
        result = (uint8_t)((sign << 7) | m3);
    } else if (e4 >= 15) {
        // Overflow -> saturate to max normal (not NaN).
        result = (uint8_t)((sign << 7) | 0x7E);
    } else {
        // Normal value: round 23-bit mantissa to 3-bit.
        uint32_t m3 = (f_man + (1u << 19)) >> 20;
        if (m3 > 7) {
            m3 = 0;
            e4 += 1;
            if (e4 >= 15) {
                result = (uint8_t)((sign << 7) | 0x7E);
                return result;
            }
        }
        result = (uint8_t)((sign << 7) | (e4 << 3) | (m3 & 0x07));
    }
    return result;
}

// FP8 E4M3 -> FP32 software conversion.
__device__ __forceinline__ float fp8_e4m3_to_float_sw(uint8_t x)
{
    const uint32_t sign = (x >> 7) & 1;
    int exp = (int)((x >> 3) & 0x0F);
    uint32_t man = x & 0x07;

    if (exp == 0 && man == 0) return sign ? -0.0f : 0.0f;
    if (exp == 15 && man != 0) return __int_as_float(0x7FC00000);  // NaN

    float val;
    if (exp == 0) {
        // Subnormal: value = 2^(-6) * (m / 8)
        val = ldexpf((float)man / 8.0f, -6);
    } else {
        // Normal: value = 2^(exp-7) * (1 + m/8)
        val = ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    }
    return sign ? -val : val;
}

// ---------------------------------------------------------------------------
// Absmax reduction kernel
// ---------------------------------------------------------------------------

__global__ void absmax_reduce_kernel(
    const half*   __restrict__ input,
    float*        __restrict__ block_maxes,
    int n)
{
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
__global__ void absmax_final_reduce_kernel(
    const float*  __restrict__ block_maxes,
    float*        __restrict__ result,
    int n_blocks)
{
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
// Quantize kernel: FP16 / scale -> FP8 E4M3
// ---------------------------------------------------------------------------

__global__ void quantize_fp16_to_fp8_scaled_kernel(
    const half*   __restrict__ input,
    uint8_t*      __restrict__ output,
    int n,
    float inv_scale)   // 1.0f / scale
{
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n) return;

#if defined(__CUDA_FP8_TYPES_EXIST__)
    // ---- Native FP8 path (CUDA 12+ with __nv_fp8_e4m3) --------------------
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
#else
    // ---- Software fallback -------------------------------------------------
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n) {
            float val = __half2float(input[idx]) * inv_scale;
            output[idx] = float_to_fp8_e4m3_sw(val);
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Dequantize kernel: FP8 E4M3 * scale -> FP16
// ---------------------------------------------------------------------------

__global__ void dequantize_fp8_to_fp16_scaled_kernel(
    const uint8_t* __restrict__ input,
    half*          __restrict__ output,
    int n,
    float scale)
{
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n) return;

#if defined(__CUDA_FP8_TYPES_EXIST__)
    // ---- Native FP8 path ---------------------------------------------------
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
#else
    // ---- Software fallback -------------------------------------------------
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n) {
            float fval = fp8_e4m3_to_float_sw(input[idx]) * scale;
            output[idx] = __float2half(fval);
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Quantize + write-scale kernel (for the Tensor-based API)
// ---------------------------------------------------------------------------

__global__ void quantize_fp16_to_fp8_with_scale_kernel(
    const half*   __restrict__ input,
    uint8_t*      __restrict__ output,
    const float*  __restrict__ d_scale,      // device-side scale
    float*        __restrict__ d_scale_out,  // copy scale to output
    int n)
{
    // Read the scale computed by calibration.
    float scale = d_scale[0];
    if (threadIdx.x == 0 && blockIdx.x == 0 && d_scale_out != nullptr) {
        d_scale_out[0] = scale;
    }

    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;

    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n) return;

#if defined(__CUDA_FP8_TYPES_EXIST__)
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
#else
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n) {
            float val = __half2float(input[idx]) * inv_scale;
            output[idx] = float_to_fp8_e4m3_sw(val);
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers
// ---------------------------------------------------------------------------

static inline int compute_grid(int n)
{
    const int threads_needed = (n + kElemsPerThread - 1) / kElemsPerThread;
    return (threads_needed + kBlockSize - 1) / kBlockSize;
}

// ---- calibrate_fp8_scale --------------------------------------------------

float calibrate_fp8_scale(const Tensor& input, cudaStream_t stream)
{
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
    cudaMalloc(&d_block_maxes, (size_t)grid * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    // First-level reduction: per-block absmax.
    absmax_reduce_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const half*>(input.data),
        d_block_maxes,
        n);

    // Second-level reduction: reduce block results to a single scalar.
    absmax_final_reduce_kernel<<<1, kBlockSize, 0, stream>>>(
        d_block_maxes,
        d_result,
        grid);

    // Copy result back to host.
    float absmax = 0.0f;
    cudaMemcpyAsync(&absmax, d_result, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_block_maxes);
    cudaFree(d_result);

    // Avoid division by zero.
    if (absmax == 0.0f) {
        IMP_LOG_WARN("calibrate_fp8_scale: all-zero tensor, returning scale=1.0");
        return 1.0f;
    }

    float scale = absmax / kFP8E4M3Max;
    IMP_LOG_DEBUG("calibrate_fp8_scale: absmax=%.6f  scale=%.6f", absmax, scale);
    return scale;
}

// ---- quantize_fp16_to_fp8_e4m3 (Tensor API) ------------------------------

void quantize_fp16_to_fp8_e4m3(const Tensor& input, Tensor& output,
                                float* d_scale_out,
                                cudaStream_t stream)
{
    if (!input.on_device || input.data == nullptr) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: input must be a non-null device tensor");
        return;
    }
    if (!output.on_device || output.data == nullptr) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: output must be a pre-allocated device tensor");
        return;
    }
    if (input.dtype != DType::FP16) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: input dtype must be FP16");
        return;
    }

    const int n = (int)input.numel();
    if (n <= 0) return;

    if (output.numel() != input.numel()) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3: output numel (%lld) != input numel (%lld)",
                      (long long)output.numel(), (long long)input.numel());
        return;
    }

    // Step 1: Calibrate scale via absmax reduction.
    const int grid = compute_grid(n);

    float* d_block_maxes = nullptr;
    float* d_scale_device = nullptr;
    cudaMalloc(&d_block_maxes, (size_t)grid * sizeof(float));
    cudaMalloc(&d_scale_device, sizeof(float));

    absmax_reduce_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const half*>(input.data),
        d_block_maxes,
        n);

    absmax_final_reduce_kernel<<<1, kBlockSize, 0, stream>>>(
        d_block_maxes,
        d_scale_device,
        grid);

    // d_scale_device currently holds absmax; compute scale = absmax / 448.
    // We do this on device via a small single-thread kernel inline.
    // For simplicity, use the combined quantize kernel that reads the scale.

    // To compute scale = absmax / 448 on device, we use a tiny lambda-like kernel.
    // Alternatively, we memcpy to host, compute, and push back. For streaming
    // correctness we do host round-trip with stream sync.
    float absmax = 0.0f;
    cudaMemcpyAsync(&absmax, d_scale_device, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float scale = (absmax > 0.0f) ? (absmax / kFP8E4M3Max) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Write scale to caller's device pointer.
    if (d_scale_out != nullptr) {
        cudaMemcpyAsync(d_scale_out, &scale, sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    // Step 2: Quantize.
    quantize_fp16_to_fp8_scaled_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const half*>(input.data),
        static_cast<uint8_t*>(output.data),
        n,
        inv_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3 kernel launch failed: %s",
                      cudaGetErrorString(err));
    }

    cudaFree(d_block_maxes);
    cudaFree(d_scale_device);

    IMP_LOG_DEBUG("quantize_fp16_to_fp8_e4m3: n=%d  scale=%.6f", n, scale);
}

// ---- quantize_fp16_to_fp8_e4m3_scaled (raw pointer API) ------------------

void quantize_fp16_to_fp8_e4m3_scaled(const void* input_fp16, void* output_fp8,
                                       int n_elements, float scale,
                                       cudaStream_t stream)
{
    if (n_elements <= 0) return;
    if (input_fp16 == nullptr || output_fp8 == nullptr) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3_scaled: null pointer");
        return;
    }

    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;

    const int grid = compute_grid(n_elements);

    quantize_fp16_to_fp8_scaled_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const half*>(input_fp16),
        static_cast<uint8_t*>(output_fp8),
        n_elements,
        inv_scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3_scaled launch failed: %s",
                      cudaGetErrorString(err));
    }
}

// ---- dequantize_fp8_e4m3_to_fp16 (raw pointer API) ----------------------

void dequantize_fp8_e4m3_to_fp16(const void* input_fp8, void* output_fp16,
                                  int n_elements, float scale,
                                  cudaStream_t stream)
{
    if (n_elements <= 0) return;
    if (input_fp8 == nullptr || output_fp16 == nullptr) {
        IMP_LOG_ERROR("dequantize_fp8_e4m3_to_fp16: null pointer");
        return;
    }

    const int grid = compute_grid(n_elements);

    dequantize_fp8_to_fp16_scaled_kernel<<<grid, kBlockSize, 0, stream>>>(
        static_cast<const uint8_t*>(input_fp8),
        static_cast<half*>(output_fp16),
        n_elements,
        scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("dequantize_fp8_e4m3_to_fp16 launch failed: %s",
                      cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// Per-expert FP8 scale calibration kernel for MoE.
// One block per expert: finds absmax within [offsets[e], offsets[e+1]) × K,
// writes scale = absmax / 448.0.
// ---------------------------------------------------------------------------

__global__ void calibrate_fp8_scales_per_expert_kernel(
    const half*    __restrict__ input,
    const int32_t* __restrict__ offsets,
    float*         __restrict__ scales_out,
    int K)
{
    __shared__ float sdata[kBlockSize];

    const int expert = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = offsets[expert];
    const int end   = offsets[expert + 1];
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

__global__ void quantize_fp16_to_fp8_per_expert_kernel(
    const half*    __restrict__ input,
    uint8_t*       __restrict__ output,
    const int32_t* __restrict__ offsets,
    const float*   __restrict__ scales,
    int K)
{
    const int expert = blockIdx.y;
    const int start = offsets[expert];
    const int end   = offsets[expert + 1];
    const int n_elems = (end - start) * K;

    if (n_elems == 0) return;

    float scale = scales[expert];
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;

    const half* expert_in  = input  + static_cast<int64_t>(start) * K;
    uint8_t*    expert_out = output + static_cast<int64_t>(start) * K;

    int base = (blockIdx.x * blockDim.x + threadIdx.x) * kElemsPerThread;
    if (base >= n_elems) return;

#if defined(__CUDA_FP8_TYPES_EXIST__)
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
#else
    #pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
        int idx = base + i;
        if (idx < n_elems) {
            float val = __half2float(expert_in[idx]) * inv_scale;
            expert_out[idx] = float_to_fp8_e4m3_sw(val);
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Host-side launch wrappers for per-expert FP8 operations
// ---------------------------------------------------------------------------

void calibrate_fp8_scales_per_expert(const void* input_fp16, int K,
                                      const int32_t* d_offsets, int n_experts,
                                      float* d_scales_out,
                                      cudaStream_t stream)
{
    if (n_experts <= 0 || !input_fp16 || !d_offsets || !d_scales_out) return;

    calibrate_fp8_scales_per_expert_kernel<<<n_experts, kBlockSize, 0, stream>>>(
        static_cast<const half*>(input_fp16),
        d_offsets,
        d_scales_out,
        K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("calibrate_fp8_scales_per_expert launch failed: %s",
                      cudaGetErrorString(err));
    }
}

void quantize_fp16_to_fp8_e4m3_per_expert(const void* input_fp16, void* output_fp8,
                                            int K, const int32_t* d_offsets,
                                            int n_experts, const float* d_scales,
                                            cudaStream_t stream)
{
    if (n_experts <= 0 || !input_fp16 || !output_fp8 || !d_offsets || !d_scales) return;

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
        static_cast<const half*>(input_fp16),
        static_cast<uint8_t*>(output_fp8),
        d_offsets,
        d_scales,
        K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("quantize_fp16_to_fp8_e4m3_per_expert launch failed: %s",
                      cudaGetErrorString(err));
    }
}

} // namespace imp

