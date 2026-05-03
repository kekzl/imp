#include "quant/nvfp4_quant.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_utils.cuh"
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
// NVFP4 (FP4 E2M1) quantization with two-level scaling.
//
// Phase 4 of the imp quantization pipeline.  Implements NVIDIA's FP4 format
// used in Blackwell (SM100) with software emulation for earlier architectures.
//
// FP4 E2M1 format: 1 sign | 2 exponent | 1 mantissa, bias = 1
//   Representable magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
//
// Two-level scaling scheme:
//   Level 1 (tensor scale):  global_absmax / 6.0  (FP32)
//   Level 2 (micro scale):   local_absmax / (tensor_scale * 6.0)  (FP8 E4M3)
//   Quantized value:          val / (tensor_scale * micro_scale) -> FP4 E2M1
//
// Packed format: 2 FP4 values per byte.
//   Low nibble  (bits 0-3) = even-indexed element
//   High nibble (bits 4-7) = odd-indexed element
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;
static constexpr int kMicroBlockSize = 16;  // micro-block: 16 values
static constexpr float kFP4E2M1Max = 6.0f;  // max representable in FP4 E2M1
static constexpr float kFP8E4M3Max = 448.0f;

// ---------------------------------------------------------------------------
// FP4 E2M1 lookup table (unsigned magnitudes, indexed by 3-bit code 0..7)
// ---------------------------------------------------------------------------
//   code  exp(2-bit)  man(1-bit)   value
//     0      00          0         0.0   (zero)
//     1      00          1         0.5   (subnormal: 0.mantissa * 2^(1-bias) = 0.1 * 2^0)
//     2      01          0         1.0   (1.0 * 2^(1-1))
//     3      01          1         1.5   (1.1 * 2^0)
//     4      10          0         2.0   (1.0 * 2^1)
//     5      10          1         3.0   (1.1 * 2^1)
//     6      11          0         4.0   (1.0 * 2^2)
//     7      11          1         6.0   (1.1 * 2^2)

__constant__ float kFP4E2M1Dequant[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ---------------------------------------------------------------------------
// Device helper: quantize a single FP32 magnitude to FP4 E2M1 (3-bit code)
// Uses round-to-nearest-even among the 8 representable magnitudes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t float_abs_to_fp4_e2m1(float abs_val) {
    // Branchless: count of midpoint thresholds exceeded gives the E2M1 code.
    // Thresholds between adjacent representable values:
    //   0    0.5    1.0    1.5    2.0    3.0    4.0    6.0
    //     0.25  0.75  1.25  1.75  2.5   3.5    5.0
    uint8_t code = (abs_val >= 0.25f) + (abs_val >= 0.75f) + (abs_val >= 1.25f) + (abs_val >= 1.75f) +
                   (abs_val >= 2.5f) + (abs_val >= 3.5f) + (abs_val >= 5.0f);
    return code;  // 0..7
}

// HW FP32 pair → packed E2M1 byte (low = v0, high = v1). IEEE RNE rounding,
// saturates to ±6. Single PTX instruction on sm_120+.
__device__ __forceinline__ uint8_t nvfp4_pack_pair_hw(float v0, float v1) {
#if __CUDA_ARCH__ >= 1200
    uint32_t out;
    asm volatile(
        "{ .reg .b8 b;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "  cvt.u32.u8 %0, b; }\n"
        : "=r"(out)
        : "f"(v0), "f"(v1));
    return static_cast<uint8_t>(out);
#else
    uint8_t sign0 = (v0 < 0.0f) ? 1u : 0u;
    uint8_t sign1 = (v1 < 0.0f) ? 1u : 0u;
    uint8_t c0 = (sign0 << 3) | float_abs_to_fp4_e2m1(fabsf(v0));
    uint8_t c1 = (sign1 << 3) | float_abs_to_fp4_e2m1(fabsf(v1));
    return (c1 << 4) | c0;
#endif
}

// float_to_fp8_e4m3() and fp8_e4m3_to_float() are provided by fp8_utils.cuh.

// ---------------------------------------------------------------------------
// Device helper: quantize one micro-block (16 FP16 values) to NVFP4.
//
// Loads 16 values from `input + base`, computes the micro-scale via
// two-level scaling, writes the packed FP4 nibbles and FP8 micro-scale.
//
// Shared by quantize_nvfp4_kernel and quantize_nvfp4_from_absmax_kernel.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void quantize_micro_block_nvfp4(const half* __restrict__ input,
                                                           uint8_t* __restrict__ packed_out,
                                                           uint8_t* __restrict__ micro_scales,
                                                           float tensor_scale,
                                                           int64_t base,  // first element index in input
                                                           int64_t row, int64_t col_mb,
                                                           int64_t num_mb_per_row, int64_t K) {
    // Step 1: Load 16 FP16 values via vectorized half2 loads and find local absmax.
    float vals[kMicroBlockSize];
    float local_absmax = 0.0f;

    const half2* src_h2 = reinterpret_cast<const half2*>(input + base);
#pragma unroll
    for (int i = 0; i < kMicroBlockSize / 2; i++) {
        half2 h2 = src_h2[i];
        vals[i * 2] = __half2float(h2.x);
        vals[i * 2 + 1] = __half2float(h2.y);
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(vals[i * 2]), fabsf(vals[i * 2 + 1])));
    }

    // Step 2: Compute micro-scale = local_absmax / (tensor_scale * 6.0).
    // Clamp to avoid division by zero and FP8 representable range.
    float micro_scale_f = local_absmax / (tensor_scale * kFP4E2M1Max);
    if (micro_scale_f < 1.0f / 512.0f)
        micro_scale_f = 1.0f / 512.0f;  // FP8 E4M3 min subnormal
    if (micro_scale_f > kFP8E4M3Max)
        micro_scale_f = kFP8E4M3Max;

    // Convert micro-scale to FP8 E4M3.
    uint8_t micro_scale_fp8 = float_to_fp8_e4m3(micro_scale_f);

    // Reconstruct the actual micro-scale from FP8 (for quantization consistency).
    float micro_scale_actual = fp8_e4m3_to_float(micro_scale_fp8);
    if (micro_scale_actual == 0.0f)
        micro_scale_actual = 1.0f / 512.0f;

    // Store micro-scale.
    micro_scales[row * num_mb_per_row + col_mb] = micro_scale_fp8;

    // Step 3: Quantize each value to FP4 E2M1 and pack 2 per byte.
    float inv_combined_scale = 1.0f / (tensor_scale * micro_scale_actual);
    int64_t packed_base = row * (K / 2) + col_mb * (kMicroBlockSize / 2);

#pragma unroll
    for (int i = 0; i < kMicroBlockSize; i += 2) {
        float s0 = vals[i] * inv_combined_scale;
        float s1 = vals[i + 1] * inv_combined_scale;
        packed_out[packed_base + i / 2] = nvfp4_pack_pair_hw(s0, s1);
    }
}

// ---------------------------------------------------------------------------
// Kernel: absmax reduction over entire tensor (FP16 input).
// Grid-stride loop, block-level reduction to shared memory, then atomicMax
// on a global counter.  Uses integer atomicMax on the float bit pattern
// (works because absval is non-negative and IEEE754 preserves ordering).
// ---------------------------------------------------------------------------
__global__ void absmax_kernel(const half* __restrict__ input, int64_t n_elements,
                              float* __restrict__ global_max) {
    __shared__ float smem[kBlockSize];

    float local_max = 0.0f;
    int64_t idx = (int64_t)blockIdx.x * kBlockSize + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * kBlockSize;

    for (int64_t i = idx; i < n_elements; i += stride) {
        float v = fabsf(__half2float(input[i]));
        if (v > local_max)
            local_max = v;
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();

    // Block reduction.
    for (int s = kBlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (smem[threadIdx.x + s] > smem[threadIdx.x])
                smem[threadIdx.x] = smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Use integer atomicMax on the bit pattern of a non-negative float.
        unsigned int* ptr = reinterpret_cast<unsigned int*>(global_max);
        unsigned int old_bits = __float_as_uint(smem[0]);
        atomicMax(ptr, old_bits);
    }
}

// ---------------------------------------------------------------------------
// Kernel: quantize FP16 -> NVFP4 with two-level scaling.
// Each thread handles one micro-block of 16 elements.
//
// Thread mapping:
//   global_thread_id = blockIdx.x * blockDim.x + threadIdx.x
//   row  = global_thread_id / num_micro_blocks_per_row
//   col_mb = global_thread_id % num_micro_blocks_per_row
//   first element index = row * K + col_mb * 16
// ---------------------------------------------------------------------------
__global__ void quantize_nvfp4_kernel(const half* __restrict__ input,      // [N, K] FP16
                                      uint8_t* __restrict__ packed_out,    // [N, K/2] packed nibbles
                                      uint8_t* __restrict__ micro_scales,  // [N, K/16] FP8 E4M3
                                      float tensor_scale, int64_t N, int64_t K) {
    const int64_t num_mb_per_row = K / kMicroBlockSize;
    const int64_t total_mb = N * num_mb_per_row;

    int64_t mb_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mb_idx >= total_mb)
        return;

    int64_t row = mb_idx / num_mb_per_row;
    int64_t col_mb = mb_idx % num_mb_per_row;
    int64_t base = row * K + col_mb * kMicroBlockSize;

    quantize_micro_block_nvfp4(input, packed_out, micro_scales, tensor_scale, base, row, col_mb,
                               num_mb_per_row, K);
}

// ---------------------------------------------------------------------------
// Variant that reads tensor_scale from a device pointer (for async pipeline).
// Computes tensor_scale = absmax / kFP4E2M1Max on the fly.
// ---------------------------------------------------------------------------
__global__ void quantize_nvfp4_from_absmax_kernel(
    const half* __restrict__ input, uint8_t* __restrict__ packed_out, uint8_t* __restrict__ micro_scales,
    const float* __restrict__ d_absmax,  // device pointer to absmax value
    float* __restrict__ d_tensor_scale,  // output: tensor_scale for result
    int64_t N, int64_t K) {
    float tensor_scale = d_absmax[0] / kFP4E2M1Max;
    if (tensor_scale == 0.0f)
        tensor_scale = 1.0f;
    if (threadIdx.x == 0 && blockIdx.x == 0 && d_tensor_scale) {
        d_tensor_scale[0] = tensor_scale;
    }

    const int64_t num_mb_per_row = K / kMicroBlockSize;
    const int64_t total_mb = N * num_mb_per_row;

    int64_t mb_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mb_idx >= total_mb)
        return;

    int64_t row = mb_idx / num_mb_per_row;
    int64_t col_mb = mb_idx % num_mb_per_row;
    int64_t base = row * K + col_mb * kMicroBlockSize;

    quantize_micro_block_nvfp4(input, packed_out, micro_scales, tensor_scale, base, row, col_mb,
                               num_mb_per_row, K);
}

// ---------------------------------------------------------------------------
// Kernel: dequantize NVFP4 -> FP16.
// Reverses the two-level scaling.  Each thread handles one micro-block.
// ---------------------------------------------------------------------------
__global__ void dequantize_nvfp4_kernel(const uint8_t* __restrict__ packed_data,   // [N, K/2]
                                        const uint8_t* __restrict__ micro_scales,  // [N, K/16]
                                        float tensor_scale,
                                        half* __restrict__ output,  // [N, K] FP16
                                        int64_t N, int64_t K) {
    const int64_t num_mb_per_row = K / kMicroBlockSize;
    const int64_t total_mb = N * num_mb_per_row;

    int64_t mb_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mb_idx >= total_mb)
        return;

    int64_t row = mb_idx / num_mb_per_row;
    int64_t col_mb = mb_idx % num_mb_per_row;

    // Recover micro-scale.
    uint8_t ms_fp8 = micro_scales[row * num_mb_per_row + col_mb];
    float micro_scale = fp8_e4m3_to_float(ms_fp8);
    float combined_scale = tensor_scale * micro_scale;

    int64_t out_base = row * K + col_mb * kMicroBlockSize;
    int64_t packed_base = row * (K / 2) + col_mb * (kMicroBlockSize / 2);

#pragma unroll
    for (int i = 0; i < kMicroBlockSize; i += 2) {
        uint8_t byte = packed_data[packed_base + i / 2];

        // Low nibble = even element.
        uint8_t fp4_lo = byte & 0x0F;
        uint8_t sign_lo = (fp4_lo >> 3) & 1;
        uint8_t code_lo = fp4_lo & 0x07;
        float val_lo = kFP4E2M1Dequant[code_lo] * combined_scale;
        if (sign_lo)
            val_lo = -val_lo;

        // High nibble = odd element.
        uint8_t fp4_hi = (byte >> 4) & 0x0F;
        uint8_t sign_hi = (fp4_hi >> 3) & 1;
        uint8_t code_hi = fp4_hi & 0x07;
        float val_hi = kFP4E2M1Dequant[code_hi] * combined_scale;
        if (sign_hi)
            val_hi = -val_hi;

        output[out_base + i] = __float2half(val_lo);
        output[out_base + i + 1] = __float2half(val_hi);
    }
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

float calibrate_nvfp4_scales(const Tensor& input, cudaStream_t stream, float* d_reusable_max) {
    assert(input.on_device && "input must be on device");
    assert(input.qtype == QType::F16 && "input must be FP16");

    int64_t n_elements = input.numel();

    // Use caller's reusable buffer or allocate per-call
    float* d_global_max = d_reusable_max;
    bool own_alloc = false;
    if (!d_global_max) {
        IMP_CUDA_CHECK_LOG(cudaMalloc(&d_global_max, sizeof(float)));
        own_alloc = true;
    }
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_global_max, 0, sizeof(float), stream));

    int num_blocks = (int)((n_elements + kBlockSize - 1) / kBlockSize);
    if (num_blocks > 2048)
        num_blocks = 2048;  // cap grid size

    absmax_kernel<<<num_blocks, kBlockSize, 0, stream>>>(reinterpret_cast<const half*>(input.data),
                                                         n_elements, d_global_max);

    // Read back the result.
    float h_absmax = 0.0f;
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(&h_absmax, d_global_max, sizeof(float), cudaMemcpyDeviceToHost, stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

    if (own_alloc)
        IMP_CUDA_CHECK_LOG(cudaFree(d_global_max));

    if (h_absmax == 0.0f) {
        IMP_LOG_WARN("calibrate_nvfp4_scales: tensor is all zeros, using scale 1.0");
        return 1.0f;
    }

    float tensor_scale = h_absmax / kFP4E2M1Max;
    IMP_LOG_DEBUG("calibrate_nvfp4_scales: absmax=%.6f, tensor_scale=%.6f", h_absmax, tensor_scale);
    return tensor_scale;
}

void quantize_fp16_to_nvfp4(const Tensor& input, NvFP4QuantResult& result, cudaStream_t stream) {
    assert(input.on_device && "input must be on device");
    assert(input.qtype == QType::F16 && "input must be FP16");
    assert(input.ndim == 2 && "input must be 2D [N, K]");

    int64_t N = input.shape[0];
    int64_t K = input.shape[1];
    assert(K % kMicroBlockSize == 0 && "K must be multiple of 16");

    // Step 1: Calibrate tensor scale.
    float tensor_scale = calibrate_nvfp4_scales(input, stream);

    // Step 2: Allocate output buffers.
    int64_t packed_bytes = N * (K / 2);
    int64_t micro_scale_bytes = N * (K / kMicroBlockSize);

    uint8_t* d_packed = nullptr;
    uint8_t* d_micro_scales = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_packed, packed_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_micro_scales, micro_scale_bytes));

    // Step 3: Launch quantization kernel.
    int64_t total_micro_blocks = N * (K / kMicroBlockSize);
    int num_blocks = (int)((total_micro_blocks + kBlockSize - 1) / kBlockSize);

    quantize_nvfp4_kernel<<<num_blocks, kBlockSize, 0, stream>>>(reinterpret_cast<const half*>(input.data),
                                                                 d_packed, d_micro_scales, tensor_scale, N,
                                                                 K);

    // Fill result.
    result.packed_data = d_packed;
    result.micro_scales = d_micro_scales;
    result.tensor_scale = tensor_scale;
    result.N = N;
    result.K = K;

    IMP_LOG_DEBUG(
        "quantize_fp16_to_nvfp4: N=%lld K=%lld tensor_scale=%.6f "
        "packed_bytes=%lld micro_scale_bytes=%lld",
        (long long)N, (long long)K, tensor_scale, (long long)packed_bytes, (long long)micro_scale_bytes);
}

void quantize_fp16_to_nvfp4_async(const Tensor& input, NvFP4QuantResult& result, float* d_absmax_buf,
                                  float* d_tensor_scale_buf, cudaStream_t stream) {
    assert(input.on_device && "input must be on device");
    assert(input.qtype == QType::F16 && "input must be FP16");
    assert(input.ndim == 2 && "input must be 2D [N, K]");

    int64_t N = input.shape[0];
    int64_t K = input.shape[1];
    assert(K % kMicroBlockSize == 0 && "K must be multiple of 16");

    // Step 1: absmax reduction (async, no host sync)
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_absmax_buf, 0, sizeof(float), stream));
    int64_t n_elements = input.numel();
    int absmax_blocks = (int)((n_elements + kBlockSize - 1) / kBlockSize);
    if (absmax_blocks > 2048)
        absmax_blocks = 2048;

    absmax_kernel<<<absmax_blocks, kBlockSize, 0, stream>>>(reinterpret_cast<const half*>(input.data),
                                                            n_elements, d_absmax_buf);

    // Step 2: Allocate output buffers
    int64_t packed_bytes = N * (K / 2);
    int64_t micro_scale_bytes = N * (K / kMicroBlockSize);

    uint8_t* d_packed = nullptr;
    uint8_t* d_micro_scales = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_packed, packed_bytes));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_micro_scales, micro_scale_bytes));

    // Step 3: Fused quantize that reads absmax from device (no host sync!)
    int64_t total_micro_blocks = N * (K / kMicroBlockSize);
    int num_blocks = (int)((total_micro_blocks + kBlockSize - 1) / kBlockSize);

    quantize_nvfp4_from_absmax_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const half*>(input.data), d_packed, d_micro_scales, d_absmax_buf, d_tensor_scale_buf,
        N, K);

    result.packed_data = d_packed;
    result.micro_scales = d_micro_scales;
    result.tensor_scale = 0.0f;  // will be read back from d_tensor_scale_buf after sync
    result.N = N;
    result.K = K;
}

void dequantize_nvfp4_to_fp16(const NvFP4QuantResult& quant, void* output_fp16, cudaStream_t stream) {
    assert(quant.packed_data != nullptr && "packed_data must not be null");
    assert(quant.micro_scales != nullptr && "micro_scales must not be null");
    assert(output_fp16 != nullptr && "output buffer must not be null");

    int64_t N = quant.N;
    int64_t K = quant.K;
    int64_t total_micro_blocks = N * (K / kMicroBlockSize);
    int num_blocks = (int)((total_micro_blocks + kBlockSize - 1) / kBlockSize);

    dequantize_nvfp4_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(quant.packed_data),
        reinterpret_cast<const uint8_t*>(quant.micro_scales), quant.tensor_scale,
        reinterpret_cast<half*>(output_fp16), N, K);
}

// ---------------------------------------------------------------------------
// Kernel: dequantize NVFP4 MoE -> FP16 (per-expert tensor scales).
// Same as dequantize_nvfp4_kernel but reads tensor_scale from device array.
// ---------------------------------------------------------------------------
__global__ void dequantize_nvfp4_moe_kernel(const uint8_t* __restrict__ packed_data,
                                            const uint8_t* __restrict__ micro_scales,
                                            const float* __restrict__ tensor_scales,  // [n_experts] on device
                                            half* __restrict__ output,  // [n_experts, N, K] FP16
                                            int n_experts, int64_t N, int64_t K, size_t expert_stride_packed,
                                            size_t expert_stride_ms) {
    const int64_t num_mb_per_row = K / kMicroBlockSize;
    const int64_t mb_per_expert = N * num_mb_per_row;
    const int64_t total_mb = static_cast<int64_t>(n_experts) * mb_per_expert;

    int64_t mb_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mb_idx >= total_mb)
        return;

    int expert = static_cast<int>(mb_idx / mb_per_expert);
    int64_t local_mb = mb_idx % mb_per_expert;
    int64_t row = local_mb / num_mb_per_row;
    int64_t col_mb = local_mb % num_mb_per_row;

    float ts = tensor_scales[expert];
    uint8_t ms_fp8 = micro_scales[expert * expert_stride_ms + row * num_mb_per_row + col_mb];
    float combined_scale = ts * fp8_e4m3_to_float(ms_fp8);

    int64_t out_base = static_cast<int64_t>(expert) * (N * K) + row * K + col_mb * kMicroBlockSize;
    int64_t packed_base = static_cast<int64_t>(expert) * expert_stride_packed + row * (K / 2) +
                          col_mb * (kMicroBlockSize / 2);

#pragma unroll
    for (int i = 0; i < kMicroBlockSize; i += 2) {
        uint8_t byte = packed_data[packed_base + i / 2];

        uint8_t fp4_lo = byte & 0x0F;
        uint8_t sign_lo = (fp4_lo >> 3) & 1;
        float val_lo = kFP4E2M1Dequant[fp4_lo & 0x07] * combined_scale;
        if (sign_lo)
            val_lo = -val_lo;

        uint8_t fp4_hi = (byte >> 4) & 0x0F;
        uint8_t sign_hi = (fp4_hi >> 3) & 1;
        float val_hi = kFP4E2M1Dequant[fp4_hi & 0x07] * combined_scale;
        if (sign_hi)
            val_hi = -val_hi;

        output[out_base + i] = __float2half(val_lo);
        output[out_base + i + 1] = __float2half(val_hi);
    }
}

void dequantize_nvfp4_moe_to_fp16(const NvFP4MoEQuantResult& result, void* output_fp16, cudaStream_t stream) {
    assert(result.packed_data != nullptr);
    assert(result.micro_scales != nullptr);
    assert(result.tensor_scales != nullptr);
    assert(output_fp16 != nullptr);

    int64_t mb_per_expert = result.N * (result.K / kMicroBlockSize);
    int64_t total_mb = static_cast<int64_t>(result.n_experts) * mb_per_expert;
    int num_blocks = static_cast<int>((total_mb + kBlockSize - 1) / kBlockSize);

    dequantize_nvfp4_moe_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(result.packed_data),
        reinterpret_cast<const uint8_t*>(result.micro_scales), result.tensor_scales,
        reinterpret_cast<half*>(output_fp16), result.n_experts, result.N, result.K,
        result.expert_stride_packed, result.expert_stride_ms);
}

void free_nvfp4_result(NvFP4QuantResult& result) {
    if (result.packed_data) {
        IMP_CUDA_CHECK_LOG(cudaFree(result.packed_data));
        result.packed_data = nullptr;
    }
    if (result.micro_scales) {
        IMP_CUDA_CHECK_LOG(cudaFree(result.micro_scales));
        result.micro_scales = nullptr;
    }
    result.tensor_scale = 1.0f;
    result.N = 0;
    result.K = 0;
}

// ---------------------------------------------------------------------------
// MoE per-expert quantization
// ---------------------------------------------------------------------------

void quantize_packed_experts_to_nvfp4(const void* packed_ggml_data, QType qtype, int n_experts, int eff,
                                      int K, void* dequant_scratch, NvFP4MoEQuantResult& result,
                                      cudaStream_t stream) {
    assert(packed_ggml_data && "packed expert data must not be null");
    assert(dequant_scratch && "dequant scratch buffer required");
    assert(K % kMicroBlockSize == 0 && "K must be multiple of 16");

    // Compute per-expert sizes
    size_t expert_packed_bytes = static_cast<size_t>(eff) * (K / 2);
    size_t expert_ms_bytes = static_cast<size_t>(eff) * (K / kMicroBlockSize);
    size_t total_packed = static_cast<size_t>(n_experts) * expert_packed_bytes;
    size_t total_ms = static_cast<size_t>(n_experts) * expert_ms_bytes;

    // Allocate contiguous output buffers
    uint8_t* d_packed = nullptr;
    uint8_t* d_micro_scales = nullptr;
    float* d_tensor_scales = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_packed, total_packed));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_micro_scales, total_ms));
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_tensor_scales, n_experts * sizeof(float)));

    // Compute expert stride in source GGML data
    size_t src_expert_stride = static_cast<size_t>(eff) * qtype_row_bytes(qtype, K);

    // Temporary device buffer for absmax reduction
    float* d_global_max = nullptr;
    IMP_CUDA_CHECK_LOG(cudaMalloc(&d_global_max, sizeof(float)));

    int64_t n_elements = static_cast<int64_t>(eff) * K;
    int64_t total_micro_blocks = static_cast<int64_t>(eff) * (K / kMicroBlockSize);
    int quant_blocks = static_cast<int>((total_micro_blocks + kBlockSize - 1) / kBlockSize);

    for (int e = 0; e < n_experts; e++) {
        const uint8_t* src = static_cast<const uint8_t*>(packed_ggml_data) + e * src_expert_stride;
        half* scratch = static_cast<half*>(dequant_scratch);

        // Step 1: Dequant this expert slice to FP16 scratch
        dequant_gpu(src, scratch, qtype, eff, K, stream);

        // Step 2: Calibrate tensor_scale = absmax / 6.0
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_global_max, 0, sizeof(float), stream));
        int absmax_blocks = static_cast<int>((n_elements + kBlockSize - 1) / kBlockSize);
        if (absmax_blocks > 2048)
            absmax_blocks = 2048;
        absmax_kernel<<<absmax_blocks, kBlockSize, 0, stream>>>(scratch, n_elements, d_global_max);

        float h_absmax = 0.0f;
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(&h_absmax, d_global_max, sizeof(float), cudaMemcpyDeviceToHost, stream));
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));

        float ts = (h_absmax == 0.0f) ? 1.0f : (h_absmax / kFP4E2M1Max);

        // Copy tensor_scale to device array
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(d_tensor_scales + e, &ts, sizeof(float), cudaMemcpyHostToDevice, stream));

        // Step 3: Quantize FP16 scratch -> NVFP4 at expert offset
        quantize_nvfp4_kernel<<<quant_blocks, kBlockSize, 0, stream>>>(scratch,
                                                                       d_packed + e * expert_packed_bytes,
                                                                       d_micro_scales + e * expert_ms_bytes,
                                                                       ts, static_cast<int64_t>(eff),
                                                                       static_cast<int64_t>(K));

        // Must sync before scratch is reused by next expert
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    }

    IMP_CUDA_CHECK_LOG(cudaFree(d_global_max));

    // Fill result
    result.packed_data = d_packed;
    result.micro_scales = d_micro_scales;
    result.tensor_scales = d_tensor_scales;
    result.n_experts = n_experts;
    result.N = eff;
    result.K = K;
    result.expert_stride_packed = expert_packed_bytes;
    result.expert_stride_ms = expert_ms_bytes;

    IMP_LOG_DEBUG(
        "quantize_packed_experts_to_nvfp4: %d experts, eff=%d K=%d, "
        "packed=%.2f MiB, ms=%.2f MiB",
        n_experts, eff, K, total_packed / (1024.0 * 1024.0), total_ms / (1024.0 * 1024.0));
}

void free_nvfp4_moe_result(NvFP4MoEQuantResult& result) {
    if (result.packed_data) {
        IMP_CUDA_CHECK_LOG(cudaFree(result.packed_data));
        result.packed_data = nullptr;
    }
    if (result.micro_scales) {
        IMP_CUDA_CHECK_LOG(cudaFree(result.micro_scales));
        result.micro_scales = nullptr;
    }
    if (result.tensor_scales) {
        IMP_CUDA_CHECK_LOG(cudaFree(result.tensor_scales));
        result.tensor_scales = nullptr;
    }
    result.n_experts = 0;
    result.N = 0;
    result.K = 0;
    result.expert_stride_packed = 0;
    result.expert_stride_ms = 0;
}

}  // namespace imp
