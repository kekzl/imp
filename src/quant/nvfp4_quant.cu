#include "quant/nvfp4_quant.h"
#include "quant/dequant_gpu.h"
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
static constexpr int kMicroBlockSize = 16;   // micro-block: 16 values
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

__constant__ float kFP4E2M1Dequant[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// ---------------------------------------------------------------------------
// Device helper: quantize a single FP32 magnitude to FP4 E2M1 (3-bit code)
// Uses round-to-nearest-even among the 8 representable magnitudes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t float_abs_to_fp4_e2m1(float abs_val)
{
    // Clamp to representable range.
    if (abs_val <= 0.0f) return 0;
    if (abs_val >= 6.0f) return 7;  // saturate to max

    // Thresholds are midpoints between consecutive representable values:
    //   0    0.5    1.0    1.5    2.0    3.0    4.0    6.0
    //     0.25  0.75  1.25  1.75  2.5   3.5    5.0
    if (abs_val < 0.25f)  return 0;   // -> 0.0
    if (abs_val < 0.75f)  return 1;   // -> 0.5
    if (abs_val < 1.25f)  return 2;   // -> 1.0
    if (abs_val < 1.75f)  return 3;   // -> 1.5
    if (abs_val < 2.5f)   return 4;   // -> 2.0
    if (abs_val < 3.5f)   return 5;   // -> 3.0
    if (abs_val < 5.0f)   return 6;   // -> 4.0
    return 7;                          // -> 6.0
}

// ---------------------------------------------------------------------------
// Device helper: FP32 -> FP8 E4M3 software conversion with saturation.
// Reuses the same bit-manipulation approach from fp8_quant.cu.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val)
{
    const uint32_t sign = (val < 0.0f) ? 1u : 0u;
    float abs_val = fabsf(val);

    if (abs_val > kFP8E4M3Max) abs_val = kFP8E4M3Max;  // clamp

    // Smallest E4M3 subnormal: 2^(-9)
    if (abs_val < (1.0f / 512.0f)) {
        return (uint8_t)(sign << 7);
    }

    uint32_t fbits;
    memcpy(&fbits, &abs_val, sizeof(float));
    int f_exp = (int)((fbits >> 23) & 0xFF) - 127;
    uint32_t f_man = fbits & 0x7FFFFF;

    int e4 = f_exp + 7;  // E4M3 bias = 7

    uint8_t result;
    if (e4 <= 0) {
        // Subnormal in E4M3.
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
                result = (uint8_t)((sign << 7) | (1 << 3) | 0);
                return result;
            }
        }
        result = (uint8_t)((sign << 7) | m3);
    } else if (e4 >= 15) {
        // E4M3 has no Inf/NaN; saturate to max (e=14, m=7) = 448.
        result = (uint8_t)((sign << 7) | 0x7E | 0x01);
        result = (uint8_t)((sign << 7) | (14 << 3) | 7);
    } else {
        // Normal.
        uint32_t round_bit = (f_man >> 19) & 1;
        uint32_t sticky = (f_man & 0x7FFFF) ? 1 : 0;
        uint8_t m3 = (uint8_t)((f_man >> 20) & 0x07);
        if (round_bit && (sticky || (m3 & 1))) {
            m3 += 1;
            if (m3 > 7) {
                m3 = 0;
                e4 += 1;
                if (e4 >= 15) {
                    result = (uint8_t)((sign << 7) | (14 << 3) | 7);
                    return result;
                }
            }
        }
        result = (uint8_t)((sign << 7) | ((e4 & 0x0F) << 3) | (m3 & 0x07));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Device helper: FP8 E4M3 -> FP32 software conversion.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits)
{
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0x0F;
    uint32_t man  = bits & 0x07;

    float abs_val;
    if (exp == 0) {
        // Subnormal: value = 0.mantissa * 2^(1 - bias) = man * 2^(-9)
        abs_val = (float)man * (1.0f / 512.0f);
    } else {
        // Normal: value = 1.mantissa * 2^(exp - bias) = (8 + man) * 2^(exp - 10)
        abs_val = (float)(8 + man) * exp2f((float)(exp) - 10.0f);
    }
    return sign ? -abs_val : abs_val;
}

// ---------------------------------------------------------------------------
// Kernel: absmax reduction over entire tensor (FP16 input).
// Grid-stride loop, block-level reduction to shared memory, then atomicMax
// on a global counter.  Uses integer atomicMax on the float bit pattern
// (works because absval is non-negative and IEEE754 preserves ordering).
// ---------------------------------------------------------------------------
__global__ void absmax_kernel(const half* __restrict__ input,
                              int64_t n_elements,
                              float* __restrict__ global_max)
{
    __shared__ float smem[kBlockSize];

    float local_max = 0.0f;
    int64_t idx = (int64_t)blockIdx.x * kBlockSize + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * kBlockSize;

    for (int64_t i = idx; i < n_elements; i += stride) {
        float v = fabsf(__half2float(input[i]));
        if (v > local_max) local_max = v;
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
__global__ void quantize_nvfp4_kernel(
    const half* __restrict__ input,        // [N, K] FP16
    uint8_t*    __restrict__ packed_out,    // [N, K/2] packed nibbles
    uint8_t*    __restrict__ micro_scales,  // [N, K/16] FP8 E4M3
    float                    tensor_scale,
    int64_t N, int64_t K)
{
    const int64_t num_mb_per_row = K / kMicroBlockSize;
    const int64_t total_mb = N * num_mb_per_row;

    int64_t mb_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mb_idx >= total_mb) return;

    int64_t row   = mb_idx / num_mb_per_row;
    int64_t col_mb = mb_idx % num_mb_per_row;
    int64_t base  = row * K + col_mb * kMicroBlockSize;

    // Step 1: Load 16 FP16 values and find local absmax.
    float vals[kMicroBlockSize];
    float local_absmax = 0.0f;

    #pragma unroll
    for (int i = 0; i < kMicroBlockSize; i++) {
        vals[i] = __half2float(input[base + i]);
        float av = fabsf(vals[i]);
        if (av > local_absmax) local_absmax = av;
    }

    // Step 2: Compute micro-scale = local_absmax / (tensor_scale * 6.0).
    // Clamp to avoid division by zero and FP8 representable range.
    float micro_scale_f = local_absmax / (tensor_scale * kFP4E2M1Max);
    if (micro_scale_f < 1.0f / 512.0f) micro_scale_f = 1.0f / 512.0f;  // FP8 E4M3 min subnormal
    if (micro_scale_f > kFP8E4M3Max) micro_scale_f = kFP8E4M3Max;

    // Convert micro-scale to FP8 E4M3.
    uint8_t micro_scale_fp8 = float_to_fp8_e4m3(micro_scale_f);

    // Reconstruct the actual micro-scale from FP8 (for quantization consistency).
    float micro_scale_actual = fp8_e4m3_to_float(micro_scale_fp8);
    if (micro_scale_actual == 0.0f) micro_scale_actual = 1.0f / 512.0f;

    // Store micro-scale.
    micro_scales[row * num_mb_per_row + col_mb] = micro_scale_fp8;

    // Step 3: Quantize each value to FP4 E2M1 and pack 2 per byte.
    float inv_combined_scale = 1.0f / (tensor_scale * micro_scale_actual);
    int64_t packed_base = row * (K / 2) + col_mb * (kMicroBlockSize / 2);

    #pragma unroll
    for (int i = 0; i < kMicroBlockSize; i += 2) {
        // Quantize even element (low nibble).
        float scaled0 = vals[i] * inv_combined_scale;
        uint8_t sign0 = (scaled0 < 0.0f) ? 1u : 0u;
        uint8_t code0 = float_abs_to_fp4_e2m1(fabsf(scaled0));
        uint8_t fp4_0 = (sign0 << 3) | code0;

        // Quantize odd element (high nibble).
        float scaled1 = vals[i + 1] * inv_combined_scale;
        uint8_t sign1 = (scaled1 < 0.0f) ? 1u : 0u;
        uint8_t code1 = float_abs_to_fp4_e2m1(fabsf(scaled1));
        uint8_t fp4_1 = (sign1 << 3) | code1;

        // Pack: low nibble = even, high nibble = odd.
        packed_out[packed_base + i / 2] = (fp4_1 << 4) | fp4_0;
    }
}

// ---------------------------------------------------------------------------
// Kernel: dequantize NVFP4 -> FP16.
// Reverses the two-level scaling.  Each thread handles one micro-block.
// ---------------------------------------------------------------------------
__global__ void dequantize_nvfp4_kernel(
    const uint8_t* __restrict__ packed_data,    // [N, K/2]
    const uint8_t* __restrict__ micro_scales,   // [N, K/16]
    float                       tensor_scale,
    half*          __restrict__ output,          // [N, K] FP16
    int64_t N, int64_t K)
{
    const int64_t num_mb_per_row = K / kMicroBlockSize;
    const int64_t total_mb = N * num_mb_per_row;

    int64_t mb_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (mb_idx >= total_mb) return;

    int64_t row    = mb_idx / num_mb_per_row;
    int64_t col_mb = mb_idx % num_mb_per_row;

    // Recover micro-scale.
    uint8_t ms_fp8 = micro_scales[row * num_mb_per_row + col_mb];
    float micro_scale = fp8_e4m3_to_float(ms_fp8);
    float combined_scale = tensor_scale * micro_scale;

    int64_t out_base    = row * K + col_mb * kMicroBlockSize;
    int64_t packed_base = row * (K / 2) + col_mb * (kMicroBlockSize / 2);

    #pragma unroll
    for (int i = 0; i < kMicroBlockSize; i += 2) {
        uint8_t byte = packed_data[packed_base + i / 2];

        // Low nibble = even element.
        uint8_t fp4_lo = byte & 0x0F;
        uint8_t sign_lo = (fp4_lo >> 3) & 1;
        uint8_t code_lo = fp4_lo & 0x07;
        float val_lo = kFP4E2M1Dequant[code_lo] * combined_scale;
        if (sign_lo) val_lo = -val_lo;

        // High nibble = odd element.
        uint8_t fp4_hi = (byte >> 4) & 0x0F;
        uint8_t sign_hi = (fp4_hi >> 3) & 1;
        uint8_t code_hi = fp4_hi & 0x07;
        float val_hi = kFP4E2M1Dequant[code_hi] * combined_scale;
        if (sign_hi) val_hi = -val_hi;

        output[out_base + i]     = __float2half(val_lo);
        output[out_base + i + 1] = __float2half(val_hi);
    }
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

float calibrate_nvfp4_scales(const Tensor& input, cudaStream_t stream)
{
    assert(input.on_device && "input must be on device");
    assert(input.dtype == DType::FP16 && "input must be FP16");

    int64_t n_elements = input.numel();

    // Allocate and zero-initialize global max on device.
    float* d_global_max = nullptr;
    cudaMalloc(&d_global_max, sizeof(float));
    cudaMemsetAsync(d_global_max, 0, sizeof(float), stream);

    int num_blocks = (int)((n_elements + kBlockSize - 1) / kBlockSize);
    if (num_blocks > 2048) num_blocks = 2048;  // cap grid size

    absmax_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const half*>(input.data),
        n_elements,
        d_global_max
    );

    // Read back the result.
    float h_absmax = 0.0f;
    cudaMemcpyAsync(&h_absmax, d_global_max, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Reinterpret: atomicMax stored the bit pattern of absmax as uint, but
    // since we wrote via __float_as_uint and the value is non-negative, the
    // memcpy gives us the correct float.  (IEEE754 non-negative float order
    // matches unsigned integer order.)

    cudaFree(d_global_max);

    if (h_absmax == 0.0f) {
        IMP_LOG_WARN("calibrate_nvfp4_scales: tensor is all zeros, using scale 1.0");
        return 1.0f;
    }

    float tensor_scale = h_absmax / kFP4E2M1Max;
    IMP_LOG_DEBUG("calibrate_nvfp4_scales: absmax=%.6f, tensor_scale=%.6f",
                  h_absmax, tensor_scale);
    return tensor_scale;
}

void quantize_fp16_to_nvfp4(const Tensor& input, NvFP4QuantResult& result,
                             cudaStream_t stream)
{
    assert(input.on_device && "input must be on device");
    assert(input.dtype == DType::FP16 && "input must be FP16");
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
    cudaMalloc(&d_packed, packed_bytes);
    cudaMalloc(&d_micro_scales, micro_scale_bytes);

    // Step 3: Launch quantization kernel.
    int64_t total_micro_blocks = N * (K / kMicroBlockSize);
    int num_blocks = (int)((total_micro_blocks + kBlockSize - 1) / kBlockSize);

    quantize_nvfp4_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const half*>(input.data),
        d_packed,
        d_micro_scales,
        tensor_scale,
        N, K
    );

    // Fill result.
    result.packed_data  = d_packed;
    result.micro_scales = d_micro_scales;
    result.tensor_scale = tensor_scale;
    result.N = N;
    result.K = K;

    IMP_LOG_DEBUG("quantize_fp16_to_nvfp4: N=%lld K=%lld tensor_scale=%.6f "
                  "packed_bytes=%lld micro_scale_bytes=%lld",
                  (long long)N, (long long)K, tensor_scale,
                  (long long)packed_bytes, (long long)micro_scale_bytes);
}

void dequantize_nvfp4_to_fp16(const NvFP4QuantResult& quant,
                               void* output_fp16,
                               cudaStream_t stream)
{
    assert(quant.packed_data != nullptr && "packed_data must not be null");
    assert(quant.micro_scales != nullptr && "micro_scales must not be null");
    assert(output_fp16 != nullptr && "output buffer must not be null");

    int64_t N = quant.N;
    int64_t K = quant.K;
    int64_t total_micro_blocks = N * (K / kMicroBlockSize);
    int num_blocks = (int)((total_micro_blocks + kBlockSize - 1) / kBlockSize);

    dequantize_nvfp4_kernel<<<num_blocks, kBlockSize, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(quant.packed_data),
        reinterpret_cast<const uint8_t*>(quant.micro_scales),
        quant.tensor_scale,
        reinterpret_cast<half*>(output_fp16),
        N, K
    );
}

void free_nvfp4_result(NvFP4QuantResult& result)
{
    if (result.packed_data) {
        cudaFree(result.packed_data);
        result.packed_data = nullptr;
    }
    if (result.micro_scales) {
        cudaFree(result.micro_scales);
        result.micro_scales = nullptr;
    }
    result.tensor_scale = 1.0f;
    result.N = 0;
    result.K = 0;
}

// ---------------------------------------------------------------------------
// MoE per-expert quantization
// ---------------------------------------------------------------------------

void quantize_packed_experts_to_nvfp4(
    const void* packed_ggml_data, GGMLQuantType qtype,
    int n_experts, int eff, int K,
    void* dequant_scratch,
    NvFP4MoEQuantResult& result,
    cudaStream_t stream)
{
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
    cudaMalloc(&d_packed, total_packed);
    cudaMalloc(&d_micro_scales, total_ms);
    cudaMalloc(&d_tensor_scales, n_experts * sizeof(float));

    // Compute expert stride in source GGML data
    size_t src_expert_stride = static_cast<size_t>(eff) * ggml_quant_row_bytes(qtype, K);

    // Temporary device buffer for absmax reduction
    float* d_global_max = nullptr;
    cudaMalloc(&d_global_max, sizeof(float));

    int64_t n_elements = static_cast<int64_t>(eff) * K;
    int64_t total_micro_blocks = static_cast<int64_t>(eff) * (K / kMicroBlockSize);
    int quant_blocks = static_cast<int>((total_micro_blocks + kBlockSize - 1) / kBlockSize);

    for (int e = 0; e < n_experts; e++) {
        const uint8_t* src = static_cast<const uint8_t*>(packed_ggml_data) + e * src_expert_stride;
        half* scratch = static_cast<half*>(dequant_scratch);

        // Step 1: Dequant this expert slice to FP16 scratch
        dequant_gpu(src, scratch, qtype, eff, K, stream);

        // Step 2: Calibrate tensor_scale = absmax / 6.0
        cudaMemsetAsync(d_global_max, 0, sizeof(float), stream);
        int absmax_blocks = static_cast<int>((n_elements + kBlockSize - 1) / kBlockSize);
        if (absmax_blocks > 2048) absmax_blocks = 2048;
        absmax_kernel<<<absmax_blocks, kBlockSize, 0, stream>>>(
            scratch, n_elements, d_global_max);

        float h_absmax = 0.0f;
        cudaMemcpyAsync(&h_absmax, d_global_max, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float ts = (h_absmax == 0.0f) ? 1.0f : (h_absmax / kFP4E2M1Max);

        // Copy tensor_scale to device array
        cudaMemcpyAsync(d_tensor_scales + e, &ts, sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Step 3: Quantize FP16 scratch -> NVFP4 at expert offset
        quantize_nvfp4_kernel<<<quant_blocks, kBlockSize, 0, stream>>>(
            scratch,
            d_packed + e * expert_packed_bytes,
            d_micro_scales + e * expert_ms_bytes,
            ts,
            static_cast<int64_t>(eff), static_cast<int64_t>(K));

        // Must sync before scratch is reused by next expert
        cudaStreamSynchronize(stream);
    }

    cudaFree(d_global_max);

    // Fill result
    result.packed_data = d_packed;
    result.micro_scales = d_micro_scales;
    result.tensor_scales = d_tensor_scales;
    result.n_experts = n_experts;
    result.N = eff;
    result.K = K;
    result.expert_stride_packed = expert_packed_bytes;
    result.expert_stride_ms = expert_ms_bytes;

    IMP_LOG_DEBUG("quantize_packed_experts_to_nvfp4: %d experts, eff=%d K=%d, "
                  "packed=%.2f MiB, ms=%.2f MiB",
                  n_experts, eff, K,
                  total_packed / (1024.0 * 1024.0),
                  total_ms / (1024.0 * 1024.0));
}

void free_nvfp4_moe_result(NvFP4MoEQuantResult& result)
{
    if (result.packed_data) {
        cudaFree(result.packed_data);
        result.packed_data = nullptr;
    }
    if (result.micro_scales) {
        cudaFree(result.micro_scales);
        result.micro_scales = nullptr;
    }
    if (result.tensor_scales) {
        cudaFree(result.tensor_scales);
        result.tensor_scales = nullptr;
    }
    result.n_experts = 0;
    result.N = 0;
    result.K = 0;
    result.expert_stride_packed = 0;
    result.expert_stride_ms = 0;
}

} // namespace imp
