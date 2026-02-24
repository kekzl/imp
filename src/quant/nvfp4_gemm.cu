#include "quant/nvfp4_gemm.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm.h"
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cassert>

namespace imp {

// ---------------------------------------------------------------------------
// NVFP4 GEMV kernel: y[M] = A_nvfp4[M, K] @ x[K]
//
// Each warp (32 threads) handles one output row of A.  Threads cooperatively
// load chunks of the packed FP4 row, dequantize via two-level scaling, and
// accumulate the dot product with x.  Final reduction via warp shuffle.
//
// Memory layout:
//   A packed_data:   [M, K/2]  (2 FP4 values per byte)
//   A micro_scales:  [M, K/16] (FP8 E4M3)
//   x:               [K]       (FP16)
//   y:               [M]       (FP16)
// ---------------------------------------------------------------------------

static constexpr int kMicroBlockSize = 16;

// FP4 E2M1 dequant table (same as in nvfp4_quant.cu).
__constant__ float kFP4E2M1DequantGemm[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// FP8 E4M3 -> float (duplicated here to keep compilation units independent).
__device__ __forceinline__ float fp8_e4m3_to_float_gemm(uint8_t bits)
{
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp  = (bits >> 3) & 0x0F;
    uint32_t man  = bits & 0x07;

    float abs_val;
    if (exp == 0) {
        abs_val = (float)man * (1.0f / 512.0f);
    } else {
        abs_val = (float)(8 + man) * exp2f((float)(exp) - 10.0f);
    }
    return sign ? -abs_val : abs_val;
}

// ---------------------------------------------------------------------------
// Dequantize a single packed byte into two FP32 values.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dequant_byte(uint8_t packed, float scale,
                                              float& val0, float& val1)
{
    // Low nibble = even element.
    uint8_t fp4_lo = packed & 0x0F;
    uint8_t sign_lo = (fp4_lo >> 3) & 1;
    uint8_t code_lo = fp4_lo & 0x07;
    val0 = kFP4E2M1DequantGemm[code_lo] * scale;
    if (sign_lo) val0 = -val0;

    // High nibble = odd element.
    uint8_t fp4_hi = (packed >> 4) & 0x0F;
    uint8_t sign_hi = (fp4_hi >> 3) & 1;
    uint8_t code_hi = fp4_hi & 0x07;
    val1 = kFP4E2M1DequantGemm[code_hi] * scale;
    if (sign_hi) val1 = -val1;
}

// ---------------------------------------------------------------------------
// GEMV kernel.
// Grid:  (ceil(M / warps_per_block),)
// Block: (32 * warps_per_block,)     i.e., warps_per_block warps
//
// Each warp handles one row.  Within the row, each thread iterates over
// K/32 pairs of FP4 values (since 32 threads * 1 byte each = 32 nibbles =
// 16 packed bytes per iteration, covering 32 FP4 values).
//
// Actually, we iterate at micro-block granularity: each micro-block is 16
// values = 8 packed bytes.  With 32 threads per warp, we assign groups of
// threads to micro-blocks.  Strategy: each thread loads one byte (2 values)
// per iteration, so 32 threads cover 32 bytes = 64 FP4 values = 4 micro-
// blocks per iteration.
// ---------------------------------------------------------------------------

static constexpr int kWarpsPerBlock = 8;
static constexpr int kWarpSize = 32;

__global__ void gemv_nvfp4_kernel(
    const uint8_t* __restrict__ packed_data,    // [M, K/2]
    const uint8_t* __restrict__ micro_scales,   // [M, K/16]
    float                       tensor_scale,
    const half*    __restrict__ x,              // [K]
    half*          __restrict__ y,              // [M]
    int64_t M, int64_t K)
{
    const int warp_id_in_block = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;

    // Global row index for this warp.
    const int64_t row = (int64_t)blockIdx.x * kWarpsPerBlock + warp_id_in_block;
    if (row >= M) return;

    const int64_t K_half = K / 2;              // packed bytes per row
    const int64_t num_mb = K / kMicroBlockSize; // micro-blocks per row

    const uint8_t* row_packed = packed_data + row * K_half;
    const uint8_t* row_ms     = micro_scales + row * num_mb;

    float acc = 0.0f;

    // Each thread processes elements at stride of 32 (warp width) in units
    // of 2 values (1 packed byte).  So thread 'lane' processes byte indices
    // lane, lane+32, lane+64, ... within the row.
    for (int64_t byte_idx = lane; byte_idx < K_half; byte_idx += kWarpSize) {
        // The two FP4 values correspond to element indices 2*byte_idx and
        // 2*byte_idx+1.
        int64_t elem_idx = byte_idx * 2;

        // Determine which micro-block these elements belong to.
        int64_t mb = elem_idx / kMicroBlockSize;

        // Load micro-scale and compute combined scale.
        float ms = fp8_e4m3_to_float_gemm(row_ms[mb]);
        float combined_scale = tensor_scale * ms;

        // Dequantize the packed byte.
        uint8_t packed = row_packed[byte_idx];
        float v0, v1;
        dequant_byte(packed, combined_scale, v0, v1);

        // Load corresponding x values.
        float x0 = __half2float(x[elem_idx]);
        float x1 = __half2float(x[elem_idx + 1]);

        // Accumulate dot product.
        acc += v0 * x0 + v1 * x1;
    }

    // Warp-level reduction via shuffle.
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Lane 0 writes the result.
    if (lane == 0) {
        y[row] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Host launcher for GEMV.
// ---------------------------------------------------------------------------
void gemv_nvfp4(const NvFP4QuantResult& A, const Tensor& x, Tensor& y,
                cudaStream_t stream)
{
    assert(A.packed_data != nullptr && "A must be quantized");
    assert(x.on_device && "x must be on device");
    assert(y.on_device && "y must be on device");
    assert(x.dtype == DType::FP16 && "x must be FP16");
    assert(y.dtype == DType::FP16 && "y must be FP16");

    int64_t M = A.N;   // number of output rows
    int64_t K = A.K;

    // Validate x dimension.
    int64_t x_len = x.numel();
    assert(x_len == K && "x length must match A.K");

    // Validate y dimension.
    int64_t y_len = y.numel();
    assert(y_len == M && "y length must match A.N (number of rows)");

    int threads_per_block = kWarpsPerBlock * kWarpSize;  // 8 * 32 = 256
    int num_blocks = (int)((M + kWarpsPerBlock - 1) / kWarpsPerBlock);

    gemv_nvfp4_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(A.packed_data),
        reinterpret_cast<const uint8_t*>(A.micro_scales),
        A.tensor_scale,
        reinterpret_cast<const half*>(x.data),
        reinterpret_cast<half*>(y.data),
        M, K
    );

    IMP_LOG_DEBUG("gemv_nvfp4: M=%lld K=%lld blocks=%d threads=%d",
                  (long long)M, (long long)K, num_blocks, threads_per_block);
}

// ---------------------------------------------------------------------------
// GEMM for NVFP4 weights:  C = input @ A^T
//
//   A (NvFP4QuantResult): weight matrix [N, K] in NVFP4 packed format
//   input (Tensor):       activation     [M, K] in FP16
//   C (Tensor):           output         [M, N] in FP16
//
// Strategy:
//   1. Dequantize A from NVFP4 to FP16 into a temporary buffer.
//   2. Call the existing cuBLAS gemm() which computes C = input @ A_fp16^T.
//
// This avoids the per-call cudaMalloc by using a persistent scratch buffer
// (grown as needed, never shrunk).  For SM100+ with CUDA 13.1, a native
// cuBLASLt NVFP4 path can be added in the future.
// ---------------------------------------------------------------------------

// Persistent dequant scratch buffer (process-lifetime, grown as needed).
static void* s_nvfp4_dequant_buf = nullptr;
static size_t s_nvfp4_dequant_buf_size = 0;

static void* ensure_dequant_buffer(size_t needed) {
    if (needed <= s_nvfp4_dequant_buf_size) return s_nvfp4_dequant_buf;
    if (s_nvfp4_dequant_buf) cudaFree(s_nvfp4_dequant_buf);
    s_nvfp4_dequant_buf = nullptr;
    s_nvfp4_dequant_buf_size = 0;
    cudaError_t err = cudaMalloc(&s_nvfp4_dequant_buf, needed);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("gemm_nvfp4: failed to allocate %zu bytes for dequant buffer: %s",
                      needed, cudaGetErrorString(err));
        return nullptr;
    }
    s_nvfp4_dequant_buf_size = needed;
    IMP_LOG_DEBUG("gemm_nvfp4: allocated dequant scratch buffer: %zu bytes", needed);
    return s_nvfp4_dequant_buf;
}

void gemm_nvfp4(const NvFP4QuantResult& A, const Tensor& B, Tensor& C,
                cudaStream_t stream)
{
    assert(A.packed_data != nullptr && "A must be quantized");
    assert(B.on_device && "B (input) must be on device");
    assert(C.on_device && "C (output) must be on device");
    assert(B.ndim == 2 && "B (input) must be 2D [M, K]");
    assert(C.ndim == 2 && "C (output) must be 2D [M, N]");

    const int64_t N = A.N;   // weight out_features
    const int64_t K = A.K;   // weight in_features
    const int64_t M = B.shape[0];  // sequence length / batch tokens

    assert(B.shape[1] == K && "input columns must match weight in_features");
    assert(C.shape[0] == M && C.shape[1] == N && "output shape must be [M, N]");

    // For M == 1, prefer the custom GEMV kernel (bandwidth-optimized).
    if (M == 1) {
        gemv_nvfp4(A, B, C, stream);
        return;
    }

    // Dequantize weight A [N, K] from NVFP4 to FP16 into scratch buffer.
    size_t A_fp16_bytes = (size_t)(N * K) * sizeof(half);
    void* dequant_buf = ensure_dequant_buffer(A_fp16_bytes);
    if (!dequant_buf) return;

    dequantize_nvfp4_to_fp16(A, dequant_buf, stream);

    // Wrap as Tensor [N, K] and call standard cuBLAS GEMM: C = B @ A_fp16^T.
    int64_t A_shape[2] = {N, K};
    Tensor A_fp16(dequant_buf, DType::FP16, 2, A_shape, /*on_device=*/true);

    gemm(B, A_fp16, C, 1.0f, 0.0f, stream);
}

} // namespace imp
