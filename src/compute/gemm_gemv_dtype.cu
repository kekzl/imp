#include "compute/gemm.h"
#include "compute/gemm_internal.cuh"
#include "core/logging.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstring>
#include "compute/pdl_device.cuh"
#include "runtime/pdl.h"

namespace imp {

// Bridge to the cuBLAS handle owned by gemm.cu so gemm_try_sgemm stays
// byte-identical to its original gemm.cu form.
static inline cublasHandle_t get_cublas_handle() { return gemm_internal_cublas_handle(); }

// --- GEMV fast path for M=1 decode (memory-bandwidth-bound) ---
// Applies when all operands share the same dtype (excludes LM head: FP16→FP32).
// Returns true if handled.
bool gemm_try_gemv(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                   cudaStream_t stream) {
    const int64_t M = A.shape[0];
    if (M != 1 || alpha != 1.0f || beta != 0.0f)
        return false;
    if (A.qtype != B.qtype || A.qtype != C.qtype)
        return false;
    if (A.qtype != QType::F16 && A.qtype != QType::F32 && A.qtype != QType::BF16)
        return false;

    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];

    Tensor x_vec;
    x_vec.data = A.data;
    x_vec.qtype = A.qtype;
    x_vec.ndim = 1;
    x_vec.shape[0] = K;
    x_vec.stride[0] = 1;
    x_vec.on_device = true;

    Tensor y_vec;
    y_vec.data = C.data;
    y_vec.qtype = C.qtype;
    y_vec.ndim = 1;
    y_vec.shape[0] = N;
    y_vec.stride[0] = 1;
    y_vec.on_device = true;

    gemv(B, x_vec, y_vec, stream);
    return true;
}

// --- FP32 fast path using cublasSgemm ---
// B is [N,K] row-major = [K,N] col-major. We need B transposed → CUBLAS_OP_T.
// A is [M,K] row-major = [K,M] col-major. We need A as-is    → CUBLAS_OP_N.
// Returns true if handled.
bool gemm_try_sgemm(const Tensor& A, const Tensor& B, Tensor& C, float alpha, float beta,
                    cudaStream_t stream) {
    if (A.qtype != QType::F32 || B.qtype != QType::F32 || C.qtype != QType::F32)
        return false;

    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];

    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);

    cublasStatus_t st = cublasSgemm(handle,
                                    CUBLAS_OP_T,  // transa: transpose B_col [K,N] → [N,K]
                                    CUBLAS_OP_N,  // transb: A_col [K,M] used as-is
                                    (int)N,       // m
                                    (int)M,       // n
                                    (int)K,       // k
                                    &alpha, static_cast<const float*>(B.data),
                                    (int)K,  // lda = K (leading dim of B before transpose)
                                    static_cast<const float*>(A.data), (int)K,  // ldb = K (leading dim of A)
                                    &beta, static_cast<float*>(C.data), (int)N  // ldc = N
    );
    if (st != CUBLAS_STATUS_SUCCESS) {
        IMP_LOG_ERROR("imp::gemm: cublasSgemm failed (status %d)", (int)st);
    }
    return true;
}

// ---------------------------------------------------------------------------
// GEMV kernels -- each warp computes one output element (dot product of a row)
// ---------------------------------------------------------------------------

// --- FP32 GEMV kernel ---
__global__ void gemv_fp32_kernel(const float* __restrict__ A, const float* __restrict__ x,
                                 float* __restrict__ y, int M, int K) {
    // Each warp handles one row of A.
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const float* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: float4 = 4 floats per load.
    const int K_vec = K / 4;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a = A_row_v[i];
        float4 xv = x_v[i];
        sum += a.x * xv.x + a.y * xv.y + a.z * xv.z + a.w * xv.w;
    }

    // Handle remainder elements (K not divisible by 4).
    int base = K_vec * 4;
    for (int i = base + lane; i < K; i += 32) {
        sum += A_row[i] * x[i];
    }

    // Warp-level reduction via shuffle.
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = sum;
    }
}

// --- FP16 GEMV kernel ---
__global__ void gemv_fp16_kernel(const half* __restrict__ A, const half* __restrict__ x, half* __restrict__ y,
                                 int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;
    pdl_wait();

    const half* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

#if __CUDA_ARCH__ >= 1200
    // Blackwell (sm_120+): 256-bit loads via paired float4 (16 halves per iteration).
    // 2× wider than the default 128-bit path, better saturating memory bandwidth.
    const int K_vec16 = K / 16;  // 16 halves = 32 bytes = 2 × sizeof(float4)
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec16; i += 32) {
        float4 a0 = A_row_v[2 * i];
        float4 a1 = A_row_v[2 * i + 1];
        float4 x0 = x_v[2 * i];
        float4 x1 = x_v[2 * i + 1];

        const half2* a_h2_0 = reinterpret_cast<const half2*>(&a0);
        const half2* x_h2_0 = reinterpret_cast<const half2*>(&x0);
        const half2* a_h2_1 = reinterpret_cast<const half2*>(&a1);
        const half2* x_h2_1 = reinterpret_cast<const half2*>(&x1);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2_0[j], x_h2_0[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2_1[j], x_h2_1[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
    }

    // Handle elements between K_vec16*16 and K_vec8*8 (0 or 8 elements).
    int base16 = K_vec16 * 16;
    if (base16 + 8 <= K) {
        int K_vec8_rem = (K - base16) / 8;
        const float4* A_rem = reinterpret_cast<const float4*>(A_row + base16);
        const float4* x_rem = reinterpret_cast<const float4*>(x + base16);
        for (int i = lane; i < K_vec8_rem; i += 32) {
            float4 a_raw = A_rem[i];
            float4 x_raw = x_rem[i];
            const half2* a_h2 = reinterpret_cast<const half2*>(&a_raw);
            const half2* x_h2 = reinterpret_cast<const half2*>(&x_raw);
            for (int j = 0; j < 4; ++j) {
                half2 prod = __hmul2(a_h2[j], x_h2[j]);
                sum += __half2float(prod.x) + __half2float(prod.y);
            }
        }
        base16 = base16 + K_vec8_rem * 8;
    }

    // Scalar remainder.
    for (int i = base16 + lane; i < K; i += 32) {
        sum += __half2float(A_row[i]) * __half2float(x[i]);
    }
#else
    // Default path: 128-bit loads (8 halves per float4).
    const int K_vec = K / 8;  // 8 halves = 16 bytes = sizeof(float4)
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a_raw = A_row_v[i];
        float4 x_raw = x_v[i];

        // Reinterpret as half2 arrays (4 half2 per float4).
        const half2* a_h2 = reinterpret_cast<const half2*>(&a_raw);
        const half2* x_h2 = reinterpret_cast<const half2*>(&x_raw);

        for (int j = 0; j < 4; ++j) {
            half2 prod = __hmul2(a_h2[j], x_h2[j]);
            sum += __half2float(prod.x) + __half2float(prod.y);
        }
    }

    // Remainder.
    int base = K_vec * 8;
    for (int i = base + lane; i < K; i += 32) {
        sum += __half2float(A_row[i]) * __half2float(x[i]);
    }
#endif

    // Warp shuffle reduction.
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// --- BF16 GEMV kernel ---
__global__ void gemv_bf16_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const __nv_bfloat16* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: 8 bf16 per float4.
    const int K_vec = K / 8;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);
    const float4* x_v = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a_raw = A_row_v[i];
        float4 x_raw = x_v[i];

        const __nv_bfloat162* a_h2 = reinterpret_cast<const __nv_bfloat162*>(&a_raw);
        const __nv_bfloat162* x_h2 = reinterpret_cast<const __nv_bfloat162*>(&x_raw);

        for (int j = 0; j < 4; ++j) {
            __nv_bfloat162 prod = __hmul2(a_h2[j], x_h2[j]);
            sum += __bfloat162float(prod.x) + __bfloat162float(prod.y);
        }
    }

    // Remainder.
    int base = K_vec * 8;
    for (int i = base + lane; i < K; i += 32) {
        sum += __bfloat162float(A_row[i]) * __bfloat162float(x[i]);
    }

    // Warp shuffle reduction.
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2bfloat16(sum);
    }
}

// ---------------------------------------------------------------------------
// gemv:  y = A @ x
//   A [M, K],  x [K] or [K, batch],  y [M] or [M, batch]
//   Custom CUDA kernels for the memory-bandwidth-bound case.
//   For batched case (x has 2 dims), we loop over batch columns.
// ---------------------------------------------------------------------------
void gemv(const Tensor& A, const Tensor& x, Tensor& y, cudaStream_t stream) {
    const int M = (int)A.shape[0];
    const int K = (int)A.shape[1];

    // Determine batch size from x's shape.
    int batch = 1;
    if (x.ndim == 2) {
        batch = (int)x.shape[1];
    }

    const int blocks = gemv_blocks(M);

    for (int b = 0; b < batch; ++b) {
        switch (A.qtype) {
            case QType::F32: {
                const float* A_ptr = static_cast<const float*>(A.data);
                const float* x_ptr = static_cast<const float*>(x.data) + (int64_t)b * K;
                float* y_ptr = static_cast<float*>(y.data) + (int64_t)b * M;
                gemv_fp32_kernel<<<blocks, kGemvThreads, 0, stream>>>(A_ptr, x_ptr, y_ptr, M, K);
                IMP_CUDA_CHECK_LAUNCH();
                break;
            }
            case QType::F16: {
                const half* A_ptr = static_cast<const half*>(A.data);
                const half* x_ptr = static_cast<const half*>(x.data) + (int64_t)b * K;
                half* y_ptr = static_cast<half*>(y.data) + (int64_t)b * M;
                pdl::enable_kernel(gemv_fp16_kernel);
                pdl::launch(gemv_fp16_kernel, dim3(blocks), dim3(kGemvThreads), size_t(0), stream, A_ptr, x_ptr, y_ptr, M, K);
                IMP_CUDA_CHECK_LAUNCH();
                break;
            }
            case QType::BF16: {
                const __nv_bfloat16* A_ptr = static_cast<const __nv_bfloat16*>(A.data);
                const __nv_bfloat16* x_ptr = static_cast<const __nv_bfloat16*>(x.data) + (int64_t)b * K;
                __nv_bfloat16* y_ptr = static_cast<__nv_bfloat16*>(y.data) + (int64_t)b * M;
                gemv_bf16_kernel<<<blocks, kGemvThreads, 0, stream>>>(A_ptr, x_ptr, y_ptr, M, K);
                IMP_CUDA_CHECK_LAUNCH();
                break;
            }
            default: {
                // Fallback: use cuBLAS gemv for other dtypes via gemm with N=1.
                // Construct a temporary Tensor view for the column vectors.
                Tensor x_col;
                x_col.data = static_cast<char*>(x.data) + b * K * dtype_size(x.qtype);
                x_col.qtype = x.qtype;
                x_col.ndim = 2;
                x_col.shape[0] = K;
                x_col.shape[1] = 1;
                x_col.stride[0] = 1;
                x_col.stride[1] = K;
                x_col.on_device = true;

                Tensor y_col;
                y_col.data = static_cast<char*>(y.data) + b * M * dtype_size(y.qtype);
                y_col.qtype = y.qtype;
                y_col.ndim = 2;
                y_col.shape[0] = M;
                y_col.shape[1] = 1;
                y_col.stride[0] = 1;
                y_col.stride[1] = M;
                y_col.on_device = true;

                gemm(A, x_col, y_col, 1.0f, 0.0f, stream);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 GEMV kernel -- 16 FP8 values per load (16 bytes)
// Each warp handles one row. Dequant on-the-fly; ROWSCALE selects a per-row
// (output-channel) scale lookup instead of the single per-tensor scale (the
// fp8_ssm_proj sidecar quantizes heterogeneous packed rows, where one tensor
// scale wastes e4m3 range).
// ---------------------------------------------------------------------------
template <bool ROWSCALE>
__global__ void gemv_fp8_e4m3_kernel(const uint8_t* __restrict__ A, const half* __restrict__ x,
                                     half* __restrict__ y, int M, int K, float scale,
                                     const float* __restrict__ row_scales) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;
    if constexpr (ROWSCALE)
        scale = row_scales[row];

    const uint8_t* A_row = A + (int64_t)row * K;

    float sum = 0.0f;

    // Vectorized loads: 16 FP8 values per load (16 bytes = sizeof(float4))
    const int K_vec = K / 16;
    const float4* A_row_v = reinterpret_cast<const float4*>(A_row);

    // x is FP16 -- load 8 halves at a time (16 bytes)
    const float4* x_v = reinterpret_cast<const float4*>(x);

    for (int i = lane; i < K_vec; i += 32) {
        float4 a_raw = A_row_v[i];

        // 16 FP8 values need 16 FP16 values = 2 float4 loads from x
        float4 x_raw0 = x_v[2 * i];
        float4 x_raw1 = x_v[2 * i + 1];

        // Reinterpret FP8 bytes
        const uint8_t* a_bytes = reinterpret_cast<const uint8_t*>(&a_raw);
        const half* x_lo = reinterpret_cast<const half*>(&x_raw0);  // x[0..7]
        const half* x_hi = reinterpret_cast<const half*>(&x_raw1);  // x[8..15]

// Dequant and accumulate 16 FP8 values in two groups of 8,
// avoiding per-element j<8 branch for x_lo vs x_hi selection.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &a_bytes[j], 1);
            float a_val = (float)fp8_val * scale;
            sum += a_val * __half2float(x_lo[j]);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &a_bytes[8 + j], 1);
            float a_val = (float)fp8_val * scale;
            sum += a_val * __half2float(x_hi[j]);
        }
    }

    // Handle remainder
    int base = K_vec * 16;
    for (int i = base + lane; i < K; i += 32) {
        __nv_fp8_e4m3 fp8_val;
        memcpy(&fp8_val, &A_row[i], 1);
        float a_val = (float)fp8_val * scale;
        sum += a_val * __half2float(*(reinterpret_cast<const half*>(x) + i));
    }

    // Warp reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        y[row] = __float2half(sum);
    }
}

// ---------------------------------------------------------------------------
// Fused Q6_K GEMV kernel -- dequant-and-dot in one pass.
// Q6_K block = 210 bytes for 256 elements: ql[128] + qh[64] + scales[16] + d[2].
// Each warp computes one output row's dot product.
// ---------------------------------------------------------------------------
__global__ void gemv_q6k_kernel(const uint8_t* __restrict__ W, const half* __restrict__ x,
                                half* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const int blocks_per_row = K / 256;
    const size_t row_bytes = (size_t)blocks_per_row * 210;
    const uint8_t* W_row = W + (size_t)row * row_bytes;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 210;
        const uint8_t* ql = bp;                            // ql[128]
        const uint8_t* qh = bp + 128;                      // qh[64]
        const int8_t* sc = (const int8_t*)(bp + 192);      // scales[16]
        float d = __half2float(*(const half*)(bp + 208));  // d[2]
        const int base = b * 256;

        // Coalesced loads: 4 ql bytes + 2 qh bytes per thread
        uint8_t ql_a = ql[lane];            // [0..31]
        uint8_t ql_b = ql[lane + 32];       // [32..63]
        uint8_t ql_c = ql[64 + lane];       // [64..95]
        uint8_t ql_d = ql[64 + lane + 32];  // [96..127]
        uint8_t qh0 = qh[lane];             // [0..31]
        uint8_t qh1 = qh[32 + lane];        // [32..63]

        // Dequant 8 values per thread (elements at lane, lane+32, ..., lane+224)
        int q0 = (int)(((qh0 & 0x03) << 4) | (ql_a & 0x0F)) - 32;
        int q1 = (int)((((qh0 >> 2) & 0x03) << 4) | (ql_b & 0x0F)) - 32;
        int q2 = (int)((((qh0 >> 4) & 0x03) << 4) | ((ql_a >> 4) & 0x0F)) - 32;
        int q3 = (int)((((qh0 >> 6) & 0x03) << 4) | ((ql_b >> 4) & 0x0F)) - 32;
        int q4 = (int)(((qh1 & 0x03) << 4) | (ql_c & 0x0F)) - 32;
        int q5 = (int)((((qh1 >> 2) & 0x03) << 4) | (ql_d & 0x0F)) - 32;
        int q6 = (int)((((qh1 >> 4) & 0x03) << 4) | ((ql_c >> 4) & 0x0F)) - 32;
        int q7 = (int)((((qh1 >> 6) & 0x03) << 4) | ((ql_d >> 4) & 0x0F)) - 32;

        // Scale lookups: 16 scales per block, 2 sub-blocks of 32 elements each
        // lane/16 selects between two scale groups within each 32-lane sub-block
        int sc_idx = lane >> 4;  // 0 or 1
        sum += d * ((float)sc[sc_idx] * (float)q0 * __half2float(x[base + lane]) +
                    (float)sc[sc_idx + 2] * (float)q1 * __half2float(x[base + lane + 32]) +
                    (float)sc[sc_idx + 4] * (float)q2 * __half2float(x[base + lane + 64]) +
                    (float)sc[sc_idx + 6] * (float)q3 * __half2float(x[base + lane + 96]) +
                    (float)sc[sc_idx + 8] * (float)q4 * __half2float(x[base + lane + 128]) +
                    (float)sc[sc_idx + 10] * (float)q5 * __half2float(x[base + lane + 160]) +
                    (float)sc[sc_idx + 12] * (float)q6 * __half2float(x[base + lane + 192]) +
                    (float)sc[sc_idx + 14] * (float)q7 * __half2float(x[base + lane + 224]));
    }

    // Warp shuffle reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[row] = __float2half(sum);
}

void gemv_q6k(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream) {
    gemv_q6k_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(static_cast<const uint8_t*>(W), x, y, M, K);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// Fused Q8_0 GEMV kernel -- dequant-and-dot in one pass.
// Q8_0 block = 34 bytes for 32 elements: d[2] + qs[32].
// Each warp computes one output row's dot product. Each thread handles one
// element per block (32 threads = 32 elements = 1 block).
// ---------------------------------------------------------------------------
__global__ void gemv_q8_0_kernel(const uint8_t* __restrict__ W, const half* __restrict__ x,
                                 half* __restrict__ y, int M, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M)
        return;

    const int blocks_per_row = K / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;
    const uint8_t* W_row = W + (size_t)row * row_bytes;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* bp = W_row + b * 34;
        float d = __half2float(*(const half*)bp);
        int8_t q = ((const int8_t*)(bp + 2))[lane];
        sum += d * (float)q * __half2float(x[b * 32 + lane]);
    }

    // Warp shuffle reduction
    sum = warp_reduce_sum(sum);

    if (lane == 0)
        y[row] = __float2half(sum);
}

void gemv_q8_0(const void* W, const half* x, half* y, int M, int K, cudaStream_t stream) {
    gemv_q8_0_kernel<<<gemv_blocks(M), kGemvThreads, 0, stream>>>(static_cast<const uint8_t*>(W), x, y, M, K);
    IMP_CUDA_CHECK_LAUNCH();
}

// ---------------------------------------------------------------------------
// FP8 E4M3 GEMV
// ---------------------------------------------------------------------------

void gemv_fp8(const Tensor& A, const Tensor& x, Tensor& y, float scale, cudaStream_t stream) {
    const int M = (int)A.shape[0];
    const int K = (int)A.shape[1];

    gemv_fp8_e4m3_kernel<false>
        <<<gemv_blocks(M), kGemvThreads, 0, stream>>>(static_cast<const uint8_t*>(A.data),
                                                      static_cast<const half*>(x.data),
                                                      static_cast<half*>(y.data), M, K, scale, nullptr);
    IMP_CUDA_CHECK_LAUNCH();
}

void gemv_fp8_rowscale(const Tensor& A, const Tensor& x, Tensor& y, const float* d_row_scales,
                       cudaStream_t stream) {
    const int M = (int)A.shape[0];
    const int K = (int)A.shape[1];

    gemv_fp8_e4m3_kernel<true>
        <<<gemv_blocks(M), kGemvThreads, 0, stream>>>(static_cast<const uint8_t*>(A.data),
                                                      static_cast<const half*>(x.data),
                                                      static_cast<half*>(y.data), M, K, 0.0f,
                                                      d_row_scales);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
