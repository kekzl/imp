// ggml-compatible MMVQ kernels for Q4_K and Q5_1.
// Ported from llama.cpp vecdotq.cuh / mmvq.cu for exact numerical parity.
// Self-contained: no ggml header dependencies.

#include "ggml_mmvq.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace imp {

// -------------------------------------------------------------------------
// Block types (matching ggml layout exactly)
// -------------------------------------------------------------------------

// Q4_K: 256 elements per block, 144 bytes
struct ggml_block_q4_K {
    half d;              // super-block scale
    half dmin;           // super-block min
    uint8_t scales[12];  // sub-block scales and mins, 6-bit quantized
    uint8_t qs[128];     // 4-bit quants (256 values packed)
};
static_assert(sizeof(ggml_block_q4_K) == 144, "Q4_K block must be 144 bytes");

// Q5_1: 32 elements per block, 24 bytes (union dm accessed as half2)
struct ggml_block_q5_1 {
    half d;              // delta
    half m;              // min
    uint8_t qh[4];       // 5th bit of quants
    uint8_t qs[16];      // lower 4 bits of quants
};
static_assert(sizeof(ggml_block_q5_1) == 24, "Q5_1 block must be 24 bytes");

// Q8_1 for ggml MMVQ: matches ggml layout (half2 ds = {d, s})
// We use this internally so vec_dot reads are identical to ggml.
struct ggml_block_q8_1 {
    half d;              // delta (scale)
    half s;              // sum = d * sum(qs[i])
    int8_t qs[32];       // quantized values
};
static_assert(sizeof(ggml_block_q8_1) == 36, "Q8_1 block must be 36 bytes");

// -------------------------------------------------------------------------
// Constants (from ggml-common.h)
// -------------------------------------------------------------------------

static constexpr int QK4_K = 256;
static constexpr int QK5_1 = 32;
static constexpr int QK8_1 = 32;

// Q4_K: qk=256, qi = QK_K/(4*QR4_K) = 256/8 = 32, QR4_K=2, vdr=2
static constexpr int QI4_K = 32;  // QK_K / (4 * QR4_K)
static constexpr int QR4_K = 2;

// Q5_1: qk=32, qi = QK5_1/(4*QR5_1) = 32/8 = 4, QR5_1=2, vdr=2
static constexpr int QI5_1 = 4;   // QK5_1 / (4 * QR5_1)
static constexpr int QR5_1 = 2;

static constexpr int WARP_SIZE = 32;

// -------------------------------------------------------------------------
// Device helpers
// -------------------------------------------------------------------------

static __device__ __forceinline__ int get_int_b4(const void* x, const int& i32) {
    return ((const int*)x)[i32];
}

static __device__ __forceinline__ int ggml_dp4a(const int a, const int b, int c) {
    return __dp4a(a, b, c);
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

// -------------------------------------------------------------------------
// vec_dot_q5_1_q8_1 — ported exactly from ggml vecdotq.cuh
// -------------------------------------------------------------------------

static constexpr int VDR_Q5_1 = 2;  // VDR_Q5_1_Q8_1_MMVQ

template <int vdr>
static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int* vl, const int* vh, const int* u, const half2& dm5, const half2& ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
        vi0    |= (vh[i] <<  4) & 0x00000010;
        vi0    |= (vh[i] << 11) & 0x00001000;
        vi0    |= (vh[i] << 18) & 0x00100000;
        vi0    |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_dp4a(vi0, u[2*i+0], sumi);

        int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
        vi1    |= (vh[i] >> 12) & 0x00000010;
        vi1    |= (vh[i] >>  5) & 0x00001000;
        vi1    |= (vh[i] <<  2) & 0x00100000;
        vi1    |= (vh[i] <<  9) & 0x10000000;
        sumi = ggml_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;

    return sumi * d5d8 + m5s8 / (QI5_1 / vdr);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void* __restrict__ vbq, const ggml_block_q8_1* __restrict__ bq8_1,
    const int& kbx, const int& iqs) {

    const ggml_block_q5_1* bq5_1 = (const ggml_block_q5_1*)vbq + kbx;

    int vl[VDR_Q5_1];
    int vh[VDR_Q5_1];
    int u[2 * VDR_Q5_1];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        vl[i]    = get_int_b4(bq5_1->qs, iqs + i);
        vh[i]    = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }

    // Construct half2 from imp's separate d/m and d/s fields
    const half2 dm5 = make_half2(bq5_1->d, bq5_1->m);
    const half2 ds8 = make_half2(bq8_1->d, bq8_1->s);

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1>(vl, vh, u, dm5, ds8);
}

// -------------------------------------------------------------------------
// vec_dot_q4_K_q8_1 — ported exactly from ggml vecdotq.cuh (vmmq variant)
// -------------------------------------------------------------------------

static constexpr int VDR_Q4_K = 2;  // VDR_Q4_K_Q8_1_MMVQ

// vec_dot_q4_K_q8_1_impl_vmmq — exact copy from ggml
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
    const half2& dm4, const float* __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_dp4a(v1i, u[2*i+1], ggml_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = ggml_dp4a(0x01010101, u[2*i+1], ggml_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void* __restrict__ vbq, const ggml_block_q8_1* __restrict__ bq8_1,
    const int& kbx, const int& iqs) {

    const ggml_block_q4_K* bq4_K = (const ggml_block_q4_K*)vbq + kbx;

    int   v[2];
    int   u[2 * QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs/2) / (QK8_1 / (4 * 2)));  // QI8_1 = 8, QI8_1/2 = 4

    const int* q4 = (const int*)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t* scales = (const uint16_t*)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const ggml_block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        // ggml reads __low2float(bq8i->ds) which is the d field
        d8[i] = __half2float(bq8i->d);

        const int* q8 = (const int*)bq8i->qs + ((iqs/2) % 4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    const half2 dm = make_half2(bq4_K->d, bq4_K->dmin);

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

// -------------------------------------------------------------------------
// Q8_1 quantization kernel (ggml-compatible)
// -------------------------------------------------------------------------

static __global__ void quantize_fp16_to_q8_1_ggml_kernel(
    const half* __restrict__ x, ggml_block_q8_1* __restrict__ y, int K) {

    const int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_blocks = K / QK8_1;
    if (block_id >= num_blocks) return;

    const half* xb = x + block_id * QK8_1;
    ggml_block_q8_1* yb = y + block_id;

    // Find max absolute value
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        float v = __half2float(xb[i]);
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        float v = __half2float(xb[i]);
        int8_t q = (int8_t)roundf(v * id);
        yb->qs[i] = q;
        sum += (float)q;
    }

    yb->d = __float2half(d);
    yb->s = __float2half(d * sum);
}

// FP32 input variant: quantizes from FP32 instead of FP16.
// This matches llama's precision where the residual stream is FP32.
static __global__ void quantize_fp32_to_q8_1_ggml_kernel(
    const float* __restrict__ x, ggml_block_q8_1* __restrict__ y, int K) {

    const int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_blocks = K / QK8_1;
    if (block_id >= num_blocks) return;

    const float* xb = x + block_id * QK8_1;
    ggml_block_q8_1* yb = y + block_id;

    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        amax = fmaxf(amax, fabsf(xb[i]));
    }

    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (127.0f / amax) : 0.0f;

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_1; ++i) {
        int8_t q = (int8_t)roundf(xb[i] * id);
        yb->qs[i] = q;
        sum += (float)q;
    }

    yb->d = __float2half(d);
    yb->s = __float2half(d * sum);
}

// -------------------------------------------------------------------------
// MMVQ kernel — warp-per-row, simplified from ggml mul_mat_vec_q
// -------------------------------------------------------------------------

// Template tag for dispatch
enum class QType { Q4_K, Q5_1 };

template <QType qtype>
static __global__ void mmvq_kernel(
    const void* __restrict__ W,
    const ggml_block_q8_1* __restrict__ x_q8,
    half* __restrict__ y,
    int N, int K) {

    // Each warp computes one output element: y[row, col]
    // Grid: (N, M), Block: (32, nwarps)
    constexpr int nwarps = 4;

    constexpr int qk  = (qtype == QType::Q4_K) ? QK4_K : QK5_1;
    constexpr int qi  = (qtype == QType::Q4_K) ? QI4_K : QI5_1;
    constexpr int vdr = (qtype == QType::Q4_K) ? VDR_Q4_K : VDR_Q5_1;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;  // output row (N dimension of W)
    const int col = blockIdx.y;  // input row (M dimension of x)

    if (row >= N) return;

    const int blocks_per_row = K / qk;

    // Quantized input: offset to correct row
    const ggml_block_q8_1* yq = x_q8 + col * (K / QK8_1);

    float tmp = 0.0f;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));

        if constexpr (qtype == QType::Q4_K) {
            tmp += vec_dot_q4_K_q8_1(W, &yq[kby], row * blocks_per_row + kbx, kqs);
        } else {
            tmp += vec_dot_q5_1_q8_1(W, &yq[kby], row * blocks_per_row + kbx, kqs);
        }
    }

    // Warp reduction across nwarps via shared memory
    __shared__ float smem[nwarps - 1][WARP_SIZE];

    if (threadIdx.y > 0) {
        smem[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();

    if (threadIdx.y > 0) return;

    // Warp 0 accumulates partial sums from other warps
#pragma unroll
    for (int w = 0; w < nwarps - 1; ++w) {
        tmp += smem[w][threadIdx.x];
    }

    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        y[col * N + row] = __float2half(tmp);
    }
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

void ggml_mmvq_q4k(
    const void* W, const half* x, half* y,
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream) {

    const int num_q8_blocks = (K / QK8_1);
    const int total_q8_blocks = M * num_q8_blocks;
    size_t need = (size_t)total_q8_blocks * sizeof(ggml_block_q8_1);
    if (need > scratch_size) return;
    ggml_block_q8_1* x_q8 = (ggml_block_q8_1*)scratch;

    {
        const int threads = 256;
        const int nblk = (total_q8_blocks + threads - 1) / threads;
        quantize_fp16_to_q8_1_ggml_kernel<<<nblk, threads, 0, stream>>>(x, x_q8, M * K);
    }
    {
        constexpr int nwarps = 4;
        dim3 block(WARP_SIZE, nwarps);
        dim3 grid(N, M);
        mmvq_kernel<QType::Q4_K><<<grid, block, 0, stream>>>(W, x_q8, y, N, K);
    }
}

void ggml_mmvq_q5_1(
    const void* W, const half* x, half* y,
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream) {

    const int num_q8_blocks = (K / QK8_1);
    const int total_q8_blocks = M * num_q8_blocks;
    size_t need = (size_t)total_q8_blocks * sizeof(ggml_block_q8_1);
    if (need > scratch_size) return;
    ggml_block_q8_1* x_q8 = (ggml_block_q8_1*)scratch;

    // Quantize FP16 input to Q8_1
    {
        const int threads = 256;
        const int blocks = (total_q8_blocks + threads - 1) / threads;
        quantize_fp16_to_q8_1_ggml_kernel<<<blocks, threads, 0, stream>>>(x, x_q8, M * K);
    }

    // MMVQ kernel: grid(N, M), block(32, 4)
    {
        constexpr int nwarps = 4;
        dim3 block(WARP_SIZE, nwarps);
        dim3 grid(N, M);
        mmvq_kernel<QType::Q5_1><<<grid, block, 0, stream>>>(W, x_q8, y, N, K);
    }
}

// FP32 input variants — quantize FP32→Q8_1 (matches llama's FP32 residual stream)
void ggml_mmvq_q4k_f32(
    const void* W, const float* x, half* y,
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream) {

    const int num_q8_blocks = (K / QK8_1);
    const int total_q8_blocks = M * num_q8_blocks;
    size_t need = (size_t)total_q8_blocks * sizeof(ggml_block_q8_1);
    if (need > scratch_size) return;
    ggml_block_q8_1* x_q8 = (ggml_block_q8_1*)scratch;

    {
        const int threads = 256;
        const int nblk = (total_q8_blocks + threads - 1) / threads;
        quantize_fp32_to_q8_1_ggml_kernel<<<nblk, threads, 0, stream>>>(x, x_q8, M * K);
    }
    {
        constexpr int nwarps = 4;
        dim3 block(WARP_SIZE, nwarps);
        dim3 grid(N, M);
        mmvq_kernel<QType::Q4_K><<<grid, block, 0, stream>>>(W, x_q8, y, N, K);
    }
}

void ggml_mmvq_q5_1_f32(
    const void* W, const float* x, half* y,
    int M, int N, int K,
    void* scratch, size_t scratch_size,
    cudaStream_t stream) {

    const int num_q8_blocks = (K / QK8_1);
    const int total_q8_blocks = M * num_q8_blocks;
    size_t need = (size_t)total_q8_blocks * sizeof(ggml_block_q8_1);
    if (need > scratch_size) return;
    ggml_block_q8_1* x_q8 = (ggml_block_q8_1*)scratch;

    {
        const int threads = 256;
        const int nblk = (total_q8_blocks + threads - 1) / threads;
        quantize_fp32_to_q8_1_ggml_kernel<<<nblk, threads, 0, stream>>>(x, x_q8, M * K);
    }
    {
        constexpr int nwarps = 4;
        dim3 block(WARP_SIZE, nwarps);
        dim3 grid(N, M);
        mmvq_kernel<QType::Q5_1><<<grid, block, 0, stream>>>(W, x_q8, y, N, K);
    }
}

} // namespace imp
