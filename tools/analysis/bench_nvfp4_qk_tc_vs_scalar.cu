// tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu
// Phase 0 microbench (BitDecoding port plan):
// compare scalar-FFMA Q.K dot vs HMMA-MMA Q.K dot on dequantized NVFP4 KV.
// No imp dependencies. Build via the wrapper script.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// NVFP4 dequant: 1 byte = 2 packed E2M1 nibbles → 2 half via PTX cvt
// (same as imp's fp4_byte_to_half2 in src/compute/ptx92_utils.cuh)
// ---------------------------------------------------------------------------
__device__ __forceinline__ __half2 fp4_byte_to_half2(uint32_t byte_val) {
    uint32_t fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
        : "=r"(fp16x2) : "r"(byte_val));
    return *reinterpret_cast<__half2*>(&fp16x2);
}

// UE4M3 → fp32: 4-bit unbiased exp, 3-bit mantissa-equivalent. NVFP4 group scale.
__device__ __forceinline__ float ue4m3_decode(uint8_t s) {
    int e = (s >> 3) & 0xF;
    int m = s & 0x7;
    if (e == 0 && m == 0) return 0.0f;
    float val = (1.0f + m / 8.0f) * exp2f(static_cast<float>(e) - 7.0f);
    return val;
}

// ---------------------------------------------------------------------------
// Reference kernel: scalar FFMA Q.K dot on NVFP4 KV
// (mirrors imp's current path at attention_paged_nvfp4.cu:142-153).
// Q: half [HEAD_DIM]                — single query, single head (decode shape)
// K: uint8 [seqlen_kv, HEAD_DIM/2]  — packed NVFP4 (2 elems per byte)
// K_scales: uint8 [seqlen_kv, HEAD_DIM/16]  — UE4M3 per 16-element group
// out: float [seqlen_kv]            — per-token Q.K dot
// ---------------------------------------------------------------------------
template <int HEAD_DIM, int WARP_SIZE = 32>
__global__ void qk_dot_scalar_kernel(
    const __half* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ K_scales,
    float* __restrict__ out, int seqlen_kv) {
    constexpr int ELEMS = HEAD_DIM / WARP_SIZE;
    const int tok = blockIdx.x;
    const int lane = threadIdx.x;
    if (tok >= seqlen_kv) return;

    float q_reg[ELEMS];
    {
        const __half2* Q2 = reinterpret_cast<const __half2*>(Q + lane * ELEMS);
#pragma unroll
        for (int i = 0; i < ELEMS / 2; i++) {
            __half2 h2 = Q2[i];
            q_reg[2 * i]     = __half2float(h2.x);
            q_reg[2 * i + 1] = __half2float(h2.y);
        }
    }

    const int sc_groups = HEAD_DIM / 16;
    const int lane_group = (lane * ELEMS) / 16;
    float k_scale = ue4m3_decode(K_scales[tok * sc_groups + lane_group]);
    const __half2 k_scale_h2 = __float2half2_rn(k_scale);

    float dot = 0.0f;
    const uint8_t* k_bytes = K + tok * (HEAD_DIM / 2) + lane * ELEMS / 2;
#pragma unroll
    for (int i = 0; i < ELEMS / 2; i++) {
        __half2 kh2 = fp4_byte_to_half2(k_bytes[i]);
        kh2 = __hmul2(kh2, k_scale_h2);
        float2 kf = __half22float2(kh2);
        dot = __fmaf_rn(q_reg[2 * i],     kf.x, dot);
        dot = __fmaf_rn(q_reg[2 * i + 1], kf.y, dot);
    }

    for (int off = 16; off > 0; off >>= 1)
        dot += __shfl_xor_sync(0xffffffff, dot, off);
    if (lane == 0) out[tok] = dot;
}

// ---------------------------------------------------------------------------
// Phase 0 main: smoke that the scalar reference path compiles + runs.
// ---------------------------------------------------------------------------
int main() {
    constexpr int HEAD_DIM = 128;
    constexpr int seqlen_kv = 4096;

    std::vector<__half> Q_h(HEAD_DIM);
    std::vector<uint8_t> K_h(seqlen_kv * HEAD_DIM / 2);
    std::vector<uint8_t> Ks_h(seqlen_kv * HEAD_DIM / 16);
    for (int i = 0; i < HEAD_DIM; i++) Q_h[i] = __float2half(0.01f * (i % 17 - 8));
    for (size_t i = 0; i < K_h.size(); i++)  K_h[i]  = static_cast<uint8_t>(i & 0xff);
    for (size_t i = 0; i < Ks_h.size(); i++) Ks_h[i] = static_cast<uint8_t>(0x38);

    __half* d_Q = nullptr;
    uint8_t* d_K = nullptr;
    uint8_t* d_Ks = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_Q, HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_K, K_h.size());
    cudaMalloc(&d_Ks, Ks_h.size());
    cudaMalloc(&d_out, seqlen_kv * sizeof(float));
    cudaMemcpy(d_Q, Q_h.data(), HEAD_DIM * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_h.data(), K_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, Ks_h.data(), Ks_h.size(), cudaMemcpyHostToDevice);

    qk_dot_scalar_kernel<HEAD_DIM><<<seqlen_kv, 32>>>(d_Q, d_K, d_Ks, d_out, seqlen_kv);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> out(seqlen_kv);
    cudaMemcpy(out.data(), d_out, seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Phase 0 scalar reference: out[0]=%.4f out[100]=%.4f out[4095]=%.4f\n",
           out[0], out[100], out[4095]);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_Ks); cudaFree(d_out);
    return 0;
}
