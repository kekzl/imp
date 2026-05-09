// tools/analysis/bench_nvfp4_pv_tc_vs_scalar.cu
// Phase 2 debug: isolate the V-accumulation WMMA pattern.
//
// Computes o_contribution[hd_local] = sum_t weights[t] * V[t][hd_local]
// for 16 tokens × HEAD_DIM=128. Compares scalar reference against WMMA.
//
// Goal: rule out (or confirm) the WMMA layout for the V-accum step is the
// source of the Phase-2 production-kernel output degeneration. If this
// microbench shows numerical equivalence, the bug is in the production
// kernel's INTEGRATION (block-softmax merge, per-lane scatter, smem reuse).
// If this bench fails, the WMMA layout itself is wrong.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

__device__ __forceinline__ __half2 fp4_byte_to_half2(uint32_t byte_val) {
    uint32_t fp16x2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
        : "=r"(fp16x2) : "r"(byte_val));
    return *reinterpret_cast<__half2*>(&fp16x2);
}

__device__ __forceinline__ float ue4m3_decode(uint8_t s) {
    int e = (s >> 3) & 0xF;
    int m = s & 0x7;
    if (e == 0 && m == 0) return 0.0f;
    return (1.0f + m / 8.0f) * exp2f(static_cast<float>(e) - 7.0f);
}

// ---------------------------------------------------------------------------
// Reference scalar PV: o_contribution[hd] = sum_t weights[t] * V_dequant[t][hd]
// One block produces all HEAD_DIM elements of the output.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void pv_scalar_kernel(
    const float* __restrict__ weights,        // [16] per-token weight
    const uint8_t* __restrict__ V,            // [16, HEAD_DIM/2] packed FP4
    const uint8_t* __restrict__ V_scales,     // [16, HEAD_DIM/16] UE4M3
    float* __restrict__ out) {
    constexpr int sc_groups = HEAD_DIM / 16;
    const int hd = threadIdx.x;
    if (hd >= HEAD_DIM) return;

    float acc = 0.0f;
    for (int t = 0; t < 16; t++) {
        int byte_off = t * (HEAD_DIM / 2) + hd / 2;
        uint32_t b = V[byte_off];
        __half2 hh = fp4_byte_to_half2(b);
        __half v = (hd & 1) ? hh.y : hh.x;
        float scale = ue4m3_decode(V_scales[t * sc_groups + (hd / 16)]);
        float v_f = __half2float(v) * scale;
        acc += weights[t] * v_f;
    }
    out[hd] = acc;
}

// ---------------------------------------------------------------------------
// WMMA-PV: per-warp, computes one 16-wide hd_chunk per iteration.
// Total HEAD_DIM/16 iterations cover the full output.
//
// A: matrix_a [m=16, k=16] row_major, A[m, k] = weights[k] (replicated rows)
// B: matrix_b [k=16, n=16] row_major (NOT col_major like QK), B[k, n] = V[k][n]
// C/D: accumulator [m=16, n=16] FP32. Row 0 = output for this hd_chunk.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void pv_tc_kernel(
    const float* __restrict__ weights,
    const uint8_t* __restrict__ V,
    const uint8_t* __restrict__ V_scales,
    float* __restrict__ out) {
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be multiple of 16");
    constexpr int K_TILES = HEAD_DIM / 16;
    constexpr int sc_groups = HEAD_DIM / 16;

    const int lane = threadIdx.x;
    using namespace nvcuda;

    extern __shared__ __half smem[];
    __half* sQ = smem;                 // [16, 16]
    __half* sK = smem + 16 * 16;       // [16, 16]
    float* sFV = reinterpret_cast<float*>(sK + 16 * 16);  // [16, 16]

    // Replicate weights into A operand once
    for (int i = lane; i < 16 * 16; i += 32) {
        sQ[i] = __float2half(weights[i % 16]);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    for (int kt = 0; kt < K_TILES; kt++) {
        const int hd_off = kt * 16;

        // Dequant V[16, 16] into sK row_major
        for (int i = lane; i < 16 * 16; i += 32) {
            int t = i / 16;
            int hd_local = i % 16;
            int hd_global = hd_off + hd_local;
            int byte_off = t * (HEAD_DIM / 2) + hd_global / 2;
            uint32_t b = V[byte_off];
            __half2 hh = fp4_byte_to_half2(b);
            __half v = (hd_global & 1) ? hh.y : hh.x;
            float scale = ue4m3_decode(V_scales[t * sc_groups + (hd_global / 16)]);
            sK[i] = __float2half(__half2float(v) * scale);
        }
        __syncwarp();

        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, sQ, 16);
        wmma::load_matrix_sync(b_frag, sK, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        wmma::store_matrix_sync(sFV, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        // Lane (% 16) writes one hd_local of row 0 to global out
        if (lane < 16) {
            out[hd_off + lane] = sFV[lane];
        }
        __syncwarp();
    }
}

int main() {
    constexpr int HEAD_DIM = 128;
    constexpr int N_TOKENS = 16;

    std::vector<float> w_h(N_TOKENS);
    std::vector<uint8_t> V_h(N_TOKENS * HEAD_DIM / 2);
    std::vector<uint8_t> Vs_h(N_TOKENS * (HEAD_DIM / 16));

    // Synthetic: weights = exp(-t/8), V = small magnitudes, scales mid-range.
    for (int t = 0; t < N_TOKENS; t++) w_h[t] = expf(-t * 0.125f);
    for (size_t i = 0; i < V_h.size(); i++) {
        uint8_t lo = (i * 3) & 0x03;
        uint8_t hi = ((i * 5) + 1) & 0x03;
        V_h[i] = static_cast<uint8_t>(lo | (hi << 4));
    }
    for (size_t i = 0; i < Vs_h.size(); i++) Vs_h[i] = 0x20;  // ~0.125

    float* d_w = nullptr;
    uint8_t* d_V = nullptr;
    uint8_t* d_Vs = nullptr;
    float* d_out_scalar = nullptr;
    float* d_out_tc = nullptr;
    cudaMalloc(&d_w, N_TOKENS * sizeof(float));
    cudaMalloc(&d_V, V_h.size());
    cudaMalloc(&d_Vs, Vs_h.size());
    cudaMalloc(&d_out_scalar, HEAD_DIM * sizeof(float));
    cudaMalloc(&d_out_tc,     HEAD_DIM * sizeof(float));
    cudaMemcpy(d_w, w_h.data(), N_TOKENS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_h.data(), V_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vs, Vs_h.data(), Vs_h.size(), cudaMemcpyHostToDevice);

    pv_scalar_kernel<HEAD_DIM><<<1, HEAD_DIM>>>(d_w, d_V, d_Vs, d_out_scalar);
    cudaDeviceSynchronize();

    size_t smem_bytes = (16 * 16 + 16 * 16) * sizeof(__half) + 16 * 16 * sizeof(float);
    pv_tc_kernel<HEAD_DIM><<<1, 32, smem_bytes>>>(d_w, d_V, d_Vs, d_out_tc);
    cudaDeviceSynchronize();

    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        printf("FAIL kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> out_s(HEAD_DIM), out_t(HEAD_DIM);
    cudaMemcpy(out_s.data(), d_out_scalar, HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_t.data(), d_out_tc,     HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_err = 0.0f, max_val = 0.0f;
    for (int i = 0; i < HEAD_DIM; i++) {
        float e = fabsf(out_s[i] - out_t[i]);
        if (e > max_abs_err) max_abs_err = e;
        if (fabsf(out_s[i]) > max_val) max_val = fabsf(out_s[i]);
    }

    printf("Phase 2 V-WMMA microbench: HEAD_DIM=%d N_TOKENS=%d\n", HEAD_DIM, N_TOKENS);
    printf("  scalar[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           out_s[0], out_s[1], out_s[2], out_s[3], out_s[4], out_s[5], out_s[6], out_s[7]);
    printf("  tc    [0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           out_t[0], out_t[1], out_t[2], out_t[3], out_t[4], out_t[5], out_t[6], out_t[7]);
    printf("  max_abs_err=%.4e  max_val=%.4f  rel=%.4e\n",
           max_abs_err, max_val, max_abs_err / (max_val + 1e-9f));

    cudaFree(d_w);
    cudaFree(d_V);
    cudaFree(d_Vs);
    cudaFree(d_out_scalar);
    cudaFree(d_out_tc);
    return (max_abs_err < 1e-2f * max_val) ? 0 : 2;
}
