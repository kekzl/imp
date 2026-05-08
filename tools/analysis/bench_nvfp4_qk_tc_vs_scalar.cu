// tools/analysis/bench_nvfp4_qk_tc_vs_scalar.cu
// Phase 0 microbench (BitDecoding port plan):
// compare scalar-FFMA Q.K dot vs HMMA-MMA Q.K dot on dequantized NVFP4 KV.
// No imp dependencies. Build via the wrapper script.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
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
// TC kernel: WMMA Q.K dot on dequantized NVFP4 KV.
// Uses the nvcuda::wmma API (16×16×16 fragments) to avoid the m16n8k16
// ldmatrix-layout gotcha while still issuing HMMA on Tensor Cores.
// Each block processes 16 KV tokens. Q is replicated into rows 0..15 of A;
// only row 0 of D contributes the per-token dot.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void qk_dot_tc_kernel(
    const __half* __restrict__ Q,
    const uint8_t* __restrict__ K,
    const uint8_t* __restrict__ K_scales,
    float* __restrict__ out, int seqlen_kv) {
    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be multiple of 16 for wmma 16×16×16");
    constexpr int K_TILES = HEAD_DIM / 16;
    constexpr int N_TILE = 16;

    using namespace nvcuda;

    const int n_block = blockIdx.x;
    const int tok_base = n_block * N_TILE;
    if (tok_base >= seqlen_kv) return;

    const int lane = threadIdx.x;

    extern __shared__ __half smem[];
    __half* sQ = smem;                     // [16,16]
    __half* sK = smem + 16 * 16;           // [16,16]  (n_tok × hd_chunk)

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half> c_frag;
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    const int sc_groups = HEAD_DIM / 16;

    for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
        const int hd_off = k_tile * 16;

        // Load Q[hd_off : hd_off+16] into sQ rows 0..15 (replicated).
        // sQ is [16 rows, 16 cols] row-major — every row holds the same Q chunk.
        for (int i = lane; i < 16 * 16; i += 32) {
            int col = i % 16;
            sQ[i] = Q[hd_off + col];
        }

        // Load K[tok_base..tok_base+16, hd_off..hd_off+16] dequantized into sK.
        // sK is [16 tokens, 16 hd_cols] row-major.
        for (int i = lane; i < 16 * 16; i += 32) {
            int k_tok = tok_base + i / 16;
            int k_hd  = hd_off + (i % 16);
            if (k_tok < seqlen_kv) {
                int byte_off = k_tok * (HEAD_DIM / 2) + k_hd / 2;
                uint32_t b = K[byte_off];
                __half2 hh = fp4_byte_to_half2(b);
                __half v = (k_hd & 1) ? hh.y : hh.x;
                float scale = ue4m3_decode(K_scales[k_tok * sc_groups + (k_hd / 16)]);
                sK[i] = __float2half(__half2float(v) * scale);
            } else {
                sK[i] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // A: Q replicated, row-major [16,16]. ld stride = 16.
        wmma::load_matrix_sync(a_frag, sQ, 16);
        // B: K [n_tok, hd_chunk] is row-major in memory but we want col_major for
        // the matrix_b with shape [k=16, n=16]. Loading the same row-major buffer
        // with col_major declaration effectively transposes it: each row of sK
        // becomes a column of B → token i ↔ B column i. That's what we want
        // (Q.K^T over the 16-token tile).
        wmma::load_matrix_sync(b_frag, sK, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store the 16×16 accumulator back to shared mem and pluck out row 0
    // (each row should equal Q.K_tok[i] for the matching column i).
    __half* sOut = smem + 0;  // reuse sQ region — stride 16
    wmma::store_matrix_sync(sOut, c_frag, 16, wmma::mem_row_major);

    __syncthreads();

    // Output token i sits at sOut[0 * 16 + i] (row 0, col i).
    if (lane < N_TILE) {
        int tok = tok_base + lane;
        if (tok < seqlen_kv) out[tok] = __half2float(sOut[lane]);
    }
}

// ---------------------------------------------------------------------------
// Phase 0 main: run scalar reference + TC kernel, assert numerical equivalence.
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
    float* d_out_scalar = nullptr;
    float* d_out_tc = nullptr;
    cudaMalloc(&d_Q, HEAD_DIM * sizeof(__half));
    cudaMalloc(&d_K, K_h.size());
    cudaMalloc(&d_Ks, Ks_h.size());
    cudaMalloc(&d_out_scalar, seqlen_kv * sizeof(float));
    cudaMalloc(&d_out_tc, seqlen_kv * sizeof(float));
    cudaMemcpy(d_Q, Q_h.data(), HEAD_DIM * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_h.data(), K_h.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ks, Ks_h.data(), Ks_h.size(), cudaMemcpyHostToDevice);

    constexpr int N_TILE_TC = 16;
    size_t smem_bytes = (16 * 16 + 16 * 16) * sizeof(__half);  // sQ + sK

    // Warmup
    for (int i = 0; i < 3; i++) {
        qk_dot_scalar_kernel<HEAD_DIM><<<seqlen_kv, 32>>>(d_Q, d_K, d_Ks, d_out_scalar, seqlen_kv);
        qk_dot_tc_kernel<HEAD_DIM><<<(seqlen_kv + N_TILE_TC - 1) / N_TILE_TC, 32, smem_bytes>>>(
            d_Q, d_K, d_Ks, d_out_tc, seqlen_kv);
    }
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    constexpr int REPS = 100;

    cudaEventRecord(a);
    for (int i = 0; i < REPS; i++)
        qk_dot_scalar_kernel<HEAD_DIM><<<seqlen_kv, 32>>>(d_Q, d_K, d_Ks, d_out_scalar, seqlen_kv);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float scalar_ms = 0.0f;
    cudaEventElapsedTime(&scalar_ms, a, b);
    scalar_ms /= REPS;

    cudaEventRecord(a);
    for (int i = 0; i < REPS; i++)
        qk_dot_tc_kernel<HEAD_DIM><<<(seqlen_kv + N_TILE_TC - 1) / N_TILE_TC, 32, smem_bytes>>>(
            d_Q, d_K, d_Ks, d_out_tc, seqlen_kv);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float tc_ms = 0.0f;
    cudaEventElapsedTime(&tc_ms, a, b);
    tc_ms /= REPS;

    cudaEventDestroy(a);
    cudaEventDestroy(b);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FAIL kernel: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> out_s(seqlen_kv), out_t(seqlen_kv);
    cudaMemcpy(out_s.data(), d_out_scalar, seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_t.data(), d_out_tc,     seqlen_kv * sizeof(float), cudaMemcpyDeviceToHost);

    float max_abs_err = 0.0f, max_val = 0.0f;
    for (int i = 0; i < seqlen_kv; i++) {
        float e = fabsf(out_s[i] - out_t[i]);
        if (e > max_abs_err) max_abs_err = e;
        if (fabsf(out_s[i]) > max_val) max_val = fabsf(out_s[i]);
    }

    printf("Phase 0 microbench: HEAD_DIM=%d seqlen_kv=%d\n", HEAD_DIM, seqlen_kv);
    printf("  scalar (FFMA): %.4f ms / iter\n", scalar_ms);
    printf("  tc     (HMMA): %.4f ms / iter\n", tc_ms);
    printf("  speedup: %.2fx (TC vs scalar)\n", scalar_ms / tc_ms);
    printf("  max_abs_err=%.4e  rel=%.4e\n", max_abs_err, max_abs_err / (max_val + 1e-9f));
    printf("  scalar[0..2]=%.4f %.4f %.4f\n", out_s[0], out_s[1], out_s[2]);
    printf("  tc    [0..2]=%.4f %.4f %.4f\n", out_t[0], out_t[1], out_t[2]);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_Ks);
    cudaFree(d_out_scalar); cudaFree(d_out_tc);
    return (max_abs_err < 1e-2f * max_val) ? 0 : 2;
}
