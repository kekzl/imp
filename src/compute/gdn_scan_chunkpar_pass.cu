// Kernel 2 of the chunk-parallel GDN prefill scan (gdn.chunkpar_scan): the
// sequential inter-chunk state pass on tensor cores. Kernel 1, the launcher
// and the narrative are in gdn_scan_chunkpar.cu; shared pieces in
// gdn_scan_chunkpar.cuh.
#include "compute/gdn_scan_chunkpar.cuh"
#include "core/logging.h"

#include <cuda_bf16.h>
#include <mutex>

namespace imp {
namespace chunkpar {
namespace {

// ---------------------------------------------------------------------------
// Kernel 2 — sequential inter-chunk state pass + outputs, on tensor cores.
// Grid (n_heads, kColSplit): blockIdx.y owns COLS = HD/kColSplit state columns;
// u_eff, y and the state update are column-local, so the split only re-stages
// the (read-only) W / Qeff / K_d rows per CTA. 256 threads = 8 warps: the CTA is latency-bound
// (11 us per chunk against ~4 us of MMA + staging at 4 warps, 1 CTA/SM by
// shared memory), so the row GEMMs split their 4 x 4 warp tiles over 8 warps
// and the state update runs one m-tile per warp.
//
// Per chunk, three GEMMs as mma.sync m16n8k8 tf32 with FP32 accumulate:
//   u_eff [L x COLS]  = U_A - W    [L x SS] @ H [SS x COLS]
//   y     [L x COLS]  = (Y_A + Qeff [L x SS] @ H) * scale
//   H'    [SS x COLS] = D0L * H + K_d^T [SS x L] @ u_eff [L x COLS]
// H lives in shared memory (FP32): the B operand of the first two GEMMs and
// the C-init / D of the third. The carried state itself is never rounded —
// D0L*H enters the accumulator in FP32 — only the per-chunk increments see
// the operand rounding. The two GEMMs that feed the state (u_eff, H update)
// run as 3xFP16 m16n8k16 (a = a_hi + a_lo in fp16, 22 significant bits like
// 3xTF32, at the FP16/FP32-accumulate rate = 2x TF32 on GeForce Blackwell
// and k = 16 per instruction): after #1851 ncu read the tensor pipe 67%
// active with math_pipe_throttle the top stall, i.e. the TF32 rate was the
// limit; 3xFP16 reads K2 -15% on both hybrids (27B 328 -> 279 ms, 35B 102 ->
// 87 per pp4096) with the unit-test state diff at 1.3e-6 (3xTF32: 9.5e-7).
// Plain tf32 on u_eff is out (state diff 8.5e-5..1.1e-4); the y GEMM is an
// output term and runs plain fp16 (the tf32 precision class). The k16
// fragment pattern (rows 2tg / 2tg+1, A as float2 at column 2tg) collided on
// the padded strides (SA = 132 cannot serve both the row-GEMM float2 rows
// and the K_d^T rows 2tg; SH = 40 puts rows 0/4 on one bank group): ncu read
// bank conflicts 28% of the shared wavefronts, mio_throttle 1.3. The staged
// block is a swz128 tile and the [k][n] tiles use stride COLS + 4 now:
// conflicts 13%, K2 another -5% (27B 280 -> 266 ms, 35B 87 -> 83), 41% tensor
// pipe active, the rest wait / scoreboard stalls at 8 warps and 1 CTA/SM. The scalar float4 form of this
// kernel ran 242 us per 512-token strip at 6.6 TFLOPS; the three GEMMs are
// 1.6 GFLOP per strip and head-set.
// The staged [row][k] block is a swz128 tile (both fragment patterns land on
// distinct bank groups), the [k][n] tiles use stride COLS + 4.
// ---------------------------------------------------------------------------
// C[16 x NTILE*8] = s_a[m0.., 0..SS) @ H[0..SS, 0..NTILE*8) for one warp's
// 16-row strip, mma C layout. (A free function: nvcc 13.3 segfaults on a
// generic lambda carrying the X3 tag inside the kernel.)
// C[16 x NTILE*8] = s_a[m0.., 0..SS) @ H[0..SS, 0..NTILE*8) for one warp's
// 16-row strip, mma C layout, k16 fp16 fragments: X3 = 3xFP16 (state-feeding
// terms), else plain fp16 (output-only terms). s_a is a swz128 [64 x 128]
// tile, s_h a [k][n] tile of stride SH. (A free function: nvcc 13.3
// segfaults on a generic lambda carrying the X3 tag inside the kernel.)
template <int SS, int SH, int NTILE, bool X3>
__device__ __forceinline__ void gemm_rows_x_h(const float* __restrict__ s_a, const float* __restrict__ s_h,
                                              int m0, int g, int tg, float (&acc)[NTILE][4]) {
#pragma unroll
    for (int nt = 0; nt < NTILE; nt++)
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    for (int k0 = 0; k0 < SS; k0 += 16) {
        float a[8];
        const float2 p0 = *reinterpret_cast<const float2*>(&s_a[swz128(m0 + g, k0 + 2 * tg)]);
        const float2 p1 = *reinterpret_cast<const float2*>(&s_a[swz128(m0 + g + 8, k0 + 2 * tg)]);
        const float2 p2 = *reinterpret_cast<const float2*>(&s_a[swz128(m0 + g, k0 + 2 * tg + 8)]);
        const float2 p3 = *reinterpret_cast<const float2*>(&s_a[swz128(m0 + g + 8, k0 + 2 * tg + 8)]);
        a[0] = p0.x;
        a[1] = p0.y;
        a[2] = p1.x;
        a[3] = p1.y;
        a[4] = p2.x;
        a[5] = p2.y;
        a[6] = p3.x;
        a[7] = p3.y;
        const F16A af = f16_split_a(a);
#pragma unroll
        for (int nt = 0; nt < NTILE; nt++) {
            float b[4];
            b[0] = s_h[(k0 + 2 * tg) * SH + nt * 8 + g];
            b[1] = s_h[(k0 + 2 * tg + 1) * SH + nt * 8 + g];
            b[2] = s_h[(k0 + 2 * tg + 8) * SH + nt * 8 + g];
            b[3] = s_h[(k0 + 2 * tg + 9) * SH + nt * 8 + g];
            const F16B bf = f16_split_b(b);
            if constexpr (X3)
                mma_f16x3(acc[nt], af, bf);
            else
                mma_f16x1(acc[nt], af, bf);
        }
    }
}

constexpr int kPassThreads = 256;  // K2 CTA: 8 warps (see the kernel comment)

template <int HD, int SS, typename StateT>
__global__ void __launch_bounds__(kPassThreads, 1) gdn_chunkpar_pass_kernel(
    float* __restrict__ ws_base, StateT* __restrict__ h_state, half* __restrict__ y_out,
    int strip_t0, int strip_tokens, int n_chunks, int n_heads, int load_statet, int store_statet) {
    constexpr int COLS = HD / kColSplit;  // state columns this CTA owns
    constexpr int SH = COLS + 4;          // stride of H [k][n] and u_eff [k][n]: 4 mod 32 puts the k16
                                          // fragment rows (2tg, 2tg+1) on distinct bank groups
    constexpr int NT = kPassThreads;
    constexpr int NW = NT / 32;           // warps per CTA
    constexpr int NTILE = COLS / 8;       // n-tiles across the CTA's columns
    // Row GEMMs (u_eff, y): kChunk/16 m-tiles x NTILE n-tiles split over NW
    // warps -> each warp owns one m-tile and NTW n-tiles.
    constexpr int NTW = (kChunk / 16) * NTILE / NW;
    static_assert(COLS % 8 == 0 && SS % 64 == 0 && kChunk % 16 == 0, "mma tiling");
    static_assert(NW % (kChunk / 16) == 0 && NTW >= 1, "row-GEMM warp tiling");
    const int h = blockIdx.x;
    const int d_base = blockIdx.y * COLS;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, tg = lane % 4;  // mma fragment coordinates (groupID, thread-in-group)
    const float scale = rsqrtf(static_cast<float>(HD));
    const int inner = n_heads * HD;

    const ChunkparWs ws = chunkpar_ws_layout<HD, SS>(ws_base, n_heads);
    float* H32_h = ws.H32 + static_cast<size_t>(h) * SS * HD;

    extern __shared__ float smem[];
    float* s_a = smem;               // [kChunk][SS] swz128  staging: W, then Qeff, then K_d
    float* s_h = s_a + kChunk * SS;  // [SS][SH]             state slice, FP32
    float* s_ue = s_h + SS * SH;     // [kChunk][SH]  u_eff

    // This CTA's state slice: H[s][d_base + dl].
    for (int idx = tid; idx < SS * COLS; idx += NT) {
        const int s = idx / COLS, dl = idx % COLS;
        const size_t gi = static_cast<size_t>(s) * HD + d_base + dl;
        s_h[s * SH + dl] = load_statet
                               ? static_cast<float>(h_state[static_cast<size_t>(h) * SS * HD + gi])
                               : H32_h[gi];
    }
    __syncthreads();

    // Staging is software-pipelined: the next [kChunk x SS] factor block is
    // loaded into registers (8 float4 per thread) BEFORE the current GEMM
    // phase and committed to s_a after the barrier that closes it, so the
    // L2 round trips of the staging overlap the MMA work instead of sitting
    // on the critical path (ncu on the un-pipelined form: long_scoreboard
    // 4.25 stalls per issue, the top item). Rows >= L are staged as zeros
    // (the state update needs finite K_d rows there; the row GEMMs discard
    // them).
    constexpr int F4_PER_ROW = SS / 4;
    constexpr int F4T = kChunk * F4_PER_ROW / NT;  // float4 per thread per block
    static_assert(kChunk * F4_PER_ROW % NT == 0, "staging split");
    float4 pf[F4T];
    auto prefetch = [&](const float* src_rows, int L) {
        const float4* src = reinterpret_cast<const float4*>(src_rows);
#pragma unroll
        for (int i = 0; i < F4T; i++) {
            const int idx = tid + i * NT;
            pf[i] = (idx / F4_PER_ROW < L) ? src[idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto commit = [&]() {
#pragma unroll
        for (int i = 0; i < F4T; i++) {
            const int idx = tid + i * NT;
            *reinterpret_cast<float4*>(&s_a[swz128(idx / F4_PER_ROW, (idx % F4_PER_ROW) * 4)]) = pf[i];
        }
    };

    const int m0 = (warp % (kChunk / 16)) * 16;       // this warp's row tile in the row GEMMs
    const int n0 = (warp / (kChunk / 16)) * NTW * 8;  // its first column in the row GEMMs
    const int r0 = m0 + g, r1 = r0 + 8;

    prefetch(ws.W + static_cast<size_t>(h) * kChunk * SS, min(kChunk, strip_tokens));
    for (int c = 0; c < n_chunks; c++) {
        const int L = min(kChunk, strip_tokens - c * kChunk);
        const size_t slot = static_cast<size_t>(c) * n_heads + h;
        const float* UA_s = ws.UA + slot * kChunk * HD;
        const float* YA_s = ws.YA + slot * kChunk * HD;
        const float d0l = ws.D0L[slot];

        // -- u_eff = U_A - W @ H --
        commit();  // W
        prefetch(ws.QE + slot * kChunk * SS, L);
        if (L < kChunk)  // tail rows are K operands of the state update: zero them
            for (int idx = tid; idx < (kChunk - L) * COLS; idx += NT)
                s_ue[(L + idx / COLS) * SH + idx % COLS] = 0.0f;
        __syncthreads();
        if (m0 < L) {
            float2 ua[NTW][2];
#pragma unroll
            for (int nt = 0; nt < NTW; nt++) {
                const int col = n0 + nt * 8 + 2 * tg;
                ua[nt][0] = (r0 < L) ? *reinterpret_cast<const float2*>(&UA_s[r0 * HD + d_base + col])
                                     : make_float2(0.0f, 0.0f);
                ua[nt][1] = (r1 < L) ? *reinterpret_cast<const float2*>(&UA_s[r1 * HD + d_base + col])
                                     : make_float2(0.0f, 0.0f);
            }
            float acc[NTW][4];
            gemm_rows_x_h<SS, SH, NTW, true>(s_a, s_h + n0, m0, g, tg, acc);  // feeds the state: 3xFP16
#pragma unroll
            for (int nt = 0; nt < NTW; nt++) {
                const int col = n0 + nt * 8 + 2 * tg;
                if (r0 < L) {
                    s_ue[r0 * SH + col] = ua[nt][0].x - acc[nt][0];
                    s_ue[r0 * SH + col + 1] = ua[nt][0].y - acc[nt][1];
                }
                if (r1 < L) {
                    s_ue[r1 * SH + col] = ua[nt][1].x - acc[nt][2];
                    s_ue[r1 * SH + col + 1] = ua[nt][1].y - acc[nt][3];
                }
            }
        }
        __syncthreads();

        // -- y = (Y_A + Qeff @ H) * scale, from the PRE-update state --
        commit();  // Qeff
        prefetch(ws.KD + slot * kChunk * SS, L);
        __syncthreads();
        if (m0 < L) {
            float2 ya[NTW][2];
#pragma unroll
            for (int nt = 0; nt < NTW; nt++) {
                const int col = n0 + nt * 8 + 2 * tg;
                ya[nt][0] = (r0 < L) ? *reinterpret_cast<const float2*>(&YA_s[r0 * HD + d_base + col])
                                     : make_float2(0.0f, 0.0f);
                ya[nt][1] = (r1 < L) ? *reinterpret_cast<const float2*>(&YA_s[r1 * HD + d_base + col])
                                     : make_float2(0.0f, 0.0f);
            }
            float acc[NTW][4];
            gemm_rows_x_h<SS, SH, NTW, false>(s_a, s_h + n0, m0, g, tg, acc);  // output only: plain fp16
            const size_t t_base = static_cast<size_t>(strip_t0) + c * kChunk;
#pragma unroll
            for (int nt = 0; nt < NTW; nt++) {
                const int col = n0 + nt * 8 + 2 * tg;
                if (r0 < L)
                    *reinterpret_cast<__half2*>(&y_out[(t_base + r0) * inner + h * HD + d_base + col]) =
                        __floats2half2_rn((ya[nt][0].x + acc[nt][0]) * scale,
                                          (ya[nt][0].y + acc[nt][1]) * scale);
                if (r1 < L)
                    *reinterpret_cast<__half2*>(&y_out[(t_base + r1) * inner + h * HD + d_base + col]) =
                        __floats2half2_rn((ya[nt][1].x + acc[nt][2]) * scale,
                                          (ya[nt][1].y + acc[nt][3]) * scale);
            }
        }
        __syncthreads();

        // -- H = D0L * H + K_d^T @ u_eff --
        commit();  // K_d, rows >= L zero (the partial k-block needs finite rows)
        if (c + 1 < n_chunks)
            prefetch(ws.W + (slot + n_heads) * kChunk * SS, min(kChunk, strip_tokens - (c + 1) * kChunk));
        __syncthreads();
        {
            constexpr int MT = SS / 16 / NW;  // m-tiles (16 state rows each) per warp
            static_assert(MT >= 1, "state-update warp tiling");
            const int mbase = warp * MT * 16;
            float acc[MT][NTILE][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTILE; nt++) {
                    const int r = mbase + mt * 16 + g, col = nt * 8 + 2 * tg;
                    acc[mt][nt][0] = d0l * s_h[r * SH + col];
                    acc[mt][nt][1] = d0l * s_h[r * SH + col + 1];
                    acc[mt][nt][2] = d0l * s_h[(r + 8) * SH + col];
                    acc[mt][nt][3] = d0l * s_h[(r + 8) * SH + col + 1];
                }
            const int kmax16 = (L + 15) & ~15;  // rows [L, kmax16) of K_d and u_eff are zero
            for (int k0 = 0; k0 < kmax16; k0 += 16) {
                F16B bf[NTILE];
#pragma unroll
                for (int nt = 0; nt < NTILE; nt++) {
                    float b[4];
                    b[0] = s_ue[(k0 + 2 * tg) * SH + nt * 8 + g];
                    b[1] = s_ue[(k0 + 2 * tg + 1) * SH + nt * 8 + g];
                    b[2] = s_ue[(k0 + 2 * tg + 8) * SH + nt * 8 + g];
                    b[3] = s_ue[(k0 + 2 * tg + 9) * SH + nt * 8 + g];
                    bf[nt] = f16_split_b(b);
                }
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    // A[m = s][k = t] = K_d[t][s], staged row-major as [t][s]
                    const int r = mbase + mt * 16 + g;
                    float a[8];
                    a[0] = s_a[swz128(k0 + 2 * tg, r)];
                    a[1] = s_a[swz128(k0 + 2 * tg + 1, r)];
                    a[2] = s_a[swz128(k0 + 2 * tg, r + 8)];
                    a[3] = s_a[swz128(k0 + 2 * tg + 1, r + 8)];
                    a[4] = s_a[swz128(k0 + 2 * tg + 8, r)];
                    a[5] = s_a[swz128(k0 + 2 * tg + 9, r)];
                    a[6] = s_a[swz128(k0 + 2 * tg + 8, r + 8)];
                    a[7] = s_a[swz128(k0 + 2 * tg + 9, r + 8)];
                    const F16A af = f16_split_a(a);
#pragma unroll
                    for (int nt = 0; nt < NTILE; nt++)
                        mma_f16x3(acc[mt][nt], af, bf[nt]);  // state update: 3xFP16
                }
            }
            // Each warp owns its 16 state rows: no cross-warp hazard before the barrier.
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTILE; nt++) {
                    const int r = mbase + mt * 16 + g, col = nt * 8 + 2 * tg;
                    s_h[r * SH + col] = acc[mt][nt][0];
                    s_h[r * SH + col + 1] = acc[mt][nt][1];
                    s_h[(r + 8) * SH + col] = acc[mt][nt][2];
                    s_h[(r + 8) * SH + col + 1] = acc[mt][nt][3];
                }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < SS * COLS; idx += NT) {
        const int s = idx / COLS, dl = idx % COLS;
        const size_t gi = static_cast<size_t>(s) * HD + d_base + dl;
        const float v = s_h[s * SH + dl];
        if (store_statet)
            h_state[static_cast<size_t>(h) * SS * HD + gi] = static_cast<StateT>(v);
        else
            H32_h[gi] = v;
    }
}

}  // namespace

template <typename StateT>
void chunkpar_pass_128(float* ws_base, StateT* h_state, half* y, int strip_t0, int strip_tokens,
                       int n_chunks, int n_heads, int load_statet, int store_statet, cudaStream_t stream) {
    constexpr int HD = 128, SS = 128;
    constexpr size_t smem = (kChunk * SS + SS * (HD / kColSplit + 4) + kChunk * (HD / kColSplit + 4)) *
                            sizeof(float);
    static std::once_flag attr_once;
    std::call_once(attr_once, [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&gdn_chunkpar_pass_kernel<HD, SS, StateT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    });
    gdn_chunkpar_pass_kernel<HD, SS, StateT><<<dim3(n_heads, kColSplit), kPassThreads, smem, stream>>>(
        ws_base, h_state, y, strip_t0, strip_tokens, n_chunks, n_heads, load_statet, store_statet);
    IMP_CUDA_CHECK_LAUNCH();
}

template void chunkpar_pass_128<float>(float*, float*, half*, int, int, int, int, int, int, cudaStream_t);
template void chunkpar_pass_128<__nv_bfloat16>(float*, __nv_bfloat16*, half*, int, int, int, int, int, int,
                                               cudaStream_t);

}  // namespace chunkpar
}  // namespace imp
