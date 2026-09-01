#include "compute/gdn_internal.cuh"
#include "compute/gdn_scan_chunkpar.cuh"
#include "core/logging.h"

#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace imp {

// ---------------------------------------------------------------------------
// Chunk-PARALLEL GDN delta-rule prefill scan.
//
// Every prior scan variant in this repo (fused, chunkwise 1b.1, WY 2a/2b/2c)
// launches <<<n_heads, HD>>> = 32 CTAs on 170 SMs and walks the tokens (or the
// chunks) sequentially inside the CTA. At Qwen3.6-35B pp512 that kernel is 42%
// of the prefill wall (658 us/layer, 1.28 us/tok) on 19% of the SMs.
//
// This variant makes the CHUNKS independent by splitting the WY solution on
// its linearity in the incoming state H_0:
//
//     u = u_A - W @ H_0
//
// where u_A solves the chunk's triangular system with RHS beta_t * v_t and W
// solves the SAME system with RHS beta_t * D[0..t+1] * k~_t (both independent
// of H_0). Substituting into the WY forms (gdn_scan_chunkwise_wy_kernel):
//
//     y_t  = Qeff[t] @ H_0 + Y_A[t]
//     H_L  = D[0..L] * H_0 - K_d^T (W @ H_0) + K_d^T U_A
//
// with  Qeff[t,:] = D[0..t+1] q~_t - sum_j P[t,j] W[j,:]
//       Y_A[t,:]  = sum_j P[t,j] U_A[j,:]
//       P[t,j]    = D[j+1..t+1] QK[t,j]          (j <= t)
//       K_d[t,:]  = D[t+1..L] k~_t
//
// All five per-chunk arrays (W, K_d, U_A, Qeff, Y_A) are independent of H_0,
// so kernel 1 computes them with grid (n_chunks x n_heads) — full-device
// parallelism. Kernel 2 then runs the cheap sequential chain per head: per
// chunk it is three L x 128 x 128 matmuls against the register-resident
// state, with the strip factors staged through shared memory.
//
// Perf notes from the first (refuted) cut, measured on Qwen3.6-35B pp512:
//   - K1 with the triangular-solve histories in GLOBAL memory ran 379 us:
//     the j-loop is one dependent L2 round-trip per term. The histories now
//     live in shared memory, aliased over the K/Q staging region (the K/Q
//     data they replace has already been folded into the Gram matrices and
//     the decay-scaled global copies by then).
//   - K2 as one CTA per head with scalar smem reads ran 628 us — as slow as
//     the sequential scan it replaces (every FMA carried a 4 B shared-memory
//     operand, 1 CTA/SM). Now: float4 staging reads, and the state columns
//     split across gridDim.y CTAs (column-local everything), 2 CTAs/SM.
//   - Accumulator-split (2-4 partial chains) and 2-row-unroll variants of the
//     K2 loops measured WORSE (242 -> 295/309 us: register growth vs the
//     2-CTA residency); the simple single-chain float4 form ships. The K1
//     solve keeps 4 partial accumulators — there they pay.
//   - #1847 shipped state (Qwen3.6-35B class kernel sums): pp512 144.5 ->
//     98.5 ms (-32%), pp4096 1485 -> 786 ms (-47%); K1 196 us, K2 242 us per
//     512-token strip, both ~4x over the compute floor (ncu: short_scoreboard
//     2.4 + barrier 2.1 stalls/issue in K1).
//   - K2 on tensor cores (mma.sync m16n8k8 tf32, this file): 242 -> 65 us
//     with plain tf32 on all three GEMMs, but PPL 6.8216 -> 6.8304 (+0.13%,
//     the state path compounds the 10-bit operand rounding); with 3xTF32 on
//     the two GEMMs that feed the carried state (u_eff, H update) and plain
//     tf32 on y: 90 us, PPL 6.8122, FP32 state 8.9e-7 vs the fused kernel.
//     Class pp512 145 -> 67/69 ms (-53%), pp4096 1530/1574 -> 546/542 ms
//     (-65%), e2e pp4096 12.5k -> 21.5k tok/s (#1848).
//   - K1 phases per CTA (ncu, test geometry): A 54 us (64 serial row loads +
//     the scalar Gram), B 45 (solve), C 33 (Qeff/Y_A). Now: float4-per-lane
//     row loads, per-token decay/beta in parallel, Gram as 3xTF32 mma, P@W /
//     P@U_A as 3xTF32 mma -> 75 us per CTA, 201 -> 128 us per strip in situ.
//     Plain tf32 on P@W is NOT safe: Qeff = D q~ - P W is a difference of
//     O(1) terms. The 35B PPL is no judge below ~0.5% (MoE routing flips
//     between fp32-equivalent kernels: 6.8122..6.8493 across variants with
//     state diffs of 1e-6); Qwen3.8-27B (deterministic) reads fused 4.6283
//     -> 4.6148. The solve (45 us, 128 barriers) was then 60% of K1.
//   - Blockwise solve (this file): T built once in place of KK, RHS staged
//     into the histories, per 16-row block an off-diagonal 3xTF32 mma update
//     + a register-resident diagonal block per thread; 8 barriers instead of
//     128. K1 per CTA 75 -> 49.5 us, in situ 128 -> 82 us per strip (K2 91).
//     Class pp512 144 -> 42.5/42.3 ms (-71%), pp4096 1497/1489 -> 307/308 ms
//     (-79%), e2e pp4096 12.9k -> 27.0k tok/s; Qwen3.8 PPL 4.6273.
//   - K2 at 8 warps + software-pipelined staging (gdn_scan_chunkpar_pass.cu),
//     the strip sized per n_heads (2026-09-02): ncu on the 4-warp K2 read
//     long_scoreboard 4.25 stalls/issue (the global staging loads) at 16.7%
//     warps active, 1 CTA/SM by shared memory. 8 warps: K2 -12/-14% (27B/
//     35B); prefetching the next factor block into registers before each
//     GEMM phase: another -19/-21%. Strip: 48 heads x 8 chunks = 384 K1
//     CTAs = 2.26 waves; 7 chunks (336, 1.98 waves) reads K1 -12% on the
//     27B, 10 chunks on 32 heads -11%; 14-16 chunks read K1 -21% but K2
//     +11% (the strip's factor set no longer fits L2), hence the L2 cap in
//     chunkpar_strip_chunks. pp4096 kernel sums vs #1850: 27B K1 365 -> 322
//     ms, K2 474 -> 332; 35B K1 150 -> 134, K2 152 -> 103; e2e 27B 10.5k ->
//     11.0k tok/s (+5%), 35B 27.3k -> 28.9k (+5.5%). K1's shared tiles then
//     got the XOR swizzle (swz128): ncu bank conflicts 11.1M -> 0.56M on
//     17.3M -> 3.7M wavefronts, but K1 only -1..-1.5% (4/4 pairs): with the
//     conflicts gone the top stall is math_pipe_throttle 2.9 (3xTF32 issue).
//
// Numerics: identical formulas to gdn_scan_chunkwise_wy_kernel (log-space
// cumulative decay, the same softplus/sigmoid/L2-norm forms), reassociated
// per the split above. Validated against the fused kernel by
// GDNScanTest.ChunkparMatchesFused (nonzero initial state, mild + hard
// decay heads — hard decay alone makes the H_0 coupling invisible).
//
// Scope: HD=SS=128 (every staged GDN checkpoint), single-sequence prefill,
// no d_real_n (padded verify chunks are tiny and stay on the fused kernel).
// StateT float or __nv_bfloat16 for the committed pool state; the state stays
// FP32 in registers within a strip and in the FP32 side buffer across strips,
// so a scan rounds to BF16 exactly once (at the final commit), matching the
// fused kernel's behaviour.
// ---------------------------------------------------------------------------

namespace chunkpar {
namespace {

// ---------------------------------------------------------------------------
// Kernel 1 — per-(chunk, head) state-independent factors.
// Grid (n_chunks, n_heads), block HD threads.
//
// Shared-memory phases (one allocation, region 1 reused):
//   region 1 [2*kChunk*SS]: phase A = k~ | q~ staging; phase B/C = the solve
//                           histories U_A | W (k~/q~ are already folded into
//                           the Gram matrices and the global RHS copies).
//   region 2: KK -> T in place (phase B mma A operand) | QK -> P in place
//             (phase C mma A operand), both padded stride kChunk+4.
//   region 3: beta[kChunk], logD[kChunk+1].
// Phase A: float4-per-lane row loads, parallel per-token decay/beta, Gram
// matrices as 3xTF32 mma. Phase B: blockwise forward substitution (16-row
// diagonal blocks per thread in registers, off-diagonal updates as 3xTF32
// mma). Phase C: P @ W and P @ U_A as 3xTF32 mma.
// ---------------------------------------------------------------------------
// 2*HD threads: the solve has 2*HD independent columns (HD of U_A + HD of W),
// one per thread — 8 warps hide the smem/global latency that 4 could not
// (the 128-thread cut ran 215 us/launch with each thread walking BOTH chains).
template <int HD, int SS>
__global__ void __launch_bounds__(2 * HD, 1) gdn_chunkpar_intra_kernel(
    const float* __restrict__ conv_f32, const half* __restrict__ alpha_all,
    const half* __restrict__ beta_all, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ ws_base, int strip_t0, int strip_tokens,
    int n_heads, int n_groups, int conv_channels, int grouped_layout) {
    static_assert(HD == SS, "chunkpar assumes HD == SS");
    const int c = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int col = tid & (HD - 1);
    const bool w_half = tid >= HD;  // this thread owns a W column (else U_A)
    const int t0 = strip_t0 + c * kChunk;
    const int L = min(kChunk, strip_tokens - c * kChunk);
    if (L <= 0)
        return;

    const int g_idx = grouped_layout ? (h / (n_heads / n_groups)) : (h % n_groups);
    const int BC_size = n_groups * SS;
    const float A_h = A_log[h];
    const float dtb_h = dt_bias[h];

    const ChunkparWs ws = chunkpar_ws_layout<HD, SS>(ws_base, n_heads);
    const size_t slot = static_cast<size_t>(c) * n_heads + h;
    float* W_s = ws.W + slot * kChunk * SS;
    float* KD_s = ws.KD + slot * kChunk * SS;
    float* UA_s = ws.UA + slot * kChunk * HD;
    float* QE_s = ws.QE + slot * kChunk * SS;
    float* YA_s = ws.YA + slot * kChunk * HD;

    constexpr int SQ = kChunk + 4;  // padded stride of QK/P: the phase-C mma A operand
    extern __shared__ float smem[];
    float* s_k = smem;                     // [kChunk * SS]  phase A
    float* s_q = s_k + kChunk * SS;        // [kChunk * SS]  phase A
    float* s_u = s_k;                      // [kChunk * HD]  phase B/C alias
    float* s_w = s_q;                      // [kChunk * SS]  phase B/C alias
    float* s_kk = s_q + kChunk * SS;       // [kChunk * SQ]  KK, then T in place (phase-B mma A operand)
    float* s_qk = s_kk + kChunk * SQ;      // [kChunk * SQ]  QK, then P in place (phase-C mma A operand)
    float* s_beta = s_qk + kChunk * SQ;    // [kChunk]
    float* s_logD = s_beta + kChunk;       // [kChunk + 1]
    const int warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, tg = lane % 4;  // mma fragment coordinates

    // ---- phase A: load raw K, Q — a warp per row, one float4 per lane ----
    // (64 serial per-thread row loads used to be the biggest single item of
    // this phase: one exposed L2/DRAM round-trip per token.)
    static_assert(SS == 128, "float4-per-lane row load assumes SS == 128");
    for (int t = warp; t < L; t += 2 * HD / 32) {
        const float* row = conv_f32 + static_cast<size_t>(t0 + t) * conv_channels;
        *reinterpret_cast<float4*>(&s_q[swz128(t, lane * 4)]) = *reinterpret_cast<const float4*>(
            row + g_idx * SS + lane * 4);
        *reinterpret_cast<float4*>(&s_k[swz128(t, lane * 4)]) = *reinterpret_cast<const float4*>(
            row + BC_size + g_idx * SS + lane * 4);
    }
    // Per-token decay / learning rate — same formulas as gdn_scan_fused_kernel:
    // the transcendental part per token in parallel, the prefix sum (same
    // sequential order as before) on thread 0 after the barrier.
    if (tid < L) {
        float alpha_h = __half2float(alpha_all[static_cast<size_t>(t0 + tid) * n_heads + h]);
        float dt_val = alpha_h + dtb_h;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        s_logD[tid + 1] = fmaxf(A_h * dt_val, -20.0f);
        float beta_h = __half2float(beta_all[static_cast<size_t>(t0 + tid) * n_heads + h]);
        s_beta[tid] = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));
    }
    __syncthreads();
    if (tid == 0) {
        s_logD[0] = 0.0f;
        for (int t = 0; t < L; t++)
            s_logD[t + 1] += s_logD[t];
    }

    // Per-token L2 norm, one warp per token: rsqrtf(max(sum_sq, 1e-12)).
    {
        for (int t = warp; t < L; t += 2 * HD / 32) {
            float k_sq = 0.0f, q_sq = 0.0f;
#pragma unroll
            for (int i = lane; i < SS; i += 32) {
                k_sq += s_k[swz128(t, i)] * s_k[swz128(t, i)];
                q_sq += s_q[swz128(t, i)] * s_q[swz128(t, i)];
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                k_sq += __shfl_xor_sync(0xffffffffu, k_sq, off);
                q_sq += __shfl_xor_sync(0xffffffffu, q_sq, off);
            }
            const float k_inv = rsqrtf(fmaxf(k_sq, 1e-12f));
            const float q_inv = rsqrtf(fmaxf(q_sq, 1e-12f));
#pragma unroll
            for (int i = lane; i < SS; i += 32) {
                s_k[swz128(t, i)] *= k_inv;
                s_q[swz128(t, i)] *= q_inv;
            }
        }
    }
    __syncthreads();

    // Decay-scaled global copies (these survive the region-1 alias):
    //   KD  = D[t+1..L] k~          W(RHS) = beta_t D[0..t+1] k~
    //   QE  = D[0..t+1] q~          (phase C subtracts the P W sum in place)
    const float logD_L = s_logD[L];
    for (int t = 0; t < L; t++) {
        const float d_t1 = expf(s_logD[t + 1]);
        if (w_half) {
            QE_s[t * SS + col] = d_t1 * s_q[swz128(t, col)];
        } else {
            const float k_tc = s_k[swz128(t, col)];
            KD_s[t * SS + col] = expf(logD_L - s_logD[t + 1]) * k_tc;
            W_s[t * SS + col] = s_beta[t] * d_t1 * k_tc;
        }
    }
    if (tid == 0)
        ws.D0L[slot] = expf(logD_L);

    // Gram matrices KK = K~ K~^T and QK = Q~ K~^T (lower triangle incl. the
    // diagonal, zero elsewhere) as 3xTF32 mma: warp w takes matrix w/4 and
    // m-tile w%4 across all 8 n-tiles. Both feed the state (T and P
    // coefficients), hence the compensated form.
    {
        const bool is_qk = warp >= 4;
        const float* A = is_qk ? s_q : s_k;
        const int m0 = (warp % 4) * 16;
        float acc[kChunk / 8][4];
#pragma unroll
        for (int nt = 0; nt < kChunk / 8; nt++)
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
        for (int k0 = 0; k0 < SS; k0 += 8) {
            float a[4];
            a[0] = A[swz128(m0 + g, k0 + tg)];
            a[1] = A[swz128(m0 + g + 8, k0 + tg)];
            a[2] = A[swz128(m0 + g, k0 + tg + 4)];
            a[3] = A[swz128(m0 + g + 8, k0 + tg + 4)];
#pragma unroll
            for (int nt = 0; nt < kChunk / 8; nt++) {
                float b[2];  // B[k][n] = K~[n][k]
                b[0] = s_k[swz128(nt * 8 + g, k0 + tg)];
                b[1] = s_k[swz128(nt * 8 + g, k0 + tg + 4)];
                mma_frag<true>(acc[nt], a, b);
            }
        }
        float* dst = is_qk ? s_qk : s_kk;
        constexpr int ds = SQ;
        const int i0 = m0 + g, i1 = i0 + 8;
#pragma unroll
        for (int nt = 0; nt < kChunk / 8; nt++) {
            const int j = nt * 8 + 2 * tg;
            dst[i0 * ds + j] = (j <= i0 && i0 < L) ? acc[nt][0] : 0.0f;
            dst[i0 * ds + j + 1] = (j + 1 <= i0 && i0 < L) ? acc[nt][1] : 0.0f;
            dst[i1 * ds + j] = (j <= i1 && i1 < L) ? acc[nt][2] : 0.0f;
            dst[i1 * ds + j + 1] = (j + 1 <= i1 && i1 < L) ? acc[nt][3] : 0.0f;
        }
    }
    __syncthreads();  // region 1 is dead as k~/q~ from here on

    // ---- phase B: blockwise forward triangular solve for U_A and W ----
    // T[t][j] = beta_t D[j+1..t+1] KK[t][j] (j < t) is built once, in place
    // of KK, and P = D QK (independent of the solve) alongside it. The RHS of
    // both systems is staged into the histories (rows >= L zero). Then per
    // 16-row block: the off-diagonal update hist[b] -= T[b, <b] @ hist[<b] as
    // 3xTF32 mma (it feeds the state), one barrier, the 16x16 diagonal block
    // per thread in registers (its own column; the T reads are warp-broadcast),
    // one barrier. 8 barriers instead of 128, and the up-to-63-term serial
    // j-chain of the row-at-a-time form (45 us per CTA, 60% of this kernel)
    // becomes a tensor-core product.
    {
        constexpr int F4_PER_ROW = HD / 4;
        for (int idx = tid; idx < kChunk * F4_PER_ROW; idx += 2 * HD) {
            const int t = idx / F4_PER_ROW, c4 = (idx % F4_PER_ROW) * 4;
            float4 u = make_float4(0.0f, 0.0f, 0.0f, 0.0f), w = u;
            if (t < L) {
                const float* row = conv_f32 + static_cast<size_t>(t0 + t) * conv_channels;
                u = *reinterpret_cast<const float4*>(row + 2 * BC_size + h * HD + c4);
                const float bt = s_beta[t];
                u.x *= bt;
                u.y *= bt;
                u.z *= bt;
                u.w *= bt;
                w = *reinterpret_cast<const float4*>(&W_s[t * SS + c4]);  // RHS from phase A
            }
            *reinterpret_cast<float4*>(&s_u[swz128(t, c4)]) = u;
            *reinterpret_cast<float4*>(&s_w[swz128(t, c4)]) = w;
        }
        for (int idx = tid; idx < kChunk * kChunk; idx += 2 * HD) {
            const int i = idx / kChunk, j = idx % kChunk;
            float* kk = &s_kk[i * SQ + j];
            *kk = (j < i && i < L) ? s_beta[i] * expf(s_logD[i + 1] - s_logD[j + 1]) * *kk : 0.0f;
            if (j <= i && i < L)
                s_qk[i * SQ + j] *= expf(s_logD[i + 1] - s_logD[j + 1]);
        }
    }
    __syncthreads();
    float* const hist = w_half ? s_w : s_u;
    constexpr int BS = 16;
    static_assert(kChunk % BS == 0 && BS == 16, "diagonal block = one mma m-tile");
    for (int b = 0; b < kChunk / BS; b++) {
        const int r0b = b * BS;
        if (r0b >= L)
            break;
        if (b > 0) {
            // hist[r0b..r0b+16) -= T[r0b.., 0..r0b) @ hist[0..r0b): warp w takes
            // history w/4 (0 = U_A, 1 = W) and columns (w%4)*32 .. +32.
            float* H = (warp < 4) ? s_u : s_w;
            const int nbase = (warp % 4) * 32;
            float acc[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
                acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
            for (int k0 = 0; k0 < r0b; k0 += 8) {
                float a[4];
                a[0] = s_kk[(r0b + g) * SQ + k0 + tg];
                a[1] = s_kk[(r0b + g + 8) * SQ + k0 + tg];
                a[2] = s_kk[(r0b + g) * SQ + k0 + tg + 4];
                a[3] = s_kk[(r0b + g + 8) * SQ + k0 + tg + 4];
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    float bb[2];
                    bb[0] = H[swz128(k0 + tg, nbase + nt * 8 + g)];
                    bb[1] = H[swz128(k0 + tg + 4, nbase + nt * 8 + g)];
                    mma_frag<true>(acc[nt], a, bb);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int cb = nbase + nt * 8 + 2 * tg;
                float2* p0 = reinterpret_cast<float2*>(&H[swz128(r0b + g, cb)]);
                float2* p1 = reinterpret_cast<float2*>(&H[swz128(r0b + g + 8, cb)]);
                const float2 v0 = *p0, v1 = *p1;
                *p0 = make_float2(v0.x - acc[nt][0], v0.y - acc[nt][1]);
                *p1 = make_float2(v1.x - acc[nt][2], v1.y - acc[nt][3]);
            }
            __syncthreads();
        }
        {
            float x[BS];
#pragma unroll
            for (int i = 0; i < BS; i++)
                x[i] = hist[swz128(r0b + i, col)];
#pragma unroll
            for (int i = 1; i < BS; i++) {
                const float* Ti = &s_kk[(r0b + i) * SQ + r0b];
#pragma unroll
                for (int j = 0; j < i; j++)
                    x[i] -= Ti[j] * x[j];
            }
#pragma unroll
            for (int i = 0; i < BS; i++)
                hist[swz128(r0b + i, col)] = x[i];
        }
        __syncthreads();
    }
    {
        constexpr int F4 = kChunk * SS / 4, F4R = SS / 4;
        const float4* su4 = reinterpret_cast<const float4*>(s_u);
        const float4* sw4 = reinterpret_cast<const float4*>(s_w);
        float4* ua4 = reinterpret_cast<float4*>(UA_s);
        float4* w4 = reinterpret_cast<float4*>(W_s);
        for (int idx = tid; idx < 2 * F4; idx += 2 * HD) {
            const int e = (idx < F4) ? idx : idx - F4;
            const int t = e / F4R, c4 = e % F4R;
            const int src = t * F4R + (c4 ^ ((t & 7) << 1));  // de-swizzle (float4 units)
            if (idx < F4)
                ua4[e] = su4[src];
            else
                w4[e] = sw4[src];
        }
    }

    // ---- phase C: Qeff (in place on QE) and Y_A on tensor cores ----
    // P[t][j] = D[j+1..t+1] QK[t][j] (j <= t, built with T above), then
    //   Qeff = D q~ - P @ W      Y_A = P @ U_A
    // as two [L x 128] GEMMs over K = j. Output terms, but NOT tf32-safe:
    // Qeff is the difference of two O(1) terms (D q~ and P W cancel), so the
    // 10-bit operand rounding on P W becomes an O(1e-2) relative error on
    // Qeff - measured PPL 6.8122 -> 6.8845 (+0.9%) with plain tf32 here.
    // 3xTF32 on both. History rows [L, kChunk) are zero from the staging, so
    // the K range may run to the next multiple of 8; the solve loop ended
    // with a barrier.
    const int kmax = (L + 7) & ~7;
    {
        const int m0 = (warp % 4) * 16;           // t rows of this warp
        const int nbase = (warp / 4) * (HD / 2);  // its 64 output columns
        if (m0 < L) {
            const int r0 = m0 + g, r1 = r0 + 8;
#pragma unroll 1
            for (int which = 0; which < 2; which++) {  // 0: P @ W -> Qeff, 1: P @ U_A -> Y_A
                const float* B = which ? s_u : s_w;
                float acc[HD / 16][4];
#pragma unroll
                for (int nt = 0; nt < HD / 16; nt++)
                    acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
                for (int k0 = 0; k0 < kmax; k0 += 8) {
                    float a[4];
                    a[0] = s_qk[(m0 + g) * SQ + k0 + tg];
                    a[1] = s_qk[(m0 + g + 8) * SQ + k0 + tg];
                    a[2] = s_qk[(m0 + g) * SQ + k0 + tg + 4];
                    a[3] = s_qk[(m0 + g + 8) * SQ + k0 + tg + 4];
#pragma unroll
                    for (int nt = 0; nt < HD / 16; nt++) {
                        float b[2];
                        b[0] = B[swz128(k0 + tg, nbase + nt * 8 + g)];
                        b[1] = B[swz128(k0 + tg + 4, nbase + nt * 8 + g)];
                        mma_frag<true>(acc[nt], a, b);
                    }
                }
#pragma unroll
                for (int nt = 0; nt < HD / 16; nt++) {
                    const int cb = nbase + nt * 8 + 2 * tg;
                    if (which == 0) {
                        if (r0 < L) {
                            float2* q = reinterpret_cast<float2*>(&QE_s[r0 * SS + cb]);
                            const float2 v = *q;
                            *q = make_float2(v.x - acc[nt][0], v.y - acc[nt][1]);
                        }
                        if (r1 < L) {
                            float2* q = reinterpret_cast<float2*>(&QE_s[r1 * SS + cb]);
                            const float2 v = *q;
                            *q = make_float2(v.x - acc[nt][2], v.y - acc[nt][3]);
                        }
                    } else {
                        if (r0 < L)
                            *reinterpret_cast<float2*>(&YA_s[r0 * HD + cb]) =
                                make_float2(acc[nt][0], acc[nt][1]);
                        if (r1 < L)
                            *reinterpret_cast<float2*>(&YA_s[r1 * HD + cb]) =
                                make_float2(acc[nt][2], acc[nt][3]);
                    }
                }
            }
        }
    }
}


}  // namespace

void chunkpar_intra_128(const float* conv_f32, const half* alpha, const half* beta, const float* A_log,
                        const float* dt_bias, float* ws_base, int strip_t0, int strip_tokens, int n_chunks,
                        int n_heads, int n_groups, int conv_channels, int grouped_layout,
                        cudaStream_t stream) {
    constexpr int HD = 128, SS = 128;
    constexpr size_t smem = (2 * kChunk * SS + 2 * kChunk * (kChunk + 4) + kChunk + (kChunk + 1)) *
                            sizeof(float);
    static std::once_flag attr_once;
    std::call_once(attr_once, [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&gdn_chunkpar_intra_kernel<HD, SS>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    });
    gdn_chunkpar_intra_kernel<HD, SS>
        <<<dim3(n_chunks, n_heads), 2 * HD, smem, stream>>>(conv_f32, alpha, beta, A_log, dt_bias, ws_base,
                                                            strip_t0, strip_tokens, n_heads, n_groups,
                                                            conv_channels, grouped_layout);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace chunkpar

namespace {

using namespace chunkpar;

// Strip length for this n_heads. Two constraints: kernel 1 runs strip x
// n_heads CTAs at one CTA per SM, so the strip decides how full the last wave
// is (48 heads x 8 = 384 CTAs = 2.26 waves on 170 SMs: the third wave runs
// 26% of the SMs); and kernel 2 re-reads the five [64 x 128] FP32 factor
// blocks per (chunk, head) that kernel 1 just wrote, so the strip's factor
// set (strip x n_heads x 160 KB) must stay L2-resident - at 16 x 48 = 126 MB
// it does not, and kernel 2 read +11% slower from DRAM while kernel 1 read
// -21% (Qwen3.8-27B pp4096, 2026-09-02). Auto: the strip in [4, cap] with
// the fullest last wave, the larger one on ties, cap = the largest strip
// whose factor set fits two thirds of L2 (max kMaxStripChunks).
int chunkpar_strip_chunks(int n_heads, int requested) {
    if (requested > 0)
        return std::min(requested, kMaxStripChunks);
    static int sm_count = 0, l2_bytes = 0;
    if (sm_count == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev);
        if (sm_count <= 0)
            sm_count = 170;
        if (l2_bytes <= 0)
            l2_bytes = 96 << 20;
    }
    const size_t factor_bytes_per_chunk = static_cast<size_t>(5) * kChunk * 128 * sizeof(float) * n_heads;
    const size_t l2_budget = static_cast<size_t>(l2_bytes) * 2 / 3;
    int cap = static_cast<int>(l2_budget / factor_bytes_per_chunk);
    cap = std::max(1, std::min(cap, kMaxStripChunks));
    int best = cap;
    double best_eff = -1.0;
    for (int sc = std::min(4, cap); sc <= cap; sc++) {
        const int ctas = sc * n_heads;
        const int waves = (ctas + sm_count - 1) / sm_count;
        const double eff = static_cast<double>(ctas) / (static_cast<double>(waves) * sm_count);
        if (eff >= best_eff - 1e-9) {
            best_eff = eff;
            best = sc;
        }
    }
    return best;
}

template <typename StateT>
void chunkpar_launch(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                     const float* A_log, const float* dt_bias, StateT* h_state, half* y, int n_tokens,
                     int n_heads, int n_groups, cudaStream_t stream, int grouped_layout, float* ws_base,
                     int strip_chunks) {
    const int strip_len = strip_chunks * kChunk;
    const int n_strips = (n_tokens + strip_len - 1) / strip_len;
    for (int si = 0; si < n_strips; si++) {
        const int t0 = si * strip_len;
        const int strip_tokens = min(strip_len, n_tokens - t0);
        const int n_chunks = (strip_tokens + kChunk - 1) / kChunk;
        chunkpar_intra_128(conv_f32, alpha, beta, A_log, dt_bias, ws_base, t0, strip_tokens, n_chunks,
                           n_heads, n_groups, conv_channels, grouped_layout, stream);
        chunkpar_pass_128<StateT>(ws_base, h_state, y, t0, strip_tokens, n_chunks, n_heads,
                                  /*load_statet=*/si == 0 ? 1 : 0,
                                  /*store_statet=*/si == n_strips - 1 ? 1 : 0, stream);
    }
}

}  // namespace

size_t gdn_scan_chunkpar_workspace_bytes(int n_heads) {
    const size_t slots = static_cast<size_t>(chunkpar::kMaxStripChunks) * n_heads;
    return (5 * slots * kChunk * 128 + slots + static_cast<size_t>(n_heads) * 128 * 128) * sizeof(float);
}

void gdn_scan_chunkpar_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                           const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                           int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                           int grouped_layout, float* ws, size_t ws_bytes, int strip_chunks) {
    if (head_dim_ssm != 128 || state_size != 128)
        throw std::runtime_error("gdn_scan_chunkpar_f32: no kernel for HD=" + std::to_string(head_dim_ssm) +
                                 " SS=" + std::to_string(state_size));
    if (!ws || ws_bytes < gdn_scan_chunkpar_workspace_bytes(n_heads))
        throw std::runtime_error("gdn_scan_chunkpar_f32: workspace too small");
    chunkpar_launch<float>(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                           n_heads, n_groups, stream, grouped_layout, ws,
                           chunkpar_strip_chunks(n_heads, strip_chunks));
}

void gdn_scan_chunkpar_bf16(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, __nv_bfloat16* h_state, half* y,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int grouped_layout, float* ws, size_t ws_bytes,
                            int strip_chunks) {
    if (head_dim_ssm != 128 || state_size != 128)
        throw std::runtime_error("gdn_scan_chunkpar_bf16: no kernel for HD=" + std::to_string(head_dim_ssm) +
                                 " SS=" + std::to_string(state_size));
    if (!ws || ws_bytes < gdn_scan_chunkpar_workspace_bytes(n_heads))
        throw std::runtime_error("gdn_scan_chunkpar_bf16: workspace too small");
    chunkpar_launch<__nv_bfloat16>(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y, n_tokens,
                                   n_heads, n_groups, stream, grouped_layout, ws,
                                   chunkpar_strip_chunks(n_heads, strip_chunks));
}

}  // namespace imp
