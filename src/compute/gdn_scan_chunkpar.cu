#include "compute/gdn_internal.cuh"
#include "core/logging.h"

#include <cuda_bf16.h>
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

namespace {

constexpr int kChunk = 64;       // tokens per chunk (WY tile)
constexpr int kStripChunks = 8;  // chunks per strip => 512 tokens per strip
constexpr int kColSplit = 4;     // K2 CTAs per head (state columns split): 32 columns each

// Workspace float layout, per strip slot (slot = c * n_heads + h):
//   W    [slots][kChunk*SS]   phase A: solve RHS; phase B: solved W
//   KD   [slots][kChunk*SS]
//   UA   [slots][kChunk*HD]
//   QE   [slots][kChunk*SS]   phase A: D[0..t+1] q~; phase C: finished Qeff
//   YA   [slots][kChunk*HD]
//   D0L  [slots]
//   H32  [n_heads*SS*HD]      FP32 inter-strip state
struct ChunkparWs {
    float* W;
    float* KD;
    float* UA;
    float* QE;
    float* YA;
    float* D0L;
    float* H32;
};

template <int HD, int SS>
__host__ __device__ inline ChunkparWs chunkpar_ws_layout(float* base, int n_heads) {
    const size_t slots = static_cast<size_t>(kStripChunks) * n_heads;
    const size_t arr = slots * kChunk * SS;  // SS == HD
    ChunkparWs w;
    w.W = base;
    w.KD = base + arr;
    w.UA = base + 2 * arr;
    w.QE = base + 3 * arr;
    w.YA = base + 4 * arr;
    w.D0L = base + 5 * arr;
    w.H32 = w.D0L + slots;
    return w;
}

// mma.sync m16n8k8 tf32 helpers, shared by both kernels.
__device__ __forceinline__ uint32_t f32_to_tf32(float f) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(f));
    return r;
}

__device__ __forceinline__ void mma_tf32_16x8x8(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Fragment MMA on FP32 operands. X3 = the 3xTF32 error-compensated form
// (a = a_hi + a_lo, three MMAs: a_lo*b_hi + a_hi*b_lo + a_hi*b_hi), ~FP32
// accuracy on the products; plain tf32 otherwise. Plain tf32 on all three
// chunk GEMMs read PPL +0.13% on Qwen3.6-35B (6.8216 -> 6.8304): the state
// path compounds the 10-bit operand rounding across chunks.
template <bool X3>
__device__ __forceinline__ void mma_frag(float* c, const float* a, const float* b) {
    uint32_t ah[4], bh[2];
#pragma unroll
    for (int i = 0; i < 4; i++)
        ah[i] = f32_to_tf32(a[i]);
#pragma unroll
    for (int i = 0; i < 2; i++)
        bh[i] = f32_to_tf32(b[i]);
    if constexpr (X3) {
        uint32_t al[4], bl[2];
#pragma unroll
        for (int i = 0; i < 4; i++)
            al[i] = f32_to_tf32(a[i] - __uint_as_float(ah[i]));
#pragma unroll
        for (int i = 0; i < 2; i++)
            bl[i] = f32_to_tf32(b[i] - __uint_as_float(bh[i]));
        mma_tf32_16x8x8(c, al, bh);
        mma_tf32_16x8x8(c, ah, bl);
    }
    mma_tf32_16x8x8(c, ah, bh);
}

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
        *reinterpret_cast<float4*>(&s_q[t * SS + lane * 4]) =
            *reinterpret_cast<const float4*>(row + g_idx * SS + lane * 4);
        *reinterpret_cast<float4*>(&s_k[t * SS + lane * 4]) =
            *reinterpret_cast<const float4*>(row + BC_size + g_idx * SS + lane * 4);
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
                k_sq += s_k[t * SS + i] * s_k[t * SS + i];
                q_sq += s_q[t * SS + i] * s_q[t * SS + i];
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
                s_k[t * SS + i] *= k_inv;
                s_q[t * SS + i] *= q_inv;
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
            QE_s[t * SS + col] = d_t1 * s_q[t * SS + col];
        } else {
            KD_s[t * SS + col] = expf(logD_L - s_logD[t + 1]) * s_k[t * SS + col];
            W_s[t * SS + col] = s_beta[t] * d_t1 * s_k[t * SS + col];
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
            a[0] = A[(m0 + g) * SS + k0 + tg];
            a[1] = A[(m0 + g + 8) * SS + k0 + tg];
            a[2] = A[(m0 + g) * SS + k0 + tg + 4];
            a[3] = A[(m0 + g + 8) * SS + k0 + tg + 4];
#pragma unroll
            for (int nt = 0; nt < kChunk / 8; nt++) {
                float b[2];  // B[k][n] = K~[n][k]
                b[0] = s_k[(nt * 8 + g) * SS + k0 + tg];
                b[1] = s_k[(nt * 8 + g) * SS + k0 + tg + 4];
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
            *reinterpret_cast<float4*>(&s_u[t * HD + c4]) = u;
            *reinterpret_cast<float4*>(&s_w[t * HD + c4]) = w;
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
                    bb[0] = H[(k0 + tg) * HD + nbase + nt * 8 + g];
                    bb[1] = H[(k0 + tg + 4) * HD + nbase + nt * 8 + g];
                    mma_frag<true>(acc[nt], a, bb);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int cb = nbase + nt * 8 + 2 * tg;
                float2* p0 = reinterpret_cast<float2*>(&H[(r0b + g) * HD + cb]);
                float2* p1 = reinterpret_cast<float2*>(&H[(r0b + g + 8) * HD + cb]);
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
                x[i] = hist[(r0b + i) * HD + col];
#pragma unroll
            for (int i = 1; i < BS; i++) {
                const float* Ti = &s_kk[(r0b + i) * SQ + r0b];
#pragma unroll
                for (int j = 0; j < i; j++)
                    x[i] -= Ti[j] * x[j];
            }
#pragma unroll
            for (int i = 0; i < BS; i++)
                hist[(r0b + i) * HD + col] = x[i];
        }
        __syncthreads();
    }
    {
        constexpr int F4 = kChunk * SS / 4;
        const float4* su4 = reinterpret_cast<const float4*>(s_u);
        const float4* sw4 = reinterpret_cast<const float4*>(s_w);
        float4* ua4 = reinterpret_cast<float4*>(UA_s);
        float4* w4 = reinterpret_cast<float4*>(W_s);
        for (int idx = tid; idx < 2 * F4; idx += 2 * HD) {
            if (idx < F4)
                ua4[idx] = su4[idx];
            else
                w4[idx - F4] = sw4[idx - F4];
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
                        b[0] = B[(k0 + tg) * HD + nbase + nt * 8 + g];
                        b[1] = B[(k0 + tg + 4) * HD + nbase + nt * 8 + g];
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

// ---------------------------------------------------------------------------
// Kernel 2 — sequential inter-chunk state pass + outputs, on tensor cores.
// Grid (n_heads, kColSplit): blockIdx.y owns COLS = HD/kColSplit state columns;
// u_eff, y and the state update are column-local, so the split only re-stages
// the (read-only) W / Qeff / K_d rows per CTA. 128 threads = 4 warps.
//
// Per chunk, three GEMMs as mma.sync m16n8k8 tf32 with FP32 accumulate:
//   u_eff [L x COLS]  = U_A - W    [L x SS] @ H [SS x COLS]
//   y     [L x COLS]  = (Y_A + Qeff [L x SS] @ H) * scale
//   H'    [SS x COLS] = D0L * H + K_d^T [SS x L] @ u_eff [L x COLS]
// H lives in shared memory (FP32): the B operand of the first two GEMMs and
// the C-init / D of the third. The carried state itself is never rounded —
// D0L*H enters the accumulator in FP32 — only the per-chunk increments see
// the tf32 operand rounding (10-bit mantissa). The scalar float4 form of this
// kernel ran 242 us per 512-token strip at 6.6 TFLOPS; the three GEMMs are
// 1.6 GFLOP per strip and head-set.
// Shared strides are padded so the fragment loads are bank-conflict free
// (SA = SS + 4 for the [row][k] staging, SH = COLS + 8 for the [k][n] tiles).
// ---------------------------------------------------------------------------
// C[16 x NTILE*8] = s_a[m0.., 0..SS) @ H[0..SS, 0..NTILE*8) for one warp's
// 16-row strip, mma C layout. (A free function: nvcc 13.3 segfaults on a
// generic lambda carrying the X3 tag inside the kernel.)
template <int SS, int SA, int SH, int NTILE, bool X3>
__device__ __forceinline__ void gemm_rows_x_h(const float* __restrict__ s_a, const float* __restrict__ s_h,
                                              int m0, int g, int tg, float (&acc)[NTILE][4]) {
#pragma unroll
    for (int nt = 0; nt < NTILE; nt++)
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    for (int k0 = 0; k0 < SS; k0 += 8) {
        float a[4];
        a[0] = s_a[(m0 + g) * SA + k0 + tg];
        a[1] = s_a[(m0 + g + 8) * SA + k0 + tg];
        a[2] = s_a[(m0 + g) * SA + k0 + tg + 4];
        a[3] = s_a[(m0 + g + 8) * SA + k0 + tg + 4];
#pragma unroll
        for (int nt = 0; nt < NTILE; nt++) {
            float b[2];
            b[0] = s_h[(k0 + tg) * SH + nt * 8 + g];
            b[1] = s_h[(k0 + tg + 4) * SH + nt * 8 + g];
            mma_frag<X3>(acc[nt], a, b);
        }
    }
}

template <int HD, int SS, typename StateT>
__global__ void __launch_bounds__(128, 1) gdn_chunkpar_pass_kernel(
    float* __restrict__ ws_base, StateT* __restrict__ h_state, half* __restrict__ y_out,
    int strip_t0, int strip_tokens, int n_chunks, int n_heads, int load_statet, int store_statet) {
    constexpr int COLS = HD / kColSplit;  // state columns this CTA owns
    constexpr int SA = SS + 4;            // padded stride of the staged [row][k] block
    constexpr int SH = COLS + 8;          // padded stride of H [k][n] and u_eff [k][n]
    constexpr int NT = 128;
    constexpr int NTILE = COLS / 8;       // n-tiles per row strip
    static_assert(COLS % 8 == 0 && SS % 64 == 0 && kChunk % 16 == 0, "mma tiling");
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
    float* s_a = smem;               // [kChunk][SA]  staging: W, then Qeff, then K_d
    float* s_h = s_a + kChunk * SA;  // [SS][SH]      state slice, FP32
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

    // rows [0, L) of a [kChunk x SS] row-major global block -> s_a (padded stride)
    auto stage = [&](const float* src_rows, int L) {
        constexpr int F4_PER_ROW = SS / 4;
        const float4* src = reinterpret_cast<const float4*>(src_rows);
        for (int idx = tid; idx < L * F4_PER_ROW; idx += NT) {
            const int r = idx / F4_PER_ROW, c4 = idx % F4_PER_ROW;
            *reinterpret_cast<float4*>(&s_a[r * SA + c4 * 4]) = src[idx];
        }
    };

    for (int c = 0; c < n_chunks; c++) {
        const int L = min(kChunk, strip_tokens - c * kChunk);
        const size_t slot = static_cast<size_t>(c) * n_heads + h;
        const float* UA_s = ws.UA + slot * kChunk * HD;
        const float* YA_s = ws.YA + slot * kChunk * HD;
        const int m0 = warp * 16;

        // -- u_eff = U_A - W @ H --
        stage(ws.W + slot * kChunk * SS, L);
        if (L < kChunk)  // tail rows are K operands of the state update: zero them
            for (int idx = tid; idx < (kChunk - L) * COLS; idx += NT)
                s_ue[(L + idx / COLS) * SH + idx % COLS] = 0.0f;
        __syncthreads();
        if (m0 < L) {
            float acc[NTILE][4];
            gemm_rows_x_h<SS, SA, SH, NTILE, true>(s_a, s_h, m0, g, tg, acc);  // feeds the state: 3xTF32
            const int r0 = m0 + g, r1 = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < NTILE; nt++) {
                const int col = nt * 8 + 2 * tg;
                if (r0 < L) {
                    const float2 u = *reinterpret_cast<const float2*>(&UA_s[r0 * HD + d_base + col]);
                    s_ue[r0 * SH + col] = u.x - acc[nt][0];
                    s_ue[r0 * SH + col + 1] = u.y - acc[nt][1];
                }
                if (r1 < L) {
                    const float2 u = *reinterpret_cast<const float2*>(&UA_s[r1 * HD + d_base + col]);
                    s_ue[r1 * SH + col] = u.x - acc[nt][2];
                    s_ue[r1 * SH + col + 1] = u.y - acc[nt][3];
                }
            }
        }
        __syncthreads();

        // -- y = (Y_A + Qeff @ H) * scale, from the PRE-update state --
        stage(ws.QE + slot * kChunk * SS, L);
        __syncthreads();
        if (m0 < L) {
            float acc[NTILE][4];
            gemm_rows_x_h<SS, SA, SH, NTILE, false>(s_a, s_h, m0, g, tg, acc);  // output only: tf32
            const size_t t_base = static_cast<size_t>(strip_t0) + c * kChunk;
            const int r0 = m0 + g, r1 = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < NTILE; nt++) {
                const int col = nt * 8 + 2 * tg;
                if (r0 < L) {
                    const float2 ya = *reinterpret_cast<const float2*>(&YA_s[r0 * HD + d_base + col]);
                    *reinterpret_cast<__half2*>(&y_out[(t_base + r0) * inner + h * HD + d_base + col]) =
                        __floats2half2_rn((ya.x + acc[nt][0]) * scale, (ya.y + acc[nt][1]) * scale);
                }
                if (r1 < L) {
                    const float2 ya = *reinterpret_cast<const float2*>(&YA_s[r1 * HD + d_base + col]);
                    *reinterpret_cast<__half2*>(&y_out[(t_base + r1) * inner + h * HD + d_base + col]) =
                        __floats2half2_rn((ya.x + acc[nt][2]) * scale, (ya.y + acc[nt][3]) * scale);
                }
            }
        }
        __syncthreads();

        // -- H = D0L * H + K_d^T @ u_eff --
        stage(ws.KD + slot * kChunk * SS, L);
        const int kmax = (L + 7) & ~7;
        if (L < kmax)  // the partial k-block: its K_d rows must be finite (u_eff rows are zero)
            for (int idx = tid; idx < (kmax - L) * SS; idx += NT)
                s_a[(L + idx / SS) * SA + idx % SS] = 0.0f;
        __syncthreads();
        {
            const float d0l = ws.D0L[slot];
            constexpr int MT = SS / 16 / 4;  // m-tiles (16 state rows each) per warp
            const int mbase = warp * MT * 16;
            float acc[MT][NTILE][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTILE; nt++) {
                    const int r0 = mbase + mt * 16 + g, col = nt * 8 + 2 * tg;
                    acc[mt][nt][0] = d0l * s_h[r0 * SH + col];
                    acc[mt][nt][1] = d0l * s_h[r0 * SH + col + 1];
                    acc[mt][nt][2] = d0l * s_h[(r0 + 8) * SH + col];
                    acc[mt][nt][3] = d0l * s_h[(r0 + 8) * SH + col + 1];
                }
            for (int k0 = 0; k0 < kmax; k0 += 8) {
                float b[NTILE][2];
#pragma unroll
                for (int nt = 0; nt < NTILE; nt++) {
                    b[nt][0] = s_ue[(k0 + tg) * SH + nt * 8 + g];
                    b[nt][1] = s_ue[(k0 + tg + 4) * SH + nt * 8 + g];
                }
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    // A[m = s][k = t] = K_d[t][s], staged row-major as [t][s]
                    const int r = mbase + mt * 16 + g;
                    float a[4];
                    a[0] = s_a[(k0 + tg) * SA + r];
                    a[1] = s_a[(k0 + tg) * SA + r + 8];
                    a[2] = s_a[(k0 + tg + 4) * SA + r];
                    a[3] = s_a[(k0 + tg + 4) * SA + r + 8];
#pragma unroll
                    for (int nt = 0; nt < NTILE; nt++)
                        mma_frag<true>(acc[mt][nt], a, b[nt]);  // state update: 3xTF32
                }
            }
            // Each warp owns its 32 state rows: no cross-warp hazard before the barrier.
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NTILE; nt++) {
                    const int r0 = mbase + mt * 16 + g, col = nt * 8 + 2 * tg;
                    s_h[r0 * SH + col] = acc[mt][nt][0];
                    s_h[r0 * SH + col + 1] = acc[mt][nt][1];
                    s_h[(r0 + 8) * SH + col] = acc[mt][nt][2];
                    s_h[(r0 + 8) * SH + col + 1] = acc[mt][nt][3];
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

template <int HD, int SS, typename StateT>
void chunkpar_launch(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                     const float* A_log, const float* dt_bias, StateT* h_state, half* y, int n_tokens,
                     int n_heads, int n_groups, cudaStream_t stream, int grouped_layout, float* ws_base) {
    const size_t smem1 =
        (2 * kChunk * SS + 2 * kChunk * (kChunk + 4) + kChunk + (kChunk + 1)) * sizeof(float);
    const size_t smem2 =
        (kChunk * (SS + 4) + SS * (HD / kColSplit + 8) + kChunk * (HD / kColSplit + 8)) * sizeof(float);
    static std::once_flag attr_once;
    std::call_once(attr_once, [&] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&gdn_chunkpar_intra_kernel<HD, SS>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem1));
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&gdn_chunkpar_pass_kernel<HD, SS, StateT>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem2));
    });

    const int strip_len = kStripChunks * kChunk;
    const int n_strips = (n_tokens + strip_len - 1) / strip_len;
    for (int si = 0; si < n_strips; si++) {
        const int t0 = si * strip_len;
        const int strip_tokens = min(strip_len, n_tokens - t0);
        const int n_chunks = (strip_tokens + kChunk - 1) / kChunk;
        gdn_chunkpar_intra_kernel<HD, SS><<<dim3(n_chunks, n_heads), 2 * HD, smem1, stream>>>(
            conv_f32, alpha, beta, A_log, dt_bias, ws_base, t0, strip_tokens, n_heads, n_groups,
            conv_channels, grouped_layout);
        IMP_CUDA_CHECK_LAUNCH();
        gdn_chunkpar_pass_kernel<HD, SS, StateT><<<dim3(n_heads, kColSplit), 128, smem2, stream>>>(
                ws_base, h_state, y, t0, strip_tokens, n_chunks, n_heads,
                /*load_statet=*/si == 0 ? 1 : 0, /*store_statet=*/si == n_strips - 1 ? 1 : 0);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace

size_t gdn_scan_chunkpar_workspace_bytes(int n_heads) {
    const size_t slots = static_cast<size_t>(kStripChunks) * n_heads;
    return (5 * slots * kChunk * 128 + slots + static_cast<size_t>(n_heads) * 128 * 128) * sizeof(float);
}

void gdn_scan_chunkpar_f32(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                           const float* A_log, const float* dt_bias, float* h_state, half* y, int n_tokens,
                           int n_heads, int head_dim_ssm, int state_size, int n_groups, cudaStream_t stream,
                           int grouped_layout, float* ws, size_t ws_bytes) {
    if (head_dim_ssm != 128 || state_size != 128)
        throw std::runtime_error("gdn_scan_chunkpar_f32: no kernel for HD=" + std::to_string(head_dim_ssm) +
                                 " SS=" + std::to_string(state_size));
    if (!ws || ws_bytes < gdn_scan_chunkpar_workspace_bytes(n_heads))
        throw std::runtime_error("gdn_scan_chunkpar_f32: workspace too small");
    chunkpar_launch<128, 128, float>(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state, y,
                                     n_tokens, n_heads, n_groups, stream, grouped_layout, ws);
}

void gdn_scan_chunkpar_bf16(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                            const float* A_log, const float* dt_bias, __nv_bfloat16* h_state, half* y,
                            int n_tokens, int n_heads, int head_dim_ssm, int state_size, int n_groups,
                            cudaStream_t stream, int grouped_layout, float* ws, size_t ws_bytes) {
    if (head_dim_ssm != 128 || state_size != 128)
        throw std::runtime_error("gdn_scan_chunkpar_bf16: no kernel for HD=" + std::to_string(head_dim_ssm) +
                                 " SS=" + std::to_string(state_size));
    if (!ws || ws_bytes < gdn_scan_chunkpar_workspace_bytes(n_heads))
        throw std::runtime_error("gdn_scan_chunkpar_bf16: workspace too small");
    chunkpar_launch<128, 128, __nv_bfloat16>(conv_f32, conv_channels, alpha, beta, A_log, dt_bias, h_state,
                                             y, n_tokens, n_heads, n_groups, stream, grouped_layout, ws);
}

}  // namespace imp
