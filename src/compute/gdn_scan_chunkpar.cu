#include "compute/gdn_internal.cuh"
#include "core/logging.h"

#include <cuda_bf16.h>
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
//   - Shipped state (2026-09-01, Qwen3.6-35B class kernel sums): pp512
//     144.5 -> 98.5 ms (-32%), pp4096 1485 -> 786 ms (-47%); K1 196 us, K2
//     242 us per 512-token strip. Still ~4x over the compute floor: ncu says
//     short_scoreboard 2.4 + barrier 2.1 stalls/issue in K1 — the remaining
//     lever is an MMA form of the three chunk matmuls, not more scalar ILP.
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
constexpr int kColSplit = 2;     // K2 CTAs per head (state columns split)

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

// ---------------------------------------------------------------------------
// Kernel 1 — per-(chunk, head) state-independent factors.
// Grid (n_chunks, n_heads), block HD threads.
//
// Shared-memory phases (one allocation, region 1 reused):
//   region 1 [2*kChunk*SS]: phase A = k~ | q~ staging; phase B/C = the solve
//                           histories U_A | W (k~/q~ are already folded into
//                           the Gram matrices and the global RHS copies).
//   region 2 [2*kChunk*kChunk]: KK | QK Gram matrices (whole kernel).
//   region 3: beta[kChunk], logD[kChunk+1], row scratch[kChunk].
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

    extern __shared__ float smem[];
    float* s_k = smem;                       // [kChunk * SS]  phase A
    float* s_q = s_k + kChunk * SS;          // [kChunk * SS]  phase A
    float* s_u = s_k;                        // [kChunk * HD]  phase B/C alias
    float* s_w = s_q;                        // [kChunk * SS]  phase B/C alias
    float* s_kk = s_q + kChunk * SS;         // [kChunk * kChunk]
    float* s_qk = s_kk + kChunk * kChunk;    // [kChunk * kChunk]
    float* s_beta = s_qk + kChunk * kChunk;  // [kChunk]
    float* s_logD = s_beta + kChunk;         // [kChunk + 1]
    float* s_row = s_logD + kChunk + 1;      // [kChunk] T/P row scratch

    // ---- phase A: load raw K, Q (half the threads each) ----
    for (int t = 0; t < L; t++) {
        const float* row = conv_f32 + static_cast<size_t>(t0 + t) * conv_channels;
        if (w_half)
            s_q[t * SS + col] = row[g_idx * SS + col];
        else
            s_k[t * SS + col] = row[BC_size + g_idx * SS + col];
    }
    __syncthreads();

    // Per-token decay / learning rate (thread 0, L small) — same formulas as
    // gdn_scan_fused_kernel.
    if (tid == 0) {
        float log_D = 0.0f;
        s_logD[0] = 0.0f;
        for (int t = 0; t < L; t++) {
            float alpha_h = __half2float(alpha_all[static_cast<size_t>(t0 + t) * n_heads + h]);
            float dt_val = alpha_h + dtb_h;
            dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
            log_D += fmaxf(A_h * dt_val, -20.0f);
            s_logD[t + 1] = log_D;
            float beta_h = __half2float(beta_all[static_cast<size_t>(t0 + t) * n_heads + h]);
            s_beta[t] = 1.0f / (1.0f + expf(-fmaxf(fminf(beta_h, 20.0f), -20.0f)));
        }
    }

    // Per-token L2 norm, one warp per token: rsqrtf(max(sum_sq, 1e-12)).
    {
        const int warp = tid / 32, lane = tid % 32;
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

    // Gram matrices (lower triangle incl. diagonal).
    for (int idx = tid; idx < kChunk * kChunk; idx += 2 * HD) {
        const int i = idx / kChunk, j = idx % kChunk;
        if (j > i || i >= L) {
            s_kk[idx] = 0.0f;
            s_qk[idx] = 0.0f;
            continue;
        }
        float kk = 0.0f, qk = 0.0f;
#pragma unroll 8
        for (int s = 0; s < SS; s++) {
            const float k_js = s_k[j * SS + s];
            kk += s_k[i * SS + s] * k_js;
            qk += s_q[i * SS + s] * k_js;
        }
        s_kk[idx] = kk;
        s_qk[idx] = qk;
    }
    __syncthreads();  // region 1 is dead as k~/q~ from here on

    // ---- phase B: forward triangular solve for U_A and W ----
    // T[t,j] = beta_t * exp(logD[t+1]-logD[j+1]) * KK[t,j]. One column per
    // thread (U_A half / W half); histories in shared memory (region 1). The
    // per-step global reads (v, the W RHS) are issued BEFORE the row barrier
    // so their latency overlaps the row build; the global copies of the
    // solutions are written once, batched, after the loop.
    float* const hist = w_half ? s_w : s_u;
    for (int t = 0; t < L; t++) {
        float rhs;
        if (w_half) {
            rhs = W_s[t * SS + col];  // RHS from phase A
        } else {
            const float* row = conv_f32 + static_cast<size_t>(t0 + t) * conv_channels;
            rhs = s_beta[t] * row[2 * BC_size + h * HD + col];
        }
        if (tid < t)
            s_row[tid] = s_beta[t] * expf(s_logD[t + 1] - s_logD[tid + 1]) * s_kk[t * kChunk + tid];
        __syncthreads();
        // Four partial accumulators: a single chain serializes one 4-cycle
        // FMA on a shared-memory operand per term (ncu: short_scoreboard
        // 2.45 + barrier 2.12 stalls per issue on the fused-chain form).
        {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int j = 0;
            for (; j + 3 < t; j += 4) {
                a0 += s_row[j] * hist[j * HD + col];
                a1 += s_row[j + 1] * hist[(j + 1) * HD + col];
                a2 += s_row[j + 2] * hist[(j + 2) * HD + col];
                a3 += s_row[j + 3] * hist[(j + 3) * HD + col];
            }
            for (; j < t; j++)
                a0 += s_row[j] * hist[j * HD + col];
            rhs -= (a0 + a1) + (a2 + a3);
        }
        hist[t * HD + col] = rhs;
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

    // ---- phase C: Qeff (in place on QE, W half) and Y_A (U_A half) ----
    // P[t,j] = exp(logD[t+1]-logD[j+1]) * QK[t,j] for j <= t.
    for (int t = 0; t < L; t++) {
        float pre = 0.0f;
        if (w_half)
            pre = QE_s[t * SS + col];  // D[0..t+1] q~ from phase A, prefetched
        if (tid <= t)
            s_row[tid] = expf(s_logD[t + 1] - s_logD[tid + 1]) * s_qk[t * kChunk + tid];
        __syncthreads();
        {
            const float* h2 = w_half ? s_w : s_u;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int j = 0;
            for (; j + 3 <= t; j += 4) {
                a0 += s_row[j] * h2[j * HD + col];
                a1 += s_row[j + 1] * h2[(j + 1) * HD + col];
                a2 += s_row[j + 2] * h2[(j + 2) * HD + col];
                a3 += s_row[j + 3] * h2[(j + 3) * HD + col];
            }
            for (; j <= t; j++)
                a0 += s_row[j] * h2[j * HD + col];
            const float sum = (a0 + a1) + (a2 + a3);
            if (w_half)
                QE_s[t * SS + col] = pre - sum;
            else
                YA_s[t * HD + col] = sum;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 — sequential inter-chunk state pass + outputs.
// Grid (n_heads, kColSplit): blockIdx.y owns HD/kColSplit state columns —
// u_eff, y and the state update are all column-local, so the split only
// re-stages the (read-only) W / Qeff / K_d rows per CTA. Block 2*HD/kColSplit
// threads, adjacent lanes share a column (SPLIT=2, like the fused kernel).
// ---------------------------------------------------------------------------
template <int HD, int SS, typename StateT>
__global__ void __launch_bounds__(2 * HD / kColSplit, kColSplit) gdn_chunkpar_pass_kernel(
    float* __restrict__ ws_base, StateT* __restrict__ h_state, half* __restrict__ y_out,
    int strip_t0, int strip_tokens, int n_chunks, int n_heads, int load_statet, int store_statet) {
    constexpr int COLS = HD / kColSplit;  // d-columns this CTA owns
    const int h = blockIdx.x;
    const int d_base = blockIdx.y * COLS;
    const int dl = threadIdx.x / 2;  // local column
    const int d = d_base + dl;
    const int part = threadIdx.x % 2;
    constexpr int SS_PER = SS / 2;
    const int s_base = part * SS_PER;
    const float scale = rsqrtf(static_cast<float>(HD));
    const int inner = n_heads * HD;

    const ChunkparWs ws = chunkpar_ws_layout<HD, SS>(ws_base, n_heads);
    float* H32_h = ws.H32 + static_cast<size_t>(h) * SS * HD;

    // This CTA's slice of the state, FP32, one half-column per thread.
    float H_reg[SS_PER];
    if (load_statet) {
        const StateT* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            H_reg[s] = static_cast<float>(H_col[(s_base + s) * HD]);
    } else {
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            H_reg[s] = H32_h[(s_base + s) * HD + d];
    }

    extern __shared__ float smem[];
    float* s_a = smem;                // [kChunk * SS] staging (W, then Qeff, then K_d)
    float* s_ue = s_a + kChunk * SS;  // [kChunk * COLS] u_eff, local columns only
    constexpr int NT = 2 * COLS;

    for (int c = 0; c < n_chunks; c++) {
        const int L = min(kChunk, strip_tokens - c * kChunk);
        const size_t slot = static_cast<size_t>(c) * n_heads + h;
        const float* UA_s = ws.UA + slot * kChunk * HD;

        // -- stage W (float4), compute u_eff = U_A - W @ H_0 --
        {
            const float4* src = reinterpret_cast<const float4*>(ws.W + slot * kChunk * SS);
            float4* dst = reinterpret_cast<float4*>(s_a);
            for (int idx = threadIdx.x; idx < L * SS / 4; idx += NT)
                dst[idx] = src[idx];
        }
        __syncthreads();
        for (int t = 0; t < L; t++) {
            float wh = 0.0f;
            const float4* a4 = reinterpret_cast<const float4*>(&s_a[t * SS + s_base]);
#pragma unroll
            for (int ii = 0; ii < SS_PER / 4; ii++) {
                const float4 a = a4[ii];
                wh += a.x * H_reg[4 * ii] + a.y * H_reg[4 * ii + 1] + a.z * H_reg[4 * ii + 2] +
                      a.w * H_reg[4 * ii + 3];
            }
            wh += __shfl_xor_sync(0xffffffffu, wh, 1);
            if (part == 0)
                s_ue[t * COLS + dl] = UA_s[t * HD + d] - wh;
        }
        __syncthreads();

        // -- stage Qeff, emit y from the PRE-update state --
        {
            const float4* src = reinterpret_cast<const float4*>(ws.QE + slot * kChunk * SS);
            float4* dst = reinterpret_cast<float4*>(s_a);
            for (int idx = threadIdx.x; idx < L * SS / 4; idx += NT)
                dst[idx] = src[idx];
        }
        __syncthreads();
        const float* YA_s = ws.YA + slot * kChunk * HD;
        for (int t = 0; t < L; t++) {
            float qeh = 0.0f;
            const float4* a4 = reinterpret_cast<const float4*>(&s_a[t * SS + s_base]);
#pragma unroll
            for (int ii = 0; ii < SS_PER / 4; ii++) {
                const float4 a = a4[ii];
                qeh += a.x * H_reg[4 * ii] + a.y * H_reg[4 * ii + 1] + a.z * H_reg[4 * ii + 2] +
                       a.w * H_reg[4 * ii + 3];
            }
            qeh += __shfl_xor_sync(0xffffffffu, qeh, 1);
            if (part == 0) {
                const size_t t_glob = static_cast<size_t>(strip_t0) + c * kChunk + t;
                y_out[t_glob * inner + h * HD + d] = __float2half((YA_s[t * HD + d] + qeh) * scale);
            }
        }
        __syncthreads();

        // -- stage K_d, advance the state: H = D0L*H + K_d^T u_eff --
        {
            const float4* src = reinterpret_cast<const float4*>(ws.KD + slot * kChunk * SS);
            float4* dst = reinterpret_cast<float4*>(s_a);
            for (int idx = threadIdx.x; idx < L * SS / 4; idx += NT)
                dst[idx] = src[idx];
        }
        __syncthreads();
        const float d0l = ws.D0L[slot];
#pragma unroll
        for (int ib = 0; ib < SS_PER / 4; ib++) {
            float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int t = 0; t < L; t++) {
                const float ue = s_ue[t * COLS + dl];
                const float4 a = *reinterpret_cast<const float4*>(&s_a[t * SS + s_base + 4 * ib]);
                acc.x += a.x * ue;
                acc.y += a.y * ue;
                acc.z += a.z * ue;
                acc.w += a.w * ue;
            }
            H_reg[4 * ib] = d0l * H_reg[4 * ib] + acc.x;
            H_reg[4 * ib + 1] = d0l * H_reg[4 * ib + 1] + acc.y;
            H_reg[4 * ib + 2] = d0l * H_reg[4 * ib + 2] + acc.z;
            H_reg[4 * ib + 3] = d0l * H_reg[4 * ib + 3] + acc.w;
        }
        __syncthreads();
    }

    if (store_statet) {
        StateT* H_col = h_state + static_cast<size_t>(h) * SS * HD + d;
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            H_col[(s_base + s) * HD] = static_cast<StateT>(H_reg[s]);
    } else {
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            H32_h[(s_base + s) * HD + d] = H_reg[s];
    }
}

template <int HD, int SS, typename StateT>
void chunkpar_launch(const float* conv_f32, int conv_channels, const half* alpha, const half* beta,
                     const float* A_log, const float* dt_bias, StateT* h_state, half* y, int n_tokens,
                     int n_heads, int n_groups, cudaStream_t stream, int grouped_layout, float* ws_base) {
    const size_t smem1 =
        (2 * kChunk * SS + 2 * kChunk * kChunk + kChunk + (kChunk + 1) + kChunk) * sizeof(float);
    const size_t smem2 = (kChunk * SS + kChunk * (HD / kColSplit)) * sizeof(float);
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
        gdn_chunkpar_pass_kernel<HD, SS, StateT>
            <<<dim3(n_heads, kColSplit), 2 * HD / kColSplit, smem2, stream>>>(
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
