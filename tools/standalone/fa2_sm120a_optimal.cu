// fa2_sm120a_optimal.cu
// -----------------------------------------------------------------------------
// A self-contained, from-scratch FlashAttention-2 forward kernel for the
// RTX 5090 (GB202, sm_120a — consumer Blackwell). No imp engine headers, no
// CUTLASS, no cuDNN. One file: kernel + CPU reference + correctness check +
// timing. This is the *reference* shape of the optimal sm_120a attention kernel
// described in docs/sm120_optimal_kernel.md — meant to be read and run, not to
// beat imp's production FA2 (which adds f16-acc variants, TWOSLOT, INT8/FP8-QK,
// and per-head GQA plumbing on top of exactly this skeleton).
//
// What it demonstrates (the "optimal kernel" properties):
//   * Register-resident O accumulator — O never touches shared memory across the
//     whole KV loop. This is THE defining FA2 property and the reason raw
//     mma.sync is required (WMMA hides the accumulator->row mapping, so you
//     cannot apply the per-row online-softmax rescale to a WMMA O fragment).
//   * Online softmax (running row max + sum), no global S materialization.
//   * Tensor-core QK^T and P·V via mma.sync.m16n8k16 (HMMA on sm_120 — there is
//     no wgmma / tcgen05 / TMEM on consumer Blackwell).
//   * Bq=128 / Bkv=64 / D=128: 8 warps (256 threads), each warp owns one 16-row
//     query tile. 1 block/SM, latency-hidden by the software pipeline.
//
// Profiling-driven optimization trail (ncu on RTX 5090, BH=8 causal). Each step
// targets the bottleneck the previous ncu run revealed — measured, not guessed.
//   reference                         50.3 / ~80 / ~91   TFLOP/s  (S = 4k / 8k / 16k)
//   A. V staged transposed -> contiguous PV B-fragment load (LSU/MIO relief)
//   B. ldmatrix.x4 for K/P/V fragments — 1 instr per 16x16 tile
//   C. PV f16-accumulate (O as packed half2): lower mma latency, 218 -> 154 regs
//   D. drop the smem Q stage; Q A-fragments straight to registers
//   E. NO P smem round-trip: the QK-output and PV-A column->lane maps coincide,
//      so P is repacked from S_acc registers (no shuffle, no smem, no syncwarp)
//   F. PAD K_s/V_s to stride HD+8: killed a 7.8M-event 8-way bank conflict . +25%
//   G. V via cp.async + ldmatrix.trans (not a regular transposing load): hides
//      the global-load latency that showed up as a long-scoreboard stall ... +25%
//   H. double-buffer K/V (ping-pong cp.async, prefetch tile j+1) .......... +13%
//   final                             113.7 / 157.7 / 186.8 TFLOP/s   (~2.1-2.3x ref)
//   (Past imp's production FA2 ~135 at S>=8k. QK f16-accumulate was tried and
//    REVERTED — unpack overhead > latency win.)
//
// Remaining bottleneck: the cp.async `wait` stall (~1.6 cyc/issue) now dominates
// (DRAM 4%, L1/TEX 29%, SM 14% — still latency-bound, at 16.7% occupancy). A
// deeper 3-stage pipeline would chip at it but is INFEASIBLE here: ldmatrix needs
// 16-byte row alignment (KS a multiple of 8 halves), so the minimum pad is HD+8,
// and 3 slots of K+V then need 102 KB > the 99 KB smem cap. The hard ceiling is
// the absence of an async MMA (tcgen05/TMEM on sm_120): the mma dependency chain +
// per-tile barrier cap concurrency regardless of residency (forcing 2 blocks/SM
// measured -24%). The remaining gap to a B200 FA4 kernel is silicon, not code.
//
// Still simplified vs production: f32 QK^T accumulate, MHA only (no GQA grouping).
//
// Build & run (host has no CUDA toolkit — use the CUDA 13.3 container):
//   docker run --rm --gpus all -v "$PWD":/w -w /w nvidia/cuda:13.3.0-devel-ubuntu24.04 \
//     sh -c 'nvcc -O3 -std=c++17 -arch=sm_120a fa2_sm120a_optimal.cu -o fa2 && ./fa2'
// (imp's own image already has nvcc: imp:test works as the image too.)
// -----------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---- problem / tile config -------------------------------------------------
#define HD 128   // head dim (fixed for this reference)
#define BQ 128   // query rows per CTA  -> 8 warps, each owns a 16-row mma tile
#define BKV 64   // key/value rows per KV tile
#define NWARPS (BQ / 16)  // 8
#define NTHREADS (NWARPS * 32)  // 256
#define KS (HD + 8)  // padded d-stride (kill 8-way conflict); mult of 8 halves for 16B ldmatrix align

#define CUDA_CHECK(x)                                                                  \
    do {                                                                               \
        cudaError_t e_ = (x);                                                          \
        if (e_ != cudaSuccess) {                                                       \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,       \
                   __LINE__);                                                          \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

// ---- raw sm_120 tensor-core primitive -------------------------------------
// D[16x8] += A[16x16] * B[16x8]   (A row-major f16, B col-major f16, C/D f32)
__device__ __forceinline__ void mma_m16n8k16(float& d0, float& d1, float& d2, float& d3,
                                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1, float c0, float c1, float c2,
                                             float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// f16-accumulate variant: D,C are f16 (2 packed-half2 regs each instead of 4
// f32). Lower issue latency than f32-acc AND halves the O accumulator register
// footprint. d0=packed(D[gid][c0],D[gid][c1]), d1=packed(D[gid+8][c0],D[gid+8][c1]).
__device__ __forceinline__ void mma_m16n8k16_f16(uint32_t& d0, uint32_t& d1, uint32_t a0,
                                                 uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                 uint32_t b1, uint32_t c0, uint32_t c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
}

__device__ __forceinline__ __half2 u2h2(uint32_t x) {
    __half2 h;
    memcpy(&h, &x, 4);
    return h;
}
__device__ __forceinline__ uint32_t h22u(__half2 h) {
    uint32_t x;
    memcpy(&x, &h, 4);
    return x;
}
__device__ __forceinline__ uint32_t mul_h2(uint32_t x, __half2 s) { return h22u(__hmul2(u2h2(x), s)); }

__device__ __forceinline__ void cp_async16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Issue the cp.async loads for KV tile `jj` into double-buffer slot `slot` and
// commit them as one group. K and V both row-major; V is transposed at read time
// by ldmatrix.trans in PV.
__device__ __forceinline__ void load_kv(half* K_s, half* V_s, const half* Kb, const half* Vb,
                                        int jj, int slot, int S) {
    const int base = jj * BKV;
    half* Kd = K_s + slot * BKV * KS;
    half* Vd = V_s + slot * BKV * KS;
    for (int i = threadIdx.x; i < BKV * HD / 8; i += NTHREADS) {
        int row = (i * 8) / HD, col = (i * 8) % HD;
        int gr = base + row;
        if (gr < S) {
            cp_async16(&Kd[row * KS + col], &Kb[(int64_t)gr * HD + col]);
            cp_async16(&Vd[row * KS + col], &Vb[(int64_t)gr * HD + col]);
        } else {
            *reinterpret_cast<float4*>(&Kd[row * KS + col]) = make_float4(0, 0, 0, 0);
            *reinterpret_cast<float4*>(&Vd[row * KS + col]) = make_float4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
}

// ldmatrix: load a 16x16 b16 tile (= 4 mma fragment regs) in ONE instruction.
// x4       : A operand, source row-major (no transpose).
// x4.trans : B operand, source row-major -> hardware-transposed into the fragment.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const half* p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3, const half* p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

__device__ __forceinline__ uint32_t pack2(half lo, half hi) {
    uint32_t r;
    half2 h = __halves2half2(lo, hi);
    memcpy(&r, &h, 4);
    return r;
}

// ---- the kernel ------------------------------------------------------------
// Q,K,V,O are [n_bh][S][HD] row-major, fp16. n_bh = batch*heads (flattened).
// grid = (ceil(S/BQ), n_bh), block = NTHREADS.
__global__ void __launch_bounds__(NTHREADS, 1)
    fa2_sm120a(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
               half* __restrict__ O, int S, float scale, bool causal) {
    constexpr int KC_QK = HD / 16;   // 8  k-chunks of 16 for QK^T
    constexpr int N_S = BKV / 8;     // 8  n8-tiles spanning the BKV score columns
    constexpr int KC_PV = BKV / 16;  // 4  k-chunks of 16 for P·V
    constexpr int N_O = HD / 8;      // 16 n8-tiles spanning the HD output columns

    const int q_tile = blockIdx.x;
    const int bh = blockIdx.y;
    const int warp = threadIdx.x / 32;  // 0..7  -> this warp's 16-row query tile
    const int lane = threadIdx.x % 32;
    const int gid = lane >> 2;     // 0..7  : row within the 16-tile (lo half)
    const int tig = lane & 3;      // 0..3  : column pair within an 8/16 group
    const int q_base = q_tile * BQ + warp * 16;  // first absolute query row of this warp

    const half* Qb = Q + (int64_t)bh * S * HD;
    const half* Kb = K + (int64_t)bh * S * HD;
    const half* Vb = V + (int64_t)bh * S * HD;
    half* Ob = O + (int64_t)bh * S * HD;

    // ---- shared memory ----
    // Q_s : staged once       (BQ*HD)     = 128*128 halves = 32 KB
    // K_s : one KV tile        (BKV*HD)   = 64*128  halves = 16 KB
    // V_t : one KV tile, TRANSPOSED to [d][kv] with padded stride VTS, so the
    //       PV B-fragment (col-major, k-contiguous) is a single uint32 load
    //       instead of two non-contiguous half reads (the LSU/MIO bottleneck).
    // P_s : per-warp P repack  (BQ*BKV)   = 128*64  halves = 16 KB
    extern __shared__ half smem[];
    half* K_s = smem;                // [2][BKV*KS] double-buffer (ping-pong)
    half* V_s = K_s + 2 * BKV * KS;  // [2][BKV*KS]; P needs no smem (built from S_acc registers)

    // ---- load this warp's Q A-fragments straight from global into registers
    //      ONCE (kept resident across the whole KV loop). No smem Q stage: Q is
    //      read exactly once per CTA, and dropping the 32 KB stage is what lets
    //      the kernel fit 2 blocks/SM. DRAM has ample headroom (~2% busy). ----
    const int qr0 = q_base + gid, qr1 = q_base + gid + 8;
    uint32_t Qf[KC_QK][4];
#pragma unroll
    for (int kc = 0; kc < KC_QK; kc++) {
        Qf[kc][0] = (qr0 < S) ? *reinterpret_cast<const uint32_t*>(&Qb[(int64_t)qr0 * HD + kc * 16 + tig * 2]) : 0u;
        Qf[kc][1] = (qr1 < S) ? *reinterpret_cast<const uint32_t*>(&Qb[(int64_t)qr1 * HD + kc * 16 + tig * 2]) : 0u;
        Qf[kc][2] = (qr0 < S) ? *reinterpret_cast<const uint32_t*>(&Qb[(int64_t)qr0 * HD + kc * 16 + 8 + tig * 2]) : 0u;
        Qf[kc][3] = (qr1 < S) ? *reinterpret_cast<const uint32_t*>(&Qb[(int64_t)qr1 * HD + kc * 16 + 8 + tig * 2]) : 0u;
    }

    // ---- register-resident O accumulator (PACKED f16: half the registers) +
    //      online-softmax state. O_h2[n2][0]=row gid, O_h2[n2][1]=row gid+8. ----
    uint32_t O_h2[N_O][2];
#pragma unroll
    for (int n = 0; n < N_O; n++) O_h2[n][0] = O_h2[n][1] = 0u;
    // two rows owned by this lane: rlo = warp*16+gid, rhi = warp*16+gid+8
    float m_lo = -INFINITY, m_hi = -INFINITY;  // running row max
    float l_lo = 0.f, l_hi = 0.f;              // running row sum

    // n_kv MUST be uniform across the whole block: the loop body has block-wide
    // __syncthreads() + cooperative K/V loads, so every warp must run the same
    // trip count. Size it to the block's *last* query row (q_tile*BQ + BQ - 1);
    // warps/rows that shouldn't see a tile get fully -INF-masked below.
    const int q_block_max = min(q_tile * BQ + BQ, S);  // exclusive
    const int n_kv = causal ? ((q_block_max + BKV - 1) / BKV) : ((S + BKV - 1) / BKV);

    // Double-buffered KV pipeline: prefetch tile j+1 while computing tile j, so
    // the cp.async global latency (the long-scoreboard + wait stalls) overlaps
    // compute instead of stalling. Prologue issues tile 0.
    load_kv(K_s, V_s, Kb, Vb, 0, 0, S);
    for (int j = 0; j < n_kv; j++) {
        const int kv0 = j * BKV;
        const int slot = j & 1;
        if (j + 1 < n_kv) {
            load_kv(K_s, V_s, Kb, Vb, j + 1, (j + 1) & 1, S);  // prefetch next
            cp_async_wait<1>();  // 2 groups in flight -> drain the older (tile j)
        } else {
            cp_async_wait<0>();  // last tile: drain it
        }
        __syncthreads();
        const half* K_sl = K_s + slot * BKV * KS;  // this tile's slot
        const half* V_sl = V_s + slot * BKV * KS;

        // ---- Phase 1: S = Q @ K^T via ldmatrix.x4 -> 8 n8-tile accumulators.
        //      K_s is [n=kv][k=d], same layout as V_t, so ldmatrix.x4 (NO trans)
        //      yields the col-major B fragment directly. One x4 = 2 n-tiles;
        //      quadrants -> n-tile0=(k0,k2), n-tile1=(k1,k3) (f32-acc QK^T). ----
        const int lr = lane % 16, lc = (lane / 16) * 8;  // ldmatrix per-lane row/col-base
        float S_acc[N_S][4];
#pragma unroll
        for (int nb = 0; nb < N_S / 2; nb++) {
            float sa0 = 0, sa1 = 0, sa2 = 0, sa3 = 0, sb0 = 0, sb1 = 0, sb2 = 0, sb3 = 0;
#pragma unroll
            for (int kc = 0; kc < KC_QK; kc++) {
                uint32_t k0, k1, k2, k3;
                ldmatrix_x4(k0, k1, k2, k3, &K_sl[(nb * 16 + lr) * KS + kc * 16 + lc]);
                mma_m16n8k16(sa0, sa1, sa2, sa3, Qf[kc][0], Qf[kc][1], Qf[kc][2], Qf[kc][3], k0, k2,
                             sa0, sa1, sa2, sa3);
                mma_m16n8k16(sb0, sb1, sb2, sb3, Qf[kc][0], Qf[kc][1], Qf[kc][2], Qf[kc][3], k1, k3,
                             sb0, sb1, sb2, sb3);
            }
            S_acc[2 * nb][0] = sa0; S_acc[2 * nb][1] = sa1;
            S_acc[2 * nb][2] = sa2; S_acc[2 * nb][3] = sa3;
            S_acc[2 * nb + 1][0] = sb0; S_acc[2 * nb + 1][1] = sb1;
            S_acc[2 * nb + 1][2] = sb2; S_acc[2 * nb + 1][3] = sb3;
        }

        // ---- scale + causal mask ----
        const int rlo = q_base + gid, rhi = q_base + gid + 8;
#pragma unroll
        for (int nt = 0; nt < N_S; nt++) {
            int c0 = kv0 + nt * 8 + tig * 2, c1 = c0 + 1;
            S_acc[nt][0] = (rlo < S && c0 <= (causal ? rlo : S - 1)) ? S_acc[nt][0] * scale : -INFINITY;
            S_acc[nt][1] = (rlo < S && c1 <= (causal ? rlo : S - 1)) ? S_acc[nt][1] * scale : -INFINITY;
            S_acc[nt][2] = (rhi < S && c0 <= (causal ? rhi : S - 1)) ? S_acc[nt][2] * scale : -INFINITY;
            S_acc[nt][3] = (rhi < S && c1 <= (causal ? rhi : S - 1)) ? S_acc[nt][3] * scale : -INFINITY;
        }

        // ---- Phase 2: online softmax (row max/sum across the 4-lane group) ----
        float rmax_lo = -INFINITY, rmax_hi = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < N_S; nt++) {
            rmax_lo = fmaxf(rmax_lo, fmaxf(S_acc[nt][0], S_acc[nt][1]));
            rmax_hi = fmaxf(rmax_hi, fmaxf(S_acc[nt][2], S_acc[nt][3]));
        }
        // reduce across tig=0..3 (lanes gid*4 .. gid*4+3 hold the other columns)
        for (int o = 1; o <= 2; o <<= 1) {
            rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffff, rmax_lo, o));
            rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffff, rmax_hi, o));
        }
        float m_lo_new = fmaxf(m_lo, rmax_lo), m_hi_new = fmaxf(m_hi, rmax_hi);
        float a_lo = isinf(m_lo) ? 0.f : __expf(m_lo - m_lo_new);  // O rescale factor
        float a_hi = isinf(m_hi) ? 0.f : __expf(m_hi - m_hi_new);

        float rsum_lo = 0.f, rsum_hi = 0.f;
#pragma unroll
        for (int nt = 0; nt < N_S; nt++) {
            float p0 = isinf(m_lo_new) ? 0.f : __expf(S_acc[nt][0] - m_lo_new);
            float p1 = isinf(m_lo_new) ? 0.f : __expf(S_acc[nt][1] - m_lo_new);
            float p2 = isinf(m_hi_new) ? 0.f : __expf(S_acc[nt][2] - m_hi_new);
            float p3 = isinf(m_hi_new) ? 0.f : __expf(S_acc[nt][3] - m_hi_new);
            S_acc[nt][0] = p0; S_acc[nt][1] = p1; S_acc[nt][2] = p2; S_acc[nt][3] = p3;
            rsum_lo += p0 + p1;
            rsum_hi += p2 + p3;
        }
        for (int o = 1; o <= 2; o <<= 1) {
            rsum_lo += __shfl_xor_sync(0xffffffff, rsum_lo, o);
            rsum_hi += __shfl_xor_sync(0xffffffff, rsum_hi, o);
        }
        m_lo = m_lo_new; m_hi = m_hi_new;
        l_lo = l_lo * a_lo + rsum_lo;
        l_hi = l_hi * a_hi + rsum_hi;

        // rescale the register-resident O by the per-row factor (rows rlo/rhi)
        __half2 alo2 = __float2half2_rn(a_lo), ahi2 = __float2half2_rn(a_hi);
#pragma unroll
        for (int n = 0; n < N_O; n++) {
            O_h2[n][0] = mul_h2(O_h2[n][0], alo2);
            O_h2[n][1] = mul_h2(O_h2[n][1], ahi2);
        }

        // ---- Phase 3: O += P @ V  (into the register-resident accumulator) ----
        // NO P smem round-trip: the QK output column->lane mapping and the PV
        // A-fragment column->lane mapping COINCIDE, so each lane already holds in
        // S_acc exactly the P values its A-fragment needs. Build a0..a3 by
        // repacking f32->f16 in registers -> kills 32 smem writes + a syncwarp +
        // 16 ldmatrix reads + the write->read dependency (the short-scoreboard
        // stall) and frees the 16 KB P buffer. (P[gid][kc*16+tig*2] = S_acc[2kc][0].)
#pragma unroll
        for (int kc = 0; kc < KC_PV; kc++) {
            uint32_t a0 = pack2(__float2half(S_acc[2 * kc][0]), __float2half(S_acc[2 * kc][1]));
            uint32_t a1 = pack2(__float2half(S_acc[2 * kc][2]), __float2half(S_acc[2 * kc][3]));
            uint32_t a2 = pack2(__float2half(S_acc[2 * kc + 1][0]), __float2half(S_acc[2 * kc + 1][1]));
            uint32_t a3 = pack2(__float2half(S_acc[2 * kc + 1][2]), __float2half(S_acc[2 * kc + 1][3]));
            // B = V via ldmatrix.x4.trans on row-major V_s[kv=k][d=n]: the trans
            // turns the row-major [k][n] storage into the col-major B fragment.
            // Addressing: lr splits the k-half (matrix 0/1), lc splits the n-tile
            // (matrix 0,1 vs 2,3) -> n-tile0=(v0,v1), n-tile1=(v2,v3).
#pragma unroll
            for (int nb = 0; nb < N_O / 2; nb++) {
                uint32_t v0, v1, v2, v3;
                ldmatrix_x4_trans(v0, v1, v2, v3, &V_sl[(kc * 16 + lr) * KS + nb * 16 + lc]);
                int n2a = nb * 2, n2b = nb * 2 + 1;
                mma_m16n8k16_f16(O_h2[n2a][0], O_h2[n2a][1], a0, a1, a2, a3, v0, v1, O_h2[n2a][0],
                                 O_h2[n2a][1]);
                mma_m16n8k16_f16(O_h2[n2b][0], O_h2[n2b][1], a0, a1, a2, a3, v2, v3, O_h2[n2b][0],
                                 O_h2[n2b][1]);
            }
        }
        __syncthreads();  // reads of this slot done before j+1 prefetches into it (j+2)
    }

    // ---- normalize by row sum and write O ----
    float inv_lo = l_lo > 0.f ? 1.f / l_lo : 0.f;
    float inv_hi = l_hi > 0.f ? 1.f / l_hi : 0.f;
    const int rlo = q_base + gid, rhi = q_base + gid + 8;
#pragma unroll
    for (int n2 = 0; n2 < N_O; n2++) {
        int c0 = n2 * 8 + tig * 2, c1 = c0 + 1;
        __half2 lo = u2h2(O_h2[n2][0]);  // row rlo: (col c0, col c1)
        __half2 hi = u2h2(O_h2[n2][1]);  // row rhi
        if (rlo < S) {
            Ob[(int64_t)rlo * HD + c0] = __float2half(__low2float(lo) * inv_lo);
            Ob[(int64_t)rlo * HD + c1] = __float2half(__high2float(lo) * inv_lo);
        }
        if (rhi < S) {
            Ob[(int64_t)rhi * HD + c0] = __float2half(__low2float(hi) * inv_hi);
            Ob[(int64_t)rhi * HD + c1] = __float2half(__high2float(hi) * inv_hi);
        }
    }
}

// ---- CPU reference (double precision) --------------------------------------
static void cpu_attention(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O, int BH, int S,
                          float scale, bool causal) {
    for (int b = 0; b < BH; b++)
        for (int i = 0; i < S; i++) {
            int lim = causal ? i : S - 1;
            std::vector<double> s(lim + 1);
            double mx = -1e30;
            for (int j = 0; j <= lim; j++) {
                double dot = 0;
                for (int d = 0; d < HD; d++)
                    dot += (double)Q[((int64_t)b * S + i) * HD + d] * K[((int64_t)b * S + j) * HD + d];
                s[j] = dot * scale;
                mx = std::max(mx, s[j]);
            }
            double sum = 0;
            for (int j = 0; j <= lim; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            for (int d = 0; d < HD; d++) {
                double acc = 0;
                for (int j = 0; j <= lim; j++) acc += s[j] * V[((int64_t)b * S + j) * HD + d];
                O[((int64_t)b * S + i) * HD + d] = (float)(acc / sum);
            }
        }
}

int main(int argc, char** argv) {
    int S = (argc > 1) ? atoi(argv[1]) : 512;
    int BH = (argc > 2) ? atoi(argv[2]) : 16;  // batch*heads
    bool causal = true;
    float scale = 1.0f / std::sqrt((float)HD);

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  (sm_%d%d, %d SMs, smemOptin=%zu KB)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.sharedMemPerBlockOptin / 1024);
    printf("Problem: BH=%d S=%d HD=%d causal=%d  (BQ=%d BKV=%d, %d warps)\n", BH, S, HD, causal, BQ,
           BKV, NWARPS);

    size_t n = (size_t)BH * S * HD;
    std::vector<float> hQ(n), hK(n), hV(n), hO_ref((size_t)BH * S * HD), hO_gpu((size_t)BH * S * HD);
    srand(1234);
    for (size_t i = 0; i < n; i++) {
        hQ[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
        hK[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
        hV[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.f;
    }

    std::vector<half> hQh(n), hKh(n), hVh(n);
    for (size_t i = 0; i < n; i++) { hQh[i] = __float2half(hQ[i]); hKh[i] = __float2half(hK[i]); hVh[i] = __float2half(hV[i]); }

    half *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, n * 2));
    CUDA_CHECK(cudaMalloc(&dK, n * 2));
    CUDA_CHECK(cudaMalloc(&dV, n * 2));
    CUDA_CHECK(cudaMalloc(&dO, n * 2));
    CUDA_CHECK(cudaMemcpy(dQ, hQh.data(), n * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hKh.data(), n * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hVh.data(), n * 2, cudaMemcpyHostToDevice));

    size_t smem = (4 * BKV * (HD + 8)) * sizeof(half);  // 2-slot double-buffer: K_s[2] + V_s[2]
    CUDA_CHECK(cudaFuncSetAttribute(fa2_sm120a, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    printf("smem/block = %zu KB\n", smem / 1024);

    dim3 grid((S + BQ - 1) / BQ, BH);
    dim3 block(NTHREADS);

    // correctness
    fa2_sm120a<<<grid, block, smem>>>(dQ, dK, dV, dO, S, scale, causal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> hOh(n);
    CUDA_CHECK(cudaMemcpy(hOh.data(), dO, n * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++) hO_gpu[i] = __half2float(hOh[i]);

    cpu_attention(hQ, hK, hV, hO_ref, BH, S, scale, causal);
    double max_abs = 0, sum_abs = 0;
    for (size_t i = 0; i < n; i++) {
        double e = std::fabs((double)hO_gpu[i] - hO_ref[i]);
        max_abs = std::max(max_abs, e);
        sum_abs += e;
    }
    printf("Correctness: max_abs_err=%.4e  mean_abs_err=%.4e  %s\n", max_abs, sum_abs / n,
           max_abs < 5e-3 ? "PASS" : "FAIL");

    // timing — warm the clocks for >1.5s of wall time first (sm_120 idles at
    // ~360-810 MHz and takes ~1s to ramp to ~2850 MHz under load; a short
    // warmup reads artificially LOW — see docs/sm120_optimal_kernel.md notes).
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    {
        cudaEvent_t w0, w1; cudaEventCreate(&w0); cudaEventCreate(&w1);
        float wms = 0; int wi = 0;
        cudaEventRecord(w0);
        while (wms < 1500.f) {  // busy until >1.5s elapsed
            for (int i = 0; i < 50; i++) fa2_sm120a<<<grid, block, smem>>>(dQ, dK, dV, dO, S, scale, causal);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(w1); cudaEventSynchronize(w1);
            cudaEventElapsedTime(&wms, w0, w1); wi += 50;
        }
        printf("(warmup: %d iters / %.0f ms to ramp clocks)\n", wi, wms);
    }
    int reps = 200;
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) fa2_sm120a<<<grid, block, smem>>>(dQ, dK, dV, dO, S, scale, causal);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    double per = ms / reps;
    // causal FLOPs: 2 * 2 * BH * (S*(S+1)/2) * HD  (QK + PV, MACs*2)
    double flops = 2.0 * 2.0 * BH * ((double)S * (S + 1) / 2.0) * HD;
    printf("Time: %.3f ms/iter   %.1f TFLOP/s (fp16, %% of 838 datasheet = %.1f%%)\n", per,
           flops / (per * 1e-3) / 1e12, flops / (per * 1e-3) / 1e12 / 838.0 * 100.0);
    printf("NOTE: trust this number only with warm clocks (~2850 MHz SM / 13801 MHz mem / ~500 W);\n"
           "      a contended GPU (other CUDA processes) depresses it. Sample nvidia-smi during the run.\n");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
