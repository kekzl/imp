// Dense Q6_K small-M IMMA kernel (sm_120a): BM=32, one 256-K super-block per step, cp.async staging.
// Launched from launch_q6k (mmq_q8_imma.cu) for dense M <= 32; MoE and M >= 64 keep
// mmq_imma_q6k_raw_kernel (mmq_q8_imma_q6k.cu). Same per-kb math and order as that kernel at BM=32.

#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kQ6Bytes = 210;
constexpr int kQ6Chunks = 14;  // 16-B chunks covering a 210-B block at any even 16-B offset
constexpr int kQ6BRowW = kQ6kK256BRowW;
constexpr int kQ6ARow = kQ6kK256ARow;
constexpr int kQ6Subs = 8;  // 32-wide activation sub-blocks per step
static_assert(kQ6BRowW * 4 >= (kQ6Chunks * 16) + 4, "q6k_rd of byte 208 at offset 14 reads word 56");

// cp.async of src_bytes (0..16) into a 16-B smem chunk; the rest is zero-filled.
__device__ __forceinline__ void cp_async_cg_16_n(void* smem, const void* glob, int src_bytes) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(glob), "r"(src_bytes));
}

// Staged row of block idx, advanced to the word holding block byte 0; sh = bit shift of that byte.
// The window starts (W + idx * 210) & 15 bytes before the block; W is 16-aligned (launch_q6k).
__device__ __forceinline__ const uint32_t* q6k_win(const uint32_t* row, uintptr_t w_base, size_t idx,
                                                   uint32_t& sh) {
    const auto mis = static_cast<uint32_t>((w_base + (idx * kQ6Bytes)) & 15u);
    sh = (mis & 3u) * 8u;
    return row + (mis >> 2);
}

// Block bytes [off, off + 4) of a staged window row; off is a multiple of 4.
__device__ __forceinline__ uint32_t q6k_rd(const uint32_t* row, int off, uint32_t sh) {
    return __funnelshift_r(row[off >> 2], row[(off >> 2) + 1], sh);
}

// One step: A tile [32][256] s8, A scales [32][8], B 16-B windows [128][14 chunks] of super-block sb.
// Rows >= M and >= n_valid are zero-filled; the last chunk stops at the block end (no read past it).
__device__ __forceinline__ void q6k_k256_load(int tid, const int8_t* __restrict__ A,
                                              const __half* __restrict__ Asc, const uint8_t* __restrict__ W,
                                              uint8_t* sA, __half* sAsc, uint32_t* sB, int M, int K,
                                              int base_n, int n_valid, int sblk_count, int sb) {
    const int subs = K / 32;
#pragma unroll
    for (int i = tid; i < 32 * 16; i += kThreads) {
        const int row = i >> 4;
        const int col = (i & 15) * 16;
        cp_async_cg_16(sA + static_cast<ptrdiff_t>((row * kQ6ARow) + col),
                       A + (static_cast<size_t>(row) * K) + (static_cast<size_t>(sb) * 256) + col, row < M);
    }
    if (tid < 32 * 4) {
        const int row = tid >> 2;
        const int c = (tid & 3) * 2;
        cp_async_ca_4(sAsc + static_cast<ptrdiff_t>((row * kQ6Subs) + c),
                      Asc + (static_cast<size_t>(row) * subs) + (static_cast<size_t>(sb) * kQ6Subs) + c,
                      row < M);
    }
#pragma unroll
    for (int i = tid; i < kBN * kQ6Chunks; i += kThreads) {
        const int row = i / kQ6Chunks;
        const int c = i - (row * kQ6Chunks);
        const uint8_t* blk = W + ((((static_cast<size_t>(base_n + row)) * sblk_count) + sb) * kQ6Bytes);
        const auto mis = static_cast<int>(reinterpret_cast<uintptr_t>(blk) & 15u);
        const uint8_t* src = blk - mis + static_cast<ptrdiff_t>(c * 16);
        const int bytes = row < n_valid ? max(0, min(16, mis + kQ6Bytes - (c * 16))) : 0;
        cp_async_cg_16_n(sB + static_cast<ptrdiff_t>((row * kQ6BRowW) + (c * 4)), bytes > 0 ? src : W, bytes);
    }
}

// acc[nf] over one 128-K group g of the staged super-block (4 sub-blocks of 32, per-16 scales).
template <int kNF>
__device__ __forceinline__ void q6k_k256_group(float (&acc)[kNF][4], const uint8_t* sA, const __half* sAsc,
                                               const uint32_t* sB, int g, int arow_lo, int bcol0, int ncol0,
                                               int rl, int cl, uintptr_t w_base, size_t idx0,
                                               int sblk_count) {
    const int arow_hi = arow_lo + 8;
    const uint8_t* arl = sA + static_cast<ptrdiff_t>(arow_lo * kQ6ARow);
    const uint8_t* arh = sA + static_cast<ptrdiff_t>(arow_hi * kQ6ARow);
    uint32_t a[4][4];
    float da_lo[4], da_hi[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int kb = (4 * g) + q;
        const int acol = (kb * 32) + (cl * 4);
        a[q][0] = *reinterpret_cast<const uint32_t*>(arl + acol);
        a[q][1] = *reinterpret_cast<const uint32_t*>(arh + acol);
        a[q][2] = *reinterpret_cast<const uint32_t*>(arl + acol + 16);
        a[q][3] = *reinterpret_cast<const uint32_t*>(arh + acol + 16);
        da_lo[q] = __half2float(sAsc[(arow_lo * kQ6Subs) + kb]);
        da_hi[q] = __half2float(sAsc[(arow_hi * kQ6Subs) + kb]);
    }
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) {
        const int bcol = bcol0 + (nf * 8) + rl;
        uint32_t bsh = 0;
        const uint32_t* brow = q6k_win(sB + static_cast<ptrdiff_t>(bcol * kQ6BRowW), w_base,
                                       idx0 + (static_cast<size_t>(bcol) * sblk_count), bsh);
        const int qo = (g * 64) + (cl * 4);
        const uint32_t qlw[4] = {q6k_rd(brow, qo, bsh), q6k_rd(brow, qo + 16, bsh),
                                 q6k_rd(brow, qo + 32, bsh), q6k_rd(brow, qo + 48, bsh)};
        const int ho = 128 + (g * 32) + (cl * 4);
        const uint32_t qh0 = q6k_rd(brow, ho, bsh);
        const uint32_t qh1 = q6k_rd(brow, ho + 16, bsh);

        const int ncol = ncol0 + (nf * 8);
        uint32_t sh_lo = 0;
        uint32_t sh_hi = 0;
        const uint32_t* srow_lo = q6k_win(sB + static_cast<ptrdiff_t>(ncol * kQ6BRowW), w_base,
                                          idx0 + (static_cast<size_t>(ncol) * sblk_count), sh_lo);
        const uint32_t* srow_hi = q6k_win(sB + static_cast<ptrdiff_t>((ncol + 1) * kQ6BRowW), w_base,
                                          idx0 + (static_cast<size_t>(ncol + 1) * sblk_count), sh_hi);
        const uint32_t sc_lo[2] = {q6k_rd(srow_lo, 192 + (g * 8), sh_lo),
                                   q6k_rd(srow_lo, 196 + (g * 8), sh_lo)};
        const uint32_t sc_hi[2] = {q6k_rd(srow_hi, 192 + (g * 8), sh_hi),
                                   q6k_rd(srow_hi, 196 + (g * 8), sh_hi)};
        const float dlo = __half2float(
            __ushort_as_half(static_cast<unsigned short>(q6k_rd(srow_lo, 208, sh_lo))));
        const float dhi = __half2float(
            __ushort_as_half(static_cast<unsigned short>(q6k_rd(srow_hi, 208, sh_hi))));
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint32_t nib_shift = (q >= 2) ? 4u : 0u;
            const uint32_t qh_shift = static_cast<uint32_t>(q * 2);
            const uint32_t l0 = (q & 1) != 0 ? qlw[2] : qlw[0];
            const uint32_t l1 = (q & 1) != 0 ? qlw[3] : qlw[1];
            const uint32_t b0 = __vsub4(((l0 >> nib_shift) & 0x0F0F0F0Fu) |
                                            (((qh0 >> qh_shift) & 0x03030303u) << 4),
                                        0x20202020u);
            const uint32_t b1 = __vsub4(((l1 >> nib_shift) & 0x0F0F0F0Fu) |
                                            (((qh1 >> qh_shift) & 0x03030303u) << 4),
                                        0x20202020u);
            int32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            int32_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
                : "r"(a[q][0]), "r"(a[q][1]), "r"(a[q][2]), "r"(a[q][3]), "r"(b0), "r"(0), "r"(0), "r"(0),
                  "r"(0), "r"(0));
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=r"(q0), "=r"(q1), "=r"(q2), "=r"(q3)
                : "r"(a[q][0]), "r"(a[q][1]), "r"(a[q][2]), "r"(a[q][3]), "r"(0), "r"(b1), "r"(0), "r"(0),
                  "r"(0), "r"(0));
            // per-16 scales 2q, 2q+1 of this group: bytes 2(q&1), 2(q&1)+1 of word q>>1
            const int sb0 = (q & 1) * 16;
            const float al1 = dlo * static_cast<float>(static_cast<int8_t>(sc_lo[q >> 1] >> sb0));
            const float al2 = dlo * static_cast<float>(static_cast<int8_t>(sc_lo[q >> 1] >> (sb0 + 8)));
            const float ah1 = dhi * static_cast<float>(static_cast<int8_t>(sc_hi[q >> 1] >> sb0));
            const float ah2 = dhi * static_cast<float>(static_cast<int8_t>(sc_hi[q >> 1] >> (sb0 + 8)));
            acc[nf][0] += da_lo[q] * fmaf(al1, static_cast<float>(p0), al2 * static_cast<float>(q0));
            acc[nf][1] += da_lo[q] * fmaf(ah1, static_cast<float>(p1), ah2 * static_cast<float>(q1));
            acc[nf][2] += da_hi[q] * fmaf(al1, static_cast<float>(p2), al2 * static_cast<float>(q2));
            acc[nf][3] += da_hi[q] * fmaf(ah1, static_cast<float>(p3), ah2 * static_cast<float>(q3));
        }
    }
}

// Rows row_lo and row_lo + 8 of the warp tile: fp32 partials (SPLITK) or fp16 (+= when BETA1).
template <bool BETA1, bool SPLITK, int kNF>
__device__ __forceinline__ void q6k_k256_store(const float (&acc)[kNF][4], __half* __restrict__ C,
                                               float* __restrict__ Cs, int row_lo, int col0, int M, int N) {
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) {
        const int col = col0 + (nf * 8);
        if (col + 1 >= N)
            continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row_lo + (h * 8);
            if (row >= M)
                continue;
            const size_t o = (static_cast<size_t>(row) * N) + col;
            if constexpr (SPLITK) {
                *reinterpret_cast<float2*>(&Cs[o]) = make_float2(acc[nf][2 * h], acc[nf][(2 * h) + 1]);
            } else {
                __half2* p = reinterpret_cast<__half2*>(&C[o]);
                const __half2 v = __floats2half2_rn(acc[nf][2 * h], acc[nf][(2 * h) + 1]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
        }
    }
}

}  // namespace

template <bool BETA1, bool SPLITK>
__global__ void __launch_bounds__(kThreads) mmq_imma_q6k_k256_kernel(
    const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale, const uint8_t* __restrict__ Wq6k,
    __half* __restrict__ out, int M, int N, int K, float* __restrict__ split_out, int sb_per_split) {
    constexpr int kWN = 4;
    constexpr int kTileN = kBN / kWN;
    constexpr int kNF = kTileN / 8;

    const int base_n = blockIdx.x * kBN;
    const int n_valid = min(kBN, N - base_n);
    const int sblk_count = K / 256;
    const int sb_begin = SPLITK ? static_cast<int>(blockIdx.z) * sb_per_split : 0;
    const int sb_end = SPLITK ? min(sblk_count, sb_begin + sb_per_split) : sblk_count;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;
    const int cl = lane & 3;
    const int arow_lo = (warp_m * 16) + rl;
    const int bcol0 = warp_n * kTileN;
    const int ncol0 = bcol0 + (cl * 2);

    extern __shared__ __align__(16) uint8_t smem_k256[];
    auto sA = [&](int st) { return smem_k256 + (static_cast<size_t>(st) * q6k_k256_stage_bytes()); };
    auto sAsc = [&](int st) { return reinterpret_cast<__half*>(sA(st) + (32 * kQ6ARow)); };
    auto sB = [&](int st) {
        return reinterpret_cast<uint32_t*>(sA(st) + (32 * kQ6ARow) + (sizeof(__half) * 32 * kQ6Subs));
    };

    float acc[kNF][4];
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf)
        acc[nf][0] = acc[nf][1] = acc[nf][2] = acc[nf][3] = 0.0f;

    const uintptr_t w_base = reinterpret_cast<uintptr_t>(Wq6k);
    const bool rows_live = warp_m * 16 < M;  // M <= 16: the warp_m = 1 half only loads and syncs
    // Launch guarantees sb_begin < sb_end.
    q6k_k256_load(tid, X_s8, x_scale, Wq6k, sA(0), sAsc(0), sB(0), M, K, base_n, n_valid, sblk_count,
                  sb_begin);
    cp_async_commit();
    for (int sb = sb_begin; sb < sb_end; ++sb) {
        const int stage = (sb - sb_begin) & 1;
        if (sb + 1 < sb_end) {
            const int ns = stage ^ 1;
            q6k_k256_load(tid, X_s8, x_scale, Wq6k, sA(ns), sAsc(ns), sB(ns), M, K, base_n, n_valid,
                          sblk_count, sb + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        const size_t idx0 = (static_cast<size_t>(base_n) * sblk_count) + sb;
        if (rows_live) {
#pragma unroll
            for (int g = 0; g < 2; ++g)
                q6k_k256_group<kNF>(acc, sA(stage), sAsc(stage), sB(stage), g, arow_lo, bcol0, ncol0, rl, cl,
                                    w_base, idx0, sblk_count);
        }
        __syncthreads();
    }

    q6k_k256_store<BETA1, SPLITK, kNF>(acc, out,
                                       split_out + (SPLITK ? static_cast<size_t>(blockIdx.z) * M * N : 0),
                                       arow_lo, base_n + ncol0, M, N);
}

template __global__ void mmq_imma_q6k_k256_kernel<false, false>(const int8_t*, const __half*, const uint8_t*,
                                                                __half*, int, int, int, float*, int);
template __global__ void mmq_imma_q6k_k256_kernel<true, false>(const int8_t*, const __half*, const uint8_t*,
                                                               __half*, int, int, int, float*, int);
template __global__ void mmq_imma_q6k_k256_kernel<false, true>(const int8_t*, const __half*, const uint8_t*,
                                                               __half*, int, int, int, float*, int);

// Dense M <= 32 on the k256 kernel: K-split when it pays, else one pass. False = smem opt-in failed.
bool launch_q6k_k256(const uint8_t* w6, __half* out_f16, int M, int N, int K, float beta, bool capturing,
                     cudaStream_t stream) {
    using K256 = void (*)(const int8_t*, const __half*, const uint8_t*, __half*, int, int, int, float*, int);
    const K256 k0 = mmq_imma_q6k_k256_kernel<false, false>;
    const K256 k1 = mmq_imma_q6k_k256_kernel<true, false>;
    const K256 ks = mmq_imma_q6k_k256_kernel<false, true>;
    static bool smem_ok = false;
    if (!smem_ok)
        smem_ok = imma_smem_opt_in(reinterpret_cast<const void*>(k0), q6k_k256_smem_bytes()) &&
                  imma_smem_opt_in(reinterpret_cast<const void*>(k1), q6k_k256_smem_bytes()) &&
                  imma_smem_opt_in(reinterpret_cast<const void*>(ks), q6k_k256_smem_bytes());
    if (!smem_ok)
        return false;
    const bool beta1 = beta == 1.0f;
    const int n_tiles = (N + kBN - 1) / kBN;
    const int sblks = K / 256;
    const int s = imma_dense_splits(N, sblks);
    const int per = (sblks + s - 1) / s;
    const int used = (sblks + per - 1) / per;
    const size_t slice = static_cast<size_t>(M) * N;
    if (used > 1 && imma_ensure_splitk(slice * used, capturing)) {
        ks<<<dim3(n_tiles, 1, used), kThreads, q6k_k256_smem_bytes(), stream>>>(g_imma_act.xs8,
                                                                                g_imma_act.xscale, w6,
                                                                                out_f16, M, N, K,
                                                                                g_imma_splitk.buf, per);
        IMP_CUDA_CHECK_LAUNCH();
        const int total = static_cast<int>(slice);
        imma_splitk_finalize(used, total, out_f16, beta1, stream);
        return true;
    }
    (beta1 ? k1 : k0)<<<dim3(n_tiles, 1, 1), kThreads, q6k_k256_smem_bytes(), stream>>>(g_imma_act.xs8,
                                                                                        g_imma_act.xscale, w6,
                                                                                        out_f16, M, N, K,
                                                                                        nullptr, 0);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
