// Q5_K RAW-read IMMA prefill kernel (sm_120a): the Q4_K raw kernel (mmq_q8_imma_q4k.cu) plus the
// fifth bit. A 176-B super-block is d, dmin, 12-B scales, 32-B qh, 128-B qs; element l of 32-byte
// group g takes bit 2g (low nibble) / 2g+1 (high nibble) of qh[l]. Values 0..31 enter the s8 MMA as
// q - 16, so alpha = d*sc, beta = 16*alpha - dmin*m. Replaces dequant-all-experts + FP16 grouped
// GEMM for Q5_K MoE tensors (Qwen3.6-35B-A3B UD-Q4_K_M down_proj: 791 us/layer of dequant at pp512).

#include "compute/mmq_q8_imma_internal.cuh"

#include <cstring>

namespace imp {

namespace {

constexpr int kQ5Block = 176;
constexpr int kQsOff = 48;  // d, dmin, scales[12], qh[32]
constexpr int kQRow = 32 + 16;

__device__ __forceinline__ void q5k_scale_min(int j, const uint8_t* q, uint32_t& sc, uint32_t& mn) {
    if (j < 4) {
        sc = q[j] & 63u;
        mn = q[j + 4] & 63u;
    } else {
        sc = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

template <int BM>
__device__ __forceinline__ void load_kstep_q5k(int tid, const int8_t* __restrict__ A,
                                               const __half* __restrict__ Asc, const float* __restrict__ Ars,
                                               const uint8_t* __restrict__ Wq5k, int base_n, int sblk_count,
                                               int8_t (*sA)[kRow], uint8_t (*sBq)[kQRow],
                                               uint8_t (*sBqh)[kQRow], uint8_t (*sBh)[16], __half (*sAsc)[2],
                                               float (*sArs)[2], int base_m, int M, int K, int subs,
                                               int k_base, int base_n_rows) {
#pragma unroll
    for (int i = tid; i < BM * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col], A + static_cast<size_t>(base_m + row) * K + k_base + col, valid);
    }
    const int ks = k_base / kBK;
    const int sblk = ks >> 2;  // super-block index along K
    const int grp = ks & 3;    // 32-byte nibble group within it
#pragma unroll
    for (int i = tid; i < kBN * 5; i += kThreads) {
        const int row = i / 5;
        const int part = i % 5;
        const bool bvalid = (base_n_rows < 0) || (row < base_n_rows);
        const uint8_t* blk = Wq5k + (static_cast<size_t>(base_n + row) * sblk_count + sblk) * kQ5Block;
        if (part == 0) {
            cp_async_cg_16(&sBh[row][0], blk, bvalid);  // d, dmin, 12-B scales
        } else if (part < 3) {
            const int off = (part - 1) * 16;
            cp_async_cg_16(&sBqh[row][off], blk + 16 + off, bvalid);  // qh, shared by all 4 groups
        } else {
            const int off = (part - 3) * 16;
            cp_async_cg_16(&sBq[row][off], blk + kQsOff + grp * 32 + off, bvalid);
        }
    }
    const int kb0 = k_base / 32;
#pragma unroll
    for (int i = tid; i < BM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        cp_async_ca_4(&sAsc[i][0], Asc + static_cast<size_t>(base_m + i) * subs + kb0, valid);
        cp_async_ca_8(&sArs[i][0], Ars + static_cast<size_t>(base_m + i) * subs + kb0, valid);
    }
}

// Four 5-bit weights of one u32 lane group as s8 (q - 16): nibbles at `shift`, fifth bit at `hbit`.
__device__ __forceinline__ uint32_t q5k_s8x4(uint32_t qs, uint32_t qh, uint32_t shift, uint32_t hbit) {
    const uint32_t lo = (qs >> shift) & 0x0F0F0F0Fu;
    const uint32_t hi = ((qh >> hbit) & 0x01010101u) << 4;
    return __vsub4(lo | hi, 0x10101010u);
}

}  // namespace

template <int BM, bool BETA1>
__global__ void __launch_bounds__(kThreads) mmq_imma_q5k_raw_kernel(
    const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale, const float* __restrict__ x_rowsum,
    const uint8_t* __restrict__ Wq5k, __half* __restrict__ out, int M, int N, int K,
    const int32_t* __restrict__ expert_offsets, size_t w_stride_blocks) {
    constexpr int kWM = (BM == 128) ? 4 : 2;
    constexpr int kWN = (BM == 128) ? 2 : 4;
    constexpr int kTileM = BM / kWM;
    constexpr int kTileN = kBN / kWN;
    constexpr int kMF = kTileM / 16;
    constexpr int kNF = kTileN / 8;

    int rows = M;
    size_t row_off = 0;
    const int e = blockIdx.z;
    if (expert_offsets != nullptr) {
        const int32_t o0 = __ldg(&expert_offsets[e]);
        const int32_t o1 = __ldg(&expert_offsets[e + 1]);
        row_off = static_cast<size_t>(o0);
        rows = o1 - o0;
    }
    const int base_m = blockIdx.y * BM;
    const int base_n = blockIdx.x * kBN;
    if (base_m >= rows || rows == 0)
        return;
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);

    const int subs = K / 32;
    const int sblk_count = K / 256;
    const int8_t* A = X_s8 + row_off * K;
    const __half* Asc = x_scale + row_off * subs;
    const float* Ars = x_rowsum + row_off * subs;
    const uint8_t* W = Wq5k + static_cast<size_t>(e) * w_stride_blocks * kQ5Block;
    __half* C = out + row_off * N;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;
    const int cl = lane & 3;

    // Dynamic smem (q5k_smem_bytes): the staged qh rows push the Q4_K layout past the 48 KB static cap.
    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto* sA = reinterpret_cast<int8_t(*)[BM][kRow]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * BM * kRow;
    auto* sBq = reinterpret_cast<uint8_t(*)[kBN][kQRow]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * kBN * kQRow;
    auto* sBqh = reinterpret_cast<uint8_t(*)[kBN][kQRow]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * kBN * kQRow;
    auto* sBh = reinterpret_cast<uint8_t(*)[kBN][16]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * kBN * 16;
    auto* sAsc = reinterpret_cast<__half(*)[BM][2]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * BM * 2 * sizeof(__half);
    auto* sArs = reinterpret_cast<float(*)[BM][2]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * BM * 2 * sizeof(float);
    auto* sBab = reinterpret_cast<__half2(*)[kBN][2]>(smem_raw + off);  // (alpha, beta) per (col, kb)

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int ksteps = K / kBK;
    load_kstep_q5k<BM>(tid, A, Asc, Ars, W, base_n, sblk_count, sA[0], sBq[0], sBqh[0], sBh[0], sAsc[0],
                       sArs[0], base_m, rows, K, subs, 0, n_rem);
    cp_async_commit();

    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int nstage = (ks + 1) & 1;
            load_kstep_q5k<BM>(tid, A, Asc, Ars, W, base_n, sblk_count, sA[nstage], sBq[nstage], sBqh[nstage],
                               sBh[nstage], sAsc[nstage], sArs[nstage], base_m, rows, K, subs, (ks + 1) * kBK,
                               n_rem);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        const int j0 = (2 * ks) & 7;  // sub-block within the super-block
        {
#pragma unroll
            for (int i = tid; i < kBN * 2; i += kThreads) {
                const int row = i >> 1;
                const int kb = i & 1;
                const uint8_t* h = &sBh[stage][row][0];
                __half d_h, dmin_h;
                memcpy(&d_h, h, 2);
                memcpy(&dmin_h, h + 2, 2);
                uint32_t sc, mn;
                q5k_scale_min(j0 + kb, h + 4, sc, mn);
                const float a = __half2float(d_h) * static_cast<float>(sc);
                const float b = 16.0f * a - __half2float(dmin_h) * static_cast<float>(mn);
                sBab[stage][row][kb] = __floats2half2_rn(a, b);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int kc = kb * 32;
            const uint32_t shift = kb * 4;
            const uint32_t hbit = static_cast<uint32_t>(j0 + kb);  // 2 * group + nibble half
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int arow_lo = warp_m * kTileM + mf * 16 + rl;
                const int arow_hi = arow_lo + 8;
                const int acol = kc + cl * 4;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol]);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol]);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol + 16]);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol + 16]);
                const float da_lo = __half2float(sAsc[stage][arow_lo][kb]);
                const float da_hi = __half2float(sAsc[stage][arow_hi][kb]);
                const float rs_lo = sArs[stage][arow_lo][kb];
                const float rs_hi = sArs[stage][arow_hi][kb];

#pragma unroll
                for (int nf = 0; nf < kNF; ++nf) {
                    const int bcol = warp_n * kTileN + nf * 8 + rl;
                    const uint32_t b0 =
                        q5k_s8x4(*reinterpret_cast<const uint32_t*>(&sBq[stage][bcol][cl * 4]),
                                 *reinterpret_cast<const uint32_t*>(&sBqh[stage][bcol][cl * 4]), shift, hbit);
                    const uint32_t b1 = q5k_s8x4(
                        *reinterpret_cast<const uint32_t*>(&sBq[stage][bcol][cl * 4 + 16]),
                        *reinterpret_cast<const uint32_t*>(&sBqh[stage][bcol][cl * 4 + 16]), shift, hbit);

                    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0),
                          "r"(0));
                    const int ncol_lo = warp_n * kTileN + nf * 8 + cl * 2;
                    const __half2 ab_lo = sBab[stage][ncol_lo][kb];
                    const __half2 ab_hi = sBab[stage][ncol_lo + 1][kb];
                    const float al = __half2float(__low2half(ab_lo));
                    const float bl = __half2float(__high2half(ab_lo));
                    const float ah = __half2float(__low2half(ab_hi));
                    const float bh = __half2float(__high2half(ab_hi));
                    acc[mf][nf][0] += da_lo * fmaf(al, static_cast<float>(c0), bl * rs_lo);
                    acc[mf][nf][1] += da_lo * fmaf(ah, static_cast<float>(c1), bh * rs_lo);
                    acc[mf][nf][2] += da_hi * fmaf(al, static_cast<float>(c2), bl * rs_hi);
                    acc[mf][nf][3] += da_hi * fmaf(ah, static_cast<float>(c3), bh * rs_hi);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
        const int row_lo = base_m + warp_m * kTileM + mf * 16 + rl;
        const int row_hi = row_lo + 8;
#pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
            const int col = base_n + warp_n * kTileN + nf * 8 + cl * 2;
            if (col + 1 >= N)
                continue;  // N-tail: OOB columns are never stored
            if (row_lo < rows) {
                __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row_lo) * N + col]);
                __half2 v = __floats2half2_rn(acc[mf][nf][0], acc[mf][nf][1]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
            if (row_hi < rows) {
                __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row_hi) * N + col]);
                __half2 v = __floats2half2_rn(acc[mf][nf][2], acc[mf][nf][3]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
        }
    }
}

// Explicit instantiations launched by the dispatch in mmq_q8_imma.cu.
template __global__ void mmq_imma_q5k_raw_kernel<32, false>(const int8_t*, const __half*, const float*,
                                                            const uint8_t*, __half*, int, int, int,
                                                            const int32_t*, size_t);
template __global__ void mmq_imma_q5k_raw_kernel<128, true>(const int8_t*, const __half*, const float*,
                                                            const uint8_t*, __half*, int, int, int,
                                                            const int32_t*, size_t);
template __global__ void mmq_imma_q5k_raw_kernel<128, false>(const int8_t*, const __half*, const float*,
                                                             const uint8_t*, __half*, int, int, int,
                                                             const int32_t*, size_t);

}  // namespace imp
