// Q4_K RAW-read IMMA prefill kernel (sm_120a). Split out of mmq_q8_imma.cu
// (recompile-blast-radius gate); tile constants/cp.async in mmq_q8_imma_internal.cuh,
// kernel template declared there, launched from mmq_q8_imma.cu dispatch.

#include "compute/mmq_q8_imma_internal.cuh"

#include <cstring>

namespace imp {

namespace {

// Q4_K RAW-read kernel: reads GGUF 144-B super-blocks directly, zero extra weight VRAM
// (plane-repack duplicated all expert weights, hit 32-GB wall on Qwen3-30B MoE: pp512
// 8x slower under UVM paging). One 128-wide K-step = 4 sub-blocks = two 32-byte nibble groups
// per B row (sub-block pair = low/high nibbles), 3 barriers per 16 MMAs per warp (#2441);
// B-fragment fetch = one u32 load + shift/mask/vsub4.
// alpha/beta computed from staged 16-B block headers after tile lands (alpha=d*sc6,
// beta=8*alpha-dmin*m6; same algebra as mmq_q4k_imma_reorder, see mmq_q4k_imma_layout.h).

__device__ __forceinline__ void q4k_scale_min(int j, const uint8_t* q, uint32_t& sc,
                                              uint32_t& mn) {
    if (j < 4) {
        sc = q[j] & 63u;
        mn = q[j + 4] & 63u;
    } else {
        sc = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Header -> (alpha, beta) per (col, kb) of K-step ks, one thread per entry.
__device__ __forceinline__ void q4k_header_pass(int tid, int ks, const uint8_t (*sBh)[16],
                                                __half2 (*sBab)[kQ4kSub]) {
    const int j0 = (kQ4kSub * ks) & 7;  // first sub-block within the super-block
#pragma unroll
    for (int i = tid; i < kBN * kQ4kSub; i += kThreads) {
        const int row = i >> 2;
        const int kb = i & 3;
        const uint8_t* h = &sBh[row][0];
        __half d_h, dmin_h;
        memcpy(&d_h, h, 2);
        memcpy(&dmin_h, h + 2, 2);
        uint32_t sc, mn;
        q4k_scale_min(j0 + kb, h + 4, sc, mn);
        const float a = __half2float(d_h) * static_cast<float>(sc);
        const float b = 8.0f * a - __half2float(dmin_h) * static_cast<float>(mn);
        sBab[row][kb] = __floats2half2_rn(a, b);
    }
}

// Four 8x8 b16 tiles (8 rows x 16 B each); lane l addresses row (l & 7) of tile l >> 3.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                            const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(s));
}

template <int BM>
__device__ __forceinline__ void load_kstep_q4k(int tid, const int8_t* __restrict__ A,
                                               const __half* __restrict__ Asc, const float* __restrict__ Ars,
                                               const uint8_t* __restrict__ Wq4k, int base_n, int sblk_count,
                                               int8_t (*sA)[kQ4kRow], uint8_t (*sBq)[kQ4kQRow],
                                               uint8_t (*sBh)[16], __half (*sAsc)[kQ4kSub],
                                               float (*sArs)[kQ4kSub], int base_m, int M, int K, int subs,
                                               int k_base, int base_n_rows) {
#pragma unroll
    for (int i = tid; i < BM * (kQ4kBK / 16); i += kThreads) {
        const int row = i >> 3;
        const int col = (i & 7) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col],
                       A + static_cast<size_t>(base_m + row) * K + k_base + col, valid);
    }
    const int ks = k_base / kQ4kBK;
    const int sblk = ks >> 1;  // super-block index along K
    const int half = ks & 1;   // 64-byte nibble-group pair within it
    // #2218 bounded: half = ks & 1 <= 1: half * 64 <= 64 B (in-block offset)
#pragma unroll
    for (int i = tid; i < kBN * 5; i += kThreads) {
        const int row = i / 5;
        const int part = i % 5;
        const bool bvalid = (base_n_rows < 0) || (row < base_n_rows);
        const uint8_t* blk = Wq4k + (static_cast<size_t>(base_n + row) * sblk_count + sblk) * 144;
        if (part == 0) {
            cp_async_cg_16(&sBh[row][0], blk, bvalid);  // d, dmin, 12-B scales
        } else {
            const int off = (part - 1) * 16;
            cp_async_cg_16(&sBq[row][off], blk + 16 + static_cast<ptrdiff_t>(half * 64) + off, bvalid);
        }
    }
    const int kb0 = k_base / 32;
#pragma unroll
    for (int i = tid; i < BM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        cp_async_ca_8(&sAsc[i][0], Asc + static_cast<size_t>(base_m + i) * subs + kb0, valid);
        cp_async_cg_16(&sArs[i][0], Ars + static_cast<size_t>(base_m + i) * subs + kb0, valid);
    }
}

}  // namespace

template <int BM, bool BETA1>
__global__ void __launch_bounds__(kThreads)
    mmq_imma_q4k_raw_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                            const float* __restrict__ x_rowsum, const uint8_t* __restrict__ Wq4k,
                            __half* __restrict__ out, int M, int N, int K,
                            const int32_t* __restrict__ expert_offsets, size_t w_stride_blocks) {
    constexpr int kWM = (BM == 128) ? 4 : 2;
    constexpr int kWN = (BM == 128) ? 2 : 4;
    constexpr int kTileM = BM / kWM;
    constexpr int kTileN = kBN / kWN;
    constexpr int kMF = kTileM / 16;
    constexpr int kNF = kTileN / 8;
    constexpr bool kHoistB = kMF == 1;  // B fragments of all nf loaded once per nibble group

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
    if (base_m >= rows || rows == 0) return;
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);

    const int subs = K / 32;
    const int sblk_count = K / 256;
    const int8_t* A = X_s8 + row_off * K;
    const __half* Asc = x_scale + row_off * subs;
    const float* Ars = x_rowsum + row_off * subs;
    const uint8_t* W = Wq4k + static_cast<size_t>(e) * w_stride_blocks * 144;
    __half* C = out + row_off * N;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;
    const int cl = lane & 3;
    // #2218 bounded: cl = lane & 3 <= 3: smem column cl * 4 <= 12

    // Dynamic smem (q4k_smem_bytes): two 128-K stages exceed the 48 KB static cap at BM=128.
    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto* sA = reinterpret_cast<int8_t (*)[BM][kQ4kRow]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * BM * kQ4kRow;
    auto* sBq = reinterpret_cast<uint8_t (*)[kBN][kQ4kQRow]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * kBN * kQ4kQRow;
    auto* sBh = reinterpret_cast<uint8_t (*)[kBN][16]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * kBN * 16;
    auto* sArs = reinterpret_cast<float (*)[BM][kQ4kSub]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * BM * kQ4kSub * sizeof(float);
    auto* sBab = reinterpret_cast<__half2(*)[kBN][kQ4kSub]>(smem_raw + off);  // (alpha, beta) per (col, kb)
    off += static_cast<size_t>(kStages) * kBN * kQ4kSub * sizeof(__half2);
    auto* sAsc = reinterpret_cast<__half(*)[BM][kQ4kSub]>(smem_raw + off);

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int ksteps = K / kQ4kBK;
    load_kstep_q4k<BM>(tid, A, Asc, Ars, W, base_n, sblk_count, sA[0], sBq[0], sBh[0], sAsc[0], sArs[0],
                       base_m, rows, K, subs, 0, n_rem);
    cp_async_commit();

    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int nstage = (ks + 1) & 1;
            load_kstep_q4k<BM>(tid, A, Asc, Ars, W, base_n, sblk_count, sA[nstage], sBq[nstage], sBh[nstage],
                               sAsc[nstage], sArs[nstage], base_m, rows, K, subs, (ks + 1) * kQ4kBK, n_rem);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        q4k_header_pass(tid, ks, sBh[stage], sBab[stage]);
        __syncthreads();

#pragma unroll 1
        for (int g = 0; g < kQ4kSub / 2; ++g) {  // nibble group: sub-blocks 2g (low) and 2g + 1 (high)
            // B: one ldmatrix.x4 = raw words (k 0-15, k 16-31) of two n-fragments
            uint32_t raw[kNF][2];
            if constexpr (kHoistB) {
#pragma unroll
                for (int nf = 0; nf < kNF; nf += 2) {
                    const int m = lane >> 3;
                    const int col = warp_n * kTileN + nf * 8 + (lane & 7) + (m >> 1) * 8;
                    ldmatrix_x4(raw[nf][0], raw[nf][1], raw[nf + 1][0], raw[nf + 1][1],
                                &sBq[stage][col][g * 32 + (m & 1) * 16]);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int kb = 2 * g + h;
                const uint32_t shift = h * 4;
#pragma unroll
                for (int mf = 0; mf < kMF; ++mf) {
                    const int arow_lo = warp_m * kTileM + mf * 16 + rl;
                    const int arow_hi = arow_lo + 8;
                    // A: one ldmatrix.x4 = (rows 0-7, 8-15) x (k 0-15, 16-31) of the 16 x 32 fragment
                    uint32_t a0, a1, a2, a3;
                    {
                        const int m = lane >> 3;
                        const int row = warp_m * kTileM + mf * 16 + (lane & 7) + (m & 1) * 8;
                        ldmatrix_x4(a0, a1, a2, a3, &sA[stage][row][kb * 32 + (m >> 1) * 16]);
                    }
                    const float da_lo = __half2float(sAsc[stage][arow_lo][kb]);
                    const float da_hi = __half2float(sAsc[stage][arow_hi][kb]);
                    const float rs_lo = sArs[stage][arow_lo][kb];
                    const float rs_hi = sArs[stage][arow_hi][kb];

#pragma unroll
                    for (int nf = 0; nf < kNF; ++nf) {
                        if constexpr (!kHoistB) {
                            const int bcol = warp_n * kTileN + nf * 8 +
                                             rl;  // BM=128: scalar (hoisted = 254 regs)
                            raw[nf][0] = *reinterpret_cast<const uint32_t*>(
                                &sBq[stage][bcol][g * 32 + cl * 4]);
                            raw[nf][1] = *reinterpret_cast<const uint32_t*>(
                                &sBq[stage][bcol][g * 32 + cl * 4 + 16]);
                        }
                        const uint32_t b0 = __vsub4((raw[nf][0] >> shift) & 0x0F0F0F0Fu, 0x08080808u);
                        const uint32_t b1 = __vsub4((raw[nf][1] >> shift) & 0x0F0F0F0Fu, 0x08080808u);

                        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= 800
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0),
                              "r"(0));
#endif
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
            if (col + 1 >= N) continue;  // N-tail: OOB columns are never stored
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
template __global__ void mmq_imma_q4k_raw_kernel<32, false>(const int8_t*, const __half*,
                                                           const float*, const uint8_t*, __half*,
                                                           int, int, int, const int32_t*, size_t);
template __global__ void mmq_imma_q4k_raw_kernel<128, true>(const int8_t*, const __half*,
                                                           const float*, const uint8_t*, __half*,
                                                           int, int, int, const int32_t*, size_t);
template __global__ void mmq_imma_q4k_raw_kernel<128, false>(const int8_t*, const __half*,
                                                            const float*, const uint8_t*, __half*,
                                                            int, int, int, const int32_t*, size_t);

}  // namespace imp
