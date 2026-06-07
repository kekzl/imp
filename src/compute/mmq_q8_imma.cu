// =============================================================================
// mmq_q8_imma.cu — INT8 IMMA prefill GEMM family (sm_120a)
// =============================================================================
//
// Fused-dequant prefill GEMMs on the int8 tensor cores for Q8_0 and Q4_K
// (dense + MoE-grouped). See mmq_q8_imma.h for the contract and the design
// notes vs the 2026-05-18 phase-2B ceiling (SMEM-staged scales, 128x128x64
// tiles, +16-B row pad killing 28.3M bank conflicts, one syncthreads pair
// per K-step; 3-stage cp.async REGRESSED -21% — do not re-add).
//
// Unified math (α/β form; Q8_0 repacks with α=d, β=0):
//   out[m,n] = Σ_kb d_a[m,kb] · ( α[n,kb]·Σ_{k∈kb} a_s8·w_s8 + β[n,kb]·rs[m,kb] )
// where rs is the int rowsum of the quantized activation sub-block (couples
// to Q4_K's collapsed (q-8)+dmin terms, see mmq_q4k_imma_layout.h).

#include "compute/mmq_q8_imma.h"
#include "core/logging.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace imp {

namespace {

constexpr int kBN = 128;
constexpr int kBK = 64;   // 2 sub-blocks per K-step
constexpr int kPad = 16;  // smem row pad (bytes): 80-B stride = conflict-free rl-lane access
constexpr int kRow = kBK + kPad;
constexpr int kStages = 2;
constexpr int kThreads = 256;

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 16 : 0;  // src-size 0 → zero-fill (OOB M-tail rows)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_8(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_ca_4(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(glob),
                 "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// One K-step tile load, all 256 threads cooperating (loops are fully
// unrolled compile-time for both BM variants):
//   A   [BM][kBK]  s8     M-tail rows zero-filled
//   B   [kBN][kBK] s8     weight rows always full (N % kBN == 0 gate)
//   Asc [BM][2]    half   activation scale, d-plane cols (kb0, kb0+1)
//   Ars [BM][2]    float  activation rowsum, same cols
//   Bsc [kBN][2][2] half  weight (α, β) interleaved for both kb cols
template <int BM>
__device__ __forceinline__ void load_kstep(int tid, const int8_t* __restrict__ A,
                                           const __half* __restrict__ Asc,
                                           const float* __restrict__ Ars,
                                           const int8_t* __restrict__ B,
                                           const __half* __restrict__ Bsc, int8_t (*sA)[kRow],
                                           int8_t (*sB)[kRow], __half (*sAsc)[2],
                                           float (*sArs)[2], __half (*sBsc)[2][2], int base_m,
                                           int M, int K, int subs, int k_base) {
#pragma unroll
    for (int i = tid; i < BM * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col],
                       A + static_cast<size_t>(base_m + row) * K + k_base + col, valid);
    }
#pragma unroll
    for (int i = tid; i < kBN * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        cp_async_cg_16(&sB[row][col], B + static_cast<size_t>(row) * K + k_base + col, true);
    }
    const int kb0 = k_base / 32;
#pragma unroll
    for (int i = tid; i < BM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        cp_async_ca_4(&sAsc[i][0], Asc + static_cast<size_t>(base_m + i) * subs + kb0, valid);
        cp_async_ca_8(&sArs[i][0], Ars + static_cast<size_t>(base_m + i) * subs + kb0, valid);
    }
#pragma unroll
    for (int i = tid; i < kBN; i += kThreads) {
        cp_async_ca_8(&sBsc[i][0][0], Bsc + (static_cast<size_t>(i) * subs + kb0) * 2, true);
    }
}

// out = (or +=, BETA1) the α/β-scaled IMMA over [BM,kBN] tiles. Grouped MoE
// form: gridDim.z = ne with device expert_offsets; dense passes offsets ==
// nullptr (z extent 1).
template <int BM, bool BETA1>
__global__ void __launch_bounds__(kThreads)
    mmq_imma_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                    const float* __restrict__ x_rowsum, const int8_t* __restrict__ W_s8,
                    const __half* __restrict__ w_sc, __half* __restrict__ out, int M, int N,
                    int K, const int32_t* __restrict__ expert_offsets, size_t w_stride,
                    size_t wsc_stride) {
    constexpr int kWM = (BM == 128) ? 4 : 2;  // warp grid
    constexpr int kWN = (BM == 128) ? 2 : 4;
    constexpr int kTileM = BM / kWM;       // 32 / 16
    constexpr int kTileN = kBN / kWN;      // 64 / 32
    constexpr int kMF = kTileM / 16;       // 2 / 1
    constexpr int kNF = kTileN / 8;        // 8 / 4
    static_assert(kWM * kWN * 32 == kThreads, "warp grid must fill the CTA");

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

    const int subs = K / 32;
    const int8_t* A = X_s8 + row_off * K;
    const __half* Asc = x_scale + row_off * subs;
    const float* Ars = x_rowsum + row_off * subs;
    const int8_t* B = W_s8 + static_cast<size_t>(e) * w_stride + static_cast<size_t>(base_n) * K;
    const __half* Bsc = w_sc + static_cast<size_t>(e) * wsc_stride +
                        static_cast<size_t>(base_n) * subs * 2;
    __half* C = out + row_off * N;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;  // 0..7
    const int cl = lane & 3;   // 0..3

    __shared__ int8_t sA[kStages][BM][kRow];
    __shared__ int8_t sB[kStages][kBN][kRow];
    __shared__ __half sAsc[kStages][BM][2];
    __shared__ float sArs[kStages][BM][2];
    __shared__ __half sBsc[kStages][kBN][2][2];

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int ksteps = K / kBK;
    load_kstep<BM>(tid, A, Asc, Ars, B, Bsc, sA[0], sB[0], sAsc[0], sArs[0], sBsc[0], base_m,
                   rows, K, subs, 0);
    cp_async_commit();

    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int nstage = (ks + 1) & 1;
            load_kstep<BM>(tid, A, Asc, Ars, B, Bsc, sA[nstage], sB[nstage], sAsc[nstage],
                           sArs[nstage], sBsc[nstage], base_m, rows, K, subs, (ks + 1) * kBK);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int kc = kb * 32;
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int arow_lo = warp_m * kTileM + mf * 16 + rl;
                const int arow_hi = arow_lo + 8;
                const int acol = kc + cl * 4;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol]);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol]);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol + 16]);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol + 16]);
                // activation scale + rowsum for the fragment's two rows,
                // hoisted over all n-frags
                const float da_lo = __half2float(sAsc[stage][arow_lo][kb]);
                const float da_hi = __half2float(sAsc[stage][arow_hi][kb]);
                const float rs_lo = sArs[stage][arow_lo][kb];
                const float rs_hi = sArs[stage][arow_hi][kb];

#pragma unroll
                for (int nf = 0; nf < kNF; ++nf) {
                    const int bcol = warp_n * kTileN + nf * 8 + rl;
                    const int bk = kc + cl * 4;
                    uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[stage][bcol][bk]);
                    uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[stage][bcol][bk + 16]);

                    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= 800
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0),
                          "r"(0), "r"(0));
#endif
                    // c0,c1 = rows (rl) cols (cl*2, cl*2+1); c2,c3 = rows (rl+8)
                    const int ncol_lo = warp_n * kTileN + nf * 8 + cl * 2;
                    const __half2 ab_lo = *reinterpret_cast<const __half2*>(&sBsc[stage][ncol_lo][kb][0]);
                    const __half2 ab_hi =
                        *reinterpret_cast<const __half2*>(&sBsc[stage][ncol_lo + 1][kb][0]);
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

    // Epilogue: FP16 store, M-tail predicated (N is a multiple of kBN).
#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
        const int row_lo = base_m + warp_m * kTileM + mf * 16 + rl;
        const int row_hi = row_lo + 8;
#pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
            const int col = base_n + warp_n * kTileN + nf * 8 + cl * 2;
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

// -----------------------------------------------------------------------------
// Q4_K RAW-read kernel: reads the GGUF 144-B super-blocks directly — zero
// extra weight VRAM (the plane-repack variant duplicated all expert weights
// and hit the 32-GB wall on Qwen3-30B MoE: pp512 8x SLOWER under UVM
// paging). One 64-wide K-step = exactly one 32-byte nibble group per B row
// (sub-block pair = low/high nibbles of the same bytes), so the B-fragment
// fetch is one u32 load + shift/mask/vsub4. The (α, β) scale pairs are
// computed from the staged 16-B block headers in a cooperative pass after
// the tile lands (α = d·sc6, β = 8·α − dmin·m6; algebra identical to the
// mmq_q4k_imma_reorder form, see mmq_q4k_imma_layout.h).
// -----------------------------------------------------------------------------

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

constexpr int kQRow = 32 + 16;  // staged qs row: 32 B group + 16-B pad (bank stride 12 words)

template <int BM>
__device__ __forceinline__ void load_kstep_q4k(int tid, const int8_t* __restrict__ A,
                                               const __half* __restrict__ Asc,
                                               const float* __restrict__ Ars,
                                               const uint8_t* __restrict__ Wq4k, int base_n,
                                               int N, int sblk_count, int8_t (*sA)[kRow],
                                               uint8_t (*sBq)[kQRow], uint8_t (*sBh)[16],
                                               __half (*sAsc)[2], float (*sArs)[2], int base_m,
                                               int M, int K, int subs, int k_base) {
#pragma unroll
    for (int i = tid; i < BM * 4; i += kThreads) {
        const int row = i >> 2;
        const int col = (i & 3) * 16;
        const bool valid = (base_m + row) < M;
        cp_async_cg_16(&sA[row][col],
                       A + static_cast<size_t>(base_m + row) * K + k_base + col, valid);
    }
    const int ks = k_base / kBK;
    const int sblk = ks >> 2;          // super-block index along K
    const int grp = ks & 3;            // 32-byte nibble group within it
#pragma unroll
    for (int i = tid; i < kBN * 3; i += kThreads) {
        const int row = i / 3;
        const int part = i % 3;
        const uint8_t* blk = Wq4k + (static_cast<size_t>(base_n + row) * sblk_count + sblk) * 144;
        if (part == 0) {
            cp_async_cg_16(&sBh[row][0], blk, true);  // d, dmin, 12-B scales
        } else {
            const int off = (part - 1) * 16;
            cp_async_cg_16(&sBq[row][off], blk + 16 + grp * 32 + off, true);
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

    __shared__ int8_t sA[kStages][BM][kRow];
    __shared__ uint8_t sBq[kStages][kBN][kQRow];
    __shared__ uint8_t sBh[kStages][kBN][16];
    __shared__ __half sAsc[kStages][BM][2];
    __shared__ float sArs[kStages][BM][2];
    __shared__ __half2 sBab[kStages][kBN][2];  // (α, β) per (col, kb)

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int ksteps = K / kBK;
    load_kstep_q4k<BM>(tid, A, Asc, Ars, W, base_n, N, sblk_count, sA[0], sBq[0], sBh[0],
                       sAsc[0], sArs[0], base_m, rows, K, subs, 0);
    cp_async_commit();

    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int nstage = (ks + 1) & 1;
            load_kstep_q4k<BM>(tid, A, Asc, Ars, W, base_n, N, sblk_count, sA[nstage],
                               sBq[nstage], sBh[nstage], sAsc[nstage], sArs[nstage], base_m, rows,
                               K, subs, (ks + 1) * kBK);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // header → (α, β) cooperative pass: one thread per (col, kb)
        {
            const int j0 = (2 * ks) & 7;  // sub-block within the super-block
#pragma unroll
            for (int i = tid; i < kBN * 2; i += kThreads) {
                const int row = i >> 1;
                const int kb = i & 1;
                const uint8_t* h = &sBh[stage][row][0];
                __half d_h, dmin_h;
                memcpy(&d_h, h, 2);
                memcpy(&dmin_h, h + 2, 2);
                uint32_t sc, mn;
                q4k_scale_min(j0 + kb, h + 4, sc, mn);
                const float a = __half2float(d_h) * static_cast<float>(sc);
                const float b =
                    8.0f * a - __half2float(dmin_h) * static_cast<float>(mn);
                sBab[stage][row][kb] = __floats2half2_rn(a, b);
            }
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
            const int kc = kb * 32;
            const uint32_t shift = kb * 4;
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
                    const uint32_t raw0 =
                        *reinterpret_cast<const uint32_t*>(&sBq[stage][bcol][cl * 4]);
                    const uint32_t raw1 =
                        *reinterpret_cast<const uint32_t*>(&sBq[stage][bcol][cl * 4 + 16]);
                    const uint32_t b0 = __vsub4((raw0 >> shift) & 0x0F0F0F0Fu, 0x08080808u);
                    const uint32_t b1 = __vsub4((raw1 >> shift) & 0x0F0F0F0Fu, 0x08080808u);

                    int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= 800
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0),
                          "r"(0), "r"(0));
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
        __syncthreads();
    }

#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
        const int row_lo = base_m + warp_m * kTileM + mf * 16 + rl;
        const int row_hi = row_lo + 8;
#pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
            const int col = base_n + warp_n * kTileN + nf * 8 + cl * 2;
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

// -----------------------------------------------------------------------------
// Q8_0 SoA split: raw 34-B blocks {half d; int8 qs[32]} (2-aligned! memcpy
// only) → qs plane [N][K] s8 + interleaved (α=d, β=0) plane [N][K/32][2].
// -----------------------------------------------------------------------------
__global__ void q8_split_kernel(const uint8_t* __restrict__ src, int8_t* __restrict__ qs_plane,
                                __half* __restrict__ sc_plane, int n_blocks_total) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks_total) return;
    const uint8_t* blk = src + static_cast<size_t>(b) * 34;
    __half d;
    memcpy(&d, blk, 2);
    sc_plane[static_cast<size_t>(b) * 2] = d;
    sc_plane[static_cast<size_t>(b) * 2 + 1] = __float2half(0.0f);
    int8_t* dst = qs_plane + static_cast<size_t>(b) * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = static_cast<int8_t>(blk[2 + i]);
}

// Activation quantizer: 8 warps per block, grid-stride over (m, sub) pairs
// (the shared 32-thread-block version ran at ~150 GB/s). Emits s8 + half
// scale + float rowsum (the rowsum couples to the Q4_K β term; ~free here).
__global__ void quantize_act_fast_kernel(const __half* __restrict__ X, int M, int K,
                                         int8_t* __restrict__ xs8, __half* __restrict__ xscale,
                                         float* __restrict__ xrowsum) {
    const int subs = K / 32;
    const int total = M * subs;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int nwarps = gridDim.x * (blockDim.x >> 5);
    for (int idx = warp; idx < total; idx += nwarps) {
        const int m = idx / subs;
        const int s = idx - m * subs;
        const size_t off = static_cast<size_t>(m) * K + s * 32;
        const float v = __half2float(X[off + lane]);
        float amax = fabsf(v);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
        const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = __float2int_rn(v * inv);
        q = max(-127, min(127, q));
        xs8[off + lane] = static_cast<int8_t>(q);
        int sq = q;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xFFFFFFFFu, sq, o);
        if (lane == 0) {
            xscale[static_cast<size_t>(m) * subs + s] =
                __float2half((amax > 0.0f) ? (amax / 127.0f) : 0.0f);
            xrowsum[static_cast<size_t>(m) * subs + s] = static_cast<float>(sq);
        }
    }
}

struct WeightPlanes {
    int8_t* qs = nullptr;
    __half* sc = nullptr;  // interleaved (α, β) [N][K/32][2]
    int N = 0;             // total rows (ne × per-expert rows for MoE)
    int K = 0;
};
struct ActScratch {
    int8_t* xs8 = nullptr;
    __half* xscale = nullptr;
    float* xrowsum = nullptr;
    size_t cap_mk = 0;
    size_t cap_msubs = 0;
};

std::mutex g_mtx;
std::unordered_map<const void*, WeightPlanes> g_weights;
ActScratch g_act;

bool stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing(stream, &st) == cudaSuccess &&
           st == cudaStreamCaptureStatusActive;
}

bool ensure_weight(const void* src, int N, int K, cudaStream_t stream, bool capturing) {
    // Q8_0 only: the SoA planes cost 1.06x the source — fine for dense Q8
    // models, but Q4_K (esp. MoE experts) reads the raw blocks in-kernel
    // instead (the plane variant duplicated all expert weights and hit the
    // 32-GB VRAM wall on Qwen3-30B: pp512 8x SLOWER under UVM paging).
    auto it = g_weights.find(src);
    if (it != g_weights.end() && it->second.N == N && it->second.K == K) return true;
    if (capturing) return false;  // never allocate inside graph capture

    WeightPlanes w;
    w.N = N;
    w.K = K;
    const size_t subs = static_cast<size_t>(K) / 32;
    if (cudaMalloc(&w.qs, static_cast<size_t>(N) * K) != cudaSuccess) return false;
    if (cudaMalloc(&w.sc, static_cast<size_t>(N) * subs * 2 * sizeof(__half)) != cudaSuccess) {
        cudaFree(w.qs);
        return false;
    }
    const int total = N * static_cast<int>(subs);
    q8_split_kernel<<<(total + 255) / 256, 256, 0, stream>>>(static_cast<const uint8_t*>(src),
                                                             w.qs, w.sc, total);
    g_weights[src] = w;
    return true;
}

bool ensure_act(int M, int K, bool capturing) {
    const size_t mk = static_cast<size_t>(M) * K;
    const size_t msubs = static_cast<size_t>(M) * (K / 32);
    if (g_act.xs8 && g_act.cap_mk >= mk && g_act.cap_msubs >= msubs) return true;
    if (capturing) return false;
    if (g_act.xs8) {
        cudaFree(g_act.xs8);
        cudaFree(g_act.xscale);
        cudaFree(g_act.xrowsum);
        g_act = ActScratch{};
    }
    if (cudaMalloc(&g_act.xs8, mk) != cudaSuccess) return false;
    if (cudaMalloc(&g_act.xscale, msubs * sizeof(__half)) != cudaSuccess) return false;
    if (cudaMalloc(&g_act.xrowsum, msubs * sizeof(float)) != cudaSuccess) return false;
    g_act.cap_mk = mk;
    g_act.cap_msubs = msubs;
    return true;
}

void quantize_act(const __half* x, int M, int K, cudaStream_t stream) {
    // NO memoization: workspace buffers (moe gathered, layer activations) are
    // REUSED across layers with the same pointer — a (ptr, M, K) memo served
    // layer-1 activations to every later layer (PPL 31.6 → 441k, found
    // 2026-06-07). The kernel costs ~7 µs; quantize unconditionally.
    const int total_warps = M * (K / 32);
    const int blocks = min(2048, (total_warps + 7) / 8);
    quantize_act_fast_kernel<<<blocks, 256, 0, stream>>>(x, M, K, g_act.xs8, g_act.xscale,
                                                         g_act.xrowsum);
}

bool gemm_common(const void* w_blocks, bool q4k, const __half* x_f16, __half* out_f16, int M,
                 int N, int K, cudaStream_t stream, float beta, const int32_t* d_offsets,
                 int h_max_rows, int expanded, int ne) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const bool capturing = stream_capturing(stream);
    if (!q4k && !ensure_weight(w_blocks, ne * N, K, stream, capturing)) return false;
    const int act_rows = d_offsets ? expanded : M;
    if (!ensure_act(act_rows, K, capturing)) return false;
    quantize_act(x_f16, act_rows, K, stream);

    const int grid_m_rows = d_offsets ? h_max_rows : M;
    const bool small_m = d_offsets && h_max_rows < 96;
    const int bm = small_m ? 32 : 128;
    dim3 grid(N / kBN, (grid_m_rows + bm - 1) / bm, ne);

    if (q4k) {
        // raw-read kernel: zero extra weight VRAM
        const uint8_t* w4 = static_cast<const uint8_t*>(w_blocks);
        const size_t w_stride_blocks = static_cast<size_t>(N) * (K / 256);
        if (small_m) {
            mmq_imma_q4k_raw_kernel<32, false><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w4, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
        } else if (beta == 1.0f) {
            mmq_imma_q4k_raw_kernel<128, true><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w4, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
        } else {
            mmq_imma_q4k_raw_kernel<128, false><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w4, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
        }
        return true;
    }

    const auto& w = g_weights[w_blocks];
    const size_t w_stride = static_cast<size_t>(N) * K;
    const size_t wsc_stride = static_cast<size_t>(N) * (K / 32) * 2;
    if (small_m) {
        mmq_imma_kernel<32, false><<<grid, kThreads, 0, stream>>>(
            g_act.xs8, g_act.xscale, g_act.xrowsum, w.qs, w.sc, out_f16, M, N, K, d_offsets,
            w_stride, wsc_stride);
    } else if (beta == 1.0f) {
        mmq_imma_kernel<128, true><<<grid, kThreads, 0, stream>>>(
            g_act.xs8, g_act.xscale, g_act.xrowsum, w.qs, w.sc, out_f16, M, N, K, d_offsets,
            w_stride, wsc_stride);
    } else {
        mmq_imma_kernel<128, false><<<grid, kThreads, 0, stream>>>(
            g_act.xs8, g_act.xscale, g_act.xrowsum, w.qs, w.sc, out_f16, M, N, K, d_offsets,
            w_stride, wsc_stride);
    }
    return true;
}

}  // namespace

bool mmq_q8_imma_gemm(const void* w_q8_blocks, const __half* x_f16, __half* out_f16, int M, int N,
                      int K, cudaStream_t stream, float beta) {
    if (M < 64 || N % kBN != 0 || K % kBK != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    return gemm_common(w_q8_blocks, false, x_f16, out_f16, M, N, K, stream, beta, nullptr, 0, 0,
                       1);
}

bool mmq_q4k_imma_gemm(const void* w_q4k_blocks, const __half* x_f16, __half* out_f16, int M,
                       int N, int K, cudaStream_t stream, float beta) {
    if (M < 64 || N % kBN != 0 || K % 256 != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    return gemm_common(w_q4k_blocks, true, x_f16, out_f16, M, N, K, stream, beta, nullptr, 0, 0,
                       1);
}

bool mmq_imma_moe_gemm(const void* w_blocks, bool qtype_is_q4k, const __half* x_f16,
                       __half* out_f16, const int32_t* d_offsets, int h_max_rows, int expanded,
                       int ne, int N, int K, cudaStream_t stream) {
    if (N % kBN != 0 || K % (qtype_is_q4k ? 256 : kBK) != 0) return false;
    if (h_max_rows <= 0 || expanded <= 0 || ne <= 0) return false;
    const bool ok = gemm_common(w_blocks, qtype_is_q4k, x_f16, out_f16, /*M=*/0, N, K, stream,
                                0.0f, d_offsets, h_max_rows, expanded, ne);
    static bool logged = false;
    if (ok && !logged) {
        logged = true;
        IMP_LOG_INFO("MoE IMMA prefill ACTIVE (%s, ne=%d N=%d K=%d max_rows=%d)",
                     qtype_is_q4k ? "Q4_K" : "Q8_0", ne, N, K, h_max_rows);
    }
    return ok;
}

void mmq_q8_imma_release_all() {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (auto& [_, w] : g_weights) {
        cudaFree(w.qs);
        cudaFree(w.sc);
    }
    g_weights.clear();
    if (g_act.xs8) {
        cudaFree(g_act.xs8);
        cudaFree(g_act.xscale);
        cudaFree(g_act.xrowsum);
        g_act = ActScratch{};
    }
}

}  // namespace imp
