// Q8_0 plane IMMA prefill kernel at 2 CTAs/SM (#2267e, sm_120a). Same contract as the legacy
// mmq_imma_kernel (mmq_q8_imma.cu, pure alpha, WB=false); declared in mmq_q8_imma_internal.cuh,
// launched from the mmq_q8_imma.cu dispatch.
// Registers: 128 at BM=128 (__launch_bounds__(256, 2)), 0 spill; smem 45088 B static per CTA.
// K loop: mbarrier stage handoff, no __syncthreads. full[s] = 256 cp.async arrivals, empty[s] = 8 warps.
// Per element: fma(alpha, fma(d_a, c_biased, -d_a*bias), acc); IMMA C = bias, no I2F in the loop.

#include "compute/mmq_q8_imma_internal.cuh"

namespace imp {

namespace {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
// One arrival once every prior cp.async of this thread has landed (counted in the init count).
__device__ __forceinline__ void mbar_cp_async_arrive(uint64_t* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
    for (;;) {
        uint32_t done = 0;
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done)
            : "r"(smem_u32(bar)), "r"(parity)
            : "memory");
        if (done != 0) return;
    }
}

// Opaque %tid.x: per-thread load addresses are rebuilt per K-step, not held (or spilled) across the loop.
__device__ __forceinline__ int tid_x_volatile() {
    int t;
    asm volatile("mov.u32 %0, %%tid.x;\n" : "=r"(t));
    return t;
}

// IMMA C input 0x4B400000 = 1.5*2^23 as f32: for |c| < 2^22 the s32 result read as f32 is 1.5*2^23 + c.
constexpr int32_t kImmaBias = 0x4B400000;
constexpr float kImmaBiasF = 12582912.0f;

template <int BM>
struct Q8Tile {
    static constexpr int kWM = (BM == 128) ? 4 : 2;  // warp grid
    static constexpr int kWN = (BM == 128) ? 2 : 4;
    static constexpr int kTileM = BM / kWM;  // 32 / 16
    static constexpr int kTileN = kBN / kWN;  // 64 / 32
    static constexpr int kMF = kTileM / 16;  // 2 / 1
    static constexpr int kNF = kTileN / 8;   // 8 / 4
    static_assert(kWM * kWN * 32 == kThreads, "warp grid must fill the CTA");
};

struct Stage {
    int8_t (*sA)[kRow];
    int8_t (*sB)[kRow];
    __half (*sAsc)[2];
    __half (*sBsc)[kBN][2];
};

// One K-step's cp.async from CTA-uniform tile bases with 32-bit per-thread offsets.
// A/B chunk j of thread t: row (t>>2) + 64j, col (t&3)*16 (A and B share the offset).
template <int BM>
__device__ __forceinline__ void load_kstep_q8(int tid, const int8_t* __restrict__ A_tile,
                                              const __half* __restrict__ Asc_tile,
                                              const int8_t* __restrict__ B_tile,
                                              const __half* __restrict__ Bsc_tile, const Stage& st,
                                              int row_left, int n_rem, int K, int subs, int k_base) {
    constexpr int kAChunks = (BM * 4 + kThreads - 1) / kThreads;
    constexpr int kBChunks = kBN * 4 / kThreads;
    static_assert(kBN * 2 == kThreads, "one (alpha, beta) word per thread");
    const int r0 = tid >> 2;
    const int c0 = (tid & 3) * 16;
    const int ab_off = r0 * K + c0 + k_base;
#pragma unroll
    for (int j = 0; j < kAChunks; ++j) {
        const int row = r0 + (j * (kThreads / 4));
        const int off = ab_off + (j * (kThreads / 4) * K);
        if (row < BM)
            cp_async_cg_16(&st.sA[row][c0], A_tile + off, row < row_left);
    }
#pragma unroll
    for (int j = 0; j < kBChunks; ++j) {
        const int row = r0 + (j * (kThreads / 4));
        const int off = ab_off + (j * (kThreads / 4) * K);
        cp_async_cg_16(&st.sB[row][c0], B_tile + off, (n_rem < 0) || (row < n_rem));
    }
    const int kb0 = k_base / 32;
    const int asc_off = (tid * subs) + kb0;
    if (tid < BM) cp_async_ca_4(&st.sAsc[tid][0], Asc_tile + asc_off, tid < row_left);
    const int n = tid >> 1, kb = tid & 1;
    const int bsc_off = ((n * subs) + kb0 + kb) * 2;
    cp_async_ca_4(&st.sBsc[kb][n][0], Bsc_tile + bsc_off, (n_rem < 0) || (n < n_rem));
}

// Both 32-wide sub-blocks of one staged K-step into acc. kb loop rolled: unrolled spills 16-44 B at 128 regs.
template <int BM>
__device__ __forceinline__ void mma_kstep_q8(float (&acc)[Q8Tile<BM>::kMF][Q8Tile<BM>::kNF][4],
                                             const Stage& st,
                                             int warp_m, int warp_n, int rl, int cl) {
    using T = Q8Tile<BM>;
#pragma unroll 1
    for (int kb = 0; kb < 2; ++kb) {
        const int kc = kb * 32;
        uint32_t a[T::kMF][4];
        float da_lo[T::kMF], da_hi[T::kMF], nb_lo[T::kMF], nb_hi[T::kMF];
#pragma unroll
        for (int mf = 0; mf < T::kMF; ++mf) {
            const int arow_lo = warp_m * T::kTileM + mf * 16 + rl;
            const int arow_hi = arow_lo + 8;
            const int acol = kc + cl * 4;
            a[mf][0] = *reinterpret_cast<const uint32_t*>(&st.sA[arow_lo][acol]);
            a[mf][1] = *reinterpret_cast<const uint32_t*>(&st.sA[arow_hi][acol]);
            a[mf][2] = *reinterpret_cast<const uint32_t*>(&st.sA[arow_lo][acol + 16]);
            a[mf][3] = *reinterpret_cast<const uint32_t*>(&st.sA[arow_hi][acol + 16]);
            da_lo[mf] = __half2float(st.sAsc[arow_lo][kb]);
            da_hi[mf] = __half2float(st.sAsc[arow_hi][kb]);
            nb_lo[mf] = -da_lo[mf] * kImmaBiasF;  // half mantissa x 1.5*2^23: exact in f32
            nb_hi[mf] = -da_hi[mf] * kImmaBiasF;
        }
        // nf outer, mf inner: each B fragment and its alpha pair is read from smem once per nf
#pragma unroll
        for (int nf = 0; nf < T::kNF; ++nf) {
            const int bcol = warp_n * T::kTileN + nf * 8 + rl;
            const int bk = kc + cl * 4;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&st.sB[bcol][bk]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&st.sB[bcol][bk + 16]);
            // c0,c1 = rows (rl) cols (cl*2, cl*2+1); c2,c3 = rows (rl+8)
            const int ncol_lo = warp_n * T::kTileN + nf * 8 + cl * 2;
            const uint2 ab = *reinterpret_cast<const uint2*>(&st.sBsc[kb][ncol_lo][0]);
            const float al = __half2float(__low2half(*reinterpret_cast<const __half2*>(&ab.x)));
            const float ah = __half2float(__low2half(*reinterpret_cast<const __half2*>(&ab.y)));
#pragma unroll
            for (int mf = 0; mf < T::kMF; ++mf) {
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= 800
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                    : "r"(a[mf][0]), "r"(a[mf][1]), "r"(a[mf][2]), "r"(a[mf][3]), "r"(b0), "r"(b1),
                      "r"(kImmaBias));
#endif
                acc[mf][nf][0] = fmaf(al, fmaf(da_lo[mf], __int_as_float(c0), nb_lo[mf]), acc[mf][nf][0]);
                acc[mf][nf][1] = fmaf(ah, fmaf(da_lo[mf], __int_as_float(c1), nb_lo[mf]), acc[mf][nf][1]);
                acc[mf][nf][2] = fmaf(al, fmaf(da_hi[mf], __int_as_float(c2), nb_hi[mf]), acc[mf][nf][2]);
                acc[mf][nf][3] = fmaf(ah, fmaf(da_hi[mf], __int_as_float(c3), nb_hi[mf]), acc[mf][nf][3]);
            }
        }
    }
}

// Tile store, M-tail predicated, N-tail columns skipped. SPLITK: fp32 partial slice (finalize reduces).
// Otherwise FP16, += when BETA1.
template <int BM, bool BETA1, bool SPLITK>
__device__ __forceinline__ void store_tile_q8(const float (&acc)[Q8Tile<BM>::kMF][Q8Tile<BM>::kNF][4],
                                              __half* __restrict__ C, float* __restrict__ Cs, int rows, int N,
                                              int base_m, int base_n, int warp_m, int warp_n, int rl,
                                              int cl) {
    using T = Q8Tile<BM>;
#pragma unroll
    for (int mf = 0; mf < T::kMF; ++mf) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = base_m + warp_m * T::kTileM + mf * 16 + rl + h * 8;
            if (row >= rows) continue;
#pragma unroll
            for (int nf = 0; nf < T::kNF; ++nf) {
                const int col = base_n + warp_n * T::kTileN + nf * 8 + cl * 2;
                if (col + 1 >= N) continue;
                const float v0 = acc[mf][nf][2 * h], v1 = acc[mf][nf][2 * h + 1];
                if (SPLITK) {
                    Cs[static_cast<size_t>(row) * N + col] = v0;
                    Cs[static_cast<size_t>(row) * N + col + 1] = v1;
                } else {
                    __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row) * N + col]);
                    const __half2 v = __floats2half2_rn(v0, v1);
                    *p = BETA1 ? __hadd2(*p, v) : v;
                }
            }
        }
    }
}

}  // namespace

template <int BM, bool BETA1, bool SPLITK>
__global__ void __launch_bounds__(kThreads, 2)
    mmq_imma_q8_mbar_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                            const float* __restrict__ /*x_rowsum: Q8_0 has no beta term*/,
                            const int8_t* __restrict__ W_s8, const __half* __restrict__ w_sc,
                            __half* __restrict__ out, int M, int N, int K,
                            const int32_t* __restrict__ expert_offsets, size_t w_stride, size_t wsc_stride,
                            float* __restrict__ split_out, int ks_per_split) {
    using T = Q8Tile<BM>;
    static_assert(kStages == 2, "stage parity below assumes 2 stages");
    int rows = M;
    size_t row_off = 0;
    const int e = SPLITK ? 0 : blockIdx.z;  // SPLITK reuses gridDim.z for the K-split
    if (expert_offsets != nullptr) {
        const int32_t o0 = __ldg(&expert_offsets[e]);
        row_off = static_cast<size_t>(o0);
        rows = __ldg(&expert_offsets[e + 1]) - o0;
    }
    const int base_m = blockIdx.y * BM;
    const int base_n = blockIdx.x * kBN;
    const int ksteps = K / kBK;
    const int ks_begin = SPLITK ? static_cast<int>(blockIdx.z) * ks_per_split : 0;
    const int ks_end = SPLITK ? min(ksteps, ks_begin + ks_per_split) : ksteps;
    if (base_m >= rows || ks_begin >= ks_end) return;  // CTA-uniform, before any barrier
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);  // -1 = full tile
    const int subs = K / 32;
    const int row_left = rows - base_m;
    // CTA-uniform tile bases; per-thread parts stay 32-bit (128 rows x K < 2^31)
    const int8_t* A_tile = X_s8 + (row_off + base_m) * K;
    const __half* Asc_tile = x_scale + (row_off + base_m) * subs;
    const int8_t* B_tile = W_s8 + static_cast<size_t>(e) * w_stride + static_cast<size_t>(base_n) * K;
    const __half* Bsc_tile =
        w_sc + static_cast<size_t>(e) * wsc_stride + static_cast<size_t>(base_n) * subs * 2;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_m = (tid >> 5) / T::kWN;
    const int warp_n = (tid >> 5) % T::kWN;

    __shared__ int8_t sA[kStages][BM][kRow];
    __shared__ int8_t sB[kStages][kBN][kRow];
    __shared__ __half sAsc[kStages][BM][2];
    __shared__ __half sBsc[kStages][2][kBN][2];
    __shared__ uint64_t full_bar[kStages];
    __shared__ uint64_t empty_bar[kStages];
    auto stage = [&](int s) { return Stage{sA[s], sB[s], sAsc[s], sBsc[s]}; };  // no local-memory array
    if (tid == 0) {
        for (int s = 0; s < kStages; ++s) {
            mbar_init(&full_bar[s], kThreads);
            mbar_init(&empty_bar[s], kThreads / 32);
        }
    }
    __syncthreads();

    float acc[T::kMF][T::kNF][4] = {};
    load_kstep_q8<BM>(tid, A_tile, Asc_tile, B_tile, Bsc_tile, stage(0), row_left, n_rem, K, subs,
                      ks_begin * kBK);
    mbar_cp_async_arrive(&full_bar[0]);
    const int nsteps = ks_end - ks_begin;
    for (int i = 0; i < nsteps; ++i) {
        const int s = i & 1;
        mbar_wait(&full_bar[s], static_cast<uint32_t>(i >> 1) & 1u);
        if (i + 1 < nsteps) {
            // stage s^1 last held step i-1: every warp must have released it
            if (i >= 1) mbar_wait(&empty_bar[s ^ 1], static_cast<uint32_t>((i - 1) >> 1) & 1u);
            load_kstep_q8<BM>(tid_x_volatile(), A_tile, Asc_tile, B_tile, Bsc_tile, stage(s ^ 1), row_left,
                              n_rem, K, subs, (ks_begin + i + 1) * kBK);
            mbar_cp_async_arrive(&full_bar[s ^ 1]);
        }
        mma_kstep_q8<BM>(acc, stage(s), warp_m, warp_n, lane >> 2, lane & 3);
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    }

    float* Cs = SPLITK ? split_out + static_cast<size_t>(blockIdx.z) * (static_cast<size_t>(M) * N) : nullptr;
    store_tile_q8<BM, BETA1, SPLITK>(acc, out + row_off * N, Cs, rows, N, base_m, base_n, warp_m, warp_n,
                                     lane >> 2, lane & 3);
}

#define IMP_Q8_MBAR_INST(BM, BETA1, SPLITK)                                                                  \
    template __global__ void mmq_imma_q8_mbar_kernel<BM, BETA1, SPLITK>(                                    \
        const int8_t*, const __half*, const float*, const int8_t*, const __half*, __half*, int, int, int, \
        const int32_t*, size_t, size_t, float*, int);
IMP_Q8_MBAR_INST(32, false, false)
IMP_Q8_MBAR_INST(32, false, true)
IMP_Q8_MBAR_INST(128, false, false)
IMP_Q8_MBAR_INST(128, true, false)
#undef IMP_Q8_MBAR_INST

}  // namespace imp
