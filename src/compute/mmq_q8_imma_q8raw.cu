// Dense Q8_0 small-M IMMA kernel (sm_120a): BM=32, eight GGUF 34-B blocks (256 K) per step, cp.async
// staging. K % 256 == 0 makes every 272-B step of a row 16-B aligned. Launched from gemm_common (qkind 5)
// for dense uncached Q8_0 at M 2..32; the dp4a kernel read the weight once per 16 rows.

#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"

#include <utility>

namespace imp {

namespace {

constexpr int kQ8Bytes = 34;                   // fp16 d + 32 x s8
constexpr int kQ8Blocks = 8;                   // blocks per step
constexpr int kQ8Step = kQ8Blocks * kQ8Bytes;  // 272 B = 17 chunks of 16 B
constexpr int kQ8Chunks = kQ8Step / 16;
constexpr int kQ8ARow = 256 + 16;
constexpr int kQ8BRowW = kQ8Step / 4;  // 68 % 32 = 4: rl-lane word reads conflict-free
static_assert(kQ8Step % 16 == 0, "a step must be whole 16-B chunks");

constexpr size_t q8raw_stage_bytes() {
    return (size_t{32} * kQ8ARow) + (size_t{32} * kQ8Blocks * sizeof(__half)) +
           (static_cast<size_t>(kBN) * kQ8Step);
}
constexpr size_t q8raw_smem_bytes() { return kStages * q8raw_stage_bytes(); }

// One step: A tile [32][256] s8, A scales [32][8], B rows [128][272 B] of blocks 8 st .. 8 st + 7.
// Rows >= M and >= n_valid are zero-filled.
__device__ __forceinline__ void q8raw_load(int tid, const int8_t* __restrict__ A,
                                           const __half* __restrict__ Asc, const uint8_t* __restrict__ W,
                                           uint8_t* sA, __half* sAsc, uint8_t* sB, int M, int K, int base_n,
                                           int n_valid, int st) {
    const int subs = K / 32;
#pragma unroll
    for (int i = tid; i < 32 * 16; i += kThreads) {
        const int row = i >> 4;
        const int col = (i & 15) * 16;
        cp_async_cg_16(sA + static_cast<ptrdiff_t>((row * kQ8ARow) + col),
                       A + (static_cast<size_t>(row) * K) + (static_cast<size_t>(st) * 256) + col, row < M);
    }
    if (tid < 32 * 4) {
        const int row = tid >> 2;
        const int c = (tid & 3) * 2;
        cp_async_ca_4(sAsc + static_cast<ptrdiff_t>((row * kQ8Blocks) + c),
                      Asc + (static_cast<size_t>(row) * subs) + (static_cast<size_t>(st) * kQ8Blocks) + c,
                      row < M);
    }
    const size_t row_bytes = static_cast<size_t>(subs) * kQ8Bytes;
#pragma unroll
    for (int i = tid; i < kBN * kQ8Chunks; i += kThreads) {
        const int row = i / kQ8Chunks;
        const int c = i - (row * kQ8Chunks);
        const bool live = row < n_valid;
        const uint8_t* src = W + ((live ? static_cast<size_t>(base_n + row) : 0) * row_bytes) +
                             (static_cast<size_t>(st) * kQ8Step) + (static_cast<size_t>(c) * 16);
        cp_async_cg_16(sB + static_cast<ptrdiff_t>((row * kQ8Step) + (c * 16)), src, live);
    }
}

// Block bytes [off, off + 4) of a staged row; off % 4 is 0 (odd block) or 2 (even block).
template <bool kOdd>
__device__ __forceinline__ uint32_t q8raw_rd(const uint32_t* row, int off) {
    if constexpr (kOdd)
        return row[off >> 2];
    else
        return __byte_perm(row[off >> 2], row[(off >> 2) + 1], 0x5432);
}

template <int J, int kNF>
__device__ __forceinline__ void q8raw_block(float (&acc)[kNF][4], const uint8_t* sA, const __half* sAsc,
                                            const uint8_t* sB, int arow_lo, int bcol0, int ncol0, int rl,
                                            int cl) {
    const int arow_hi = arow_lo + 8;
    const int acol = (J * 32) + (cl * 4);
    const uint8_t* arl = sA + static_cast<ptrdiff_t>(arow_lo * kQ8ARow);
    const uint8_t* arh = sA + static_cast<ptrdiff_t>(arow_hi * kQ8ARow);
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(arl + acol);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(arh + acol);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(arl + acol + 16);
    const uint32_t a3 = *reinterpret_cast<const uint32_t*>(arh + acol + 16);
    const float da_lo = __half2float(sAsc[(arow_lo * kQ8Blocks) + J]);
    const float da_hi = __half2float(sAsc[(arow_hi * kQ8Blocks) + J]);
    constexpr bool kOdd = (J & 1) != 0;
    constexpr ptrdiff_t kDOff = static_cast<ptrdiff_t>(J) * (kQ8Bytes / 2);  // half index of block J's d
    const int qo = (J * kQ8Bytes) + 2 + (cl * 4);
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) {
        const auto* brow = reinterpret_cast<const uint32_t*>(
            sB + (static_cast<ptrdiff_t>(bcol0 + (nf * 8) + rl) * kQ8Step));
        const uint32_t b0 = q8raw_rd<kOdd>(brow, qo);
        const uint32_t b1 = q8raw_rd<kOdd>(brow, qo + 16);
        const int ncol = ncol0 + (nf * 8);
        const auto* d_lo = reinterpret_cast<const __half*>(sB + (static_cast<ptrdiff_t>(ncol) * kQ8Step));
        const auto* d_hi = reinterpret_cast<const __half*>(sB + (static_cast<ptrdiff_t>(ncol + 1) * kQ8Step));
        const float wl = __half2float(d_lo[kDOff]);
        const float wh = __half2float(d_hi[kDOff]);
        int32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
        acc[nf][0] = fmaf(da_lo * wl, static_cast<float>(p0), acc[nf][0]);
        acc[nf][1] = fmaf(da_lo * wh, static_cast<float>(p1), acc[nf][1]);
        acc[nf][2] = fmaf(da_hi * wl, static_cast<float>(p2), acc[nf][2]);
        acc[nf][3] = fmaf(da_hi * wh, static_cast<float>(p3), acc[nf][3]);
    }
}

template <int kNF, int... J>
__device__ __forceinline__ void q8raw_step(float (&acc)[kNF][4], const uint8_t* sA, const __half* sAsc,
                                           const uint8_t* sB, int arow_lo, int bcol0, int ncol0, int rl,
                                           int cl, std::integer_sequence<int, J...> /*blocks*/) {
    (q8raw_block<J, kNF>(acc, sA, sAsc, sB, arow_lo, bcol0, ncol0, rl, cl), ...);
}

// SPLITK: gridDim.z = K-split, fp32 partials to split_out[z][M][N], st_per_split 256-K steps each.
template <bool BETA1, bool SPLITK>
__global__ void __launch_bounds__(kThreads) mmq_imma_q8raw_kernel(
    const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale, const uint8_t* __restrict__ Wq8,
    __half* __restrict__ out, int M, int N, int K, float* __restrict__ split_out, int st_per_split) {
    constexpr int kWN = 4;
    constexpr int kTileN = kBN / kWN;
    constexpr int kNF = kTileN / 8;

    const int base_n = blockIdx.x * kBN;
    const int n_valid = min(kBN, N - base_n);
    const int steps = K / 256;
    const int st_begin = SPLITK ? static_cast<int>(blockIdx.z) * st_per_split : 0;
    const int st_end = SPLITK ? min(steps, st_begin + st_per_split) : steps;

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

    extern __shared__ __align__(16) uint8_t smem_q8raw[];
    auto sA = [&](int s) { return smem_q8raw + (static_cast<size_t>(s) * q8raw_stage_bytes()); };
    auto sAsc = [&](int s) { return reinterpret_cast<__half*>(sA(s) + (32 * kQ8ARow)); };
    auto sB = [&](int s) { return sA(s) + (32 * kQ8ARow) + (sizeof(__half) * 32 * kQ8Blocks); };

    float acc[kNF][4];
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf)
        acc[nf][0] = acc[nf][1] = acc[nf][2] = acc[nf][3] = 0.0f;

    const bool rows_live = warp_m * 16 < M;  // M <= 16: the warp_m = 1 half only loads and syncs
    // Launch guarantees st_begin < st_end.
    q8raw_load(tid, X_s8, x_scale, Wq8, sA(0), sAsc(0), sB(0), M, K, base_n, n_valid, st_begin);
    cp_async_commit();
    for (int st = st_begin; st < st_end; ++st) {
        const int stage = (st - st_begin) & 1;
        if (st + 1 < st_end) {
            const int ns = stage ^ 1;
            q8raw_load(tid, X_s8, x_scale, Wq8, sA(ns), sAsc(ns), sB(ns), M, K, base_n, n_valid, st + 1);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();
        if (rows_live)
            q8raw_step<kNF>(acc, sA(stage), sAsc(stage), sB(stage), arow_lo, bcol0, ncol0, rl, cl,
                            std::make_integer_sequence<int, kQ8Blocks>{});
        __syncthreads();
    }

    imma_small_m_store<BETA1, SPLITK, kNF>(acc, out,
                                           split_out + (SPLITK ? static_cast<size_t>(blockIdx.z) * M * N : 0),
                                           arow_lo, base_n + ncol0, M, N);
}

}  // namespace

bool launch_q8_raw(const uint8_t* w8, __half* out_f16, int M, int N, int K, float beta, bool capturing,
                   cudaStream_t stream) {
    using Q8Raw = void (*)(const int8_t*, const __half*, const uint8_t*, __half*, int, int, int, float*, int);
    const Q8Raw k0 = mmq_imma_q8raw_kernel<false, false>;
    const Q8Raw k1 = mmq_imma_q8raw_kernel<true, false>;
    const Q8Raw ks = mmq_imma_q8raw_kernel<false, true>;
    static bool smem_ok = false;
    if (!smem_ok)
        smem_ok = imma_smem_opt_in(reinterpret_cast<const void*>(k0), q8raw_smem_bytes()) &&
                  imma_smem_opt_in(reinterpret_cast<const void*>(k1), q8raw_smem_bytes()) &&
                  imma_smem_opt_in(reinterpret_cast<const void*>(ks), q8raw_smem_bytes());
    if (!smem_ok)
        return false;
    const bool beta1 = beta == 1.0f;
    const int n_tiles = (N + kBN - 1) / kBN;
    const int steps = K / 256;
    const int s = imma_dense_splits(N, steps);
    const int per = (steps + s - 1) / s;
    const int used = (steps + per - 1) / per;
    const size_t slice = static_cast<size_t>(M) * N;
    if (used > 1 && imma_ensure_splitk(slice * used, capturing)) {
        ks<<<dim3(n_tiles, 1, used), kThreads, q8raw_smem_bytes(), stream>>>(g_imma_act.xs8,
                                                                             g_imma_act.xscale, w8, out_f16,
                                                                             M, N, K, g_imma_splitk.buf, per);
        IMP_CUDA_CHECK_LAUNCH();
        imma_splitk_finalize(used, static_cast<int>(slice), out_f16, beta1, stream);
        return true;
    }
    (beta1 ? k1 : k0)<<<dim3(n_tiles, 1, 1), kThreads, q8raw_smem_bytes(), stream>>>(g_imma_act.xs8,
                                                                                     g_imma_act.xscale, w8,
                                                                                     out_f16, M, N, K,
                                                                                     nullptr, 0);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace imp
