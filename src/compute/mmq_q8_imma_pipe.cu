// BM=128 Q8_0 IMMA prefill kernel (#2267): kQ8Stages-deep cp.async ring, 1 barrier per K-step,
// ldmatrix.x4 fragments from 64-B rows with 16-B chunk c stored at c ^ ((row >> 1) & 3): 8 rows = 8 bank groups.
// Per output: same kb order and `acc += (d_a * alpha) * float(c)` as mmq_imma_kernel: bit-identical.

#include "compute/mmq_q8_imma.h"
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"

#include <cstddef>

namespace imp {

namespace {

constexpr int kQ8BM = 128;
constexpr int kQ8Stages = 4;
constexpr int kTileM = 32, kTileN = 64;  // 4 x 2 warps
constexpr int kMF = kTileM / 16;         // 2
constexpr int kNF = kTileN / 8;          // 8
static_assert(4 * 2 * 32 == kThreads && kQ8BM == 4 * kTileM && kBN == 2 * kTileN, "warp grid");
static_assert(kQ8Stages >= 2 && kThreads == 2 * kQ8BM && kThreads == 2 * kBN, "load split");

struct Q8PipeStage {
    int8_t A[kQ8BM * kBK];
    int8_t B[kBN * kBK];
    __half2 Asc[kQ8BM];                 // (d kb0, d kb1) per row
    alignas(16) __half Bsc[kBN][2][2];  // [n][kb](alpha, beta)
};
constexpr size_t kQ8PipeSmem = kQ8Stages * sizeof(Q8PipeStage);

using Acc = float[kMF][kNF][4];

__device__ __forceinline__ int q8_swz(int row, int chunk) {
    return row * kBK + ((chunk ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t saddr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(saddr));
}

// One K-step into st: A/B 2 x 16 B per thread; threads 0-127 Asc (4 B), 128-255 Bsc (8 B, kb0 + kb1).
__device__ __forceinline__ void q8_pipe_load(Q8PipeStage& st, int tid, const int8_t* __restrict__ A,
                                             const __half* __restrict__ Asc, const int8_t* __restrict__ B,
                                             const __half* __restrict__ Bsc, int base_m, int rows, int K,
                                             int subs, int k_base, int n_rem) {
#pragma unroll
    for (int i = tid; i < kQ8BM * 4; i += kThreads) {
        const int row = i >> 2, c = i & 3;
        cp_async_cg_16(&st.A[q8_swz(row, c)], A + static_cast<size_t>(base_m + row) * K + k_base + c * 16,
                       (base_m + row) < rows);
    }
#pragma unroll
    for (int i = tid; i < kBN * 4; i += kThreads) {
        const int row = i >> 2, c = i & 3;
        cp_async_cg_16(&st.B[q8_swz(row, c)], B + static_cast<size_t>(row) * K + k_base + c * 16,
                       (n_rem < 0) || (row < n_rem));
    }
    const int kb0 = k_base / 32;
    if (tid < kQ8BM) {
        cp_async_ca_4(&st.Asc[tid], Asc + static_cast<size_t>(base_m + tid) * subs + kb0, (base_m + tid) < rows);
    } else {
        const int n = tid - kQ8BM;
        cp_async_ca_8(&st.Bsc[n][0][0], Bsc + (static_cast<size_t>(n) * subs + kb0) * 2,
                      (n_rem < 0) || (n < n_rem));
    }
}

// Per-lane ldmatrix offsets (stage-relative). Row bases are multiples of 8: swizzle key (lane >> 1) & 3.
// A x4 (mf, kb): lanes 0-7 rows 0-7 chunk 2kb, 8-15 rows 8-15 chunk 2kb, 16-23 / 24-31 chunk 2kb+1.
// B x4 (nf): lanes 8j..8j+7 rows 0-7 chunk j -> regs (b0, b1) of kb0, (b0, b1) of kb1.
struct Q8Lane {
    int warp_m, warp_n, rl, cl, swz, a_hi;
    uint32_t a_off, b_off;
};

__device__ __forceinline__ Q8Lane q8_lane(int tid) {
    Q8Lane L;
    const int warp_id = tid >> 5, lane = tid & 31;
    L.warp_m = warp_id / 2;
    L.warp_n = warp_id % 2;
    L.rl = lane >> 2;
    L.cl = lane & 3;
    L.swz = (lane >> 1) & 3;
    L.a_hi = lane >> 4;
    L.a_off = static_cast<uint32_t>(offsetof(Q8PipeStage, A) +
                                    (L.warp_m * kTileM + (lane & 7) + ((lane >> 3) & 1) * 8) * kBK);
    L.b_off = static_cast<uint32_t>(offsetof(Q8PipeStage, B) + (L.warp_n * kTileN + (lane & 7)) * kBK +
                                    (((lane >> 3) ^ L.swz) << 4));
    return L;
}

// One K-step of MMAs on stage st (shared address sbase). Per output: kb0 then kb1.
__device__ __forceinline__ void q8_pipe_mma(Acc& acc, const Q8PipeStage& st, uint32_t sbase, const Q8Lane& L) {
    uint32_t a[2][kMF][4];
    float da_lo[2][kMF], da_hi[2][kMF];
#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
#pragma unroll
        for (int kb = 0; kb < 2; ++kb)
            ldmatrix_x4(a[kb][mf], sbase + L.a_off + mf * 16 * kBK + (((kb * 2 + L.a_hi) ^ L.swz) << 4));
        const int arow_lo = L.warp_m * kTileM + mf * 16 + L.rl;
        const __half2 s_lo = st.Asc[arow_lo];
        const __half2 s_hi = st.Asc[arow_lo + 8];
        da_lo[0][mf] = __half2float(__low2half(s_lo));
        da_lo[1][mf] = __half2float(__high2half(s_lo));
        da_hi[0][mf] = __half2float(__low2half(s_hi));
        da_hi[1][mf] = __half2float(__high2half(s_hi));
    }
#pragma unroll
    for (int nf = 0; nf < kNF; ++nf) {
        uint32_t b[4];
        ldmatrix_x4(b, sbase + L.b_off + nf * 8 * kBK);
        // columns (ncol, ncol + 1) x kb 0/1: one 16-B word [n][kb](alpha, beta)
        const int ncol_lo = L.warp_n * kTileN + nf * 8 + L.cl * 2;
        const uint4 ab = *reinterpret_cast<const uint4*>(&st.Bsc[ncol_lo][0][0]);
        const float al[2] = {__half2float(__low2half(*reinterpret_cast<const __half2*>(&ab.x))),
                             __half2float(__low2half(*reinterpret_cast<const __half2*>(&ab.y)))};
        const float ah[2] = {__half2float(__low2half(*reinterpret_cast<const __half2*>(&ab.z))),
                             __half2float(__low2half(*reinterpret_cast<const __half2*>(&ab.w)))};
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#if __CUDA_ARCH__ >= 800
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                    : "r"(a[kb][mf][0]), "r"(a[kb][mf][1]), "r"(a[kb][mf][2]), "r"(a[kb][mf][3]),
                      "r"(b[kb * 2]), "r"(b[kb * 2 + 1]), "r"(0), "r"(0), "r"(0), "r"(0));
#endif
                acc[mf][nf][0] += (da_lo[kb][mf] * al[kb]) * static_cast<float>(c0);
                acc[mf][nf][1] += (da_lo[kb][mf] * ah[kb]) * static_cast<float>(c1);
                acc[mf][nf][2] += (da_hi[kb][mf] * al[kb]) * static_cast<float>(c2);
                acc[mf][nf][3] += (da_hi[kb][mf] * ah[kb]) * static_cast<float>(c3);
            }
        }
    }
}

// FP16 store, M-tail predicated, N-tail columns never stored.
template <bool BETA1>
__device__ __forceinline__ void q8_pipe_store(const Acc& acc, __half* __restrict__ C, int N, int rows, int base_m,
                                              int base_n, const Q8Lane& L) {
#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
        const int row_lo = base_m + L.warp_m * kTileM + mf * 16 + L.rl;
#pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
            const int col = base_n + L.warp_n * kTileN + nf * 8 + L.cl * 2;
            if (col + 1 >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = row_lo + h * 8;
                if (row >= rows) continue;
                __half2* p = reinterpret_cast<__half2*>(&C[static_cast<size_t>(row) * N + col]);
                const __half2 v = __floats2half2_rn(acc[mf][nf][2 * h], acc[mf][nf][2 * h + 1]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
        }
    }
}

template <bool BETA1>
__global__ void __launch_bounds__(kThreads, 1)
    mmq_q8_pipe_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                       const int8_t* __restrict__ W_s8, const __half* __restrict__ w_sc, __half* __restrict__ out,
                       int M, int N, int K, const int32_t* __restrict__ expert_offsets, size_t w_stride,
                       size_t wsc_stride) {
    int rows = M;
    size_t row_off = 0;
    const int e = blockIdx.z;
    if (expert_offsets != nullptr) {
        const int32_t o0 = __ldg(&expert_offsets[e]);
        row_off = static_cast<size_t>(o0);
        rows = __ldg(&expert_offsets[e + 1]) - o0;
    }
    const int base_m = blockIdx.y * kQ8BM;
    const int base_n = blockIdx.x * kBN;
    if (base_m >= rows) return;  // also rows == 0
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);  // -1 = full tile

    const int subs = K / 32;
    const int8_t* A = X_s8 + row_off * K;
    const __half* Asc = x_scale + row_off * subs;
    const int8_t* B = W_s8 + static_cast<size_t>(e) * w_stride + static_cast<size_t>(base_n) * K;
    const __half* Bsc = w_sc + static_cast<size_t>(e) * wsc_stride + static_cast<size_t>(base_n) * subs * 2;

    extern __shared__ __align__(16) unsigned char q8_pipe_smem[];
    Q8PipeStage* stg = reinterpret_cast<Q8PipeStage*>(q8_pipe_smem);
    const uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(q8_pipe_smem));
    const int tid = threadIdx.x;
    const Q8Lane L = q8_lane(tid);

    Acc acc;
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int nk = K / kBK;
#pragma unroll
    for (int s = 0; s < kQ8Stages - 1; ++s) {
        if (s < nk)
            q8_pipe_load(stg[s], tid, A, Asc, B, Bsc, base_m, rows, K, subs, s * kBK, n_rem);
        cp_async_commit();
    }
    for (int ks = 0; ks < nk; ++ks) {
        cp_async_wait_group<kQ8Stages - 2>();
        __syncthreads();  // stage ks visible; stage (ks - 1) % S drained by every warp
        const int nxt = ks + kQ8Stages - 1;
        if (nxt < nk)
            q8_pipe_load(stg[nxt % kQ8Stages], tid, A, Asc, B, Bsc, base_m, rows, K, subs, nxt * kBK, n_rem);
        cp_async_commit();
        const int stage = ks % kQ8Stages;
        q8_pipe_mma(acc, stg[stage], smem_base + static_cast<uint32_t>(stage * sizeof(Q8PipeStage)), L);
    }
    q8_pipe_store<BETA1>(acc, out + row_off * N, N, rows, base_m, base_n, L);
}

}  // namespace

void launch_q8_pipe(bool beta1, dim3 grid, cudaStream_t stream, const int8_t* xs8, const __half* xscale,
                    const int8_t* ws8, const __half* wsc, __half* out, int M, int N, int K,
                    const int32_t* d_offsets, size_t w_stride, size_t wsc_stride) {
    static const bool attr_set = [] {
        cudaFuncSetAttribute(mmq_q8_pipe_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(kQ8PipeSmem));
        cudaFuncSetAttribute(mmq_q8_pipe_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(kQ8PipeSmem));
        return true;
    }();
    (void)attr_set;
    auto* k = beta1 ? mmq_q8_pipe_kernel<true> : mmq_q8_pipe_kernel<false>;
    k<<<grid, kThreads, kQ8PipeSmem, stream>>>(xs8, xscale, ws8, wsc, out, M, N, K, d_offsets, w_stride,
                                               wsc_stride);
    IMP_CUDA_CHECK_LAUNCH();
}

bool mmq_q8_imma_plane128(const int8_t* xs8, const __half* xscale, const int8_t* ws8, const __half* wsc,
                          __half* out, int M, int N, int K, bool beta1, cudaStream_t stream) {
    if (M < 1 || N % 2 != 0 || K % kBK != 0) return false;
    const dim3 grid((N + kBN - 1) / kBN, (M + kQ8BM - 1) / kQ8BM, 1);
    launch_q8_pipe(beta1, grid, stream, xs8, xscale, ws8, wsc, out, M, N, K, nullptr, 0, 0);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace imp
