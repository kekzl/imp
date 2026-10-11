// Q8_0 dense IMMA prefill on the cached planes with a 128-K step (4 sub-blocks per barrier pair) and
// ldmatrix fragments, the Q4_K levers of #2622/#2623 (#2416). Per output the k order and the per-sub-block
// FMUL+FFMA match mmq_imma_kernel<BM, BETA1, false>: rows are bit-identical to it (#2152).
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"

namespace imp {

namespace {

constexpr int kK8Sub = kQ4kSub;  // 4 sub-blocks of 32 per K-step
constexpr int kK8BK = kQ4kBK;    // 128
constexpr int kK8Row = kQ4kRow;  // 144-B stride: ldmatrix rows conflict-free

constexpr size_t q8k128_smem_bytes(int BM) {
    return static_cast<size_t>(kStages) *
           (static_cast<size_t>(BM + kBN) * kK8Row + static_cast<size_t>(kBN) * kK8Sub * 2 * sizeof(__half) +
            static_cast<size_t>(BM) * kK8Sub * sizeof(__half));
}

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                        const void* smem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(s));
}

// A[BM][128] s8 (M-tail zero-filled), B[kBN][128] s8, Asc[BM][4] half, Bsc[kBN][4][2] half (alpha, beta).
template <int BM>
__device__ __forceinline__ void load_kstep_q8k128(
    int tid, const int8_t* __restrict__ A, const __half* __restrict__ Asc, const int8_t* __restrict__ B,
    const __half* __restrict__ Bsc, int8_t (*sA)[kK8Row], int8_t (*sB)[kK8Row], __half (*sAsc)[kK8Sub],
    __half (*sBsc)[kK8Sub][2], int base_m, int M, int K, int subs, int k_base, int base_n_rows) {
#pragma unroll
    for (int i = tid; i < BM * (kK8BK / 16); i += kThreads) {
        const int row = i >> 3;
        const int col = (i & 7) * 16;
        cp_async_cg_16(&sA[row][col], A + static_cast<size_t>(base_m + row) * K + k_base + col,
                       (base_m + row) < M);
    }
#pragma unroll
    for (int i = tid; i < kBN * (kK8BK / 16); i += kThreads) {
        const int row = i >> 3;
        const int col = (i & 7) * 16;
        cp_async_cg_16(&sB[row][col], B + static_cast<size_t>(row) * K + k_base + col,
                       (base_n_rows < 0) || (row < base_n_rows));
    }
    const int kb0 = k_base / 32;
#pragma unroll
    for (int i = tid; i < BM; i += kThreads)
        cp_async_ca_8(&sAsc[i][0], Asc + static_cast<size_t>(base_m + i) * subs + kb0, (base_m + i) < M);
#pragma unroll
    for (int i = tid; i < kBN; i += kThreads)
        cp_async_cg_16(&sBsc[i][0][0], Bsc + (static_cast<size_t>(i) * subs + kb0) * 2,
                       (base_n_rows < 0) || (i < base_n_rows));
}

// Fragment layout of a thread's accumulators: rows row0 + mf * 16 (+8), cols col0 + nf * 8 (+1).
template <int MF, int NF>
__device__ __forceinline__ void load_residual(__half2 (&cres)[MF][NF][2], const __half* __restrict__ out,
                                              int row0, int col0, int M, int N) {
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        const int r = row0 + mf * 16;
#pragma unroll
        for (int nf = 0; nf < NF; ++nf) {
            const int col = col0 + nf * 8;
            const bool cok = col + 1 < N;
            cres[mf][nf][0] = cok && r < M
                                  ? *reinterpret_cast<const __half2*>(&out[static_cast<size_t>(r) * N + col])
                                  : __half2{};
            cres[mf][nf][1] = cok && r + 8 < M ? *reinterpret_cast<const __half2*>(
                                                     &out[static_cast<size_t>(r + 8) * N + col])
                                               : __half2{};
        }
    }
}

// FP16 store, M-tail predicated (N is a multiple of kBN); beta 1 adds the prefetched residual.
template <int MF, int NF, bool BETA1>
__device__ __forceinline__ void store_tile(const float (&acc)[MF][NF][4], const __half2 (&cres)[MF][NF][2],
                                           __half* __restrict__ out, int row0, int col0, int M, int N) {
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
#pragma unroll
        for (int nf = 0; nf < NF; ++nf) {
            const int col = col0 + nf * 8;
            if (col + 1 >= N)
                continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = row0 + mf * 16 + h * 8;
                if (r >= M)
                    continue;
                const __half2 v = __floats2half2_rn(acc[mf][nf][2 * h], acc[mf][nf][2 * h + 1]);
                *reinterpret_cast<__half2*>(
                    &out[static_cast<size_t>(r) * N + col]) = BETA1 ? __hadd2(cres[mf][nf][h], v) : v;
            }
        }
    }
}

template <int BM, bool BETA1>
__global__ void __launch_bounds__(kThreads) mmq_imma_q8k128_kernel(
    const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale, const int8_t* __restrict__ W_s8,
    const __half* __restrict__ w_sc, __half* __restrict__ out, int M, int N, int K) {
    constexpr int kWM = (BM == 128) ? 4 : 2;
    constexpr int kWN = (BM == 128) ? 2 : 4;
    constexpr int kTileM = BM / kWM;
    constexpr int kTileN = kBN / kWN;
    constexpr int kMF = kTileM / 16;
    constexpr int kNF = kTileN / 8;
    static_assert(kNF % 2 == 0, "B fragments load in nf pairs");

    const int base_m = blockIdx.y * BM;
    const int base_n = blockIdx.x * kBN;
    if (base_m >= M)
        return;
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);

    const int subs = K / 32;
    const int8_t* B = W_s8 + static_cast<size_t>(base_n) * K;
    const __half* Bsc = w_sc + static_cast<size_t>(base_n) * subs * 2;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id / kWN;
    const int warp_n = warp_id % kWN;
    const int rl = lane >> 2;
    const int cl = lane & 3;
    const int lm = lane >> 3;  // ldmatrix matrix index of this lane's row address

    extern __shared__ __align__(16) uint8_t smem_raw[];
    size_t off = 0;
    auto* sA = reinterpret_cast<int8_t (*)[BM][kK8Row]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * BM * kK8Row;
    auto* sB = reinterpret_cast<int8_t (*)[kBN][kK8Row]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * kBN * kK8Row;
    auto* sBsc = reinterpret_cast<__half(*)[kBN][kK8Sub][2]>(smem_raw + off);
    off += static_cast<size_t>(kStages) * kBN * kK8Sub * 2 * sizeof(__half);
    auto* sAsc = reinterpret_cast<__half(*)[BM][kK8Sub]>(smem_raw + off);

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    // This thread's fragment origin in the output: rows row0 + mf * 16 (+8), cols col0 + nf * 8 (+1).
    const int row0 = base_m + warp_m * kTileM + rl;
    const int col0 = base_n + warp_n * kTileN + cl * 2;
    // beta 1: the residual tile is read up front, its latency hidden under the K loop (same __hadd2 below).
    __half2 cres[kMF][kNF][2];
    if constexpr (BETA1)
        load_residual<kMF, kNF>(cres, out, row0, col0, M, N);

    const int ksteps = K / kK8BK;
    load_kstep_q8k128<BM>(tid, X_s8, x_scale, B, Bsc, sA[0], sB[0], sAsc[0], sBsc[0], base_m, M, K, subs, 0,
                          n_rem);
    cp_async_commit();

    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int nstage = (ks + 1) & 1;
            load_kstep_q8k128<BM>(tid, X_s8, x_scale, B, Bsc, sA[nstage], sB[nstage], sAsc[nstage],
                                  sBsc[nstage], base_m, M, K, subs, (ks + 1) * kK8BK, n_rem);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < kK8Sub; ++kb) {
            uint32_t a[kMF][4];
            float da_lo[kMF], da_hi[kMF];
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int row = warp_m * kTileM + mf * 16 + (lane & 7) + (lm & 1) * 8;
                ldsm_x4(a[mf][0], a[mf][1], a[mf][2], a[mf][3], &sA[stage][row][kb * 32 + (lm >> 1) * 16]);
                const int arow_lo = warp_m * kTileM + mf * 16 + rl;
                da_lo[mf] = __half2float(sAsc[stage][arow_lo][kb]);
                da_hi[mf] = __half2float(sAsc[stage][arow_lo + 8][kb]);
            }
#pragma unroll
            for (int nf = 0; nf < kNF; nf += 2) {
                uint32_t b[2][2];
                {
                    const int col = warp_n * kTileN + nf * 8 + (lane & 7) + (lm >> 1) * 8;
                    ldsm_x4(b[0][0], b[0][1], b[1][0], b[1][1], &sB[stage][col][kb * 32 + (lm & 1) * 16]);
                }
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    // c0,c1 = rows (rl) cols (cl*2, cl*2+1); c2,c3 = rows (rl+8)
                    const int ncol_lo = warp_n * kTileN + (nf + p) * 8 + cl * 2;
                    const float al = __half2float(sBsc[stage][ncol_lo][kb][0]);
                    const float ah = __half2float(sBsc[stage][ncol_lo + 1][kb][0]);
#pragma unroll
                    for (int mf = 0; mf < kMF; ++mf) {
                        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        asm volatile(
                            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                            : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                            : "r"(a[mf][0]), "r"(a[mf][1]), "r"(a[mf][2]), "r"(a[mf][3]), "r"(b[p][0]),
                              "r"(b[p][1]), "r"(0), "r"(0), "r"(0), "r"(0));
                        acc[mf][nf + p][0] += (da_lo[mf] * al) * static_cast<float>(c0);
                        acc[mf][nf + p][1] += (da_lo[mf] * ah) * static_cast<float>(c1);
                        acc[mf][nf + p][2] += (da_hi[mf] * al) * static_cast<float>(c2);
                        acc[mf][nf + p][3] += (da_hi[mf] * ah) * static_cast<float>(c3);
                    }
                }
            }
        }
        __syncthreads();
    }

    store_tile<kMF, kNF, BETA1>(acc, cres, out, row0, col0, M, N);
}

template <typename Kernel>
bool k128_smem_opt_in(Kernel k, size_t bytes) {
    const cudaError_t err = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(bytes));
    if (err != cudaSuccess)
        IMP_LOG_ERROR("mmq imma q8 k128: smem opt-in %zu B failed: %s", bytes, cudaGetErrorString(err));
    return err == cudaSuccess;
}

}  // namespace

bool launch_q8_k128(const int8_t* xs8, const __half* xscale, const int8_t* w_qs, const __half* w_sc,
                    __half* out, int M, int N, int K, int bm, bool beta1, dim3 grid, cudaStream_t stream) {
    if (K % kK8BK != 0 || (bm != 32 && bm != 128))
        return false;
    static const bool smem_ok = k128_smem_opt_in(mmq_imma_q8k128_kernel<32, false>, q8k128_smem_bytes(32)) &&
                                k128_smem_opt_in(mmq_imma_q8k128_kernel<32, true>, q8k128_smem_bytes(32)) &&
                                k128_smem_opt_in(mmq_imma_q8k128_kernel<128, false>,
                                                 q8k128_smem_bytes(128)) &&
                                k128_smem_opt_in(mmq_imma_q8k128_kernel<128, true>, q8k128_smem_bytes(128));
    if (!smem_ok)
        return false;
    if (bm == 32) {
        const auto k32 = beta1 ? mmq_imma_q8k128_kernel<32, true> : mmq_imma_q8k128_kernel<32, false>;
        k32<<<grid, kThreads, q8k128_smem_bytes(32), stream>>>(xs8, xscale, w_qs, w_sc, out, M, N, K);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (beta1) {
        mmq_imma_q8k128_kernel<128, true>
            <<<grid, kThreads, q8k128_smem_bytes(128), stream>>>(xs8, xscale, w_qs, w_sc, out, M, N, K);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        mmq_imma_q8k128_kernel<128, false>
            <<<grid, kThreads, q8k128_smem_bytes(128), stream>>>(xs8, xscale, w_qs, w_sc, out, M, N, K);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}

}  // namespace imp
