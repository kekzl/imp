// =============================================================================
// mmq_q8_imma.cu — Q8_0 INT8 IMMA prefill GEMM (sm_120a)
// =============================================================================
//
// See mmq_q8_imma.h for the design rationale (phase-2B ceiling fixes: SMEM-
// staged scales, 128×128×64 tiles, symmetric epilogue). Reuses the activation
// quantizer from the Q4_K IMMA stack (quantize_fp16_to_int8_subblock).

#include "compute/mmq_q8_imma.h"
#include "core/logging.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace imp {

namespace {

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 64;  // 2 sub-blocks per K-step
constexpr int kPad = 16;  // smem row pad (bytes): 80-B stride = conflict-free rl-lane access
constexpr int kRow = kBK + kPad;
constexpr int kStages = 2;
constexpr int kWarpsM = 4;
constexpr int kWarpsN = 2;
constexpr int kThreads = kWarpsM * kWarpsN * 32;  // 256
constexpr int kWarpTileM = kBM / kWarpsM;          // 32 → 2 m16 frags
constexpr int kWarpTileN = kBN / kWarpsN;          // 64 → 8 n8 frags
constexpr int kMF = kWarpTileM / 16;               // 2
constexpr int kNF = kWarpTileN / 8;                // 8

__device__ __forceinline__ void cp_async_cg_16(void* smem, const void* glob, bool valid) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    int src_size = valid ? 16 : 0;  // src-size 0 → zero-fill (OOB M-tail rows)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(glob),
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

// One K-step tile load, all 256 threads cooperating:
//   A  [kBM][kBK] s8 = 8192 B → 32 B/thread (2 × cp16), M-tail rows zero-filled
//   B  [kBN][kBK] s8 = 8192 B → 32 B/thread (2 × cp16)
//   Ad [kBM][2] half = 512 B → first 128 threads, 4 B each (d-plane cols
//      (kb0, kb0+1) are contiguous and 4-B aligned: kb0 is even)
//   Bd [kBN][2] half = 512 B → next 128 threads
__device__ __forceinline__ void load_kstep(int tid, const int8_t* __restrict__ A,
                                           const int8_t* __restrict__ Ad_src,
                                           const int8_t* __restrict__ B,
                                           const int8_t* __restrict__ Bd_src,
                                           int8_t (*sA)[kRow], int8_t (*sB)[kRow],
                                           __half (*sAd)[2], __half (*sBd)[2], int base_m, int M,
                                           int K, int subs, int k_base) {
    {
        // A: row = tid/2 in 0..127, col = (tid&1)*32 + {0,16}
        const int row = tid >> 1;
        const int col = (tid & 1) * 32;
        const bool valid = (base_m + row) < M;
        const int8_t* src = A + static_cast<size_t>(base_m + row) * K + k_base + col;
        cp_async_cg_16(&sA[row][col], src, valid);
        cp_async_cg_16(&sA[row][col + 16], src + 16, valid);
    }
    {
        // B: weight rows are always full (N % kBN == 0 gate)
        const int row = tid >> 1;
        const int col = (tid & 1) * 32;
        const int8_t* src = B + static_cast<size_t>(row) * K + k_base + col;  // B pre-offset to base_n
        cp_async_cg_16(&sB[row][col], src, true);
        cp_async_cg_16(&sB[row][col + 16], src + 16, true);
    }
    {
        const int kb0 = k_base / 32;
        if (tid < kBM) {
            const bool valid = (base_m + tid) < M;
            const __half* src = reinterpret_cast<const __half*>(Ad_src) +
                                static_cast<size_t>(base_m + tid) * subs + kb0;
            cp_async_ca_4(&sAd[tid][0], src, valid);
        } else if (tid < kBM + kBN) {
            const int row = tid - kBM;
            const __half* src =
                reinterpret_cast<const __half*>(Bd_src) + static_cast<size_t>(row) * subs + kb0;
            cp_async_ca_4(&sBd[row][0], src, true);
        }
    }
}

// out[M,N] = (or +=, BETA1) per-block-scaled IMMA over [kBM,kBN] tiles.
template <bool BETA1>
__global__ void __launch_bounds__(kThreads)
    mmq_q8_imma_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                       const int8_t* __restrict__ W_s8, const __half* __restrict__ w_d,
                       __half* __restrict__ out, int M, int N, int K) {
    const int n_block = blockIdx.x;
    const int m_block = blockIdx.y;
    const int base_m = m_block * kBM;
    const int base_n = n_block * kBN;
    if (base_m >= M) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp_id >> 1;  // 0..3
    const int warp_n = warp_id & 1;   // 0..1
    const int rl = lane >> 2;         // 0..7
    const int cl = lane & 3;          // 0..3

    __shared__ int8_t sA[kStages][kBM][kRow];
    __shared__ int8_t sB[kStages][kBN][kRow];
    __shared__ __half sAd[kStages][kBM][2];
    __shared__ __half sBd[kStages][kBN][2];

    float acc[kMF][kNF][4];
#pragma unroll
    for (int i = 0; i < kMF; ++i)
#pragma unroll
        for (int j = 0; j < kNF; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int subs = K / 32;
    const int ksteps = K / kBK;
    const int8_t* B_base = W_s8 + static_cast<size_t>(base_n) * K;
    const int8_t* Bd_base = reinterpret_cast<const int8_t*>(w_d + static_cast<size_t>(base_n) * subs);
    const int8_t* Ad_base = reinterpret_cast<const int8_t*>(x_scale);

    load_kstep(tid, X_s8, Ad_base, B_base, Bd_base, sA[0], sB[0], sAd[0], sBd[0], base_m, M, K,
               subs, 0);
    cp_async_commit();

    for (int ks = 0; ks < ksteps; ++ks) {
        const int stage = ks & 1;
        if (ks + 1 < ksteps) {
            const int nstage = (ks + 1) & 1;
            load_kstep(tid, X_s8, Ad_base, B_base, Bd_base, sA[nstage], sB[nstage], sAd[nstage],
                       sBd[nstage], base_m, M, K, subs, (ks + 1) * kBK);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {  // 2 sub-blocks per K-step
            const int kc = kb * 32;
#pragma unroll
            for (int mf = 0; mf < kMF; ++mf) {
                const int arow_lo = warp_m * kWarpTileM + mf * 16 + rl;
                const int arow_hi = arow_lo + 8;
                const int acol = kc + cl * 4;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol]);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol]);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_lo][acol + 16]);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[stage][arow_hi][acol + 16]);
                // d_a for this fragment's two rows, hoisted over all 8 n-frags
                const float da_lo = __half2float(sAd[stage][arow_lo][kb]);
                const float da_hi = __half2float(sAd[stage][arow_hi][kb]);

#pragma unroll
                for (int nf = 0; nf < kNF; ++nf) {
                    const int bcol = warp_n * kWarpTileN + nf * 8 + rl;
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
                    const int ncol_lo = warp_n * kWarpTileN + nf * 8 + cl * 2;
                    const float dw_lo = __half2float(sBd[stage][ncol_lo][kb]);
                    const float dw_hi = __half2float(sBd[stage][ncol_lo + 1][kb]);
                    acc[mf][nf][0] += (da_lo * dw_lo) * static_cast<float>(c0);
                    acc[mf][nf][1] += (da_lo * dw_hi) * static_cast<float>(c1);
                    acc[mf][nf][2] += (da_hi * dw_lo) * static_cast<float>(c2);
                    acc[mf][nf][3] += (da_hi * dw_hi) * static_cast<float>(c3);
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: FP16 store, M-tail predicated (N is a multiple of kBN).
#pragma unroll
    for (int mf = 0; mf < kMF; ++mf) {
        const int row_lo = base_m + warp_m * kWarpTileM + mf * 16 + rl;
        const int row_hi = row_lo + 8;
#pragma unroll
        for (int nf = 0; nf < kNF; ++nf) {
            const int col = base_n + warp_n * kWarpTileN + nf * 8 + cl * 2;
            if (row_lo < M) {
                __half2* p = reinterpret_cast<__half2*>(&out[static_cast<size_t>(row_lo) * N + col]);
                __half2 v = __floats2half2_rn(acc[mf][nf][0], acc[mf][nf][1]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
            if (row_hi < M) {
                __half2* p = reinterpret_cast<__half2*>(&out[static_cast<size_t>(row_hi) * N + col]);
                __half2 v = __floats2half2_rn(acc[mf][nf][2], acc[mf][nf][3]);
                *p = BETA1 ? __hadd2(*p, v) : v;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Q8_0 SoA split: raw 34-B blocks {half d; int8 qs[32]} (2-aligned! memcpy
// only) → qs plane [N][K] s8 + d plane [N][K/32] half. One-time per weight.
// -----------------------------------------------------------------------------
__global__ void q8_split_kernel(const uint8_t* __restrict__ src, int8_t* __restrict__ qs_plane,
                                __half* __restrict__ d_plane, int n_blocks_total, int subs) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks_total) return;
    const uint8_t* blk = src + static_cast<size_t>(b) * 34;
    __half d;
    memcpy(&d, blk, 2);
    d_plane[b] = d;
    int8_t* dst = qs_plane + static_cast<size_t>(b) * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[i] = static_cast<int8_t>(blk[2 + i]);
    (void)subs;
}

// Faster activation quantizer than the shared 32-thread-block version
// (quantize_fp16_to_int8_subblock): same math, but 8 warps per block with a
// grid-stride loop over (m, sub) pairs — the shared kernel's 65k 32-thread
// blocks cap SM occupancy and ran at ~150 GB/s (32.5 µs per 512×4096 GEMM,
// nsys 2026-06-07); this version is plain-coalesced and occupancy-bound.
__global__ void quantize_act_fast_kernel(const __half* __restrict__ X, int M, int K,
                                         int8_t* __restrict__ xs8, __half* __restrict__ xscale) {
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
        if (lane == 0)
            xscale[static_cast<size_t>(m) * subs + s] =
                __float2half((amax > 0.0f) ? (amax / 127.0f) : 0.0f);
    }
}

struct WeightPlanes {
    int8_t* qs = nullptr;
    __half* d = nullptr;
    int N = 0;
    int K = 0;
};
struct ActScratch {
    int8_t* xs8 = nullptr;
    __half* xscale = nullptr;
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
    auto it = g_weights.find(src);
    if (it != g_weights.end() && it->second.N == N && it->second.K == K) return true;
    if (capturing) return false;  // never allocate inside graph capture

    WeightPlanes w;
    w.N = N;
    w.K = K;
    if (cudaMalloc(&w.qs, static_cast<size_t>(N) * K) != cudaSuccess) return false;
    if (cudaMalloc(&w.d, static_cast<size_t>(N) * (K / 32) * sizeof(__half)) != cudaSuccess) {
        cudaFree(w.qs);
        return false;
    }
    const int total = N * (K / 32);
    q8_split_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint8_t*>(src), w.qs, w.d, total, K / 32);
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
        g_act = ActScratch{};
    }
    if (cudaMalloc(&g_act.xs8, mk) != cudaSuccess) return false;
    if (cudaMalloc(&g_act.xscale, msubs * sizeof(__half)) != cudaSuccess) return false;
    g_act.cap_mk = mk;
    g_act.cap_msubs = msubs;
    return true;
}

}  // namespace

bool mmq_q8_imma_gemm(const void* w_q8_blocks, const __half* x_f16, __half* out_f16, int M, int N,
                      int K, cudaStream_t stream, float beta) {
    if (M < 64 || N % kBN != 0 || K % kBK != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;

    std::lock_guard<std::mutex> lk(g_mtx);
    const bool capturing = stream_capturing(stream);
    if (!ensure_weight(w_q8_blocks, N, K, stream, capturing)) return false;
    if (!ensure_act(M, K, capturing)) return false;

    {
        const int total_warps_needed = M * (K / 32);
        const int blocks = min(2048, (total_warps_needed + 7) / 8);
        quantize_act_fast_kernel<<<blocks, 256, 0, stream>>>(x_f16, M, K, g_act.xs8, g_act.xscale);
    }

    const auto& w = g_weights[w_q8_blocks];
    dim3 grid(N / kBN, (M + kBM - 1) / kBM);
    if (beta == 1.0f)
        mmq_q8_imma_kernel<true><<<grid, kThreads, 0, stream>>>(g_act.xs8, g_act.xscale, w.qs, w.d,
                                                                out_f16, M, N, K);
    else
        mmq_q8_imma_kernel<false><<<grid, kThreads, 0, stream>>>(g_act.xs8, g_act.xscale, w.qs,
                                                                 w.d, out_f16, M, N, K);
    static bool logged = false;
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("Q8_0 IMMA prefill ACTIVE (M=%d N=%d K=%d, tile 128x128x64, 8 warps)", M, N, K);
    }
    return true;
}

void mmq_q8_imma_release_all() {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (auto& [_, w] : g_weights) {
        cudaFree(w.qs);
        cudaFree(w.d);
    }
    g_weights.clear();
    if (g_act.xs8) {
        cudaFree(g_act.xs8);
        cudaFree(g_act.xscale);
        g_act = ActScratch{};
    }
}

}  // namespace imp
