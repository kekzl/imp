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
#include "compute/mmq_q8_imma_internal.cuh"
#include "core/logging.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace imp {

namespace {

// One K-step tile load, all 256 threads cooperating (loops are fully
// unrolled compile-time for both BM variants):
//   A   [BM][kBK]  s8     M-tail rows zero-filled
//   B   [kBN][kBK] s8     weight rows always full (N % kBN == 0 gate)
//   Asc [BM][2]    half   activation scale, d-plane cols (kb0, kb0+1)
//   Ars [BM][2]    float  activation rowsum, same cols
//   Bsc [kBN][2][2] half  weight (α, β) interleaved for both kb cols
template <int BM, bool WB>
__device__ __forceinline__ void load_kstep(int tid, const int8_t* __restrict__ A,
                                           const __half* __restrict__ Asc,
                                           const float* __restrict__ Ars,
                                           const int8_t* __restrict__ B,
                                           const __half* __restrict__ Bsc, int8_t (*sA)[kRow],
                                           int8_t (*sB)[kRow], __half (*sAsc)[2],
                                           float (*sArs)[2], __half (*sBsc)[2][2], int base_m,
                                           int M, int K, int subs, int k_base,
                                           int base_n_rows) {
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
        cp_async_cg_16(&sB[row][col], B + static_cast<size_t>(row) * K + k_base + col,
                       (base_n_rows < 0) || (row < base_n_rows));
    }
    const int kb0 = k_base / 32;
#pragma unroll
    for (int i = tid; i < BM; i += kThreads) {
        const bool valid = (base_m + i) < M;
        cp_async_ca_4(&sAsc[i][0], Asc + static_cast<size_t>(base_m + i) * subs + kb0, valid);
        if (WB)
            cp_async_ca_8(&sArs[i][0], Ars + static_cast<size_t>(base_m + i) * subs + kb0, valid);
    }
#pragma unroll
    for (int i = tid; i < kBN; i += kThreads) {
        cp_async_ca_8(&sBsc[i][0][0], Bsc + (static_cast<size_t>(i) * subs + kb0) * 2,
                      (base_n_rows < 0) || (i < base_n_rows));
    }
}

// out = (or +=, BETA1) the α/β-scaled IMMA over [BM,kBN] tiles. Grouped MoE
// form: gridDim.z = ne with device expert_offsets; dense passes offsets ==
// nullptr (z extent 1).
// SPLITK (dense-only): gridDim.z = K-split index instead of expert. With a
// single M-tile the grid is only N/kBN blocks — far too few to hide the
// K-loop latency (the spec-decode verify bottleneck, issue #667). Each split
// computes ks_per_split K-steps and stores its fp32 partial tile to
// split_out[z][M][N]; mmq_splitk_finalize_kernel reduces the slices (fixed
// order — bit-reproducible) and applies the beta/residual form.
template <int BM, bool BETA1, bool WB /* weight beta term (Q4_K); false = pure alpha (Q8_0) */,
          bool SPLITK = false>
__global__ void __launch_bounds__(kThreads)
    mmq_imma_kernel(const int8_t* __restrict__ X_s8, const __half* __restrict__ x_scale,
                    const float* __restrict__ x_rowsum, const int8_t* __restrict__ W_s8,
                    const __half* __restrict__ w_sc, __half* __restrict__ out, int M, int N,
                    int K, const int32_t* __restrict__ expert_offsets, size_t w_stride,
                    size_t wsc_stride, float* __restrict__ split_out = nullptr,
                    int ks_per_split = 0) {
    constexpr int kWM = (BM == 128) ? 4 : 2;  // warp grid
    constexpr int kWN = (BM == 128) ? 2 : 4;
    constexpr int kTileM = BM / kWM;       // 32 / 16
    constexpr int kTileN = kBN / kWN;      // 64 / 32
    constexpr int kMF = kTileM / 16;       // 2 / 1
    constexpr int kNF = kTileN / 8;        // 8 / 4
    static_assert(kWM * kWN * 32 == kThreads, "warp grid must fill the CTA");

    int rows = M;
    size_t row_off = 0;
    const int e = SPLITK ? 0 : blockIdx.z;  // SPLITK reuses gridDim.z for the K-split
    if (expert_offsets != nullptr) {
        const int32_t o0 = __ldg(&expert_offsets[e]);
        const int32_t o1 = __ldg(&expert_offsets[e + 1]);
        row_off = static_cast<size_t>(o0);
        rows = o1 - o0;
    }
    const int base_m = blockIdx.y * BM;
    const int base_n = blockIdx.x * kBN;
    if (base_m >= rows || rows == 0) return;
    const int n_rem = (N - base_n >= kBN) ? -1 : (N - base_n);  // -1 = full tile

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
    int ks_begin = 0;
    int ks_end = ksteps;
    if (SPLITK) {
        ks_begin = blockIdx.z * ks_per_split;
        ks_end = min(ksteps, ks_begin + ks_per_split);
        if (ks_begin >= ks_end)
            return;
    }
    load_kstep<BM, WB>(tid, A, Asc, Ars, B, Bsc, sA[0], sB[0], sAsc[0], sArs[0], sBsc[0],
                       base_m, rows, K, subs, ks_begin * kBK, n_rem);
    cp_async_commit();

    for (int ks = ks_begin; ks < ks_end; ++ks) {
        const int stage = (ks - ks_begin) & 1;
        if (ks + 1 < ks_end) {
            const int nstage = (ks + 1 - ks_begin) & 1;
            load_kstep<BM, WB>(tid, A, Asc, Ars, B, Bsc, sA[nstage], sB[nstage], sAsc[nstage],
                               sArs[nstage], sBsc[nstage], base_m, rows, K, subs, (ks + 1) * kBK,
                               n_rem);
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
                const float rs_lo = WB ? sArs[stage][arow_lo][kb] : 0.0f;
                const float rs_hi = WB ? sArs[stage][arow_hi][kb] : 0.0f;

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
                    if (WB) {
                        acc[mf][nf][0] += da_lo * fmaf(al, static_cast<float>(c0), bl * rs_lo);
                        acc[mf][nf][1] += da_lo * fmaf(ah, static_cast<float>(c1), bh * rs_lo);
                        acc[mf][nf][2] += da_hi * fmaf(al, static_cast<float>(c2), bl * rs_hi);
                        acc[mf][nf][3] += da_hi * fmaf(ah, static_cast<float>(c3), bh * rs_hi);
                    } else {
                        // pure-alpha Q8_0 fast path (the unified beta form
                        // cost Q8 ~6%: 11.4k -> 10.7k pp512, fixed here)
                        acc[mf][nf][0] += (da_lo * al) * static_cast<float>(c0);
                        acc[mf][nf][1] += (da_lo * ah) * static_cast<float>(c1);
                        acc[mf][nf][2] += (da_hi * al) * static_cast<float>(c2);
                        acc[mf][nf][3] += (da_hi * ah) * static_cast<float>(c3);
                    }
                }
            }
        }
        __syncthreads();
    }

    // SPLITK epilogue: store the fp32 partial tile to this split's slice;
    // the finalize kernel reduces slices and applies beta/residual.
    if constexpr (SPLITK) {
        float* Cs = split_out + static_cast<size_t>(blockIdx.z) * (static_cast<size_t>(M) * N);
#pragma unroll
        for (int mf = 0; mf < kMF; ++mf) {
            const int row_lo = base_m + warp_m * kTileM + mf * 16 + rl;
            const int row_hi = row_lo + 8;
#pragma unroll
            for (int nf = 0; nf < kNF; ++nf) {
                const int col = base_n + warp_n * kTileN + nf * 8 + cl * 2;
                if (col + 1 >= N) continue;
                if (row_lo < rows) {
                    Cs[static_cast<size_t>(row_lo) * N + col] = acc[mf][nf][0];
                    Cs[static_cast<size_t>(row_lo) * N + col + 1] = acc[mf][nf][1];
                }
                if (row_hi < rows) {
                    Cs[static_cast<size_t>(row_hi) * N + col] = acc[mf][nf][2];
                    Cs[static_cast<size_t>(row_hi) * N + col + 1] = acc[mf][nf][3];
                }
            }
        }
    } else {
        // Epilogue: FP16 store, M-tail predicated (N is a multiple of kBN).
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
}

// Reduce the SPLITK partial slices (fixed order — bit-reproducible) and
// apply the beta form: out = sum (beta 0) or out += sum (beta 1).
__global__ void mmq_splitk_finalize_kernel(const float* __restrict__ split_out, int n_splits,
                                           int total, __half* __restrict__ out, int beta1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = 0.0f;
    for (int s = 0; s < n_splits; ++s)
        v += split_out[static_cast<size_t>(s) * total + idx];
    out[idx] = beta1 ? __hadd(out[idx], __float2half(v)) : __float2half(v);
}

// -----------------------------------------------------------------------------
// Q6_K RAW-read kernel (per-16 scales, symmetric — no beta/rowsum term).
//
// The 210-B super-blocks are only 2-aligned (blocks cp.async at all sizes),
// so a one-time 224-B-stride repack (plain byte copy, +6.7% of the Q6_K
// bytes — ~110 MB for the 30B down_proj experts) restores 16-B alignment;
// the forge 2026-05-28 "2-aligned quant repack" finding, applied here.
//
// Per-16 scale granularity vs the k32 MMA: HALF-MMA SPLIT — issue the
// m16n8k32 twice per sub-block, once as (b0, 0) and once as (0, b1); the
// zeroed operand register makes each MMA return the 16-wide partial sum,
// which gets its own α = d·sc16. Same int-op count as a k16 MMA pair,
// zero layout restructuring.
//
// Quad mapping (see dequant_gpu.cu Q6_K header): K-step ks covers sub-block
// pair j = (2ks)%8, j+1 → group g = j>>2, quads (j%4, j%4+1). The pair
// shares ql bytes [g*64 .. +63] (quad&1 selects the 32-byte half, quad>=2
// the nibble) and qh bytes [g*32 .. +31] (shift quad*2).
// -----------------------------------------------------------------------------

__global__ void q6k_repack_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                  size_t n_blocks) {
    const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n_blocks * 105;  // 210 B as 105 u16 copies
    if (b >= total) return;
    const size_t blk = b / 105;
    const size_t off = (b % 105) * 2;
    uint16_t v;
    memcpy(&v, src + blk * 210 + off, 2);
    memcpy(dst + blk * kQ6Stride + off, &v, 2);
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

// TUNING LADDER 2026-06-07 (Qwen3-8B Q8_0 pp512, baseline 12 131 tok/s) —
// three refuted attempts, kernel is at its local optimum in this structure:
//   __launch_bounds__(256,2):  9 768 (-19%) — 184-reg kernel spills under
//                              the 128-reg cap; 64-fp32 accumulator file.
//   NT=64 tile (acc 32):       9 685 (-20%) — bigger tiles win, matching
//                              the 2026-05-18 phase-2B finding (+108% from
//                              tile growth). 1 CTA/SM stands.
//   ldmatrix.x4 A+B fetch:    11 619 (-4%)  — smem fetch is NOT the binding
//                              constraint after the +16-B row pad; matches
//                              the 2B.5 neutral-to-negative result.
// Remaining structural ideas (BK=128 sync-halving w/ dynamic smem, warp
// specialization) are larger rewrites; the model-level gap vs llama.cpp is
// 1.13x — the smallest on the board. Spend elsewhere first.
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

struct Q6kRepack {
    uint8_t* blocks = nullptr;  // 224-B-stride super-blocks
    size_t n_blocks = 0;
};

std::mutex g_mtx;
std::unordered_map<const void*, WeightPlanes> g_weights;
std::unordered_map<const void*, Q6kRepack> g_q6k;
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
    IMP_CUDA_CHECK_LAUNCH();
    g_weights[src] = w;
    return true;
}

bool ensure_q6k(const void* src, size_t n_blocks, cudaStream_t stream, bool capturing) {
    auto it = g_q6k.find(src);
    if (it != g_q6k.end() && it->second.n_blocks == n_blocks) return true;
    if (capturing) return false;
    Q6kRepack r;
    r.n_blocks = n_blocks;
    if (cudaMalloc(&r.blocks, n_blocks * kQ6Stride) != cudaSuccess) return false;
    const size_t total = n_blocks * 105;
    q6k_repack_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint8_t*>(src), r.blocks, n_blocks);
    IMP_CUDA_CHECK_LAUNCH();
    g_q6k[src] = r;
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

// SPLITK partial-slice scratch (grow-only, freed in mmq_q8_imma_release_all).
struct SplitKScratch {
    float* buf = nullptr;
    size_t cap = 0;
};
SplitKScratch g_splitk;

bool ensure_splitk(size_t floats, bool capturing) {
    if (g_splitk.buf && g_splitk.cap >= floats) return true;
    if (capturing) return false;
    if (g_splitk.buf) cudaFree(g_splitk.buf);
    if (cudaMalloc(&g_splitk.buf, floats * sizeof(float)) != cudaSuccess) {
        g_splitk.buf = nullptr;
        g_splitk.cap = 0;
        return false;
    }
    g_splitk.cap = floats;
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
    IMP_CUDA_CHECK_LAUNCH();
}

bool gemm_common(const void* w_blocks, int qkind /*0=q8 1=q4k 2=q6k*/, const __half* x_f16,
                 __half* out_f16, int M, int N, int K, cudaStream_t stream, float beta,
                 const int32_t* d_offsets, int h_max_rows, int expanded, int ne) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const bool capturing = stream_capturing(stream);
    if (qkind == 0 && !ensure_weight(w_blocks, ne * N, K, stream, capturing)) return false;
    if (qkind == 2 &&
        !ensure_q6k(w_blocks, static_cast<size_t>(ne) * N * (K / 256), stream, capturing))
        return false;
    // qkind 3 (Q5_1) reads raw blocks — nothing to prepare
    const int act_rows = d_offsets ? expanded : M;
    if (!ensure_act(act_rows, K, capturing)) return false;
    quantize_act(x_f16, act_rows, K, stream);

    const int grid_m_rows = d_offsets ? h_max_rows : M;
    const bool small_m = d_offsets && h_max_rows < 96;
    const int bm = small_m ? 32 : 128;
    dim3 grid((N + kBN - 1) / kBN, (grid_m_rows + bm - 1) / bm, ne);

    if (qkind == 1) {
        // raw-read kernel: zero extra weight VRAM
        const uint8_t* w4 = static_cast<const uint8_t*>(w_blocks);
        const size_t w_stride_blocks = static_cast<size_t>(N) * (K / 256);
        if (small_m) {
            mmq_imma_q4k_raw_kernel<32, false><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w4, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        } else if (beta == 1.0f) {
            mmq_imma_q4k_raw_kernel<128, true><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w4, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            mmq_imma_q4k_raw_kernel<128, false><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w4, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        }
        return true;
    }
    if (qkind == 3) {
        const uint8_t* w5 = static_cast<const uint8_t*>(w_blocks);
        const size_t w_stride_blocks = static_cast<size_t>(N) * (K / 32);
        if (small_m) {
            mmq_imma_q51_raw_kernel<32, false><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w5, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        } else if (beta == 1.0f) {
            mmq_imma_q51_raw_kernel<128, true><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w5, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            mmq_imma_q51_raw_kernel<128, false><<<grid, kThreads, 0, stream>>>(
                g_act.xs8, g_act.xscale, g_act.xrowsum, w5, out_f16, M, N, K, d_offsets,
                w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        }
        return true;
    }
    if (qkind == 2) {
        const uint8_t* w6 = g_q6k[w_blocks].blocks;
        const size_t w_stride_blocks = static_cast<size_t>(N) * (K / 256);
        static bool smem_set6 = false;
        if (!smem_set6) {
            smem_set6 = true;
            cudaFuncSetAttribute(mmq_imma_q6k_raw_kernel<32, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(q6k_smem_bytes(32)));
            cudaFuncSetAttribute(mmq_imma_q6k_raw_kernel<128, false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(q6k_smem_bytes(128)));
            cudaFuncSetAttribute(mmq_imma_q6k_raw_kernel<128, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(q6k_smem_bytes(128)));
        }
        if (small_m) {
            mmq_imma_q6k_raw_kernel<32, false>
                <<<grid, kThreads, q6k_smem_bytes(32), stream>>>(g_act.xs8, g_act.xscale, w6,
                                                                 out_f16, M, N, K, d_offsets,
                                                                 w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        } else if (beta == 1.0f) {
            mmq_imma_q6k_raw_kernel<128, true>
                <<<grid, kThreads, q6k_smem_bytes(128), stream>>>(g_act.xs8, g_act.xscale, w6,
                                                                  out_f16, M, N, K, d_offsets,
                                                                  w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        } else {
            mmq_imma_q6k_raw_kernel<128, false>
                <<<grid, kThreads, q6k_smem_bytes(128), stream>>>(g_act.xs8, g_act.xscale, w6,
                                                                  out_f16, M, N, K, d_offsets,
                                                                  w_stride_blocks);
            IMP_CUDA_CHECK_LAUNCH();
        }
        return true;
    }

    const auto& w = g_weights[w_blocks];
    const size_t w_stride = static_cast<size_t>(N) * K;
    const size_t wsc_stride = static_cast<size_t>(N) * (K / 32) * 2;

    // Dense small-M split-K: with one M-tile the grid is N/kBN blocks (~20 on
    // a 2.5k-wide weight) — far too few to hide the K-loop latency. The call
    // costs ~35 µs regardless of N, which is the spec-decode verify
    // bottleneck (issue #667). Split the K-steps across gridDim.z into fp32
    // partial slices and reduce; the finalize kernel applies beta/residual.
    if (d_offsets == nullptr && M <= 32) {
        const int ksteps_total = K / kBK;
        const int n_tiles = (N + kBN - 1) / kBN;
        int S = 1;
        while (S < 8 && n_tiles * S < 256 && ksteps_total / (S * 2) >= 2) S *= 2;
        if (S > 1) {
            const int ks_per_split = (ksteps_total + S - 1) / S;
            const int used = (ksteps_total + ks_per_split - 1) / ks_per_split;
            const size_t slice = static_cast<size_t>(M) * N;
            if (used > 1 && ensure_splitk(slice * used, capturing)) {
                dim3 sgrid(n_tiles, 1, used);
                mmq_imma_kernel<32, false, false, true><<<sgrid, kThreads, 0, stream>>>(
                    g_act.xs8, g_act.xscale, g_act.xrowsum, w.qs, w.sc, out_f16, M, N, K,
                    nullptr, w_stride, wsc_stride, g_splitk.buf, ks_per_split);
                IMP_CUDA_CHECK_LAUNCH();
                const int total = static_cast<int>(slice);
                mmq_splitk_finalize_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                    g_splitk.buf, used, total, out_f16, beta == 1.0f ? 1 : 0);
                IMP_CUDA_CHECK_LAUNCH();
                return true;
            }
        }
    }

    // plane path = Q8_0 only since the raw-read kernels: pure-alpha (WB=false)
    if (small_m) {
        mmq_imma_kernel<32, false, false><<<grid, kThreads, 0, stream>>>(
            g_act.xs8, g_act.xscale, g_act.xrowsum, w.qs, w.sc, out_f16, M, N, K, d_offsets,
            w_stride, wsc_stride);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (beta == 1.0f) {
        mmq_imma_kernel<128, true, false><<<grid, kThreads, 0, stream>>>(
            g_act.xs8, g_act.xscale, g_act.xrowsum, w.qs, w.sc, out_f16, M, N, K, d_offsets,
            w_stride, wsc_stride);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        mmq_imma_kernel<128, false, false><<<grid, kThreads, 0, stream>>>(
            g_act.xs8, g_act.xscale, g_act.xrowsum, w.qs, w.sc, out_f16, M, N, K, d_offsets,
            w_stride, wsc_stride);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}

}  // namespace

bool mmq_q8_imma_gemm(const void* w_q8_blocks, const __half* x_f16, __half* out_f16, int M, int N,
                      int K, cudaStream_t stream, float beta) {
    // M >= 2 (was >= 64): small-M callers (spec-decode verify chunks, short
    // prompts) are exactly where the dequant->cuBLAS fallback hurts most; the
    // tiles zero-fill the M-tail (same machinery as the MoE per-expert path).
    if (M < 2 || N % 2 != 0 || K % kBK != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    return gemm_common(w_q8_blocks, 0, x_f16, out_f16, M, N, K, stream, beta, nullptr, 0, 0, 1);
}

bool mmq_q4k_imma_gemm(const void* w_q4k_blocks, const __half* x_f16, __half* out_f16, int M,
                       int N, int K, cudaStream_t stream, float beta) {
    if (M < 2 || N % 2 != 0 || K % 256 != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    return gemm_common(w_q4k_blocks, 1, x_f16, out_f16, M, N, K, stream, beta, nullptr, 0, 0, 1);
}

bool mmq_q6k_imma_gemm(const void* w_q6k_blocks, const __half* x_f16, __half* out_f16, int M,
                       int N, int K, cudaStream_t stream, float beta) {
    if (M < 64 || N % 2 != 0 || K % 256 != 0) return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    return gemm_common(w_q6k_blocks, 2, x_f16, out_f16, M, N, K, stream, beta, nullptr, 0, 0, 1);
}

bool mmq_imma_moe_gemm(const void* w_blocks, int qkind, const __half* x_f16, __half* out_f16,
                       const int32_t* d_offsets, int h_max_rows, int expanded, int ne, int N,
                       int K, cudaStream_t stream) {
    if (N % 2 != 0) return false;
    if (K % ((qkind == 1 || qkind == 2) ? 256 : kBK) != 0) return false;
    if (h_max_rows <= 0 || expanded <= 0 || ne <= 0) return false;
    const bool ok = gemm_common(w_blocks, qkind, x_f16, out_f16, /*M=*/0, N, K, stream, 0.0f,
                                d_offsets, h_max_rows, expanded, ne);
    static bool logged = false;
    if (ok && !logged) {
        logged = true;
        IMP_LOG_INFO("MoE IMMA prefill ACTIVE (%s, ne=%d N=%d K=%d max_rows=%d)",
                     qkind == 1 ? "Q4_K"
                                : (qkind == 2 ? "Q6_K" : (qkind == 3 ? "Q5_1" : "Q8_0")),
                     ne, N, K, h_max_rows);
    }
    return ok;
}

void mmq_q8_imma_release_all() {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_splitk.buf) {
        cudaFree(g_splitk.buf);
        g_splitk.buf = nullptr;
        g_splitk.cap = 0;
    }
    for (auto& [_, w] : g_weights) {
        cudaFree(w.qs);
        cudaFree(w.sc);
    }
    g_weights.clear();
    for (auto& [_, r] : g_q6k) cudaFree(r.blocks);
    g_q6k.clear();
    if (g_act.xs8) {
        cudaFree(g_act.xs8);
        cudaFree(g_act.xscale);
        cudaFree(g_act.xrowsum);
        g_act = ActScratch{};
    }
}

}  // namespace imp
