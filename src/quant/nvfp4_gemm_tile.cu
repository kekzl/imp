// Own NVFP4 prefill GEMM (ROADMAP row 123): CTA tile 128 x BN, ring of BK-element stages,
// 8 MMA warps of 64 x BN/4, ldmatrix fragments, native mma.sync kind::mxf4nvf4 on the CUTLASS
// bytes. Fragment and SF lane mappings: nvfp4_gemm_smallm_v2.cu (CUTLASS SM120_16x8x64_TN_VS).
// SfAtom: 128 rows x 4 K-groups = 512 B, row r at (r%32)*16 + (r/32)*4; atoms K-major per
// 128-row tile, so a 128-row slab reads BK/64 contiguous atoms per stage.
// Mainloops: TMA (one producer lane: 2D tensor loads of the A/W tiles with hardware swizzle +
// 1D bulk loads of the SF slabs, mbarrier ring) and sync (all warps cp.async, padded rows).
// Measured against this design (test_nvfp4_gemm_tile bench, RTX 5090): one producer warp of
// 16 B cp.async 4x slower than CUTLASS, per-row 64 B bulk copies 13x, 5 stages / grouped-M
// raster / persistent CTAs / direct epilogue stores 0..+8 % slower; 128x128 is L2-bound (77 %).
// gemm_mxfp8_tile / gemm_mxfp4_tile: the same kernels on MXFP8 (kind::mxf8f6f4 m16n8k32, half the BK) and
// MXFP4 bytes (scale_vec::2X ue8m0, one 128-K SfAtom per stage).

#include "quant/nvfp4_gemm_tile.h"
#include "quant/nvfp4_gemm_tile_core.cuh"

#include "core/logging.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

namespace imp {
namespace {

using namespace nvfp4_tile;  // NOLINT(google-build-using-namespace): this TU's own PTX helpers

// Stream-K arrival flags, one per CTA: 0 at load, set by a contributor, reset by the tile's finisher in the
// same launch (no per-launch memset). One stream-K GEMM at a time, as for the shared partial workspace.
constexpr int kMaxSkCtas = 1024;
__device__ int g_sk_flags[kMaxSkCtas];

struct TileArgs {
    const uint8_t* a;
    const uint8_t* a_sf;
    const uint8_t* w;
    const uint8_t* w_sf;
    half* d;
    float alpha;
    int M, N, K;
    // Stream-K (Cfg::kSk): one fp32 partial tile per CTA (workspace); each CTA owns units
    // [b * sk_per, (b + 1) * sk_per) of tiles * KT. Arrival flags: g_sk_flags.
    float* sk_parts;
    int sk_per;
    const float* col_scale;  // MXFP8 only: D column n *= col_scale[n] (FP8 row-scale view), may be null
};

// alpha, times col_scale[col], col + 1 for the MXFP8 row-scale view.
template <class C>
__device__ __forceinline__ float2 out_scale(const TileArgs& p, int col) {
    if constexpr (C::kMx8)
        if (p.col_scale != nullptr && col < p.N)  // N % 8 == 0: col + 1 < N too
            return make_float2(p.alpha * p.col_scale[col], p.alpha * p.col_scale[col + 1]);
    return make_float2(p.alpha, p.alpha);
}

// Sync mainloop stage load: 16 B cp.async chunks over the 256 MMA threads (rows >= M/N zero-fill).
template <class C>
__device__ __forceinline__ void load_stage(uint32_t s_base, const TileArgs& p, int m0, int n0, int kt) {
    constexpr int kData = kBM * C::kChunks;
    constexpr int kSf = C::kSfBytes / 16;
    const size_t row_bytes = C::row_bytes(p.K);
    const size_t k_byte = static_cast<size_t>(kt) * C::kRB;
    const size_t sf_k = C::sf_k(kt);
    const size_t sf_slab = C::sf_slab(p.K);
    for (int i = threadIdx.x; i < 2 * kData + 2 * kSf; i += kMmaThreads) {
        if (i < 2 * kData) {
            const bool is_w = i >= kData;
            const int j = is_w ? i - kData : i;
            const int r = j / C::kChunks;
            const int c = j % C::kChunks;
            const int g = (is_w ? n0 : m0) + r;
            const bool ok = g < (is_w ? p.N : p.M);
            const uint8_t* src = (is_w ? p.w : p.a) + (static_cast<size_t>(ok ? g : 0) * row_bytes) + k_byte +
                                 (static_cast<size_t>(c) * 16);
            cp_async16(s_base + (is_w ? C::kATile : 0) + (r * C::kRS) + (c * 16), src, ok ? 16 : 0);
        } else {
            const int j = i - 2 * kData;
            const bool is_w = j >= kSf;
            const int c = is_w ? j - kSf : j;
            const uint8_t* src = is_w ? p.w_sf + (static_cast<size_t>(n0 / 128) * sf_slab)
                                      : p.a_sf + (static_cast<size_t>(m0 / kBM) * sf_slab);
            cp_async16(s_base + (is_w ? C::kSfW : C::kSfA) + c * 16,
                       src + sf_k + (static_cast<size_t>(c) * 16), 16);
        }
    }
}

// TMA producer lane: arm the stage's tx count, then the A/W tiles (OOB rows land as 0) and the
// SF slabs (one per 128 rows; a slab past N re-reads the last one: finite scales on zero rows).
template <class C>
__device__ __forceinline__ void tma_stage(uint32_t s_base, uint32_t full, const CUtensorMap* ma,
                                          const CUtensorMap* mw, const TileArgs& p, int m0, int n0, int kt) {
    const size_t sf_slab = C::sf_slab(p.K);
    const size_t sf_k = C::sf_k(kt);
    mbar_expect_tx(full, C::kTxBytes);
    tma_2d(s_base, ma, kt * C::kRB, m0, full);
    tma_2d(s_base + C::kATile, mw, kt * C::kRB, n0, full);
    bulk_1d(s_base + C::kSfA, p.a_sf + (static_cast<size_t>(m0 / kBM) * sf_slab) + sf_k, C::kSfBytes, full);
#pragma unroll
    for (int j = 0; j < C::kBN / 128; ++j)
        bulk_1d(s_base + C::kSfW + j * C::kSfBytes,
                p.w_sf + (static_cast<size_t>(min(n0 / 128 + j, (p.N - 1) / 128)) * sf_slab) + sf_k,
                C::kSfBytes, full);
}


// Per-CTA ring geometry shared by the mainloops and the epilogue.
struct Ring {
    uint8_t* smem;  // 1024-aligned generic base
    uint32_t s0;    // same, shared-window address
    uint32_t bars;  // full[s] at s*8, empty[s] after
    int m0, n0, KT, warp, lane, wm, wn;
};

// MMA warps' consumer loop: wait full[s], compute, release empty[s]. With REFILL, thread 0
// also refills the PREVIOUS stage's slot: the 8 warps left it while thread 0 computed this one,
// so the empty wait rarely blocks (refilling this slot waited on the slowest warp: 128x128 +15 %).
template <class C, bool REFILL>
__device__ __forceinline__ void tma_consume(float (&acc)[4][C::kNF][4], const Ring& g, const TileArgs& p,
                                            const CUtensorMap* ma, const CUtensorMap* mw) {
    for (int kt = 0; kt < g.KT; ++kt) {
        const int s = kt % C::kStages;
        mbar_wait(g.bars + s * 8, (kt / C::kStages) & 1);
        mma_stage<C>(acc, g.s0 + s * C::kStageBytes, g.smem + s * C::kStageBytes, g.wm, g.wn, g.lane,
                     C::sf_g0(kt));
        __syncwarp();
        if (g.lane == 0)
            mbar_arrive(g.bars + (C::kStages + s) * 8);
        const int kp = kt - 1;
        if (REFILL && threadIdx.x == 0 && kp >= 0 && kp + C::kStages < g.KT) {
            const int sp = kp % C::kStages;
            mbar_wait(g.bars + (C::kStages + sp) * 8, (kp / C::kStages) & 1);
            tma_stage<C>(g.s0 + sp * C::kStageBytes, g.bars + sp * 8, ma, mw, p, g.m0, g.n0, kp + C::kStages);
        }
    }
}

// TMA mainloop. False for the producer warp (kPw), which leaves after its last issue: every
// stage was consumed before the MMA warps leave their loop.
template <class C>
__device__ __forceinline__ bool tma_mainloop(float (&acc)[4][C::kNF][4], const Ring& g, const TileArgs& p,
                                             const CUtensorMap* ma, const CUtensorMap* mw) {
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(g.bars + s * 8, 1);
            mbar_init(g.bars + (C::kStages + s) * 8, kMmaThreads / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if constexpr (!C::kPw) {
        if (threadIdx.x == 0)
            for (int s = 0; s < C::kStages && s < g.KT; ++s)
                tma_stage<C>(g.s0 + s * C::kStageBytes, g.bars + s * 8, ma, mw, p, g.m0, g.n0, s);
        tma_consume<C, true>(acc, g, p, ma, mw);
        return true;
    } else {
        if (g.warp != kMmaThreads / 32) {
            tma_consume<C, false>(acc, g, p, ma, mw);
            return true;
        }
        if (g.lane == 0)
            for (int kt = 0; kt < g.KT; ++kt) {
                const int s = kt % C::kStages;
                if (kt >= C::kStages)
                    mbar_wait(g.bars + (C::kStages + s) * 8, ((kt / C::kStages) - 1) & 1);
                tma_stage<C>(g.s0 + s * C::kStageBytes, g.bars + s * 8, ma, mw, p, g.m0, g.n0, kt);
            }
        return false;
    }
}

template <class C>
__device__ __forceinline__ void sync_mainloop(float (&acc)[4][C::kNF][4], const Ring& g, const TileArgs& p) {
#pragma unroll
    for (int s = 0; s < C::kStages - 1; ++s) {
        if (s < g.KT)
            load_stage<C>(g.s0 + s * C::kStageBytes, p, g.m0, g.n0, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < g.KT; ++kt) {
        cp_async_wait<C::kStages - 2>();
        __syncthreads();
        const int pf = kt + C::kStages - 1;
        if (pf < g.KT)
            load_stage<C>(g.s0 + (pf % C::kStages) * C::kStageBytes, p, g.m0, g.n0, pf);
        cp_async_commit();
        const int s = kt % C::kStages;
        mma_stage<C>(acc, g.s0 + s * C::kStageBytes, g.smem + s * C::kStageBytes, g.wm, g.wn, g.lane,
                     C::sf_g0(kt));
    }
    cp_async_wait<0>();
}

// alpha, fp16, staged in so (stride kOS) for 16 B row-contiguous stores (256 MMA threads).
template <class C>
__device__ __forceinline__ void store_tile(const float (&acc)[4][C::kNF][4], half* so, int m0, int n0, int wm,
                                           int wn, int lane, const TileArgs& p) {
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));  // every warp left the ring / the last store loop
    const int T0 = lane & 3;
    const int T1 = lane >> 2;
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf) {
            const float2 sc = out_scale<C>(p, n0 + wn + nf * 8 + T0 * 2);
#pragma unroll
            for (int h = 0; h < 2; ++h)
                *reinterpret_cast<__half2*>(so + ((wm + mf * 16 + T1 + h * 8) * C::kOS) + wn + nf * 8 +
                                            T0 * 2) = __floats2half2_rn(acc[mf][nf][h * 2] * sc.x,
                                                                        acc[mf][nf][h * 2 + 1] * sc.y);
        }
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
    for (int i = threadIdx.x; i < kBM * (C::kBN / 8); i += kMmaThreads) {
        const int r = i / (C::kBN / 8);
        const int c = (i % (C::kBN / 8)) * 8;
        const int row = m0 + r;
        const int col = n0 + c;
        if (row < p.M && col < p.N)
            *reinterpret_cast<uint4*>(p.d + (static_cast<size_t>(row) * p.N) +
                                      col) = *reinterpret_cast<const uint4*>(so + (r * C::kOS) + c);
    }
}

// Stream-K. CTA b walks its unit range [begin, end) forward in segments (tile t, k-steps
// [kb, ke)): its first segment can start mid tile (kb > 0): a partial, written right at the start;
// its last can end mid tile (kb == 0, ke < KT): it holds k-step 0, finishes the tile and reads the
// partials of CTAs b+1.. at the end of its range. Whole tiles store directly.
// Partials sum in a fixed order (deterministic, not bit-equal to the data-parallel kernel).
struct SkSeg {
    int t, kb, ke;
};

__device__ __forceinline__ SkSeg sk_seg(int it, int end, int KT) {
    const int t = it / KT;
    const int kb = it - (t * KT);
    return {t, kb, min(KT, kb + (end - it))};
}

__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

template <class C>
__device__ __forceinline__ void store_direct(const float (&acc)[4][C::kNF][4], const TileArgs& p, int m0,
                                             int n0, int wm, int wn, int lane) {
    const int T0 = lane & 3;
    const int T1 = lane >> 2;
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf) {
            const int col = n0 + wn + nf * 8 + T0 * 2;
            const float2 sc = out_scale<C>(p, col);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm + mf * 16 + T1 + h * 8;
                if (row < p.M && col < p.N)
                    *reinterpret_cast<__half2*>(p.d + (static_cast<size_t>(row) * p.N) +
                                                col) = __floats2half2_rn(acc[mf][nf][h * 2] * sc.x,
                                                                         acc[mf][nf][(h * 2) + 1] * sc.y);
            }
        }
}

// Partial tile of CTA b in fragment order: float4 j of thread tid at (j * 256 + tid), coalesced.
template <class C>
__device__ __forceinline__ float4* sk_slot(const TileArgs& p, int b) {
    return reinterpret_cast<float4*>(p.sk_parts) + (static_cast<size_t>(b) * 4 * C::kNF * kMmaThreads);
}

template <class C>
__device__ __forceinline__ void sk_mainloop(const Ring& g, const TileArgs& p, const CUtensorMap* ma,
                                            const CUtensorMap* mw) {
    const int KT = p.K / C::kBK;
    const int n_tiles = (p.N + C::kBN - 1) / C::kBN;
    const int units = n_tiles * ((p.M + kBM - 1) / kBM) * KT;
    const int b = blockIdx.x;
    const int begin = b * p.sk_per;
    const int end = min(begin + p.sk_per, units);
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(g.bars + s * 8, 1);
            mbar_init(g.bars + (C::kStages + s) * 8, kMmaThreads / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    int ring = 0;
    if (g.warp == kMmaThreads / 32) {
        if (g.lane == 0)
            for (int it = begin; it < end;) {
                const SkSeg sg = sk_seg(it, end, KT);
                const int m0 = (sg.t / n_tiles) * kBM;
                const int n0 = (sg.t % n_tiles) * C::kBN;
                for (int kt = sg.kb; kt < sg.ke; ++kt, ++ring) {
                    const int s = ring % C::kStages;
                    if (ring >= C::kStages)
                        mbar_wait(g.bars + (C::kStages + s) * 8, ((ring / C::kStages) - 1) & 1);
                    tma_stage<C>(g.s0 + s * C::kStageBytes, g.bars + s * 8, ma, mw, p, m0, n0, kt);
                }
                it += sg.ke - sg.kb;
            }
        return;
    }
    float acc[4][C::kNF][4];
    for (int it = begin; it < end;) {
        const SkSeg sg = sk_seg(it, end, KT);
        const int m0 = (sg.t / n_tiles) * kBM;
        const int n0 = (sg.t % n_tiles) * C::kBN;
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < C::kNF; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    acc[i][j][r] = 0.0f;
        for (int kt = sg.kb; kt < sg.ke; ++kt, ++ring) {
            const int s = ring % C::kStages;
            mbar_wait(g.bars + s * 8, (ring / C::kStages) & 1);
            mma_stage<C>(acc, g.s0 + s * C::kStageBytes, g.smem + s * C::kStageBytes, g.wm, g.wn, g.lane,
                         C::sf_g0(kt));
            __syncwarp();
            if (g.lane == 0)
                mbar_arrive(g.bars + (C::kStages + s) * 8);
        }
        float4* own = sk_slot<C>(p, b);
        if (sg.kb > 0) {  // contributor: this tile's k-step 0 belongs to an earlier CTA
#pragma unroll
            for (int mf = 0; mf < 4; ++mf)
#pragma unroll
                for (int nf = 0; nf < C::kNF; ++nf)
                    own[((mf * C::kNF + nf) * kMmaThreads) + threadIdx.x] =
                        make_float4(acc[mf][nf][0], acc[mf][nf][1], acc[mf][nf][2], acc[mf][nf][3]);
            __threadfence();
            asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
            if (threadIdx.x == 0)
                st_release(g_sk_flags + b, 1);
        } else {
            if (sg.ke < KT) {  // finisher: CTAs b+1 .. last hold k-steps [ke, KT)
                const int last = ((sg.t * KT) + KT - 1) / p.sk_per;
                for (int q = b + 1; q <= last; ++q) {
                    if (threadIdx.x == 0) {
                        while (ld_acquire(g_sk_flags + q) == 0) {}
                        g_sk_flags[q] = 0;  // the only reader: every set flag is 0 again when the launch ends
                    }
                    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
                    const float4* src = sk_slot<C>(p, q);
#pragma unroll
                    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
                        for (int nf = 0; nf < C::kNF; ++nf) {
                            const float4 v = __ldcg(src + ((mf * C::kNF + nf) * kMmaThreads) + threadIdx.x);
                            acc[mf][nf][0] += v.x;
                            acc[mf][nf][1] += v.y;
                            acc[mf][nf][2] += v.z;
                            acc[mf][nf][3] += v.w;
                        }
                }
            }
            if constexpr (C::kSkStaged)
                store_tile<C>(acc, reinterpret_cast<half*>(g.smem + (C::kStages * C::kStageBytes) + 128), m0,
                              n0, g.wm, g.wn, g.lane, p);
            else
                store_direct<C>(acc, p, m0, n0, g.wm, g.wn, g.lane);
        }
        it += sg.ke - sg.kb;
    }
}

template <class C>
__global__ void __launch_bounds__(C::kThreads, 1) gemm_nvfp4_tile_kernel(
    TileArgs p, const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_w) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + C::kAlign - 1) & ~uint32_t(C::kAlign - 1)) - raw;
    const int warp = threadIdx.x >> 5;
    const Ring g{smem_raw + pad,
                 raw + pad,
                 raw + pad + C::kStages * C::kStageBytes,
                 static_cast<int>(blockIdx.y) * kBM,
                 static_cast<int>(blockIdx.x) * C::kBN,
                 p.K / C::kBK,
                 warp,
                 static_cast<int>(threadIdx.x & 31),
                 (warp & 1) * 64,
                 ((warp >> 1) & 3) * C::kWN};

    if constexpr (C::kSk) {
        sk_mainloop<C>(g, p, &map_a, &map_w);
    } else {
        float acc[4][C::kNF][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < C::kNF; ++j)
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    acc[i][j][r] = 0.0f;

        if constexpr (C::kTma) {
            if (!tma_mainloop<C>(acc, g, p, &map_a, &map_w))
                return;
        } else {
            sync_mainloop<C>(acc, g, p);
        }
        store_tile<C>(acc, reinterpret_cast<half*>(g.smem), g.m0, g.n0, g.wm, g.wn, g.lane, p);
    }
}

// cuTensorMapEncodeTiled via the runtime's driver entry point (no link-time libcuda dependency).
using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
                                   CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
                                   CUtensorMapFloatOOBfill);

EncodeTiledFn encode_fn() {
    static const EncodeTiledFn fn = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q{};
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) !=
                cudaSuccess ||
            q != cudaDriverEntryPointSuccess)
            return static_cast<EncodeTiledFn>(nullptr);
        return reinterpret_cast<EncodeTiledFn>(p);
    }();
    return fn;
}

// [rows, row_bytes] uint8 operand, box [box_bytes, box_rows], swizzle matching box_bytes.
bool encode_operand(CUtensorMap* map, const void* base, int rows, int row_bytes, int box_bytes,
                    int box_rows) {
    const EncodeTiledFn fn = encode_fn();
    if (fn == nullptr)
        return false;
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(row_bytes), static_cast<cuuint64_t>(rows)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(row_bytes)};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(box_bytes), static_cast<cuuint32_t>(box_rows)};
    const cuuint32_t elem[2] = {1, 1};
    const CUtensorMapSwizzle swz = box_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
    return fn(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dims, strides, box, elem,
              CU_TENSOR_MAP_INTERLEAVE_NONE, swz, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <class C>
bool launch_tile(const TileArgs& p, cudaStream_t stream) {
    if (p.K % C::kBK != 0)
        return false;
    CUtensorMap map_a{};
    CUtensorMap map_w{};
    if constexpr (C::kTma) {
        if (reinterpret_cast<uintptr_t>(p.a) % 16 != 0 || reinterpret_cast<uintptr_t>(p.w) % 16 != 0 ||
            !encode_operand(&map_a, p.a, p.M, static_cast<int>(C::row_bytes(p.K)), C::kRB, kBM) ||
            !encode_operand(&map_w, p.w, p.N, static_cast<int>(C::row_bytes(p.K)), C::kRB, C::kBN))
            return false;
    }
    static const bool smem_ok = cudaFuncSetAttribute(gemm_nvfp4_tile_kernel<C>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    dim3 grid((p.N + C::kBN - 1) / C::kBN, (p.M + kBM - 1) / kBM);
    if constexpr (C::kSk) {  // caller set sk_per and the workspace
        const int units = static_cast<int>(grid.x * grid.y) * (p.K / C::kBK);
        grid = dim3((units + p.sk_per - 1) / p.sk_per);
    }
    gemm_nvfp4_tile_kernel<C><<<grid, C::kThreads, C::kSmemBytes, stream>>>(p, map_a, map_w);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

namespace {

using SkCfg = Cfg<128, 128, 3, true, true, true>;  // staged epilogue: M=512 N=1024 15.8 vs 19.0 us direct x4
constexpr size_t kSkPartBytes = size_t(kBM) * 128 *
                                sizeof(float);  // stream-K workspace: one fp32 tile per CTA

int sm_count() {
    static const int sms = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            return 170;  // RTX 5090
        return n > 0 ? n : 170;
    }();
    return sms;
}

// Data-parallel tile by wave cost per column of work (128x256 1.0, 128x128 1.12, isolated
// full-wave bench; 128x256 only from 3/4 of a wave). Stream-K only when that tile idles
// >= 15 % of its waves: M=512 N=17408 88.4 vs 92.1 us, but M=2048 N=6144 132.5 vs 128.5.
int pick_variant(int M, int N, int sms, bool sk_fits) {
    const int m_tiles = (M + kBM - 1) / kBM;
    const int t128 = ((N + 127) / 128) * m_tiles;
    const int t256 = ((N + 255) / 256) * m_tiles;
    const auto waves = [&](int ctas) {
        return std::ceil(static_cast<float>(ctas) / static_cast<float>(sms));
    };
    const float c256 = waves(t256) * 256.0f;
    const float c128 = waves(t128) * 128.0f * 1.12f;
    const bool use256 = 4 * t256 >= 3 * sms && c256 <= c128;
    const int dp = use256 ? t256 : t128;
    const float idle = 1.0f - (static_cast<float>(dp) / (waves(dp) * static_cast<float>(sms)));
    if (sk_fits && idle >= 0.15f)
        return 3;
    return use256 ? 1 : 0;
}

// All formats run the same byte geometry: an MXFP8 stage of BK elements moves what an NVFP4 / MXFP4 stage
// of 2 x BK does, so the NVFP4 tile choices carry over with BK halved.
template <Fmt F>
bool run_tile(TileArgs p, cudaStream_t stream, int variant, void* workspace, size_t workspace_bytes) {
    constexpr int kB1 = F == Fmt::kMx8 ? 64 : 128;
    constexpr int kB2 = 2 * kB1;
    using Sk = Cfg<128, kB1, 3, true, true, true, F>;
    const int M = p.M;
    const int N = p.N;
    const int K = p.K;
    if (M <= 0 || N <= 0 || K <= 0 || N % 8 != 0 || !p.a || !p.a_sf || !p.w || !p.w_sf || !p.d)
        return false;
    const int sms = sm_count();
    const int units = ((N + 127) / 128) * ((M + kBM - 1) / kBM) * (K / Sk::kBK);
    const bool sk_fits = K % Sk::kBK == 0 && workspace != nullptr && sms <= kMaxSkCtas &&
                         workspace_bytes >= static_cast<size_t>(sms) * kSkPartBytes;
    p.sk_parts = sk_fits ? static_cast<float*>(workspace) : nullptr;
    p.sk_per = (units + sms - 1) / sms;
    if (variant < 0)
        variant = pick_variant(M, N, sms, sk_fits);
    switch (variant) {
        case 1:
            if (launch_tile<Cfg<256, kB1, 3, true, false, false, F>>(p, stream))
                return true;
            break;
        case 2:
            return launch_tile<Cfg<128, kB2, 2, false, false, false, F>>(p, stream);
        case 3:
            if (sk_fits && launch_tile<Sk>(p, stream))
                return true;
            [[fallthrough]];
        default:
            if (launch_tile<Cfg<128, kB1, 4, true, true, false, F>>(p, stream))
                return true;
            break;
    }
    // no TMA map (alignment): sync mainloop
    return launch_tile<Cfg<128, kB2, 2, false, false, false, F>>(p, stream);
}

}  // namespace

bool gemm_nvfp4_tile(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf, float alpha,
                     void* d_fp16, int M, int N, int K, cudaStream_t stream, int variant, void* workspace,
                     size_t workspace_bytes) {
    const TileArgs p{static_cast<const uint8_t*>(a_data),
                     static_cast<const uint8_t*>(a_sf),
                     static_cast<const uint8_t*>(w_data),
                     static_cast<const uint8_t*>(w_sf),
                     static_cast<half*>(d_fp16),
                     alpha,
                     M,
                     N,
                     K,
                     nullptr,
                     0,
                     nullptr};
    return run_tile<Fmt::kNv4>(p, stream, variant, workspace, workspace_bytes);
}

bool gemm_mxfp8_tile(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf,
                     const float* col_scale, void* d_fp16, int M, int N, int K, cudaStream_t stream,
                     int variant, void* workspace, size_t workspace_bytes) {
    const TileArgs p{static_cast<const uint8_t*>(a_data),
                     static_cast<const uint8_t*>(a_sf),
                     static_cast<const uint8_t*>(w_data),
                     static_cast<const uint8_t*>(w_sf),
                     static_cast<half*>(d_fp16),
                     1.0f,
                     M,
                     N,
                     K,
                     nullptr,
                     0,
                     col_scale};
    return run_tile<Fmt::kMx8>(p, stream, variant, workspace, workspace_bytes);
}

bool gemm_mxfp4_tile(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf, float alpha,
                     void* d_fp16, int M, int N, int K, cudaStream_t stream, int variant, void* workspace,
                     size_t workspace_bytes) {
    const TileArgs p{static_cast<const uint8_t*>(a_data),
                     static_cast<const uint8_t*>(a_sf),
                     static_cast<const uint8_t*>(w_data),
                     static_cast<const uint8_t*>(w_sf),
                     static_cast<half*>(d_fp16),
                     alpha,
                     M,
                     N,
                     K,
                     nullptr,
                     0,
                     nullptr};
    return run_tile<Fmt::kMx4>(p, stream, variant, workspace, workspace_bytes);
}

size_t gemm_tile_streamk_workspace_bytes() { return static_cast<size_t>(sm_count()) * kSkPartBytes; }

// Shared with gemm_f16_tile.cu (same TMA operand maps, 2-byte elements counted as bytes).
bool nvfp4_tile::encode_2d(CUtensorMap* map, const void* base, int rows, int row_bytes, int box_bytes,
                           int box_rows) {
    return encode_operand(map, base, rows, row_bytes, box_bytes, box_rows);
}

}  // namespace imp
