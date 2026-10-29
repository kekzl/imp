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

#include "quant/nvfp4_gemm_tile.h"

#include "core/logging.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

constexpr int kBM = 128;
constexpr int kMmaThreads = 256;

template <int BN, int BK, int STAGES, bool TMA, bool PW = false>
struct Cfg {
    static constexpr int kBN = BN;
    static constexpr int kBK = BK;
    static constexpr int kStages = STAGES;
    static constexpr bool kTma = TMA;
    // PW: a 9th warp issues the TMA loads. 9 warps put 3 on one SM sub-partition, which caps
    // ptxas at 168 registers (128x256 spilled 140 B); without it thread 0 refills each slot.
    static constexpr bool kPw = TMA && PW;
    static constexpr int kThreads = kMmaThreads + (kPw ? 32 : 0);
    static constexpr int kWN = BN / 4;  // warp tile 64 x kWN
    static constexpr int kNF = kWN / 8;
    static constexpr int kChunks = BK / 32;  // 16 B chunks per row per stage
    // TMA: dense rows, swizzled chunks (64B / 128B mode); sync: rows padded by 16 B. Both put
    // the 8 rows of an ldmatrix on 32 distinct banks.
    static constexpr int kRS = TMA ? BK / 2 : BK / 2 + 16;
    static constexpr int kSfBytes = (BK / 64) * 512;  // SfAtoms per 128-row slab per stage
    static constexpr int kATile = kBM * kRS;
    static constexpr int kWTile = BN * kRS;
    static constexpr int kSfA = kATile + kWTile;  // stage offsets of the SF slabs
    static constexpr int kSfW = kSfA + kSfBytes;
    static constexpr int kStageBytes = kSfW + (BN / 128) * kSfBytes;
    static constexpr uint32_t kTxBytes = (kBM + BN) * (BK / 2) + (1 + BN / 128) * kSfBytes;
    static constexpr int kOS = BN + 8;   // epilogue staging stride in halves: half2 writes conflict-free
    static constexpr int kAlign = 1024;  // 128B-swizzle atom; stage and tile offsets keep it
    static constexpr int kSmemBytes = STAGES * kStageBytes + 128 + kAlign;
    static_assert(STAGES * kStageBytes >= kBM * kOS * 2, "epilogue staging must fit the ring");
    static_assert(2 * STAGES * 8 <= 128, "mbarrier area");
    static_assert(TMA || BN == 128, "sync mainloop loads one 128-row W slab");
    static_assert(!TMA || (kATile % kAlign == 0 && kStageBytes % kAlign == 0), "TMA tiles 1024-aligned");

    // Physical 16 B chunk of logical chunk c in row r (CU_TENSOR_MAP_SWIZZLE_64B / _128B).
    __device__ static __forceinline__ int chunk(int r, int c) {
        if constexpr (!TMA)
            return c;
        else if constexpr (BK == 128)
            return c ^ ((r >> 1) & 3);
        else
            return c ^ (r & 7);
    }
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "TILE_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n\t"
        "@!P bra TILE_WAIT_%=;\n\t}" ::"r"(bar),
        "r"(parity)
        : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int x, int y, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
        "[%4];" ::"r"(dst),
        "l"(map), "r"(x), "r"(y), "r"(bar)
        : "memory");
}
__device__ __forceinline__ void bulk_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
        "l"(src), "r"(bytes), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void mma_mxf4nvf4(float acc[4], const uint32_t a[4], uint32_t b0, uint32_t b1,
                                             uint32_t sfa, uint32_t sfb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, "
        "%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
          "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
#endif
}

// ldmatrix x4: lane l addresses row l%8 of 8x16 B matrix l/8; register j of every lane is the
// u32 at (row T1, byte T0*4) of matrix j, the m16n8k64 fragment order.
__device__ __forceinline__ void ldsm_x4(uint32_t r[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// Byte of row r's 4 K-group scales inside a stage's SF slabs (r < 128 per slab).
template <class C>
__device__ __forceinline__ int sf_off(int r, int c) {
    return ((r >> 7) * C::kSfBytes) + (c * 512) + ((r & 31) * 16) + (((r & 127) >> 5) * 4);
}

struct TileArgs {
    const uint8_t* a;
    const uint8_t* a_sf;
    const uint8_t* w;
    const uint8_t* w_sf;
    half* d;
    float alpha;
    int M, N, K;
};

// Sync mainloop stage load: 16 B cp.async chunks over the 256 MMA threads (rows >= M/N zero-fill).
template <class C>
__device__ __forceinline__ void load_stage(uint32_t s_base, const TileArgs& p, int m0, int n0, int kt) {
    constexpr int kData = kBM * C::kChunks;
    constexpr int kSf = C::kSfBytes / 16;
    const size_t row_bytes = static_cast<size_t>(p.K) / 2;
    const size_t k_byte = static_cast<size_t>(kt) * (C::kBK / 2);
    const size_t sf_k = static_cast<size_t>(kt) * C::kSfBytes;
    const size_t sf_slab = static_cast<size_t>(p.K / 64) * 512;
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
    const size_t sf_slab = static_cast<size_t>(p.K / 64) * 512;
    const size_t sf_k = static_cast<size_t>(kt) * C::kSfBytes;
    mbar_expect_tx(full, C::kTxBytes);
    tma_2d(s_base, ma, kt * (C::kBK / 2), m0, full);
    tma_2d(s_base + C::kATile, mw, kt * (C::kBK / 2), n0, full);
    bulk_1d(s_base + C::kSfA, p.a_sf + (static_cast<size_t>(m0 / kBM) * sf_slab) + sf_k, C::kSfBytes, full);
#pragma unroll
    for (int j = 0; j < C::kBN / 128; ++j)
        bulk_1d(s_base + C::kSfW + j * C::kSfBytes,
                p.w_sf + (static_cast<size_t>(min(n0 / 128 + j, (p.N - 1) / 128)) * sf_slab) + sf_k,
                C::kSfBytes, full);
}

// One stage of MMAs for a 64 x kWN warp tile: BK/64 k64 steps of 4 x kNF m16n8k64.
template <class C>
__device__ __forceinline__ void mma_stage(float (&acc)[4][C::kNF][4], uint32_t sa_s, const uint8_t* sst,
                                          int wm, int wn, int lane) {
    const int T1 = lane >> 2;
    const int lr = lane & 7;
    const int lm = lane >> 3;
    const int sfa_lane = ((lane & 1) * 8) + (lane >> 2);
#pragma unroll
    for (int c = 0; c < C::kBK / 64; ++c) {
        uint32_t af[4][4];
        uint32_t sfa[4];
#pragma unroll
        for (int mf = 0; mf < 4; ++mf) {
            const int r = wm + mf * 16 + lr + (lm & 1) * 8;
            ldsm_x4(af[mf], sa_s + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm >> 1)) * 16));
            sfa[mf] = *reinterpret_cast<const uint32_t*>(sst + C::kSfA +
                                                         sf_off<C>(wm + mf * 16 + sfa_lane, c));
        }
#pragma unroll
        for (int np = 0; np < C::kNF / 2; ++np) {
            uint32_t bf[4];
            const int r = wn + (np * 2 + (lm >> 1)) * 8 + lr;
            ldsm_x4(bf, sa_s + C::kATile + (r * C::kRS) + (C::chunk(r, (c * 2) + (lm & 1)) * 16));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nf = np * 2 + h;
                const uint32_t sfb = *reinterpret_cast<const uint32_t*>(sst + C::kSfW +
                                                                        sf_off<C>(wn + nf * 8 + T1, c));
#pragma unroll
                for (int mf = 0; mf < 4; ++mf)
                    mma_mxf4nvf4(acc[mf][nf], af[mf], h == 0 ? bf[0] : bf[2], h == 0 ? bf[1] : bf[3], sfa[mf],
                                 sfb);
            }
        }
    }
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
        mma_stage<C>(acc, g.s0 + s * C::kStageBytes, g.smem + s * C::kStageBytes, g.wm, g.wn, g.lane);
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
        mma_stage<C>(acc, g.s0 + s * C::kStageBytes, g.smem + s * C::kStageBytes, g.wm, g.wn, g.lane);
    }
    cp_async_wait<0>();
}

// alpha, fp16, staged in the dead ring for 16 B row-contiguous stores (256 MMA threads).
template <class C>
__device__ __forceinline__ void store_tile(const float (&acc)[4][C::kNF][4], const Ring& g,
                                           const TileArgs& p) {
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));  // every warp left the ring
    const int T0 = g.lane & 3;
    const int T1 = g.lane >> 2;
    half* so = reinterpret_cast<half*>(g.smem);
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                *reinterpret_cast<__half2*>(so + ((g.wm + mf * 16 + T1 + h * 8) * C::kOS) + g.wn + nf * 8 +
                                            T0 * 2) = __floats2half2_rn(acc[mf][nf][h * 2] * p.alpha,
                                                                        acc[mf][nf][h * 2 + 1] * p.alpha);
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
    for (int i = threadIdx.x; i < kBM * (C::kBN / 8); i += kMmaThreads) {
        const int r = i / (C::kBN / 8);
        const int c = (i % (C::kBN / 8)) * 8;
        const int row = g.m0 + r;
        const int col = g.n0 + c;
        if (row < p.M && col < p.N)
            *reinterpret_cast<uint4*>(p.d + (static_cast<size_t>(row) * p.N) +
                                      col) = *reinterpret_cast<const uint4*>(so + (r * C::kOS) + c);
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
    store_tile<C>(acc, g, p);
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
            !encode_operand(&map_a, p.a, p.M, p.K / 2, C::kBK / 2, kBM) ||
            !encode_operand(&map_w, p.w, p.N, p.K / 2, C::kBK / 2, C::kBN))
            return false;
    }
    static const bool smem_ok = cudaFuncSetAttribute(gemm_nvfp4_tile_kernel<C>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess;
    if (!smem_ok)
        return false;
    const dim3 grid((p.N + C::kBN - 1) / C::kBN, (p.M + kBM - 1) / kBM);
    gemm_nvfp4_tile_kernel<C><<<grid, C::kThreads, C::kSmemBytes, stream>>>(p, map_a, map_w);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace

bool gemm_nvfp4_tile(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf, float alpha,
                     void* d_fp16, int M, int N, int K, cudaStream_t stream, int variant) {
    if (M <= 0 || N <= 0 || K <= 0 || N % 8 != 0 || !a_data || !a_sf || !w_data || !w_sf || !d_fp16)
        return false;
    const TileArgs p{static_cast<const uint8_t*>(a_data),
                     static_cast<const uint8_t*>(a_sf),
                     static_cast<const uint8_t*>(w_data),
                     static_cast<const uint8_t*>(w_sf),
                     static_cast<half*>(d_fp16),
                     alpha,
                     M,
                     N,
                     K};
    if (variant < 0) {  // auto: 128x256 once its grid fills 3/4 of the card, else 128x128 (twice the CTAs)
        static const int sms = [] {
            int dev = 0, n = 0;
            if (cudaGetDevice(&dev) != cudaSuccess ||
                cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
                return 170;  // RTX 5090
            return n > 0 ? n : 170;
        }();
        const int ctas_256 = ((N + 255) / 256) * ((M + kBM - 1) / kBM);
        variant = 4 * ctas_256 >= 3 * sms ? 1 : 0;  // 160 CTAs: 128x256 45.4 vs 51.4 us; 80: 45.0 vs 30.7
        if (variant == 1 && launch_tile<Cfg<256, 128, 3, true>>(p, stream))
            return true;
        if (launch_tile<Cfg<128, 128, 4, true, true>>(p, stream))
            return true;
        return launch_tile<Cfg<128, 256, 2, false>>(p, stream);  // no TMA map (alignment): sync mainloop
    }
    switch (variant) {
        case 1:
            return launch_tile<Cfg<256, 128, 3, true>>(p, stream);
        case 2:
            return launch_tile<Cfg<128, 256, 2, false>>(p, stream);
        default:
            return launch_tile<Cfg<128, 128, 4, true, true>>(p, stream);
    }
}

}  // namespace imp
