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
// MXFP4 bytes (scale_vec::2X ue8m0, one 128-K SfAtom per stage); gemm_fp8_tile: per-tensor E4M3, MXFP8
// geometry without SF slabs, the block-scaled MMA with UE8M0 1.0, tensor scales and beta in the epilogue.

#include "quant/nvfp4_gemm_tile.h"
#include "quant/nvfp4_gemm_tile_core.cuh"
#include "quant/nvfp4_gemm_tile_epilogue.cuh"
#include "quant/nvfp4_gemm_tile_sk.cuh"
#include "quant/nvfp4_gemm_tile_plan.h"

#include "core/logging.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

using namespace nvfp4_tile;  // NOLINT(google-build-using-namespace): this TU's own PTX helpers

// Sync mainloop stage load: 16 B cp.async chunks over the 256 MMA threads (rows >= M/N zero-fill).
template <class C>
__device__ __forceinline__ void load_stage(uint32_t s_base, const TileArgs& p, int m0, int n0, int kt) {
    constexpr int kData = kBM * C::kChunks;
    constexpr int kSf = C::kSfBytes / 16;
    const size_t row_bytes = C::row_bytes(p.K);
    const size_t k_byte = static_cast<size_t>(kt) * C::kRB;
    const size_t sf_k = C::sf_k(kt);
    const size_t sf_slab = C::sf_slab(p.K);
    for (int i = threadIdx.x; i < 2 * kData + 2 * kSf; i += C::kMmaThr) {
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

// MMA warps' consumer loop: wait full[s], compute, release empty[s]. With REFILL, thread 0
// also refills the PREVIOUS stage's slot: the 8 warps left it while thread 0 computed this one,
// so the empty wait rarely blocks (refilling this slot waited on the slowest warp: 128x128 +15 %).
template <class C, bool REFILL>
__device__ __forceinline__ void tma_consume(float (&acc)[4][C::kNF][4], const Ring& g, const TileArgs& p,
                                            const CUtensorMap* ma, const CUtensorMap* mw) {
    // A fragments + scales one K step ahead, across the stage boundary: the next step's loads issue behind
    // this step's MMAs. The slot is released once its last MMAs issued (their operands have landed).
    constexpr int kSteps = C::kRB / 32;
    static_assert(kSteps % 2 == 0, "double buffer: step 0 of every stage in f[0]");
    AFrag<C> f[2];
    mbar_wait_sleep(g.bars, 0);
    load_a<C>(f[0], g.s0, g.smem, g.wm, g.wn, g.lane, 0, C::sf_g0(0), g.n0 & 127);
    for (int kt = 0; kt < g.KT; ++kt) {
        const int s = kt % C::kStages;
        const uint32_t sa = g.s0 + s * C::kStageBytes;
#pragma unroll
        for (int c = 0; c < kSteps; ++c) {
            mma_from<C>(acc, f[c & 1], sa, g.wn, g.lane, c);
            if (c + 1 < kSteps) {
                load_a<C>(f[(c + 1) & 1], sa, g.smem + s * C::kStageBytes, g.wm, g.wn, g.lane, c + 1,
                          C::sf_g0(kt), g.n0 & 127);
            } else {
                __syncwarp();
                if (g.lane == 0)
                    mbar_arrive(g.bars + (C::kStages + s) * 8);
                if (kt + 1 < g.KT) {
                    const int s1 = (kt + 1) % C::kStages;
                    mbar_wait_sleep(g.bars + s1 * 8, ((kt + 1) / C::kStages) & 1);
                    load_a<C>(f[0], g.s0 + s1 * C::kStageBytes, g.smem + s1 * C::kStageBytes, g.wm, g.wn,
                              g.lane, 0, C::sf_g0(kt + 1), g.n0 & 127);
                }
            }
        }
        const int kp = kt - 1;
        if (REFILL && threadIdx.x == 0 && kp >= 0 && kp + C::kStages < g.KT) {
            const int sp = kp % C::kStages;
            mbar_wait_sleep(g.bars + (C::kStages + sp) * 8, (kp / C::kStages) & 1);
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
            mbar_init(g.bars + (C::kStages + s) * 8, C::kMmaThr / 32);
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
        if (g.warp != C::kMmaThr / 32) {
            tma_consume<C, false>(acc, g, p, ma, mw);
            return true;
        }
        if (g.lane == 0)
            for (int kt = 0; kt < g.KT; ++kt) {
                const int s = kt % C::kStages;
                if (kt >= C::kStages)
                    mbar_wait_sleep(g.bars + (C::kStages + s) * 8, ((kt / C::kStages) - 1) & 1);
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

template <class C>
__global__ void __launch_bounds__(C::kThreads, C::kMinB) gemm_nvfp4_tile_kernel(
    TileArgs p, const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_w,
    const __grid_constant__ CUtensorMap map_d) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + C::kAlign - 1) & ~uint32_t(C::kAlign - 1)) - raw;
    const int warp = threadIdx.x >> 5;
    const Ring g{smem_raw + pad,
                 raw + pad,
                 raw + pad + C::kStages * C::kStageBytes,
                 static_cast<int>(C::kSk ? blockIdx.y : blockIdx.x) * kBM,
                 static_cast<int>(C::kSk ? blockIdx.x : blockIdx.y) * C::kBN,
                 (p.K + C::kBK - 1) / C::kBK,  // kPartK: a partial last stage
                 warp,
                 static_cast<int>(threadIdx.x & 31),
                 (warp & 1) * 64,
                 ((warp >> 1) % (C::kMW / 2)) * C::kWN};

    if constexpr (C::kSk) {
        sk_mainloop<C>(g, p, &map_a, &map_w, &map_d);
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
        if constexpr (C::kTma)
            store_tma<C>(acc, g, p, &map_d);
        else
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

// [M, N] fp16 output with row stride ldd, box [64 columns, 128 rows] = 128 B rows, 128B swizzle (store_tma).
bool encode_out(CUtensorMap* map, void* base, int M, int N, int ldd) {
    const EncodeTiledFn fn = encode_fn();
    if (fn == nullptr)
        return false;
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(N), static_cast<cuuint64_t>(M)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(ldd) * sizeof(half)};
    const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(kBM)};
    const cuuint32_t elem[2] = {1, 1};
    return fn(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, base, dims, strides, box, elem,
              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <class C>
bool launch_tile(const TileArgs& p, cudaStream_t stream) {
    if (p.K % C::kBK != 0 && !(C::kPartK && p.K % 64 == 0))
        return false;
    CUtensorMap map_a{};
    CUtensorMap map_w{};
    CUtensorMap map_d{};
    if constexpr (C::kTma) {
        if (reinterpret_cast<uintptr_t>(p.a) % 16 != 0 || reinterpret_cast<uintptr_t>(p.w) % 16 != 0 ||
            !encode_operand(&map_a, p.a, p.M, static_cast<int>(C::row_bytes(p.K)), C::kRB, kBM) ||
            !encode_operand(&map_w, p.w, p.N, static_cast<int>(C::row_bytes(p.K)), C::kRB, C::kBN))
            return false;
        if constexpr (!C::kSk || C::kSkStaged)
            if (p.d32 == nullptr &&
                (reinterpret_cast<uintptr_t>(p.d) % 16 != 0 || !encode_out(&map_d, p.d, p.M, p.N, p.ldd)))
                return false;
    }
    // MINB > 1: the whole carveout as shared memory, or the driver may plan one CTA per SM.
    static const bool smem_ok = cudaFuncSetAttribute(gemm_nvfp4_tile_kernel<C>,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     C::kSmemBytes) == cudaSuccess &&
                                (C::kMinB == 1 ||
                                 cudaFuncSetAttribute(gemm_nvfp4_tile_kernel<C>,
                                                      cudaFuncAttributePreferredSharedMemoryCarveout,
                                                      cudaSharedmemCarveoutMaxShared) == cudaSuccess);
    if (!smem_ok)
        return false;
    // Data-parallel tiles walk M fastest: neighbouring CTAs share a W tile
    // (128x128 at M=512 N=17408: 115 -> 105 us).
    dim3 grid = C::kSk ? dim3((p.N + C::kBN - 1) / C::kBN, (p.M + kBM - 1) / kBM)
                       : dim3((p.M + kBM - 1) / kBM, (p.N + C::kBN - 1) / C::kBN);
    if constexpr (C::kSk) {  // caller set the plan and the workspace; hybrid: grid = sk_per
        const int units = static_cast<int>(grid.x * grid.y) * ((p.K + C::kBK - 1) / C::kBK);
        grid = dim3(p.sk_split > 0 ? p.sk_per : (units + p.sk_per - 1) / p.sk_per);
    }
    gemm_nvfp4_tile_kernel<C><<<grid, C::kThreads, C::kSmemBytes, stream>>>(p, map_a, map_w, map_d);
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

// Shape and operands every config needs: SF slabs for the block-scaled formats, scale scalars for FP8.
template <Fmt F>
bool args_ok(const TileArgs& p) {
    if (p.M <= 0 || p.N <= 0 || p.K <= 0 || p.N % 8 != 0 || !p.a || !p.w || (!p.d && !p.d32))
        return false;
    if constexpr (F == Fmt::kF8)
        return p.a_scale != nullptr && p.w_scale != nullptr;
    return p.a_sf != nullptr && p.w_sf != nullptr;
}

// FP8 128x64 on 4 MMA warps + producer, 3 stages, 2 CTAs per SM (2 x SMs slots): 7 uniform stream-K, 8 hybrid
// (else 7), 9 data-parallel; stream-K falls back to 9.
template <Fmt F>
bool launch_two_cta(const TileArgs& p, int variant, bool sk_fits, int sms, int kt, cudaStream_t stream) {
    using Sk2 = Cfg<64, 64, 3, true, true, true, F, 4, 2>;
    if (sk_fits && variant != 9) {
        TileArgs h = p;
        if (variant == 8 && hybrid_plan(h, 2 * sms, kt, 64) && launch_tile<Sk2>(h, stream))
            return true;
        h = p;
        const int units = ((p.N + 63) / 64) * ((p.M + kBM - 1) / kBM) * kt;
        h.sk_per = (units + (2 * sms) - 1) / (2 * sms);
        if (launch_tile<Sk2>(h, stream))
            return true;
    }
    return launch_tile<Cfg<64, 64, 3, true, true, false, F, 4, 2>>(p, stream);
}

// One variant (see the header), falling through to the next simpler plan; false = every TMA config refused.
template <Fmt F>
bool launch_variant(const TileArgs& p, int variant, bool sk_fits, int sms, cudaStream_t stream) {
    constexpr int kB1 = F == Fmt::kMx8 || F == Fmt::kF8 ? 64 : 128;
    using Sk = Cfg<128, kB1, 3, true, true, true, F>;
    const int kt = (p.K + Sk::kBK - 1) / Sk::kBK;
    if constexpr (F == Fmt::kF8)
        if (variant >= 7 && variant <= 9 && launch_two_cta<F>(p, variant, sk_fits, sms, kt, stream))
            return true;
    switch (variant) {
        case 1:
            if (launch_tile<Cfg<256, kB1, 3, true, false, false, F>>(p, stream))
                return true;
            break;
        case 2:
            return launch_tile<Cfg<128, 2 * kB1, 2, false, false, false, F>>(p, stream);
        case 6: {  // 128x64, every tile split in 2 k-pieces
            TileArgs h = p;
            if (sk_fits && hybrid_plan(h, sms, kt, 64) &&
                launch_tile<Cfg<64, kB1, 4, true, true, true, F>>(h, stream))
                return true;
            [[fallthrough]];
        }
        case 5:
            if (launch_tile<Cfg<64, kB1, 4, true, true, false, F>>(p, stream))
                return true;
            break;
        case 4: {
            TileArgs h = p;
            if (sk_fits && hybrid_plan(h, sms, kt) && launch_tile<Sk>(h, stream))
                return true;
            [[fallthrough]];
        }
        case 3:
            if (sk_fits && launch_tile<Sk>(p, stream))
                return true;
            [[fallthrough]];
        default:
            if (launch_tile<Cfg<128, kB1, 4, true, true, false, F>>(p, stream))
                return true;
            break;
    }
    return false;
}

// All formats run the same byte geometry: an MXFP8 stage of BK elements moves what an NVFP4 / MXFP4 stage
// of 2 x BK does, so the NVFP4 tile choices carry over with BK halved.
template <Fmt F>
bool run_tile(TileArgs p, cudaStream_t stream, int variant, void* workspace, size_t workspace_bytes) {
    constexpr int kB1 = F == Fmt::kMx8 || F == Fmt::kF8 ? 64 : 128;
    constexpr int kB2 = 2 * kB1;
    using Sk = Cfg<128, kB1, 3, true, true, true, F>;
    const int M = p.M;
    const int N = p.N;
    const int K = p.K;
    if (!args_ok<F>(p))
        return false;
    if (p.ldd == 0)
        p.ldd = N;
    const int sms = sm_count();
    const int units = ((N + 127) / 128) * ((M + kBM - 1) / kBM) * ((K + Sk::kBK - 1) / Sk::kBK);
    const bool sk_fits = (K % Sk::kBK == 0 || (Sk::kPartK && K % 64 == 0)) && workspace != nullptr &&
                         sms <= kMaxSkCtas && workspace_bytes >= static_cast<size_t>(sms) * kSkPartBytes;
    p.sk_parts = sk_fits ? static_cast<float*>(workspace) : nullptr;
    p.sk_per = (units + sms - 1) / sms;
    if (variant < 0)
        variant = pick_variant(M, N, (K + Sk::kBK - 1) / Sk::kBK, sms, sk_fits);
    if (launch_variant<F>(p, variant, sk_fits, sms, stream))
        return true;
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
                     nullptr,
                     nullptr,
                     nullptr,
                     0.0f};
    return run_tile<Fmt::kNv4>(p, stream, variant, workspace, workspace_bytes);
}

bool gemm_nvfp4_tile_f32(const void* a_data, const void* a_sf, const void* w_data, const void* w_sf,
                         float alpha, float* d_fp32, int M, int N, int K, cudaStream_t stream,
                         void* workspace, size_t workspace_bytes) {
    TileArgs p{static_cast<const uint8_t*>(a_data),
               static_cast<const uint8_t*>(a_sf),
               static_cast<const uint8_t*>(w_data),
               static_cast<const uint8_t*>(w_sf),
               nullptr,
               alpha,
               M,
               N,
               K};
    p.d32 = d_fp32;
    return run_tile<Fmt::kNv4>(p, stream, -1, workspace, workspace_bytes);
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
                     col_scale,
                     nullptr,
                     nullptr,
                     0.0f};
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
                     nullptr,
                     nullptr,
                     nullptr,
                     0.0f};
    return run_tile<Fmt::kMx4>(p, stream, variant, workspace, workspace_bytes);
}

bool gemm_fp8_tile(const void* a_data, const void* w_data, const float* a_scale, const float* w_scale,
                   float beta, void* d_fp16, int M, int N, int K, cudaStream_t stream, int variant,
                   void* workspace, size_t workspace_bytes) {
    const TileArgs p{static_cast<const uint8_t*>(a_data),
                     nullptr,
                     static_cast<const uint8_t*>(w_data),
                     nullptr,
                     static_cast<half*>(d_fp16),
                     1.0f,
                     M,
                     N,
                     K,
                     nullptr,
                     0,
                     nullptr,
                     a_scale,
                     w_scale,
                     beta};
    return run_tile<Fmt::kF8>(p, stream, variant, workspace, workspace_bytes);
}

size_t gemm_tile_streamk_workspace_bytes() { return static_cast<size_t>(sm_count()) * kSkPartBytes; }

// Shared with gemm_f16_tile.cu (same TMA operand maps, 2-byte elements counted as bytes).
bool nvfp4_tile::encode_2d(CUtensorMap* map, const void* base, int rows, int row_bytes, int box_bytes,
                           int box_rows) {
    return encode_operand(map, base, rows, row_bytes, box_bytes, box_rows);
}

}  // namespace imp
