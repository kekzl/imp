// Own grouped NVFP4 MoE prefill GEMM (ROADMAP row 123) on the CUTLASS 3.x grouped operands
// (GroupedNvfp4DeviceArgs): per expert e, D_e[M_e, N] = alpha_e * A_e B_e^T. Persistent CTAs walk a
// device-built tile list (expert by binary search on the per-expert tile prefix). A: one TMA map over
// the expert-sorted activations (rows of the next expert in a tile's tail compute but never store).
// B: a per-CTA tensormap in global memory, re-pointed with tensormap.replace on an expert change
// (two slots: a slot is rewritten only after its loads were consumed, K/128 >= stages).

#include "compute/gemm_grouped_nvfp4_tile.h"

#include "core/logging.h"
#include "core/process_diag.h"
#include "quant/nvfp4_gemm_tile_core.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

using namespace nvfp4_tile;  // NOLINT(google-build-using-namespace): the shared tile kernel parts

using C = Cfg<128, 128, 3, true, true>;  // 128x128, producer warp, 3 stages + own staging buffer
constexpr int kStageArea = C::kStages * C::kStageBytes;
constexpr int kStagingBytes = kBM * C::kOS * 2;

constexpr int kMaxExperts = 512;
constexpr size_t kPrefixBytes = (kMaxExperts + 1) * sizeof(int);  // smem tile prefix
constexpr int kSmemBytes = kStageArea + 128 + kStagingBytes + static_cast<int>(kPrefixBytes) + C::kAlign;
constexpr size_t kMapBytes = 128;
constexpr int kMaxCtas = 1024;
// Two B tensormap slots per CTA (one launch at a time, like the CUTLASS grouped static workspace).
constexpr size_t kMapArea = static_cast<size_t>(kMaxCtas) * 2 * kMapBytes;
__device__ __align__(128) uint8_t g_maps[kMapArea];

struct GArgs {
    GroupedNvfp4DeviceArgs a;
    int ne, N, K;
};

__device__ __forceinline__ int find_expert(const int* prefix, int ne, int t) {
    int lo = 0, hi = ne;  // largest e with prefix[e] <= t
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (prefix[mid] <= t)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__device__ __forceinline__ const void* b_ptr(const GroupedNvfp4DeviceArgs& a, int e) {
    return a.d_B_ptrs ? a.d_B_ptrs[e]
                      : static_cast<const uint8_t*>(a.base_B_packed) +
                            (static_cast<int64_t>(e) * a.b_expert_stride_packed);
}
__device__ __forceinline__ const uint8_t* sfb_ptr(const GroupedNvfp4DeviceArgs& a, int e) {
    return static_cast<const uint8_t*>(a.d_SFB_ptrs ? a.d_SFB_ptrs[e]
                                                    : static_cast<const uint8_t*>(a.base_B_sf) +
                                                          (static_cast<int64_t>(e) * a.b_expert_stride_sf));
}

// Copy the B map template into this CTA's slot and point it at expert e's weights.
__device__ __forceinline__ const CUtensorMap* rebind_b(uint8_t* slot, const CUtensorMap& tmpl,
                                                       const void* base) {
    const uint64_t* src = reinterpret_cast<const uint64_t*>(&tmpl);
    uint64_t* dst = reinterpret_cast<uint64_t*>(slot);
#pragma unroll
    for (int i = 0; i < 16; ++i)
        dst[i] = src[i];
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;" ::"l"(slot),
                 "l"(reinterpret_cast<uint64_t>(base))
                 : "memory");
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(slot) : "memory");
    return reinterpret_cast<const CUtensorMap*>(slot);
}

struct Tile {
    int e, m0, n0, rows;
};

__device__ __forceinline__ Tile tile_at(const GArgs& g, const int* prefix, int t, int n_tiles) {
    const int e = find_expert(prefix, g.ne, t);
    const int local = t - prefix[e];
    return {e, (local / n_tiles) * kBM, (local % n_tiles) * C::kBN, g.a.d_M_per[e]};
}

struct Ctx {
    uint8_t* smem;
    uint32_t s0, bars;
    half* so;  // epilogue staging, separate from the ring (the producer refills it meanwhile)
    int* sp;   // [ne + 1] tile prefix
    int KT, n_tiles, total;
    size_t sf_slab;
};

// Warp 0: exclusive scan of tiles per expert, 32 experts per step.
__device__ __forceinline__ void tile_prefix_scan(const GArgs& g, int* sp, int n_tiles, int lane) {
    int carry = 0;
    for (int base = 0; base < g.ne; base += 32) {
        const int e = base + lane;
        const int v = e < g.ne ? ((g.a.d_M_per[e] + kBM - 1) / kBM) * n_tiles : 0;
        int inc = v;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d)
                inc += u;
        }
        if (e < g.ne)
            sp[e] = carry + inc - v;
        carry += __shfl_sync(0xffffffffu, inc, 31);
    }
    if (lane == 0)
        sp[g.ne] = carry;
}

// Producer lane: every tile of this CTA, the B map re-pointed on an expert change.
__device__ __forceinline__ void produce(const GArgs& g, const Ctx& x, const CUtensorMap& map_a,
                                        const CUtensorMap& map_b) {
    int ring = 0, cur = -1, slot = 0;
    const CUtensorMap* mb = nullptr;
    for (int t = static_cast<int>(blockIdx.x); t < x.total; t += static_cast<int>(gridDim.x)) {
        const Tile tl = tile_at(g, x.sp, t, x.n_tiles);
        if (tl.e != cur) {
            slot ^= 1;
            mb = rebind_b(g_maps + ((static_cast<size_t>(blockIdx.x) * 2 + slot) * kMapBytes), map_b,
                          b_ptr(g.a, tl.e));
            cur = tl.e;
        }
        const uint8_t* sfa = static_cast<const uint8_t*>(g.a.base_A_sf) + g.a.d_sfa_offsets[tl.e] +
                             (static_cast<size_t>(tl.m0 / kBM) * x.sf_slab);
        const uint8_t* sfb = sfb_ptr(g.a, tl.e) + (static_cast<size_t>(tl.n0 / 128) * x.sf_slab);
        const int row0 = g.a.d_expert_offsets[tl.e] + tl.m0;
        for (int kt = 0; kt < x.KT; ++kt, ++ring) {
            const int s = ring % C::kStages;
            const uint32_t full = x.bars + s * 8;
            const uint32_t sb = x.s0 + s * C::kStageBytes;
            if (ring >= C::kStages)
                mbar_wait(x.bars + (C::kStages + s) * 8, ((ring / C::kStages) - 1) & 1);
            // K % 128 == 64: the last stage holds one 64-K SfAtom per operand; TMA zero-fills the data past
            // K, the second SF atom is zeroed here (a never-written slot could read as a NaN scale).
            const bool half_k = kt == x.KT - 1 && (g.K % C::kBK) != 0;
            const uint32_t sf_bytes = half_k ? C::kSfBytes / 2 : C::kSfBytes;
            if (half_k) {
                uint4* z0 = reinterpret_cast<uint4*>(x.smem + (static_cast<size_t>(s) * C::kStageBytes) +
                                                     C::kSfA + sf_bytes);
                uint4* z1 = reinterpret_cast<uint4*>(x.smem + (static_cast<size_t>(s) * C::kStageBytes) +
                                                     C::kSfW + sf_bytes);
                for (uint32_t i = 0; i < sf_bytes / 16; ++i) {
                    z0[i] = make_uint4(0, 0, 0, 0);
                    z1[i] = make_uint4(0, 0, 0, 0);
                }
            }
            mbar_expect_tx(full, C::kTxBytes - (2 * (C::kSfBytes - sf_bytes)));
            tma_2d(sb, &map_a, kt * (C::kBK / 2), row0, full);
            tma_2d(sb + C::kATile, mb, kt * (C::kBK / 2), tl.n0, full);
            bulk_1d(sb + C::kSfA, sfa + (static_cast<size_t>(kt) * C::kSfBytes), sf_bytes, full);
            bulk_1d(sb + C::kSfW, sfb + (static_cast<size_t>(kt) * C::kSfBytes), sf_bytes, full);
        }
    }
}

// alpha_e, fp16, staged in so, 16 B row stores of the expert's valid rows.
__device__ __forceinline__ void store_tile(const float (&acc)[4][C::kNF][4], const GArgs& g, const Ctx& x,
                                           const Tile& tl, int wm, int wn, int lane) {
    const float alpha = g.a.d_alpha[tl.e];
    const int T0 = lane & 3;
    const int T1 = lane >> 2;
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));  // the last tile's store loop left `so`
#pragma unroll
    for (int mf = 0; mf < 4; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = wm + mf * 16 + T1 + h * 8;
                const int col = wn + nf * 8 + T0 * 2;
                *reinterpret_cast<__half2*>(x.so + (static_cast<size_t>(row) * C::kOS) +
                                            col) = __floats2half2_rn(acc[mf][nf][h ? 2 : 0] * alpha,
                                                                     acc[mf][nf][h ? 3 : 1] * alpha);
            }
    asm volatile("bar.sync 1, %0;" ::"n"(kMmaThreads));
    half* d = static_cast<half*>(g.a.base_D) + (static_cast<size_t>(g.a.d_expert_offsets[tl.e]) * g.N);
    for (int i = static_cast<int>(threadIdx.x); i < kBM * (C::kBN / 8); i += kMmaThreads) {
        const int r = i / (C::kBN / 8);
        const int c = (i % (C::kBN / 8)) * 8;
        if (tl.m0 + r < tl.rows && tl.n0 + c < g.N)
            *reinterpret_cast<uint4*>(d + (static_cast<size_t>(tl.m0 + r) * g.N) + tl.n0 + c) =
                *reinterpret_cast<const uint4*>(x.so + (static_cast<size_t>(r) * C::kOS) + c);
    }
}

__device__ __forceinline__ void consume(const GArgs& g, const Ctx& x, int warp, int lane) {
    const int wm = (warp & 1) * 64;
    const int wn = (warp >> 1) * C::kWN;
    int ring = 0;
    for (int t = static_cast<int>(blockIdx.x); t < x.total; t += static_cast<int>(gridDim.x)) {
        const Tile tl = tile_at(g, x.sp, t, x.n_tiles);
        float acc[4][C::kNF][4] = {};
        for (int kt = 0; kt < x.KT; ++kt, ++ring) {
            const int s = ring % C::kStages;
            mbar_wait(x.bars + s * 8, (ring / C::kStages) & 1);
            mma_stage<C>(acc, x.s0 + s * C::kStageBytes, x.smem + (static_cast<size_t>(s) * C::kStageBytes),
                         wm, wn, lane);
            __syncwarp();
            if (lane == 0)
                mbar_arrive(x.bars + (C::kStages + s) * 8);
        }
        store_tile(acc, g, x, tl, wm, wn, lane);
    }
}

__global__ void __launch_bounds__(C::kThreads, 1) gemm_grouped_nvfp4_tile_kernel(
    GArgs g, const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + C::kAlign - 1) & ~uint32_t(C::kAlign - 1)) - raw;
    Ctx x{};
    x.smem = smem_raw + pad;
    x.s0 = raw + pad;
    x.bars = x.s0 + kStageArea;
    x.so = reinterpret_cast<half*>(x.smem + kStageArea + 128);
    x.sp = reinterpret_cast<int*>(x.smem + kStageArea + 128 + kStagingBytes);
    x.KT = (g.K + C::kBK - 1) / C::kBK;  // K % 128 == 64: a half last stage
    x.n_tiles = (g.N + C::kBN - 1) / C::kBN;
    x.sf_slab = static_cast<size_t>(g.K / 64) * 512;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (threadIdx.x == 0) {
        for (int s = 0; s < C::kStages; ++s) {
            mbar_init(x.bars + s * 8, 1);
            mbar_init(x.bars + (C::kStages + s) * 8, kMmaThreads / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0)
        tile_prefix_scan(g, x.sp, x.n_tiles, lane);
    __syncthreads();
    x.total = x.sp[g.ne];
    if (warp == kMmaThreads / 32) {
        if (lane == 0)
            produce(g, x, map_a, map_b);
        return;
    }
    consume(g, x, warp, lane);
}

}  // namespace

bool gemm_grouped_nvfp4_tile(int n_experts, int N, int K, int expanded, const GroupedNvfp4DeviceArgs& args,
                             cudaStream_t stream) {
    static const int sms = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            return 0;
        return n;
    }();
    static const bool smem_ok = cudaFuncSetAttribute(gemm_grouped_nvfp4_tile_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     kSmemBytes) == cudaSuccess;
    if (!smem_ok || sms <= 0 || sms > kMaxCtas || n_experts <= 0 || n_experts > kMaxExperts ||
        expanded <= 0 || N % 8 != 0 || K % 64 != 0 || K / C::kBK < C::kStages ||
        args.base_A_packed == nullptr || args.base_A_sf == nullptr || args.base_D == nullptr)
        return false;
    CUtensorMap map_a{};
    CUtensorMap map_b{};  // template: rows N, the base is replaced per expert in the kernel
    if (!encode_2d(&map_a, args.base_A_packed, expanded, K / 2, C::kBK / 2, kBM) ||
        !encode_2d(&map_b, args.base_A_packed, N, K / 2, C::kBK / 2, C::kBN))
        return false;
    const GArgs g{args, n_experts, N, K};
    gemm_grouped_nvfp4_tile_kernel<<<sms, C::kThreads, kSmemBytes, stream>>>(g, map_a, map_b);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

bool gemm_grouped_nvfp4_dispatch(int n_experts, int N, int K, int expanded,
                                 const GroupedNvfp4DeviceArgs& args, cudaStream_t stream) {
    if (process_diag_moe_tile() && gemm_grouped_nvfp4_tile(n_experts, N, K, expanded, args, stream)) {
        static bool logged = false;  // activity proof, once
        if (!logged) {
            logged = true;
            IMP_LOG_INFO("gemm.moe_tile: own grouped NVFP4 GEMM ACTIVE (experts=%d N=%d K=%d)", n_experts, N,
                         K);
        }
        return true;
    }
    return gemm_grouped_cutlass_3x_nvfp4_device_args(n_experts, N, K, args, stream);
}

}  // namespace imp
