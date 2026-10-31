// Own grouped NVFP4 MoE prefill GEMM (ROADMAP row 123) on the grouped operands
// (GroupedNvfp4DeviceArgs): per expert e, D_e[M_e, N] = alpha_e * A_e B_e^T. Persistent CTAs walk a
// device-built tile list (expert by binary search on the per-expert tile prefix). A: one TMA map over
// the expert-sorted activations (rows of the next expert in a tile's tail compute but never store).
// B: a per-CTA tensormap in global memory, re-pointed with tensormap.replace on an expert change
// (kMapSlots slots: a slot is rewritten after (kMapSlots - 1) tiles of >= 1 K stage each, so its last load
// was consumed before the producer passed the stage ring: (kMapSlots - 1) >= kStages).

#include "compute/gemm_grouped_nvfp4_tile.h"

#include "core/logging.h"
#include "memory/engine_arena.h"
#include "quant/nvfp4_gemm_tile_core.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

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
// B tensormap slots per CTA (one launch at a time): 4 > 3 stages covers one K stage per tile (K 64..128).
constexpr int kMapSlots = 4;
static_assert(kMapSlots - 1 >= C::kStages, "a slot must outlive the stage ring");
constexpr size_t kMapArea = static_cast<size_t>(kMaxCtas) * kMapSlots * kMapBytes;
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
            slot = (slot + 1) % kMapSlots;
            mb = rebind_b(g_maps + ((static_cast<size_t>(blockIdx.x) * kMapSlots + slot) * kMapBytes), map_b,
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
                const int col = C::ncol(wn, nf) + T0 * 2;
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

namespace {

int sm_count() {
    static const int sms = [] {
        int dev = 0, n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
            return 0;
        return n;
    }();
    return sms;
}

// Host-args staging, one T2 slice per arena generation (a model swap re-takes it).
void* s_staging = nullptr;
uint64_t s_staging_gen = ~0ull;

void* staging() {
    if (engine_arena().generation() != s_staging_gen)
        return nullptr;
    return s_staging;
}

size_t align128(size_t x) { return (x + 127) & ~size_t(127); }

}  // namespace

bool gemm_grouped_nvfp4_supports(int n_experts, int N, int K) {
    static const bool smem_ok = cudaFuncSetAttribute(gemm_grouped_nvfp4_tile_kernel,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     kSmemBytes) == cudaSuccess;
    const int sms = sm_count();
    return smem_ok && sms > 0 && sms <= kMaxCtas && n_experts > 0 && n_experts <= kMaxExperts && N > 0 &&
           N % 8 == 0 && K > 0 && K % 64 == 0;
}

bool gemm_grouped_nvfp4_tile(int n_experts, int N, int K, int expanded, const GroupedNvfp4DeviceArgs& args,
                             cudaStream_t stream) {
    if (!gemm_grouped_nvfp4_supports(n_experts, N, K) || expanded <= 0 || args.base_A_packed == nullptr ||
        args.base_A_sf == nullptr || args.base_D == nullptr)
        return false;
    CUtensorMap map_a{};
    CUtensorMap map_b{};  // template: rows N, the base is replaced per expert in the kernel
    if (!encode_2d(&map_a, args.base_A_packed, expanded, K / 2, C::kBK / 2, kBM) ||
        !encode_2d(&map_b, args.base_A_packed, N, K / 2, C::kBK / 2, C::kBN))
        return false;
    const GArgs g{args, n_experts, N, K};
    gemm_grouped_nvfp4_tile_kernel<<<sm_count(), C::kThreads, kSmemBytes, stream>>>(g, map_a, map_b);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

bool gemm_grouped_nvfp4_dispatch(int n_experts, int N, int K, int expanded,
                                 const GroupedNvfp4DeviceArgs& args, cudaStream_t stream) {
    if (!gemm_grouped_nvfp4_tile(n_experts, N, K, expanded, args, stream))
        return false;
    static bool logged = false;  // activity proof, once
    if (!logged) {
        logged = true;
        IMP_LOG_INFO("MoE prefill: own grouped NVFP4 GEMM ACTIVE (experts=%d N=%d K=%d)", n_experts, N, K);
    }
    return true;
}

bool gemm_grouped_nvfp4_host_args(int n_experts, int N, int K, int expanded, const GroupedNvfp4HostArgs& args,
                                  cudaStream_t stream) {
    if (!gemm_grouped_nvfp4_supports(n_experts, N, K) || args.offsets[n_experts] > expanded)
        return false;
    char* d_base = static_cast<char*>(staging());
    if (d_base == nullptr) {
        IMP_LOG_ERROR("grouped NVFP4 host-args: no staging slice (gemm_grouped_nvfp4_prewarm not run)");
        return false;
    }
    const size_t ne = static_cast<size_t>(n_experts);
    const size_t o_m = 0;
    const size_t o_off = align128(o_m + (ne * sizeof(int32_t)));
    const size_t o_sfa = align128(o_off + ((ne + 1) * sizeof(int32_t)));
    const size_t o_alpha = align128(o_sfa + ((ne + 1) * sizeof(int64_t)));
    const size_t o_b = align128(o_alpha + (ne * sizeof(float)));
    const size_t o_sfb = align128(o_b + (ne * sizeof(void*)));
    const size_t total = o_sfb + (ne * sizeof(void*));
    if (total > kGroupedNvfp4StagingBytes)
        return false;
    std::vector<char> h(total, 0);
    auto* m_per = reinterpret_cast<int32_t*>(h.data() + o_m);
    auto* offs = reinterpret_cast<int32_t*>(h.data() + o_off);
    auto* sfa = reinterpret_cast<int64_t*>(h.data() + o_sfa);
    auto* alpha = reinterpret_cast<float*>(h.data() + o_alpha);
    auto* b = reinterpret_cast<const void**>(h.data() + o_b);
    auto* sfb = reinterpret_cast<const void**>(h.data() + o_sfb);
    for (size_t e = 0; e < ne; ++e) {
        m_per[e] = args.offsets[e + 1] - args.offsets[e];
        offs[e] = args.offsets[e];
        sfa[e] = static_cast<int64_t>(args.sfa_offsets[e]);
        alpha[e] = args.alpha[e];
        b[e] = args.B[e];
        sfb[e] = args.SFB[e];
    }
    offs[ne] = args.offsets[ne];
    sfa[ne] = sfa[ne - 1];
    // Pageable source: the call returns once the bytes are staged, so `h` may go out of scope.
    if (cudaMemcpyAsync(d_base, h.data(), total, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return false;
    GroupedNvfp4DeviceArgs d{};
    d.d_M_per = reinterpret_cast<const int32_t*>(d_base + o_m);
    d.d_expert_offsets = reinterpret_cast<const int32_t*>(d_base + o_off);
    d.d_sfa_offsets = reinterpret_cast<const int64_t*>(d_base + o_sfa);
    d.d_alpha = reinterpret_cast<const float*>(d_base + o_alpha);
    d.d_B_ptrs = reinterpret_cast<const void* const*>(d_base + o_b);
    d.d_SFB_ptrs = reinterpret_cast<const void* const*>(d_base + o_sfb);
    d.base_A_packed = args.base_A_packed;
    d.base_A_sf = args.base_A_sf;
    d.base_D = args.base_D;
    return gemm_grouped_nvfp4_dispatch(n_experts, N, K, expanded, d, stream);
}

void gemm_grouped_nvfp4_prewarm() {
    const uint64_t gen = engine_arena().generation();
    if (s_staging != nullptr && s_staging_gen == gen)
        return;
    auto slab = engine_arena().take_bytes(kGroupedNvfp4StagingBytes);
    if (slab.empty()) {
        IMP_LOG_ERROR(
            "grouped NVFP4: staging (%zu KiB) unavailable from the T2 arena, host-args MoE prefill "
            "will fail",
            kGroupedNvfp4StagingBytes >> 10);
        s_staging = nullptr;
        return;
    }
    s_staging = slab.data();
    s_staging_gen = gen;
}

}  // namespace imp
