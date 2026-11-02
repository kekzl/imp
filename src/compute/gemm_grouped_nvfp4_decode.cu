// Grouped NVFP4 MoE GEMM for decode rows (docs/audit/VLLM_GAP_2026_11_02.md lever 1). 16-row M tiles
// (one m16n8k64 row fragment) instead of the prefill kernel's 128, persistent CTAs over a device-built
// (expert, m tile, n tile) list, a 7-stage TMA ring fed by one producer warp, four MMA warps of 32 B rows.
// Gated epilogues run gate and up in one launch: a tile's 128 B rows are 64 gate + 64 up rows of the same
// output columns, so act(gate) * up and the NVFP4 quantize of the down input stay in registers.
// Per-row bits equal gemm_grouped_nvfp4_tile: same 64-K MMA order, fp16 rounding before the activation.

#include "compute/gemm_grouped_nvfp4_decode.h"

#include "compute/gemm_cutlass_sm120.h"

#include "core/logging.h"
#include "quant/fp8_utils.cuh"
#include "quant/nvfp4_gemm_tile_ptx.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace imp {
namespace {

using namespace nvfp4_tile;  // NOLINT(google-build-using-namespace): the shared PTX primitives

constexpr int kBM = 16;
constexpr int kBK = 128;      // K per stage
constexpr int kRB = kBK / 2;  // 64 B per row per stage, 64B swizzle
constexpr int kBR = 128;      // B rows per tile
constexpr int kWarps = 4;     // MMA warps, 32 B rows each
constexpr int kThreads = (kWarps + 1) * 32;
constexpr int kStages = 7;
constexpr int kATile = kBM * kRB;  // 1024
constexpr int kBTile = kBR * kRB;  // 8192
constexpr int kSfStage = 1024;     // 2 SfAtoms (64 K each) per operand per stage
constexpr int kOffSfA = kATile + kBTile;
constexpr int kOffSfB = kOffSfA + kSfStage;  // gated: gate atoms here, up atoms at + kSfStage
constexpr int stage_bytes(bool gated) { return kOffSfB + ((gated ? 2 : 1) * kSfStage); }
static_assert(stage_bytes(false) % 1024 == 0 && stage_bytes(true) % 1024 == 0, "1024-aligned stages");
constexpr int kMaxExperts = 512;
constexpr int kScanBytes = 3 * (kMaxExperts + 1) * 4;
constexpr int smem_bytes(bool gated) { return (kStages * stage_bytes(gated)) + 128 + kScanBytes + 1024; }
constexpr int kMaxCtas = 256;
// B tensormap slots per CTA and source: a slot is rewritten after (kSlots - 1) tiles of KT stages each,
// so its last load was consumed once (kSlots - 1) * KT >= kStages (checked in supports).
constexpr int kSlots = 4;
constexpr size_t kMapBytes = 128;
__device__ __align__(128) uint8_t g_dec_maps[static_cast<size_t>(kMaxCtas) * 2 * kSlots * kMapBytes];
__device__ unsigned int g_dec_clipped = 0;

constexpr bool gated(MoeDecodeEpi e) { return e == MoeDecodeEpi::kSwiglu || e == MoeDecodeEpi::kGeglu; }

struct DArgs {
    GroupedNvfp4DecodeArgs a;
    int ne, N, K, n_tiles;
};

struct Smem {
    uint8_t* ring;
    uint32_t s0, bars;
    int* sp;   // [ne + 1] tile prefix
    int* sfa;  // [ne] A SF slab offsets (in-kernel form)
    int* sfo;  // [ne] output SF slab offsets
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

// Byte of row r's 4 K-group scales inside one 512 B SfAtom (r < 128).
__device__ __forceinline__ int sf_row(int r) { return ((r & 31) * 16) + (((r >> 5) & 3) * 4); }

// Compact M16 SF layout of the act epilogues' output (the down input, d_sfa_offsets == nullptr): per expert,
// per 16-row M tile, per 64-K atom 16 rows x 4 UE4M3 = 64 B, row i at i * 4. Bytes <= 1.5 x rows x N of the
// FP16 buffer it lives in; SfAtom would pad every expert to 128 rows.
constexpr int kM16Atom = 64;

__device__ __forceinline__ const CUtensorMap* rebind(uint8_t* slot, const CUtensorMap& tmpl,
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

// Warp 0: exclusive scans of tiles, compact A SF bytes and output SF bytes per expert, 32 experts per step.
__device__ __forceinline__ void scan(const DArgs& g, const Smem& x, int lane) {
    const int at_a = g.K / 64;
    const int at_o = (g.N + 63) / 64;
    int c0 = 0, c1 = 0, c2 = 0;
    for (int base = 0; base < g.ne; base += 32) {
        const int e = base + lane;
        const int m = e < g.ne ? g.a.d_M_per[e] : 0;
        const int mt = (m + kBM - 1) / kBM;
        const int v0 = mt * g.n_tiles;
        const int v1 = mt * at_a * kM16Atom;
        const int v2 = mt * at_o * kM16Atom;
        int i0 = v0, i1 = v1, i2 = v2;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int u0 = __shfl_up_sync(0xffffffffu, i0, d);
            const int u1 = __shfl_up_sync(0xffffffffu, i1, d);
            const int u2 = __shfl_up_sync(0xffffffffu, i2, d);
            if (lane >= d) {
                i0 += u0;
                i1 += u1;
                i2 += u2;
            }
        }
        if (e < g.ne) {
            x.sp[e] = c0 + i0 - v0;
            x.sfa[e] = c1 + i1 - v1;
            x.sfo[e] = c2 + i2 - v2;
        }
        c0 += __shfl_sync(0xffffffffu, i0, 31);
        c1 += __shfl_sync(0xffffffffu, i1, 31);
        c2 += __shfl_sync(0xffffffffu, i2, 31);
    }
    if (lane == 0)
        x.sp[g.ne] = c0;
}

struct Tile {
    int e, m0, n0, rows;
};

template <MoeDecodeEpi EPI>
__device__ __forceinline__ Tile tile_at(const DArgs& g, const int* sp, int t) {
    constexpr int kOC = gated(EPI) ? kBR / 2 : kBR;
    const int e = find_expert(sp, g.ne, t);
    const int local = t - sp[e];
    return {e, (local / g.n_tiles) * kBM, (local % g.n_tiles) * kOC, g.a.d_M_per[e]};
}

template <MoeDecodeEpi EPI>
__device__ __forceinline__ void produce(const DArgs& g, const Smem& x, const CUtensorMap& map_a,
                                        const CUtensorMap& map_b) {
    constexpr bool kG = gated(EPI);
    constexpr int kSB = stage_bytes(kG);
    const int KT = (g.K + kBK - 1) / kBK;
    const size_t slab = static_cast<size_t>(g.K / 64) * 512;  // SF bytes per 128-row slab (A and B share K)
    const int total = x.sp[g.ne];
    int ring = 0, cur = -1, slot = 0;
    const CUtensorMap* mb = nullptr;
    const CUtensorMap* mg = nullptr;
    uint8_t* maps = g_dec_maps + (static_cast<size_t>(blockIdx.x) * 2 * kSlots * kMapBytes);
    for (int t = static_cast<int>(blockIdx.x); t < total; t += static_cast<int>(gridDim.x)) {
        const Tile tl = tile_at<EPI>(g, x.sp, t);
        if (tl.e != cur) {
            slot = (slot + 1) % kSlots;
            mb = rebind(maps + (slot * kMapBytes), map_b, g.a.d_B_ptrs[tl.e]);
            if constexpr (kG)
                mg = rebind(maps + ((kSlots + slot) * kMapBytes), map_b, g.a.d_G_ptrs[tl.e]);
            cur = tl.e;
        }
        const bool compact = g.a.d_sfa_offsets == nullptr;  // M16 layout of an act epilogue's output
        const uint8_t* sfa = static_cast<const uint8_t*>(g.a.base_A_sf) +
                             (compact
                                  ? x.sfa[tl.e] + (static_cast<size_t>(tl.m0 / kBM) * (g.K / 64) * kM16Atom)
                                  : g.a.d_sfa_offsets[tl.e] + (static_cast<size_t>(tl.m0 / 128) * slab));
        const uint32_t sfa_stage = compact ? 2 * kM16Atom : kSfStage;
        const size_t b_slab = static_cast<size_t>(tl.n0 / 128) * slab;
        const uint8_t* sfb = static_cast<const uint8_t*>(g.a.d_SFB_ptrs[tl.e]) + b_slab;
        const uint8_t* sfg = kG ? static_cast<const uint8_t*>(g.a.d_SFG_ptrs[tl.e]) + b_slab : nullptr;
        const int row0 = g.a.d_expert_offsets[tl.e] + tl.m0;
        for (int kt = 0; kt < KT; ++kt, ++ring) {
            const int s = ring % kStages;
            const uint32_t full = x.bars + (s * 8);
            const uint32_t sb = x.s0 + (s * kSB);
            if (ring >= kStages)
                mbar_wait(x.bars + ((kStages + s) * 8), ((ring / kStages) - 1) & 1);
            // K % 128 == 64: the last stage holds one SfAtom per operand; TMA zero-fills the data past K,
            // the second atom is zeroed here (a stale slot could hold a NaN scale).
            const bool half_k = kt == KT - 1 && (g.K % kBK) != 0;
            const uint32_t sf_bytes = half_k ? kSfStage / 2 : kSfStage;
            const uint32_t sfa_bytes = half_k ? sfa_stage / 2 : sfa_stage;
            if (half_k) {
                uint8_t* st = x.ring + (static_cast<size_t>(s) * kSB);
                for (int o = kOffSfA; o < kSB; o += kSfStage) {  // SFA, SFB (and the up atoms when gated)
                    const uint32_t b = o == kOffSfA ? sfa_bytes : sf_bytes;
                    uint4* z = reinterpret_cast<uint4*>(st + o + b);
                    for (uint32_t i = 0; i < b / 16; ++i)
                        z[i] = make_uint4(0, 0, 0, 0);
                }
            }
            mbar_expect_tx(full, kATile + kBTile + ((kG ? 2 : 1) * sf_bytes) + sfa_bytes);
            tma_2d(sb, &map_a, kt * kRB, row0, full);
            if constexpr (kG) {
                tma_2d(sb + kATile, mg, kt * kRB, tl.n0, full);
                tma_2d(sb + kATile + (kBTile / 2), mb, kt * kRB, tl.n0, full);
                bulk_1d(sb + kOffSfB, sfg + (static_cast<size_t>(kt) * kSfStage), sf_bytes, full);
                bulk_1d(sb + kOffSfB + kSfStage, sfb + (static_cast<size_t>(kt) * kSfStage), sf_bytes, full);
            } else {
                tma_2d(sb + kATile, mb, kt * kRB, tl.n0, full);
                bulk_1d(sb + kOffSfB, sfb + (static_cast<size_t>(kt) * kSfStage), sf_bytes, full);
            }
            bulk_1d(sb + kOffSfA, sfa + (static_cast<size_t>(kt) * sfa_stage), sfa_bytes, full);
        }
    }
}

// Tile row of warp w's fragment nf (8 B rows). Gated: nf 0-1 gate rows w*16.., nf 2-3 the up rows 64 below.
template <MoeDecodeEpi EPI>
__device__ __forceinline__ int brow(int w, int nf) {
    if constexpr (gated(EPI))
        return ((nf >> 1) * 64) + (w * 16) + ((nf & 1) * 8);
    else
        return (w * 32) + (nf * 8);
}

__device__ __forceinline__ int swz(int r, int c) { return c ^ ((r >> 1) & 3); }

template <MoeDecodeEpi EPI>
__device__ __forceinline__ void mma_tile_stage(float (&acc)[4][4], uint32_t sa, const uint8_t* st, int w,
                                               int lane, int a_sf, int a_step, int bl) {
    const int lr = lane & 7;
    const int lm = lane >> 3;
    const int T1 = lane >> 2;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        uint32_t af[4];
        const int ra = lr + ((lm & 1) * 8);
        ldsm_x4(af, sa + (ra * kRB) + (swz(ra, (c * 2) + (lm >> 1)) * 16));
        const uint32_t sfa = *reinterpret_cast<const uint32_t*>(st + kOffSfA + (c ? a_step : 0) + a_sf);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
            uint32_t bf[4];
            const int rb = brow<EPI>(w, (np * 2) + (lm >> 1)) + lr;
            ldsm_x4(bf, sa + kATile + (rb * kRB) + (swz(rb, (c * 2) + (lm & 1)) * 16));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nf = (np * 2) + h;
                const int r = brow<EPI>(w, nf) + T1;
                // Gated: gate rows read the gate atoms, up rows (r >= 64) the up atoms; both at slab row bl +
                // r % 64.
                const int so = gated(EPI) ? kOffSfB + ((r >> 6) * kSfStage) + sf_row(bl + (r & 63))
                                          : kOffSfB + sf_row(r);
                const uint32_t sfb = *reinterpret_cast<const uint32_t*>(st + so +
                                                                        (static_cast<size_t>(c) * 512));
                mma_mxf4nvf4(acc[nf], af, h ? bf[2] : bf[0], h ? bf[3] : bf[1], sfa, sfb);
            }
        }
    }
}

__device__ __forceinline__ uint8_t pack_fp4(float lo, float hi) {
    uint32_t out;
    asm volatile(
        "{ .reg .b8 b;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "  cvt.u32.u8 %0, b; }\n"
        : "=r"(out)
        : "f"(lo), "f"(hi));
    return static_cast<uint8_t>(out);
}

__device__ __forceinline__ float rnd16(float v) { return __half2float(__float2half_rn(v)); }

// fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel's activations, from fp16-rounded gate / up.
template <MoeDecodeEpi EPI>
__device__ __forceinline__ float act(float g, float u) {
    if constexpr (EPI == MoeDecodeEpi::kSwiglu) {
        return (g / (1.0f + __expf(-g))) * u;
    } else if constexpr (EPI == MoeDecodeEpi::kGeglu) {
        constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
        constexpr float kGeluCoeff = 0.044715f;
        const float gelu = g * 0.5f * (1.0f + tanhf(kGeluSqrt2OverPi * (g + (kGeluCoeff * g * g * g))));
        return fminf(fmaxf(gelu * u, -65504.0f), 65504.0f);
    } else {
        return (u > 0.0f) ? (u * u) : 0.0f;
    }
}

// One 16-column block of a row held by the 4 lanes of a quad (2 fragments x 2 values each): absmax over the
// quad, UE4M3 scale, FP4 bytes (quantize_micro_block_nvfp4_from_vals).
__device__ __forceinline__ void quant_block(const float (&v)[4], bool valid, uint8_t* prow, int col,
                                            uint8_t* sf, int T0, float as) {
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (!valid)
        return;
    const float scale_f = amax / (as * 6.0f);  // as: static act tensor scale, 1.0 = amax / 6
    if (scale_f > 448.0f)
        g_dec_clipped = 1u;
    const uint8_t ue = float_to_fp8_e4m3(scale_f);
    float actual = fp8_e4m3_to_float_fast(ue);
    if (actual == 0.0f)
        actual = 1.0f / 512.0f;
    const float inv = 1.0f / (as * actual);
    prow[(col / 2)] = pack_fp4(v[0] * inv, v[1] * inv);
    prow[(col / 2) + 4] = pack_fp4(v[2] * inv, v[3] * inv);
    if (T0 == 0)
        *sf = ue;
}

template <MoeDecodeEpi EPI>
__device__ __forceinline__ void store(const float (&acc)[4][4], const DArgs& g, const Smem& x, const Tile& tl,
                                      int w, int lane) {
    const int T0 = lane & 3;
    const int T1 = lane >> 2;
    const int row_base = g.a.d_expert_offsets[tl.e] + tl.m0;
    const float alpha = g.a.d_alpha[tl.e];
    if constexpr (EPI == MoeDecodeEpi::kF16) {
        half* d = static_cast<half*>(g.a.base_D);
#pragma unroll
        for (int nf = 0; nf < 4; ++nf)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = T1 + (h * 8);
                const int col = tl.n0 + (w * 32) + (nf * 8) + (T0 * 2);
                if (tl.m0 + row < tl.rows && col < g.N)
                    *reinterpret_cast<__half2*>(d + (static_cast<size_t>(row_base + row) * g.N) +
                                                col) = __floats2half2_rn(acc[nf][h ? 2 : 0] * alpha,
                                                                         acc[nf][h ? 3 : 1] * alpha);
            }
    } else {
        const int at_o = (g.N + 63) / 64;
        uint8_t* packed = static_cast<uint8_t*>(g.a.base_D);
        uint8_t* sf = static_cast<uint8_t*>(g.a.base_D_sf) + x.sfo[tl.e];
        const float galpha = gated(EPI) ? g.a.d_galpha[tl.e] : 0.0f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = T1 + (h * 8);
            const int lrow = tl.m0 + row;
            uint8_t* prow = packed + (static_cast<size_t>(row_base + row) * (g.N / 2));
            // Gated: one block (gate nf 0-1 x up nf 2-3); relu2: two blocks (nf 0-1, nf 2-3).
#pragma unroll
            for (int b = 0; b < (gated(EPI) ? 1 : 2); ++b) {
                float v[4];
#pragma unroll
                for (int f = 0; f < 2; ++f)
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const float a = acc[(b * 2) + f][(h * 2) + j];
                        v[(f * 2) + j] = gated(EPI) ? act<EPI>(rnd16(a * galpha),
                                                               rnd16(acc[f + 2][(h * 2) + j] * alpha))
                                                    : act<EPI>(0.0f, rnd16(a * alpha));
                    }
                const int col0 = tl.n0 + (gated(EPI) ? w * 16 : (w * 32) + (b * 16));
                quant_block(v, lrow < tl.rows && col0 < g.N, prow, col0 + (T0 * 2),
                            sf + (static_cast<size_t>(((lrow / kBM) * at_o) + (col0 >> 6)) * kM16Atom) +
                                (static_cast<size_t>(row) * 4) + ((col0 >> 4) & 3),
                            T0, g.a.act_scale);
            }
        }
    }
}

template <MoeDecodeEpi EPI>
__device__ __forceinline__ void consume(const DArgs& g, const Smem& x, int w, int lane) {
    constexpr int kSB = stage_bytes(gated(EPI));
    const int KT = (g.K + kBK - 1) / kBK;
    const int total = x.sp[g.ne];
    const bool compact = g.a.d_sfa_offsets == nullptr;
    const int sfa_lane = ((lane & 1) * 8) + (lane >> 2);  // A row whose scales this lane supplies
    const int a_step = compact ? kM16Atom : 512;
    int ring = 0;
    for (int t = static_cast<int>(blockIdx.x); t < total; t += static_cast<int>(gridDim.x)) {
        const Tile tl = tile_at<EPI>(g, x.sp, t);
        const int a_sf = compact ? sfa_lane * 4 : sf_row((tl.m0 & 127) + sfa_lane);
        float acc[4][4] = {};
        for (int kt = 0; kt < KT; ++kt, ++ring) {
            const int s = ring % kStages;
            mbar_wait(x.bars + (s * 8), (ring / kStages) & 1);
            mma_tile_stage<EPI>(acc, x.s0 + (s * kSB), x.ring + (static_cast<size_t>(s) * kSB), w, lane, a_sf,
                                a_step, tl.n0 & 127);
            __syncwarp();
            if (lane == 0)
                mbar_arrive(x.bars + ((kStages + s) * 8));
        }
        store<EPI>(acc, g, x, tl, w, lane);
    }
}

template <MoeDecodeEpi EPI>
__global__ void __launch_bounds__(kThreads, 1) gemm_grouped_nvfp4_decode_kernel(
    DArgs g, const __grid_constant__ CUtensorMap map_a, const __grid_constant__ CUtensorMap map_b) {
    extern __shared__ __align__(128) uint8_t smem_raw[];
    const uint32_t raw = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const uint32_t pad = ((raw + 1023) & ~uint32_t(1023)) - raw;
    constexpr int kRing = kStages * stage_bytes(gated(EPI));
    Smem x{};
    x.ring = smem_raw + pad;
    x.s0 = raw + pad;
    x.bars = x.s0 + kRing;
    x.sp = reinterpret_cast<int*>(x.ring + kRing + 128);
    x.sfa = x.sp + kMaxExperts + 1;
    x.sfo = x.sfa + kMaxExperts + 1;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (threadIdx.x == 0) {
        for (int s = 0; s < kStages; ++s) {
            mbar_init(x.bars + (s * 8), 1);
            mbar_init(x.bars + ((kStages + s) * 8), kWarps);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0)
        scan(g, x, lane);
    __syncthreads();
    if (warp == kWarps) {
        if (lane == 0)
            produce<EPI>(g, x, map_a, map_b);
        return;
    }
    consume<EPI>(g, x, warp, lane);
}

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

// Opt-in smem per instance, set by gemm_grouped_nvfp4_decode_prewarm only: cudaFuncSetAttribute inside a
// global-mode stream capture (decode / prefill graphs) fails, a lazy first call there refused every launch.
bool g_smem_ok[4] = {false, false, false, false};

template <MoeDecodeEpi EPI>
void set_smem() {
    g_smem_ok[static_cast<int>(EPI)] = cudaFuncSetAttribute(gemm_grouped_nvfp4_decode_kernel<EPI>,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            smem_bytes(gated(EPI))) == cudaSuccess;
}

bool smem_ready(MoeDecodeEpi epi) { return g_smem_ok[static_cast<int>(epi)]; }

size_t align128(size_t v) { return (v + 127) & ~size_t(127); }

}  // namespace

bool gemm_grouped_nvfp4_decode_supports(int n_experts, int N, int K, MoeDecodeEpi epi) {
    const int sms = sm_count();
    const int KT = (K + kBK - 1) / kBK;
    const int n_mod = gated(epi) ? 64 : epi == MoeDecodeEpi::kRelu2 ? 16 : 8;
    return sms > 0 && sms <= kMaxCtas && n_experts > 0 && n_experts <= kMaxExperts && N > 0 &&
           N % n_mod == 0 && K > 0 && K % 64 == 0 && (kSlots - 1) * KT >= kStages && smem_ready(epi);
}

size_t gemm_grouped_nvfp4_decode_act_sf_offset(int expanded, int N) {
    return align128(static_cast<size_t>(expanded) * (N / 2));
}

size_t gemm_grouped_nvfp4_decode_act_bytes(int n_experts, int expanded, int N) {
    // M16 tiles: sum over experts of ceil(M_e / 16) <= expanded / 16 + active experts.
    const size_t tiles = static_cast<size_t>(std::min(n_experts, expanded)) + (expanded / kBM);
    return gemm_grouped_nvfp4_decode_act_sf_offset(expanded, N) + (tiles * ((N + 63) / 64) * kM16Atom);
}

bool gemm_grouped_nvfp4_decode(int n_experts, int N, int K, int expanded, MoeDecodeEpi epi,
                               const GroupedNvfp4DecodeArgs& args, cudaStream_t stream) {
    if (!gemm_grouped_nvfp4_decode_supports(n_experts, N, K, epi) || expanded <= 0 ||
        args.base_A_packed == nullptr || args.base_A_sf == nullptr || args.base_D == nullptr ||
        args.d_B_ptrs == nullptr || args.d_SFB_ptrs == nullptr || args.d_alpha == nullptr)
        return false;
    if (gated(epi) && (args.d_G_ptrs == nullptr || args.d_SFG_ptrs == nullptr || args.d_galpha == nullptr))
        return false;
    if (epi != MoeDecodeEpi::kF16 && args.base_D_sf == nullptr)
        return false;
    CUtensorMap map_a{};
    CUtensorMap map_b{};  // template: rows N, the base is replaced per expert in the kernel
    if (!encode_2d(&map_a, args.base_A_packed, expanded, K / 2, kRB, kBM) ||
        !encode_2d(&map_b, args.base_A_packed, N, K / 2, kRB, gated(epi) ? kBR / 2 : kBR))
        return false;
    const int oc = gated(epi) ? kBR / 2 : kBR;
    const DArgs g{args, n_experts, N, K, (N + oc - 1) / oc};
    const int grid = sm_count();
    switch (epi) {
        case MoeDecodeEpi::kF16:
            gemm_grouped_nvfp4_decode_kernel<MoeDecodeEpi::kF16>
                <<<grid, kThreads, smem_bytes(false), stream>>>(g, map_a, map_b);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        case MoeDecodeEpi::kSwiglu:
            gemm_grouped_nvfp4_decode_kernel<MoeDecodeEpi::kSwiglu>
                <<<grid, kThreads, smem_bytes(true), stream>>>(g, map_a, map_b);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        case MoeDecodeEpi::kGeglu:
            gemm_grouped_nvfp4_decode_kernel<MoeDecodeEpi::kGeglu>
                <<<grid, kThreads, smem_bytes(true), stream>>>(g, map_a, map_b);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        case MoeDecodeEpi::kRelu2:
            gemm_grouped_nvfp4_decode_kernel<MoeDecodeEpi::kRelu2>
                <<<grid, kThreads, smem_bytes(false), stream>>>(g, map_a, map_b);
            IMP_CUDA_CHECK_LAUNCH();
            break;
    }
    return true;
}

void gemm_grouped_nvfp4_decode_prewarm() {
    (void)sm_count();
    set_smem<MoeDecodeEpi::kF16>();
    set_smem<MoeDecodeEpi::kSwiglu>();
    set_smem<MoeDecodeEpi::kGeglu>();
    set_smem<MoeDecodeEpi::kRelu2>();
}

bool gemm_grouped_nvfp4_decode_scale_clipped() {
    unsigned int v = 0;
    if (cudaMemcpyFromSymbol(&v, g_dec_clipped, sizeof(v)) != cudaSuccess)
        return false;
    return v != 0;
}

}  // namespace imp
