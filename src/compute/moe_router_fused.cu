// Fused MoE router: CTA tile = BM rows x all experts (<= 128), mma.sync m16n8k16 with an fp16
// accumulator per 32-wide k tile promoted into fp32 (fp32-accumulate MMA runs at 1/4 rate on sm_120:
// 253 vs 1956 TFLOPS), ldmatrix on XOR-swizzled 64 B smem rows, cp.async pipeline (the
// gemm_f16_narrow_prefill structure). Small M splits K over CTAs: each parks its fragments in ws,
// the last to take the tile's ticket sums them in split order (deterministic per shape) and runs the
// epilogue: logits to smem, one warp per row: softmax / sigmoid (+ bias), top-k, renorm.

#include "compute/moe_router_fused.h"
#include "core/logging.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>

namespace imp {
namespace {

constexpr int kBK = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kNPad = 128;
constexpr int kChunkHalves = 8;  // one 16 B cp.async
constexpr int kChunksPerRow = kBK / kChunkHalves;
constexpr int kMaxSplit = 16;
constexpr int kMinKTilesPerCta = 4;
constexpr int kTargetCtas = 192;
constexpr int kMaxTiles = 4096;  // ticket slots
constexpr size_t kTicketBytes = kMaxTiles * sizeof(unsigned);
constexpr int kLogitLd = kNPad + 4;  // smem logits row stride (floats)
constexpr int kMaxTopK = 32;
constexpr int kSlots = kNPad / 32;  // experts per lane in the epilogue

struct Args {
    MoeRouterFusedArgs a;
    int split;
    int ktiles;
    unsigned* tickets;
    float4* partials;  // [tile][split][frag][thread]
};

__device__ __forceinline__ void cp_async16(uint32_t s, const void* g, bool valid) {
    const int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 16 B chunk c (0..3) of 64 B row r: the 8 rows of an ldmatrix 8x8 matrix land on distinct bank quads.
__device__ __forceinline__ uint32_t swz(int r, int c) {
    return static_cast<uint32_t>((r * kBK * 2) + ((c ^ ((r >> 1) & 3)) * 16));
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int WM>
struct Cfg {
    static constexpr int kBM = 16 * WM;
    static constexpr int kWN = kWarps / WM;     // column groups
    static constexpr int kWCols = kNPad / kWN;  // columns per warp
    static constexpr int kNF = kWCols / 8;      // n8 fragments per warp
    static constexpr int kStageBytes = (kBM + kNPad) * kBK * 2;
    static constexpr int kStages = 4;
    static constexpr int kPipeBytes = kStages * kStageBytes;
    static constexpr int kLogitBytes = kBM * kLogitLd * 4;
    static constexpr int kSmemBytes = kPipeBytes > kLogitBytes ? kPipeBytes : kLogitBytes;
    static constexpr int kRowsPerPass = kThreads / kChunksPerRow;
    static constexpr int kLoads = (kBM + kNPad + kRowsPerPass - 1) /
                                  kRowsPerPass;  // last pass partial at BM 16
    static_assert(kSmemBytes <= 48 * 1024, "static smem");
};

// One 16 B chunk per thread per pass: x rows of the tile, then the router rows (zero past ne / M).
template <int WM>
struct Loader {
    using C = Cfg<WM>;
    const half* src[C::kLoads];
    uint32_t dst[C::kLoads];
    unsigned ok = 0;    // has a source row (else zero-fill)
    unsigned live = 0;  // the slot exists in the BM + kNPad tile rows

    __device__ __forceinline__ Loader(const MoeRouterFusedArgs& a, int row0, int kt0) {
        const int kc = static_cast<int>(threadIdx.x) % kChunksPerRow;
        const int r0 = static_cast<int>(threadIdx.x) / kChunksPerRow;
        const size_t k_off = (static_cast<size_t>(kt0) * kBK) + static_cast<size_t>(kc * kChunkHalves);
#pragma unroll
        for (int i = 0; i < C::kLoads; ++i) {
            const int r = r0 + (i * C::kRowsPerPass);
            const bool is_x = r < C::kBM;
            const bool valid = is_x ? row0 + r < a.M : (r < C::kBM + kNPad && r - C::kBM < a.ne);
            const half* base = is_x ? a.x : a.w;
            const int src_row = valid ? (is_x ? row0 + r : r - C::kBM) : 0;
            src[i] = base + (static_cast<size_t>(src_row) * a.K) + k_off;
            dst[i] = swz(r, kc);
            ok |= valid ? (1u << i) : 0u;
            live |= r < C::kBM + kNPad ? (1u << i) : 0u;
        }
    }

    __device__ __forceinline__ void load(uint32_t s, int step) const {
#pragma unroll
        for (int i = 0; i < C::kLoads; ++i)
            if ((live >> i) & 1u)
                cp_async16(s + dst[i], src[i] + (step * kBK), (ok >> i) & 1u);
    }
};

// K tiles [kt0, kt0 + n_tiles) of this CTA into acc; fp16 partial per k tile, promoted to fp32.
template <int WM>
__device__ __forceinline__ void mainloop(const MoeRouterFusedArgs& a, uint32_t s0, int row0, int kt0,
                                         int n_tiles, float (&acc)[Cfg<WM>::kNF][4]) {
    using C = Cfg<WM>;
    constexpr int NF = C::kNF;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wc0 = (warp / WM) * C::kWCols;
    const Loader<WM> ld(a, row0, kt0);
#pragma unroll
    for (int s = 0; s < C::kStages - 1; ++s) {
        if (s < n_tiles)
            ld.load(s0 + (s * C::kStageBytes), s);
        cp_async_commit();
    }
    const int a_row = ((warp % WM) * 16) + (lane & 15);
    const int a_kh = lane >> 4;
    const int b_row = C::kBM + wc0 + (lane & 7) + ((lane >> 4) << 3);
    const int b_kh = (lane >> 3) & 1;
    for (int kt = 0; kt < n_tiles; ++kt) {
        cp_async_wait<C::kStages - 2>();
        __syncthreads();
        const int nxt = kt + C::kStages - 1;
        if (nxt < n_tiles)
            ld.load(s0 + ((nxt % C::kStages) * C::kStageBytes), nxt);
        cp_async_commit();
        const uint32_t st = s0 + ((kt % C::kStages) * C::kStageBytes);
        uint32_t h[NF][2] = {};
#pragma unroll
        for (int kk = 0; kk < kBK / 16; ++kk) {
            uint32_t af[4];
            ldsm_x4(st + swz(a_row, (kk * 2) + a_kh), af[0], af[1], af[2], af[3]);
#pragma unroll
            for (int jb = 0; jb < NF / 2; ++jb) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4(st + swz(b_row + (jb * 16), (kk * 2) + b_kh), b0, b1, b2, b3);
                mma_f16(h[2 * jb], af, b0, b1);
                mma_f16(h[(2 * jb) + 1], af, b2, b3);
            }
        }
#pragma unroll
        for (int f = 0; f < NF; ++f) {  // rows g / g + 8, columns 2t, 2t + 1
            const float2 lo = __half22float2(*reinterpret_cast<const half2*>(&h[f][0]));
            const float2 hi = __half22float2(*reinterpret_cast<const half2*>(&h[f][1]));
            acc[f][0] += lo.x;
            acc[f][1] += lo.y;
            acc[f][2] += hi.x;
            acc[f][3] += hi.y;
        }
    }
    cp_async_wait<0>();
    __syncthreads();  // the pipeline smem holds the logits next
}

// Split-K: park the fragments, take the tile's ticket; the last CTA sums every split in order into
// acc and re-arms the ticket. False for the other CTAs (they exit).
template <int WM>
__device__ __forceinline__ bool split_reduce(const Args& args, int tile, int ks,
                                             float (&acc)[Cfg<WM>::kNF][4], int& s_last) {
    constexpr int NF = Cfg<WM>::kNF;
    const int S = args.split;
    float4* tile_ws = args.partials + (static_cast<size_t>(tile) * S * NF * kThreads);
    float4* own = tile_ws + (static_cast<size_t>(ks) * NF * kThreads);
#pragma unroll
    for (int f = 0; f < NF; ++f)
        __stcg(own + ((static_cast<size_t>(f) * kThreads) + threadIdx.x),
               make_float4(acc[f][0], acc[f][1], acc[f][2], acc[f][3]));
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned t = atomicAdd(args.tickets + tile, 1u);
        s_last = (t == static_cast<unsigned>(S - 1)) ? 1 : 0;
        if (s_last)
            args.tickets[tile] = 0u;  // all S arrived: re-arm for the next launch
    }
    __syncthreads();
    if (!s_last)
        return false;
    __threadfence();
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        float4 sum = __ldcg(tile_ws + ((static_cast<size_t>(f) * kThreads) + threadIdx.x));
        for (int q = 1; q < S; ++q) {
            const float4 v = __ldcg(tile_ws + (static_cast<size_t>(q) * NF * kThreads) +
                                    (static_cast<size_t>(f) * kThreads) + threadIdx.x);
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        acc[f][0] = sum.x;
        acc[f][1] = sum.y;
        acc[f][2] = sum.z;
        acc[f][3] = sum.w;
    }
    return true;
}

// Scores of one row's slots e = lane + 32 j: p = softmax / sigmoid (weights), sel = p + bias (selection).
__device__ __forceinline__ void row_scores(const MoeRouterFusedArgs& a, const float* logits, int lane,
                                           float (&p)[kSlots], float (&sel)[kSlots]) {
    float mx = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
        p[j] = (lane + (32 * j)) < a.ne ? logits[lane + (32 * j)] : -FLT_MAX;
        mx = fmaxf(mx, p[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kSlots; ++j)
        sum += (lane + (32 * j)) < a.ne ? expf(p[j] - mx) : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float inv = 1.0f / sum;
#pragma unroll
    for (int j = 0; j < kSlots; ++j) {
        const int e = lane + (32 * j);
        const float s = a.use_sigmoid ? 1.0f / (1.0f + expf(-p[j])) : expf(p[j] - mx) * inv;
        p[j] = e < a.ne ? s : -FLT_MAX;
        sel[j] = (e < a.ne && a.score_bias) ? p[j] + __half2float(a.score_bias[e]) : p[j];
    }
}

// Warp argmax over the lane slots; ties pick the lowest expert index (topk_gating_kernel's order).
__device__ __forceinline__ void warp_argmax(const float (&sel)[kSlots], int lane, float& best, int& bi) {
    best = sel[0];
    bi = lane;
#pragma unroll
    for (int j = 1; j < kSlots; ++j)
        if (sel[j] > best) {
            best = sel[j];
            bi = lane + (32 * j);
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
}

// Unbiased score of expert bi (broadcast from its owner lane), whose selection slot is cleared.
__device__ __forceinline__ float take_expert(const float (&p)[kSlots], float (&sel)[kSlots], int bi,
                                             int lane) {
    const int owner = bi & 31;
    const int slot = bi >> 5;
    float pv = 0.0f;
#pragma unroll
    for (int j = 0; j < kSlots; ++j)
        if (j == slot) {
            pv = p[j];
            if (lane == owner)
                sel[j] = -FLT_MAX;
        }
    return __shfl_sync(0xffffffffu, pv, owner);
}

// One warp per row: top-k on sel, weights = unbiased scores, renorm summed in pick order.
__device__ __forceinline__ void route_row(const MoeRouterFusedArgs& a, const float* logits, int row) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float p[kSlots], sel[kSlots];
    row_scores(a, logits, lane, p, sel);
    int my_idx = 0;
    float my_val = 0.0f;
    float norm = 0.0f;
    for (int k = 0; k < a.top_k; ++k) {
        float best;
        int bi;
        warp_argmax(sel, lane, best, bi);
        const float pv = take_expert(p, sel, bi, lane);
        norm += pv;
        if (lane == k) {
            my_idx = bi;
            my_val = pv;
        }
    }
    const float mult = (a.normalize && norm > 0.0f) ? 1.0f / norm : 1.0f;
    if (lane < a.top_k) {
        const size_t o = (static_cast<size_t>(row) * a.top_k) + lane;
        a.ids[o] = my_idx;
        a.weights[o] = my_val * mult;
    }
}

template <int WM>
__global__ void __launch_bounds__(kThreads) moe_router_fused_kernel(Args args) {
    using C = Cfg<WM>;
    constexpr int NF = C::kNF;
    __shared__ __align__(128) unsigned char smem_raw[C::kSmemBytes];
    __shared__ int s_last;
    const uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw));
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tile = static_cast<int>(blockIdx.x);
    const int ks = static_cast<int>(blockIdx.y);
    const int row0 = tile * C::kBM;
    const int kt0 = (ks * args.ktiles) / args.split;
    const int kt1 = ((ks + 1) * args.ktiles) / args.split;

    float acc[NF][4] = {};
    mainloop<WM>(args.a, s0, row0, kt0, kt1 - kt0, acc);
    if (args.split > 1 && !split_reduce<WM>(args, tile, ks, acc, s_last))
        return;

    float* s_logits = reinterpret_cast<float*>(smem_raw);
    const int r = ((warp % WM) * 16) + (lane >> 2);
    const int c0 = ((warp / WM) * C::kWCols) + ((lane & 3) * 2);
#pragma unroll
    for (int f = 0; f < NF; ++f) {
        *reinterpret_cast<float2*>(
            s_logits + static_cast<ptrdiff_t>((r * kLogitLd) + c0 + (f * 8))) = make_float2(acc[f][0],
                                                                                            acc[f][1]);
        *reinterpret_cast<float2*>(
            s_logits + static_cast<ptrdiff_t>(((r + 8) * kLogitLd) + c0 + (f * 8))) = make_float2(acc[f][2],
                                                                                                  acc[f][3]);
    }
    __syncthreads();
    for (int rr = warp; rr < C::kBM; rr += kWarps)
        if (row0 + rr < args.a.M)
            route_row(args.a, s_logits + static_cast<ptrdiff_t>(rr * kLogitLd), row0 + rr);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

bool shape_ok(const MoeRouterFusedArgs& a, const void* ws, size_t ws_bytes) {
    if (!a.x || !a.w || !a.ids || !a.weights || a.M < 1 || a.K < kBK || (a.K % kBK) != 0)
        return false;
    if (a.ne < 8 || a.ne > kNPad || (a.ne % 8) != 0 || a.top_k < 1 || a.top_k > kMaxTopK || a.top_k > a.ne)
        return false;
    const auto misaligned = [](const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) != 0; };
    return !misaligned(a.x) && !misaligned(a.w) && ws && !misaligned(ws) && ws_bytes >= kTicketBytes;
}

template <int WM>
bool launch(const Args& args, int tiles, cudaStream_t stream) {
    moe_router_fused_kernel<WM><<<dim3(tiles, args.split), kThreads, 0, stream>>>(args);
    IMP_CUDA_CHECK_LAUNCH();
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

size_t moe_router_fused_workspace_bytes(int /*max_rows*/) {
    // Split > 1 only while tiles * split <= kTargetCtas: the partials of the largest tile cap that.
    return kTicketBytes + (static_cast<size_t>(kTargetCtas) * Cfg<2>::kBM * kNPad * sizeof(float));
}

bool moe_router_fused(const MoeRouterFusedArgs& a, void* ws, size_t ws_bytes, cudaStream_t stream) {
    if (!shape_ok(a, ws, ws_bytes))
        return false;
    // 32-row tiles once they fill the card, else 16; split K until the grid reaches the target and
    // every CTA keeps >= kMinKTilesPerCta k tiles; the partials must fit the workspace.
    const int ktiles = a.K / kBK;
    const bool wide = ceil_div(a.M, 32) * 2 > kTargetCtas;
    const int bm = wide ? 32 : 16;
    const int tiles = ceil_div(a.M, bm);
    if (tiles > kMaxTiles)
        return false;
    int split = 1;
    while (split < kMaxSplit && tiles * split * 2 <= kTargetCtas && ktiles / (split * 2) >= kMinKTilesPerCta)
        split *= 2;
    const size_t part_bytes = static_cast<size_t>(tiles) * split * bm * kNPad * sizeof(float);
    if (split > 1 && kTicketBytes + part_bytes > ws_bytes)
        return false;

    Args args{};
    args.a = a;
    args.split = split;
    args.ktiles = ktiles;
    args.tickets = static_cast<unsigned*>(ws);
    args.partials = reinterpret_cast<float4*>(static_cast<unsigned char*>(ws) + kTicketBytes);
    return wide ? launch<2>(args, tiles, stream) : launch<1>(args, tiles, stream);
}

}  // namespace imp
