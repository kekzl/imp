// Small-M NVFP4 GEMM, weight streaming (batched decode, M 1..32): y[m,n] = ts * W[n,:] . x[m,:] (+ beta y).
// Both sides plain NVFP4 (nibbles [rows, K/2], linear UE4M3 scales [rows, K/16]), mxf4nvf4 m16n8k64.
//   CTA = kMT x 16 weight rows x the whole K, 8 warps; warp w owns the 128-element k-steps ks % 8 == w.
//   W is the MMA A operand (16 rows), the activations B (8 tokens per n-tile, kNT tiles).
//   Global -> registers, no smem ring: a quad loads one 64-byte k-step of a row (LDG.128 per lane) and a
//   quad transpose (3 shuffles) turns it into the A / B fragments of the step's two MMAs.
//   fp32 accumulate, warp partials summed in warp order through smem, ts and beta in fp32.
// A row's bits depend on K only: tokens never share an MMA column, the K order is fixed (kMT, kU, M do not
// move it). K % 64 == 0 (a K % 128 == 64 tail step runs its first MMA only), N % (16 kMT) == 0.

#include "quant/nvfp4_gemm.h"

#include "core/logging.h"
#include "core/pdl_device.cuh"
#include "core/pdl_launch.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {
namespace {

constexpr int kWarps = 8;   // fixes the K order: never chosen by M or N
constexpr int kStepB = 64;  // bytes per row per k-step (128 elements, two m16n8k64 MMAs)
constexpr int kStepSf = 8;  // scale bytes per row per k-step

__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldg_u32_stream(const uint8_t* p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void mma_mxf4nvf4(float (&acc)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                             uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sfa,
                                             uint32_t sfb) {
#if __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, "
        "%15};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(static_cast<uint16_t>(0)),
          "h"(static_cast<uint16_t>(0)), "r"(sfb), "h"(static_cast<uint16_t>(0)),
          "h"(static_cast<uint16_t>(0)));
#endif
}

// Branch-free w[i & 3]: three SELs on lane-constant predicates (a ternary chain compiled to branches).
__device__ __forceinline__ uint32_t pick4(const uint32_t (&w)[4], int i) {
    const uint32_t lo = (i & 1) ? w[1] : w[0];
    const uint32_t hi = (i & 1) ? w[3] : w[2];
    return (i & 2) ? hi : lo;
}

// Lane t4 of a quad holds bytes [16 t4, 16 t4 + 16) of a row's 64-byte k-step; afterwards o[j] = word t4 of
// quad lane j, i.e. the fragment word at byte 16 j + 4 t4: o[0] / o[1] = MMA 0 k lo / hi, o[2] / o[3] =
// MMA 1.
__device__ __forceinline__ void quad_transpose(const uint4& v, int t4, uint32_t (&o)[4]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    uint32_t r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t send = pick4(w, t4 ^ i);  // the word lane t4 ^ i needs
        r[i] = i == 0 ? send : __shfl_xor_sync(0xffffffffu, send, i);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        o[j] = pick4(r, j ^ t4);
}

// Registers of one k-step: weight rows g / g + 8 per m-tile, their scale words (lane's SFA row), the
// tokens' nibbles and scale words (token nt * 8 + g supplies both the B column and the SFB lane).
template <int kMT, int kNT>
struct Step {
    uint4 w[kMT][2];
    uint32_t wsf[kMT][2];
    uint4 x[kNT];
    uint32_t xsf[kNT][2];
};

template <int kMT, int kNT>
struct Ptrs {
    const uint8_t* w[kMT][2];
    const uint8_t* wsf[kMT];
    const uint8_t* x[kNT];
    const uint8_t* xsf[kNT];
    bool xv[kNT];
};

template <int kMT, int kNT>
__device__ __forceinline__ void load_w(Step<kMT, kNT>& s, const Ptrs<kMT, kNT>& p, int ks, int ks_full,
                                       int ks_total, int t4) {
    const bool live = ks < ks_total;
    const bool hi = ks < ks_full;  // a K % 128 == 64 tail step holds bytes 0..31 (lanes t4 0-1) only
    const bool data = live && (hi || t4 < 2);
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
        for (int r = 0; r < 2; ++r)
            s.w[mt][r] = data ? ldg_stream(p.w[mt][r] + (static_cast<int64_t>(ks) * kStepB))
                              : make_uint4(0, 0, 0, 0);
        const uint8_t* sf = p.wsf[mt] + (static_cast<int64_t>(ks) * kStepSf);
        s.wsf[mt][0] = live ? ldg_u32_stream(sf) : 0u;
        s.wsf[mt][1] = hi ? ldg_u32_stream(sf + 4) : 0u;
    }
}

template <int kMT, int kNT>
__device__ __forceinline__ void load_x(Step<kMT, kNT>& s, const Ptrs<kMT, kNT>& p, int ks, int ks_full,
                                       int ks_total, int t4) {
    const bool live = ks < ks_total;
    const bool hi = ks < ks_full;
    const bool data = live && (hi || t4 < 2);
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
        const int64_t o = static_cast<int64_t>(ks) * kStepB;
        const int64_t so = static_cast<int64_t>(ks) * kStepSf;
        s.x[nt] = (p.xv[nt] && data) ? __ldg(reinterpret_cast<const uint4*>(p.x[nt] + o))
                                     : make_uint4(0, 0, 0, 0);
        s.xsf[nt][0] = (p.xv[nt] && live) ? __ldg(reinterpret_cast<const uint32_t*>(p.xsf[nt] + so)) : 0u;
        s.xsf[nt][1] = (p.xv[nt] && hi) ? __ldg(reinterpret_cast<const uint32_t*>(p.xsf[nt] + so + 4)) : 0u;
    }
}

// Branch-free: a step past K (or the missing half of a tail step) holds zero nibbles and zero scales, so its
// MMAs add +0 (a branch around the shuffles and MMAs cost BSSY / WARPSYNC per step).
template <int kMT, int kNT>
__device__ __forceinline__ void mma_step(float (&acc)[kMT][kNT][4], const Step<kMT, kNT>& s, int t4) {
    uint32_t wf[kMT][2][4];
    uint32_t xf[kNT][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int r = 0; r < 2; ++r)
            quad_transpose(s.w[mt][r], t4, wf[mt][r]);
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt)
        quad_transpose(s.x[nt], t4, xf[nt]);
#pragma unroll
    for (int c = 0; c < 2; ++c) {
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
            for (int nt = 0; nt < kNT; ++nt)
                mma_mxf4nvf4(acc[mt][nt], wf[mt][0][2 * c], wf[mt][1][2 * c], wf[mt][0][(2 * c) + 1],
                             wf[mt][1][(2 * c) + 1], xf[nt][2 * c], xf[nt][(2 * c) + 1], s.wsf[mt][c],
                             s.xsf[nt][c]);
    }
}

// One CTA: rows n0 .. n0 + 16 kMT - 1 of W, all M tokens, the whole K. Shared by the single and multi
// kernels.
template <int kMT, int kNT, int kU>
__device__ __forceinline__ void ws_cta(const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsf,
                                       const uint8_t* __restrict__ x, const uint8_t* __restrict__ xsf,
                                       float ts, float beta, half* __restrict__ y, int M, int N, int K,
                                       int n0) {
    constexpr int kR = kMT * 16;
    constexpr int kRP = kR + 4;  // padded row: the 4 t4 lanes of a token pair land 8 banks apart
    __shared__ float red[kWarps][kNT * 8][kRP];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int64_t row_b = K / 2;
    const int64_t row_sf = K / 16;
    const int ks_full = K / 128;
    const int ks_total = (K + 127) / 128;
    constexpr int kPerRound = kWarps * kU;
    const int rounds = (ks_total + kPerRound - 1) / kPerRound;

    pdl_trigger();
    // #2218 bounded: n0 + kR <= N (grid covers N / kR tiles); ks * 64 + 16 t4 + 16 <= K / 2 on data loads,
    // tokens < M, ks * 8 + 8 <= K / 16 on full steps (+4 on the tail).
    Ptrs<kMT, kNT> p;
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt) {
        const int r0 = n0 + (mt * 16) + g;
        p.w[mt][0] = w + (r0 * row_b) + static_cast<ptrdiff_t>(t4 * 16);
        p.w[mt][1] = p.w[mt][0] + (8 * row_b);
        p.wsf[mt] = wsf + ((n0 + (mt * 16) + ((lane & 1) * 8) + g) * row_sf);
    }
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
        const int tok = (nt * 8) + g;
        p.xv[nt] = tok < M;
        const int t = p.xv[nt] ? tok : 0;
        p.x[nt] = x + (t * row_b) + static_cast<ptrdiff_t>(t4 * 16);
        p.xsf[nt] = xsf + (t * row_sf);
    }

    float acc[kMT][kNT][4];
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int nt = 0; nt < kNT; ++nt)
#pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[mt][nt][r] = 0.0f;

    // Two register sets: round r + 1's loads are in flight while round r's MMAs run.
    Step<kMT, kNT> s0[kU], s1[kU];
    auto ks_of = [&](int round, int u) { return (round * kPerRound) + (u * kWarps) + warp; };
    auto load_round_w = [&](Step<kMT, kNT>(&s)[kU], int round) {
#pragma unroll
        for (int u = 0; u < kU; ++u)
            load_w(s[u], p, ks_of(round, u), ks_full, ks_total, t4);
    };
    auto load_round_x = [&](Step<kMT, kNT>(&s)[kU], int round) {
#pragma unroll
        for (int u = 0; u < kU; ++u)
            load_x(s[u], p, ks_of(round, u), ks_full, ks_total, t4);
    };
    auto mma_round = [&](const Step<kMT, kNT>(&s)[kU], int round) {
#pragma unroll
        for (int u = 0; u < kU; ++u)
            mma_step(acc, s[u], t4);
    };
    // Weights are immutable: rounds 0 and 1 stream in before the grid dependency (pdl_device.cuh).
    load_round_w(s0, 0);
    if (rounds > 1)
        load_round_w(s1, 1);
    pdl_wait();
    load_round_x(s0, 0);
    if (rounds > 1)
        load_round_x(s1, 1);
    for (int round = 0; round < rounds; round += 2) {
        mma_round(s0, round);
        if (round + 2 < rounds) {
            load_round_w(s0, round + 2);
            load_round_x(s0, round + 2);
        }
        if (round + 1 >= rounds)
            break;
        mma_round(s1, round + 1);
        if (round + 3 < rounds) {
            load_round_w(s1, round + 3);
            load_round_x(s1, round + 3);
        }
    }

    // C fragment: c0 / c1 weight row g, tokens 2 t4 and 2 t4 + 1; c2 / c3 row g + 8.
#pragma unroll
    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
        for (int nt = 0; nt < kNT; ++nt) {
            const int tok = (nt * 8) + (t4 * 2);
            const int r = (mt * 16) + g;
            red[warp][tok][r] = acc[mt][nt][0];
            red[warp][tok + 1][r] = acc[mt][nt][1];
            red[warp][tok][r + 8] = acc[mt][nt][2];
            red[warp][tok + 1][r + 8] = acc[mt][nt][3];
        }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < kNT * 8 * kR; i += kWarps * 32) {
        const int tok = i / kR;
        const int r = i % kR;
        if (tok >= M)
            continue;
        float sum = 0.0f;
#pragma unroll
        for (int ww = 0; ww < kWarps; ++ww)
            sum += red[ww][tok][r];
        half* o = y + (static_cast<int64_t>(tok) * N) + n0 + r;
        float v = sum * ts;
        if (beta != 0.0f)
            v += beta * __half2float(*o);
        *o = __float2half(v);
    }
}

template <int kMT, int kNT, int kU>
__global__ void __launch_bounds__(kWarps * 32) nvfp4_smallm_ws_kernel(
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ wsf, const uint8_t* __restrict__ x,
    const uint8_t* __restrict__ xsf, float ts, float beta, half* __restrict__ y, int M, int N, int K) {
    ws_cta<kMT, kNT, kU>(w, wsf, x, xsf, ts, beta, y, M, N, K, static_cast<int>(blockIdx.x) * kMT * 16);
}

// Up to kSmallMV2MaxSiblings weights on one activation (q|k|v, gate|up), beta 0: grid.x walks their row tiles
// back to back; constant-index selects keep the argument struct out of local memory.
struct WsMultiArgs {
    const uint8_t* w[kSmallMV2MaxSiblings];
    const uint8_t* s[kSmallMV2MaxSiblings];
    half* y[kSmallMV2MaxSiblings];
    float ts[kSmallMV2MaxSiblings];
    int N[kSmallMV2MaxSiblings];
    int tiles_end[kSmallMV2MaxSiblings];  // inclusive prefix sums of row tiles
};

template <int kMT, int kNT, int kU>
__global__ void __launch_bounds__(kWarps * 32) nvfp4_smallm_ws_multi_kernel(WsMultiArgs a,
                                                                            const uint8_t* __restrict__ x,
                                                                            const uint8_t* __restrict__ xsf,
                                                                            int M, int K) {
    const int t = static_cast<int>(blockIdx.x);
    const bool t1 = t >= a.tiles_end[0];
    const bool t2 = t >= a.tiles_end[1];
    const int first = t2 ? a.tiles_end[1] : (t1 ? a.tiles_end[0] : 0);
    const uint8_t* w = t2 ? a.w[2] : (t1 ? a.w[1] : a.w[0]);
    const uint8_t* s = t2 ? a.s[2] : (t1 ? a.s[1] : a.s[0]);
    half* y = t2 ? a.y[2] : (t1 ? a.y[1] : a.y[0]);
    const float ts = t2 ? a.ts[2] : (t1 ? a.ts[1] : a.ts[0]);
    const int N = t2 ? a.N[2] : (t1 ? a.N[1] : a.N[0]);
    ws_cta<kMT, kNT, kU>(w, s, x, xsf, ts, 0.0f, y, M, N, K, (t - first) * kMT * 16);
}

bool ptr_ok(const void* p) { return p != nullptr && (reinterpret_cast<uintptr_t>(p) % 16) == 0; }

bool scale_ok(const void* p) { return p != nullptr && (reinterpret_cast<uintptr_t>(p) % 4) == 0; }

bool args_ok(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, const half* y, int M, int N, int K,
             int rows) {
    if (M < 1 || M > kNvfp4SmallMWsMaxM || N < rows || (N % rows) != 0 || K < 64 || (K % 64) != 0)
        return false;
    return ptr_ok(W.packed_data) && scale_ok(W.micro_scales) && ptr_ok(Xq.packed_data) &&
           scale_ok(Xq.micro_scales) && y != nullptr;
}

template <int kMT, int kU, typename Launch>
void by_m(int M, Launch&& go) {
    if (M <= 8)
        go(nvfp4_smallm_ws_kernel<kMT, 1, kU>, nvfp4_smallm_ws_multi_kernel<kMT, 1, kU>);
    else if (M <= 16)
        go(nvfp4_smallm_ws_kernel<kMT, 2, kU>, nvfp4_smallm_ws_multi_kernel<kMT, 2, kU>);
    else
        go(nvfp4_smallm_ws_kernel<kMT, 4, kU>, nvfp4_smallm_ws_multi_kernel<kMT, 4, kU>);
}

template <typename Launch>
bool by_cfg(int rows, int unroll, int M, Launch&& go) {
    if (rows == 16 && unroll == 1)
        by_m<1, 1>(M, go);
    else if (rows == 16 && unroll == 2)
        by_m<1, 2>(M, go);
    else if (rows == 32 && unroll == 1)
        by_m<2, 1>(M, go);
    else if (rows == 32 && unroll == 2)
        by_m<2, 2>(M, go);
    else
        return false;
    return true;
}

}  // namespace

// CTA config per (M, combined N); bits do not depend on it. Graph replay, cold weights, M 16, 32 rows x u2:
// Gemma-4 o 4.85 us, gate|up 4.97, down 3.05, Qwen3-8B down 20.68 (64 rows: 6.34 / 4.82 / 2.87 / 22.53).
// M > 16: u1 above N 4096 (32 x 4 tokens x u2 holds 254 registers).
WsCfg gemm_nvfp4_smallm_ws_cfg(int M, int N_total, int n_gcd) {
    const bool m32 = n_gcd % 32 == 0;
    const bool big = N_total >= 4096;
    if (M <= 16)
        return m32 ? WsCfg{32, 2} : WsCfg{16, 2};
    if (!m32)
        return WsCfg{16, 1};
    return big ? WsCfg{32, 1} : WsCfg{32, 2};
}

bool gemm_nvfp4_smallm_ws_supports(int N, int K) { return N >= 16 && N % 16 == 0 && K >= 64 && K % 64 == 0; }

bool gemm_nvfp4_smallm_ws_tuned(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M, int N,
                                int K, float beta, cudaStream_t stream, int rows, int unroll) {
    if (!args_ok(W, Xq, y, M, N, K, rows))
        return false;
    const float ts = W.tensor_scale * Xq.tensor_scale;
    const auto* w = static_cast<const uint8_t*>(W.packed_data);
    const auto* ws = static_cast<const uint8_t*>(W.micro_scales);
    const auto* x = static_cast<const uint8_t*>(Xq.packed_data);
    const auto* xs = static_cast<const uint8_t*>(Xq.micro_scales);
    return by_cfg(rows, unroll, M, [&](auto k, auto) {
        pdl::enable_kernel(k);
        pdl::launch(k, dim3(N / rows), dim3(kWarps * 32), size_t(0), stream, w, ws, x, xs, ts, beta, y, M, N,
                    K);
        IMP_CUDA_CHECK_LAUNCH();
    });
}

bool gemm_nvfp4_smallm_ws(const NvFP4QuantResult& W, const NvFP4QuantResult& Xq, half* y, int M, int N, int K,
                          float beta, cudaStream_t stream) {
    const WsCfg c = gemm_nvfp4_smallm_ws_cfg(M, N, N);
    return gemm_nvfp4_smallm_ws_tuned(W, Xq, y, M, N, K, beta, stream, c.rows, c.unroll);
}

bool gemm_nvfp4_smallm_ws_multi_tuned(const SmallMV2Sibling* t, int count, const NvFP4QuantResult& Xq, int M,
                                      int K, cudaStream_t stream, int rows, int unroll) {
    if (count < 2 || count > kSmallMV2MaxSiblings)
        return false;
    WsMultiArgs a{};
    int tiles = 0;
    for (int i = 0; i < count; ++i) {
        if (t[i].w == nullptr || !args_ok(*t[i].w, Xq, t[i].y, M, t[i].N, K, rows))
            return false;
        a.w[i] = static_cast<const uint8_t*>(t[i].w->packed_data);
        a.s[i] = static_cast<const uint8_t*>(t[i].w->micro_scales);
        a.y[i] = t[i].y;
        a.ts[i] = t[i].w->tensor_scale * Xq.tensor_scale;
        a.N[i] = t[i].N;
        tiles += t[i].N / rows;
        a.tiles_end[i] = tiles;
    }
    for (int i = count; i < kSmallMV2MaxSiblings; ++i)
        a.tiles_end[i] = tiles;
    const auto* x = static_cast<const uint8_t*>(Xq.packed_data);
    const auto* xs = static_cast<const uint8_t*>(Xq.micro_scales);
    return by_cfg(rows, unroll, M, [&](auto, auto k) {
        pdl::enable_kernel(k);
        pdl::launch(k, dim3(tiles), dim3(kWarps * 32), size_t(0), stream, a, x, xs, M, K);
        IMP_CUDA_CHECK_LAUNCH();
    });
}

bool gemm_nvfp4_smallm_ws_multi(const SmallMV2Sibling* t, int count, const NvFP4QuantResult& Xq, int M, int K,
                                cudaStream_t stream) {
    int n_total = 0;
    int n_gcd = 64;  // only divisibility by 16 / 32 / 64 matters
    for (int i = 0; i < count && i < kSmallMV2MaxSiblings; ++i) {
        n_total += t[i].N;
        while (n_gcd > 16 && t[i].N % n_gcd != 0)
            n_gcd /= 2;
    }
    const WsCfg c = gemm_nvfp4_smallm_ws_cfg(M, n_total, n_gcd);
    return gemm_nvfp4_smallm_ws_multi_tuned(t, count, Xq, M, K, stream, c.rows, c.unroll);
}

}  // namespace imp
