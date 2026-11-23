// Inkling2-Loop mix and gate kernels for the shape of a decode or lookahead call (inkling_fused.cuh): the
// state, query and iteration counts are template parameters, so a thread keeps its values in registers
// and issues its loads before the first use. One-row step 5.03 -> 4.14 ms, bit-equal to inkling_fused.cu.

#include "compute/inkling_fused.cuh"

#include "compute/inkling_detail.cuh"
#include "core/logging.h"

namespace imp::inkling {

namespace {

using namespace detail;

bool g_fixed = true;

constexpr int kGateIt = 9;  // H = 2304

// Mix, step 1 with the state count N and the query-dot count NQ (0 without the softmax) fixed at compile
// time: every value stays in a register (the runtime-bounded form kept 384 B of stack per thread).
template <int N, int NQ>
__global__ void mix_partial_t(MixArgs a, int H, int slices) {
    constexpr int G = N * (N + 1) / 2, K = G + NQ * N, W = kMixSlice / 32;
    __shared__ float red[K * W];
    const int t = blockIdx.x / slices, b = blockIdx.x % slices;
    const int i = b * kMixSlice + threadIdx.x;
    float v[K];
#pragma unroll
    for (int k = 0; k < K; ++k)
        v[k] = 0.0f;
    if (i < H) {
        float x[N];
        int own = N;
        float po = 0.0f, r = 0.0f;
        if (a.pre.kind != 0) {
            po = mix_pre(a.pre, i, H, r);
            own = a.pre.copy ? N - 2 : N - 1;
        }
#pragma unroll
        for (int j = 0; j < N; ++j)
            x[j] = j < own ? a.st.s[j][(size_t)t * H + i] : (j == N - 1 ? po : r);
        int at = 0;
#pragma unroll
        for (int j = 0; j < N; ++j)
#pragma unroll
            for (int k = j; k < N; ++k)
                v[at++] = x[j] * x[k];
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float qv = a.query[(size_t)q * H + i];
#pragma unroll
            for (int j = 0; j < N; ++j)
                v[at++] = qv * x[j];
        }
    }
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float s = warp_sum(v[k]);
        if ((threadIdx.x & 31) == 0)
            red[k * W + warp] = s;
    }
    __syncthreads();
    const int id = threadIdx.x;
    if (id >= K)
        return;
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < W; ++w)
        tot += red[id * W + w];
    // Where mix_out_k reads it: Gram entry (j, k) at j * 5 + k, query dot (q, j) at 25 + q * 5 + j.
    int pos = 25 + ((id - G) / N) * 5 + (id - G) % N;
    int base = 0;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        if (id >= base && id < base + N - j)
            pos = j * 5 + j + (id - base);
        base += N - j;
    }
    a.partial[((size_t)t * slices + b) * kMixVals + pos] = tot;
}

template <int N>
bool mix_partial_n(const MixArgs& a, int T, int H, int slices, cudaStream_t s) {
    const unsigned grid = static_cast<unsigned>(T * slices);
    switch (a.mix ? a.nq : 0) {
        case 0:
            mix_partial_t<N, 0><<<grid, kMixSlice, 0, s>>>(a, H, slices);
            IMP_CUDA_CHECK_LAUNCH();
            return true;
        case 1:
            mix_partial_t<N, 1><<<grid, kMixSlice, 0, s>>>(a, H, slices);
            IMP_CUDA_CHECK_LAUNCH();
            return true;
        case 3:
            mix_partial_t<N, 3><<<grid, kMixSlice, 0, s>>>(a, H, slices);
            IMP_CUDA_CHECK_LAUNCH();
            return true;
        case 4:
            mix_partial_t<N, 4><<<grid, kMixSlice, 0, s>>>(a, H, slices);
            IMP_CUDA_CHECK_LAUNCH();
            return true;
        default:
            return false;
    }
}

// Mix, step 2 with the softmax, N states and NQ queries, kMixPs partial rows per token: the partials, the
// states and the norm weights are loaded before their first use, once (the runtime-bounded form loaded
// the states again for every query). Same expressions in the same order.
constexpr int kMixPs = 18;  // H = 2304
template <int N, int NQ>
__global__ void mix_out_t(MixArgs a, int H) {
    __shared__ float tot[kMixVals];
    const int t = blockIdx.x / kMixPs, b = blockIdx.x % kMixPs;
    if (threadIdx.x < kMixVals) {
        const float* p = a.partial + (size_t)t * kMixPs * kMixVals + threadIdx.x;
        float pv[kMixPs];
#pragma unroll
        for (int k = 0; k < kMixPs; ++k)
            pv[k] = p[(size_t)k * kMixVals];
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < kMixPs; ++k)
            v += pv[k];
        tot[threadIdx.x] = v;
    }
    __syncthreads();
    const int i = b * kMixSlice + threadIdx.x;
    if (i >= H)
        return;
    const size_t at = (size_t)t * H + i;
    float x[N], nw[NQ];
#pragma unroll
    for (int j = 0; j < N; ++j)
        x[j] = a.st.s[j][at];
#pragma unroll
    for (int q = 0; q < NQ; ++q)
        nw[q] = a.out_h[q] ? a.norm_w[q][i] : 0.0f;
    const float scale = rsqrtf(static_cast<float>(H));
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float c[N];
        float mx = -FLT_MAX, sum = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            c[j] = tot[25 + q * 5 + j] * rsqrtf(tot[j * 5 + j] / H + 1e-6f) * scale;
            mx = fmaxf(mx, c[j]);
        }
#pragma unroll
        for (int j = 0; j < N; ++j) {
            c[j] = expf(c[j] - mx);
            sum += c[j];
        }
#pragma unroll
        for (int j = 0; j < N; ++j)
            c[j] = c[j] / sum * rsqrtf(tot[j * 5 + j] / H + 1e-6f);
        float v = 0.0f, vv = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            v += c[j] * x[j];
            vv += c[j] * c[j] * tot[j * 5 + j];
#pragma unroll
            for (int k = j + 1; k < N; ++k)
                vv += 2.0f * c[j] * c[k] * tot[j * 5 + k];
        }
        if (a.out_f[q])
            a.out_f[q][at] = v;
        if (a.out_h[q])
            a.out_h[q][at] = __float2half(v * rsqrtf(fmaxf(vv, 0.0f) / H + a.eps) * nw[q]);
    }
}

// Gate steps 1 and 3 for H = IT * kThreads: a thread issues its IT loads per array before it uses one
// (the runtime-bounded loop waits on each load in turn). Same sums in the same order.
template <int IT>
__global__ void gate_down_t(GateNormArgs a, float* gd) {
    constexpr int H = IT * kThreads;
    const int rows = a.rank + 1;
    const int tg = blockIdx.x / rows, r = blockIdx.x % rows;
    const int t = tg / a.n, g = tg % a.n;
    const float* h = a.h[g] + (size_t)t * H + threadIdx.x;
    const float* msg = a.msg[g] + (size_t)t * H + threadIdx.x;
    float* o = gd + (size_t)tg * (2 * a.rank + 2);
    __shared__ float red[kThreads / 32];
    float hv[IT], mv[IT], sh = 0.0f, sm = 0.0f;
#pragma unroll
    for (int k = 0; k < IT; ++k)
        hv[k] = h[(size_t)k * kThreads], mv[k] = msg[(size_t)k * kThreads];
    if (r == a.rank) {
#pragma unroll
        for (int k = 0; k < IT; ++k) {
            sh += hv[k] * hv[k];
            sm += mv[k] * mv[k];
        }
        sh = block_sum(sh, red), sm = block_sum(sm, red);
        if (threadIdx.x == 0) {
            o[(size_t)2 * a.rank] = rsqrtf(sh / H + a.eps);
            o[(size_t)2 * a.rank + 1] = rsqrtf(sm / H + a.eps);
        }
        return;
    }
    const half* d = a.down[g] + (size_t)r * 2 * H + threadIdx.x;
    half dh[IT], dm[IT];
#pragma unroll
    for (int k = 0; k < IT; ++k)
        dh[k] = d[(size_t)k * kThreads], dm[k] = d[(size_t)H + (size_t)k * kThreads];
#pragma unroll
    for (int k = 0; k < IT; ++k) {
        sh += __half2float(dh[k]) * hv[k];
        sm += __half2float(dm[k]) * mv[k];
    }
    sh = block_sum(sh, red), sm = block_sum(sm, red);
    if (threadIdx.x == 0) {
        o[r] = sh;
        o[a.rank + r] = sm;
    }
}

template <int IT>
__global__ void gate_rms_t(GateNormArgs a, GateOut in) {
    constexpr int H = IT * kThreads;
    __shared__ float sh[kThreads / 32];
    const int t = blockIdx.x / a.n, g = blockIdx.x % a.n;
    const float* x = in.o[g] + (size_t)t * H + threadIdx.x;
    const float* nw = a.norm_w[g] + threadIdx.x;
    float xv[IT], wv[IT], ss = 0.0f;
#pragma unroll
    for (int k = 0; k < IT; ++k)
        xv[k] = x[(size_t)k * kThreads], wv[k] = nw[(size_t)k * kThreads];
#pragma unroll
    for (int k = 0; k < IT; ++k)
        ss += xv[k] * xv[k];
    const float r = rsqrtf(block_sum(ss, sh) / H + a.eps);
    half* o = a.out_h[g] + (size_t)t * H + threadIdx.x;
#pragma unroll
    for (int k = 0; k < IT; ++k)
        o[(size_t)k * kThreads] = __float2half(xv[k] * r * wv[k]);
}

// Gate, step 2 for rank R: the 2 * R / 8 weight loads of an element are issued before the first use.
template <int R>
__global__ void gate_mix_t(GateNormArgs a, const float* gd, GateOut out, int H, int col_blocks) {
    __shared__ float gs[R];
    const int tg = blockIdx.x / col_blocks;
    const int i = (blockIdx.x % col_blocks) * 32 + threadIdx.x;
    const int t = tg / a.n, g = tg % a.n;
    const float* gr = gd + (size_t)tg * (2 * R + 2);
    const float rh = gr[(size_t)2 * R], rm = gr[(size_t)2 * R + 1];
#pragma unroll
    for (int k = 0; k < R / 32; ++k) {
        const int r = threadIdx.x + k * 32;
        gs[r] = gr[r] * rh + gr[R + r] * rm;
    }
    __syncthreads();
    if (i >= H)
        return;
    const half* fw = a.forget[g] + (size_t)i * R;
    const half* gw = a.gain[g] + (size_t)i * R;
    const size_t at = (size_t)t * H + i;
    uint4 pf[R / 8], pg[R / 8];
#pragma unroll
    for (int k = 0; k < R / 8; ++k) {
        pf[k] = *reinterpret_cast<const uint4*>(fw + (size_t)k * 8);
        pg[k] = *reinterpret_cast<const uint4*>(gw + (size_t)k * 8);
    }
    const float hv = a.h[g][at], mv = a.msg[g][at];
    float f = 0.0f, gn = 0.0f;
#pragma unroll
    for (int r = 0; r < R / 8; ++r) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            f += __half2float(reinterpret_cast<const half*>(&pf[r])[k]) * gs[r * 8 + k];
            gn += __half2float(reinterpret_cast<const half*>(&pg[r])[k]) * gs[r * 8 + k];
        }
    }
    out.o[g][at] = (1.0f - tanhf(f)) * hv + tanhf(gn) * mv;
}

template <int N>
bool mix_out_n(const MixArgs& a, int T, int H, cudaStream_t s) {
    const unsigned grid = static_cast<unsigned>(T * kMixPs);
    switch (a.nq) {
        case 1:
            mix_out_t<N, 1><<<grid, kMixSlice, 0, s>>>(a, H);
            IMP_CUDA_CHECK_LAUNCH();
            return true;
        case 3:
            mix_out_t<N, 3><<<grid, kMixSlice, 0, s>>>(a, H);
            IMP_CUDA_CHECK_LAUNCH();
            return true;
        case 4:
            mix_out_t<N, 4><<<grid, kMixSlice, 0, s>>>(a, H);
            IMP_CUDA_CHECK_LAUNCH();
            return true;
        default:
            return false;
    }
}

}  // namespace

void set_fixed_shape_kernels(bool on) { g_fixed = on; }

namespace detail {

bool mix_partial_fixed(const MixArgs& a, int T, int H, int slices, cudaStream_t s) {
    if (!g_fixed)
        return false;
    switch (a.st.n) {
        case 1:
            return mix_partial_n<1>(a, T, H, slices, s);
        case 2:
            return mix_partial_n<2>(a, T, H, slices, s);
        case 3:
            return mix_partial_n<3>(a, T, H, slices, s);
        case 4:
            return mix_partial_n<4>(a, T, H, slices, s);
        case 5:
            return mix_partial_n<5>(a, T, H, slices, s);
        default:
            return false;
    }
}

bool mix_out_fixed(const MixArgs& a, int T, int H, int slices, cudaStream_t s) {
    if (!g_fixed || !a.mix || slices != kMixPs)
        return false;
    switch (a.st.n) {
        case 1:
            return mix_out_n<1>(a, T, H, s);
        case 2:
            return mix_out_n<2>(a, T, H, s);
        case 3:
            return mix_out_n<3>(a, T, H, s);
        case 4:
            return mix_out_n<4>(a, T, H, s);
        case 5:
            return mix_out_n<5>(a, T, H, s);
        default:
            return false;
    }
}

bool gate_down_fixed(const GateNormArgs& a, float* gd, int T, int H, cudaStream_t s) {
    if (!g_fixed || H != kGateIt * kThreads)
        return false;
    gate_down_t<kGateIt><<<T * a.n*(a.rank + 1), kThreads, 0, s>>>(a, gd);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

bool gate_mix_fixed(const GateNormArgs& a, const float* gd, const GateOut& o, int T, int H, cudaStream_t s) {
    if (!g_fixed || a.rank != 64)
        return false;
    const int col_blocks = (H + 31) / 32;
    gate_mix_t<64><<<T * a.n * col_blocks, 32, 0, s>>>(a, gd, o, H, col_blocks);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

bool gate_rms_fixed(const GateNormArgs& a, const GateOut& o, int T, int H, cudaStream_t s) {
    if (!g_fixed || H != kGateIt * kThreads)
        return false;
    gate_rms_t<kGateIt><<<T * a.n, kThreads, 0, s>>>(a, o);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

}  // namespace detail

}  // namespace imp::inkling
