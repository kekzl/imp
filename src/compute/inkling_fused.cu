// Inkling2-Loop fused per-site kernels (inkling_ops.h): one launch where the reference has a chain of
// projection post-processing, so a decode step is about 1000 graph nodes instead of 2400.

#include "compute/inkling_ops.h"

#include "compute/inkling_detail.cuh"
#include "compute/inkling_fused.cuh"
#include "core/logging.h"

#include <algorithm>

namespace imp::inkling {

namespace {

using namespace detail;

// Mix, step 1. One block per (token, 128-element slice), one thread per element: this slice's share
// of the Gram matrix and of the query dots. Many small blocks: one block reads only a few GB/s.
__global__ void mix_partial_k(MixArgs a, int H, int slices) {
    __shared__ float red[kMixVals * (kMixSlice / 32)];
    const int t = blockIdx.x / slices, b = blockIdx.x % slices;
    const int i = b * kMixSlice + threadIdx.x;
    float acc[kMixVals];
    for (int k = 0; k < kMixVals; ++k)
        acc[k] = 0.0f;
    if (i < H) {
        float x[5];
        int own = a.st.n;  // states read from memory
        if (a.pre.kind != 0) {
            float r = 0.0f;
            x[a.st.n - 1] = mix_pre(a.pre, i, H, r);
            own = a.st.n - 1;
            if (a.pre.copy)
                x[--own] = r;
        }
        for (int j = 0; j < own; ++j)
            x[j] = a.st.s[j][(size_t)t * H + i];
        for (int j = 0; j < a.st.n; ++j)
            for (int k = j; k < a.st.n; ++k)
                acc[j * 5 + k] = x[j] * x[k];
        if (a.mix)
            for (int q = 0; q < a.nq; ++q)
                for (int j = 0; j < a.st.n; ++j)
                    acc[25 + q * 5 + j] = a.query[(size_t)q * H + i] * x[j];
    }
    // Reduce only the entries in use: the upper Gram triangle and nq x n dots.
    float cmp[kMixVals];
    int K = 0;
    for (int j = 0; j < a.st.n; ++j)
        for (int k = j; k < a.st.n; ++k)
            cmp[K++] = acc[j * 5 + k];
    for (int q = 0; a.mix && q < a.nq; ++q)
        for (int j = 0; j < a.st.n; ++j)
            cmp[K++] = acc[25 + q * 5 + j];
    block_sum_many(cmp, K, red);
    K = 0;
    for (int j = 0; j < a.st.n; ++j)
        for (int k = j; k < a.st.n; ++k)
            acc[j * 5 + k] = cmp[K++];
    for (int q = 0; a.mix && q < a.nq; ++q)
        for (int j = 0; j < a.st.n; ++j)
            acc[25 + q * 5 + j] = cmp[K++];
    if (threadIdx.x < kMixVals)
        a.partial[((size_t)t * slices + b) * kMixVals + threadIdx.x] = acc[threadIdx.x];
}

// Mix, step 1 for calls of many rows. One block per row, kThreads threads striding the row: the row's
// Gram matrix and query dots in one pass (the slice form reduces 45 values per 128 elements: 289 us per
// call of 778 rows). Totals at partial[t * kMixVals].
__global__ void mix_rows_partial_k(MixArgs a, int H) {
    __shared__ float red[kMixVals * (kThreads / 32)];
    const int t = blockIdx.x, n = a.st.n;
    float g[15], d[20];
#pragma unroll
    for (int k = 0; k < 15; ++k)
        g[k] = 0.0f;
#pragma unroll
    for (int k = 0; k < 20; ++k)
        d[k] = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float x[5];
#pragma unroll
        for (int j = 0; j < 5; ++j)
            x[j] = j < n ? a.st.s[j][(size_t)t * H + i] : 0.0f;
        int K = 0;
#pragma unroll
        for (int j = 0; j < 5; ++j)
#pragma unroll
            for (int k = j; k < 5; ++k)
                g[K++] += x[j] * x[k];
        if (a.mix) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float qv = q < a.nq ? a.query[(size_t)q * H + i] : 0.0f;
#pragma unroll
                for (int j = 0; j < 5; ++j)
                    d[q * 5 + j] += qv * x[j];
            }
        }
    }
    // The layout mix_out_k reads: Gram entry (j, k) at j * 5 + k, query dots at 25 + q * 5 + j.
    float acc[kMixVals];
    int K = 0;
#pragma unroll
    for (int j = 0; j < 5; ++j)
#pragma unroll
        for (int k = 0; k < 5; ++k)
            acc[j * 5 + k] = k >= j ? g[K++] : 0.0f;
#pragma unroll
    for (int k = 0; k < 20; ++k)
        acc[25 + k] = d[k];
    block_sum_many(acc, kMixVals, red);
    if (threadIdx.x < kMixVals)
        a.partial[(size_t)t * kMixVals + threadIdx.x] = acc[threadIdx.x];
}

// Mix, step 2. Same grid: sum the slices' partials, softmax per query, write this slice. The RMS of
// v_q comes from the Gram matrix (|v|^2 = c^T G c), so no pass over v is needed.
// ps: partial rows per token (slices, or 1 after mix_rows_partial_k).
__global__ void mix_out_k(MixArgs a, int H, int slices, int ps) {
    __shared__ float tot[kMixVals];
    const int t = blockIdx.x / slices, b = blockIdx.x % slices;
    if (threadIdx.x < kMixVals) {
        float v = 0.0f;
        for (int k = 0; k < ps; ++k)
            v += a.partial[((size_t)t * ps + k) * kMixVals + threadIdx.x];
        tot[threadIdx.x] = v;
    }
    __syncthreads();
    const int i = b * kMixSlice + threadIdx.x;
    if (i >= H)
        return;
    const int n = a.st.n;
    const float scale = rsqrtf(static_cast<float>(H));
    for (int q = 0; q < a.nq; ++q) {
        float c[5];
        if (a.mix) {
            float mx = -FLT_MAX, sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                c[j] = tot[25 + q * 5 + j] * rsqrtf(tot[j * 5 + j] / H + 1e-6f) * scale;
                mx = fmaxf(mx, c[j]);
            }
            for (int j = 0; j < n; ++j) {
                c[j] = expf(c[j] - mx);
                sum += c[j];
            }
            for (int j = 0; j < n; ++j)
                c[j] = c[j] / sum * rsqrtf(tot[j * 5 + j] / H + 1e-6f);
        } else {
            for (int j = 0; j < n; ++j)
                c[j] = j == n - 1 ? 1.0f : 0.0f;
        }
        float v = 0.0f, vv = 0.0f;
        for (int j = 0; j < n; ++j) {
            v += c[j] * a.st.s[j][(size_t)t * H + i];
            vv += c[j] * c[j] * tot[j * 5 + j];
            for (int k = j + 1; k < n; ++k)
                vv += 2.0f * c[j] * c[k] * tot[j * 5 + k];
        }
        if (a.out_f[q])
            a.out_f[q][(size_t)t * H + i] = v;
        if (a.out_h[q])
            a.out_h[q][(size_t)t * H + i] = __float2half(v * rsqrtf(fmaxf(vv, 0.0f) / H + a.eps) *
                                                         a.norm_w[q][i]);
    }
}

// Gate, step 1. One block per (token, gate, row): rows 0..rank-1 are the raw dots of a down row with h
// and with msg, row `rank` is the pair of unit-RMS factors. kThreads threads per row: a thread's loads
// come one after the other (about 0.3 us each), 32 threads took 72 of them (22.8 us), 256 take 9.
// gd per (token, gate): [rank] dots with h, [rank] dots with msg, rh, rm.
__global__ void gate_down_k(GateNormArgs a, float* gd, int H) {
    const int rows = a.rank + 1;
    const int tg = blockIdx.x / rows, r = blockIdx.x % rows;
    const int t = tg / a.n, g = tg % a.n;
    const float* h = a.h[g] + (size_t)t * H;
    const float* msg = a.msg[g] + (size_t)t * H;
    float* o = gd + (size_t)tg * (2 * a.rank + 2);
    __shared__ float red[kThreads / 32];
    float sh = 0.0f, sm = 0.0f;
    if (r == a.rank) {
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            sh += h[i] * h[i];
            sm += msg[i] * msg[i];
        }
        sh = block_sum(sh, red), sm = block_sum(sm, red);
        if (threadIdx.x == 0) {
            o[(size_t)2 * a.rank] = rsqrtf(sh / H + a.eps);
            o[(size_t)2 * a.rank + 1] = rsqrtf(sm / H + a.eps);
        }
        return;
    }
    const half* d = a.down[g] + (size_t)r * 2 * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        sh += __half2float(d[i]) * h[i];
        sm += __half2float(d[H + i]) * msg[i];
    }
    sh = block_sum(sh, red), sm = block_sum(sm, red);
    if (threadIdx.x == 0) {
        o[r] = sh;
        o[a.rank + r] = sm;
    }
}

// Gate, step 2. 32-thread blocks, one thread per element: o = (1 - tanh(forget . g)) * h + tanh(gain . g) *
// msg.
__global__ void gate_mix_k(GateNormArgs a, const float* gd, GateOut out, int H, int col_blocks) {
    __shared__ float gs[kGateMaxRank];
    const int tg = blockIdx.x / col_blocks;
    const int i = (blockIdx.x % col_blocks) * 32 + threadIdx.x;
    const int t = tg / a.n, g = tg % a.n;
    const float* gr = gd + (size_t)tg * (2 * a.rank + 2);
    const float rh = gr[(size_t)2 * a.rank], rm = gr[(size_t)2 * a.rank + 1];
    for (int r = threadIdx.x; r < a.rank; r += 32)
        gs[r] = gr[r] * rh + gr[a.rank + r] * rm;
    __syncthreads();
    if (i >= H)
        return;
    const half* fw = a.forget[g] + (size_t)i * a.rank;
    const half* gw = a.gain[g] + (size_t)i * a.rank;
    float f = 0.0f, gn = 0.0f;
    if ((a.rank & 7) == 0) {
        // 8 halves per load.
        for (int r = 0; r < a.rank; r += 8) {
            const uint4 pf = *reinterpret_cast<const uint4*>(fw + r);
            const uint4 pg = *reinterpret_cast<const uint4*>(gw + r);
            for (int k = 0; k < 8; ++k) {
                f += __half2float(reinterpret_cast<const half*>(&pf)[k]) * gs[r + k];
                gn += __half2float(reinterpret_cast<const half*>(&pg)[k]) * gs[r + k];
            }
        }
    } else {
        for (int r = 0; r < a.rank; ++r) {
            f += __half2float(fw[r]) * gs[r];
            gn += __half2float(gw[r]) * gs[r];
        }
    }
    const size_t at = (size_t)t * H + i;
    out.o[g][at] = (1.0f - tanhf(f)) * a.h[g][at] + tanhf(gn) * a.msg[g][at];
}

// Gate, step 3. One block per (token, gate): out_h = fp16(rms_norm(o) * norm_w).
__global__ void gate_rms_k(GateNormArgs a, GateOut in, int H) {
    __shared__ float sh[kThreads / 32];
    const int t = blockIdx.x / a.n, g = blockIdx.x % a.n;
    const float* x = in.o[g] + (size_t)t * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        ss += x[i] * x[i];
    const float r = rsqrtf(block_sum(ss, sh) / H + a.eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        a.out_h[g][(size_t)t * H + i] = __float2half(x[i] * r * a.norm_w[g][i]);
}

// One block per head, one thread per channel of the head; rows in order (the conv state runs along them).
__global__ void kv_site_k(KvSiteArgs a) {
    __shared__ float sh[4];
    const int head = blockIdx.x, C = a.n_heads * a.hd;
    const int c = head * a.hd + threadIdx.x;
    const int pos0 = *a.d_pos;
    float s0 = a.state[c], s1 = a.state[(size_t)C + c], s2 = a.state[(size_t)2 * C + c];
    for (int t = 0; t < a.T; ++t) {
        const float x = __half2float(a.x[(size_t)t * a.stride + c]);
        float y;
        if (a.static_w) {
            const half* d = a.dyn + (size_t)t * a.stride + (size_t)head * 4;
            const float acc = (a.static_w[c] + __half2float(d[0])) * x +
                              (a.static_w[(size_t)C + c] + __half2float(d[1])) * s2 +
                              (a.static_w[(size_t)2 * C + c] + __half2float(d[2])) * s1 +
                              (a.static_w[(size_t)3 * C + c] + __half2float(d[3])) * s0;
            y = (acc + x) * (1.0f / 3.0f);
        } else {
            const float* w = a.w + (size_t)c * 4;
            y = (w[0] * s0 + w[1] * s1 + w[2] * s2 + w[3] * x + x) * 0.4472135954999579f;
        }
        const float ss = block_sum(y * y, sh);
        const float f = a.is_key ? fminf(a.thr / (sqrtf(ss) + 1e-6f), 1.0f) : rsqrtf(ss / a.hd + a.eps);
        const int slot = kv_slot(pos0 + t, a.ring, a.blocks, a.kv_block);
        a.cache[((size_t)slot * a.n_heads + head) * a.hd + threadIdx.x] = __float2half(y * f);
        if (a.log)
            a.log[(size_t)t * C + c] = x;
        s0 = s1, s1 = s2, s2 = x;
    }
    a.state[c] = s0, a.state[(size_t)C + c] = s1, a.state[(size_t)2 * C + c] = s2;
}

// ---- prefill forms: one thread per (row, channel); the per-channel row loops above are for a few rows ----

// Input u rows before row 0 (u < 0) comes from state[3 + u].
__device__ inline float tap_h(const half* x, const float* state, int u, int c, int C, int stride) {
    return u >= 0 ? __half2float(x[(size_t)u * stride + c]) : state[(size_t)(3 + u) * C + c];
}
__device__ inline float tap_f(const float* x, const float* state, int u, int c, int C) {
    return u >= 0 ? x[(size_t)u * C + c] : state[(size_t)(3 + u) * C + c];
}

// One block per (row, head).
__global__ void kv_site_rows_k(KvSiteArgs a) {
    __shared__ float sh[4];
    const int t = blockIdx.x / a.n_heads, head = blockIdx.x % a.n_heads, C = a.n_heads * a.hd;
    const int c = head * a.hd + threadIdx.x;
    const float x = tap_h(a.x, a.state, t, c, C, a.stride);
    const float s2 = tap_h(a.x, a.state, t - 1, c, C, a.stride),
                s1 = tap_h(a.x, a.state, t - 2, c, C, a.stride);
    const float s0 = tap_h(a.x, a.state, t - 3, c, C, a.stride);
    float y;
    if (a.static_w) {
        const half* d = a.dyn + (size_t)t * a.stride + (size_t)head * 4;
        const float acc = (a.static_w[c] + __half2float(d[0])) * x +
                          (a.static_w[(size_t)C + c] + __half2float(d[1])) * s2 +
                          (a.static_w[(size_t)2 * C + c] + __half2float(d[2])) * s1 +
                          (a.static_w[(size_t)3 * C + c] + __half2float(d[3])) * s0;
        y = (acc + x) * (1.0f / 3.0f);
    } else {
        const float* w = a.w + (size_t)c * 4;
        y = (w[0] * s0 + w[1] * s1 + w[2] * s2 + w[3] * x + x) * 0.4472135954999579f;
    }
    const float ss = block_sum(y * y, sh);
    const float f = a.is_key ? fminf(a.thr / (sqrtf(ss) + 1e-6f), 1.0f) : rsqrtf(ss / a.hd + a.eps);
    const int pos = *a.d_pos + t;
    const int slot = kv_slot(pos, a.ring, a.blocks, a.kv_block);
    a.cache[((size_t)slot * a.n_heads + head) * a.hd + threadIdx.x] = __float2half(y * f);
}

// dst <- the last three inputs after T rows (dst = state: the call's state update).
__global__ void state_from_h_k(const half* x, const float* state, float* dst, int T, int C, int stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float v[3];
    for (int i = 0; i < 3; ++i)
        v[i] = tap_h(x, state, T - 3 + i, c, C, stride);
    for (int i = 0; i < 3; ++i)
        dst[(size_t)i * C + c] = v[i];
}
__global__ void state_from_f_k(const float* x, const float* state, float* dst, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float v[3];
    for (int i = 0; i < 3; ++i)
        v[i] = tap_f(x, state, T - 3 + i, c, C);
    for (int i = 0; i < 3; ++i)
        dst[(size_t)i * C + c] = v[i];
}

__global__ void conv_residual_rows_k(const half* x, const float* w, const float* state, const float* resid,
                                     bool add, float* out, float* copy, size_t n, int C) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    const int c = static_cast<int>(id % C), t = static_cast<int>(id / C);
    const float* wc = w + (size_t)c * 4;
    const float v = tap_h(x, state, t, c, C, C);
    const float a = (wc[0] * tap_h(x, state, t - 3, c, C, C) + wc[1] * tap_h(x, state, t - 2, c, C, C) +
                     wc[2] * tap_h(x, state, t - 1, c, C, C) + wc[3] * v + v) *
                    0.4472135954999579f;
    const float r = resid[id];
    if (copy)
        copy[id] = r;
    out[id] = add ? r + a : a;
}

__global__ void expert_combine_k(const half* ey, const float* weights, float* y, size_t n, int top_k, int C) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    const size_t t = id / C, c = id % C;
    float v = 0.0f;
    for (int k = 0; k < top_k; ++k)
        v += weights[t * top_k + k] * __half2float(ey[(t * top_k + k) * C + c]);
    y[id] = v;
}

__global__ void conv_tail_rows_k(const float* y, const float* w, const float* state, const float* resid,
                                 float* out, size_t n, int C) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    const int c = static_cast<int>(id % C), t = static_cast<int>(id / C);
    const float* wc = w + (size_t)c * 4;
    const float v = y[id];
    const float a = (wc[0] * tap_f(y, state, t - 3, c, C) + wc[1] * tap_f(y, state, t - 2, c, C) +
                     wc[2] * tap_f(y, state, t - 1, c, C) + wc[3] * v + v) *
                    0.4472135954999579f;
    out[id] = resid[id] + a;
}

constexpr int kRowLoopMaxT = 8;  // up to here the per-channel row loops run; above, the per-row forms

// One thread per channel, rows in order.
__global__ void conv_residual_k(const half* x, const float* w, float* state, const float* resid, bool add,
                                float* out, float* copy, int T, int C, float* log) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    const float* wc = w + (size_t)c * 4;
    float s0 = state[c], s1 = state[(size_t)C + c], s2 = state[(size_t)2 * C + c];
    for (int t = 0; t < T; ++t) {
        const size_t i = (size_t)t * C + c;
        const float v = __half2float(x[i]);
        // Multiply and add stay two rounded steps; mix_pre mirrors it.
        const float a = __fmul_rn(wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * v + v, 0.4472135954999579f);
        const float r = resid[i];
        if (copy)
            copy[i] = r;
        out[i] = add ? __fadd_rn(r, a) : a;
        if (log)
            log[i] = v;
        s0 = s1, s1 = s2, s2 = v;
    }
    state[c] = s0, state[(size_t)C + c] = s1, state[(size_t)2 * C + c] = s2;
}

__global__ void moe_tail_k(const half* ey, const float* weights, const float* w, float* state,
                           const float* resid, float* out, int T, int top_k, int C, float* log) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    const float* wc = w + (size_t)c * 4;
    float s0 = state[c], s1 = state[(size_t)C + c], s2 = state[(size_t)2 * C + c];
    for (int t = 0; t < T; ++t) {
        float v = 0.0f;
        for (int k = 0; k < top_k; ++k)
            v += weights[(size_t)t * top_k + k] * __half2float(ey[((size_t)t * top_k + k) * C + c]);
        // The residual joins in the multiply's FMA; mix_pre mirrors it.
        const float sum = wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * v + v;
        out[(size_t)t * C + c] = __fmaf_rn(sum, 0.4472135954999579f, resid[(size_t)t * C + c]);
        if (log)
            log[(size_t)t * C + c] = v;
        s0 = s1, s1 = s2, s2 = v;
    }
    state[c] = s0, state[(size_t)C + c] = s1, state[(size_t)2 * C + c] = s2;
}

}  // namespace

size_t mix_partial_floats(int T, int H) {
    return static_cast<size_t>(T) * ((H + kMixSlice - 1) / kMixSlice) * kMixVals;
}

void mix(const MixArgs& a, int T, int H, cudaStream_t s) {
    const MixPre& p = a.pre;
    const bool pre_ok = p.kind == 0 || (T == 1 && a.st.n >= (p.copy ? 2 : 1) && p.out == a.st.s[a.st.n - 1] &&
                                        (!p.copy || p.copy == a.st.s[a.st.n - 2]));
    if (a.st.n < 1 || a.st.n > 5 || a.nq < 1 || a.nq > 4 || !a.partial || !pre_ok)
        throw_runtime_error(
            "inkling mix: %d states or %d queries out of range, or a conv site that does not fit", a.st.n,
            a.nq);
    const int slices = (H + kMixSlice - 1) / kMixSlice;
    const bool rows = T > kRowLoopMaxT && p.kind == 0;
    if (rows)
        mix_rows_partial_k<<<T, kThreads, 0, s>>>(a, H);
    else if (!mix_partial_fixed(a, T, H, slices, s))
        mix_partial_k<<<T * slices, kMixSlice, 0, s>>>(a, H, slices);
    IMP_CUDA_CHECK_LAUNCH();
    if (rows || !mix_out_fixed(a, T, H, slices, s))
        mix_out_k<<<T * slices, kMixSlice, 0, s>>>(a, H, slices, rows ? 1 : slices);
    IMP_CUDA_CHECK_LAUNCH();
}

void gate_rms(const GateNormArgs& a, const GateOut& o, int T, int H, cudaStream_t s) {
    if (gate_rms_fixed(a, o, T, H, s))
        return;
    gate_rms_k<<<T * a.n, kThreads, 0, s>>>(a, o, H);
    IMP_CUDA_CHECK_LAUNCH();
}

void gate_norm(const GateNormArgs& a, const GateOut& o, float* gd, int T, int H, cudaStream_t s,
               const GateWork* w) {
    if (a.rank > kGateMaxRank || a.n < 1 || a.n > 3)
        throw_runtime_error("inkling gate_norm: rank %d or %d gates out of range", a.rank, a.n);
    if (w && T > kRowLoopMaxT && gate_norm_rows(a, o, *w, T, H, s))
        return;
    const int col_blocks = (H + 31) / 32;
    if (!gate_down_fixed(a, gd, T, H, s))
        gate_down_k<<<T * a.n*(a.rank + 1), kThreads, 0, s>>>(a, gd, H);
    IMP_CUDA_CHECK_LAUNCH();
    if (!gate_mix_fixed(a, gd, o, T, H, s))
        gate_mix_k<<<T * a.n * col_blocks, 32, 0, s>>>(a, gd, o, H, col_blocks);
    IMP_CUDA_CHECK_LAUNCH();
    gate_rms(a, o, T, H, s);
}

void kv_site(const KvSiteArgs& a, cudaStream_t s) {
    if (a.hd > 128 || a.hd % 32 != 0)
        throw_runtime_error("inkling kv_site: head_dim %d out of range", a.hd);
    if (a.T > kRowLoopMaxT) {
        kv_site_rows_k<<<a.T * a.n_heads, a.hd, 0, s>>>(a);
        IMP_CUDA_CHECK_LAUNCH();
        const int C = a.n_heads * a.hd;
        const unsigned grid = (C + kThreads - 1) / kThreads;
        for (int i = 0; i < a.marks.n; ++i) {
            state_from_h_k<<<grid, kThreads, 0, s>>>(a.x, a.state, a.marks.dst[i], a.marks.row[i], C,
                                                     a.stride);
            IMP_CUDA_CHECK_LAUNCH();
        }
        state_from_h_k<<<grid, kThreads, 0, s>>>(a.x, a.state, a.state, a.T, C, a.stride);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    kv_site_k<<<a.n_heads, a.hd, 0, s>>>(a);
    IMP_CUDA_CHECK_LAUNCH();
}

void conv_residual(const half* x, const float* w, float* state, const float* resid, bool add, float* out,
                   float* copy, int T, int C, cudaStream_t s, float* log, const ConvMarks* marks) {
    if (T > kRowLoopMaxT) {
        const size_t n = static_cast<size_t>(T) * C;
        conv_residual_rows_k<<<static_cast<unsigned>((n + kThreads - 1) / kThreads), kThreads, 0, s>>>(
            x, w, state, resid, add, out, copy, n, C);
        IMP_CUDA_CHECK_LAUNCH();
        const unsigned sg = (C + kThreads - 1) / kThreads;
        for (int i = 0; marks && i < marks->n; ++i) {
            state_from_h_k<<<sg, kThreads, 0, s>>>(x, state, marks->dst[i], marks->row[i], C, C);
            IMP_CUDA_CHECK_LAUNCH();
        }
        state_from_h_k<<<sg, kThreads, 0, s>>>(x, state, state, T, C, C);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    conv_residual_k<<<(C + kThreads - 1) / kThreads, kThreads, 0, s>>>(x, w, state, resid, add, out, copy, T,
                                                                       C, log);
    IMP_CUDA_CHECK_LAUNCH();
}

void moe_tail(const half* ey, const float* weights, const float* w, float* state, const float* resid,
              float* out, float* scratch, int T, int top_k, int C, cudaStream_t s, float* log,
              const ConvMarks* marks) {
    if (T > kRowLoopMaxT) {
        const size_t n = static_cast<size_t>(T) * C;
        const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
        expert_combine_k<<<grid, kThreads, 0, s>>>(ey, weights, scratch, n, top_k, C);
        IMP_CUDA_CHECK_LAUNCH();
        conv_tail_rows_k<<<grid, kThreads, 0, s>>>(scratch, w, state, resid, out, n, C);
        IMP_CUDA_CHECK_LAUNCH();
        const unsigned sg = (C + kThreads - 1) / kThreads;
        for (int i = 0; marks && i < marks->n; ++i) {
            state_from_f_k<<<sg, kThreads, 0, s>>>(scratch, state, marks->dst[i], marks->row[i], C);
            IMP_CUDA_CHECK_LAUNCH();
        }
        state_from_f_k<<<sg, kThreads, 0, s>>>(scratch, state, state, T, C);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    moe_tail_k<<<(C + kThreads - 1) / kThreads, kThreads, 0, s>>>(ey, weights, w, state, resid, out, T, top_k,
                                                                  C, log);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp::inkling
