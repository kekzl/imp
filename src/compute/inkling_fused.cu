// Inkling2-Loop fused per-site kernels (inkling_ops.h): one launch where the reference has a chain of
// projection post-processing, so a decode step is about 1000 graph nodes instead of 2400.

#include "compute/inkling_ops.h"

#include "compute/inkling_detail.cuh"
#include "core/logging.h"

#include <algorithm>

namespace imp::inkling {

namespace {

using namespace detail;

constexpr int kGateMaxRank = 128;

constexpr int kMixSlice = 128;  // elements per block
constexpr int kMixVals = 45;    // 5 x 5 Gram matrix of the states, then 4 x 5 query dots

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
        for (int j = 0; j < a.st.n; ++j)
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

// Mix, step 2. Same grid: sum the slices' partials, softmax per query, write this slice. The RMS of
// v_q comes from the Gram matrix (|v|^2 = c^T G c), so no pass over v is needed.
__global__ void mix_out_k(MixArgs a, int H, int slices) {
    __shared__ float tot[kMixVals];
    const int t = blockIdx.x / slices, b = blockIdx.x % slices;
    if (threadIdx.x < kMixVals) {
        float v = 0.0f;
        for (int k = 0; k < slices; ++k)
            v += a.partial[((size_t)t * slices + k) * kMixVals + threadIdx.x];
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

// Gate, step 1. One 32-thread block per (token, gate, row): rows 0..rank-1 are the raw dots of a
// down row with h and with msg, row `rank` is the pair of unit-RMS factors. Many small blocks: one
// block reads only a few GB/s, and a gate's weights are 1.2 MB.
// gd per (token, gate): [rank] dots with h, [rank] dots with msg, rh, rm.
__global__ void gate_down_k(GateNormArgs a, float* gd, int H) {
    const int rows = a.rank + 1;
    const int tg = blockIdx.x / rows, r = blockIdx.x % rows;
    const int t = tg / a.n, g = tg % a.n;
    const float* h = a.h[g] + (size_t)t * H;
    const float* msg = a.msg[g] + (size_t)t * H;
    float* o = gd + (size_t)tg * (2 * a.rank + 2);
    float sh = 0.0f, sm = 0.0f;
    if (r == a.rank) {
        for (int i = threadIdx.x; i < H; i += 32) {
            sh += h[i] * h[i];
            sm += msg[i] * msg[i];
        }
        sh = warp_sum(sh), sm = warp_sum(sm);
        if (threadIdx.x == 0) {
            o[(size_t)2 * a.rank] = rsqrtf(sh / H + a.eps);
            o[(size_t)2 * a.rank + 1] = rsqrtf(sm / H + a.eps);
        }
        return;
    }
    const half* d = a.down[g] + (size_t)r * 2 * H;
    for (int i = threadIdx.x; i < H; i += 32) {
        sh += __half2float(d[i]) * h[i];
        sm += __half2float(d[H + i]) * msg[i];
    }
    sh = warp_sum(sh), sm = warp_sum(sm);
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
    for (int r = 0; r < a.rank; ++r) {
        f += __half2float(fw[r]) * gs[r];
        gn += __half2float(gw[r]) * gs[r];
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
        const int slot = a.ring > 0 ? (pos0 + t) % a.ring : pos0 + t;
        a.cache[((size_t)slot * a.n_heads + head) * a.hd + threadIdx.x] = __float2half(y * f);
        if (a.log)
            a.log[(size_t)t * C + c] = x;
        s0 = s1, s1 = s2, s2 = x;
    }
    a.state[c] = s0, a.state[(size_t)C + c] = s1, a.state[(size_t)2 * C + c] = s2;
}

constexpr int kAttnThreads = 64;
constexpr int kAttnMaxHd = 128;

// One block per (token, head): each thread runs an online softmax over its strided keys.
__global__ void attention_k(AttnArgs a) {
    __shared__ float sq[kAttnMaxHd];
    __shared__ float sr[32];
    __shared__ float sh[kAttnThreads / 32];
    __shared__ float sacc[kAttnThreads * kAttnMaxHd];
    const int t = blockIdx.x / a.n_heads, head = blockIdx.x % a.n_heads;
    const int kvh = head / (a.n_heads / a.n_kv);
    const int pos = *a.d_pos + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const size_t row = (size_t)t * a.stride;
    const float tau = a.log_alpha > 0.0f
                          ? 1.0f + a.log_alpha * logf(fmaxf(static_cast<float>(pos + 1) / a.log_floor, 1.0f))
                          : 1.0f;
    // q = clip(q * scale) * tau
    float ss = 0.0f;
    for (int i = threadIdx.x; i < a.hd; i += blockDim.x) {
        sq[i] = __half2float(a.q[row + (size_t)head * a.hd + i]) * a.q_scale[head * a.hd + i];
        ss += sq[i] * sq[i];
    }
    if ((int)threadIdx.x < a.d_rel)
        sr[threadIdx.x] = __half2float(a.rel[row + (size_t)head * a.d_rel + threadIdx.x]) *
                          a.rel_scale[head * a.d_rel + threadIdx.x];
    const float qf = fminf(a.q_thr / (sqrtf(block_sum(ss, sh)) + 1e-6f), 1.0f) * tau * a.scale;
    float rss = 0.0f;
    for (int i = 0; i < a.d_rel; ++i)
        rss += sr[i] * sr[i];
    const float rf = fminf(a.rel_thr / (sqrtf(rss) + 1e-6f), 1.0f) * tau;

    float acc[kAttnMaxHd];
    for (int i = 0; i < a.hd; ++i)
        acc[i] = 0.0f;
    float m = -FLT_MAX, l = 0.0f;
    for (int k = lo + threadIdx.x; k <= pos; k += blockDim.x) {
        const int slot = a.ring > 0 ? k % a.ring : k;
        const half* kp = a.k + ((size_t)slot * a.n_kv + kvh) * a.hd;
        const half* vp = a.v + ((size_t)slot * a.n_kv + kvh) * a.hd;
        float dot = 0.0f;
        for (int i = 0; i < a.hd; ++i)
            dot += sq[i] * __half2float(kp[i]);
        const int d = pos - k;
        float bias = 0.0f;
        if (d < a.rel_extent)
            for (int i = 0; i < a.d_rel; ++i)
                bias += sr[i] * a.proj[(size_t)d * a.d_rel + i];
        const float sc = dot * qf + bias * rf;
        if (sc > m) {
            const float f = expf(m - sc);
            l *= f;
            for (int i = 0; i < a.hd; ++i)
                acc[i] *= f;
            m = sc;
        }
        const float p = expf(sc - m);
        l += p;
        for (int i = 0; i < a.hd; ++i)
            acc[i] += p * __half2float(vp[i]);
    }
    const float sink = __half2float(a.sink[row + head]);
    const float gm = fmaxf(block_max(m, sh), sink);
    const float f = l > 0.0f ? expf(m - gm) : 0.0f;
    const float total = block_sum(l * f, sh) + expf(sink - gm);
    for (int i = 0; i < a.hd; ++i)
        sacc[threadIdx.x * kAttnMaxHd + i] = acc[i] * f;
    __syncthreads();
    for (int i = threadIdx.x; i < a.hd; i += blockDim.x) {
        float v = 0.0f;
        for (int th = 0; th < (int)blockDim.x; ++th)
            v += sacc[th * kAttnMaxHd + i];
        const float g = __half2float(a.gate[row + (size_t)head * a.hd + i]);
        a.out[((size_t)t * a.n_heads + head) * a.hd + i] = __float2half(v / total / (1.0f + expf(-g)));
    }
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
    const int slot = a.ring > 0 ? pos % a.ring : pos;
    a.cache[((size_t)slot * a.n_heads + head) * a.hd + threadIdx.x] = __float2half(y * f);
}

// state <- the last three inputs after T rows.
__global__ void state_from_h_k(const half* x, float* state, int T, int C, int stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float v[3];
    for (int i = 0; i < 3; ++i)
        v[i] = tap_h(x, state, T - 3 + i, c, C, stride);
    for (int i = 0; i < 3; ++i)
        state[(size_t)i * C + c] = v[i];
}
__global__ void state_from_f_k(const float* x, float* state, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float v[3];
    for (int i = 0; i < 3; ++i)
        v[i] = tap_f(x, state, T - 3 + i, c, C);
    for (int i = 0; i < 3; ++i)
        state[(size_t)i * C + c] = v[i];
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

constexpr int kSplitMaxT = 8;

// Decode attention, step 1. One warp per (token, head, key chunk). Keys go 32 at a time, one per lane
// (their loads overlap; one key after the other cost about 1 us each): scores, then the value sums by
// one warp reduction per dim, kept by the lane that owns the dim.
// part per (token, head, chunk): running max, sum, then hd weighted value sums.
__global__ void attn_part_k(AttnArgs a, int S, int CK) {
    __shared__ float sq[kAttnMaxHd];
    __shared__ float sr[32];
    __shared__ float sp[32];
    const int c = blockIdx.x % S, th = blockIdx.x / S;
    const int t = th / a.n_heads, head = th % a.n_heads;
    const int kvh = head / (a.n_heads / a.n_kv);
    const int lane = threadIdx.x, per = a.hd / 32;
    float* o = a.part + ((size_t)th * S + c) * (a.hd + 2);
    const int pos = *a.d_pos + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int k0 = lo + c * CK, k1 = min(pos, k0 + CK - 1);
    float m = -FLT_MAX, l = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (k0 <= pos) {
        const size_t row = (size_t)t * a.stride;
        const float tau = a.log_alpha > 0.0f
                              ? 1.0f +
                                    a.log_alpha * logf(fmaxf(static_cast<float>(pos + 1) / a.log_floor, 1.0f))
                              : 1.0f;
        float ss = 0.0f;
        for (int j = 0; j < per; ++j) {
            const int i = lane * per + j;
            const float q = __half2float(a.q[row + (size_t)head * a.hd + i]) * a.q_scale[head * a.hd + i];
            sq[i] = q;
            ss += q * q;
        }
        const float qf = fminf(a.q_thr / (sqrtf(warp_sum(ss)) + 1e-6f), 1.0f) * tau * a.scale;
        const float r = lane < a.d_rel ? __half2float(a.rel[row + (size_t)head * a.d_rel + lane]) *
                                             a.rel_scale[head * a.d_rel + lane]
                                       : 0.0f;
        sr[lane] = r;
        const float rf = fminf(a.rel_thr / (sqrtf(warp_sum(r * r)) + 1e-6f), 1.0f) * tau;
        __syncwarp();
        for (int g0 = k0; g0 <= k1; g0 += 32) {
            const int k = g0 + lane;
            const bool valid = k <= k1;
            const int slot = a.ring > 0 ? k % a.ring : k;
            const size_t at = ((size_t)slot * a.n_kv + kvh) * a.hd;
            float sc = -FLT_MAX;
            if (valid) {
                float dot = 0.0f;
                for (int i = 0; i < a.hd; ++i)
                    dot += sq[i] * __half2float(a.k[at + i]);
                float bias = 0.0f;
                const int d = pos - k;
                if (d < a.rel_extent)
                    for (int i = 0; i < a.d_rel; ++i)
                        bias += sr[i] * a.proj[(size_t)d * a.d_rel + i];
                sc = dot * qf + bias * rf;
            }
            float gm = sc;
            for (int off = 16; off > 0; off >>= 1)
                gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, off));
            if (gm > m) {
                const float f = expf(m - gm);
                l *= f;
                for (int j = 0; j < per; ++j)
                    acc[j] *= f;
                m = gm;
            }
            const float p = valid ? expf(sc - m) : 0.0f;
            l += warp_sum(p);
            // Values: each lane owns `per` dims and walks the group's keys (one coalesced row per key).
            sp[lane] = p;
            __syncwarp();
            const int n_g = min(32, k1 - g0 + 1);
            for (int kk = 0; kk < n_g; ++kk) {
                const int vs = a.ring > 0 ? (g0 + kk) % a.ring : g0 + kk;
                const half* vp = a.v + ((size_t)vs * a.n_kv + kvh) * a.hd + (size_t)lane * per;
                for (int j = 0; j < per; ++j)
                    acc[j] += sp[kk] * __half2float(vp[j]);
            }
            __syncwarp();
        }
    }
    for (int j = 0; j < per; ++j)
        o[2 + lane * per + j] = acc[j];
    if (lane == 0) {
        o[0] = m;
        o[1] = l;
    }
}

// Decode attention, step 2. One block per (token, head), one thread per dim: joins the chunks that
// hold keys, adds the sink to the denominator, applies the output gate.
__global__ void attn_reduce_k(AttnArgs a, int S, int CK) {
    const int t = blockIdx.x / a.n_heads, head = blockIdx.x % a.n_heads;
    const int i = threadIdx.x;
    const int pos = *a.d_pos + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int n_c = min(S, (pos - lo) / CK + 1);
    const float* p0 = a.part + (size_t)blockIdx.x * S * (a.hd + 2);
    const size_t row = (size_t)t * a.stride;
    const float sink = __half2float(a.sink[row + head]);
    float gm = sink;
    for (int c = 0; c < n_c; ++c)
        gm = fmaxf(gm, p0[(size_t)c * (a.hd + 2)]);
    float total = expf(sink - gm), v = 0.0f;
    for (int c = 0; c < n_c; ++c) {
        const float* pc = p0 + (size_t)c * (a.hd + 2);
        if (pc[1] <= 0.0f)
            continue;
        const float w = expf(pc[0] - gm);
        total += pc[1] * w;
        v += pc[2 + i] * w;
    }
    const float g = __half2float(a.gate[row + (size_t)head * a.hd + i]);
    a.out[((size_t)t * a.n_heads + head) * a.hd + i] = __float2half(v / total / (1.0f + expf(-g)));
}

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
        const float a = (wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * v + v) * 0.4472135954999579f;
        const float r = resid[i];
        if (copy)
            copy[i] = r;
        out[i] = add ? r + a : a;
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
        const float a = (wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * v + v) * 0.4472135954999579f;
        out[(size_t)t * C + c] = resid[(size_t)t * C + c] + a;
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
    if (a.st.n < 1 || a.st.n > 5 || a.nq < 1 || a.nq > 4 || !a.partial) {
        IMP_LOG_ERROR("inkling mix: %d states or %d queries out of range", a.st.n, a.nq);
        return;
    }
    const int slices = (H + kMixSlice - 1) / kMixSlice;
    mix_partial_k<<<T * slices, kMixSlice, 0, s>>>(a, H, slices);
    IMP_CUDA_CHECK_LAUNCH();
    mix_out_k<<<T * slices, kMixSlice, 0, s>>>(a, H, slices);
    IMP_CUDA_CHECK_LAUNCH();
}

void gate_norm(const GateNormArgs& a, const GateOut& o, float* gd, int T, int H, cudaStream_t s) {
    if (a.rank > kGateMaxRank || a.n < 1 || a.n > 3) {
        IMP_LOG_ERROR("inkling gate_norm: rank %d or %d gates out of range", a.rank, a.n);
        return;
    }
    const int col_blocks = (H + 31) / 32;
    gate_down_k<<<T * a.n*(a.rank + 1), 32, 0, s>>>(a, gd, H);
    IMP_CUDA_CHECK_LAUNCH();
    gate_mix_k<<<T * a.n * col_blocks, 32, 0, s>>>(a, gd, o, H, col_blocks);
    IMP_CUDA_CHECK_LAUNCH();
    gate_rms_k<<<T * a.n, kThreads, 0, s>>>(a, o, H);
    IMP_CUDA_CHECK_LAUNCH();
}

void kv_site(const KvSiteArgs& a, cudaStream_t s) {
    if (a.hd > 128 || a.hd % 32 != 0) {
        IMP_LOG_ERROR("inkling kv_site: head_dim %d out of range", a.hd);
        return;
    }
    if (a.T > kRowLoopMaxT) {
        kv_site_rows_k<<<a.T * a.n_heads, a.hd, 0, s>>>(a);
        IMP_CUDA_CHECK_LAUNCH();
        const int C = a.n_heads * a.hd;
        state_from_h_k<<<(C + kThreads - 1) / kThreads, kThreads, 0, s>>>(a.x, a.state, a.T, C, a.stride);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    kv_site_k<<<a.n_heads, a.hd, 0, s>>>(a);
    IMP_CUDA_CHECK_LAUNCH();
}

void attention(const AttnArgs& a, cudaStream_t s) {
    if (a.hd > kAttnMaxHd || a.d_rel > 32) {
        IMP_LOG_ERROR("inkling attention: head_dim %d or d_rel %d out of range", a.hd, a.d_rel);
        return;
    }
    // Few rows (decode, verify): chunked over the keys. Prefill has rows x heads blocks already.
    if (a.part && a.T <= kSplitMaxT && a.hd % 32 == 0 && a.hd <= 128 && a.d_rel <= 32) {
        const int CK = a.window > 0 ? 32 : 64;
        const int S = ((a.window > 0 ? a.window : a.max_ctx) + CK - 1) / CK;
        attn_part_k<<<a.T * a.n_heads * S, 32, 0, s>>>(a, S, CK);
        IMP_CUDA_CHECK_LAUNCH();
        attn_reduce_k<<<a.T * a.n_heads, a.hd, 0, s>>>(a, S, CK);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    attention_k<<<a.T * a.n_heads, kAttnThreads, 0, s>>>(a);
    IMP_CUDA_CHECK_LAUNCH();
}

size_t attention_part_floats(int n_heads, int hd, int window, int max_ctx) {
    const size_t chunks = static_cast<size_t>(std::max((window + 31) / 32, (max_ctx + 63) / 64));
    return static_cast<size_t>(kSplitMaxT) * n_heads * chunks * (hd + 2);
}

void conv_residual(const half* x, const float* w, float* state, const float* resid, bool add, float* out,
                   float* copy, int T, int C, cudaStream_t s, float* log) {
    if (T > kRowLoopMaxT) {
        const size_t n = static_cast<size_t>(T) * C;
        conv_residual_rows_k<<<static_cast<unsigned>((n + kThreads - 1) / kThreads), kThreads, 0, s>>>(
            x, w, state, resid, add, out, copy, n, C);
        IMP_CUDA_CHECK_LAUNCH();
        state_from_h_k<<<(C + kThreads - 1) / kThreads, kThreads, 0, s>>>(x, state, T, C, C);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    conv_residual_k<<<(C + kThreads - 1) / kThreads, kThreads, 0, s>>>(x, w, state, resid, add, out, copy, T,
                                                                       C, log);
    IMP_CUDA_CHECK_LAUNCH();
}

void moe_tail(const half* ey, const float* weights, const float* w, float* state, const float* resid,
              float* out, float* scratch, int T, int top_k, int C, cudaStream_t s, float* log) {
    if (T > kRowLoopMaxT) {
        const size_t n = static_cast<size_t>(T) * C;
        const unsigned grid = static_cast<unsigned>((n + kThreads - 1) / kThreads);
        expert_combine_k<<<grid, kThreads, 0, s>>>(ey, weights, scratch, n, top_k, C);
        IMP_CUDA_CHECK_LAUNCH();
        conv_tail_rows_k<<<grid, kThreads, 0, s>>>(scratch, w, state, resid, out, n, C);
        IMP_CUDA_CHECK_LAUNCH();
        state_from_f_k<<<(C + kThreads - 1) / kThreads, kThreads, 0, s>>>(scratch, state, T, C);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    moe_tail_k<<<(C + kThreads - 1) / kThreads, kThreads, 0, s>>>(ey, weights, w, state, resid, out, T, top_k,
                                                                  C, log);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp::inkling
