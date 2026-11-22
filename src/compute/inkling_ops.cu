// Inkling2-Loop building blocks (inkling_ops.h). Reference-grade kernels: one block per token or
// per (token, head), no tensor cores.

#include "compute/inkling_ops.h"

#include "core/logging.h"

#include <cfloat>

namespace imp::inkling {

namespace {

constexpr int kThreads = 256;

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum; sh holds one float per warp.
__device__ inline float block_sum(float v, float* sh) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i)
        t += sh[i];
    __syncthreads();
    return t;
}

__device__ inline float block_max(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = -FLT_MAX;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i)
        t = fmaxf(t, sh[i]);
    __syncthreads();
    return t;
}

__device__ inline float bf16_bits(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ inline float e4m3(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    const float mag = e == 0 ? scalbnf(static_cast<float>(m), -9) : scalbnf(static_cast<float>(8 + m), e - 10);
    return (b & 0x80) ? -mag : mag;
}

inline unsigned grid_for(size_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

__global__ void bf16_to_f16_k(const uint16_t* in, half* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(bf16_bits(in[i]));
}
__global__ void bf16_to_f32_k(const uint16_t* in, float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = bf16_bits(in[i]);
}
__global__ void f32_to_f16_k(const float* in, half* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(in[i]);
}
__global__ void f16_to_f32_k(const half* in, float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __half2float(in[i]);
}
__global__ void add_f32_k(const float* a, const float* b, float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = b ? a[i] + b[i] : a[i];
}

__global__ void embed_rows_k(const half* table, const int32_t* ids, float* out, int H) {
    const half* row = table + (size_t)ids[blockIdx.x] * H;
    float* o = out + (size_t)blockIdx.x * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        o[i] = __half2float(row[i]);
}

__global__ void rms_norm_k(const float* x, const float* w, float* of, half* oh, int H, float eps) {
    __shared__ float sh[kThreads / 32];
    const float* xr = x + (size_t)blockIdx.x * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = xr[i] * r * (w ? w[i] : 1.0f);
        if (of)
            of[(size_t)blockIdx.x * H + i] = v;
        if (oh)
            oh[(size_t)blockIdx.x * H + i] = __float2half(v);
    }
}

__global__ void block_mix_k(MixStates st, const float* query, int nq, MixOut out, int H) {
    __shared__ float sh[kThreads / 32];
    const size_t base = (size_t)blockIdx.x * H;
    float rs[5];
    for (int j = 0; j < st.n; ++j) {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            ss += st.s[j][base + i] * st.s[j][base + i];
        rs[j] = rsqrtf(block_sum(ss, sh) / H + 1e-6f);
    }
    const float scale = rsqrtf(static_cast<float>(H));
    for (int q = 0; q < nq; ++q) {
        float p[5];
        float mx = -FLT_MAX;
        for (int j = 0; j < st.n; ++j) {
            float d = 0.0f;
            for (int i = threadIdx.x; i < H; i += blockDim.x)
                d += query[(size_t)q * H + i] * st.s[j][base + i];
            p[j] = block_sum(d, sh) * rs[j] * scale;
            mx = fmaxf(mx, p[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < st.n; ++j) {
            p[j] = expf(p[j] - mx);
            sum += p[j];
        }
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            float v = 0.0f;
            for (int j = 0; j < st.n; ++j)
                v += p[j] / sum * rs[j] * st.s[j][base + i];
            out.o[q][base + i] = v;
        }
    }
}

__global__ void gate_cat_k(const float* h, const float* msg, half* cat, int H, float eps) {
    __shared__ float sh[kThreads / 32];
    const size_t base = (size_t)blockIdx.x * H;
    for (int part = 0; part < 2; ++part) {
        const float* x = (part == 0 ? h : msg) + base;
        float ss = 0.0f;
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            ss += x[i] * x[i];
        const float r = rsqrtf(block_sum(ss, sh) / H + eps);
        for (int i = threadIdx.x; i < H; i += blockDim.x)
            cat[(size_t)blockIdx.x * 2 * H + (size_t)part * H + i] = __float2half(x[i] * r);
    }
}

__global__ void gate_apply_k(const float* h, const float* msg, const half* forget, const half* gain,
                             float* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (1.0f - tanhf(__half2float(forget[i]))) * h[i] + tanhf(__half2float(gain[i])) * msg[i];
}

// One thread per (token, head).
__global__ void scaled_l2_clip_k(const half* x, const float* scale, float* out, int total, int n_heads,
                                 int hd, float thr, int pos0, float log_alpha, float log_floor) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total)
        return;
    const int head = id % n_heads, t = id / n_heads;
    const half* xr = x + (size_t)id * hd;
    const float* sc = scale ? scale + (size_t)head * hd : nullptr;
    float ss = 0.0f;
    for (int i = 0; i < hd; ++i) {
        const float v = __half2float(xr[i]) * (sc ? sc[i] : 1.0f);
        ss += v * v;
    }
    float f = fminf(thr / (sqrtf(ss) + 1e-6f), 1.0f);
    if (log_alpha > 0.0f)
        f *= 1.0f + log_alpha * logf(fmaxf(static_cast<float>(pos0 + t + 1) / log_floor, 1.0f));
    for (int i = 0; i < hd; ++i)
        out[(size_t)id * hd + i] = __half2float(xr[i]) * (sc ? sc[i] : 1.0f) * f;
}

// Input u tokens before this call's token 0 (u < 0) comes from state[3 + u].
__device__ inline float conv_in(const float* x, const float* state, int u, int c, int C) {
    return u >= 0 ? x[(size_t)u * C + c] : state[(size_t)(3 + u) * C + c];
}

__global__ void conv_static_k(const float* x, const float* w, const float* state, float* out, size_t n, int C) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    const int c = static_cast<int>(id % C), t = static_cast<int>(id / C);
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j)
        acc += w[(size_t)c * 4 + j] * conv_in(x, state, t - 3 + j, c, C);
    out[id] = (acc + x[id]) * 0.4472135954999579f;
}

__global__ void conv_dynamic_k(const float* x, const float* sw, const half* dyn, const float* state, float* out,
                               size_t n, int C, int n_heads) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    const int c = static_cast<int>(id % C), t = static_cast<int>(id / C);
    const int head = c / (C / n_heads);
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j)
        acc += (sw[(size_t)j * C + c] + __half2float(dyn[((size_t)t * n_heads + head) * 4 + j])) *
               conv_in(x, state, t - j, c, C);
    out[id] = (acc + x[id]) * (1.0f / 3.0f);
}

__global__ void conv_state_update_k(const float* x, float* state, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C)
        return;
    float v[3];
    for (int i = 0; i < 3; ++i)
        v[i] = conv_in(x, state, T - 3 + i, c, C);
    for (int i = 0; i < 3; ++i)
        state[(size_t)i * C + c] = v[i];
}

__global__ void kv_store_k(const float* x, half* cache, int total, int n_heads, int hd, bool is_key, float thr,
                           float eps, int pos0, int ring) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= total)
        return;
    const int head = id % n_heads, t = id / n_heads;
    const float* xr = x + (size_t)id * hd;
    float ss = 0.0f;
    for (int i = 0; i < hd; ++i)
        ss += xr[i] * xr[i];
    const float f = is_key ? fminf(thr / (sqrtf(ss) + 1e-6f), 1.0f) : rsqrtf(ss / hd + eps);
    const int slot = ring > 0 ? (pos0 + t) % ring : pos0 + t;
    half* o = cache + ((size_t)slot * n_heads + head) * hd;
    for (int i = 0; i < hd; ++i)
        o[i] = __float2half(xr[i] * f);
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
    const int pos = a.pos0 + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const size_t qh = (size_t)t * a.n_heads + head;
    for (int i = threadIdx.x; i < a.hd; i += blockDim.x)
        sq[i] = a.q[qh * a.hd + i];
    if ((int)threadIdx.x < a.d_rel)
        sr[threadIdx.x] = a.rel[qh * a.d_rel + threadIdx.x];
    __syncthreads();

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
                bias += sr[i] * a.proj[(size_t)i * a.rel_extent + d];
        const float sc = dot * a.scale + bias;
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
    const float sink = __half2float(a.sink[qh]);
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
        a.out[qh * a.hd + i] = v / total;
    }
}

__global__ void gate_mul_k(const float* attn, const half* gate, half* out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half(attn[i] / (1.0f + expf(-__half2float(gate[i]))));
}

// One block per token, one thread per expert (n_experts <= 128).
__global__ void route_topk_k(const float* x, const float* w, const float* bias, int32_t* idx, float* weights,
                             int H, int n_experts, int top_k, float route_scale) {
    __shared__ float sel[128];
    __shared__ float chosen[16];
    __shared__ float sh[4];
    const int e = threadIdx.x;
    const float* xr = x + (size_t)blockIdx.x * H;
    float score = 0.0f;
    if (e < n_experts) {
        float d = 0.0f;
        for (int i = 0; i < H; ++i)
            d += xr[i] * w[(size_t)e * H + i];
        score = 1.0f / (1.0f + expf(-d));
        sel[e] = score + bias[e];
    }
    __syncthreads();
    int rank = top_k;
    if (e < n_experts) {
        rank = 0;
        for (int j = 0; j < n_experts; ++j)
            rank += (sel[j] > sel[e]) || (sel[j] == sel[e] && j < e);
    }
    const float sum = block_sum(rank < top_k ? score : 0.0f, sh);
    if (rank < top_k) {
        idx[(size_t)blockIdx.x * top_k + rank] = e;
        weights[(size_t)blockIdx.x * top_k + rank] = score / (sum + 1e-20f) * route_scale;
    }
    (void)chosen;
}

__global__ void expert_act_k(const half* gu, half* out, size_t n, float gate_cap, float up_cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float g = __half2float(gu[2 * i]), u = __half2float(gu[2 * i + 1]);
    out[i] = __float2half(gate_cap * tanhf(g / gate_cap) / (1.0f + expf(-g)) * up_cap * tanhf(u / up_cap));
}

__global__ void expert_combine_k(const half* y, const float* weights, float* out, size_t n, int top_k, int H) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    const size_t t = id / H, i = id % H;
    float v = 0.0f;
    for (int k = 0; k < top_k; ++k)
        v += weights[t * top_k + k] * __half2float(y[(t * top_k + k) * H + i]);
    out[id] = v;
}

constexpr int kFp8Rows = 8;  // rows per block, one warp each

__global__ void gemv_mxfp8_slots_k(const uint8_t* w, const uint8_t* sc, const int32_t* idx, const half* x,
                                   half* y, int rows, int K, int x_group, int blocks_per_slot) {
    const int slot = blockIdx.x / blocks_per_slot;
    const int row = (blockIdx.x % blocks_per_slot) * kFp8Rows + (threadIdx.x >> 5);
    if (row >= rows)
        return;
    const int lane = threadIdx.x & 31;
    const size_t e = static_cast<size_t>(idx[slot]);
    const uint8_t* wr = w + (e * rows + row) * K;
    const uint8_t* sr = sc + (e * rows + row) * (K / 32);
    const half* xr = x + (size_t)(slot / x_group) * K;
    float acc = 0.0f;
    for (int b = 0; b < K / 32; ++b) {
        const int k = b * 32 + lane;
        acc += e4m3(wr[k]) * scalbnf(1.0f, static_cast<int>(sr[b]) - 127) * __half2float(xr[k]);
    }
    acc = warp_sum(acc);
    if (lane == 0)
        y[(size_t)slot * rows + row] = __float2half(acc);
}

__global__ void softcap_f32_k(const half* x, float* out, size_t n, float cap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = cap * tanhf(__half2float(x[i]) / cap);
}

}  // namespace

void bf16_to_f16(const uint16_t* in, half* out, size_t n, cudaStream_t s) {
    bf16_to_f16_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void bf16_to_f32(const uint16_t* in, float* out, size_t n, cudaStream_t s) {
    bf16_to_f32_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void f32_to_f16(const float* in, half* out, size_t n, cudaStream_t s) {
    f32_to_f16_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void f16_to_f32(const half* in, float* out, size_t n, cudaStream_t s) {
    f16_to_f32_k<<<grid_for(n), kThreads, 0, s>>>(in, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void add_f32(const float* a, const float* b, float* out, size_t n, cudaStream_t s) {
    add_f32_k<<<grid_for(n), kThreads, 0, s>>>(a, b, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void embed_rows(const half* table, const int32_t* ids, float* out, int T, int H, cudaStream_t s) {
    embed_rows_k<<<T, kThreads, 0, s>>>(table, ids, out, H);
    IMP_CUDA_CHECK_LAUNCH();
}
void rms_norm(const float* x, const float* w, float* out_f, half* out_h, int T, int H, float eps,
              cudaStream_t s) {
    rms_norm_k<<<T, kThreads, 0, s>>>(x, w, out_f, out_h, H, eps);
    IMP_CUDA_CHECK_LAUNCH();
}
void block_mix(const MixStates& st, const float* query, int nq, const MixOut& out, int T, int H,
               cudaStream_t s) {
    block_mix_k<<<T, kThreads, 0, s>>>(st, query, nq, out, H);
    IMP_CUDA_CHECK_LAUNCH();
}
void gate_cat(const float* h, const float* msg, half* cat, int T, int H, float eps, cudaStream_t s) {
    gate_cat_k<<<T, kThreads, 0, s>>>(h, msg, cat, H, eps);
    IMP_CUDA_CHECK_LAUNCH();
}
void gate_apply(const float* h, const float* msg, const half* forget, const half* gain, float* out,
                size_t n, cudaStream_t s) {
    gate_apply_k<<<grid_for(n), kThreads, 0, s>>>(h, msg, forget, gain, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void scaled_l2_clip(const half* x, const float* scale, float* out, int T, int n_heads, int head_dim,
                    float thr, int pos0, float log_alpha, float log_floor, cudaStream_t s) {
    const int total = T * n_heads;
    scaled_l2_clip_k<<<grid_for(total), kThreads, 0, s>>>(x, scale, out, total, n_heads, head_dim, thr, pos0,
                                                          log_alpha, log_floor);
    IMP_CUDA_CHECK_LAUNCH();
}
void conv_static(const float* x, const float* w, const float* state, float* out, int T, int C,
                 cudaStream_t s) {
    const size_t n = (size_t)T * C;
    conv_static_k<<<grid_for(n), kThreads, 0, s>>>(x, w, state, out, n, C);
    IMP_CUDA_CHECK_LAUNCH();
}
void conv_dynamic(const float* x, const float* static_w, const half* dyn, const float* state, float* out,
                  int T, int C, int n_heads, cudaStream_t s) {
    const size_t n = (size_t)T * C;
    conv_dynamic_k<<<grid_for(n), kThreads, 0, s>>>(x, static_w, dyn, state, out, n, C, n_heads);
    IMP_CUDA_CHECK_LAUNCH();
}
void conv_state_update(const float* x, float* state, int T, int C, cudaStream_t s) {
    conv_state_update_k<<<grid_for(C), kThreads, 0, s>>>(x, state, T, C);
    IMP_CUDA_CHECK_LAUNCH();
}
void kv_store(const float* x, half* cache, int T, int n_heads, int head_dim, bool is_key, float thr,
              float eps, int pos0, int ring, cudaStream_t s) {
    const int total = T * n_heads;
    kv_store_k<<<grid_for(total), kThreads, 0, s>>>(x, cache, total, n_heads, head_dim, is_key, thr, eps, pos0,
                                                    ring);
    IMP_CUDA_CHECK_LAUNCH();
}
void attention(const AttnArgs& a, cudaStream_t s) {
    if (a.hd > kAttnMaxHd || a.d_rel > 32) {
        IMP_LOG_ERROR("inkling attention: head_dim %d or d_rel %d out of range", a.hd, a.d_rel);
        return;
    }
    attention_k<<<a.T * a.n_heads, kAttnThreads, 0, s>>>(a);
    IMP_CUDA_CHECK_LAUNCH();
}
void gate_mul(const float* attn, const half* gate, half* out, size_t n, cudaStream_t s) {
    gate_mul_k<<<grid_for(n), kThreads, 0, s>>>(attn, gate, out, n);
    IMP_CUDA_CHECK_LAUNCH();
}
void route_topk(const float* x, const float* w, const float* bias, int32_t* idx, float* weights, int T,
                int H, int n_experts, int top_k, float route_scale, cudaStream_t s) {
    if (n_experts > 128 || top_k > 16) {
        IMP_LOG_ERROR("inkling router: %d experts / top %d out of range", n_experts, top_k);
        return;
    }
    route_topk_k<<<T, 128, 0, s>>>(x, w, bias, idx, weights, H, n_experts, top_k, route_scale);
    IMP_CUDA_CHECK_LAUNCH();
}
void expert_act(const half* gu, half* out, int slots, int I, float gate_cap, float up_cap, cudaStream_t s) {
    const size_t n = (size_t)slots * I;
    expert_act_k<<<grid_for(n), kThreads, 0, s>>>(gu, out, n, gate_cap, up_cap);
    IMP_CUDA_CHECK_LAUNCH();
}
void expert_combine(const half* y, const float* weights, float* out, int T, int top_k, int H,
                    cudaStream_t s) {
    const size_t n = (size_t)T * H;
    expert_combine_k<<<grid_for(n), kThreads, 0, s>>>(y, weights, out, n, top_k, H);
    IMP_CUDA_CHECK_LAUNCH();
}
void gemv_mxfp8_slots(const uint8_t* w, const uint8_t* sc, const int32_t* idx, const half* x, half* y,
                      int slots, int rows, int K, int x_group, cudaStream_t s) {
    const int bps = (rows + kFp8Rows - 1) / kFp8Rows;
    gemv_mxfp8_slots_k<<<slots * bps, kFp8Rows * 32, 0, s>>>(w, sc, idx, x, y, rows, K, x_group, bps);
    IMP_CUDA_CHECK_LAUNCH();
}
void softcap_f32(const half* x, float* out, size_t n, float cap, cudaStream_t s) {
    softcap_f32_k<<<grid_for(n), kThreads, 0, s>>>(x, out, n, cap);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp::inkling
