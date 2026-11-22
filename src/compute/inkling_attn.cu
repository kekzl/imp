// Inkling2-Loop attention kernels (inkling_ops.h): the row form for prefill and the key-chunked form
// for calls of a few rows (decode, lookahead).

#include "compute/inkling_ops.h"

#include "compute/inkling_detail.cuh"
#include "core/logging.h"

#include <algorithm>

namespace imp::inkling {

namespace {

using namespace detail;

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

constexpr int kSplitMaxT = 8;

constexpr int kPartKeys = 64, kPartThreads = 256;

// Tiles of a key chunk: hd / 8 loads of 8 halves per key and side, the bias row of every key's distance.
__device__ __forceinline__ void attn_load_tiles(const AttnArgs& a, half (*sk)[kAttnMaxHd],
                                                half (*sv)[kAttnMaxHd], float (*sproj)[32], int k0, int n,
                                                int kvh, int pos) {
    const int per = a.hd / 8;
    for (int i = threadIdx.x; i < n * per; i += blockDim.x) {
        const int key = i / per, cc = (i % per) * 8;
        const int slot = a.ring > 0 ? (k0 + key) % a.ring : k0 + key;
        const size_t at = ((size_t)slot * a.n_kv + kvh) * a.hd + cc;
        *reinterpret_cast<uint4*>(&sk[key][cc]) = *reinterpret_cast<const uint4*>(a.k + at);
        *reinterpret_cast<uint4*>(&sv[key][cc]) = *reinterpret_cast<const uint4*>(a.v + at);
    }
    for (int i = threadIdx.x; i < n * a.d_rel; i += blockDim.x) {
        const int key = i / a.d_rel, j = i % a.d_rel, d = pos - (k0 + key);
        sproj[key][j] = d < a.rel_extent ? a.proj[(size_t)d * a.d_rel + j] : 0.0f;
    }
}

// Decode attention, step 1. One block of kPartThreads (= 2 head_dim = 4 keys-per-chunk) per (token,
// head, key chunk). A thread's loads come one after the other (about 0.3 us each): keys, values and the bias
// rows of the chunk are loaded by all threads into shared memory (8 loads each), then the scores run
// one key per thread pair and the value sums one dim per thread. Was one warp per chunk with 128
// loads in a row per lane: 30.5 us at 600 tokens of context.
// part per (token, head, chunk): max score, sum of exp(score - max), then hd weighted value sums.
__global__ void attn_part_k(AttnArgs a, int S) {
    __shared__ __align__(16) half sk[kPartKeys][kAttnMaxHd];
    __shared__ __align__(16) half sv[kPartKeys][kAttnMaxHd];
    __shared__ float sproj[kPartKeys][32];
    __shared__ float sq[kAttnMaxHd];
    __shared__ float sr[32];
    __shared__ float sd[4][kPartKeys];
    __shared__ float sp[kPartKeys];
    __shared__ float sa[kAttnMaxHd];
    __shared__ float red[kPartThreads / 32];
    const int c = blockIdx.x % S, th = blockIdx.x / S, tid = threadIdx.x;
    const int t = th / a.n_heads, head = th % a.n_heads;
    const int kvh = head / (a.n_heads / a.n_kv);
    float* o = a.part + ((size_t)th * S + c) * (a.hd + 2);
    const int pos = *a.d_pos + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int k0 = lo + c * kPartKeys, n = min(pos, k0 + kPartKeys - 1) - k0 + 1;
    if (n <= 0) {
        if (tid < a.hd)
            o[2 + tid] = 0.0f;
        if (tid == 0)
            o[0] = -FLT_MAX, o[1] = 0.0f;
        return;
    }
    const size_t row = (size_t)t * a.stride;
    const float tau = a.log_alpha > 0.0f
                          ? 1.0f + a.log_alpha * logf(fmaxf(static_cast<float>(pos + 1) / a.log_floor, 1.0f))
                          : 1.0f;
    const float q = tid < a.hd
                        ? __half2float(a.q[row + (size_t)head * a.hd + tid]) * a.q_scale[head * a.hd + tid]
                        : 0.0f;
    if (tid < a.hd)
        sq[tid] = q;
    const float r = tid < a.d_rel ? __half2float(a.rel[row + (size_t)head * a.d_rel + tid]) *
                                        a.rel_scale[head * a.d_rel + tid]
                                  : 0.0f;
    if (tid < 32)
        sr[tid] = r;
    attn_load_tiles(a, sk, sv, sproj, k0, n, kvh, pos);
    const float qf = fminf(a.q_thr / (sqrtf(block_sum(q * q, red)) + 1e-6f), 1.0f) * tau * a.scale;
    const float rf = fminf(a.rel_thr / (sqrtf(block_sum(r * r, red)) + 1e-6f), 1.0f) * tau;
    // Scores: four threads per key, a quarter of the dims each.
    const int key = tid % kPartKeys, part_i = tid / kPartKeys;
    float dot = 0.0f;
    if (key < n)
        for (int i = part_i * (a.hd / 4); i < (part_i + 1) * (a.hd / 4); ++i)
            dot += sq[i] * __half2float(sk[key][i]);
    sd[part_i][key] = dot;
    __syncthreads();
    float sc = -FLT_MAX;
    if (tid < n) {
        float bias = 0.0f;
        for (int j = 0; j < a.d_rel; ++j)
            bias += sr[j] * sproj[tid][j];
        sc = (sd[0][tid] + sd[1][tid] + sd[2][tid] + sd[3][tid]) * qf + bias * rf;
    }
    float m = sc;
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((tid & 31) == 0)
        red[tid >> 5] = m;
    __syncthreads();
    for (unsigned i = 0; i < blockDim.x / 32; ++i)
        m = fmaxf(m, red[i]);
    __syncthreads();
    const float p = tid < n ? expf(sc - m) : 0.0f;
    if (tid < kPartKeys)
        sp[tid] = p;
    const float l = block_sum(p, red);
    // Values: two threads per dim, half the keys each.
    const int dim = tid % a.hd, kh = tid / a.hd;
    float acc = 0.0f;
    for (int kk = kh * (kPartKeys / 2); kk < min(n, (kh + 1) * (kPartKeys / 2)); ++kk)
        acc += sp[kk] * __half2float(sv[kk][dim]);
    if (kh == 1)
        sa[dim] = acc;
    __syncthreads();
    if (kh == 0)
        o[2 + dim] = acc + sa[dim];
    if (tid == 0)
        o[0] = m, o[1] = l;
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

}  // namespace

void attention(const AttnArgs& a, cudaStream_t s) {
    if (a.hd > kAttnMaxHd || a.d_rel > 32) {
        IMP_LOG_ERROR("inkling attention: head_dim %d or d_rel %d out of range", a.hd, a.d_rel);
        return;
    }
    // Few rows (decode, verify): chunked over the keys. Prefill has rows x heads blocks already.
    if (a.part && a.T <= kSplitMaxT && 2 * a.hd == kPartThreads && a.d_rel <= 32) {
        const int CK = kPartKeys;
        const int S = ((a.window > 0 ? a.window : a.max_ctx) + CK - 1) / CK;
        attn_part_k<<<a.T * a.n_heads * S, kPartThreads, 0, s>>>(a, S);
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

}  // namespace imp::inkling
