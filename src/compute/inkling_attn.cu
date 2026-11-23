// Inkling2-Loop attention kernels (inkling_ops.h): the row form for prefill and the key-chunked form
// for calls of a few rows (decode, lookahead).

#include "compute/inkling_ops.h"

#include "compute/inkling_detail.cuh"
#include "compute/inkling_fused.cuh"
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

// Decode attention for 1..kSplitMaxT rows of a model with three heads per KV head, step 1. The rows of a
// call read the same keys, and so do the heads of one KV head: one block per (KV head, key chunk) loads
// the chunk once and serves every (row, head) pair. Chunks start at row 0's first key, a row masks what
// lies outside its own window. attn_part_k per (row, head, chunk) took 4.17 ms of a 19.4 ms eight-row call.
constexpr int kRowsHeads = 3;                        // heads per KV head
constexpr int kRowsPairs = kSplitMaxT * kRowsHeads;  // (row, head) pairs of a block
constexpr int kRowsRel = 16;
struct RowsSmem {
    half sk[kPartKeys][kAttnMaxHd];
    half sv[kPartKeys][kAttnMaxHd];
    float proj[kPartKeys + kSplitMaxT][kRowsRel];  // bias rows of the distances the chunk can have
    float q[kRowsPairs][kAttnMaxHd];
    float r[kRowsPairs][kRowsRel];
    float sc[kRowsPairs][kPartKeys];  // scores, then softmax terms
    float qf[kRowsPairs], rf[kRowsPairs];
};

struct RowsAt {
    int c, kvh, hpk, pairs, pos0, k0, n, dlo;
};

// Key, value and bias tiles of the chunk, every pair's query and distance vector.
__device__ __forceinline__ void rows_load(const AttnArgs& a, RowsSmem& m, const RowsAt& x) {
    const int tid = threadIdx.x, per = a.hd / 8;
    for (int i = tid; i < x.n * per; i += blockDim.x) {
        const int key = i / per, cc = (i % per) * 8;
        const int slot = a.ring > 0 ? (x.k0 + key) % a.ring : x.k0 + key;
        const size_t at = ((size_t)slot * a.n_kv + x.kvh) * a.hd + cc;
        *reinterpret_cast<uint4*>(&m.sk[key][cc]) = *reinterpret_cast<const uint4*>(a.k + at);
        *reinterpret_cast<uint4*>(&m.sv[key][cc]) = *reinterpret_cast<const uint4*>(a.v + at);
    }
    for (int i = tid; i < (kPartKeys + a.T) * a.d_rel; i += blockDim.x) {
        const int d = x.dlo + i / a.d_rel, j = i % a.d_rel;
        m.proj[i / a.d_rel][j] = d >= 0 && d < a.rel_extent ? a.proj[(size_t)d * a.d_rel + j] : 0.0f;
    }
    for (int i = tid; i < x.pairs * per; i += blockDim.x) {
        const int p = i / per, cc = (i % per) * 8;
        const int head = x.kvh * x.hpk + p % x.hpk;
        half2 qh[4];
        *reinterpret_cast<uint4*>(qh) = *reinterpret_cast<const uint4*>(a.q + (size_t)(p / x.hpk) * a.stride +
                                                                        (size_t)head * a.hd + cc);
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(qh[j]);
            m.q[p][cc + 2 * j] = f.x * a.q_scale[head * a.hd + cc + 2 * j];
            m.q[p][cc + 2 * j + 1] = f.y * a.q_scale[head * a.hd + cc + 2 * j + 1];
        }
    }
    for (int i = tid; i < x.pairs * a.d_rel; i += blockDim.x) {
        const int p = i / a.d_rel, j = i % a.d_rel;
        const int head = x.kvh * x.hpk + p % x.hpk;
        m.r[p][j] = __half2float(a.rel[(size_t)(p / x.hpk) * a.stride + (size_t)head * a.d_rel + j]) *
                    a.rel_scale[head * a.d_rel + j];
    }
}

// Clip factors: a warp per pair.
__device__ __forceinline__ void rows_clip(const AttnArgs& a, RowsSmem& m, const RowsAt& x) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int p = warp; p < x.pairs; p += blockDim.x / 32) {
        float ss = 0.0f;
        for (int i = lane; i < a.hd; i += 32)
            ss += m.q[p][i] * m.q[p][i];
        const float rv = lane < a.d_rel ? m.r[p][lane] : 0.0f;
        ss = warp_sum(ss);
        const float rss = warp_sum(rv * rv);
        if (lane != 0)
            continue;
        const int row = p / x.hpk;
        const float at = static_cast<float>(x.pos0 + row + 1);
        const float tau = a.log_alpha > 0.0f ? 1.0f + a.log_alpha * logf(fmaxf(at / a.log_floor, 1.0f))
                                             : 1.0f;
        m.qf[p] = fminf(a.q_thr / (sqrtf(ss) + 1e-6f), 1.0f) * tau * a.scale;
        m.rf[p] = fminf(a.rel_thr / (sqrtf(rss) + 1e-6f), 1.0f) * tau;
    }
}

// Scores: a thread per (pair, key); a key outside the pair's row window scores -FLT_MAX.
__device__ __forceinline__ void rows_scores(const AttnArgs& a, RowsSmem& m, const RowsAt& x) {
    for (int i = threadIdx.x; i < x.pairs * kPartKeys; i += blockDim.x) {
        const int p = i / kPartKeys, key = i % kPartKeys;
        const int pos = x.pos0 + p / x.hpk, k = x.k0 + key;
        const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
        float sc = -FLT_MAX;
        if (key < x.n && k >= lo && k <= pos) {
            float dot = 0.0f;
            for (int j = 0; j < a.hd; j += 2) {
                const float2 kf = __half22float2(*reinterpret_cast<const half2*>(&m.sk[key][j]));
                dot += m.q[p][j] * kf.x + m.q[p][j + 1] * kf.y;
            }
            const float* pr = m.proj[pos - k - x.dlo];
            float bias = 0.0f;
            for (int j = 0; j < a.d_rel; ++j)
                bias += m.r[p][j] * pr[j];
            sc = dot * m.qf[p] + bias * m.rf[p];
        }
        m.sc[p][key] = sc;
    }
}

// Chunk maximum and softmax terms: a warp per pair, two keys per lane. o: the pair's part row.
__device__ __forceinline__ void rows_softmax(const AttnArgs& a, RowsSmem& m, const RowsAt& x, int S) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int p = warp; p < x.pairs; p += blockDim.x / 32) {
        const float s0 = m.sc[p][lane], s1 = m.sc[p][lane + 32];
        float mx = fmaxf(s0, s1);
        for (int off = 16; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        const float p0 = s0 > -FLT_MAX ? expf(s0 - mx) : 0.0f, p1 = s1 > -FLT_MAX ? expf(s1 - mx) : 0.0f;
        m.sc[p][lane] = p0, m.sc[p][lane + 32] = p1;
        const float l = warp_sum(p0 + p1);
        if (lane == 0) {
            const int th = (p / x.hpk) * a.n_heads + x.kvh * x.hpk + p % x.hpk;
            float* o = a.part + ((size_t)th * S + x.c) * (a.hd + 2);
            o[0] = mx, o[1] = l;
        }
    }
}

__global__ void attn_part_rows_k(AttnArgs a, int S) {
    extern __shared__ __align__(16) unsigned char raw[];
    RowsSmem& m = *reinterpret_cast<RowsSmem*>(raw);
    RowsAt x{};
    x.c = blockIdx.x % S, x.kvh = blockIdx.x / S;
    x.hpk = a.n_heads / a.n_kv, x.pairs = a.T * x.hpk;
    x.pos0 = *a.d_pos;
    const int base = a.window > 0 ? max(0, x.pos0 - a.window + 1) : 0;
    x.k0 = base + x.c * kPartKeys;
    x.n = min(x.pos0 + a.T - 1, x.k0 + kPartKeys - 1) - x.k0 + 1;
    if (x.n <= 0)
        return;  // past the last row's keys: attn_reduce_k stops before this chunk
    x.dlo = x.pos0 - (x.k0 + kPartKeys - 1);  // smallest distance a key of the chunk can have
    rows_load(a, m, x);
    __syncthreads();
    rows_clip(a, m, x);
    __syncthreads();
    rows_scores(a, m, x);
    __syncthreads();
    rows_softmax(a, m, x, S);
    __syncthreads();
    // Value sums: a thread per (pair, dim).
    for (int i = threadIdx.x; i < x.pairs * a.hd; i += blockDim.x) {
        const int p = i / a.hd, dim = i % a.hd;
        float acc = 0.0f;
        for (int key = 0; key < x.n; ++key)
            acc += m.sc[p][key] * __half2float(m.sv[key][dim]);
        const int th = (p / x.hpk) * a.n_heads + x.kvh * x.hpk + p % x.hpk;
        a.part[((size_t)th * S + x.c) * (a.hd + 2) + 2 + dim] = acc;
    }
}

// attn_part_rows_k for the model's shape (head_dim 128, d_rel 16, 3 heads per KV head) with the row count
// at compile time: a thread issues its 8 key / value loads, 5 bias loads and its query loads before the
// first store (a runtime-bounded loop sent them one after the other). Same expressions in the same
// order: bit-equal (FixedShapeKernelsAreBitEqual).
constexpr int kFixHd = 128, kFixPer = kFixHd / 8;
// Chunk columns of the grid: a block takes every kFixGrid-th chunk, so a context above 64 x kFixGrid
// positions costs no blocks that only read the position (131072 context: 16384 blocks per launch, 4.258 ms
// per step against 4.104 at 4096).
constexpr int kFixGrid = 128;
template <int T>
__device__ __forceinline__ void rows_load_t(const AttnArgs& a, RowsSmem& m, const RowsAt& x) {
    constexpr int kPairs = T * kRowsHeads;
    constexpr int kKv = kPartKeys * kFixPer / kPartThreads;                              // 4
    constexpr int kPr = ((kPartKeys + T) * kRowsRel + kPartThreads - 1) / kPartThreads;  // 5
    constexpr int kQ = (kPairs * kFixPer + kPartThreads - 1) / kPartThreads;             // 1..2
    constexpr int kR = (kPairs * kRowsRel + kPartThreads - 1) / kPartThreads;            // 1..2
    const int tid = threadIdx.x;
    uint4 kk[kKv], vv[kKv];
    float pr[kPr];
    uint4 qh[kQ];
    float qs[kQ][8];
    half rh[kR];
    float rs[kR];
    for (int j = 0; j < kKv; ++j) {
        const int i = tid + j * kPartThreads;
        const int key = min(i / kFixPer, x.n - 1), cc = (i % kFixPer) * 8;
        const int slot = a.ring > 0 ? (x.k0 + key) % a.ring : x.k0 + key;
        const size_t at = ((size_t)slot * a.n_kv + x.kvh) * kFixHd + cc;
        kk[j] = *reinterpret_cast<const uint4*>(a.k + at);
        vv[j] = *reinterpret_cast<const uint4*>(a.v + at);
    }
    for (int j = 0; j < kPr; ++j) {
        const int i = tid + j * kPartThreads;
        const int d = x.dlo + i / kRowsRel;
        const bool ok = i < (kPartKeys + T) * kRowsRel && d >= 0 && d < a.rel_extent;
        const float v = a.proj[ok ? (size_t)d * kRowsRel + i % kRowsRel : 0];
        pr[j] = ok ? v : 0.0f;
    }
    for (int j = 0; j < kQ; ++j) {
        const int i = min(tid + j * kPartThreads, kPairs * kFixPer - 1);
        const int p = i / kFixPer, cc = (i % kFixPer) * 8;
        const int head = x.kvh * kRowsHeads + p % kRowsHeads;
        qh[j] = *reinterpret_cast<const uint4*>(a.q + (size_t)(p / kRowsHeads) * a.stride +
                                                (size_t)head * kFixHd + cc);
        for (int e = 0; e < 8; ++e)
            qs[j][e] = a.q_scale[head * kFixHd + cc + e];
    }
    for (int j = 0; j < kR; ++j) {
        const int i = min(tid + j * kPartThreads, kPairs * kRowsRel - 1);
        const int p = i / kRowsRel, e = i % kRowsRel;
        const int head = x.kvh * kRowsHeads + p % kRowsHeads;
        rh[j] = a.rel[(size_t)(p / kRowsHeads) * a.stride + (size_t)head * kRowsRel + e];
        rs[j] = a.rel_scale[head * kRowsRel + e];
    }
    for (int j = 0; j < kKv; ++j) {
        const int i = tid + j * kPartThreads;
        const int key = i / kFixPer, cc = (i % kFixPer) * 8;
        if (key < x.n) {
            *reinterpret_cast<uint4*>(&m.sk[key][cc]) = kk[j];
            *reinterpret_cast<uint4*>(&m.sv[key][cc]) = vv[j];
        }
    }
    for (int j = 0; j < kPr; ++j) {
        const int i = tid + j * kPartThreads;
        if (i < (kPartKeys + T) * kRowsRel)
            m.proj[i / kRowsRel][i % kRowsRel] = pr[j];
    }
    for (int j = 0; j < kQ; ++j) {
        const int i = tid + j * kPartThreads;
        if (i >= kPairs * kFixPer)
            continue;
        const int p = i / kFixPer, cc = (i % kFixPer) * 8;
        const half2* h2 = reinterpret_cast<const half2*>(&qh[j]);
        for (int e = 0; e < 4; ++e) {
            const float2 f = __half22float2(h2[e]);
            m.q[p][cc + 2 * e] = f.x * qs[j][2 * e];
            m.q[p][cc + 2 * e + 1] = f.y * qs[j][2 * e + 1];
        }
    }
    for (int j = 0; j < kR; ++j) {
        const int i = tid + j * kPartThreads;
        if (i < kPairs * kRowsRel)
            m.r[i / kRowsRel][i % kRowsRel] = __half2float(rh[j]) * rs[j];
    }
}

template <int T>
__global__ void attn_part_rows_t(AttnArgs a, int S) {
    constexpr int kPairs = T * kRowsHeads;
    extern __shared__ __align__(16) unsigned char raw[];
    RowsSmem& m = *reinterpret_cast<RowsSmem*>(raw);
    RowsAt x{};
    const int G = min(S, kFixGrid);
    x.kvh = blockIdx.x / G;
    x.hpk = kRowsHeads, x.pairs = kPairs;
    x.pos0 = *a.d_pos;
    const int base = a.window > 0 ? max(0, x.pos0 - a.window + 1) : 0;
    for (x.c = blockIdx.x % G; x.c < S; x.c += G) {
        x.k0 = base + x.c * kPartKeys;
        x.n = min(x.pos0 + T - 1, x.k0 + kPartKeys - 1) - x.k0 + 1;
        if (x.n <= 0)
            return;
        x.dlo = x.pos0 - (x.k0 + kPartKeys - 1);
        rows_load_t<T>(a, m, x);
        __syncthreads();
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        // Clip factors: a warp per pair.
        for (int p = warp; p < kPairs; p += kPartThreads / 32) {
            float ss = 0.0f;
            for (int i = lane; i < kFixHd; i += 32)
                ss += m.q[p][i] * m.q[p][i];
            const float rv = lane < kRowsRel ? m.r[p][lane] : 0.0f;
            ss = warp_sum(ss);
            const float rss = warp_sum(rv * rv);
            if (lane != 0)
                continue;
            const int row = p / kRowsHeads;
            const float at = static_cast<float>(x.pos0 + row + 1);
            const float tau = a.log_alpha > 0.0f ? 1.0f + a.log_alpha * logf(fmaxf(at / a.log_floor, 1.0f))
                                                 : 1.0f;
            m.qf[p] = fminf(a.q_thr / (sqrtf(ss) + 1e-6f), 1.0f) * tau * a.scale;
            m.rf[p] = fminf(a.rel_thr / (sqrtf(rss) + 1e-6f), 1.0f) * tau;
        }
        __syncthreads();
        // Scores: a thread per (pair, key).
        for (int i = threadIdx.x; i < kPairs * kPartKeys; i += kPartThreads) {
            const int p = i / kPartKeys, key = i % kPartKeys;
            const int pos = x.pos0 + p / kRowsHeads, k = x.k0 + key;
            const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
            float sc = -FLT_MAX;
            if (key < x.n && k >= lo && k <= pos) {
                float dot = 0.0f;
                for (int j = 0; j < kFixHd; j += 2) {
                    const float2 kf = __half22float2(*reinterpret_cast<const half2*>(&m.sk[key][j]));
                    dot += m.q[p][j] * kf.x + m.q[p][j + 1] * kf.y;
                }
                const float* pr = m.proj[pos - k - x.dlo];
                float bias = 0.0f;
                for (int j = 0; j < kRowsRel; ++j)
                    bias += m.r[p][j] * pr[j];
                sc = dot * m.qf[p] + bias * m.rf[p];
            }
            m.sc[p][key] = sc;
        }
        __syncthreads();
        rows_softmax(a, m, x, S);
        __syncthreads();
        // Value sums: a thread per (pair, dim).
        for (int i = threadIdx.x; i < kPairs * kFixHd; i += kPartThreads) {
            const int p = i / kFixHd, dim = i % kFixHd;
            float acc = 0.0f;
            for (int key = 0; key < x.n; ++key)
                acc += m.sc[p][key] * __half2float(m.sv[key][dim]);
            const int th = (p / kRowsHeads) * a.n_heads + x.kvh * kRowsHeads + p % kRowsHeads;
            a.part[((size_t)th * S + x.c) * (kFixHd + 2) + 2 + dim] = acc;
        }
        __syncthreads();
    }
}

template <int T>
bool part_rows_launch(const AttnArgs& a, int S, cudaStream_t s) {
    static const bool smem = cudaFuncSetAttribute(attn_part_rows_t<T>,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(sizeof(RowsSmem))) == cudaSuccess;
    if (!smem)
        return false;
    attn_part_rows_t<T><<<a.n_kv * std::min(S, kFixGrid), kPartThreads, sizeof(RowsSmem), s>>>(a, S);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

// false = this shape (or set_fixed_shape_kernels(false)) takes attn_part_rows_k.
bool part_rows_fixed(const AttnArgs& a, int S, cudaStream_t s) {
    if (!fixed_shape_kernels() || a.hd != kFixHd || a.d_rel != kRowsRel)
        return false;
    switch (a.T) {
        case 1:
            return part_rows_launch<1>(a, S, s);
        case 2:
            return part_rows_launch<2>(a, S, s);
        case 3:
            return part_rows_launch<3>(a, S, s);
        case 4:
            return part_rows_launch<4>(a, S, s);
        case 5:
            return part_rows_launch<5>(a, S, s);
        case 6:
            return part_rows_launch<6>(a, S, s);
        case 7:
            return part_rows_launch<7>(a, S, s);
        case 8:
            return part_rows_launch<8>(a, S, s);
        default:
            return false;
    }
}

// Decode attention, step 2. One block per (token, head), one thread per dim: joins the chunks that
// hold keys, adds the sink to the denominator, applies the output gate.
// rows: the chunks start at row 0's first key (attn_part_rows_k).
__global__ void attn_reduce_k(AttnArgs a, int S, int CK, bool rows, bool batched) {
    const int t = blockIdx.x / a.n_heads, head = blockIdx.x % a.n_heads;
    const int i = threadIdx.x;
    const int pos = *a.d_pos + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    const int pos0 = *a.d_pos, base = a.window > 0 ? max(0, pos0 - a.window + 1) : 0;
    const int n_c = rows ? min(S, (pos0 + a.T - 1 - base) / CK + 1) : min(S, (pos - lo) / CK + 1);
    const float* p0 = a.part + (size_t)blockIdx.x * S * (a.hd + 2);
    const size_t row = (size_t)t * a.stride;
    const float sink = __half2float(a.sink[row + head]);
    float gm = sink;
    float total = 0.0f, v = 0.0f;
    if (batched) {
        // Four chunks' loads issued before the first use; the sums keep the chunk order.
        for (int c0 = 0; c0 < n_c; c0 += 4) {
            float h[4];
            for (int j = 0; j < 4; ++j)
                h[j] = p0[(size_t)min(c0 + j, n_c - 1) * (a.hd + 2)];
            for (int j = 0; j < 4; ++j)
                gm = fmaxf(gm, h[j]);
        }
        total = expf(sink - gm);
        for (int c0 = 0; c0 < n_c; c0 += 4) {
            float h[4], l[4], x[4];
            for (int j = 0; j < 4; ++j) {
                const float* pc = p0 + (size_t)min(c0 + j, n_c - 1) * (a.hd + 2);
                h[j] = pc[0], l[j] = pc[1], x[j] = pc[2 + i];
            }
            for (int j = 0; j < 4; ++j) {
                if (c0 + j >= n_c || l[j] <= 0.0f)
                    continue;
                const float w = expf(h[j] - gm);
                total += l[j] * w;
                v += x[j] * w;
            }
        }
    } else {
        for (int c = 0; c < n_c; ++c)
            gm = fmaxf(gm, p0[(size_t)c * (a.hd + 2)]);
        total = expf(sink - gm);
        for (int c = 0; c < n_c; ++c) {
            const float* pc = p0 + (size_t)c * (a.hd + 2);
            if (pc[1] <= 0.0f)
                continue;
            const float w = expf(pc[0] - gm);
            total += pc[1] * w;
            v += pc[2 + i] * w;
        }
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
    // Few rows (decode, lookahead, paced prefill): the rows and the heads of a KV head share the key loads.
    static const bool rows_smem = cudaFuncSetAttribute(attn_part_rows_k,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(sizeof(RowsSmem))) == cudaSuccess;
    if (rows_smem && a.part && a.T >= 1 && a.T <= kSplitMaxT && a.hd == kAttnMaxHd && a.d_rel <= kRowsRel &&
        a.n_heads == kRowsHeads * a.n_kv && a.stride % 8 == 0) {
        const int S = ((a.window > 0 ? a.window + a.T - 1 : a.max_ctx) + kPartKeys - 1) / kPartKeys;
        if (!part_rows_fixed(a, S, s)) {
            attn_part_rows_k<<<a.n_kv * S, kPartThreads, sizeof(RowsSmem), s>>>(a, S);
            IMP_CUDA_CHECK_LAUNCH();
        }
        attn_reduce_k<<<a.T * a.n_heads, a.hd, 0, s>>>(a, S, kPartKeys, true, fixed_shape_kernels());
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    // Other head layouts: one block per (row, head, chunk). Prefill has rows x heads blocks already.
    if (a.part && a.T <= kSplitMaxT && 2 * a.hd == kPartThreads && a.d_rel <= 32) {
        const int CK = kPartKeys;
        const int S = ((a.window > 0 ? a.window : a.max_ctx) + CK - 1) / CK;
        attn_part_k<<<a.T * a.n_heads * S, kPartThreads, 0, s>>>(a, S);
        IMP_CUDA_CHECK_LAUNCH();
        attn_reduce_k<<<a.T * a.n_heads, a.hd, 0, s>>>(a, S, CK, false, fixed_shape_kernels());
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    if (a.T > kSplitMaxT && attention_rows(a, s))
        return;
    attention_k<<<a.T * a.n_heads, kAttnThreads, 0, s>>>(a);
    IMP_CUDA_CHECK_LAUNCH();
}

size_t attention_part_floats(int n_heads, int hd, int window, int max_ctx) {
    const size_t chunks = static_cast<size_t>(std::max((window + 31) / 32, (max_ctx + 63) / 64));
    return static_cast<size_t>(kSplitMaxT) * n_heads * chunks * (hd + 2);
}

}  // namespace imp::inkling
