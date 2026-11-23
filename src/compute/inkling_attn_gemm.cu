// Inkling2-Loop prefill attention (inkling_ops.h): calls of more than 8 rows. Scores and value sums
// are tensor-core GEMMs per KV head, the softmax with distance bias and sink is a kernel between them.
// The row form attention_k took 47.5 % of the kernel time of a 3111-token prefill (492 tok/s).

#include "compute/inkling_ops.h"

#include "compute/gemm_lt_ex.h"
#include "compute/inkling_detail.cuh"
#include "core/cublas_check.h"
#include "core/logging.h"

#include <algorithm>

namespace imp::inkling {

namespace {

using namespace detail;

// Sub-batch of R rows starting at row r0 against the N keys [k_lo, k_lo + N). Pair g = kvh * R * hpk +
// t * hpk + j is head kvh * hpk + j of row t.
struct RowsWork {
    half* q;     // [pairs, hd]: clip(q * q_scale) * tau * scale
    float* r;    // [pairs, d_rel]: clip(rel * rel_scale) * tau
    float* s;    // [pairs, N] scores
    half* p;     // [pairs, N] exp(score - max)
    float* tot;  // [pairs] softmax denominator, sink included
    float* o;    // [pairs, hd] value sums
    int r0, R, N, k_lo, pos0;
};

__device__ __forceinline__ int pair_of(const AttnArgs& a, int R, int t, int head) {
    const int hpk = a.n_heads / a.n_kv;
    return (head / hpk) * R * hpk + t * hpk + head % hpk;
}

// One block per (row, head), hd threads.
__global__ void rows_prep_k(AttnArgs a, RowsWork w) {
    __shared__ float red[8];
    const int t = blockIdx.x / a.n_heads, head = blockIdx.x % a.n_heads, i = threadIdx.x;
    const int g = pair_of(a, w.R, t, head);
    const size_t row = (size_t)(w.r0 + t) * a.stride;
    const float at = static_cast<float>(w.pos0 + w.r0 + t + 1);
    const float tau = a.log_alpha > 0.0f ? 1.0f + a.log_alpha * logf(fmaxf(at / a.log_floor, 1.0f)) : 1.0f;
    const float q = __half2float(a.q[row + (size_t)head * a.hd + i]) * a.q_scale[head * a.hd + i];
    const float rv = i < a.d_rel ? __half2float(a.rel[row + (size_t)head * a.d_rel + i]) *
                                       a.rel_scale[head * a.d_rel + i]
                                 : 0.0f;
    const float qf = fminf(a.q_thr / (sqrtf(block_sum(q * q, red)) + 1e-6f), 1.0f) * tau * a.scale;
    const float rf = fminf(a.rel_thr / (sqrtf(block_sum(rv * rv, red)) + 1e-6f), 1.0f) * tau;
    w.q[(size_t)g * a.hd + i] = __float2half(q * qf);
    if (i < a.d_rel)
        w.r[(size_t)g * a.d_rel + i] = rv * rf;
}

// One block per pair: score + bias in place, then exp(score - max) as fp16 and the denominator.
__global__ void rows_softmax_k(AttnArgs a, RowsWork w) {
    __shared__ float red[kThreads / 32];
    const int hpk = a.n_heads / a.n_kv, per = w.R * hpk;
    const int g = blockIdx.x, kvh = g / per, t = (g % per) / hpk, head = kvh * hpk + g % hpk;
    const int pos = w.pos0 + w.r0 + t;
    const int lo = a.window > 0 ? max(0, pos - a.window + 1) : 0;
    float r[32];
    for (int j = 0; j < a.d_rel; ++j)
        r[j] = w.r[(size_t)g * a.d_rel + j];
    float* s = w.s + (size_t)g * w.N;
    half* p = w.p + (size_t)g * w.N;
    float m = -FLT_MAX;
    for (int kk = threadIdx.x; kk < w.N; kk += blockDim.x) {
        const int k = w.k_lo + kk;
        if (k < lo || k > pos)
            continue;
        const int d = pos - k;
        float bias = 0.0f;
        if (d < a.rel_extent)
            for (int j = 0; j < a.d_rel; ++j)
                bias += r[j] * a.proj[(size_t)d * a.d_rel + j];
        const float sc = s[kk] + bias;
        s[kk] = sc;
        m = fmaxf(m, sc);
    }
    const float sink = __half2float(a.sink[(size_t)(w.r0 + t) * a.stride + head]);
    const float gm = fmaxf(block_max(m, red), sink);
    float l = 0.0f;
    for (int kk = threadIdx.x; kk < w.N; kk += blockDim.x) {
        const int k = w.k_lo + kk;
        const float e = (k < lo || k > pos) ? 0.0f : expf(s[kk] - gm);
        p[kk] = __float2half(e);
        l += e;
    }
    l = block_sum(l, red);
    if (threadIdx.x == 0)
        w.tot[g] = l + expf(sink - gm);
}

// One block per (row, head), hd threads: normalizes and applies the output gate.
__global__ void rows_out_k(AttnArgs a, RowsWork w) {
    const int t = blockIdx.x / a.n_heads, head = blockIdx.x % a.n_heads, i = threadIdx.x;
    const int g = pair_of(a, w.R, t, head);
    const size_t row = (size_t)(w.r0 + t) * a.stride;
    const float gate = __half2float(a.gate[row + (size_t)head * a.hd + i]);
    a.out[((size_t)(w.r0 + t) * a.n_heads + head) * a.hd + i] = __float2half(w.o[(size_t)g * a.hd + i] /
                                                                             w.tot[g] / (1.0f + expf(-gate)));
}

size_t row_bytes(const AttnArgs& a, size_t N) {
    return static_cast<size_t>(a.n_heads) *
           (static_cast<size_t>(a.hd) * 6 + static_cast<size_t>(a.d_rel) * 4 + 4 + N * 6);
}

}  // namespace

size_t attention_rows_bytes(int n_heads, int hd, int d_rel, int window, int max_ctx, int max_rows) {
    AttnArgs a;
    a.n_heads = n_heads, a.hd = hd, a.d_rel = d_rel;
    // Every row of a sliding call at once; a full-context layer runs at least 8 rows per sub-batch.
    const size_t local = row_bytes(a, static_cast<size_t>(window) + max_rows) * static_cast<size_t>(max_rows);
    const size_t global = row_bytes(a, static_cast<size_t>(max_ctx)) * 8;
    return std::min(std::max(local, global) + 4096, static_cast<size_t>(512) << 20);
}

bool attention_rows(const AttnArgs& a, cudaStream_t s) {
    const int hpk = a.n_heads / a.n_kv, ld = a.n_kv * a.hd;
    if (!a.work || a.pos0 < 0 || a.n_heads % a.n_kv != 0 || a.hd > kThreads || a.d_rel > 32)
        return false;
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < a.T;) {
        RowsWork w{};
        w.r0 = r0, w.pos0 = a.pos0;
        const int p0 = a.pos0 + r0;
        w.k_lo = a.window > 0 ? std::max(0, p0 - a.window + 1) : 0;
        const size_t n_max = static_cast<size_t>(a.pos0 + a.T - w.k_lo);
        const size_t fit = (a.work_bytes - 2048) / row_bytes(a, n_max);
        if (fit < 1)
            return false;
        w.R = static_cast<int>(std::min<size_t>(fit, static_cast<size_t>(a.T - r0)));
        w.N = p0 + w.R - w.k_lo;
        const size_t pairs = static_cast<size_t>(w.R) * a.n_heads, N = static_cast<size_t>(w.N);
        // fp32 planes first, then the fp16 ones: every plane starts 4-byte aligned.
        w.s = static_cast<float*>(a.work);
        w.o = w.s + pairs * N;
        w.r = w.o + pairs * a.hd;
        w.tot = w.r + pairs * a.d_rel;
        w.p = reinterpret_cast<half*>(w.tot + pairs);
        w.q = w.p + pairs * N;
        rows_prep_k<<<w.R * a.n_heads, a.hd, 0, s>>>(a, w);
        IMP_CUDA_CHECK_LAUNCH();
        // The keys sit at slot k % ring: a range that crosses the ring end is two column ranges.
        const int slot0 = a.ring > 0 ? w.k_lo % a.ring : w.k_lo;
        const int n1 = a.ring > 0 ? std::min(w.N, a.ring - slot0) : w.N;
        const int per = w.R * hpk;
        const long long sq = static_cast<long long>(per) * a.hd, ss = static_cast<long long>(per) * w.N;
        for (int part = 0; part < 2; ++part) {
            const int n = part == 0 ? n1 : w.N - n1;
            if (n <= 0)
                continue;
            const half* kp = a.k + static_cast<size_t>(part == 0 ? slot0 : 0) * ld;
            const int off = part == 0 ? 0 : n1;
            // S[pair, key] = q . k: (keys x pairs) = K^T (hd x keys, one KV head per batch) x Q (hd x pairs).
            IMP_CUBLAS_CHECK_LOG(lt_gemm_strided_batched(CUBLAS_OP_T, CUBLAS_OP_N, n, per, a.hd, &one, kp,
                                                         CUDA_R_16F, ld, a.hd, w.q, CUDA_R_16F, a.hd, sq,
                                                         &zero, w.s + off, CUDA_R_32F, w.N, ss, a.n_kv,
                                                         CUBLAS_COMPUTE_32F, s));
        }
        rows_softmax_k<<<static_cast<unsigned>(pairs), kThreads, 0, s>>>(a, w);
        IMP_CUDA_CHECK_LAUNCH();
        for (int part = 0; part < 2; ++part) {
            const int n = part == 0 ? n1 : w.N - n1;
            if (n <= 0)
                continue;
            const half* vp = a.v + static_cast<size_t>(part == 0 ? slot0 : 0) * ld;
            const int off = part == 0 ? 0 : n1;
            // O[pair, dim] = sum_key p * v: (hd x pairs) = V (hd x keys) x P (keys x pairs).
            IMP_CUBLAS_CHECK_LOG(lt_gemm_strided_batched(CUBLAS_OP_N, CUBLAS_OP_N, a.hd, per, n, &one, vp,
                                                         CUDA_R_16F, ld, a.hd, w.p + off, CUDA_R_16F, w.N, ss,
                                                         part == 0 ? &zero : &one, w.o, CUDA_R_32F, a.hd, sq,
                                                         a.n_kv, CUBLAS_COMPUTE_32F, s));
        }
        rows_out_k<<<w.R * a.n_heads, a.hd, 0, s>>>(a, w);
        IMP_CUDA_CHECK_LAUNCH();
        r0 += w.R;
    }
    return true;
}

}  // namespace imp::inkling
