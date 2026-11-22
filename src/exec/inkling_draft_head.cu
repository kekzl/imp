// Inkling2 DSpark draft, the recurrent Markov head (inkling_draft_head.h).

#include "exec/inkling_draft_head.h"

#include "core/logging.h"

#include <cmath>

namespace imp {

namespace {

// Recurrence step, part 1. One block per output row of [3r, 2r] (128 threads, 4 inputs each: a thread's
// loads come one after the other): raw[row] = b + rc + W [state, w1[prev]].
__global__ void draft_rnn_raw_k(const half* w, const float* b, const half* rc, const float* state,
                                const half* w1, const int32_t* prev, float* raw, int r) {
    __shared__ float sh[4];
    const int row = blockIdx.x;
    const half* wr = w + (size_t)row * 2 * r;
    const half* e = w1 + (size_t)(*prev) * r;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < 2 * r; j += blockDim.x)
        acc += __half2float(wr[j]) * (j < r ? state[j] : __half2float(e[j - r]));
    acc = draft_block_sum(acc, sh);
    if (threadIdx.x == 0)
        raw[row] = b[row] + __half2float(rc[row]) + acc;
}

// Part 2, one block of r threads: state = sigmoid(raw_0) * state + (1 - sigmoid(raw_0)) * tanh(raw_1),
// latent = tanh(raw_2).
__global__ void draft_rnn_gate_k(const float* raw, float* state, float* latent, int r) {
    const int c = threadIdx.x;
    const float g = 1.0f / (1.0f + expf(-raw[c]));
    state[c] = g * state[c] + (1.0f - g) * tanhf(raw[r + c]);
    latent[c] = tanhf(raw[2 * r + c]);
}

// Candidates of a draft row: the two largest logits (base + fold_b) of every kCandSpan columns, so the
// recurrent steps score 2 x spans tokens instead of the vocabulary. One block per (row, span), 4
// columns per thread. cand0: the candidates' softcapped logits; mass: sum of exp(softcapped logit)
// over the span, for the probability of a pick against the whole vocabulary.
constexpr int kCandSpan = 4096, kCandThreads = 1024;
__global__ void draft_cands_k(const half* base, const float* fold_b, int V, int spans, float cap,
                              int32_t* cand, float* cand0, float* mass) {
    __shared__ float sb[32];
    __shared__ int si[32];
    const int row = blockIdx.x / spans, sp = blockIdx.x % spans;
    const half* x = base + (size_t)row * V;
    const int j0 = sp * kCandSpan, j1 = min(V, j0 + kCandSpan);
    float v[4], sum = 0.0f;
    int nv = 0;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        v[nv] = cap * tanhf((__half2float(x[j]) + fold_b[j]) / cap);
        sum += expf(v[nv++]);
    }
    int taken = -1;
    for (int pass = 0; pass < 2; ++pass) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (int q = 0; q < nv; ++q) {
            const int j = j0 + threadIdx.x + q * blockDim.x;
            const float vq = v[q];
            if (j != taken && vq > best) {
                best = vq;
                idx = j;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) {
                best = ob;
                idx = oi;
            }
        }
        if ((threadIdx.x & 31) == 0) {
            sb[threadIdx.x >> 5] = best;
            si[threadIdx.x >> 5] = idx;
        }
        __syncthreads();
        for (unsigned k = 0; k < blockDim.x / 32; ++k)
            if (sb[k] > best || (sb[k] == best && si[k] < idx)) {
                best = sb[k];
                idx = si[k];
            }
        __syncthreads();
        taken = idx;
        if (threadIdx.x == 0) {
            cand[((size_t)row * spans + sp) * 2 + pass] = idx < j1 ? idx : -1;
            cand0[((size_t)row * spans + sp) * 2 + pass] = best;
        }
    }
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0)
        sb[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned k = 1; k < blockDim.x / 32; ++k)
            sum += sb[k];
        mass[(size_t)row * spans + sp] = sum;
    }
}

// One block per candidate (r threads): val = softcap(base[j] + fold_b[j] + fold_w[j] . latent).
__global__ void draft_score_k(const half* base, const int32_t* cand, const half* fold_w, const float* fold_b,
                              const float* latent, float cap, float* val, int r) {
    __shared__ float sh[8];
    const int j = cand[blockIdx.x];
    float acc = j >= 0 ? __half2float(fold_w[(size_t)j * r + threadIdx.x]) * latent[threadIdx.x] : 0.0f;
    acc = draft_block_sum(acc, sh);
    if (threadIdx.x == 0)
        val[blockIdx.x] = j >= 0 ? cap * tanhf((__half2float(base[j]) + fold_b[j] + acc) / cap) : -INFINITY;
}

// One block: out = the candidate with the largest value (lowest token id on ties), conf = its softmax
// probability: the vocabulary's mass with the candidates' share replaced by their scored values.
__global__ void draft_pick_k(const float* val, const int32_t* cand, const float* cand0, const float* mass,
                             int n, int32_t* out, float* conf) {
    __shared__ float sb[32];
    __shared__ int si[32];
    __shared__ float sm;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (cand[i] >= 0 && (val[i] > best || (val[i] == best && cand[i] < idx))) {
            best = val[i];
            idx = cand[i];
        }
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ob > best || (ob == best && oi < idx)) {
            best = ob;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        sb[threadIdx.x >> 5] = best;
        si[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned k = 1; k < blockDim.x / 32; ++k)
            if (sb[k] > best || (sb[k] == best && si[k] < idx)) {
                best = sb[k];
                idx = si[k];
            }
        *out = idx;
        sm = best;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += (cand[i] >= 0 ? expf(val[i]) - expf(cand0[i]) : 0.0f) + ((i & 1) == 0 ? mass[i / 2] : 0.0f);
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0)
        sb[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned k = 1; k < blockDim.x / 32; ++k)
            sum += sb[k];
        *conf = expf(sm) / fmaxf(sum, 1e-30f);
    }
}

}  // namespace

size_t draft_head_cands(int V) { return static_cast<size_t>((V + kCandSpan - 1) / kCandSpan) * 2; }

void draft_head_pick(const DraftHeadArgs& a, cudaStream_t s) {
    const int V = a.V, r = a.rank;
    const int spans = (V + kCandSpan - 1) / kCandSpan, n_cand = spans * 2;
    draft_cands_k<<<a.rows * spans, kCandThreads, 0, s>>>(a.logits, a.fold_b, V, spans, a.cap, a.cand,
                                                          a.cand0, a.mass);
    IMP_CUDA_CHECK_LAUNCH();
    for (int i = 0; i < a.rows; ++i) {
        const int32_t* cand = a.cand + static_cast<size_t>(i) * n_cand;
        draft_rnn_raw_k<<<3 * r, 128, 0, s>>>(a.rnn_state_w, a.rnn_b, a.rc + static_cast<size_t>(i) * 3 * r,
                                              a.state, a.w1, a.tok + i, a.raw, r);
        IMP_CUDA_CHECK_LAUNCH();
        draft_rnn_gate_k<<<1, r, 0, s>>>(a.raw, a.state, a.latent, r);
        IMP_CUDA_CHECK_LAUNCH();
        draft_score_k<<<n_cand, r, 0, s>>>(a.logits + static_cast<size_t>(i) * V, cand, a.fold_w, a.fold_b,
                                           a.latent, a.cap, a.val, r);
        IMP_CUDA_CHECK_LAUNCH();
        draft_pick_k<<<1, 128, 0, s>>>(a.val, cand, a.cand0 + static_cast<size_t>(i) * n_cand,
                                       a.mass + static_cast<size_t>(i) * spans, n_cand, a.tok + i + 1,
                                       a.conf + i);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace imp
