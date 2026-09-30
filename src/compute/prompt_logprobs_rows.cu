// Fused prompt-logprobs row pass (#2257): online logsumexp, target rank and a per-thread top-N list
// in one read of the row, then a block merge. Replaces 2 + N full-row passes per row.

#include "compute/prompt_logprobs_rows.h"
#include "core/logging.h"

#include <cfloat>
#include <climits>
#include <cmath>

namespace imp {

namespace {

constexpr int kPlpRowThreads = 256;

__device__ __forceinline__ bool plp_better(float av, int ai, float bv, int bi) {
    return av > bv || (av == bv && ai < bi);
}

// Start of row `row` inside the slab at column n0; w = that slab's width.
__device__ __forceinline__ const float* plp_slab_row(const float* logits, int rows, int V, int slab_w,
                                                     int row, int n0, int& w) {
    w = min(slab_w, V - n0);
    return logits + static_cast<int64_t>(n0) * rows + static_cast<int64_t>(row) * w;
}

// Merges thread tid's list with thread o's list into tid's list (top_n best, both sorted).
__device__ __forceinline__ void plp_merge_lists(float* s_v, int* s_i, int tid, int o, int top_n) {
    float mv[kPlpMaxTopN];
    int mi[kPlpMaxTopN];
    int a = 0, b = 0;
#pragma unroll
    for (int k = 0; k < kPlpMaxTopN; ++k) {
        if (k < top_n) {
            const float av = s_v[a * kPlpRowThreads + tid], bv = s_v[b * kPlpRowThreads + o];
            const int ai = s_i[a * kPlpRowThreads + tid], bi = s_i[b * kPlpRowThreads + o];
            const bool take_a = plp_better(av, ai, bv, bi);
            mv[k] = take_a ? av : bv;
            mi[k] = take_a ? ai : bi;
            a += take_a ? 1 : 0;
            b += take_a ? 0 : 1;
        }
    }
#pragma unroll
    for (int k = 0; k < kPlpMaxTopN; ++k) {
        if (k < top_n) {
            s_v[k * kPlpRowThreads + tid] = mv[k];
            s_i[k * kPlpRowThreads + tid] = mi[k];
        }
    }
}

// Inserts (v, j) into thread tid's sorted list; returns the new N-th entry in (kv, ki).
__device__ __forceinline__ void plp_insert(float* s_v, int* s_i, int tid, int top_n, float v, int j,
                                           float& kv, int& ki) {
    int pos = top_n - 1;
    while (pos > 0 &&
           plp_better(v, j, s_v[(pos - 1) * kPlpRowThreads + tid], s_i[(pos - 1) * kPlpRowThreads + tid])) {
        s_v[pos * kPlpRowThreads + tid] = s_v[(pos - 1) * kPlpRowThreads + tid];
        s_i[pos * kPlpRowThreads + tid] = s_i[(pos - 1) * kPlpRowThreads + tid];
        --pos;
    }
    s_v[pos * kPlpRowThreads + tid] = v;
    s_i[pos * kPlpRowThreads + tid] = j;
    kv = s_v[(top_n - 1) * kPlpRowThreads + tid];
    ki = s_i[(top_n - 1) * kPlpRowThreads + tid];
}

__global__ void __launch_bounds__(kPlpRowThreads) prompt_logprobs_rows_kernel(
    const float* __restrict__ logits, int rows, int V, int slab_w, const int32_t* __restrict__ targets,
    int top_n, float* __restrict__ out_lp, int32_t* __restrict__ out_rank, int32_t* __restrict__ out_top_ids,
    float* __restrict__ out_top_lp) {
    extern __shared__ unsigned char plp_smem[];
    float* s_v = reinterpret_cast<float*>(plp_smem);  // [top_n][threads]
    int* s_i = reinterpret_cast<int*>(s_v + static_cast<size_t>(top_n) * kPlpRowThreads);
    __shared__ float s_m[kPlpRowThreads];
    __shared__ double s_s[kPlpRowThreads];
    __shared__ int s_c[kPlpRowThreads];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int target = targets[row];
    const int tn0 = (target / slab_w) * slab_w;
    int tw = 0;
    const float tv = plp_slab_row(logits, rows, V, slab_w, row, tn0, tw)[target - tn0];

    for (int k = 0; k < top_n; ++k) {
        s_v[k * kPlpRowThreads + tid] = -INFINITY;
        s_i[k * kPlpRowThreads + tid] = INT_MAX;
    }
    float kv = -INFINITY;  // this thread's N-th best so far
    int ki = INT_MAX;
    float m = -INFINITY;  // running max; s = sum exp(x - m)
    double s = 0.0;
    int above = 0;
    for (int n0 = 0; n0 < V; n0 += slab_w) {
        int w = 0;
        const float* lg = plp_slab_row(logits, rows, V, slab_w, row, n0, w);
        for (int c = tid; c < w; c += kPlpRowThreads) {
            const float v = lg[c];
            const int j = n0 + c;
            if (v > m) {
                s = s * static_cast<double>(expf(m - v)) + 1.0;
                m = v;
            } else if (v > -INFINITY) {
                s += static_cast<double>(expf(v - m));
            }
            above += (v > tv || (v == tv && j < target)) ? 1 : 0;
            if (top_n > 0 && plp_better(v, j, kv, ki))
                plp_insert(s_v, s_i, tid, top_n, v, j, kv, ki);
        }
    }

    s_m[tid] = m;
    __syncthreads();
    for (int st = kPlpRowThreads / 2; st > 0; st >>= 1) {
        if (tid < st)
            s_m[tid] = fmaxf(s_m[tid], s_m[tid + st]);
        __syncthreads();
    }
    const float row_max = s_m[0];
    s_s[tid] = (m > -INFINITY) ? s * exp(static_cast<double>(m) - static_cast<double>(row_max)) : 0.0;
    s_c[tid] = above;
    __syncthreads();
    for (int st = kPlpRowThreads / 2; st > 0; st >>= 1) {
        if (tid < st) {
            s_s[tid] += s_s[tid + st];
            s_c[tid] += s_c[tid + st];
            if (top_n > 0)
                plp_merge_lists(s_v, s_i, tid, tid + st, top_n);
        }
        __syncthreads();
    }
    const double lse = log(s_s[0]) + static_cast<double>(row_max);
    if (tid == 0) {
        out_lp[row] = static_cast<float>(static_cast<double>(tv) - lse);
        out_rank[row] = s_c[0] + 1;
    }
    if (tid < top_n) {
        const int64_t o = static_cast<int64_t>(row) * top_n + tid;
        out_top_ids[o] = s_i[static_cast<ptrdiff_t>(tid * kPlpRowThreads)];
        out_top_lp[o] = static_cast<float>(
            static_cast<double>(s_v[static_cast<ptrdiff_t>(tid * kPlpRowThreads)]) - lse);
    }
}

}  // namespace

void prompt_logprobs_rows(const float* logits, int rows, int V, int slab_w, const int32_t* targets, int top_n,
                          float* lp, int32_t* rank, int32_t* top_ids, float* top_lp, cudaStream_t stream) {
    if (rows <= 0 || V <= 0 || !logits || !targets || !lp || !rank)
        return;
    IMP_CHECK(top_n >= 0 && top_n <= kPlpMaxTopN, "prompt_logprobs_rows: top_n %d outside [0, %d]", top_n,
              kPlpMaxTopN);
    if (top_n > 0 && (!top_ids || !top_lp))
        return;
    const int sw = (slab_w <= 0 || slab_w > V) ? V : slab_w;
    const size_t smem = static_cast<size_t>(top_n) * kPlpRowThreads * (sizeof(float) + sizeof(int));
    prompt_logprobs_rows_kernel<<<rows, kPlpRowThreads, smem, stream>>>(logits, rows, V, sw, targets, top_n,
                                                                        lp, rank, top_ids, top_lp);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
