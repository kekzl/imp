// Prompt logprobs (#2207): teacher-forced log-softmax per prompt row, gathered on device.
// Same LM-head driver as perplexity_nll_partial; only [rows x (2 + 2N)] leaves the GPU.

#include "exec/executor.h"
#include "core/logging.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace imp {

namespace {

constexpr int kPlpThreads = 256;

// Block-wide (value, index) argmax; ties -> lowest index. Result in s_v[0], s_i[0].
__device__ void plp_block_argmax(float* s_v, int* s_i, float v, int i) {
    const int tid = threadIdx.x;
    s_v[tid] = v;
    s_i[tid] = i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float ov = s_v[tid + s];
            const int oi = s_i[tid + s];
            if (ov > s_v[tid] || (ov == s_v[tid] && oi < s_i[tid])) {
                s_v[tid] = ov;
                s_i[tid] = oi;
            }
        }
        __syncthreads();
    }
}

// One block per row. logprob = x[t] - logsumexp(x) (double sum, as perplexity_nll_kernel);
// rank = 1 + #{j : x[j] > x[t] or (x[j] == x[t] and j < t)}; top_n by (value desc, index asc).
__global__ void prompt_logprobs_kernel(const float* __restrict__ logits, const int32_t* __restrict__ targets,
                                       int V, int top_n, float* __restrict__ out_lp,
                                       int32_t* __restrict__ out_rank, int32_t* __restrict__ out_top_ids,
                                       float* __restrict__ out_top_lp) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lg = logits + static_cast<int64_t>(row) * V;
    const int target = targets[row];
    const float tv = lg[target];

    __shared__ float s_v[kPlpThreads];
    __shared__ int s_i[kPlpThreads];
    __shared__ double s_d[kPlpThreads];
    __shared__ int s_c[kPlpThreads];

    float mx = -INFINITY;
    int mi = 0;
    for (int j = tid; j < V; j += blockDim.x) {
        const float v = lg[j];
        if (v > mx) {
            mx = v;
            mi = j;
        }
    }
    plp_block_argmax(s_v, s_i, mx, mi);
    const float row_max = s_v[0];
    __syncthreads();

    double sum = 0.0;
    int above = 0;
    for (int j = tid; j < V; j += blockDim.x) {
        const float v = lg[j];
        sum += exp(static_cast<double>(v - row_max));
        above += (v > tv || (v == tv && j < target)) ? 1 : 0;
    }
    s_d[tid] = sum;
    s_c[tid] = above;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_d[tid] += s_d[tid + s];
            s_c[tid] += s_c[tid + s];
        }
        __syncthreads();
    }
    const double lse = log(s_d[0]) + static_cast<double>(row_max);
    if (tid == 0) {
        out_lp[row] = static_cast<float>(static_cast<double>(tv) - lse);
        out_rank[row] = s_c[0] + 1;
    }
    __syncthreads();

    // k-th pass: best element strictly after the (k-1)-th in (value desc, index asc) order.
    float prev_v = INFINITY;
    int prev_i = -1;
    for (int k = 0; k < top_n; ++k) {
        float bv = -INFINITY;
        int bi = V;
        for (int j = tid; j < V; j += blockDim.x) {
            const float v = lg[j];
            const bool after = v < prev_v || (v == prev_v && j > prev_i);
            if (after && (v > bv || (v == bv && j < bi))) {
                bv = v;
                bi = j;
            }
        }
        plp_block_argmax(s_v, s_i, bv, bi);
        prev_v = s_v[0];
        prev_i = s_i[0];
        if (tid == 0) {
            out_top_ids[static_cast<int64_t>(row) * top_n + k] = prev_i;
            out_top_lp[static_cast<int64_t>(row) * top_n + k] = static_cast<float>(
                static_cast<double>(prev_v) - lse);
        }
        __syncthreads();
    }
}

}  // namespace

void GraphExecutor::prompt_logprobs_partial(const int32_t* d_targets, int n_rows, int top_n, float* d_lp,
                                            int32_t* d_rank, int32_t* d_top_ids, float* d_top_lp,
                                            cudaStream_t stream) {
    if (!initialized_ || n_rows <= 0 || !d_targets || !d_lp || !d_rank)
        return;
    if (top_n > 0 && (!d_top_ids || !d_top_lp))
        return;
    const int V = model_->config().vocab_size;
    // allow_cutlass: same LM head as perplexity_nll_partial, so PPL from these rows matches the tool.
    for_each_lm_head_batch_(n_rows, stream, /*allow_cutlass=*/true, [&](const Tensor& lg, int row0, int csz) {
        prompt_logprobs_kernel<<<csz, kPlpThreads, 0, stream>>>(
            static_cast<const float*>(lg.data), d_targets + row0, V, top_n, d_lp + row0, d_rank + row0,
            d_top_ids ? d_top_ids + static_cast<int64_t>(row0) * top_n : nullptr,
            d_top_lp ? d_top_lp + static_cast<int64_t>(row0) * top_n : nullptr);
        IMP_CUDA_CHECK_LAUNCH();
    });
}

}  // namespace imp
