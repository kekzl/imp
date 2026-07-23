#include "compute/rowwise_topm.h"
#include "core/logging.h"

#include <cfloat>

namespace imp {

// One block per row; m sequential masked argmax passes (same block-reduce
// shape as rowwise_argmax_partial_kernel, tie-break = lowest index). The
// already-selected indices sit in shared memory and are skipped on later
// passes.
__global__ void rowwise_topm_kernel(const float* __restrict__ logits, int V, int m,
                                    int32_t* __restrict__ out) {
    const int row = blockIdx.x;
    const float* lg = logits + static_cast<int64_t>(row) * V;
    const int tid = threadIdx.x;
    __shared__ int s_sel[kRowwiseTopMMax];
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    for (int pass = 0; pass < m; ++pass) {
        float best = -FLT_MAX;
        int best_idx = V;  // sentinel: loses every tie-break
        for (int i = tid; i < V; i += blockDim.x) {
            bool taken = false;
            for (int e = 0; e < pass; ++e)
                if (s_sel[e] == i) {
                    taken = true;
                    break;
                }
            if (taken)
                continue;
            const float v = lg[i];
            if (v > best || (v == best && i < best_idx)) {
                best = v;
                best_idx = i;
            }
        }
        s_val[tid] = best;
        s_idx[tid] = best_idx;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_val[tid + s] > s_val[tid] ||
                    (s_val[tid + s] == s_val[tid] && s_idx[tid + s] < s_idx[tid])) {
                    s_val[tid] = s_val[tid + s];
                    s_idx[tid] = s_idx[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_sel[pass] = s_idx[0];
            out[static_cast<int64_t>(row) * m + pass] = s_idx[0];
        }
        __syncthreads();
    }
}

void rowwise_topm(const float* d_logits, int rows, int vocab, int m, int32_t* d_out,
                  cudaStream_t stream) {
    if (rows <= 0 || vocab <= 0 || m <= 0 || !d_logits || !d_out)
        return;
    if (m > kRowwiseTopMMax)
        m = kRowwiseTopMMax;
    rowwise_topm_kernel<<<rows, 256, 0, stream>>>(d_logits, vocab, m, d_out);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
