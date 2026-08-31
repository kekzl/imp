// =============================================================================
// mtp_topw.cu — top-W selection over MTP draft logits
// =============================================================================
//
// Two implementations behind one contract (ids in descending-logit order into
// ws.d_topk, device-only, no sync):
//
//   - mtp_topw_reference: the Stage 0 probe's single-CTA kernel — one scan of
//     the whole vocabulary per width (713 us on a 248k vocab, measured on
//     Qwen3.8-27B). Measurement-grade; also the in-tree oracle the GPU test
//     compares against.
//   - mtp_topw_fast: the serving kernel for the multi-candidate draft
//     (speculative.mtp_tree_width > 1) — pass 1 splits the vocabulary across
//     kMtpTopWBlocks blocks, pass 2 merges the partial (value, id) pairs in
//     one block.
//
// Own TU: one logical unit, and mtp_forward.cu was already at its pinned
// size — a kernel edit here must not re-ptxas the whole draft forward.
// =============================================================================

#include "compute/mtp_forward.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace imp {

__device__ __forceinline__ float mtp_logit_to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float mtp_logit_to_float(float v) { return v; }

// Top-W over an FP16/FP32 vector [vocab_size]. Single CTA; top_w (≤ kMtpMaxTopW)
// sequential argmax passes, masking previously selected indices. Writes the W
// descending-logit indices to out_idx[0..top_w). W is tiny relative to the
// lm_head GEMM that produced the logits, so the extra passes are cheap.
template <typename T>
__global__ void mtp_topk_kernel(const T* __restrict__ logits, int vocab_size,
                                int top_w, int* __restrict__ out_idx) {
    constexpr int kThreads = 256;
    __shared__ float s_val[kThreads];
    __shared__ int   s_idx[kThreads];
    __shared__ int   s_found[kMtpMaxTopW];

    int tid = threadIdx.x;
    for (int w = 0; w < top_w; ++w) {
        float best_val = -1.0e38f;
        int   best_idx = 0;
        for (int i = tid; i < vocab_size; i += kThreads) {
            bool taken = false;
            for (int f = 0; f < w; ++f) {
                if (s_found[f] == i) { taken = true; break; }
            }
            if (taken) continue;
            float v = mtp_logit_to_float(logits[i]);
            if (v > best_val) { best_val = v; best_idx = i; }
        }
        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();
        for (int off = kThreads / 2; off > 0; off >>= 1) {
            if (tid < off) {
                if (s_val[tid + off] > s_val[tid]) {
                    s_val[tid] = s_val[tid + off];
                    s_idx[tid] = s_idx[tid + off];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_found[w]  = s_idx[0];
            out_idx[w]  = s_idx[0];
        }
        __syncthreads();
    }
}


// Two-pass serving top-W. Pass 1: each block owns a contiguous vocab slice
// and runs top_w masked argmax-reduce passes over it (the probe kernel's
// selection, restricted to the slice), writing (value, id) into the partial
// arrays at [blockIdx.x * kMtpMaxTopW + w]. A slice of ~4k entries costs
// top_w * slice/256 strided reads per thread — micro against the lm_head
// GEMV that produced the logits.
template <typename T>
__global__ void mtp_topw_pass1_kernel(const T* __restrict__ logits, int vocab_size, int top_w,
                                      float* __restrict__ part_val, int* __restrict__ part_idx) {
    constexpr int kThreads = 256;
    __shared__ float s_val[kThreads];
    __shared__ int s_idx[kThreads];
    __shared__ int s_found[kMtpMaxTopW];

    const int tid = threadIdx.x;
    const int slice = (vocab_size + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * slice;
    const int hi = min(vocab_size, lo + slice);
    for (int w = 0; w < top_w; ++w) {
        float best_val = -1.0e38f;
        int best_idx = -1;
        for (int i = lo + tid; i < hi; i += kThreads) {
            bool taken = false;
            for (int f = 0; f < w; ++f) {
                if (s_found[f] == i) {
                    taken = true;
                    break;
                }
            }
            if (taken)
                continue;
            const float v = mtp_logit_to_float(logits[i]);
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }
        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();
        for (int off = kThreads / 2; off > 0; off >>= 1) {
            if (tid < off) {
                if (s_val[tid + off] > s_val[tid]) {
                    s_val[tid] = s_val[tid + off];
                    s_idx[tid] = s_idx[tid + off];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_found[w] = s_idx[0];
            part_val[blockIdx.x * kMtpMaxTopW + w] = s_val[0];
            part_idx[blockIdx.x * kMtpMaxTopW + w] = s_idx[0];
        }
        __syncthreads();
    }
}

// Pass 2: one block merges the n_blocks * top_w partial candidates (a few
// hundred entries) with the same masked argmax selection; a slice shorter
// than top_w yields -1 ids with -1e38 values, which never win.
__global__ void mtp_topw_pass2_kernel(const float* __restrict__ part_val, const int* __restrict__ part_idx,
                                      int n_blocks, int top_w, int* __restrict__ out_idx) {
    constexpr int kThreads = 256;
    __shared__ float s_val[kThreads];
    __shared__ int s_idx[kThreads];
    __shared__ int s_found[kMtpMaxTopW];

    const int tid = threadIdx.x;
    const int n = n_blocks * kMtpMaxTopW;
    for (int w = 0; w < top_w; ++w) {
        float best_val = -1.0e38f;
        int best_idx = -1;
        for (int i = tid; i < n; i += kThreads) {
            if ((i % kMtpMaxTopW) >= top_w)
                continue;  // unused width slots
            const int cand = part_idx[i];
            if (cand < 0)
                continue;
            bool taken = false;
            for (int f = 0; f < w; ++f) {
                if (s_found[f] == cand) {
                    taken = true;
                    break;
                }
            }
            if (taken)
                continue;
            const float v = part_val[i];
            if (v > best_val) {
                best_val = v;
                best_idx = cand;
            }
        }
        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();
        for (int off = kThreads / 2; off > 0; off >>= 1) {
            if (tid < off) {
                if (s_val[tid + off] > s_val[tid]) {
                    s_val[tid] = s_val[tid + off];
                    s_idx[tid] = s_idx[tid + off];
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_found[w] = s_idx[0];
            out_idx[w] = s_idx[0];
        }
        __syncthreads();
    }
}

bool mtp_topw_fast(const void* d_logits, bool fp32_logits, int vocab_size, int top_w, MtpDraftWorkspace& ws,
                   cudaStream_t stream) {
    if (ws.d_topk == nullptr || ws.d_topk_part_val == nullptr || ws.d_topk_part_idx == nullptr ||
        top_w <= 0 || top_w > kMtpMaxTopW)
        return false;
    if (fp32_logits) {
        mtp_topw_pass1_kernel<<<kMtpTopWBlocks, 256, 0, stream>>>(static_cast<const float*>(d_logits),
                                                                  vocab_size, top_w, ws.d_topk_part_val,
                                                                  ws.d_topk_part_idx);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        mtp_topw_pass1_kernel<<<kMtpTopWBlocks, 256, 0, stream>>>(static_cast<const __half*>(d_logits),
                                                                  vocab_size, top_w, ws.d_topk_part_val,
                                                                  ws.d_topk_part_idx);
        IMP_CUDA_CHECK_LAUNCH();
    }
    mtp_topw_pass2_kernel<<<1, 256, 0, stream>>>(ws.d_topk_part_val, ws.d_topk_part_idx, kMtpTopWBlocks,
                                                 top_w, ws.d_topk);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

bool mtp_topw_reference(const void* d_logits, bool fp32_logits, int vocab_size, int top_w,
                        MtpDraftWorkspace& ws, cudaStream_t stream) {
    if (ws.d_topk == nullptr || top_w <= 0 || top_w > kMtpMaxTopW)
        return false;
    if (fp32_logits) {
        mtp_topk_kernel<<<1, 256, 0, stream>>>(static_cast<const float*>(d_logits), vocab_size, top_w,
                                               ws.d_topk);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        mtp_topk_kernel<<<1, 256, 0, stream>>>(static_cast<const __half*>(d_logits), vocab_size, top_w,
                                               ws.d_topk);
        IMP_CUDA_CHECK_LAUNCH();
    }
    return true;
}


}  // namespace imp
