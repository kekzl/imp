// Factored conv window for the batched speculative verify. Companion to
// compute/gdn_factor.cuh (does the same for the recurrent state); a slot holds every GDN
// layer's conv window AND h_state (memory/ssm_state_size.h), so dropping the verify's
// spare slot needs both halves carried compactly (docs/plans/2026-09-12-factored-verify-spare.md).
// Conv half is cheaper: the drafted row shifts the kernel_size window by one, so the
// carried form is the single new tap (channels halfs) vs the whole window
// (channels*kernel_size floats) - 20 KiB vs 160 per layer at 10240 channels.
// Separate TU because ssm.cu sits at 596 of the 600-code-LOC kernel ceiling.

#include "compute/ssm.h"
#include "core/logging.h"
#include <cuda_fp16.h>

namespace imp {

namespace {

// Stash the drafted row's conv input. Reads the row the commit kernel would
// have folded into the spare window, indexed by SLOT so the row survives the
// request moving within the batch.
__global__ void ssm_conv_tap_stash_kernel(half* __restrict__ tap_pool, const half* __restrict__ x_in,
                                          int n_tokens, int channels, const int* __restrict__ d_real_n,
                                          const int* __restrict__ slots) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels)
        return;
    const int seq = blockIdx.y;
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    if (real_n <= 0)
        return;
    const half* row = x_in + (static_cast<size_t>(seq) * n_tokens + (real_n - 1)) * channels;
    tap_pool[static_cast<size_t>(slots[seq]) * channels + ch] = row[ch];
}

// Advance a window by the stashed tap: drop the oldest entry, append the tap.
// Equivalent to ssm_conv1d_commit_kernel over a single row, which is what the
// next step would have seen had the drafted row been committed at the time.
__global__ void ssm_conv_tap_apply_kernel(float* __restrict__ conv_pool, int64_t slot_stride,
                                          const half* __restrict__ tap_pool, int channels, int kernel_size,
                                          const int* __restrict__ slots) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels)
        return;
    const int slot = slots[blockIdx.y];
    float* w = conv_pool + static_cast<size_t>(slot) * static_cast<size_t>(slot_stride) + ch * kernel_size;
    for (int k = 0; k + 1 < kernel_size; k++)
        w[k] = w[k + 1];
    w[kernel_size - 1] = __half2float(tap_pool[static_cast<size_t>(slot) * channels + ch]);
}

constexpr int kTapThreads = 256;

}  // namespace

void ssm_conv_tap_stash(void* tap_pool, const void* x_in, int n_tokens, int channels, const int* d_real_n,
                        const int* slots, int n_seq, cudaStream_t stream) {
    if (!tap_pool || !x_in || !slots || n_seq <= 0 || channels <= 0 || n_tokens <= 0)
        return;
    dim3 grid((channels + kTapThreads - 1) / kTapThreads, n_seq);
    ssm_conv_tap_stash_kernel<<<grid, kTapThreads, 0, stream>>>(static_cast<half*>(tap_pool),
                                                                static_cast<const half*>(x_in), n_tokens,
                                                                channels, d_real_n, slots);
    IMP_CUDA_CHECK_LAUNCH();
}

void ssm_conv_tap_apply(void* conv_pool, int64_t slot_stride_floats, const void* tap_pool, int channels,
                        int kernel_size, const int* slots, int n_seq, cudaStream_t stream) {
    if (!conv_pool || !tap_pool || !slots || n_seq <= 0 || channels <= 0 || kernel_size <= 0)
        return;
    dim3 grid((channels + kTapThreads - 1) / kTapThreads, n_seq);
    ssm_conv_tap_apply_kernel<<<grid, kTapThreads, 0, stream>>>(static_cast<float*>(conv_pool),
                                                                slot_stride_floats,
                                                                static_cast<const half*>(tap_pool), channels,
                                                                kernel_size, slots);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
