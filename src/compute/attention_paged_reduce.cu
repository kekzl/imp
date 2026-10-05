// Split-K Phase 2 reduce for every paged decode kernel, moved verbatim from attention_paged.cu,
// plus the one-shot Q8_1 epilogue (#2439).
#include "compute/attention_paged.h"
#include "compute/attention_paged_common.cuh"
#include "compute/q8_1_quantize.cuh"
#include "core/pdl_device.cuh"
#include "core/pdl_launch.cuh"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <float.h>

namespace imp {

// Split-K Phase 2: reduces num_splits partial results into the final output. Grid:(batch,
// n_heads), Block: 128 threads; each block merges one (batch,head) pair's partials.

__global__ void paged_attention_reduce_kernel(
    const float* __restrict__ partial_out,  // [batch, n_heads, num_splits, (2+head_dim)]
    half* __restrict__ O,                   // [batch, 1, n_heads, head_dim]
    int n_heads, int head_dim, int num_splits, const half* __restrict__ attn_sinks,
    block_q8_1* __restrict__ q8_out, float* __restrict__ d8_out) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    pdl_wait();     // partials are the split-K phase 1 outputs (#2471)
    pdl_trigger();  // scheduling only: o_proj prefetches its weights during the merge
    const int partial_stride = 2 + head_dim;
    const float* base = partial_out +
                        (int64_t)((batch_idx * n_heads + head_idx) * num_splits) * partial_stride;

    // Step 1: Find global max across all splits (thread 0)
    __shared__ float s_global_max;
    __shared__ float s_global_l;

    // Per-split (m,l) pairs sit partial_stride floats apart; a single thread reading them serially
    // costs one un-overlapped memory latency each while the block waits at the syncthreads below.
    // Stage them with a parallel load first, then reduce in the SAME serial order (bit-identical,
    // only the latency is parallel). Splits are not capped at 32; kMaxStagedSplits=256 covers the
    // generous bound from paged_attention_splitk_fp8_tile_gqa_splits, bounded by scratch elsewhere.
    constexpr int kMaxStagedSplits = 256;
    __shared__ float s_m[kMaxStagedSplits];
    __shared__ float s_l[kMaxStagedSplits];
    __shared__ float s_w[kMaxStagedSplits];  // expf(m - gmax), Step 3 reuses it
    const bool staged = (num_splits <= kMaxStagedSplits);
    if (staged) {
        for (int s = tid; s < num_splits; s += blockDim.x) {
            s_m[s] = base[static_cast<int64_t>(s) * partial_stride];
            s_l[s] = base[s * partial_stride + 1];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gmax = -FLT_MAX;
        for (int s = 0; s < num_splits; s++) {
            float m = staged ? s_m[s] : base[static_cast<int64_t>(s) * partial_stride];
            gmax = fmaxf(gmax, m);
        }
        // gpt-oss learned sink (#547): virtual extra softmax column — joins
        // the global max and the denominator, dropped from the numerator.
        if (attn_sinks)
            gmax = fmaxf(gmax, __half2float(attn_sinks[head_idx]));
        s_global_max = gmax;

        // Step 2: Compute global denominator
        float gl = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = staged ? s_m[s] : base[static_cast<int64_t>(s) * partial_stride];
            float l = staged ? s_l[s] : base[s * partial_stride + 1];
            gl += expf(m - gmax) * l;
        }
        if (attn_sinks)
            gl += expf(__half2float(attn_sinks[head_idx]) - gmax);
        s_global_l = gl;
    }
    __syncthreads();

    float gmax = s_global_max;
    float gl = s_global_l;
    float inv_gl = (gl > 0.0f) ? (1.0f / gl) : 0.0f;

    // Split weight depends only on s but the loop previously called expf() once per (thread,split)
    // for the same per-split value; compute each once instead. Same expf on the same input, so the
    // weights are bit-identical.
    if (staged) {
        for (int s = tid; s < num_splits; s += blockDim.x)
            s_w[s] = expf(s_m[s] - gmax);  // expf, not __expf: must stay bit-identical
        __syncthreads();
    }

    // Step 3: Each thread handles a subset of head_dim elements
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float o_val = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float weight = staged ? s_w[s] : expf(base[static_cast<int64_t>(s) * partial_stride] - gmax);
            float o_s = base[s * partial_stride + 2 + d];
            o_val += weight * o_s;
        }
        o_val *= inv_gl;

        int out_idx = batch_idx * n_heads * head_dim + head_idx * head_dim + d;
        const half o_h = __float2half(o_val);
        stcs_half(&O[out_idx], o_h);
        if (q8_out != nullptr)  // block-uniform; a warp holds one whole 32-element Q8_1 block
            q8_1_quantize_warp(__half2float(o_h), q8_out, d8_out, out_idx);
    }
}

// Armed Q8_1 epilogue (paged_attention_arm_q8_epilogue); consumed by the next eligible reduce.
struct Q8Epilogue {
    block_q8_1* q8;
    float* d8;
    bool written;
};
static Q8Epilogue g_q8_epi{nullptr, nullptr, false};

void paged_attention_launch_reduce(float* partial, half* O, int batch_size, int n_heads, int head_dim,
                                   int num_splits, cudaStream_t stream, const half* attn_sinks) {
    dim3 grid(batch_size, n_heads);
    dim3 block(128);
    // The reduce kernel has always applied the sink term; this launcher used to
    // hard-code nullptr, so every quantised-KV split-K path silently dropped it
    // (#1345). Callers that have no sinks still pass nullptr and are unchanged.
    block_q8_1* q8 = nullptr;
    float* d8 = nullptr;
    if (g_q8_epi.q8 != nullptr && batch_size == 1 && head_dim % 32 == 0) {
        q8 = g_q8_epi.q8;
        d8 = g_q8_epi.d8;
        g_q8_epi = {nullptr, nullptr, true};
    }
    pdl::enable_kernel(paged_attention_reduce_kernel);
    pdl::launch(paged_attention_reduce_kernel, grid, block, size_t(0), stream,
                static_cast<const float*>(partial), O, n_heads, head_dim, num_splits, attn_sinks, q8, d8);
    IMP_CUDA_CHECK_LAUNCH();
}

void paged_attention_arm_q8_epilogue(bool arm, void* q8_1_out, float* d8_out) {
    g_q8_epi = {arm ? static_cast<block_q8_1*>(q8_1_out) : nullptr, arm ? d8_out : nullptr, false};
}

bool paged_attention_take_q8_epilogue() {
    const bool written = g_q8_epi.written;
    g_q8_epi = {nullptr, nullptr, false};
    return written;
}

}  // namespace imp
