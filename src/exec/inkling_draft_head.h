#pragma once
// Inkling2 DSpark draft, the recurrent Markov head (inkling_draft.cu): picks the draft tokens one after
// the other among each row's candidates.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace imp {

// Sum of v over the block's threads; sh: one float per warp.
__device__ inline float draft_block_sum(float v, float* sh) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (unsigned i = 0; i < blockDim.x / 32; ++i)
        t += sh[i];
    __syncthreads();
    return t;
}

struct DraftHeadArgs {
    const half *logits = nullptr, *rc = nullptr;  // base logits [rows][V], context term [rows][3 rank]
    const half *rnn_state_w = nullptr, *w1 = nullptr, *fold_w = nullptr;
    const float *rnn_b = nullptr, *fold_b = nullptr;
    float *state = nullptr, *raw = nullptr, *latent = nullptr;  // [rank], [3 rank], [rank]
    float *val = nullptr, *cand0 = nullptr, *mass = nullptr;    // [cands], [rows][cands], [rows][cands / 2]
    float* conf = nullptr;                                      // out [rows]: probability of the pick
    int32_t *cand = nullptr, *tok = nullptr;                    // [rows][cands]; [0] anchor, [1..] the picks
    int V = 0, rank = 0, rows = 0;
    float cap = 30.0f;
};

// Candidates scored per row.
size_t draft_head_cands(int V);
void draft_head_pick(const DraftHeadArgs& a, cudaStream_t s);

}  // namespace imp
