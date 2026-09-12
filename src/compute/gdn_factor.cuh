#pragma once

// Factored spare state for the batched speculative verify.
//
// The gated delta rule's per-token state update is a scalar decay plus a
// rank-1 outer product (gdn.cu, "Delta rule scan"):
//
//     H_new[s][d] = g * H[s][d] + k[s] * delta[d]
//
// so the state after the drafted row is fully determined by the state before
// it plus (g, k, delta). The batched verify used to carry the drafted row as a
// SECOND full state slot per batch slot - an exact duplicate of the pool,
// 79.5 MiB per slot on Qwen3.8-27B, which took max_batch 32 -> 18 and the KV
// pool 2339 -> 1429 blocks (docs/plans/2026-09-11-batched-mtp-verify.md).
// The triple is (1 + state_size + head_dim) floats per head instead of
// state_size * head_dim: 48 KiB against 1.5 MiB per layer on that model.
//
// Layout per (sequence, head), `gdn_factor_floats_per_head` floats:
//   [0]                      g, the decay scalar. EXACTLY 0 means "no pending
//                            update": g is expf(fmaxf(A*dt, -20)) and cannot
//                            reach 0, so the sentinel needs no side array.
//   [1 .. kGdnFactorPad)     padding, keeps k 16-byte aligned
//   [kGdnFactorPad + s]      k[s], the L2-normalised key, state_size of them
//   [kGdnFactorPad + SS + d] delta[d], head_dim of them

#include <cuda_runtime.h>

namespace imp {

inline constexpr int kGdnFactorPad = 4;

// Floats per (sequence, head) row of a factor buffer.
inline constexpr int gdn_factor_floats_per_head(int state_size, int head_dim) {
    return kGdnFactorPad + state_size + head_dim;
}

// Write the 0 sentinel into g for every (layer, head) of the named slots: the
// rows of rejected drafts, finished requests and reassigned slots, which the
// scan would otherwise apply on the next step. Defined in gdn_factor.cu.
void gdn_factor_clear(float* fac, int64_t layer_stride, int n_layers, const int* slots, int n_slots,
                      int n_heads, int fac_stride, cudaStream_t stream);

#ifdef __CUDACC__

// Apply a pending factored row to the state column this thread owns, in
// registers, right after the state load and before any token is processed.
// A no-op when `fac` is null or carries the 0 sentinel, so the caller does not
// have to know which sequences have a pending row.
// H_reg holds SS_PER rows starting at s_base of column d; k is indexed by the
// absolute row, delta by the column.
template <int SS_PER>
__device__ __forceinline__ void gdn_factor_apply(float (&H_reg)[SS_PER], const float* __restrict__ fac,
                                                 int state_size, int s_base, int d) {
    if (fac == nullptr)
        return;
    const float g = fac[0];
    if (g == 0.0f)
        return;
    const float* __restrict__ k = fac + kGdnFactorPad;
    const float delta_d = k[state_size + d];
#pragma unroll
    for (int s = 0; s < SS_PER; s++)
        H_reg[s] = g * H_reg[s] + k[s_base + s] * delta_d;
}

// Write the factored row for the token just processed. Mirrors the ownership
// the scan already has: every thread owns column d and rows
// [s_base, s_base + SS_PER), and k is block-wide in shared memory, so the
// d == 0 threads cover the whole of k between them and one thread writes g.
template <int SS_PER>
__device__ __forceinline__ void gdn_factor_store(float* __restrict__ fac, const float* __restrict__ k_smem,
                                                 int state_size, int s_base, int d, int part, float g,
                                                 float delta_d) {
    if (fac == nullptr)
        return;
    float* __restrict__ k_out = fac + kGdnFactorPad;
    if (part == 0)
        k_out[state_size + d] = delta_d;
    if (d == 0) {
#pragma unroll
        for (int s = 0; s < SS_PER; s++)
            k_out[s_base + s] = k_smem[s_base + s];
        if (part == 0)
            fac[0] = g;
    }
}

#endif  // __CUDACC__

}  // namespace imp
