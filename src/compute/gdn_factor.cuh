#pragma once

// Factored spare state for batched speculative verify. GDN's update is
// H_new[s][d] = g*H[s][d] + k[s]*delta[d]; a drafted row is fully determined by (g,k,delta):
// (1+state_size+head_dim) floats/head replaces a second full state slot per batch slot.
// Layout per (sequence,head), gdn_factor_floats_per_head floats:
//   [0] g decay scalar; EXACTLY 0 = no pending update (g=expf(fmaxf(A*dt,-20)) never hits 0)
//   [1..kGdnFactorPad) padding for 16-byte k alignment
//   [kGdnFactorPad+s] k[s] (L2-normalised key, state_size entries)
//   [kGdnFactorPad+SS+d] delta[d] (head_dim entries)

#include <cuda_runtime.h>
#include <cstdint>

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

// Fold a row into a full state slot: H[s][d] = g*H[s][d] + k[s]*delta[d] over every (layer, head) of
// dst_slots[i], row read at fac_slots[i]; a g = 0 row leaves the slot as is. h_pool = slot 0, layer 0
// (float or bf16), strides in bytes. Bound verify at 2 drafts: dst = fac = the live slot (1-row state),
// the row is draft 1.
void gdn_factor_fold(void* h_pool, bool bf16, int64_t slot_stride_bytes, int64_t layer_stride_bytes,
                     const float* fac, int64_t fac_layer_stride, int n_layers, const int* dst_slots,
                     const int* fac_slots, int n, int n_heads, int state_size, int head_dim, int fac_stride,
                     cudaStream_t stream);

#ifdef __CUDACC__

// Token index whose row a scan with fac_out AND out_slots stores (the one after the snapshot), -1 = none.
__device__ __forceinline__ int gdn_factor_mid_row(const int* out_slots, const float* fac_out, int snap_n,
                                                  int real_n) {
    return out_slots && fac_out && snap_n >= 1 && snap_n < real_n ? snap_n : -1;
}

// Applies a pending factored row to the state column this thread owns, in registers, right
// after the state load and before any token. No-op when fac is null or the 0 sentinel.
// H_reg holds SS_PER rows from s_base of column d; k indexed by absolute row, delta by column.
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

// Writes the factored row for the token just processed. Mirrors scan ownership: thread
// owns column d and rows [s_base, s_base+SS_PER); k is block-wide shared memory so d==0
// threads cover all of k, one thread writes g.
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
