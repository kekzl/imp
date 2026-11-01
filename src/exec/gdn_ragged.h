#pragma once

// Ragged GDN forward (cross-sequence prefill, runtime.prefill_batch), with or without the
// one-row decode riders of a mixed step (runtime.prefill_mixed_decode_hybrid). Prompt members
// take one prefill conv and one scan each; riders take ONE batched decode conv and ONE batched
// decode scan (the separate decode step's kernels), so 31 riders cost 2 launches, not 62.

#include "core/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace imp {

class SSMState;
struct InferenceState;

// Member s owns rows [h_seq_offsets[s], h_seq_offsets[s+1]) and recurrent slot h_ssm_slots[s]
// (d_ssm_slots[s] on device). The LAST n_riders members are riders with exactly one row each.
struct GdnRaggedRows {
    int n_seq = 0;
    int n_riders = 0;
    const int* h_seq_offsets = nullptr;  // HOST [n_seq + 1]
    const int* h_ssm_slots = nullptr;    // HOST [n_seq]
    const int* d_ssm_slots = nullptr;    // DEVICE [n_seq]
    [[nodiscard]] int n_prompt() const { return n_seq - n_riders; }
    [[nodiscard]] int rider_row0() const { return h_seq_offsets[n_prompt()]; }
};

// From a ragged prefill state; throws when a rider does not own exactly one row.
[[nodiscard]] GdnRaggedRows gdn_ragged_rows(const InferenceState& state);

// Conv + SiLU of every member into conv_f32 [rows, channels]. xBC [rows, channels] is the conv
// input (F16; another dtype sends riders through the per-member prefill conv too).
void gdn_ragged_conv(const GdnRaggedRows& g, SSMState& st, int ssm_idx, const Tensor& xBC, const Tensor& w,
                     const Tensor& b, float* conv_f32, int conv_kernel, cudaStream_t stream);

struct GdnRaggedScanIo {
    const float* conv_f32 = nullptr;
    int conv_channels = 0;
    const half* alpha = nullptr;
    const half* beta = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    half* y = nullptr;
    int n_heads = 0;
    int head_dim = 0;
    int state_size = 0;
    int n_groups = 0;
    int grouped_layout = 0;
    float* chunkpar_ws = nullptr;  // gdn.chunkpar_scan workspace; nullptr = no per-member route
    size_t chunkpar_bytes = 0;
    int chunkpar_strip = 0;
};

// Per-member scan when the prompt members suit it (>= 1 member of >= 128 rows, <= 4 shorter
// ones; riders do not count): members >= 128 rows on the chunk-parallel scan, shorter ones on the
// fused single-sequence scan, riders on one batched decode scan. Returns false and launches
// nothing otherwise: the caller then runs the one batched ragged launch over every member.
[[nodiscard]] bool gdn_ragged_scan_split(const GdnRaggedRows& g, SSMState& st, int ssm_idx,
                                         const GdnRaggedScanIo& io, cudaStream_t stream);

}  // namespace imp
