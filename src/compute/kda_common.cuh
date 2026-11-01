#pragma once

// Device helpers shared by the KDA scans (kda_recurrent.cu, kda_scan_chunk.cu).

#include "compute/kda.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace imp::kda_detail {

constexpr int kHD = kKdaHeadDim;  // K = V = 128
constexpr int kVSplit = 4;        // CTAs per head along V: 16 heads x 4 = 64 CTAs per sequence
constexpr int kVCols = kHD / kVSplit;

// One sequence's view: its rows, its state slab, its commit row.
struct KdaSeq {
    size_t row0;
    int n_rows;
    int real_n;
};

__device__ __forceinline__ KdaSeq kda_resolve_seq(const KdaScanArgs& a, int seq) {
    KdaSeq s{};
    if (a.seq_row_offsets) {
        s.row0 = static_cast<size_t>(a.seq_row_offsets[seq]);
        s.n_rows = a.seq_row_offsets[seq + 1] - a.seq_row_offsets[seq];
    } else {
        s.row0 = static_cast<size_t>(seq) * static_cast<size_t>(a.n_tokens);
        s.n_rows = a.n_tokens;
    }
    s.real_n = a.d_real_n ? min(s.n_rows, __ldg(a.d_real_n)) : s.n_rows;
    return s;
}

template <typename StateT>
__device__ __forceinline__ StateT* kda_state_ptr(const KdaScanArgs& a, int seq, int h) {
    StateT* base = static_cast<StateT*>(a.h_state);
    if (a.seq_slots)
        base += static_cast<size_t>(a.seq_slots[seq]) * static_cast<size_t>(a.seq_stride);
    return base + (static_cast<size_t>(h) * kHD * kHD);
}

// Log-space decay of key channel c: lower_bound * sigmoid(exp(A_log) * (f + dt_bias)).
__device__ __forceinline__ float kda_log_decay(float f, float a_exp, float dt_bias, float lower_bound) {
    const float z = a_exp * (f + dt_bias);
    return lower_bound / (1.0f + __expf(-z));
}

__device__ __forceinline__ float kda_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

template <typename StateT>
__device__ __forceinline__ float kda_load(const StateT* p) {
    if constexpr (std::is_same_v<StateT, __nv_bfloat16>)
        return __bfloat162float(*p);
    else
        return *p;
}

template <typename StateT>
__device__ __forceinline__ void kda_store(StateT* p, float v) {
    if constexpr (std::is_same_v<StateT, __nv_bfloat16>)
        *p = __float2bfloat16(v);
    else
        *p = v;
}

inline void kda_check_args(const KdaScanArgs& a, const char* who) {
    if (a.n_heads <= 0 || a.conv_channels != 3 * a.n_heads * kHD)
        throw std::runtime_error(std::string(who) + ": conv_channels must be 3 * n_heads * 128");
    if (!a.conv_f32 || !a.f || !a.beta || !a.neg_a || !a.dt_bias || !a.h_state || !a.y)
        throw std::runtime_error(std::string(who) + ": null operand");
    if (a.seq_slots && a.seq_stride <= 0)
        throw std::runtime_error(std::string(who) + ": seq_slots without seq_stride");
}

}  // namespace imp::kda_detail
