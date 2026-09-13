#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Coverage instrumentation (FA2-coverage dispatch): count of attention_cublas_prefill calls since
// last reset. A test asserts this stays 0 across a Gemma-4 prefill to prove the legacy path is
// unreachable for the target model set.
uint64_t attention_cublas_prefill_call_count();
void attention_cublas_prefill_reset_count();

// Prefill attention via cuBLAS: materialized QK^T + softmax + PV.
// Q:[q_len,n_heads*hd] K:[kv_len,n_kv_heads*hd] V:[kv_len,n_kv_heads*hd] O:[q_len,n_heads*hd] FP16.
// S: workspace [n_heads*q_len*kv_len], FP16 or FP32 (FP32 when it fits).
// q_offset = absolute position of Q[0]; causal masks K[j] for j > q_offset+i. q_offset=0 = square path.
// sliding_window>0 masks K[j] where (abs_pos-j) >= sliding_window; visible window is
// [abs_pos-sliding_window+1, abs_pos]. Default 0 (off).
// sinks (#547): per-head FP16 logits [n_heads]; adds exp(sink-max) to the softmax denominator,
// dropped after softmax. nullptr = off.
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap = 0.0f, int q_offset = 0, cudaStream_t stream = nullptr,
                              int sliding_window = 0, const void* sinks = nullptr);

// Slices attention_cublas_prefill into q-row chunks sized to the S workspace (S-matrix-overflow
// regime: long ctx x wide chunk). Sizing stays on the FP32-S path (use_fp32_s), floored to a
// multiple of 16; returns false if even a 16-row slice overflows, caller falls back to tiled FMHA.
bool attention_cublas_prefill_sliced(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                                     Tensor& S, int n_heads, int n_kv_heads, int head_dim, float scale,
                                     bool causal, float softcap = 0.0f, int q_offset = 0,
                                     cudaStream_t stream = nullptr, int sliding_window = 0,
                                     const void* sinks = nullptr);

// Force-create the static cuBLAS handle (safe to call multiple times). Engine init calls this so
// the first captured-stream call reuses it instead of cublasCreate (illegal under capture).
void attention_cublas_prewarm();

}  // namespace imp
