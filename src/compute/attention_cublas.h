#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// Coverage instrumentation (FA2-coverage dispatch): how many times
// attention_cublas_prefill has run since the last reset. A test asserts this
// stays 0 across a Gemma-4 prefill to prove the materialized legacy path is
// unreachable for the target model set (the executed-kernel coverage gate).
uint64_t attention_cublas_prefill_call_count();
void attention_cublas_prefill_reset_count();

// Prefill attention via cuBLAS materialized QK^T + softmax + PV.
//
// Q: [q_len, n_heads * head_dim] FP16
// K: [kv_len, n_kv_heads * head_dim] FP16
// V: [kv_len, n_kv_heads * head_dim] FP16
// O: [q_len, n_heads * head_dim] FP16
// S: workspace, sized for [n_heads * q_len * kv_len] in FP16 or FP32 (FP32 picked when buffer fits).
//
// q_offset is the absolute position of Q[0] in the full sequence. When causal=true,
// Q[i] (abs pos = q_offset + i) is masked against K[j] for j > q_offset + i.
// q_offset=0 reproduces the historic square path exactly.
//
// sliding_window > 0 additionally masks K[j] where (abs_pos - j) >= sliding_window,
// i.e. the visible K window is [abs_pos - sliding_window + 1, abs_pos]. Defaults to 0 (off).
//
// sinks (gpt-oss, #547): per-head learned sink logits [n_heads] FP16. Each acts
// as a virtual extra softmax column: the denominator gains exp(sink - max) and
// the column is dropped after softmax (probabilities sum to < 1). nullptr = off.
void attention_cublas_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, Tensor& S,
                              int n_heads, int n_kv_heads, int head_dim, float scale, bool causal,
                              float softcap = 0.0f, int q_offset = 0, cudaStream_t stream = nullptr,
                              int sliding_window = 0, const void* sinks = nullptr);

// attention_cublas_prefill in q-row slices sized to the S workspace.
//
// Serves the S-matrix-overflow regime (long ctx_len × wide chunk) where the
// whole-call footprint n_heads*q_len*kv_len no longer fits S: each slice of
// q rows runs the normal materialized path against the full K/V with its own
// q_offset, so causal/SWA masking, softcap, and sinks compose row-wise
// unchanged. Slices are sized to keep the accurate FP32-S path (3× elements —
// see use_fp32_s in the .cu; hd=512 FP16-S truncates scores) and floored to a
// multiple of 16. Returns false without launching when even a 16-row slice
// would overflow the workspace — the caller falls back to the O(n) tiled FMHA.
// Measured at Gemma-4 global-layer shapes (nh=16, hd=512, Sq=2048): slices of
// 64..256 rows run 3.4-3.9× faster than the whole-chunk FMHA hd=512 fallback
// at Skv 8k/16k (docs/audit/gemma4_attn_routing_2026_07_16/PERF_LOG.md entry 4).
bool attention_cublas_prefill_sliced(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O,
                                     Tensor& S, int n_heads, int n_kv_heads, int head_dim, float scale,
                                     bool causal, float softcap = 0.0f, int q_offset = 0,
                                     cudaStream_t stream = nullptr, int sliding_window = 0,
                                     const void* sinks = nullptr);

// Force-create the static cuBLAS handle. Safe to call multiple times.
// Engine init calls this so the first attention_cublas_prefill invocation
// inside a captured stream can reuse the handle without cublasCreate
// (which does internal cudaMalloc for workspace and is illegal under capture).
void attention_cublas_prewarm();

}  // namespace imp
