#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {

// Causal FA2 prefill for head_dim 512: Q / O [n, nh, 512], K / V [kv_len, nkv, 512] fp16, query
// row i at position q_offset + i. softcap > 0 applies softcap * tanh(s / softcap) to the scaled
// scores. No sliding window, no sinks. False = refused.
[[nodiscard]] bool fa2_hd512_prefill(const half* q, const half* k, const half* v, half* o, int n, int kv_len,
                                     int nh, int nkv, float scale, float softcap, int q_offset,
                                     cudaStream_t stream);
// Successful fa2_hd512_prefill launches since the last reset (test instrumentation).
uint64_t fa2_hd512_prefill_call_count();
void fa2_hd512_prefill_reset_count();

}  // namespace imp
