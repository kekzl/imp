#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace imp {

// WMMA Flash Attention 2 for sm_120, 128x64 tiles, double-buffered KV pipeline. Q:[batch,seq_q,
// n_heads,hd] K,V:[batch,seq_kv,n_kv_heads,hd] O: same as Q. sliding_window: 0=off, >0=last N KV.
// Returns false when no template fits (hd not in {64,96,128,256}, or smem over opt-in - hd=256
// needs ~176KB at Br=64 vs 99KB) or the launch errors. Callers must handle decline (old silent
// fallback was #654).
bool flash_attention_blackwell(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                               bool causal = true, int sliding_window = 0, float softcap = 0.0f,
                               cudaStream_t stream = nullptr, int q_offset = 0);

}  // namespace imp
