#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>

namespace imp {

// MXFP4 tensor core attention for prefill (sm_120): CUTLASS block-scaled MXFP4 GEMM for Q.K^T
// (~2x FP16 TC throughput), cuBLAS FP16 GEMM for P.V. Materializes full S=Q.K^T (O(seq^2));
// for long sequences use flash attention (attention_blackwell.cu) instead. Decode stays scalar
// software-dequant (GEMV is memory-bound).
// Q:[batch,seq_q,n_heads,hd] K,V:[batch,seq_kv,n_kv_heads,hd] O: same as Q, all FP16.
// Requires sm_120+, head_dim % 32 == 0, IMP_USE_CUTLASS at compile time, IMP_MXFP4_ATTENTION=1
// at runtime. Returns false if unsupported or GEMM fails.
bool attention_mxfp4_prefill(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& O, float scale,
                             bool causal, float softcap, cudaStream_t stream);

// Check if MXFP4 attention is available and enabled.
bool attention_mxfp4_available();

}  // namespace imp
