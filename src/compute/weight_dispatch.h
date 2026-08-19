#pragma once

#include "core/weight_handle.h"
#include "core/tensor.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

namespace imp {

// Dense GEMM (prefill / multi-token path): y = alpha * W @ x + beta * y.
// W is described by handle (rows, cols, primary_tier, payload).
void gemm_dispatch(cublasLtHandle_t lt, const WeightHandle& w, const Tensor& x, Tensor& y, float alpha,
                   float beta, void* workspace, size_t workspace_bytes, cudaStream_t stream);

// Decode GEMV (single-token path). Same semantics, batch=1.
void gemv_dispatch(const WeightHandle& w, const Tensor& x, Tensor& y, cudaStream_t stream);

}  // namespace imp
