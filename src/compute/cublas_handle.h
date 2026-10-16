#pragma once
// Create / destroy for the per-site lazy cuBLAS and cuBLASLt handles (H-015).
// Create failure: ERROR naming the site, then abort(). Each site keeps its own handle,
// math mode, cublasSetStream binding and IMP_REGISTER_CUDA_STATIC_RESET hook.

#include <cublasLt.h>
#include <cublas_v2.h>

namespace imp {

// cublasCreate + cublasSetMathMode(math); a failed math-mode set logs and keeps the handle.
cublasHandle_t cublas_handle_create(const char* site, cublasMath_t math);

// cublasLtCreate.
cublasLtHandle_t cublaslt_handle_create(const char* site);

// Reset-hook helpers: destroy when set, then null. Status discarded (pre-cudaDeviceReset).
void cublas_handle_destroy(cublasHandle_t& h);
void cublas_handle_destroy(cublasLtHandle_t& h);

}  // namespace imp
