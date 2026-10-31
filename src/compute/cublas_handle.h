#pragma once
// Create / destroy for the per-site lazy cuBLASLt handles (H-015). Create failure: ERROR naming
// the site, then abort(). Each site keeps its own handle and IMP_REGISTER_CUDA_STATIC_RESET hook.

#include <cublasLt.h>

namespace imp {

// cublasLtCreate.
cublasLtHandle_t cublaslt_handle_create(const char* site);

// Reset-hook helpers: destroy when set, then null. Status discarded (pre-cudaDeviceReset).
void cublas_handle_destroy(cublasLtHandle_t& h);

}  // namespace imp
