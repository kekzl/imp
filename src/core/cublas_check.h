#pragma once
// cuBLAS / cuBLASLt status checks, the cublasStatus_t twin of IMP_CUDA_CHECK_LOG (core/logging.h).
// Log-only: control flow unchanged; a deliberate discard is written (void)call with a reason.
#include <cublas_api.h>

#include "core/logging.h"

#define IMP_CUBLAS_CHECK_LOG(call)                                                     \
    do {                                                                               \
        const cublasStatus_t st_ = (call);                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                            \
            IMP_LOG_ERROR("cuBLAS error: %s at %s:%d - %s", #call, __FILE__, __LINE__, \
                          cublasGetStatusString(st_));                                 \
        }                                                                              \
    } while (0)
