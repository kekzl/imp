#include "compute/cublas_handle.h"

#include "core/cublas_check.h"
#include "core/logging.h"

#include <cstdlib>

namespace imp {

cublasHandle_t cublas_handle_create(const char* site, cublasMath_t math) {
    cublasHandle_t h = nullptr;
    const cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        IMP_LOG_ERROR("%s: cublasCreate failed (status %d: %s)", site, static_cast<int>(st),
                      cublasGetStatusString(st));
        std::abort();
    }
    IMP_CUBLAS_CHECK_LOG(cublasSetMathMode(h, math));
    return h;
}

cublasLtHandle_t cublaslt_handle_create(const char* site) {
    cublasLtHandle_t h = nullptr;
    const cublasStatus_t st = cublasLtCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        IMP_LOG_ERROR("%s: cublasLtCreate failed (status %d: %s)", site, static_cast<int>(st),
                      cublasGetStatusString(st));
        std::abort();
    }
    return h;
}

void cublas_handle_destroy(cublasHandle_t& h) {
    if (h) {
        (void)cublasDestroy(h);
        h = nullptr;
    }
}

void cublas_handle_destroy(cublasLtHandle_t& h) {
    if (h) {
        (void)cublasLtDestroy(h);
        h = nullptr;
    }
}

}  // namespace imp
