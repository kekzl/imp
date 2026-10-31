#include "compute/cublas_handle.h"

#include "core/cublas_check.h"
#include "core/logging.h"

#include <cstdlib>

namespace imp {

cublasLtHandle_t cublaslt_handle_create(const char* site) {
    cublasLtHandle_t h = nullptr;
    const cublasStatus_t st = cublasLtCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) {
        IMP_LOG_ERROR("%s: cublasLtCreate failed (status %d: %s)", site, static_cast<int>(st),
                      cublasLtGetStatusString(st));
        std::abort();
    }
    return h;
}

void cublas_handle_destroy(cublasLtHandle_t& h) {
    if (h) {
        (void)cublasLtDestroy(h);
        h = nullptr;
    }
}

}  // namespace imp
