#include "compute/weight_dispatch.h"
#include "core/logging.h"

namespace imp {

void gemm_dispatch(cublasLtHandle_t, const WeightHandle& w,
                   const Tensor&, Tensor&,
                   float, float,
                   void*, size_t,
                   cudaStream_t) {
    IMP_LOG_FATAL("gemm_dispatch: not yet implemented for tier %d (kind %s)",
                  static_cast<int>(w.primary_tier), tensor_kind_name(w.kind));
}

void gemv_dispatch(const WeightHandle& w, const Tensor&, Tensor&, cudaStream_t) {
    IMP_LOG_FATAL("gemv_dispatch: not yet implemented for tier %d (kind %s)",
                  static_cast<int>(w.primary_tier), tensor_kind_name(w.kind));
}

void gemm_grouped_dispatch(cublasLtHandle_t,
                           std::span<const WeightHandle* const>,
                           const Tensor&, Tensor&,
                           const int*, void*, size_t, cudaStream_t) {
    IMP_LOG_FATAL("gemm_grouped_dispatch: not yet implemented");
}

} // namespace imp
