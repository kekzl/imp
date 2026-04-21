#include "compute/weight_dispatch.h"
#include "compute/gemm.h"
#include "core/logging.h"

namespace imp {

void gemm_dispatch(cublasLtHandle_t, const WeightHandle& w,
                   const Tensor& x, Tensor& y,
                   float alpha, float beta,
                   void*, size_t,
                   cudaStream_t stream) {
    switch (w.primary_tier) {
        case StorageTier::FP16: {
            int64_t wshape[2] = {w.shape[0], w.shape[1]};
            Tensor w_tensor(w.payload.fp16.data, DType::FP16, 2, wshape, true);
            gemm(w_tensor, x, y, alpha, beta, stream);
            return;
        }
        case StorageTier::FP8:
        case StorageTier::NVFP4:
        case StorageTier::CUTLASS_NVFP4:
        case StorageTier::MXFP4:
            IMP_LOG_FATAL("gemm_dispatch: tier %d not yet implemented (Task 2.5)",
                          static_cast<int>(w.primary_tier));
            return;
        case StorageTier::FP32:
        case StorageTier::Undefined:
            IMP_LOG_FATAL("gemm_dispatch: handle in invalid tier %d",
                          static_cast<int>(w.primary_tier));
            return;
    }
}

void gemv_dispatch(const WeightHandle& w, const Tensor& x, Tensor& y,
                   cudaStream_t stream) {
    switch (w.primary_tier) {
        case StorageTier::FP16: {
            int64_t wshape[2] = {w.shape[0], w.shape[1]};
            Tensor w_tensor(w.payload.fp16.data, DType::FP16, 2, wshape, true);
            gemm(w_tensor, x, y, 1.0f, 0.0f, stream);
            return;
        }
        case StorageTier::FP8:
        case StorageTier::NVFP4:
        case StorageTier::CUTLASS_NVFP4:
        case StorageTier::MXFP4:
            IMP_LOG_FATAL("gemv_dispatch: tier %d not yet implemented (Task 2.5)",
                          static_cast<int>(w.primary_tier));
            return;
        default:
            IMP_LOG_FATAL("gemv_dispatch: handle in invalid tier %d",
                          static_cast<int>(w.primary_tier));
            return;
    }
}

void gemm_grouped_dispatch(cublasLtHandle_t,
                           std::span<const WeightHandle* const>,
                           const Tensor&, Tensor&,
                           const int*, void*, size_t, cudaStream_t) {
    IMP_LOG_FATAL("gemm_grouped_dispatch: not yet implemented (Task 3.4)");
}

} // namespace imp
