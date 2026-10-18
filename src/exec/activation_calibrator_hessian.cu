// GPTQ second moments (roadmap row 101): H += x^T x per (layer, kind) through one cuBLAS GEMM
// (FP16 in, FP32 accumulate), written as mean x^T x to an IMPHES01 file.

#include "exec/activation_calibrator.h"
#include "compute/cublas_handle.h"
#include "core/logging.h"
#include "quant/calibration_hessian.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <vector>

namespace imp {

void ActivationCalibrator::accumulate_hessian_(int layer, TensorKind kind, const Tensor& input,
                                               int64_t row_stride, cudaStream_t stream) {
    // Inputs already counted under the kind GPTQ looks them up by (find_hessian).
    if (kind == TensorKind::WK || kind == TensorKind::WV || kind == TensorKind::W_UP)
        return;
    const int64_t rows = input.shape[0], K = input.shape[1];
    const uint32_t key = (static_cast<uint32_t>(layer) * 256u) + static_cast<uint32_t>(kind);
    auto it = hess_.find(key);
    if (it == hess_.end()) {
        HessEntry e;
        e.K = K;
        const size_t bytes = static_cast<size_t>(K) * static_cast<size_t>(K) * sizeof(float);
        e.d_H = static_cast<float*>(alloc_->allocate(bytes, "calibration_hessian"));
        if (!e.d_H) {
            IMP_LOG_WARN("calibration: no VRAM for the %.1f MiB Hessian of layer %d kind %s",
                         bytes / 1048576.0, layer, tensor_kind_name(kind));
            return;
        }
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(e.d_H, 0, bytes, stream));
        it = hess_.emplace(key, e).first;
    } else if (it->second.K != K) {
        return;
    }
    if (!cublas_)
        cublas_ = cublas_handle_create("calibration_hessian", CUBLAS_DEFAULT_MATH);
    auto h = static_cast<cublasHandle_t>(cublas_);
    if (cublasSetStream(h, stream) != CUBLAS_STATUS_SUCCESS)
        return;
    // Row-major X [rows, K] (ld row_stride) is column-major X^T [K, rows]: H = X^T (X^T)^T.
    const float one = 1.0f;
    const auto ld = static_cast<int>(row_stride);
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(K),
                                           static_cast<int>(K), static_cast<int>(rows), &one, input.data,
                                           CUDA_R_16F, ld, input.data, CUDA_R_16F, ld, &one, it->second.d_H,
                                           CUDA_R_32F, static_cast<int>(K), CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        IMP_LOG_WARN("calibration: Hessian GEMM failed (status %d) at layer %d kind %s", static_cast<int>(st),
                     layer, tensor_kind_name(kind));
        return;
    }
    it->second.rows += static_cast<uint64_t>(rows);
}

std::string ActivationCalibrator::write_hessians(const std::string& path, cudaStream_t stream) const {
    if (hess_.empty())
        return "";
    IMP_LOG_INFO("Hessians: %zu entries to %s", hess_.size(), path.c_str());
    HessianWriter w;
    if (std::string err = w.open(path, static_cast<uint32_t>(hess_.size())); !err.empty())
        return err;
    std::vector<float> host;
    for (const auto& [key, e] : hess_) {
        const size_t n = static_cast<size_t>(e.K) * static_cast<size_t>(e.K);
        host.resize(n);
        if (cudaMemcpyAsync(host.data(), e.d_H, n * sizeof(float), cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return "Hessian D2H failed";
        const float inv = e.rows > 0 ? 1.0f / static_cast<float>(e.rows) : 0.0f;
        for (float& v : host)
            v *= inv;
        const auto kind = static_cast<TensorKind>(key % 256u);
        if (std::string err = w.add(static_cast<int>(key / 256u), tensor_kind_name(kind), e.K, e.rows,
                                    host.data());
            !err.empty())
            return err;
    }
    return w.close();
}

}  // namespace imp
