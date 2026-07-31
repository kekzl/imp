#include "exec/activation_calibrator.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

namespace imp {

namespace {

constexpr int kThreads = 256;

// One thread per input channel, striding down the rows. Consecutive threads
// read consecutive columns of the same row, so the loads coalesce.
__global__ void accum_abs_cols_kernel(const half* __restrict__ x, int64_t rows, int64_t K, int64_t row_stride,
                                      double* __restrict__ sum) {
    int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= K)
        return;
    double acc = 0.0;
    for (int64_t r = 0; r < rows; r++)
        acc += fabs(static_cast<double>(__half2float(x[r * row_stride + j])));
    sum[j] += acc;
}

}  // namespace

ActivationCalibrator::~ActivationCalibrator() {
    if (!alloc_)
        return;
    for (auto& [key, e] : entries_)
        if (e.d_sum)
            alloc_->free(e.d_sum);
}

void ActivationCalibrator::accumulate(int layer, TensorKind kind, const Tensor& input, cudaStream_t stream) {
    if (!alloc_ || input.qtype != QType::F16 || !input.on_device || input.ndim != 2 || layer < 0) {
        if (alloc_ && input.qtype != QType::F16)
            skipped_non_fp16_++;
        return;
    }
    const int64_t rows = input.shape[0];
    const int64_t K = input.shape[1];
    if (rows <= 0 || K <= 0)
        return;
    const int64_t row_stride = (input.stride[0] > 0) ? input.stride[0] : K;

    const uint32_t key = static_cast<uint32_t>(layer) * 256u + static_cast<uint32_t>(kind);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        Entry e;
        e.K = K;
        // Allocating here is why calibration forces CUDA graphs off — an
        // allocation inside a capture is an error, not a slow path.
        e.d_sum = static_cast<double*>(
            alloc_->allocate(static_cast<size_t>(K) * sizeof(double), "activation_calibration"));
        if (!e.d_sum) {
            IMP_LOG_WARN("calibration: allocation failed for layer %d kind %s (K=%lld)", layer,
                         tensor_kind_name(kind), static_cast<long long>(K));
            return;
        }
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(e.d_sum, 0, static_cast<size_t>(K) * sizeof(double), stream));
        it = entries_.emplace(key, e).first;
    } else if (it->second.K != K) {
        // Same (layer, kind) arriving with a different inner dimension means the
        // key is not identifying what we think it is. Refuse rather than blend.
        IMP_LOG_WARN("calibration: layer %d kind %s changed K %lld -> %lld, ignoring", layer,
                     tensor_kind_name(kind), static_cast<long long>(it->second.K), static_cast<long long>(K));
        return;
    }

    const int blocks = static_cast<int>((K + kThreads - 1) / kThreads);
    accum_abs_cols_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<const half*>(input.data), rows, K,
                                                           row_stride, it->second.d_sum);
    IMP_CUDA_CHECK_LAUNCH();
    it->second.rows += static_cast<uint64_t>(rows);
}

CalibrationStats ActivationCalibrator::snapshot(const std::string& model_id) const {
    CalibrationStats out;
    out.model_id = model_id;
    if (entries_.empty())
        return out;
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());
    std::vector<double> host;
    for (const auto& [key, e] : entries_) {
        if (!e.d_sum || e.rows == 0)
            continue;
        host.resize(static_cast<size_t>(e.K));
        if (cudaMemcpy(host.data(), e.d_sum, host.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            IMP_LOG_WARN("calibration: D2H copy failed for key %u", key);
            continue;
        }
        CalibrationEntry ce;
        ce.layer = static_cast<int>(key / 256u);
        ce.kind = tensor_kind_name(static_cast<TensorKind>(key % 256u));
        ce.rows = e.rows;
        ce.mean_abs.resize(host.size());
        const double inv = 1.0 / static_cast<double>(e.rows);
        for (size_t i = 0; i < host.size(); i++)
            ce.mean_abs[i] = static_cast<float>(host[i] * inv);
        out.entries.push_back(std::move(ce));
    }
    return out;
}

}  // namespace imp
