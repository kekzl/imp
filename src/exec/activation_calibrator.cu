#include "exec/activation_calibrator.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

namespace imp {

namespace {

constexpr int kThreads = 256;

// One thread per input channel, striding down rows: consecutive threads
// read consecutive columns of the same row (coalesced). Both moments in
// one pass (sum|x| for AWQ scale candidates, sum x^2 for the error weight)
// to avoid reading the activation twice.
__global__ void accum_abs_cols_kernel(const half* __restrict__ x, int64_t rows, int64_t K, int64_t row_stride,
                                      double* __restrict__ sum, double* __restrict__ sumsq) {
    int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= K)
        return;
    double acc = 0.0, accsq = 0.0;
    for (int64_t r = 0; r < rows; r++) {
        const double v = static_cast<double>(__half2float(x[r * row_stride + j]));
        acc += fabs(v);
        accsq += v * v;
    }
    sum[j] += acc;
    sumsq[j] += accsq;
}

// blockIdx.y = expert: the same column walk over that expert's row segment.
__global__ void accum_abs_cols_experts_kernel(const half* __restrict__ x, const int32_t* __restrict__ offsets,
                                              int64_t K, int64_t row_stride, double* __restrict__ sum,
                                              uint64_t* __restrict__ rows) {
    const int e = static_cast<int>(blockIdx.y);
    const int64_t r0 = offsets[e], r1 = offsets[e + 1];
    if (blockIdx.x == 0 && threadIdx.x == 0 && r1 > r0)
        rows[e] += static_cast<uint64_t>(r1 - r0);
    int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= K)
        return;
    double acc = 0.0, accsq = 0.0;
    for (int64_t r = r0; r < r1; r++) {
        const double v = static_cast<double>(__half2float(x[r * row_stride + j]));
        acc += fabs(v);
        accsq += v * v;
    }
    double* s = sum + static_cast<int64_t>(e) * 2 * K;
    s[j] += acc;
    s[K + j] += accsq;
}

}  // namespace

ActivationCalibrator::~ActivationCalibrator() {
    if (!alloc_)
        return;
    for (auto& [key, e] : entries_)
        if (e.d_sum)
            alloc_->free(e.d_sum);
    for (auto& [key, e] : expert_entries_) {
        if (e.d_sum)
            alloc_->free(e.d_sum);
        if (e.d_rows)
            alloc_->free(e.d_rows);
    }
}

void ActivationCalibrator::accumulate_experts(int layer, TensorKind kind, const Tensor& input,
                                              const int32_t* d_offsets, int ne, cudaStream_t stream) {
    if (!alloc_ || input.qtype != QType::F16 || !input.on_device || input.ndim != 2 || layer < 0 ||
        !d_offsets || ne <= 0) {
        if (alloc_ && input.qtype != QType::F16)
            skipped_non_fp16_++;
        return;
    }
    const int64_t K = input.shape[1];
    if (input.shape[0] <= 0 || K <= 0)
        return;
    const int64_t row_stride = (input.stride[0] > 0) ? input.stride[0] : K;

    const uint32_t key = static_cast<uint32_t>(layer) * 256u + static_cast<uint32_t>(kind);
    auto it = expert_entries_.find(key);
    if (it == expert_entries_.end()) {
        ExpertEntry e;
        e.K = K;
        e.ne = ne;
        const size_t sum_bytes = static_cast<size_t>(ne) * static_cast<size_t>(2 * K) * sizeof(double);
        e.d_sum = static_cast<double*>(alloc_->allocate(sum_bytes, "activation_calibration"));
        e.d_rows = static_cast<uint64_t*>(
            alloc_->allocate(static_cast<size_t>(ne) * sizeof(uint64_t), "activation_calibration"));
        if (!e.d_sum || !e.d_rows) {
            IMP_LOG_WARN("calibration: allocation failed for layer %d kind %s (%d experts, K=%lld)", layer,
                         tensor_kind_name(kind), ne, static_cast<long long>(K));
            if (e.d_sum)
                alloc_->free(e.d_sum);
            if (e.d_rows)
                alloc_->free(e.d_rows);
            return;
        }
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(e.d_sum, 0, sum_bytes, stream));
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(e.d_rows, 0, static_cast<size_t>(ne) * sizeof(uint64_t), stream));
        it = expert_entries_.emplace(key, e).first;
    } else if (it->second.K != K || it->second.ne != ne) {
        IMP_LOG_WARN("calibration: layer %d kind %s changed shape (K %lld, %d experts), ignoring", layer,
                     tensor_kind_name(kind), static_cast<long long>(K), ne);
        return;
    }

    const dim3 grid(static_cast<unsigned>((K + kThreads - 1) / kThreads), static_cast<unsigned>(ne));
    accum_abs_cols_experts_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const half*>(input.data),
                                                                 d_offsets, K, row_stride, it->second.d_sum,
                                                                 it->second.d_rows);
    IMP_CUDA_CHECK_LAUNCH();
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
        // Allocating here is why calibration forces CUDA graphs off (alloc inside
        // capture is an error). One buffer of 2K: [0,K) sum|x|, [K,2K) sum x^2,
        // avoiding two allocations per entry.
        e.d_sum = static_cast<double*>(
            alloc_->allocate(static_cast<size_t>(2 * K) * sizeof(double), "activation_calibration"));
        if (!e.d_sum) {
            IMP_LOG_WARN("calibration: allocation failed for layer %d kind %s (K=%lld)", layer,
                         tensor_kind_name(kind), static_cast<long long>(K));
            return;
        }
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(e.d_sum, 0, static_cast<size_t>(2 * K) * sizeof(double), stream));
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
                                                           row_stride, it->second.d_sum,
                                                           it->second.d_sum + K);
    IMP_CUDA_CHECK_LAUNCH();
    it->second.rows += static_cast<uint64_t>(rows);
}

CalibrationStats ActivationCalibrator::snapshot(const std::string& model_id) const {
    CalibrationStats out;
    out.model_id = model_id;
    if (empty())
        return out;
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());
    std::vector<double> host;
    for (const auto& [key, e] : entries_) {
        if (!e.d_sum || e.rows == 0)
            continue;
        host.resize(static_cast<size_t>(2 * e.K));
        if (cudaMemcpy(host.data(), e.d_sum, host.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            IMP_LOG_WARN("calibration: D2H copy failed for key %u", key);
            continue;
        }
        CalibrationEntry ce;
        ce.layer = static_cast<int>(key / 256u);
        ce.kind = tensor_kind_name(static_cast<TensorKind>(key % 256u));
        ce.rows = e.rows;
        const size_t k = static_cast<size_t>(e.K);
        ce.mean_abs.resize(k);
        ce.mean_sq.resize(k);
        const double inv = 1.0 / static_cast<double>(e.rows);
        for (size_t i = 0; i < k; i++) {
            ce.mean_abs[i] = static_cast<float>(host[i] * inv);
            ce.mean_sq[i] = static_cast<float>(host[k + i] * inv);
        }
        out.entries.push_back(std::move(ce));
    }
    std::vector<uint64_t> rows;
    for (const auto& [key, e] : expert_entries_) {
        const size_t k = static_cast<size_t>(e.K);
        host.resize(static_cast<size_t>(e.ne) * 2 * k);
        rows.resize(static_cast<size_t>(e.ne));
        if (cudaMemcpy(host.data(), e.d_sum, host.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(rows.data(), e.d_rows, rows.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
            IMP_LOG_WARN("calibration: D2H copy failed for expert key %u", key);
            continue;
        }
        const std::string kind = tensor_kind_name(static_cast<TensorKind>(key % 256u));
        for (int x = 0; x < e.ne; x++) {
            if (rows[static_cast<size_t>(x)] == 0)
                continue;  // never routed: no statistic, the expert stays round-to-nearest
            CalibrationEntry ce;
            ce.layer = static_cast<int>(key / 256u);
            ce.kind = kind + "." + std::to_string(x);
            ce.rows = rows[static_cast<size_t>(x)];
            const double* s = host.data() + static_cast<size_t>(x) * 2 * k;
            ce.mean_abs.resize(k);
            ce.mean_sq.resize(k);
            const double inv = 1.0 / static_cast<double>(ce.rows);
            for (size_t i = 0; i < k; i++) {
                ce.mean_abs[i] = static_cast<float>(s[i] * inv);
                ce.mean_sq[i] = static_cast<float>(s[k + i] * inv);
            }
            out.entries.push_back(std::move(ce));
        }
    }
    return out;
}

}  // namespace imp
