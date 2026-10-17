// GPTQ second moments (roadmap row 101): H += x^T x per (layer, kind) through one cuBLAS GEMM
// (FP16 in, FP32 accumulate), written as mean x^T x to an IMPHES01 file.

#include "exec/activation_calibrator.h"
#include "compute/cublas_handle.h"
#include "core/logging.h"
#include "quant/calibration_hessian.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace imp {

namespace {

// Rows with an L2 norm above this multiple of the call's median are left out of H: a handful of
// attention-sink tokens (norms 1e2-1e3x the rest) otherwise own the second moment.
constexpr float kOutlierRowFactor = 20.0f;

__global__ void row_norm_kernel(const half* __restrict__ x, int64_t rows, int64_t K, int64_t stride,
                                float* __restrict__ norm) {
    const int64_t r = blockIdx.x;
    float s = 0.0f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        const float v = __half2float(x[r * stride + k]);
        s += v * v;
    }
    __shared__ float red[256];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (static_cast<int>(threadIdx.x) < o)
            red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        norm[r] = sqrtf(red[0]);
}

// Copies rows into a dense [rows, K] buffer, zeroing the excluded ones.
__global__ void masked_copy_kernel(const half* __restrict__ x, int64_t K, int64_t stride,
                                   const uint8_t* __restrict__ keep, half* __restrict__ out) {
    const int64_t r = blockIdx.x;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x)
        out[r * K + k] = keep[r] ? x[r * stride + k] : __float2half(0.0f);
}

}  // namespace

void ActivationCalibrator::accumulate_hessian_(int layer, TensorKind kind, const Tensor& input,
                                               int64_t row_stride, cudaStream_t stream) {
    // Inputs already counted under the kind GPTQ looks them up by (find_hessian).
    if (kind == TensorKind::WK || kind == TensorKind::WV || kind == TensorKind::W_UP)
        return;
    const int64_t rows = input.shape[0], K = input.shape[1];
    const uint32_t key = static_cast<uint32_t>(layer) * 256u + static_cast<uint32_t>(kind);
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
    // Outlier rows out: norms, median on the host (calibration runs eager), masked dense copy.
    const size_t n_bytes = static_cast<size_t>(rows) * sizeof(float);
    const size_t x_bytes = static_cast<size_t>(rows) * static_cast<size_t>(K) * sizeof(half);
    auto* d_norm = static_cast<float*>(alloc_->allocate(n_bytes, "calibration_hessian"));
    auto* d_keep = static_cast<uint8_t*>(alloc_->allocate(static_cast<size_t>(rows), "calibration_hessian"));
    auto* d_x = static_cast<half*>(alloc_->allocate(x_bytes, "calibration_hessian"));
    struct Scratch {
        VRAMAllocator* a;
        void* p[3];
        ~Scratch() {
            for (void* q : p)
                if (q)
                    a->free(q);
        }
    } scratch{alloc_, {d_norm, d_keep, d_x}};
    if (!d_norm || !d_keep || !d_x)
        return;
    const auto* x = static_cast<const half*>(input.data);
    row_norm_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, rows, K, row_stride, d_norm);
    IMP_CUDA_CHECK_LAUNCH();
    std::vector<float> norm(static_cast<size_t>(rows));
    if (cudaMemcpyAsync(norm.data(), d_norm, n_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return;
    std::vector<float> sorted = norm;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(rows / 2), sorted.end());
    const float limit = kOutlierRowFactor * sorted[static_cast<size_t>(rows / 2)];
    std::vector<uint8_t> keep(static_cast<size_t>(rows));
    uint64_t kept = 0;
    for (int64_t r = 0; r < rows; r++) {
        keep[r] = norm[r] <= limit ? 1 : 0;
        kept += keep[r];
    }
    if (cudaMemcpyAsync(d_keep, keep.data(), keep.size(), cudaMemcpyHostToDevice, stream) != cudaSuccess)
        return;
    masked_copy_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(x, K, row_stride, d_keep, d_x);
    IMP_CUDA_CHECK_LAUNCH();

    if (!cublas_)
        cublas_ = cublas_handle_create("calibration_hessian", CUBLAS_DEFAULT_MATH);
    auto h = static_cast<cublasHandle_t>(cublas_);
    if (cublasSetStream(h, stream) != CUBLAS_STATUS_SUCCESS)
        return;
    // Row-major X [rows, K] is column-major X^T [K, rows] with ld = K: H = X^T (X^T)^T.
    const float one = 1.0f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(K),
                                           static_cast<int>(K), static_cast<int>(rows), &one, d_x, CUDA_R_16F,
                                           static_cast<int>(K), d_x, CUDA_R_16F, static_cast<int>(K), &one,
                                           it->second.d_H, CUDA_R_32F, static_cast<int>(K),
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        IMP_LOG_WARN("calibration: Hessian GEMM failed (status %d) at layer %d kind %s", static_cast<int>(st),
                     layer, tensor_kind_name(kind));
        return;
    }
    // The scratch is freed when this returns: the GEMM must have read it.
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    it->second.rows += kept;
    it->second.dropped += static_cast<uint64_t>(rows) - kept;
}

std::string ActivationCalibrator::write_hessians(const std::string& path, cudaStream_t stream) const {
    if (hess_.empty())
        return "";
    uint64_t kept = 0, dropped = 0;
    for (const auto& [key, e] : hess_) {
        kept += e.rows;
        dropped += e.dropped;
    }
    IMP_LOG_INFO("Hessians: %zu entries to %s, %llu rows, %llu outlier rows left out", hess_.size(),
                 path.c_str(), static_cast<unsigned long long>(kept),
                 static_cast<unsigned long long>(dropped));
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
