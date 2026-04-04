#pragma once

#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

namespace imp {

inline bool debug_forward_enabled() {
    static const bool enabled = (std::getenv("IMP_DEBUG_FORWARD") != nullptr);
    return enabled;
}

// Print min/max/mean/L2norm of a GPU tensor (first row only for multi-row tensors).
// Syncs the stream — only call when IMP_DEBUG_FORWARD is active.
inline void debug_tensor_stats(const char* name, const Tensor& t, cudaStream_t stream,
                               int row = 0, int max_rows = 1) {
    if (!debug_forward_enabled()) return;
    int cols = static_cast<int>(t.shape[t.ndim - 1]);
    int nrows = std::min(max_rows, static_cast<int>(t.shape[0]) - row);
    int n = cols * nrows;
    std::vector<float> host(n);

    if (t.dtype == DType::FP16) {
        std::vector<half> tmp(n);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(tmp.data(), static_cast<const half*>(t.data) + (int64_t)row * cols,
                         n * sizeof(half), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        for (int i = 0; i < n; i++) host[i] = __half2float(tmp[i]);
    } else if (t.dtype == DType::FP32) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(host.data(), static_cast<const float*>(t.data) + (int64_t)row * cols,
                         n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    } else {
        fprintf(stderr, "[DEBUG_FWD] %s: unsupported dtype %d\n", name, (int)t.dtype);
        return;
    }

    float vmin = host[0], vmax = host[0], vsum = 0, vl2 = 0;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n; i++) {
        float v = host[i];
        if (std::isnan(v)) { nan_count++; continue; }
        if (std::isinf(v)) { inf_count++; continue; }
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        vsum += v;
        vl2 += v * v;
    }
    float mean = vsum / std::max(n - nan_count - inf_count, 1);
    float l2 = std::sqrt(vl2);
    fprintf(stderr, "[DEBUG_FWD] %-30s  min=%+.6e  max=%+.6e  mean=%+.6e  L2=%.6e",
            name, vmin, vmax, mean, l2);
    if (nan_count > 0) fprintf(stderr, "  NaN=%d", nan_count);
    if (inf_count > 0) fprintf(stderr, "  Inf=%d", inf_count);
    fprintf(stderr, "\n");
}

} // namespace imp
