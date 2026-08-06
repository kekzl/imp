#pragma once

#include "core/tensor.h"
#include "core/logging.h"
#include "runtime/process_diag.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <utility>

namespace imp {

inline bool debug_forward_enabled() { return imp::process_diag_debug_forward(); }

// Decode-step counter shared between executor_forward.cu (writer) and
// executor_ssm_gdn.cu (reader). Prefill passes use step=0; each single-token
// decode pass increments it. Used for diagnostics.dump_hidden_dir config flag (was
// IMP_DUMP_HIDDEN env) filename tagging so
// GDN-internal dumps correlate with per-layer snapshots.
inline int& debug_decode_step() {
    static int s = 0;
    return s;
}

// Hidden-state npy dump for layer-diff analysis against llama.cpp.
// Returns the directory if [diagnostics] dump_hidden_dir is non-empty, else
// nullptr. Accepts "1" or "all" as shorthand for /tmp (matches the legacy
// IMP_DUMP_HIDDEN env shorthand). Resolution happens in process_diag_install().
inline const char* dump_hidden_dir() { return imp::process_diag_dump_hidden_dir(); }

// Writes a numpy .npy v1.0 file with a 2D FP32 array.
// Self-describing (python: np.load(path) just works).
inline void write_npy_fp32(const std::string& path, const float* data, int rows, int cols) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        IMP_LOG_ERROR("[DUMP_NPY] open failed: %s", path.c_str());
        return;
    }
    // Build header. Total = 6(magic) + 2(version) + 2(hlen) + header.size
    // Pad with spaces so total is a multiple of 64 (numpy convention).
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    hdr += std::to_string(rows);
    hdr += ", ";
    hdr += std::to_string(cols);
    hdr += "), }";
    size_t pre = 6 + 2 + 2;
    size_t need = pre + hdr.size() + 1;  // +1 for trailing \n
    size_t pad = (64 - (need % 64)) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    std::fwrite("\x93NUMPY", 1, 6, f);
    std::fwrite("\x01\x00", 1, 2, f);
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    std::fwrite(data, sizeof(float), static_cast<size_t>(rows) * cols, f);
    std::fclose(f);
}

// Dump a 2D [rows, cols] tensor (FP16 or FP32) as FP32 .npy.
// Early-returns when the diagnostics.dump_hidden_dir config flag (was IMP_DUMP_HIDDEN env) is
// unset. Syncs the stream (debug only).
inline void dump_tensor_npy(const char* tag, const Tensor& t, cudaStream_t stream, int layer, int step) {
    const char* dir = dump_hidden_dir();
    if (!dir)
        return;
    if (t.qtype != QType::F16 && t.qtype != QType::F32)
        return;
    int cols = static_cast<int>(t.shape[t.ndim - 1]);
    int rows = (t.ndim >= 2) ? static_cast<int>(t.shape[0]) : 1;
    int64_t row_stride = (t.ndim >= 2 && t.stride[0] > 0) ? t.stride[0] : cols;
    size_t n = static_cast<size_t>(rows) * cols;

    std::vector<float> host(n);
    cudaStreamSynchronize(stream);
    if (t.qtype == QType::F16) {
        std::vector<half> tmp(n);
        if (row_stride == cols) {
            cudaMemcpy(tmp.data(), t.data, n * sizeof(half), cudaMemcpyDeviceToHost);
        } else {
            for (int r = 0; r < rows; r++) {
                cudaMemcpy(tmp.data() + static_cast<size_t>(r) * cols,
                           static_cast<const char*>(t.data) +
                               static_cast<int64_t>(r) * row_stride * sizeof(half),
                           cols * sizeof(half), cudaMemcpyDeviceToHost);
            }
        }
        for (size_t i = 0; i < n; i++)
            host[i] = __half2float(tmp[i]);
    } else {
        if (row_stride == cols) {
            cudaMemcpy(host.data(), t.data, n * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            for (int r = 0; r < rows; r++) {
                cudaMemcpy(host.data() + static_cast<size_t>(r) * cols,
                           static_cast<const char*>(t.data) +
                               static_cast<int64_t>(r) * row_stride * sizeof(float),
                           cols * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }
    }

    // A snapshot may point at a shared workspace whose tail is uninitialised, or
    // whose valid extent for THIS model is narrower than the view (attn_out_ is
    // both: its layout depends on head count, vhd-vs-hd and the MLA compaction
    // path). The file then looks like data and is not, and a diff against another
    // model reads as a finding — a relative error of 28.6 on one model and
    // exactly 1.0000 on another, which is what uncorrelated garbage looks like.
    // Say so instead of writing it silently.
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(host[i]))
            bad++;

    char fname[512];
    std::snprintf(fname, sizeof(fname), "%s/imp_step%02d_L%02d_%s.npy", dir, step, layer, tag);
    if (bad > 0) {
        IMP_LOG_WARN("[DUMP_NPY] %s: %zu/%zu values are not finite — this view likely covers "
                     "workspace beyond the valid region; do not diff it against another model",
                     fname, bad, n);
    }
    write_npy_fp32(fname, host.data(), rows, cols);
}

// Print min/max/mean/L2norm of a GPU tensor (first row only for multi-row tensors).
// Syncs the stream — only call when the diagnostics.debug_forward config flag (was
// IMP_DEBUG_FORWARD env) is active.
inline void debug_tensor_stats(const char* name, const Tensor& t, cudaStream_t stream, int row = 0,
                               int max_rows = 1) {
    if (!debug_forward_enabled())
        return;
    int cols = static_cast<int>(t.shape[t.ndim - 1]);
    int nrows = std::min(max_rows, static_cast<int>(t.shape[0]) - row);
    int n = cols * nrows;
    std::vector<float> host(n);

    if (t.qtype == QType::F16) {
        std::vector<half> tmp(n);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(tmp.data(), static_cast<const half*>(t.data) + (int64_t)row * cols,
                                           n * sizeof(half), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        for (int i = 0; i < n; i++)
            host[i] = __half2float(tmp[i]);
    } else if (t.qtype == QType::F32) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(host.data(),
                                           static_cast<const float*>(t.data) + (int64_t)row * cols,
                                           n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    } else {
        IMP_LOG_ERROR("[DEBUG_FWD] %s: unsupported dtype %d", name, std::to_underlying(t.qtype));
        return;
    }

    float vmin = host[0], vmax = host[0], vsum = 0, vl2 = 0;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n; i++) {
        float v = host[i];
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        if (v < vmin)
            vmin = v;
        if (v > vmax)
            vmax = v;
        vsum += v;
        vl2 += v * v;
    }
    float mean = vsum / std::max(n - nan_count - inf_count, 1);
    float l2 = std::sqrt(vl2);
    std::string extra;
    if (nan_count > 0)
        extra += "  NaN=" + std::to_string(nan_count);
    if (inf_count > 0)
        extra += "  Inf=" + std::to_string(inf_count);
    IMP_LOG_DEBUG("[DEBUG_FWD] %-30s  min=%+.6e  max=%+.6e  mean=%+.6e  sum=%+.4f  L2=%.6e%s", name, vmin,
                  vmax, mean, vsum, l2, extra.c_str());
}

// Multi-row variant: dump stats over ALL rows of a tensor for cross-impl
// comparison (matching llama.cpp's eval-callback sum dump). Sync first to
// avoid races against pending stream work.
// Dump first-3 and last-3 elements of each row (matches llama eval-callback's 3x3 slice).
inline void debug_tensor_rows(const char* name, const Tensor& t, cudaStream_t stream) {
    if (!debug_forward_enabled())
        return;
    cudaStreamSynchronize(stream);
    int cols = static_cast<int>(t.shape[t.ndim - 1]);
    int nrows = static_cast<int>(t.shape[0]);
    // stride[0] is in elements, not bytes. For contiguous [n, cols] it equals cols,
    // but for a slice view of a [max_tokens, cols] buffer it still equals cols.
    int64_t row_stride = (t.ndim >= 2 && t.stride[0] > 0) ? t.stride[0] : cols;
    if (t.qtype != QType::F16 && t.qtype != QType::F32)
        return;
    const int max_rows = (nrows < 6) ? nrows : 6;
    const size_t elem_sz = (t.qtype == QType::F16) ? sizeof(half) : sizeof(float);
    // Print one header line with shape/stride so layout bugs are visible.
    IMP_LOG_DEBUG("[DEBUG_FWD_ROW] %s: shape=[%d,%d] stride0=%lld nrows_dump=%d", name, nrows, cols,
                  (long long)row_stride, max_rows);
    std::vector<float> row_f(cols);
    for (int r = 0; r < max_rows; r++) {
        const char* src = static_cast<const char*>(t.data) + (int64_t)r * row_stride * elem_sz;
        if (t.qtype == QType::F16) {
            std::vector<half> tmp(cols);
            cudaMemcpy(tmp.data(), src, cols * sizeof(half), cudaMemcpyDeviceToHost);
            for (int i = 0; i < cols; i++)
                row_f[i] = __half2float(tmp[i]);
        } else {
            cudaMemcpy(row_f.data(), src, cols * sizeof(float), cudaMemcpyDeviceToHost);
        }
        // Simple per-row L2 to detect all-zero or broadcast-identical rows quickly.
        double ss = 0.0;
        for (int i = 0; i < cols; i++)
            ss += (double)row_f[i] * row_f[i];
        IMP_LOG_DEBUG("[DEBUG_FWD_ROW] %s[%d] L2=%.4f  %+.4f %+.4f %+.4f ...  %+.4f %+.4f %+.4f", name, r,
                      std::sqrt(ss), row_f[0], row_f[1], row_f[2], row_f[cols - 3], row_f[cols - 2],
                      row_f[cols - 1]);
    }
}

inline void debug_tensor_stats_all(const char* name, const Tensor& t, cudaStream_t stream) {
    if (!debug_forward_enabled())
        return;
    cudaStreamSynchronize(stream);  // wait for pending work on this stream
    int cols = static_cast<int>(t.shape[t.ndim - 1]);
    int nrows = static_cast<int>(t.shape[0]);
    int64_t n = static_cast<int64_t>(cols) * nrows;
    if (t.qtype != QType::F16 && t.qtype != QType::F32) {
        IMP_LOG_ERROR("[DEBUG_FWD_ALL] %s: unsupported dtype %d", name, std::to_underlying(t.qtype));
        return;
    }
    std::vector<float> host(n);
    if (t.qtype == QType::F16) {
        std::vector<half> tmp(n);
        cudaMemcpy(tmp.data(), t.data, n * sizeof(half), cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < n; i++)
            host[i] = __half2float(tmp[i]);
    } else {
        cudaMemcpy(host.data(), t.data, n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    double vsum = 0.0, vss = 0.0;
    for (int64_t i = 0; i < n; i++) {
        vsum += host[i];
        vss += host[i] * host[i];
    }
    IMP_LOG_DEBUG("[DEBUG_FWD_ALL] %-30s  rows=%d cols=%d  sum=%+.4f  L2=%.4f", name, nrows, cols, vsum,
                  std::sqrt(vss));
}

}  // namespace imp
