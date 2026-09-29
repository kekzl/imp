#pragma once

#include "core/logging.h"

#include <cuda_runtime_api.h>

namespace imp {

// #2295: diagnostics.profile times a forward with events + cudaStreamSynchronize; inside a stream
// capture that sync invalidates the capture, so a capturing (or invalidated) stream never profiles.
inline bool forward_profile_allowed(bool flag, cudaStreamCaptureStatus capture) {
    return flag && capture == cudaStreamCaptureStatusNone;
}

// Profile event pairs (report-only): an unmeasured pair counts 0 ms, as before; report() WARNs once
// per step with the count and the first error.
struct ProfileElapsed {
    int n_unmeasured = 0;
    cudaError_t first_err = cudaSuccess;
    void operator()(float* ms, cudaEvent_t a, cudaEvent_t b) {
        if (const cudaError_t e = cudaEventElapsedTime(ms, a, b); e != cudaSuccess) {
            *ms = 0.0f;
            if (n_unmeasured++ == 0)
                first_err = e;
        }
    }
    void report() const {
        if (n_unmeasured > 0)
            IMP_LOG_WARN("profile: %d event pairs unmeasured (%s), counted as 0 ms", n_unmeasured,
                         cudaGetErrorString(first_err));
    }
};

}  // namespace imp
