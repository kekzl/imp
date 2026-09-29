#pragma once

#include <cuda_runtime_api.h>

namespace imp {

// #2295: diagnostics.profile times a forward with events + cudaStreamSynchronize; inside a stream
// capture that sync invalidates the capture, so a capturing (or invalidated) stream never profiles.
inline bool forward_profile_allowed(bool flag, cudaStreamCaptureStatus capture) {
    return flag && capture == cudaStreamCaptureStatusNone;
}

}  // namespace imp
