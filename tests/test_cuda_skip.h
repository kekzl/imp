#pragma once

// Single source of truth for the CUDA-presence gate: HasCudaDevice() was redefined in 6
// files and SKIP_IF_NO_CUDA() in 11, with three divergent bodies.
// Kept separate from test_model_builder.h: that header pulls in model.h + cuda_fp16/half,
// which host-compiled .cpp tests shouldn't need just to gate on a device. Needs only
// <cuda_runtime.h>.

#include <cuda_runtime.h>

namespace imp {
namespace test {

inline bool HasCudaDevice() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

}  // namespace test
}  // namespace imp

#define SKIP_IF_NO_CUDA()                               \
    do {                                                \
        if (!::imp::test::HasCudaDevice()) {            \
            GTEST_SKIP() << "No CUDA device available"; \
        }                                               \
    } while (0)
