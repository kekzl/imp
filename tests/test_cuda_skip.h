#pragma once

// CUDA-presence gate shared across the test suite.
//
// Before this header, `HasCudaDevice()` was re-defined in 6 files and the
// `SKIP_IF_NO_CUDA()` macro in 11 — with three divergent bodies (a fully
// qualified call, a raw cudaGetDevice check, an unqualified call). This is the
// single source of truth.
//
// Kept separate from test_model_builder.h on purpose: that header pulls in
// model/model.h + cuda_fp16/half, which host-compiled .cpp tests should not be
// forced to include just to gate on a CUDA device. This header needs only
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
