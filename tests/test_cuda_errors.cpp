// AUDIT_arch_2026 D-1: the sticky-error class table and the sync check that
// turn a device fault into a host signal (src/core/cuda_errors.h). CPU lane:
// cudaGetErrorString needs no device.
#include <gtest/gtest.h>
#include "core/cuda_errors.h"
#include <stdexcept>
#include <string>

namespace imp {
namespace {

TEST(CudaErrorsTest, StickyDeviceFaultsAreUnrecoverable) {
    for (cudaError_t e : {cudaErrorIllegalAddress, cudaErrorMisalignedAddress, cudaErrorLaunchFailure,
                          cudaErrorLaunchTimeout, cudaErrorHardwareStackError, cudaErrorIllegalInstruction,
                          cudaErrorECCUncorrectable, cudaErrorExternalDevice})
        EXPECT_TRUE(cuda_error_is_unrecoverable(e)) << cudaGetErrorString(e);
}

// The classes the tree clears on purpose (green-context init, a capture that
// was invalidated, cudaGraphDebugDotPrint on WSL2) and the ordinary
// allocation / argument failures must never stop the worker.
TEST(CudaErrorsTest, ClearableClassesAreRecoverable) {
    for (cudaError_t e : {cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorNotSupported,
                          cudaErrorStreamCaptureInvalidated, cudaErrorStreamCaptureUnsupported,
                          cudaErrorInvalidDevice, cudaErrorInvalidResourceHandle, cudaErrorNoDevice,
                          cudaErrorInvalidConfiguration, cudaErrorLaunchOutOfResources})
        EXPECT_FALSE(cuda_error_is_unrecoverable(e)) << cudaGetErrorString(e);
}

TEST(CudaErrorsTest, SyncFailureThrowsNamingClassAndSite) {
    EXPECT_NO_THROW(cuda_sync_or_throw(cudaSuccess, "collect_sampled_tokens"));
    try {
        cuda_sync_or_throw(cudaErrorIllegalAddress, "collect_sampled_tokens");
        FAIL() << "a failed sync returned normally";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find(cudaGetErrorString(cudaErrorIllegalAddress)), std::string::npos) << what;
        EXPECT_NE(what.find("collect_sampled_tokens"), std::string::npos) << what;
    }
}

}  // namespace
}  // namespace imp
