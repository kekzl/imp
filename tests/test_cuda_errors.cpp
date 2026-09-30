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

// #2211: a failed sampler readback sync fails the request; it never yields a token.
TEST(CudaErrorsTest, SyncedTokenIsReadOnlyAfterSuccessfulSync) {
    const int32_t h_token = 1234;
    EXPECT_EQ(synced_token_or_throw(cudaSuccess, &h_token, "forward sampler readback"), 1234);
    try {
        const int32_t tok = synced_token_or_throw(cudaErrorLaunchFailure, &h_token, "forward sampler readback");
        FAIL() << "a failed sync returned token " << tok;
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("forward sampler readback"), std::string::npos) << e.what();
    }
}

// #2307: a failed sampler scratch alloc or readback enqueue throws; the sampler never returns token 0.
TEST(CudaErrorsTest, FailedSamplerCallThrowsNamingClassAndSite) {
    EXPECT_NO_THROW(cuda_call_or_throw(cudaSuccess, "sample_topk_topp scratch"));
    try {
        cuda_call_or_throw(cudaErrorMemoryAllocation, "sample_topk_topp scratch");
        FAIL() << "a failed sampler call returned normally";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find(cudaGetErrorString(cudaErrorMemoryAllocation)), std::string::npos) << what;
        EXPECT_NE(what.find("sample_topk_topp scratch"), std::string::npos) << what;
    }
}

// #2307: a sampler that enqueued nothing (CUB scratch unavailable) fails the request.
TEST(CudaErrorsTest, UnenqueuedSamplerThrowsNamingSite) {
    EXPECT_NO_THROW(sampler_enqueued_or_throw(true, "sample_tokens"));
    try {
        sampler_enqueued_or_throw(false, "sample_tokens");
        FAIL() << "an unenqueued sampler returned normally";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("sample_tokens"), std::string::npos) << e.what();
    }
}

}  // namespace
}  // namespace imp
