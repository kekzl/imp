// Fails a test that ends with a pending CUDA runtime error. IMP_CUDA_CHECK_LAUNCH and
// IMP_CUDA_CHECK_LOG only log, so a failed launch, memset or cudaFree otherwise passes.
// Linked into the GPU test modules (imp_add_test_module ... GPU); inert without a device.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

class PendingCudaErrorFails : public ::testing::EmptyTestEventListener {
    void OnTestEnd(const ::testing::TestInfo& info) override {
        if (!has_device())
            return;
        const cudaError_t err = cudaGetLastError();
        EXPECT_EQ(err, cudaSuccess) << info.test_suite_name() << "." << info.name()
                                    << " left a CUDA error: " << cudaGetErrorString(err);
    }

    static bool has_device() {
        static const bool yes = [] {
            int n = 0;
            const bool ok = cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
            if (!ok)
                (void)cudaGetLastError();  // clears the error the failed probe set
            return ok;
        }();
        return yes;
    }
};

[[maybe_unused]] const bool kInstalled = [] {
    ::testing::UnitTest::GetInstance()->listeners().Append(new PendingCudaErrorFails());
    return true;
}();

}  // namespace
