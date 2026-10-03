// #2549: an FFN phase that refuses capture (moe_host_args_capture_guard) runs eager, the cache
// disables itself, and no exception reaches the prefill.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include "exec/executor_forward_moe_internal.h"
#include "exec/ffn_graph_cache.h"

namespace imp {
namespace {

TEST(FfnGraphCacheGpu, CaptureRefusalRunsEagerAndDisables) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    int* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, sizeof(int)), cudaSuccess);

    FfnGraphCache c;
    int completed = 0;
    int value = 0;
    auto fn = [&] {
        moe_host_args_capture_guard(stream);  // throws under capture
        ASSERT_EQ(cudaMemsetAsync(d, ++value, sizeof(int), stream), cudaSuccess);
        ++completed;
    };

    c.run(0, 16, 1, stream, fn);  // first sighting: eager
    EXPECT_EQ(completed, 1);
    EXPECT_NO_THROW(c.run(0, 16, 1, stream, fn));  // second: capture refused -> eager
    EXPECT_EQ(completed, 2);
    EXPECT_TRUE(c.disabled());
    EXPECT_EQ(c.graphs(), 0u);
    c.run(0, 16, 1, stream, fn);  // disabled: eager
    EXPECT_EQ(completed, 3);

    int h = 0;
    ASSERT_EQ(cudaMemcpyAsync(&h, d, sizeof(int), cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(h & 0xff, 3) << "the eager re-run after the refused capture executed";
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusActive;
    ASSERT_EQ(cudaStreamIsCapturing(stream, &cs), cudaSuccess);
    EXPECT_EQ(cs, cudaStreamCaptureStatusNone);

    cudaFree(d);
    cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace imp
