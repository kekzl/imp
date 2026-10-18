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

// C-029: the bucket past kMaxRowBuckets evicts the oldest; its capture reuses the evicted exec
// through cudaGraphExecUpdate, and the replay runs the new bucket's work, not the old one's.
TEST(FfnGraphCacheGpu, EvictedExecIsUpdatedForTheNextBucket) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    int* d = nullptr;
    ASSERT_EQ(cudaMalloc(&d, sizeof(int)), cudaSuccess);
    FfnGraphCache c;
    auto read = [&] {
        int h = 0;
        EXPECT_EQ(cudaMemcpyAsync(&h, d, sizeof(int), cudaMemcpyDeviceToHost, stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        return h & 0xff;
    };
    const int n = FfnGraphCache::kMaxRowBuckets + 1;
    for (int b = 1; b <= n; ++b) {
        auto fn = [&, b] { ASSERT_EQ(cudaMemsetAsync(d, b, sizeof(int), stream), cudaSuccess); };
        c.run(0, 16 * b, 1, stream, fn);  // eager
        c.run(0, 16 * b, 1, stream, fn);  // capture + launch
        EXPECT_EQ(read(), b) << "bucket " << b;
    }
    EXPECT_FALSE(c.disabled());
    EXPECT_EQ(c.graphs(), static_cast<size_t>(FfnGraphCache::kMaxRowBuckets));
    EXPECT_EQ(c.updates(), 1u) << "bucket " << n << " reused the evicted exec of bucket 1";
    c.run(0, 16 * n, 1, stream, [] { FAIL() << "a replay must not run fn"; });
    EXPECT_EQ(read(), n);
    EXPECT_EQ(c.replays(), 1u);

    c.clear();
    cudaFree(d);
    cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace imp
