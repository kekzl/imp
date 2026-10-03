// #2435: FFN prefill graph buckets. A bucket is a graph per layer, so the step sizes bound both the
// pad compute (<= 63 rows below 1024, <= 255 above) and the graph count.

#include <gtest/gtest.h>

#include "exec/ffn_graph_cache.h"

namespace imp {
namespace {

TEST(FfnGraphCache, BucketRowsStepsAndBounds) {
    EXPECT_EQ(FfnGraphCache::bucket_rows(1), 0) << "decode rows never take the prefill graph";
    EXPECT_EQ(FfnGraphCache::bucket_rows(2), 16);
    EXPECT_EQ(FfnGraphCache::bucket_rows(28), 32);
    EXPECT_EQ(FfnGraphCache::bucket_rows(64), 64);
    EXPECT_EQ(FfnGraphCache::bucket_rows(65), 128);
    EXPECT_EQ(FfnGraphCache::bucket_rows(1000), 1024);
    EXPECT_EQ(FfnGraphCache::bucket_rows(1024), 1024);
    EXPECT_EQ(FfnGraphCache::bucket_rows(1025), 1280);
    EXPECT_EQ(FfnGraphCache::bucket_rows(FfnGraphCache::kMaxRows), FfnGraphCache::kMaxRows);
    EXPECT_EQ(FfnGraphCache::bucket_rows(FfnGraphCache::kMaxRows + 1), 0);
    for (int n = 2; n <= FfnGraphCache::kMaxRows; ++n) {
        const int r = FfnGraphCache::bucket_rows(n);
        ASSERT_GE(r, n) << n;
        ASSERT_LT(r - n, n <= 1024 ? 64 : 256) << n;
    }
}

TEST(FfnGraphCache, EmptyCacheRunsEagerAndHoldsNoGraph) {
    FfnGraphCache c;
    int calls = 0;
    c.run(0, 0, 1, nullptr, [&] { ++calls; });  // rows 0 = eager, no CUDA call
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(c.graphs(), 0u);
    EXPECT_EQ(c.replays(), 0u);
    EXPECT_FALSE(c.disabled());
}

}  // namespace
}  // namespace imp
