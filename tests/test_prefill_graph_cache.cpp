// runtime.prefill_graph_prewarm: prompts up to short_max tokens keep a prefill graph runner each
// (eager first sighting, captured on the second); longer prompts share one runner that captures
// on a second consecutive sighting. No CUDA call: select() only books runners.

#include <gtest/gtest.h>

#include "runtime/prefill_graph_cache.h"

namespace imp {
namespace {

PrefillGraphCache::Key key(int n) { return {n, (n + 15) / 16, 0, -1}; }

TEST(PrefillGraphCache, ShortKeysKeepARunnerEach) {
    PrefillGraphCache c;
    c.set_short_max(64);
    EXPECT_EQ(c.select(key(20)), nullptr) << "first sighting runs eager";
    EXPECT_TRUE(c.warm(key(20)));
    CudaGraphRunner* r20 = c.select(key(20));
    ASSERT_NE(r20, nullptr);
    for (int n : {21, 22, 64}) {
        EXPECT_EQ(c.select(key(n)), nullptr);
        EXPECT_NE(c.select(key(n)), nullptr) << "a second sighting, not necessarily consecutive";
    }
    EXPECT_EQ(c.size(), 4u);
    EXPECT_EQ(c.select(key(20)), r20) << "other short lengths in between do not take its runner";
}

TEST(PrefillGraphCache, LongKeysShareOneRunnerOnConsecutiveSightings) {
    PrefillGraphCache c;
    c.set_short_max(64);
    EXPECT_EQ(c.select(key(500)), nullptr);
    EXPECT_EQ(c.select(key(501)), nullptr);
    EXPECT_EQ(c.select(key(500)), nullptr) << "alternating long lengths stay eager";
    CudaGraphRunner* r = c.select(key(500));
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(c.select(key(500)), r);
    EXPECT_EQ(c.select(key(700)), nullptr);
    EXPECT_EQ(c.select(key(700)), r) << "one shared runner, re-captured for the new shape";
    EXPECT_FALSE(c.warm(key(500))) << "the shared runner left 500";
    EXPECT_EQ(c.size(), 1u);
}

TEST(PrefillGraphCache, ShortMaxZeroIsTheSingleRunner) {
    PrefillGraphCache c;
    (void)c.select(key(30));
    CudaGraphRunner* r = c.select(key(30));
    ASSERT_NE(r, nullptr);
    (void)c.select(key(31));
    EXPECT_EQ(c.select(key(31)), r);
    EXPECT_EQ(c.size(), 1u);
    c.set_short_max(32);
    EXPECT_EQ(c.size(), 0u) << "the runner held a now-short key: dropped";
}

}  // namespace
}  // namespace imp
