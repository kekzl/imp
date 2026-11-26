// Audit D-6: the small-M activation tag retires a producer stamp when its buffer is touched (every
// view_tokens of it), instead of a forget call at each in-place writer.

#include <gtest/gtest.h>

#include "exec/smallm_xq_tag.h"

namespace imp {
namespace {

int g_no = 0;  // stand-ins for two activation buffers
int g_so = 0;

TEST(SmallMXqTag, ProducerStampIsLiveUntilItsBufferIsTouched) {
    SmallMXqTag t;
    t.stamp_producer(&g_no, 4, 256, 1.0f);
    EXPECT_TRUE(t.holds(&g_no, 4, 256, 1.0f));
    EXPECT_TRUE(t.producer_live());
    t.touch(&g_so);  // a view of another buffer
    t.touch(nullptr);
    EXPECT_TRUE(t.producer_live());
    t.touch(&g_no);  // a view of the stamped buffer: may be a writer
    EXPECT_FALSE(t.producer_live());
    EXPECT_TRUE(t.holds(&g_no, 4, 256, 1.0f))
        << "the scratch still holds the quantize, only the stamp retired";
    t.stamp_producer(&g_no, 4, 256, 1.0f);  // the next producer write
    EXPECT_TRUE(t.producer_live());
}

TEST(SmallMXqTag, DispatchQuantizeAndClearAreNotProducerStamps) {
    SmallMXqTag t;
    EXPECT_FALSE(t.producer_live());
    t.stamp_quantize(&g_no, 4, 256, 0.5f);
    EXPECT_TRUE(t.holds(&g_no, 4, 256, 0.5f));
    EXPECT_FALSE(t.holds(&g_no, 4, 256, 1.0f));
    EXPECT_FALSE(t.producer_live());
    t.stamp_producer(&g_no, 4, 256, 0.5f);
    t.clear();
    EXPECT_FALSE(t.holds(&g_no, 4, 256, 0.5f));
    EXPECT_FALSE(t.producer_live());
}

// Cross-op prenorm: the producer stamps in op A, op B's view retires it, op B's prenorm match adopts it.
TEST(SmallMXqTag, PrenormConsumerAdoptsOnlyTheProducersOwnStamp) {
    SmallMXqTag t;
    t.stamp_producer(&g_no, 8, 512, 1.0f);
    const uint64_t stamped = t.producer_gen;
    t.touch(&g_no);
    ASSERT_FALSE(t.producer_live());
    t.adopt(&g_no, stamped);
    EXPECT_TRUE(t.producer_live());

    // The producer quantized nothing (stamped 0): an older stamp on the same buffer stays retired.
    SmallMXqTag old;
    old.stamp_producer(&g_no, 8, 512, 1.0f);
    old.touch(&g_no);
    old.adopt(&g_no, 0);
    EXPECT_FALSE(old.producer_live());

    // A dispatch quantize or another buffer's stamp since the producer: nothing to adopt.
    SmallMXqTag requant;
    requant.stamp_producer(&g_no, 8, 512, 1.0f);
    const uint64_t s2 = requant.producer_gen;
    requant.touch(&g_no);
    requant.stamp_quantize(&g_no, 8, 512, 1.0f);
    requant.adopt(&g_no, s2);
    EXPECT_FALSE(requant.producer_live());
    SmallMXqTag other;
    other.stamp_producer(&g_no, 8, 512, 1.0f);
    const uint64_t s3 = other.producer_gen;
    other.stamp_producer(&g_so, 8, 512, 1.0f);
    other.touch(&g_so);
    other.adopt(&g_no, s3);
    EXPECT_FALSE(other.producer_live());
}

}  // namespace
}  // namespace imp
