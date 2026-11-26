// Audit D-6 rest: the prenorm and norm-fold handoffs are BufferStamps. A view of the stamped buffer
// retires the stamp (it may be a writer); only the consumer block's entry view carries it.

#include <gtest/gtest.h>

#include "exec/buffer_stamp.h"

namespace imp {
namespace {

int g_no = 0;  // stand-ins for norm_out_, another buffer and two norm weights
int g_h = 0;
int g_w = 0;
int g_w2 = 0;

TEST(BufferStamp, LiveUntilAViewOfItsBuffer) {
    BufferStamp<PrenormPayload> s;
    EXPECT_FALSE(s.live());
    s.stamp(&g_no, &g_w, 4, {7, &g_h});
    EXPECT_TRUE(s.matches(&g_no, &g_w, 4));
    EXPECT_FALSE(s.matches(&g_no, &g_w2, 4)) << "another norm weight";
    EXPECT_FALSE(s.matches(&g_no, &g_w, 5)) << "another row count";
    EXPECT_EQ(s.payload.xq_gen, 7u);
    EXPECT_EQ(s.payload.router, &g_h);
    s.touch(&g_h);  // a view of another buffer
    s.touch(nullptr);
    EXPECT_TRUE(s.live());
    s.touch(&g_no);  // a view of the stamped buffer: a possible writer
    EXPECT_FALSE(s.live());
    EXPECT_FALSE(s.matches(&g_no, &g_w, 4));
    s.stamp(&g_no, &g_w, 4);  // the next producer write
    EXPECT_TRUE(s.live());
    EXPECT_EQ(s.payload.xq_gen, 0u);
}

// The block entry: liveness before its view, the view, carry. Any second view retires the stamp.
TEST(BufferStamp, EntryViewCarriesOnlyALiveStamp) {
    BufferStamp<unsigned long long*> s;
    unsigned long long slot = 0;
    s.stamp(&g_no, &g_w, 1, &slot);
    bool was = s.live();
    s.touch(&g_no);
    s.carry(was);
    EXPECT_TRUE(s.matches(&g_no, &g_w, 1));
    EXPECT_EQ(s.payload, &slot);

    s.touch(&g_no);  // a writer's view between producer and consumer
    was = s.live();
    s.touch(&g_no);  // then the consumer's entry view
    s.carry(was);
    EXPECT_FALSE(s.live()) << "the entry view must not revive a retired stamp";
}

TEST(BufferStamp, ClearDropsTheStamp) {
    BufferStamp<PrenormPayload> s;
    s.stamp(&g_no, &g_w, 2, {3, &g_h});
    s.clear();
    EXPECT_FALSE(s.live());
    EXPECT_FALSE(s.matches(&g_no, &g_w, 2));
    EXPECT_EQ(s.payload.router, nullptr);
    s.carry(true);  // a carry after a clear stays empty: out is null
    EXPECT_FALSE(s.live());
}

}  // namespace
}  // namespace imp
