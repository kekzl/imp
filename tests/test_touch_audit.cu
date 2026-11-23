// TouchAudit (memory/touch_audit.h): pages written between snapshot() and compare(), per tag.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "memory/touch_audit.h"
#include "memory/vram_allocator.h"

using namespace imp;

namespace {

const TouchAudit::Row* find(const std::vector<TouchAudit::Row>& rows, const std::string& tag) {
    for (const TouchAudit::Row& r : rows)
        if (r.tag == tag)
            return &r;
    return nullptr;
}

}  // namespace

// Two writes into a 1 MiB allocation, one of 4 bytes and one across a page boundary: 3 pages count as
// written. An allocation nothing wrote counts 0, one made after the snapshot counts as new.
TEST(TouchAuditGpu, CountsThePagesWrittenSinceTheSnapshot) {
    VRAMAllocator va;
    ASSERT_TRUE(va.init());
    const size_t page = TouchAudit::kPage, big = 16 * page, odd = 3 * page + 1027;
    auto* a = static_cast<unsigned char*>(va.allocate(big, "touch_test_a"));
    auto* b = static_cast<unsigned char*>(va.allocate(odd, "touch_test_b"));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(cudaMemset(a, 0, big), cudaSuccess);
    ASSERT_EQ(cudaMemset(b, 7, odd), cudaSuccess);
    TouchAudit audit;
    EXPECT_TRUE(audit.compare(va).empty());  // no snapshot yet
    ASSERT_TRUE(audit.snapshot(va));
    {
        const std::vector<TouchAudit::Row> rows = audit.compare(va);
        const TouchAudit::Row* ra = find(rows, "touch_test_a");
        ASSERT_NE(ra, nullptr);
        EXPECT_EQ(ra->bytes, big);
        EXPECT_EQ(ra->written, 0u);
    }
    ASSERT_EQ(cudaMemset(a + 5 * page + 100, 1, 4), cudaSuccess);  // page 5
    ASSERT_EQ(cudaMemset(a + 9 * page - 8, 1, 16), cudaSuccess);   // pages 8 and 9
    ASSERT_EQ(cudaMemset(b + page, 7, page), cudaSuccess);         // same bytes: not a change
    auto* c = static_cast<unsigned char*>(va.allocate(2 * page, "touch_test_a"));
    ASSERT_NE(c, nullptr);
    const std::vector<TouchAudit::Row> rows = audit.compare(va);
    const TouchAudit::Row* ra = find(rows, "touch_test_a");
    const TouchAudit::Row* rb = find(rows, "touch_test_b");
    ASSERT_NE(ra, nullptr);
    ASSERT_NE(rb, nullptr);
    EXPECT_EQ(ra->bytes, big);
    EXPECT_EQ(ra->written, 3 * page);
    EXPECT_EQ(ra->fresh, 2 * page);
    EXPECT_EQ(ra->allocs, 1);
    EXPECT_EQ(rb->bytes, odd);
    EXPECT_EQ(rb->written, 0u);
    // The last, partial page of b: a write there is at most the allocation's tail.
    ASSERT_EQ(cudaMemset(b + 3 * page, 9, 1024), cudaSuccess);
    const std::vector<TouchAudit::Row> rows2 = audit.compare(va);
    rb = find(rows2, "touch_test_b");
    ASSERT_NE(rb, nullptr);
    EXPECT_EQ(rb->written, page);
    // Most unwritten bytes first.
    ASSERT_GE(rows2.size(), 2u);
    EXPECT_GE(rows2[0].bytes - rows2[0].written, rows2[1].bytes - rows2[1].written);
    va.free(a);
    va.free(b);
    va.free(c);
}
