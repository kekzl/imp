// #2354: WEIGHTS was charged per upload and never uncharged; freed weight sources (27B F16 LM
// head 2425 MiB, Flash-Next GDN packs 6857 MiB) stayed in the ledger and drove the whole-init
// residual negative (-495 / -3349 MiB). A keyed charge leaves with its pointer.
#include "memory/mem_account.h"
#include <gtest/gtest.h>

namespace imp {
namespace {

constexpr const char* kPool = "test_keyed_pool";

TEST(MemAccountKeyed, FreeUnchargesExactlyWhatWasCharged) {
    auto& m = MemAccount::instance();
    const int64_t base = m.pool_current(kPool);
    int a = 0, b = 0;
    m.note_alloc(kPool, &a, 1000);
    m.note_alloc(kPool, &b, 24);
    EXPECT_EQ(m.pool_current(kPool), base + 1024);
    m.note_free(&a);
    EXPECT_EQ(m.pool_current(kPool), base + 24);
    m.note_free(&b);
    EXPECT_EQ(m.pool_current(kPool), base);
}

TEST(MemAccountKeyed, FreeOfAnUnchargedPointerChangesNothing) {
    auto& m = MemAccount::instance();
    int a = 0, never = 0;
    m.note_alloc(kPool, &a, 512);
    const int64_t before = m.pool_current(kPool);
    m.note_free(&never);
    m.note_free(nullptr);
    EXPECT_EQ(m.pool_current(kPool), before);
    m.note_free(&a);
    m.note_free(&a);  // a second free of the same pointer uncharges once
    EXPECT_EQ(m.pool_current(kPool), before - 512);
}

}  // namespace
}  // namespace imp
