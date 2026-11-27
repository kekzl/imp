// #2354: WEIGHTS was charged per upload and never uncharged; freed weight sources (27B F16 LM
// head 2425 MiB, Flash-Next GDN packs 6857 MiB) stayed in the ledger and drove the whole-init
// residual negative (-495 / -3349 MiB). A keyed charge leaves with its pointer.
#include "memory/mem_account.h"
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "death_test_exec.h"

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

// VRAMAllocator::free() notes here, also from a static owner destroyed at exit (the Inkling test
// fixture). The re-executed child first registers the late note, then first uses MemAccount, so a
// Meyers instance would be destroyed before it; 8 pools fill the vector, the late one reallocates.
void note_after_exit() { MemAccount::instance().note("test_exit_pool_late", -1); }
[[noreturn]] void exit_with_a_late_note() {
    std::atexit(&note_after_exit);
    for (int i = 0; i < 8; ++i)
        MemAccount::instance().note(("test_exit_pool_" + std::to_string(i)).c_str(), 1);
    std::exit(0);
}

TEST(MemAccountExit, AStaticOwnerStillNotesAfterExit) {
    imp_test::threadsafe_death_tests();
    EXPECT_EXIT(exit_with_a_late_note(), ::testing::ExitedWithCode(0), "");
}

}  // namespace
}  // namespace imp
