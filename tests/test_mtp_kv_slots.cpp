// MTP head KV committed per slot (mtp_kv_allocate / mtp_commit_slot) on the fake backend.

#include <gtest/gtest.h>

#include "compute/mtp_forward.h"
#include "compute/mtp_forward_internal.h"
#include "memory/fake_backend.h"
#include "memory/vram_query.h"

using namespace imp;

// 4 slots of 3 granules: slot 0 is committed at allocation, the rest sits on the reserved-uncommitted
// ledger; a slot's first use commits its rows in both caches, a second use nothing; release returns all.
TEST(MtpKvSlots, CommitsASlotAtItsFirstUse) {
    const size_t slot = 3 * FakeBackend::kGranularity, kv_bytes = 4 * slot;
    const size_t ledger0 = vram_reserved_uncommitted_bytes();
    FakeBackend be;
    MtpDraftWorkspace ws;
    ws.kv_slot_elems = slot / 2;  // fp16 elements per slot
    ASSERT_TRUE(mtp_kv_allocate(ws, kv_bytes, 4, &be));
    ASSERT_NE(ws.d_k_cache_base, nullptr);
    ASSERT_NE(ws.d_v_cache_base, nullptr);
    EXPECT_EQ(be.stats().live_bytes, 2 * slot);
    EXPECT_EQ(ws.device_bytes, 2 * mtp_alloc_footprint(kv_bytes));
    EXPECT_EQ(ws.kv_pending, ws.device_bytes - 2 * slot);
    EXPECT_EQ(vram_reserved_uncommitted_bytes() - ledger0, ws.kv_pending);

    EXPECT_TRUE(mtp_commit_slot_now(ws, 0));  // committed at allocation
    EXPECT_EQ(be.stats().live_bytes, 2 * slot);
    EXPECT_TRUE(mtp_commit_slot_now(ws, 1));
    EXPECT_EQ(be.stats().live_bytes, 4 * slot);
    EXPECT_EQ(vram_reserved_uncommitted_bytes() - ledger0, ws.kv_pending);
    const size_t pending = ws.kv_pending;
    EXPECT_TRUE(mtp_commit_slot_now(ws, 1));  // again: nothing
    EXPECT_EQ(be.stats().live_bytes, 4 * slot);
    EXPECT_EQ(ws.kv_pending, pending);
    // Slot 1's rows are writable in both caches, at its offset.
    static_cast<char*>(ws.d_k_cache_base)[slot] = 1;
    static_cast<char*>(ws.d_v_cache_base)[2 * slot - 1] = 1;

    // The fake backend commits a prefix up to the range's end, so slot 3 brings slot 2 with it.
    EXPECT_TRUE(mtp_commit_slot_now(ws, 3));
    EXPECT_EQ(be.stats().live_bytes, 8 * slot);
    EXPECT_EQ(vram_reserved_uncommitted_bytes() - ledger0, ws.kv_pending);
    const int rows[] = {0, 3, 3, 1};
    EXPECT_TRUE(mtp_commit_slots(ws, rows, 4));  // all committed: no device call
    EXPECT_TRUE(mtp_commit_slot_now(ws, 7));     // out of range: left to the table check
    EXPECT_TRUE(mtp_commit_slot_now(ws, -1));
    EXPECT_EQ(be.stats().live_bytes, 8 * slot);

    mtp_kv_release(ws);
    EXPECT_EQ(be.stats().live_bytes, 0u);
    EXPECT_EQ(vram_reserved_uncommitted_bytes(), ledger0);
    EXPECT_EQ(ws.d_k_cache_base, nullptr);
    EXPECT_EQ(ws.kv_pending, 0u);
}

// The device refuses a slot: false, and the slot stays uncommitted for the next try.
TEST(MtpKvSlots, ARefusedCommitIsReported) {
    const size_t slot = 3 * FakeBackend::kGranularity;
    const size_t ledger0 = vram_reserved_uncommitted_bytes();
    FakeBackend be(/*capacity_bytes=*/3 * slot);  // slot 0 twice, one more slot of one cache
    MtpDraftWorkspace ws;
    ws.kv_slot_elems = slot / 2;
    ASSERT_TRUE(mtp_kv_allocate(ws, 4 * slot, 4, &be));
    EXPECT_FALSE(mtp_commit_slot_now(ws, 1));
    EXPECT_FALSE(mtp_commit_slot_now(ws, 1));  // still uncommitted
    EXPECT_EQ(vram_reserved_uncommitted_bytes() - ledger0, ws.kv_pending);
    mtp_kv_release(ws);
    EXPECT_EQ(vram_reserved_uncommitted_bytes(), ledger0);
    EXPECT_EQ(be.stats().live_bytes, 0u);
}
