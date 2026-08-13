// L1 of the memory architecture: Backend, the allocation-phase guard, and the
// FakeBackend test seam (docs/internals/MEMORY.md A3.1/A3.2/A6).
//
// CPU-only on purpose. imp has no GPU runner and the lane CI actually runs is
// `ctest -L unit`; the whole point of the Backend interface is that the
// allocator stack above it can be exercised on host memory. Nothing here calls
// CUDA.

#include <gtest/gtest.h>

#include "memory/backend.h"
#include "memory/fake_backend.h"

#include <cstring>
#include <string>
#include <vector>

using namespace imp;

namespace {

// Every test runs in Loading unless it says otherwise, and the phase is
// process-global, so restore it.
class PhaseFixture : public ::testing::Test {
protected:
    void SetUp() override {
        prev_ = alloc_phase();
        set_alloc_phase(AllocPhase::Loading);
        reset_steady_state_allocations();
    }
    void TearDown() override {
        set_alloc_phase(prev_);
        reset_steady_state_allocations();
    }

private:
    AllocPhase prev_ = AllocPhase::Loading;
};

constexpr size_t kMiB = 1024 * 1024;

}  // namespace

// ── Region is RAII and move-only ──────────────────────────────────────

TEST_F(PhaseFixture, RegionReleasesOnDestruction) {
    FakeBackend be;
    {
        auto r = be.acquire(4096, 256, RegionTag::EnginePersistent);
        ASSERT_TRUE(r);
        EXPECT_EQ(be.stats().live_bytes, 4096u);
        EXPECT_EQ(be.live_regions(), 1u);
    }
    EXPECT_EQ(be.stats().live_bytes, 0u);
    EXPECT_EQ(be.live_regions(), 0u);
    EXPECT_EQ(be.stats().release_count, 1u);
}

TEST_F(PhaseFixture, MoveTransfersOwnershipExactlyOnce) {
    FakeBackend be;
    {
        auto r = be.acquire(4096, 256, RegionTag::EnginePersistent);
        ASSERT_TRUE(r);
        void* base = r.region.base();

        Region moved = std::move(r.region);
        EXPECT_EQ(moved.base(), base);
        EXPECT_FALSE(r.region.valid()) << "moved-from region must not still own the memory";
        EXPECT_EQ(be.stats().live_bytes, 4096u) << "a move must not release anything";
    }
    // Exactly one release for the pair — a double-free would show as 2.
    EXPECT_EQ(be.stats().release_count, 1u);
    EXPECT_EQ(be.stats().live_bytes, 0u);
}

TEST_F(PhaseFixture, ResetIsIdempotent) {
    FakeBackend be;
    auto r = be.acquire(4096, 256, RegionTag::Other);
    ASSERT_TRUE(r);
    r.region.reset();
    r.region.reset();
    EXPECT_EQ(be.stats().release_count, 1u);
}

TEST_F(PhaseFixture, ReleasedMemoryIsPoisoned) {
    FakeBackend be;
    auto r = be.acquire(1024, 256, RegionTag::Other);
    ASSERT_TRUE(r);
    void* base = r.region.base();
    std::memset(base, 0x11, 1024);
    r.region.reset();
    // Quarantined rather than freed precisely so this read is legal.
    EXPECT_TRUE(FakeBackend::is_poisoned(base, 1024))
        << "release must scrub, so a use-after-free is a deterministic failure";
}

// ── V1: conservation ──────────────────────────────────────────────────

TEST_F(PhaseFixture, JournalConservationHoldsAfterEveryOperation) {
    FakeBackend be;
    std::vector<Region> held;
    for (int i = 0; i < 32; ++i) {
        auto r = be.acquire(static_cast<size_t>(1 + i) * 256, 256, RegionTag::ModelResident);
        ASSERT_TRUE(r) << "i=" << i;
        held.push_back(std::move(r.region));
        EXPECT_EQ(be.journal_live_bytes(), be.stats().live_bytes) << "after acquire i=" << i;
    }
    // Release in a scrambled order — conservation must not depend on it.
    for (size_t i = 0; i < held.size(); i += 3)
        held[i].reset();
    EXPECT_EQ(be.journal_live_bytes(), be.stats().live_bytes);
    held.clear();
    EXPECT_EQ(be.journal_live_bytes(), 0u);
    EXPECT_EQ(be.stats().live_bytes, 0u);
}

// ── V2: no allocation while serving ───────────────────────────────────

TEST_F(PhaseFixture, ServingPhaseAcquisitionIsCounted) {
    // NDEBUG only: a debug build aborts on this by design, which is the whole
    // point of the guard, so the counting behaviour can only be asserted here.
#ifdef NDEBUG
    FakeBackend be;
    EXPECT_EQ(steady_state_allocations(), 0u);

    set_alloc_phase(AllocPhase::Serving);
    auto r = be.acquire(4096, 256, RegionTag::ForwardScratch);
    ASSERT_TRUE(r) << "the guard counts, it does not refuse — a server must keep running";

    EXPECT_EQ(steady_state_allocations(), 1u);
    EXPECT_EQ(steady_state_allocations(RegionTag::ForwardScratch), 1u);
    EXPECT_EQ(steady_state_allocations(RegionTag::KvBlockPool), 0u);
#else
    GTEST_SKIP() << "debug build aborts on a serving-phase acquisition by design";
#endif
}

TEST_F(PhaseFixture, LoadingAndPlanningPhasesAreNotCounted) {
    FakeBackend be;
    set_alloc_phase(AllocPhase::Loading);
    { auto a = be.acquire(4096, 256, RegionTag::ModelResident); ASSERT_TRUE(a); }
    set_alloc_phase(AllocPhase::Planning);
    { auto b = be.acquire(4096, 256, RegionTag::KvBlockPool); ASSERT_TRUE(b); }
    EXPECT_EQ(steady_state_allocations(), 0u);
}

TEST_F(PhaseFixture, PhaseScopeRestoresThePreviousPhase) {
    set_alloc_phase(AllocPhase::Serving);
    {
        AllocPhaseScope scope(AllocPhase::Loading, "model_swap");
        EXPECT_EQ(alloc_phase(), AllocPhase::Loading);
    }
    EXPECT_EQ(alloc_phase(), AllocPhase::Serving);
}

TEST_F(PhaseFixture, JournalRecordsThePhaseOfEveryEvent) {
    FakeBackend be;
    { auto a = be.acquire(256, 256, RegionTag::Other); ASSERT_TRUE(a); }
    set_alloc_phase(AllocPhase::Planning);
    { auto b = be.acquire(256, 256, RegionTag::Other); ASSERT_TRUE(b); }

    // The I2 assertion a soak makes: no acquire event carries phase == Serving.
    for (const auto& e : be.journal()) {
        if (e.op == AllocEvent::Op::Acquire) {
            // Braced: the GTest macro expands to an if/else of its own, so an
            // unbraced branch here is a dangling-else the compiler warns about.
            EXPECT_NE(e.phase, AllocPhase::Serving);
        }
    }
}

// ── Capacity, failure injection, rollback ─────────────────────────────

TEST_F(PhaseFixture, CapacityIsEnforcedAndTyped) {
    FakeBackend be(4 * kMiB);
    auto ok = be.acquire(3 * kMiB, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(ok);

    auto over = be.acquire(2 * kMiB, 256, RegionTag::KvBlockPool);
    EXPECT_FALSE(over);
    EXPECT_EQ(over.error, MemError::OutOfMemory) << "exhaustion is a value, not a crash (I6)";
    EXPECT_FALSE(over.region.valid());

    // Freeing makes room again — no fragmentation bookkeeping to get wrong.
    ok.region.reset();
    auto retry = be.acquire(2 * kMiB, 256, RegionTag::KvBlockPool);
    EXPECT_TRUE(retry);
}

TEST_F(PhaseFixture, InjectedFailureLeavesNoPartialState) {
    FakeBackend be;
    be.fail_acquisition(3);

    std::vector<Region> held;
    MemError last = MemError::Ok;
    for (int i = 0; i < 4; ++i) {
        auto r = be.acquire(1024, 256, RegionTag::EnginePersistent);
        if (!r) {
            last = r.error;
            break;
        }
        held.push_back(std::move(r.region));
    }
    EXPECT_EQ(last, MemError::OutOfMemory);
    EXPECT_EQ(held.size(), 2u) << "the 3rd acquisition must fail";
    EXPECT_EQ(be.journal_live_bytes(), be.stats().live_bytes)
        << "a failed acquisition must not have booked anything";

    held.clear();
    EXPECT_EQ(be.stats().live_bytes, 0u);
}

TEST_F(PhaseFixture, ZeroBytesAndBadAlignmentAreRejected) {
    FakeBackend be;
    EXPECT_EQ(be.acquire(0, 256, RegionTag::Other).error, MemError::InvalidArgument);
    EXPECT_EQ(be.acquire(1024, 0, RegionTag::Other).error, MemError::InvalidArgument);
    EXPECT_EQ(be.acquire(1024, 300, RegionTag::Other).error, MemError::InvalidArgument)
        << "alignment must be a power of two";
}

TEST_F(PhaseFixture, AlignmentIsHonoured) {
    FakeBackend be;
    for (size_t align : {size_t{256}, size_t{512}, size_t{4096}}) {
        auto r = be.acquire(1024, align, RegionTag::Other);
        ASSERT_TRUE(r) << "align=" << align;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(r.region.base()) % align, 0u) << "align=" << align;
    }
}

// ── V9: address stability under growth (the I3 property VMM must satisfy) ──

TEST_F(PhaseFixture, GrowableRegionKeepsItsBaseAcrossCommitAndDecommit) {
    FakeBackend be;
    auto r = be.acquire_growable(16 * kMiB, 1 * kMiB, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(r);
    void* const base = r.region.base();
    EXPECT_EQ(r.region.reserved(), 16 * kMiB);
    EXPECT_EQ(r.region.committed(), 1 * kMiB);

    for (size_t mib : {2u, 4u, 8u, 16u, 8u, 2u, 12u}) {
        ASSERT_EQ(be.commit(r.region, mib * kMiB), MemError::Ok) << "target=" << mib << " MiB";
        EXPECT_EQ(r.region.base(), base)
            << "growth must not move the base — a captured graph bakes this pointer (I3)";
        EXPECT_EQ(r.region.committed(), mib * kMiB);
    }
}

TEST_F(PhaseFixture, OnlyCommittedBytesCountAgainstCapacity) {
    FakeBackend be(4 * kMiB);
    // Reserving 64 MiB on a 4 MiB budget is fine: a reservation is address
    // space, not memory. This is exactly what lets the KV pool stop guessing.
    auto r = be.acquire_growable(64 * kMiB, 1 * kMiB, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(r);
    EXPECT_EQ(be.stats().live_bytes, 1 * kMiB);

    EXPECT_EQ(be.commit(r.region, 3 * kMiB), MemError::Ok);
    EXPECT_EQ(be.commit(r.region, 8 * kMiB), MemError::OutOfMemory)
        << "committing past the budget must fail cleanly";
    EXPECT_EQ(r.region.committed(), 3 * kMiB) << "a failed commit must not change the region";
}

TEST_F(PhaseFixture, CommitBeyondTheReservationIsRejected) {
    FakeBackend be;
    auto r = be.acquire_growable(4 * kMiB, 1 * kMiB, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(r);
    EXPECT_EQ(be.commit(r.region, 8 * kMiB), MemError::InvalidArgument);
}

TEST_F(PhaseFixture, DecommittedTailIsPoisoned) {
    FakeBackend be;
    auto r = be.acquire_growable(4 * kMiB, 4 * kMiB, 256, RegionTag::KvBlockPool);
    ASSERT_TRUE(r);
    std::memset(r.region.base(), 0x22, 4 * kMiB);

    ASSERT_EQ(be.commit(r.region, 1 * kMiB), MemError::Ok);
    const auto* tail = static_cast<const char*>(r.region.base()) + 1 * kMiB;
    EXPECT_TRUE(FakeBackend::is_poisoned(tail, 3 * kMiB))
        << "decommitted pages are unmapped on device; the fake must model that";
}

TEST_F(PhaseFixture, NonGrowableBackendSaysSoInsteadOfPretending) {
    FakeBackend be(0, /*growable=*/false);
    auto r = be.acquire_growable(4 * kMiB, 1 * kMiB, 256, RegionTag::KvBlockPool);
    EXPECT_FALSE(r);
    EXPECT_EQ(r.error, MemError::NotGrowable);
}

// ── Reporting surface (I7) ────────────────────────────────────────────

TEST_F(PhaseFixture, PeakIsTrackedSeparatelyFromLive) {
    FakeBackend be;
    {
        auto a = be.acquire(2 * kMiB, 256, RegionTag::ModelResident);
        auto b = be.acquire(3 * kMiB, 256, RegionTag::KvBlockPool);
        ASSERT_TRUE(a);
        ASSERT_TRUE(b);
        EXPECT_EQ(be.stats().live_bytes, 5 * kMiB);
    }
    EXPECT_EQ(be.stats().live_bytes, 0u);
    EXPECT_EQ(be.stats().peak_bytes, 5 * kMiB)
        << "\"reserved 18 GB, 4.2 GB live\" is the statement that has to be possible";
}

TEST_F(PhaseFixture, TagNamesAndErrorNamesAreTotal) {
    // A missing case here would print "other"/"unknown" in --mem-report and
    // silently mis-attribute memory, which is the exact failure I7 is about.
    for (int i = 0; i <= static_cast<int>(RegionTag::Other); ++i) {
        const char* n = region_tag_name(static_cast<RegionTag>(i));
        ASSERT_NE(n, nullptr);
        EXPECT_NE(std::string(n), "") << "tag " << i;
    }
    EXPECT_STREQ(mem_error_name(MemError::Ok), "ok");
    EXPECT_STREQ(mem_error_name(MemError::BudgetExceeded), "budget_exceeded");
    EXPECT_STREQ(mem_error_name(MemError::NotGrowable), "not_growable");
}
