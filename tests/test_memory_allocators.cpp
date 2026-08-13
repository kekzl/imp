// Tier allocators: arena (T1/T2), block pool (T3), scratch stack (T4).
// docs/internals/MEMORY.md A2/A3.3/A3.4/A6, invariants V3-V6.
//
// CPU-only: every allocator runs over FakeBackend, so the whole stack is
// exercised in the lane CI actually runs (`ctest -L unit`).

#include <gtest/gtest.h>

#include "memory/arena.h"
#include "memory/block_pool.h"
#include "memory/fake_backend.h"
#include "memory/scratch_stack.h"

#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace imp;

namespace {
constexpr size_t kMiB = 1024 * 1024;
}

// ── Arena (T1/T2) ─────────────────────────────────────────────────────

TEST(Arena, TakesAlignedAndTracksUsage) {
    FakeBackend be;
    ArenaAllocator arena;
    ASSERT_EQ(arena.open(be, 1 * kMiB, RegionTag::EnginePersistent), MemError::Ok);
    EXPECT_EQ(arena.capacity(), 1 * kMiB);

    auto a = arena.take<float>(1000);
    ASSERT_TRUE(a);
    EXPECT_EQ(a.size(), 1000u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(a.data()) % 256, 0u);

    auto b = arena.take<float>(1000);
    ASSERT_TRUE(b);
    EXPECT_NE(a.data(), b.data()) << "two takes must not overlap";
    EXPECT_GE(reinterpret_cast<char*>(b.data()) - reinterpret_cast<char*>(a.data()),
              static_cast<ptrdiff_t>(1000 * sizeof(float)));

    EXPECT_GE(arena.used(), 2 * 1000 * sizeof(float));
    EXPECT_EQ(arena.used() + arena.remaining(), arena.capacity());
}

// The property that closes AUDIT B13, and the reason the grow-on-demand
// scratches in compute/ could move here (A7 step 8). Those pointers are kernel
// PARAMETERS baked into instantiated CUDA graphs. Their old cudaFree+cudaMalloc
// grow made a replayed graph read a freed address; an arena grow must leave the
// previous slice both valid and untouched, because the graph is still using it
// at the size it was captured for.
TEST(Arena, GrowingATenantLeavesThePreviousSliceValidAndIntact) {
    FakeBackend be;
    ArenaAllocator arena;
    ASSERT_EQ(arena.open(be, 1 * kMiB, RegionTag::EnginePersistent), MemError::Ok);

    auto first = arena.take<uint8_t>(4096);
    ASSERT_TRUE(first);
    std::memset(first.data(), 0xA5, first.size());

    // The "grow": the tenant needs more, so it takes a second, larger slice and
    // drops its reference to the first. Nothing frees.
    auto grown = arena.take<uint8_t>(16384);
    ASSERT_TRUE(grown);
    std::memset(grown.data(), 0x5A, grown.size());

    const auto* f = reinterpret_cast<const uint8_t*>(first.data());
    const auto* g = reinterpret_cast<const uint8_t*>(grown.data());
    EXPECT_TRUE(g + grown.size() <= f || f + first.size() <= g)
        << "the grown slice overlaps the one a captured graph may still be reading";
    for (size_t i = 0; i < first.size(); ++i)
        ASSERT_EQ(f[i], 0xA5) << "byte " << i << " of the pre-grow slice was clobbered";

    // And the guard the tenants use to notice a model swap: close() bumps the
    // generation, so a cached pointer is recognised as stale rather than reused
    // over a released region.
    const uint64_t before = arena.generation();
    arena.close();
    ASSERT_EQ(arena.open(be, 1 * kMiB, RegionTag::EnginePersistent), MemError::Ok);
    EXPECT_NE(arena.generation(), before);
}

TEST(Arena, ExhaustionReturnsAnEmptySpanInsteadOfReachingForTheDriver) {
    FakeBackend be;
    ArenaAllocator arena;
    ASSERT_EQ(arena.open(be, 4096, RegionTag::ModelResident), MemError::Ok);

    ASSERT_TRUE(arena.take_bytes(4000));
    auto over = arena.take_bytes(4000);
    EXPECT_FALSE(over) << "exhaustion is a value the caller handles (I6)";
    EXPECT_EQ(be.stats().acquire_count, 1u) << "the arena must not go back to the backend";
}

// V3: an arena reset frees wholesale.
TEST(Arena, ResetFreesWholesaleAndBumpsTheGeneration) {
    FakeBackend be;
    ArenaAllocator arena;
    ASSERT_EQ(arena.open(be, 64 * 1024, RegionTag::ModelResident), MemError::Ok);

    const uint64_t gen0 = arena.generation();
    for (int i = 0; i < 50; ++i)
        ASSERT_TRUE(arena.take_bytes(512)) << "i=" << i;
    const size_t peak = arena.used();
    ASSERT_GT(peak, 0u);

    arena.reset();
    EXPECT_EQ(arena.used(), 0u) << "V3: nothing is attributable to the arena after reset";
    EXPECT_EQ(arena.high_water(), peak) << "the planner's number survives the reset";
    EXPECT_NE(arena.generation(), gen0)
        << "consumers caching derived pointers must be able to notice";
    EXPECT_EQ(be.stats().live_bytes, 64u * 1024) << "reset rewinds, it does not release";
}

TEST(Arena, CloseReleasesTheRegionAndIsIdempotent) {
    FakeBackend be;
    {
        ArenaAllocator arena;
        ASSERT_EQ(arena.open(be, 8192, RegionTag::EnginePersistent), MemError::Ok);
        EXPECT_EQ(be.stats().live_bytes, 8192u);
        arena.close();
        EXPECT_EQ(be.stats().live_bytes, 0u);
        arena.close();
    }
    EXPECT_EQ(be.stats().release_count, 1u) << "no double release";
}

TEST(Arena, OpenTwiceIsRejected) {
    FakeBackend be;
    ArenaAllocator arena;
    ASSERT_EQ(arena.open(be, 4096, RegionTag::Other), MemError::Ok);
    EXPECT_EQ(arena.open(be, 4096, RegionTag::Other), MemError::InvalidArgument);
}

TEST(Arena, PropagatesTheBackendsTypedFailure) {
    FakeBackend be(1 * kMiB);
    ArenaAllocator arena;
    EXPECT_EQ(arena.open(be, 8 * kMiB, RegionTag::ModelResident), MemError::OutOfMemory);
    EXPECT_FALSE(arena.is_open());
}

// ── BlockPool + BlockRef (T3) ─────────────────────────────────────────

TEST(BlockPool, AcquireAndReleaseAreBalanced) {
    FakeBackend be;
    BlockPool pool;
    ASSERT_EQ(pool.open(be, 1024, 8, RegionTag::KvBlockPool), MemError::Ok);
    EXPECT_EQ(pool.num_blocks(), 8);
    EXPECT_EQ(pool.free_count(), 8);

    {
        auto r = pool.acquire();
        ASSERT_TRUE(r);
        EXPECT_EQ(pool.free_count(), 7);
        EXPECT_EQ(pool.live_blocks(), 1);
        EXPECT_EQ(pool.total_refs(), 1u);
    }
    EXPECT_EQ(pool.free_count(), 8);
    EXPECT_EQ(pool.total_refs(), 0u);
}

TEST(BlockPool, BlocksAreDistinctNonOverlappingAndStable) {
    FakeBackend be;
    BlockPool pool;
    ASSERT_EQ(pool.open(be, 1024, 4, RegionTag::KvBlockPool), MemError::Ok);

    std::vector<BlockRef> refs;
    std::vector<void*> bases;
    for (int i = 0; i < 4; ++i) {
        refs.push_back(pool.acquire());
        ASSERT_TRUE(refs.back()) << "i=" << i;
        auto b = pool.block(refs.back().id());
        ASSERT_EQ(b.size(), 1024u);
        bases.push_back(b.data());
    }
    std::sort(bases.begin(), bases.end());
    EXPECT_EQ(std::unique(bases.begin(), bases.end()), bases.end()) << "block ids must not alias";

    // Address stability (I3): the same id resolves to the same pointer for the
    // pool's lifetime, which is what lets a captured graph bake it.
    for (const auto& r : refs)
        EXPECT_EQ(pool.block(r.id()).data(), pool.block(r.id()).data());
}

TEST(BlockPool, ExhaustionYieldsAnInvalidRefNotACrash) {
    FakeBackend be;
    BlockPool pool;
    ASSERT_EQ(pool.open(be, 256, 2, RegionTag::KvBlockPool), MemError::Ok);
    auto a = pool.acquire();
    auto b = pool.acquire();
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    auto c = pool.acquire();
    EXPECT_FALSE(c) << "exhaustion is admission control's problem, not a fault (I6)";
    EXPECT_EQ(pool.free_count(), 0);
}

// The three-referent case from A5.1: sequence table + prefix cache + pin set.
TEST(BlockPool, ThreeReferentsAndTheBlockSurvivesUntilTheLastOneDrops) {
    FakeBackend be;
    BlockPool pool;
    ASSERT_EQ(pool.open(be, 512, 4, RegionTag::KvBlockPool), MemError::Ok);

    auto seq = pool.acquire();  // the owning sequence
    ASSERT_TRUE(seq);
    const int id = seq.id();

    auto prefix_cache = seq.share();  // content-addressed reuse
    auto pin = seq.share();           // agentic prefix pin
    EXPECT_EQ(pool.total_refs(), 3u);
    EXPECT_EQ(pool.free_count(), 3);

    // The sequence finishes (or is cancelled — same path). The block must NOT
    // return to the free list: the prefix cache and the pin still want it.
    seq.reset();
    EXPECT_EQ(pool.total_refs(), 2u);
    EXPECT_EQ(pool.free_count(), 3) << "still held by the cache and the pin";

    prefix_cache.reset();  // evicted from the hash table
    EXPECT_EQ(pool.free_count(), 3) << "still pinned";

    pin.reset();  // unpinned
    EXPECT_EQ(pool.free_count(), 4) << "last referent gone -> reclaimed";
    EXPECT_EQ(pool.total_refs(), 0u);

    auto again = pool.acquire();
    ASSERT_TRUE(again);
    EXPECT_EQ(again.id(), id) << "the id is reusable once it is genuinely free";
}

TEST(BlockRef, MoveTransfersTheRefWithoutChangingTheCount) {
    FakeBackend be;
    BlockPool pool;
    ASSERT_EQ(pool.open(be, 256, 2, RegionTag::KvBlockPool), MemError::Ok);

    auto a = pool.acquire();
    ASSERT_TRUE(a);
    const int id = a.id();

    BlockRef b = std::move(a);
    EXPECT_EQ(pool.total_refs(), 1u) << "a move is not a new referent";
    EXPECT_FALSE(a.valid());
    EXPECT_EQ(b.id(), id);

    b.reset();
    EXPECT_EQ(pool.total_refs(), 0u);
}

// V5: refcounts balance no matter where a request dies.
TEST(BlockPool, RefcountBalancesWhenAnExceptionUnwindsMidSequence) {
    FakeBackend be;
    BlockPool pool;
    ASSERT_EQ(pool.open(be, 256, 16, RegionTag::KvBlockPool), MemError::Ok);

    for (int fail_after = 0; fail_after < 8; ++fail_after) {
        try {
            std::vector<BlockRef> seq;
            std::vector<BlockRef> cache;
            for (int i = 0; i < 8; ++i) {
                seq.push_back(pool.acquire());
                if (!seq.back())
                    break;
                cache.push_back(seq.back().share());
                if (i == fail_after)
                    throw std::runtime_error("cancelled mid-prefill");
            }
        } catch (const std::exception&) {
            // The unwind is the whole mechanism: no free_sequence() to forget.
        }
        EXPECT_EQ(pool.total_refs(), 0u) << "fail_after=" << fail_after;
        EXPECT_EQ(pool.free_count(), 16) << "fail_after=" << fail_after;
    }
}

// V4: conservation under a randomised alloc/share/free workload.
TEST(BlockPool, ConservationUnderRandomisedChurn) {
    FakeBackend be;
    BlockPool pool;
    constexpr int kBlocks = 32;
    ASSERT_EQ(pool.open(be, 256, kBlocks, RegionTag::KvBlockPool), MemError::Ok);

    std::mt19937 rng(12345);
    std::vector<BlockRef> held;
    for (int step = 0; step < 5000; ++step) {
        const int op = static_cast<int>(rng() % 3);
        if (op == 0) {
            auto r = pool.acquire();
            if (r)
                held.push_back(std::move(r));
        } else if (op == 1 && !held.empty()) {
            held.push_back(held[rng() % held.size()].share());
        } else if (!held.empty()) {
            const size_t i = rng() % held.size();
            held.erase(held.begin() + static_cast<ptrdiff_t>(i));
        }
        ASSERT_EQ(pool.free_count() + pool.live_blocks(), kBlocks) << "step=" << step;
        ASSERT_GE(pool.total_refs(), static_cast<uint64_t>(pool.live_blocks())) << "step=" << step;
        ASSERT_EQ(pool.total_refs(), held.size()) << "step=" << step;
    }
    held.clear();
    EXPECT_EQ(pool.free_count(), kBlocks);
    EXPECT_EQ(pool.total_refs(), 0u);
}

TEST(BlockPool, OutOfRangeBlockIdYieldsAnEmptySpan) {
    FakeBackend be;
    BlockPool pool;
    ASSERT_EQ(pool.open(be, 256, 2, RegionTag::KvBlockPool), MemError::Ok);
    EXPECT_FALSE(pool.block(-1));
    EXPECT_FALSE(pool.block(2));
}

// ── ScratchStack (T4) ─────────────────────────────────────────────────

TEST(ScratchStack, MarkRewindsEverythingTakenAfterIt) {
    FakeBackend be;
    ScratchStack stack;
    ASSERT_EQ(stack.open(be, 64 * 1024, RegionTag::ForwardScratch), MemError::Ok);

    EXPECT_EQ(stack.used(), 0u);
    {
        auto m = stack.mark();
        ASSERT_TRUE(stack.take_bytes(1024));
        ASSERT_TRUE(stack.take_bytes(2048));
        EXPECT_GE(stack.used(), 3072u);
    }
    EXPECT_EQ(stack.used(), 0u) << "the mark unwinds the whole forward pass";
}

TEST(ScratchStack, NestedMarksRewindInReverseOrder) {
    FakeBackend be;
    ScratchStack stack;
    ASSERT_EQ(stack.open(be, 64 * 1024, RegionTag::ForwardScratch), MemError::Ok);

    auto outer = stack.mark();
    ASSERT_TRUE(stack.take_bytes(1024));
    const size_t after_outer = stack.used();
    {
        auto inner = stack.mark();
        ASSERT_TRUE(stack.take_bytes(4096));
        EXPECT_GT(stack.used(), after_outer);
    }
    EXPECT_EQ(stack.used(), after_outer) << "the inner mark returns only its own takes";
}

TEST(ScratchStack, RewindsOnTheExceptionPathToo) {
    FakeBackend be;
    ScratchStack stack;
    ASSERT_EQ(stack.open(be, 16 * 1024, RegionTag::ForwardScratch), MemError::Ok);

    try {
        auto m = stack.mark();
        ASSERT_TRUE(stack.take_bytes(8192));
        throw std::runtime_error("kernel dispatch failed");
    } catch (const std::exception&) {
    }
    EXPECT_EQ(stack.used(), 0u) << "a LIFO stack cannot leak — that is why it is a stack";
}

TEST(ScratchStack, ReportsItsHighWaterAndItsExhaustions) {
    FakeBackend be;
    ScratchStack stack;
    ASSERT_EQ(stack.open(be, 8192, RegionTag::ForwardScratch), MemError::Ok);

    for (int i = 0; i < 4; ++i) {
        auto m = stack.mark();
        ASSERT_TRUE(stack.take_bytes(static_cast<size_t>(1 + i) * 1024));
    }
    EXPECT_GE(stack.high_water(), 4u * 1024);
    EXPECT_EQ(stack.used(), 0u);
    EXPECT_EQ(stack.exhaustion_count(), 0u);

    {
        auto m = stack.mark();
        EXPECT_FALSE(stack.take_bytes(64 * 1024));
    }
    EXPECT_EQ(stack.exhaustion_count(), 1u)
        << "an under-provisioned plan must be visible, not silently absorbed";
}

TEST(ScratchStack, ReusesTheSameAddressesAcrossPasses) {
    FakeBackend be;
    ScratchStack stack;
    ASSERT_EQ(stack.open(be, 16 * 1024, RegionTag::ForwardScratch), MemError::Ok);

    void* first = nullptr;
    for (int pass = 0; pass < 3; ++pass) {
        auto m = stack.mark();
        auto s = stack.take_bytes(2048);
        ASSERT_TRUE(s);
        if (pass == 0)
            first = s.data();
        else
            EXPECT_EQ(s.data(), first)
                << "stable per slot: a captured graph replays against the same address (I3)";
    }
}

TEST(ScratchStack, TypedTakeIsAligned) {
    FakeBackend be;
    ScratchStack stack;
    ASSERT_EQ(stack.open(be, 64 * 1024, RegionTag::ForwardScratch), MemError::Ok);
    auto m = stack.mark();
    auto s = stack.take<int32_t>(100);
    ASSERT_TRUE(s);
    EXPECT_EQ(s.size(), 100u);
    EXPECT_EQ(s.size_bytes(), 400u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(s.data()) % 256, 0u);
}

// ── Spans ─────────────────────────────────────────────────────────────

TEST(Span, StableWidensToDeviceSpanAndSubspansInheritTheGuarantee) {
    FakeBackend be;
    ArenaAllocator arena;
    ASSERT_EQ(arena.open(be, 4096, RegionTag::EnginePersistent), MemError::Ok);

    StableSpan<int32_t> s = arena.take<int32_t>(64);
    ASSERT_TRUE(s);

    DeviceSpan<int32_t> widened = s;  // implicit — dropping a guarantee is safe
    EXPECT_EQ(widened.data(), s.data());
    EXPECT_EQ(widened.size(), s.size());

    StableSpan<int32_t> tail = s.subspan(16, 16);
    EXPECT_EQ(tail.data(), s.data() + 16);
    EXPECT_EQ(tail.size(), 16u);

    // Clamped, not UB, when the caller over-asks.
    EXPECT_EQ(s.subspan(60, 999).size(), 4u);
    EXPECT_TRUE(s.subspan(999, 1).empty());
    EXPECT_EQ(s.first(8).size(), 8u);
}

// The I3 mechanism itself. There is no DeviceSpan -> StableSpan conversion and
// no public StableSpan(T*, size_t): a relocatable buffer cannot be handed to a
// graph-capturable kernel wrapper. Compile-time properties, asserted here so a
// future refactor that adds an escape hatch fails this test rather than
// silently reopening the hole.
static_assert(!std::is_convertible_v<DeviceSpan<int>, StableSpan<int>>,
              "a relocatable view must never convert to a stability guarantee");
static_assert(std::is_convertible_v<StableSpan<int>, DeviceSpan<int>>,
              "dropping the guarantee must stay ergonomic");
static_assert(!std::is_constructible_v<StableSpan<int>, int*, size_t>,
              "StableSpan must not be constructible from a raw pointer");
