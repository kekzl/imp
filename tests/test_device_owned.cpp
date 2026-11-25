// DeviceOwned (memory/device_owned.h, invariant I5): one release per allocation, through the
// family that produced it. CPU lane on purpose: freed exactly once, never after a move and
// never through another family is an ownership property, so the driver is replaced by the
// release hook and the buffers are adopted host addresses.

#include <gtest/gtest.h>

#include "exec/workspace.h"
#include "memory/device_owned.h"
#include "memory/vram_allocator.h"

#include <type_traits>
#include <utility>
#include <vector>

using namespace imp;

namespace {

struct Release {
    void* ptr;
    DeviceFamily family;
    VRAMAllocator* alloc;
    cudaStream_t stream;
};

std::vector<Release>& log() {
    static std::vector<Release> l;
    return l;
}

void record(void* ptr, DeviceFamily family, VRAMAllocator* alloc, cudaStream_t stream) {
    log().push_back({ptr, family, alloc, stream});
}

class DeviceOwnedTest : public ::testing::Test {
protected:
    void SetUp() override {
        log().clear();
        prev_ = set_device_owned_release_hook(&record);
    }
    void TearDown() override { set_device_owned_release_hook(prev_); }

    // Distinct addresses that are never dereferenced.
    float slots_[4] = {};
    float* a_ = &slots_[0];
    float* b_ = &slots_[1];
    VRAMAllocator va_;  // identity only: adopt() never calls it, the hook takes the release
    cudaStream_t stream_ = reinterpret_cast<cudaStream_t>(&slots_[3]);

private:
    DeviceOwnedReleaseHook prev_ = nullptr;
};

static_assert(!std::is_copy_constructible_v<DeviceOwned<float>>);
static_assert(!std::is_copy_assignable_v<DeviceOwned<float>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceOwned<void>>);
static_assert(std::is_nothrow_move_assignable_v<DeviceOwned<void>>);

}  // namespace

TEST_F(DeviceOwnedTest, EmptyHandleReleasesNothing) {
    {
        DeviceOwned<float> h;
        EXPECT_FALSE(h);
        EXPECT_EQ(h.get(), nullptr);
        EXPECT_EQ(h.bytes(), 0u);
        EXPECT_EQ(h.family(), DeviceFamily::None);
        h.reset();
    }
    EXPECT_TRUE(log().empty());
}

TEST_F(DeviceOwnedTest, ZeroBytesAllocatesNothingInEveryFamily) {
    EXPECT_FALSE(DeviceOwned<void>::allocate(&va_, 0, "t"));
    EXPECT_FALSE(DeviceOwned<void>::allocate(nullptr, 0, "t"));
    EXPECT_FALSE(DeviceOwned<void>::allocate_plain(0));
    EXPECT_FALSE(DeviceOwned<void>::allocate_async(stream_, 0));
    EXPECT_TRUE(log().empty());
}

TEST_F(DeviceOwnedTest, AdoptWithAllocatorReleasesThroughThatAllocatorOnce) {
    {
        DeviceOwned<float> h;
        h.adopt(a_, 64, &va_);
        EXPECT_TRUE(h);
        EXPECT_EQ(h.get(), a_);
        EXPECT_EQ(h.bytes(), 64u);
        EXPECT_EQ(h.family(), DeviceFamily::Vram);
    }
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, a_);
    EXPECT_EQ(log()[0].family, DeviceFamily::Vram);
    EXPECT_EQ(log()[0].alloc, &va_);
}

TEST_F(DeviceOwnedTest, AdoptWithoutAllocatorIsThePlainFamily) {
    DeviceOwned<float> h;
    h.adopt(a_, 8, nullptr);
    EXPECT_EQ(h.family(), DeviceFamily::Plain);
    h.reset();
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].family, DeviceFamily::Plain);
    EXPECT_EQ(log()[0].alloc, nullptr);
}

TEST_F(DeviceOwnedTest, ResetIsIdempotentAndEmptiesTheHandle) {
    DeviceOwned<float> h;
    h.adopt(a_, 8, nullptr);
    h.reset();
    h.reset();
    EXPECT_FALSE(h);
    EXPECT_EQ(h.bytes(), 0u);
    EXPECT_EQ(h.family(), DeviceFamily::None);
    EXPECT_EQ(log().size(), 1u);
}

TEST_F(DeviceOwnedTest, AsyncReleasesOnTheStoredStream) {
    DeviceOwned<float> h;
    h.adopt_async(a_, 8, stream_);
    EXPECT_EQ(h.family(), DeviceFamily::Async);
    h.reset();
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].family, DeviceFamily::Async);
    EXPECT_EQ(log()[0].stream, stream_);
}

TEST_F(DeviceOwnedTest, ReleaseStreamCanBeRenamedForAsyncOnly) {
    DeviceOwned<float> h;
    h.adopt_async(a_, 8, stream_);
    h.set_release_stream(nullptr);
    h.reset();
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].stream, nullptr);

    DeviceOwned<float> p;
    p.adopt(b_, 8, nullptr);
    p.set_release_stream(stream_);
    EXPECT_EQ(p.family(), DeviceFamily::Plain);
    p.reset();
    ASSERT_EQ(log().size(), 2u);
    EXPECT_EQ(log()[1].family, DeviceFamily::Plain);
    EXPECT_EQ(log()[1].stream, nullptr);
}

TEST_F(DeviceOwnedTest, MoveConstructionTransfersTheSingleRelease) {
    {
        DeviceOwned<float> src;
        src.adopt_async(a_, 16, stream_);
        DeviceOwned<float> dst(std::move(src));
        EXPECT_FALSE(src);  // NOLINT(bugprone-use-after-move): the moved-from state is the contract
        EXPECT_EQ(src.family(), DeviceFamily::None);
        EXPECT_EQ(dst.get(), a_);
        EXPECT_EQ(dst.bytes(), 16u);
        EXPECT_EQ(dst.family(), DeviceFamily::Async);
        EXPECT_TRUE(log().empty());
    }
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].stream, stream_);
}

TEST_F(DeviceOwnedTest, MoveAssignmentReleasesTheOldBufferThroughItsOwnFamily) {
    DeviceOwned<float> dst;
    dst.adopt(a_, 8, &va_);
    DeviceOwned<float> src;
    src.adopt(b_, 8, nullptr);
    dst = std::move(src);
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, a_);
    EXPECT_EQ(log()[0].family, DeviceFamily::Vram);
    EXPECT_EQ(log()[0].alloc, &va_);
    EXPECT_EQ(dst.get(), b_);
    dst.reset();
    ASSERT_EQ(log().size(), 2u);
    EXPECT_EQ(log()[1].ptr, b_);
    EXPECT_EQ(log()[1].family, DeviceFamily::Plain);
}

TEST_F(DeviceOwnedTest, AdoptOverALiveBufferReleasesItFirst) {
    DeviceOwned<float> h;
    h.adopt(a_, 8, &va_);
    h.adopt_async(b_, 8, stream_);
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, a_);
    EXPECT_EQ(log()[0].family, DeviceFamily::Vram);
    EXPECT_EQ(h.get(), b_);
    h.reset();
    EXPECT_EQ(log().size(), 2u);
}

TEST_F(DeviceOwnedTest, AdoptingNullLeavesTheHandleEmpty) {
    DeviceOwned<float> h;
    h.adopt(nullptr, 128, &va_);
    EXPECT_FALSE(h);
    EXPECT_EQ(h.bytes(), 0u);
    EXPECT_EQ(h.family(), DeviceFamily::None);
    h.adopt_async(nullptr, 128, stream_);
    EXPECT_FALSE(h);
    h.reset();
    EXPECT_TRUE(log().empty());
}

// release() is the hand-off to a non-DeviceOwned owner (weight cache): no release, ever.
TEST_F(DeviceOwnedTest, ReleaseHandsThePointerOverWithoutFreeing) {
    {
        DeviceOwned<float> h;
        h.adopt(a_, 8, &va_);
        EXPECT_EQ(h.release(), a_);
        EXPECT_FALSE(h);
        EXPECT_EQ(h.bytes(), 0u);
        EXPECT_EQ(h.family(), DeviceFamily::None);
        EXPECT_EQ(h.release(), nullptr);
        h.reset();
    }
    EXPECT_TRUE(log().empty());
}

// An owner that is not released (error path before the hand-off) still frees once.
TEST_F(DeviceOwnedTest, UnreleasedOwnerFreesAtScopeEndReleasedOneDoesNot) {
    {
        DeviceOwned<float> kept;
        DeviceOwned<float> handed;
        kept.adopt_async(a_, 8, stream_);
        handed.adopt(b_, 8, nullptr);
        EXPECT_EQ(handed.release(), b_);
    }
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, a_);
    EXPECT_EQ(log()[0].family, DeviceFamily::Async);
    EXPECT_EQ(log()[0].stream, stream_);
}

// A struct of owners reset by whole-struct assignment (MoEWorkspace::free): every member is
// released once, a member added later needs no line in the release.
TEST_F(DeviceOwnedTest, AssigningAFreshStructReleasesEveryMember) {
    struct Bag {
        DeviceOwned<float> x;
        DeviceOwned<float> y;
        int n = 0;
    };
    Bag bag;
    bag.x.adopt(a_, 8, &va_);
    bag.y.adopt(b_, 8, nullptr);
    bag.n = 3;
    bag = Bag{};
    EXPECT_EQ(log().size(), 2u);
    EXPECT_FALSE(bag.x);
    EXPECT_FALSE(bag.y);
    EXPECT_EQ(bag.n, 0);
}

// grow_keeping_old (exec/workspace.h, audit C-11): a failed grow leaves the shared workspace
// as it was, never null with a stale token count; a grow takes the new buffer first.
TEST_F(DeviceOwnedTest, FailedGrowKeepsTheOldBufferAndSize) {
    DeviceOwned<void> buf;
    buf.adopt(a_, 64, &va_);
    size_t held = 64;
    int takes = 0;
    EXPECT_FALSE(grow_keeping_old(buf, held, 128, [&](size_t) {
        ++takes;
        return DeviceOwned<void>{};
    }));
    EXPECT_EQ(takes, 1);
    EXPECT_EQ(buf.get(), static_cast<void*>(a_));
    EXPECT_EQ(held, 64u);
    EXPECT_TRUE(log().empty());
}

TEST_F(DeviceOwnedTest, GrowTakesTheNewBufferBeforeReleasingTheOld) {
    DeviceOwned<void> buf;
    buf.adopt(a_, 64, &va_);
    size_t held = 64;
    size_t released_before_take = 99;
    EXPECT_TRUE(grow_keeping_old(buf, held, 128, [&](size_t bytes) {
        released_before_take = log().size();
        DeviceOwned<void> fresh;
        fresh.adopt(b_, bytes, &va_);
        return fresh;
    }));
    EXPECT_EQ(released_before_take, 0u);
    EXPECT_EQ(buf.get(), static_cast<void*>(b_));
    EXPECT_EQ(held, 128u);
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, static_cast<void*>(a_));
}

TEST_F(DeviceOwnedTest, GrowWithinTheHeldSizeTakesNothing) {
    DeviceOwned<void> buf;
    buf.adopt(a_, 64, &va_);
    size_t held = 64;
    EXPECT_TRUE(grow_keeping_old(buf, held, 32, [](size_t) -> DeviceOwned<void> {
        ADD_FAILURE() << "no take below the held size";
        return {};
    }));
    EXPECT_EQ(buf.get(), static_cast<void*>(a_));
    EXPECT_EQ(held, 64u);
}
