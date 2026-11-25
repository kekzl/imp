// ViewOrOwned (memory/view_or_owned.h, invariants I3 + I5): a tier view is never freed, a private
// allocation is released exactly once, and the setters report a moved address. CPU lane: views
// come from a real ArenaAllocator over a FakeBackend, releases go to the DeviceOwned hook.

#include <gtest/gtest.h>

#include "memory/arena.h"
#include "memory/fake_backend.h"
#include "memory/leaked_static.h"
#include "memory/view_or_owned.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

using namespace imp;

namespace {

std::vector<void*>& released() {
    static std::vector<void*> l;
    return l;
}

void record(void* ptr, DeviceFamily, VRAMAllocator*, cudaStream_t) { released().push_back(ptr); }

int count_of(void* p) { return static_cast<int>(std::count(released().begin(), released().end(), p)); }

static_assert(!std::is_copy_constructible_v<ViewOrOwned<void>>);
static_assert(!std::is_copy_assignable_v<ViewOrOwned<void>>);
static_assert(std::is_nothrow_move_constructible_v<ViewOrOwned<void>>);
static_assert(std::is_nothrow_move_assignable_v<ViewOrOwned<float>>);
// The view side takes a StableSpan only: no raw pointer, no DeviceSpan (I3, no passkey in a test).
static_assert(std::is_same_v<ViewOrOwned<void>::view_type, StableSpan<std::byte>>);
static_assert(std::is_same_v<ViewOrOwned<float>::view_type, StableSpan<float>>);
static_assert(!std::is_constructible_v<ViewOrOwned<void>::view_type, std::byte*, size_t>);
static_assert(!std::is_convertible_v<DeviceSpan<std::byte>, ViewOrOwned<void>::view_type>);
static_assert(!std::is_convertible_v<void*, ViewOrOwned<void>::view_type>);

class ViewOrOwnedTest : public ::testing::Test {
protected:
    void SetUp() override {
        released().clear();
        prev_ = set_device_owned_release_hook(&record);
        ASSERT_EQ(arena_.open(be_, 4096, RegionTag::EnginePersistent), MemError::Ok);
    }
    void TearDown() override {
        arena_.close();
        set_device_owned_release_hook(prev_);
    }

    DeviceOwned<void> owned(int slot, size_t bytes = 64) {
        DeviceOwned<void> o;
        o.adopt(&slots_[slot], bytes, nullptr);
        return o;
    }

    FakeBackend be_;
    ArenaAllocator arena_;
    // Distinct addresses that are never dereferenced.
    float slots_[4] = {};

private:
    DeviceOwnedReleaseHook prev_ = nullptr;
};

TEST_F(ViewOrOwnedTest, EmptyByDefault) {
    ViewOrOwned<void> h;
    EXPECT_EQ(h.get(), nullptr);
    EXPECT_EQ(h.bytes(), 0u);
    EXPECT_FALSE(static_cast<bool>(h));
    EXPECT_FALSE(h.is_view());
    EXPECT_FALSE(h.owns());
    EXPECT_FALSE(h.reset());
    EXPECT_TRUE(released().empty());
}

TEST_F(ViewOrOwnedTest, ViewSetReplaceAndResetNeverRelease) {
    const StableSpan<std::byte> s1 = arena_.take_bytes(256);
    const StableSpan<std::byte> s2 = arena_.take_bytes(512);
    ASSERT_TRUE(s1);
    ASSERT_TRUE(s2);
    {
        ViewOrOwned<void> h;
        EXPECT_FALSE(h.set_view(s1));  // first address: nothing could have captured null
        EXPECT_EQ(h.get(), s1.data());
        EXPECT_EQ(h.bytes(), 256u);
        EXPECT_TRUE(h.is_view());
        EXPECT_FALSE(h.owns());
        EXPECT_EQ(h.view().data(), s1.data());
        EXPECT_FALSE(h.set_view(s1));  // same address
        EXPECT_TRUE(h.set_view(s2));   // moved
        EXPECT_EQ(h.get(), s2.data());
        EXPECT_EQ(h.bytes(), 512u);
        EXPECT_TRUE(h.reset());
        EXPECT_FALSE(h.reset());
        EXPECT_EQ(h.get(), nullptr);
        EXPECT_FALSE(h.set_view(s1));
    }  // destroyed while holding a view
    EXPECT_TRUE(released().empty());
}

TEST_F(ViewOrOwnedTest, OwnedReplaceReleasesThePreviousAllocationOnce) {
    {
        ViewOrOwned<void> h;
        EXPECT_FALSE(h.set_owned(owned(0, 64)));
        EXPECT_EQ(h.get(), &slots_[0]);
        EXPECT_EQ(h.bytes(), 64u);
        EXPECT_TRUE(h.owns());
        EXPECT_FALSE(h.is_view());
        EXPECT_FALSE(h.view());
        EXPECT_TRUE(released().empty());

        EXPECT_TRUE(h.set_owned(owned(1, 128)));
        EXPECT_EQ(count_of(&slots_[0]), 1);
        EXPECT_EQ(released().size(), 1u);
        EXPECT_EQ(h.get(), &slots_[1]);
        EXPECT_EQ(h.bytes(), 128u);
    }  // destructor releases what is still held
    EXPECT_EQ(count_of(&slots_[0]), 1);
    EXPECT_EQ(count_of(&slots_[1]), 1);
    EXPECT_EQ(released().size(), 2u);
}

TEST_F(ViewOrOwnedTest, ResetReleasesOnceAndIsIdempotent) {
    ViewOrOwned<void> h;
    EXPECT_FALSE(h.set_owned(owned(0)));
    EXPECT_TRUE(h.reset());
    EXPECT_FALSE(h.reset());
    EXPECT_EQ(released().size(), 1u);
    EXPECT_EQ(h.get(), nullptr);
    EXPECT_FALSE(h.owns());
}

TEST_F(ViewOrOwnedTest, ViewToOwnedToViewHoldsOneOfTheTwo) {
    const StableSpan<std::byte> s1 = arena_.take_bytes(256);
    const StableSpan<std::byte> s2 = arena_.take_bytes(256);
    ViewOrOwned<void> h;
    EXPECT_FALSE(h.set_view(s1));

    EXPECT_TRUE(h.set_owned(owned(0)));  // the regrow past the slice
    EXPECT_TRUE(h.owns());
    EXPECT_FALSE(h.is_view());
    EXPECT_FALSE(h.view());
    EXPECT_EQ(h.get(), &slots_[0]);
    EXPECT_TRUE(released().empty());  // the slice stays the arena's

    EXPECT_TRUE(h.set_view(s2));  // back to a slice: the private allocation goes, once
    EXPECT_TRUE(h.is_view());
    EXPECT_FALSE(h.owns());
    EXPECT_EQ(h.get(), s2.data());
    EXPECT_EQ(count_of(&slots_[0]), 1);

    EXPECT_TRUE(h.reset());
    EXPECT_EQ(released().size(), 1u);
}

TEST_F(ViewOrOwnedTest, EmptyViewOrFailedAllocationLeavesTheHandleEmpty) {
    const StableSpan<std::byte> s1 = arena_.take_bytes(256);
    const StableSpan<std::byte> refused = arena_.take_bytes(1 << 20);  // past the capacity
    ASSERT_TRUE(refused.empty());

    ViewOrOwned<void> h;
    EXPECT_FALSE(h.set_view(refused));
    EXPECT_FALSE(static_cast<bool>(h));
    EXPECT_FALSE(h.set_owned(DeviceOwned<void>()));
    EXPECT_FALSE(static_cast<bool>(h));

    EXPECT_FALSE(h.set_owned(owned(0)));
    EXPECT_TRUE(h.set_view(refused));  // the held allocation is released, the address is gone
    EXPECT_EQ(count_of(&slots_[0]), 1);
    EXPECT_FALSE(h.is_view());
    EXPECT_EQ(h.get(), nullptr);

    EXPECT_FALSE(h.set_view(s1));
    EXPECT_TRUE(h.set_owned(DeviceOwned<void>()));  // a failed regrow: empty, reported as moved
    EXPECT_EQ(h.get(), nullptr);
    EXPECT_EQ(h.bytes(), 0u);
    EXPECT_EQ(released().size(), 1u);
}

TEST_F(ViewOrOwnedTest, MoveConstructionLeavesTheSourceEmpty) {
    const StableSpan<std::byte> s1 = arena_.take_bytes(256);
    {
        ViewOrOwned<void> a;
        EXPECT_FALSE(a.set_owned(owned(0)));
        ViewOrOwned<void> b(std::move(a));
        EXPECT_EQ(a.get(), nullptr);  // NOLINT(bugprone-use-after-move): the moved-from state is the test
        EXPECT_FALSE(a.owns());
        EXPECT_EQ(b.get(), &slots_[0]);
        EXPECT_TRUE(b.owns());

        ViewOrOwned<void> v;
        EXPECT_FALSE(v.set_view(s1));
        ViewOrOwned<void> w(std::move(v));
        EXPECT_EQ(v.get(), nullptr);  // NOLINT(bugprone-use-after-move)
        EXPECT_FALSE(v.is_view());
        EXPECT_EQ(w.get(), s1.data());
        EXPECT_TRUE(released().empty());
    }
    EXPECT_EQ(count_of(&slots_[0]), 1);
    EXPECT_EQ(released().size(), 1u);
}

TEST_F(ViewOrOwnedTest, MoveAssignmentReleasesTheTargetOnceAndNothingTwice) {
    const StableSpan<std::byte> s1 = arena_.take_bytes(256);
    {
        ViewOrOwned<void> a;
        ViewOrOwned<void> b;
        EXPECT_FALSE(a.set_owned(owned(0)));
        EXPECT_FALSE(b.set_owned(owned(1)));
        b = std::move(a);
        EXPECT_EQ(count_of(&slots_[1]), 1);  // the target's allocation
        EXPECT_EQ(count_of(&slots_[0]), 0);
        EXPECT_EQ(a.get(), nullptr);  // NOLINT(bugprone-use-after-move)
        EXPECT_EQ(b.get(), &slots_[0]);

        ViewOrOwned<void> v;
        EXPECT_FALSE(v.set_view(s1));
        b = std::move(v);  // an owner replaced by a view
        EXPECT_EQ(count_of(&slots_[0]), 1);
        EXPECT_TRUE(b.is_view());
        EXPECT_FALSE(b.owns());
        EXPECT_EQ(b.get(), s1.data());
        EXPECT_FALSE(v.is_view());  // NOLINT(bugprone-use-after-move)

        ViewOrOwned<void> c;
        EXPECT_FALSE(c.set_owned(owned(2)));
        b = std::move(c);  // a view replaced by an owner
        EXPECT_TRUE(b.owns());
        EXPECT_FALSE(b.is_view());
        EXPECT_EQ(released().size(), 2u);
    }
    EXPECT_EQ(count_of(&slots_[2]), 1);
    EXPECT_EQ(released().size(), 3u);
}

TEST_F(ViewOrOwnedTest, TypedHandleTakesATypedSpan) {
    const StableSpan<float> s = arena_.take<float>(16);
    ASSERT_TRUE(s);
    ViewOrOwned<float> h;
    EXPECT_FALSE(h.set_view(s));
    EXPECT_EQ(h.get(), s.data());
    EXPECT_EQ(h.bytes(), 16 * sizeof(float));
    DeviceOwned<float> o;
    o.adopt(&slots_[0], sizeof(float), nullptr);
    EXPECT_TRUE(h.set_owned(std::move(o)));
    EXPECT_EQ(h.get(), &slots_[0]);
    EXPECT_TRUE(h.reset());
    EXPECT_EQ(count_of(&slots_[0]), 1);
}

// The static shape (s_h_normed in engine_scheduler.cpp): no release at scope exit, one in reset.
TEST_F(ViewOrOwnedTest, LeakedStaticHandleReleasesOnlyInReset) {
    {
        LeakedStatic<ViewOrOwned<void>> s;
        EXPECT_FALSE(s->set_owned(owned(0)));
    }
    EXPECT_TRUE(released().empty());
    LeakedStatic<ViewOrOwned<void>> s;
    EXPECT_FALSE(s->set_owned(owned(1)));
    EXPECT_TRUE(s->reset());
    EXPECT_EQ(count_of(&slots_[1]), 1);
    EXPECT_EQ(released().size(), 1u);
}

}  // namespace
