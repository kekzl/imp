// WeightOwned / WeightSet (memory/weight_owned.h, invariant I5): a weight block is released
// once, through weights_free, on the stream its site names. CPU lane on purpose: the release
// hook replaces weights_free and the blocks are adopted host addresses.

#include <gtest/gtest.h>

#include "memory/weight_owned.h"

#include <type_traits>
#include <utility>
#include <vector>

using namespace imp;

namespace {

struct Release {
    void* ptr;
    cudaStream_t stream;
};

std::vector<Release>& log() {
    static std::vector<Release> l;
    return l;
}

void* g_fail_ptr = nullptr;  // the hook reports a failed free for this block

cudaError_t record(void* ptr, cudaStream_t stream) {
    log().push_back({ptr, stream});
    return ptr == g_fail_ptr ? cudaErrorInvalidValue : cudaSuccess;
}

class WeightOwnedTest : public ::testing::Test {
protected:
    void SetUp() override {
        log().clear();
        g_fail_ptr = nullptr;
        prev_ = set_weight_owned_release_hook(&record);
    }
    void TearDown() override { set_weight_owned_release_hook(prev_); }

    // Distinct addresses that are never dereferenced.
    char slots_[8] = {};
    void* a_ = &slots_[0];
    void* b_ = &slots_[1];
    void* c_ = &slots_[2];
    cudaStream_t pass_ = reinterpret_cast<cudaStream_t>(&slots_[6]);
    cudaStream_t exec_ = reinterpret_cast<cudaStream_t>(&slots_[7]);

private:
    WeightOwnedReleaseHook prev_ = nullptr;
};

static_assert(!std::is_copy_constructible_v<WeightOwned>);
static_assert(!std::is_copy_assignable_v<WeightOwned>);
static_assert(std::is_nothrow_move_constructible_v<WeightOwned>);
static_assert(std::is_nothrow_move_assignable_v<WeightOwned>);
static_assert(!std::is_copy_constructible_v<WeightSet>);

}  // namespace

TEST_F(WeightOwnedTest, EmptyHandleReleasesNothing) {
    {
        WeightOwned h;
        EXPECT_FALSE(h);
        EXPECT_EQ(h.reset(), cudaSuccess);
        EXPECT_FALSE(WeightOwned::adopt(nullptr, 64, pass_));
        EXPECT_FALSE(WeightOwned::allocate(0, pass_));
    }
    EXPECT_TRUE(log().empty());
}

TEST_F(WeightOwnedTest, ScopeEndFreesOnceOnTheAdoptionStream) {
    {
        WeightOwned h = WeightOwned::adopt(a_, 64, pass_);
        EXPECT_EQ(h.get(), a_);
        EXPECT_EQ(h.bytes(), 64u);
        EXPECT_EQ(h.release_stream(), pass_);
    }
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, a_);
    EXPECT_EQ(log()[0].stream, pass_);
}

TEST_F(WeightOwnedTest, ResetIsIdempotentAndReportsTheFreeStatus) {
    WeightOwned h = WeightOwned::adopt(a_, 8, pass_);
    g_fail_ptr = a_;
    EXPECT_EQ(h.reset(), cudaErrorInvalidValue);
    EXPECT_EQ(h.reset(), cudaSuccess);
    EXPECT_FALSE(h);
    EXPECT_EQ(h.bytes(), 0u);
    EXPECT_EQ(log().size(), 1u);
}

TEST_F(WeightOwnedTest, ResetOnNamesTheStreamOfThisRelease) {
    WeightOwned h = WeightOwned::adopt(a_, 8, nullptr);
    EXPECT_EQ(h.reset_on(exec_), cudaSuccess);
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].stream, exec_);
}

TEST_F(WeightOwnedTest, MoveTransfersTheSingleRelease) {
    {
        WeightOwned h = WeightOwned::adopt(a_, 8, pass_);
        WeightOwned moved(std::move(h));
        EXPECT_FALSE(h);  // NOLINT(bugprone-use-after-move): the moved-from state is the contract
        WeightOwned assigned = WeightOwned::adopt(b_, 8, exec_);
        assigned = std::move(moved);  // frees b_ on its own stream first
        ASSERT_EQ(log().size(), 1u);
        EXPECT_EQ(log()[0].ptr, b_);
        EXPECT_EQ(log()[0].stream, exec_);
        EXPECT_EQ(assigned.get(), a_);
    }
    ASSERT_EQ(log().size(), 2u);
    EXPECT_EQ(log()[1].ptr, a_);
    EXPECT_EQ(log()[1].stream, pass_);
}

TEST_F(WeightOwnedTest, ReleaseLeavesTheFamilyWithoutAFree) {
    {
        WeightOwned h = WeightOwned::adopt(a_, 8, pass_);
        EXPECT_EQ(h.release(), a_);
        EXPECT_FALSE(h);
    }
    EXPECT_TRUE(log().empty());
}

TEST_F(WeightOwnedTest, SetKeepsAdoptionOrderAndIgnoresNull) {
    WeightSet set;
    set.adopt(a_, 1);
    set.adopt(nullptr, 2);
    set.adopt(WeightOwned::adopt(b_, 3, pass_));
    set.adopt(WeightOwned());
    ASSERT_EQ(set.size(), 2u);
    EXPECT_EQ(set.ptr_at(0), a_);
    EXPECT_EQ(set.ptr_at(1), b_);
    EXPECT_TRUE(set.contains(b_));
    EXPECT_FALSE(set.contains(c_));
    EXPECT_FALSE(set.contains(nullptr));
    EXPECT_EQ(set.release_all(nullptr), 0);
}

TEST_F(WeightOwnedTest, SetReleasesFrontToBackOnTheNamedStream) {
    WeightSet set;
    set.adopt(a_, 1);
    set.adopt(WeightOwned::adopt(b_, 1, pass_));  // the upload stream does not survive adoption
    set.adopt(c_, 1);
    g_fail_ptr = b_;
    cudaError_t first = cudaSuccess;
    EXPECT_EQ(set.release_all(nullptr, &first), 1);
    EXPECT_EQ(first, cudaErrorInvalidValue);
    EXPECT_TRUE(set.empty());
    ASSERT_EQ(log().size(), 3u);
    EXPECT_EQ(log()[0].ptr, a_);
    EXPECT_EQ(log()[1].ptr, b_);
    EXPECT_EQ(log()[2].ptr, c_);
    for (const Release& r : log())
        EXPECT_EQ(r.stream, nullptr);
    EXPECT_EQ(set.release_all(nullptr), 0);
    EXPECT_EQ(log().size(), 3u);
}

TEST_F(WeightOwnedTest, ExtractMovesOneHandleOutAndKeepsTheRest) {
    WeightSet set;
    set.adopt(a_, 1);
    set.adopt(b_, 2);
    set.adopt(c_, 3);
    {
        WeightOwned src = set.extract(b_);
        ASSERT_TRUE(src);
        EXPECT_EQ(src.bytes(), 2u);
        EXPECT_FALSE(set.contains(b_));
        EXPECT_FALSE(set.extract(b_));  // not held any more
        EXPECT_EQ(src.reset_on(exec_), cudaSuccess);
    }
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, b_);
    EXPECT_EQ(log()[0].stream, exec_);
    ASSERT_EQ(set.size(), 2u);
    EXPECT_EQ(set.ptr_at(0), a_);
    EXPECT_EQ(set.ptr_at(1), c_);
    EXPECT_EQ(set.release_all(nullptr), 0);
}

TEST_F(WeightOwnedTest, SetDestructorFreesWhatIsLeftOnTheNullStream) {
    {
        WeightSet set;
        set.adopt(a_, 1);
        set.adopt(b_, 1);
        (void)set.extract(a_).release();  // left the family: not freed here
    }
    ASSERT_EQ(log().size(), 1u);
    EXPECT_EQ(log()[0].ptr, b_);
    EXPECT_EQ(log()[0].stream, nullptr);
}
