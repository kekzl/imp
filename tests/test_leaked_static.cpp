// LeakedStatic / StaticDeviceOwned (memory/leaked_static.h, invariant I5): a process-global owner
// releases in its reset hook and never in a static destructor (the CUDA runtime is gone by then).
// CPU lane: the driver is replaced by the DeviceOwned release hook, buffers are host addresses.

#include <gtest/gtest.h>

#include "memory/leaked_static.h"

#include <algorithm>
#include <type_traits>
#include <unordered_map>
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

int g_dtors = 0;
struct CountsDtor {
    ~CountsDtor() { ++g_dtors; }
};

// The shape of the keyed plane caches (mmq_q8_imma WeightPlanes, mmq_q4k_imma WeightCache).
struct Planes {
    DeviceOwned<float> a;
    DeviceOwned<float> b;
    int n = 0;
};

static_assert(!std::is_copy_constructible_v<Planes>);
static_assert(!std::is_copy_assignable_v<Planes>);
static_assert(std::is_nothrow_move_assignable_v<Planes>);
static_assert(!std::is_copy_constructible_v<StaticDeviceOwned<float>>);
static_assert(!std::is_move_constructible_v<StaticDeviceOwned<float>>);
static_assert(!std::is_copy_constructible_v<LeakedStatic<int>>);

// File scope like the production statics (and reachable, so a leak checker stays quiet).
LeakedStatic<std::unordered_map<const void*, Planes>> g_cache;
float g_slots[8] = {};

Planes make_planes(float* a, float* b, int n) {
    Planes p;
    p.a.adopt(a, sizeof(float), nullptr);
    p.b.adopt(b, sizeof(float), nullptr);
    p.n = n;
    return p;
}

class LeakedStaticTest : public ::testing::Test {
protected:
    void SetUp() override {
        released().clear();
        prev_ = set_device_owned_release_hook(&record);
    }
    void TearDown() override {
        g_cache->clear();
        set_device_owned_release_hook(prev_);
    }

private:
    DeviceOwnedReleaseHook prev_ = nullptr;
};

TEST_F(LeakedStaticTest, DestructorOfTheHeldObjectNeverRuns) {
    g_dtors = 0;
    { const LeakedStatic<CountsDtor> s; }
    EXPECT_EQ(g_dtors, 0);
}

TEST_F(LeakedStaticTest, OwnerThatStillHoldsAtScopeExitDoesNotRelease) {
    {
        StaticDeviceOwned<float> s;
        s.adopt(&g_slots[0], sizeof(float), nullptr);
        EXPECT_EQ(s.get(), &g_slots[0]);
        EXPECT_EQ(s.bytes(), sizeof(float));
        EXPECT_TRUE(static_cast<bool>(s));
    }
    EXPECT_TRUE(released().empty());  // the at-exit path: no call into the driver
}

TEST_F(LeakedStaticTest, ResetReleasesOnceAndIsIdempotent) {
    StaticDeviceOwned<float> s;
    s.adopt(&g_slots[0], sizeof(float), nullptr);
    s.reset();
    s.reset();
    EXPECT_EQ(count_of(&g_slots[0]), 1);
    EXPECT_EQ(s.get(), nullptr);
    EXPECT_FALSE(static_cast<bool>(s));
    EXPECT_EQ(released().size(), 1u);
}

TEST_F(LeakedStaticTest, AdoptAndMoveAssignReleaseThePreviousBuffer) {
    StaticDeviceOwned<float> s;
    s.adopt(&g_slots[0], sizeof(float), nullptr);
    s.adopt(&g_slots[1], sizeof(float), nullptr);
    EXPECT_EQ(count_of(&g_slots[0]), 1);
    DeviceOwned<float> next;
    next.adopt(&g_slots[2], sizeof(float), nullptr);
    s = std::move(next);
    EXPECT_EQ(count_of(&g_slots[1]), 1);
    EXPECT_EQ(s.get(), &g_slots[2]);
    EXPECT_EQ(next.get(), nullptr);
    s.adopt(nullptr, 0, nullptr);  // a failed allocation: empty, previous buffer released
    EXPECT_EQ(count_of(&g_slots[2]), 1);
    EXPECT_EQ(released().size(), 3u);
}

TEST_F(LeakedStaticTest, KeyedCacheOverwriteReleasesTheOldEntryOnce) {
    const void* key = &g_slots[7];
    (*g_cache)[key] = make_planes(&g_slots[0], &g_slots[1], 1);
    EXPECT_TRUE(released().empty());  // moving in releases nothing
    (*g_cache)[key] = make_planes(&g_slots[2], &g_slots[3], 2);
    EXPECT_EQ(count_of(&g_slots[0]), 1);
    EXPECT_EQ(count_of(&g_slots[1]), 1);
    EXPECT_EQ(released().size(), 2u);
    EXPECT_EQ(g_cache->find(key)->second.a.get(), &g_slots[2]);
    EXPECT_EQ(g_cache->find(key)->second.n, 2);
}

TEST_F(LeakedStaticTest, KeyedCacheEraseAndClearReleaseEveryPlaneOnce) {
    (*g_cache)[&g_slots[6]] = make_planes(&g_slots[0], &g_slots[1], 1);
    (*g_cache)[&g_slots[7]] = make_planes(&g_slots[2], &g_slots[3], 2);
    g_cache->erase(&g_slots[6]);
    EXPECT_EQ(count_of(&g_slots[0]), 1);
    EXPECT_EQ(count_of(&g_slots[1]), 1);
    EXPECT_EQ(released().size(), 2u);
    g_cache->clear();  // the reset hook
    EXPECT_EQ(count_of(&g_slots[2]), 1);
    EXPECT_EQ(count_of(&g_slots[3]), 1);
    EXPECT_EQ(released().size(), 4u);
    g_cache->clear();
    EXPECT_EQ(released().size(), 4u);
}

}  // namespace
