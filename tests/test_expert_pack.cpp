// expert_pack_shape / pack_device_experts (src/model/expert_pack.h): which per-expert sets the
// loader packs into [ne, rows, cols], and that a refused or unallocatable pack changes nothing.

#include <gtest/gtest.h>

#include "model/expert_pack.h"

#include <cstdint>
#include <vector>

using namespace imp;

namespace {

std::vector<Tensor> experts(int ne, int64_t rows, int64_t cols, QType q = QType::F16, bool dev = true) {
    static char backing[64];
    std::vector<Tensor> v;
    for (int e = 0; e < ne; ++e) {
        const int64_t shape[4] = {rows, cols, 0, 0};
        v.emplace_back(backing + e, q, 2, shape, dev);
    }
    return v;
}

TEST(ExpertPack, ShapeOfUniformDeviceF16Experts) {
    int64_t s[3] = {};
    ASSERT_TRUE(expert_pack_shape(experts(128, 512, 1536), s));
    EXPECT_EQ(s[0], 128);
    EXPECT_EQ(s[1], 512);
    EXPECT_EQ(s[2], 1536);
}

TEST(ExpertPack, RefusesQuantizedHostSingleOrMixedShapes) {
    int64_t s[3] = {};
    EXPECT_FALSE(expert_pack_shape(experts(8, 64, 64, QType::BF16), s));
    EXPECT_FALSE(expert_pack_shape(experts(8, 64, 64, QType::NVFP4), s));
    EXPECT_FALSE(expert_pack_shape(experts(8, 64, 64, QType::F16, /*dev=*/false), s));
    EXPECT_FALSE(expert_pack_shape(experts(1, 64, 64), s));
    EXPECT_FALSE(expert_pack_shape({}, s));
    auto mixed = experts(4, 64, 64);
    const int64_t other[4] = {64, 32, 0, 0};
    mixed[2] = Tensor(mixed[2].data, QType::F16, 2, other, true);
    EXPECT_FALSE(expert_pack_shape(mixed, s));
    auto hole = experts(4, 64, 64);
    hole[1].data = nullptr;
    EXPECT_FALSE(expert_pack_shape(hole, s));
}

std::vector<void*> owned_by(const std::vector<Tensor>& v) {
    std::vector<void*> a;
    for (const Tensor& t : v)
        a.push_back(t.data);
    return a;
}

TEST(ExpertPack, FailedAllocationLeavesExpertsUntouched) {
    auto v = experts(4, 64, 64);
    auto allocs = owned_by(v);
    const void* before = v[3].data;
    Tensor packed;
    int tries = 0, releases = 0;
    const bool ok = pack_device_experts(
        v, packed, allocs,
        [&](size_t) -> void* {
            ++tries;
            return nullptr;
        },
        [&](void*, size_t) { ++releases; }, nullptr);
    EXPECT_FALSE(ok);
    EXPECT_EQ(tries, 1);
    EXPECT_EQ(packed.data, nullptr);
    EXPECT_EQ(v[3].data, before);
    EXPECT_EQ(releases, 0);
    EXPECT_EQ(allocs.size(), 4u);
}

// A slab or view (data not an allocation of this load) is never released piecewise.
TEST(ExpertPack, ExpertsNotOwnedAreLeftAlone) {
    auto v = experts(4, 64, 64);
    auto allocs = owned_by(v);
    allocs.pop_back();
    Tensor packed;
    int tries = 0;
    EXPECT_FALSE(pack_device_experts(
        v, packed, allocs,
        [&](size_t) -> void* {
            ++tries;
            return nullptr;
        },
        [](void*, size_t) {}, nullptr));
    EXPECT_EQ(tries, 0);
}

TEST(ExpertPack, AlreadyPackedLayerIsLeftAlone) {
    auto v = experts(4, 64, 64);
    auto allocs = owned_by(v);
    static char buf[8];
    const int64_t shape[4] = {4, 64, 64, 0};
    Tensor packed(buf, QType::F16, 3, shape, true);
    int tries = 0;
    EXPECT_FALSE(pack_device_experts(
        v, packed, allocs,
        [&](size_t) -> void* {
            ++tries;
            return nullptr;
        },
        [](void*, size_t) {}, nullptr));
    EXPECT_EQ(tries, 0);
}

}  // namespace
