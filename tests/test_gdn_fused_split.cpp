// Qwen3-Next in_proj_qkvz / in_proj_ba -> Qwen3.5 GDN slots (#2410). Reference: HF
// Qwen3NextGatedDeltaNet.fix_query_key_value_ordering, per key-head group [q k v z] / [b a].
#include "model/gdn_fused_split.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

using imp::GdnFusedDims;
using imp::QType;
using imp::Tensor;

namespace {

constexpr int64_t kK = 3;  // columns; every element of row i holds i

Tensor rows_tensor(std::vector<uint16_t>& storage, int64_t rows) {
    storage.assign(static_cast<size_t>(rows * kK), 0);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t c = 0; c < kK; ++c)
            storage[static_cast<size_t>(i * kK + c)] = static_cast<uint16_t>(i);
    const int64_t shape[2] = {rows, kK};
    return Tensor(storage.data(), QType::BF16, 2, shape, false);
}

int64_t row_id(const Tensor& t, int64_t row) {
    const auto* p = static_cast<const uint16_t*>(t.data);
    for (int64_t c = 1; c < kK; ++c)
        EXPECT_EQ(p[row * kK + c], p[row * kK]) << "row " << row << " torn";
    return p[row * kK];
}

struct Owned {
    std::vector<void*> v;
    ~Owned() {
        for (void* p : v)
            std::free(p);
    }
};

// n_k=2, dk=2, n_v=4 (r=2), dv=1: group = [q0 q1 k0 k1 v0 v1 z0 z1], 8 rows, 16 total.
const GdnFusedDims kDims{2, 2, 4, 1};

}  // namespace

TEST(GdnFusedSplit, QkvzRegroupsPerKeyHeadGroup) {
    std::vector<uint16_t> s;
    const Tensor t = rows_tensor(s, 16);
    Owned owned;
    Tensor qkv, z;
    ASSERT_TRUE(imp::split_gdn_qkvz(t, kDims, owned.v, qkv, z));
    const std::vector<int64_t> want_qkv = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13};
    const std::vector<int64_t> want_z = {6, 7, 14, 15};
    ASSERT_EQ(qkv.shape[0], static_cast<int64_t>(want_qkv.size()));
    ASSERT_EQ(z.shape[0], static_cast<int64_t>(want_z.size()));
    for (size_t i = 0; i < want_qkv.size(); ++i)
        EXPECT_EQ(row_id(qkv, static_cast<int64_t>(i)), want_qkv[i]) << "qkv row " << i;
    for (size_t i = 0; i < want_z.size(); ++i)
        EXPECT_EQ(row_id(z, static_cast<int64_t>(i)), want_z[i]) << "z row " << i;
    EXPECT_EQ(qkv.shape[1], kK);
    EXPECT_EQ(qkv.qtype, QType::BF16);
}

TEST(GdnFusedSplit, BaRegroupsPerKeyHeadGroup) {
    std::vector<uint16_t> s;
    const Tensor t = rows_tensor(s, 8);  // [b0 b1 a0 a1 | b2 b3 a2 a3]
    Owned owned;
    Tensor b, a;
    ASSERT_TRUE(imp::split_gdn_ba(t, kDims, owned.v, b, a));
    const std::vector<int64_t> want_b = {0, 1, 4, 5};
    const std::vector<int64_t> want_a = {2, 3, 6, 7};
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(row_id(b, static_cast<int64_t>(i)), want_b[i]) << "b row " << i;
        EXPECT_EQ(row_id(a, static_cast<int64_t>(i)), want_a[i]) << "a row " << i;
    }
}

TEST(GdnFusedSplit, RefusesShapeOrQuantizedSource) {
    std::vector<uint16_t> s;
    Tensor t = rows_tensor(s, 15);
    Owned owned;
    Tensor x, y;
    EXPECT_FALSE(imp::split_gdn_qkvz(t, kDims, owned.v, x, y));
    t = rows_tensor(s, 16);
    t.qtype = QType::NVFP4;
    EXPECT_FALSE(imp::split_gdn_qkvz(t, kDims, owned.v, x, y));
    t.qtype = QType::BF16;
    EXPECT_FALSE(imp::split_gdn_qkvz(t, GdnFusedDims{3, 2, 4, 1}, owned.v, x, y));  // n_v % n_k != 0
    EXPECT_FALSE(imp::split_gdn_ba(rows_tensor(s, 7), kDims, owned.v, x, y));
}
