// Packed recurrent snapshot layout (#2419): int8 h rows + FP32 row scales, conv window and tail raw.
#include "memory/ssm_snapshot_int8.h"

#include <gtest/gtest.h>

using imp::QType;
using imp::SsmStateGeometry;

namespace {

// Qwen3.8-27B GDN slab: 48 layers, conv 10240 x 4, 48 value heads x 128 x 128, BF16 h.
SsmStateGeometry qwen38(QType h) { return SsmStateGeometry{48, 10240, 4, 48, 128, 128, h, 0}; }

}  // namespace

TEST(SsmSnapshotInt8, LayoutHalvesBf16State) {
    const SsmStateGeometry g = qwen38(QType::BF16);
    const auto l = imp::ssm_snapshot_int8_layout(g);
    EXPECT_EQ(l.rows, 48 * 128 * 4);
    EXPECT_EQ(l.row_len, 32);
    EXPECT_EQ(l.conv, imp::ssm_conv_bytes_per_layer(g));
    EXPECT_EQ(l.h_q, static_cast<size_t>(48 * 128 * 128));
    EXPECT_EQ(l.h_scale, static_cast<size_t>(48 * 128 * 4 * 4));
    EXPECT_EQ(l.total, l.per_layer * 48);
    const size_t slab = imp::ssm_bytes_per_slot(g);
    EXPECT_LT(l.total * 20, slab * 13);  // < 0.65x the BF16 slab
}

TEST(SsmSnapshotInt8, LayoutKeepsTailAndAlignment) {
    SsmStateGeometry g = qwen38(QType::F32);
    g.extra_bytes_per_slot = 1000;
    const auto l = imp::ssm_snapshot_int8_layout(g);
    EXPECT_EQ(l.extra, 1024u);
    EXPECT_EQ(l.total % 256, 0u);
    EXPECT_EQ(l.per_layer % 256, 0u);
    EXPECT_LT(l.total * 10, imp::ssm_bytes_per_slot(g) * 4);  // < 0.4x the F32 slab
}

TEST(SsmSnapshotInt8, RefusesGeometryWithoutState) {
    SsmStateGeometry g = qwen38(QType::BF16);
    g.n_heads = 0;
    EXPECT_EQ(imp::ssm_snapshot_int8_layout(g).total, 0u);
    g = qwen38(QType::INT8);
    EXPECT_EQ(imp::ssm_snapshot_int8_layout(g).total, 0u);
}
