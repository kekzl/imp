// KV dtype size and scale facts (#2460): golden values captured from main de42b750, before the
// open-coded predicates became one KvDtypeInfo table. Uniform cache, per-layer cache, planner.

#include "core/kv_dtype.h"
#include "core/qtype.h"
#include "memory/kv_cache.h"
#include "runtime/vram_budget.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace imp;

namespace {

struct UniformGolden {
    QType dtype;
    int block_size, n_kv_heads, head_dim;
    size_t block_bytes;        // KVCache::block_bytes()
    size_t scale_block_bytes;  // KVCache::scale_block_bytes()
    size_t planner_bytes;      // kv_block_bytes_per_layer(): (data + scale) * 2
};

// Every dtype the KV cache or the planner is called with, plus F32/BF16/FP8_E5M2 (unpacked
// fallback through qtype_elem_bytes), at two geometries.
const std::vector<UniformGolden> kUniform = {
    {QType::F32, 16, 8, 128, 65536, 0, 131072},       {QType::F16, 16, 8, 128, 32768, 0, 65536},
    {QType::BF16, 16, 8, 128, 32768, 0, 65536},       {QType::FP8_E4M3, 16, 8, 128, 16384, 0, 32768},
    {QType::FP8_E5M2, 16, 8, 128, 16384, 0, 32768},   {QType::INT8, 16, 8, 128, 16384, 256, 33280},
    {QType::INT4, 16, 8, 128, 8192, 256, 16896},      {QType::NVFP4, 16, 8, 128, 8192, 1024, 18432},
    {QType::MXFP4_KV, 16, 8, 128, 8192, 1024, 18432},

    {QType::F32, 32, 2, 64, 16384, 0, 32768},         {QType::F16, 32, 2, 64, 8192, 0, 16384},
    {QType::BF16, 32, 2, 64, 8192, 0, 16384},         {QType::FP8_E4M3, 32, 2, 64, 4096, 0, 8192},
    {QType::FP8_E5M2, 32, 2, 64, 4096, 0, 8192},      {QType::INT8, 32, 2, 64, 4096, 128, 8448},
    {QType::INT4, 32, 2, 64, 2048, 128, 4352},        {QType::NVFP4, 32, 2, 64, 2048, 256, 4608},
    {QType::MXFP4_KV, 32, 2, 64, 2048, 256, 4608},
};

}  // namespace

TEST(KvDtypeGolden, UniformCacheAndPlannerBytes) {
    for (const auto& g : kUniform) {
        SCOPED_TRACE(qtype_name(g.dtype));
        SCOPED_TRACE(g.head_dim);
        auto c = KVCache::for_accounting(/*n_layers=*/2, g.n_kv_heads, g.head_dim, g.dtype, /*max_blocks=*/4,
                                         g.block_size);
        EXPECT_EQ(c->block_bytes(), g.block_bytes);
        EXPECT_EQ(c->scale_block_bytes(), g.scale_block_bytes);
        EXPECT_EQ(c->scale_block_bytes(1), g.scale_block_bytes);
        EXPECT_EQ(kv_block_bytes_per_layer(g.dtype, g.block_size, g.n_kv_heads, g.head_dim), g.planner_bytes);
    }
}

TEST(KvDtypeGolden, Fp4HeadDimMustBeMultipleOf16) {
    for (QType q : {QType::NVFP4, QType::MXFP4_KV}) {
        SCOPED_TRACE(qtype_name(q));
        EXPECT_THROW(KVCache::for_accounting(1, 2, 72, q, 4), std::runtime_error);
    }
    for (QType q : {QType::F16, QType::FP8_E4M3, QType::INT8, QType::INT4}) {
        SCOPED_TRACE(qtype_name(q));
        EXPECT_NO_THROW(KVCache::for_accounting(1, 2, 72, q, 4));
    }
}

namespace {

struct PerLayerGolden {
    QType dtype;
    size_t layer_block[3];  // block_bytes(l)
    size_t layer_scale[3];  // scale_block_bytes(l)
    size_t block_bytes;     // scalar fallback: max nkv x max hd
    size_t scale_block_bytes;
};

// Layers: {nkv 8, hd 256}, {no attention}, {nkv 2, hd 512}; block_size 16.
const std::vector<PerLayerGolden> kPerLayer = {
    {QType::F16, {65536, 0, 32768}, {0, 0, 0}, 131072, 0},
    {QType::BF16, {65536, 0, 32768}, {0, 0, 0}, 131072, 0},
    {QType::FP8_E4M3, {32768, 0, 16384}, {0, 0, 0}, 65536, 0},
    {QType::NVFP4, {16384, 0, 8192}, {2048, 0, 1024}, 32768, 4096},
    {QType::MXFP4_KV, {16384, 0, 8192}, {2048, 0, 1024}, 32768, 4096},
};

const std::vector<int> kNkv = {8, 0, 2};
const std::vector<int> kHd = {256, 0, 512};

}  // namespace

TEST(KvDtypeGolden, PerLayerCacheBytes) {
    for (const auto& g : kPerLayer) {
        SCOPED_TRACE(qtype_name(g.dtype));
        auto c = KVCache::for_accounting(3, kNkv, kHd, g.dtype, /*max_blocks=*/4, /*block_size=*/16);
        for (int l = 0; l < 3; l++) {
            SCOPED_TRACE(l);
            EXPECT_EQ(c->block_bytes(l), g.layer_block[l]);
            EXPECT_EQ(c->scale_block_bytes(l), g.layer_scale[l]);
        }
        EXPECT_EQ(c->block_bytes(), g.block_bytes);
        EXPECT_EQ(c->scale_block_bytes(), g.scale_block_bytes);
    }
}

TEST(KvDtypeGolden, PerLayerRefusesPerHeadScales) {
    for (QType q : {QType::INT8, QType::INT4}) {
        SCOPED_TRACE(qtype_name(q));
        EXPECT_THROW(KVCache::for_accounting(3, kNkv, kHd, q, 4, 16), std::runtime_error);
    }
}

// The descriptor itself, against the predicate sets main open-coded:
// packed = INT4|NVFP4|MXFP4_KV, scales = INT8|INT4 (FP16 per head) + NVFP4|MXFP4_KV (1 B / 16).
namespace {

struct FlagGolden {
    QType dtype;
    bool kv, packed;
    KvScale scale;
    int scale_bytes, scale_group;
};

const std::vector<FlagGolden> kFlags = {
    {QType::F16, true, false, KvScale::None, 0, 0},
    {QType::FP8_E4M3, true, false, KvScale::None, 0, 0},
    {QType::INT8, true, false, KvScale::PerHead, 2, 0},
    {QType::INT4, true, true, KvScale::PerHead, 2, 0},
    {QType::NVFP4, true, true, KvScale::PerGroup, 1, 16},
    {QType::MXFP4_KV, true, true, KvScale::PerGroup, 1, 16},
    {QType::F32, false, false, KvScale::None, 0, 0},
    {QType::BF16, false, false, KvScale::None, 0, 0},
    {QType::FP8_E5M2, false, false, KvScale::None, 0, 0},
    {QType::FP4_E2M1, false, false, KvScale::None, 0, 0},
    {QType::Q8_0, false, false, KvScale::None, 0, 0},
};

}  // namespace

TEST(KvDtypeInfo, FlagsPerDtype) {
    for (const auto& g : kFlags) {
        SCOPED_TRACE(qtype_name(g.dtype));
        const KvDtypeInfo d = kv_dtype_info(g.dtype);
        EXPECT_EQ(d.kv, g.kv);
        EXPECT_EQ(d.packed, g.packed);
        EXPECT_EQ(d.scale, g.scale);
        EXPECT_EQ(d.has_scales(), g.scale != KvScale::None);
        EXPECT_EQ(d.scale_bytes, g.scale_bytes);
        EXPECT_EQ(d.scale_group, g.scale_group);
    }
    EXPECT_EQ(kNVFP4Group, 16);
    static_assert(kv_dtype_info(QType::NVFP4).packed && kv_dtype_info(QType::MXFP4_KV).has_scales());
}

TEST(KvDtypeInfo, BlockBytesMatchGolden) {
    for (const auto& g : kUniform) {
        SCOPED_TRACE(qtype_name(g.dtype));
        SCOPED_TRACE(g.head_dim);
        EXPECT_EQ(kv_block_data_bytes(g.dtype, g.block_size, g.n_kv_heads, g.head_dim), g.block_bytes);
        EXPECT_EQ(kv_block_scale_bytes(g.dtype, g.block_size, g.n_kv_heads, g.head_dim), g.scale_block_bytes);
        EXPECT_EQ(kv_data_bytes(g.dtype, static_cast<size_t>(g.block_size) * g.n_kv_heads * g.head_dim),
                  g.block_bytes);
    }
    // Kernel scale strides main passed: INT8/INT4 block_size * nkv halves, FP4 block_size * nkv * hd/16
    // bytes.
    EXPECT_EQ(kv_block_scale_entries(QType::INT8, 16, 8, 128), 128u);
    EXPECT_EQ(kv_block_scale_entries(QType::INT4, 16, 8, 128), 128u);
    EXPECT_EQ(kv_block_scale_entries(QType::NVFP4, 16, 8, 128), 1024u);
    EXPECT_EQ(kv_block_scale_entries(QType::MXFP4_KV, 16, 8, 128), 1024u);
    EXPECT_EQ(kv_block_scale_entries(QType::F16, 16, 8, 128), 0u);
    EXPECT_FALSE(kv_scale_head_dim_ok(QType::NVFP4, 72));
    EXPECT_TRUE(kv_scale_head_dim_ok(QType::INT4, 72));
}
