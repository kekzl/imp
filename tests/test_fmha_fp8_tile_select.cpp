#include "compute/fmha_fp8_tile_select.h"

#include <gtest/gtest.h>

using namespace imp;

namespace {
constexpr int kBkv = 64;               // SM120_Bkv in attention_fmha_sm120.cu
constexpr size_t kSm120Smem = 101376;  // cudaDevAttrMaxSharedMemoryPerBlockOptin on sm_120
}  // namespace

// #2195: the selector returned Bq=128 for HD64, which had no instance, so the launch returned false.
TEST(Fp8FmhaTileSelect, EverySelectedTileHasAnInstance) {
    int checked = 0;
    for (int hd = 32; hd <= 1024; hd += 32) {
        for (size_t smem = 0; smem <= 256 * 1024; smem += 512) {
            const int bq = fp8_fmha_select_bq(hd, kBkv, smem);
            if (bq == 0)
                continue;
            ++checked;
            EXPECT_TRUE(fp8_fmha_has_tile(bq, hd)) << "hd=" << hd << " smem=" << smem << " bq=" << bq;
            EXPECT_LE(fp8_fmha_smem_bytes(bq, kBkv, hd), smem) << "hd=" << hd << " bq=" << bq;
        }
    }
    EXPECT_GT(checked, 0);
}

TEST(Fp8FmhaTileSelect, SelectsTheLargestFittingInstance) {
    for (const Fp8FmhaTile& t : kFp8FmhaTiles) {
        const size_t need = fp8_fmha_smem_bytes(t.bq, kBkv, t.head_dim);
        const int bq = fp8_fmha_select_bq(t.head_dim, kBkv, need);
        EXPECT_GE(bq, t.bq) << "hd=" << t.head_dim << " tile bq=" << t.bq;
    }
}

TEST(Fp8FmhaTileSelect, Sm120Choices) {
    EXPECT_EQ(fp8_fmha_select_bq(64, kBkv, kSm120Smem), 64);
    EXPECT_EQ(fp8_fmha_select_bq(128, kBkv, kSm120Smem), 64);
    EXPECT_EQ(fp8_fmha_select_bq(256, kBkv, kSm120Smem), 32);
    EXPECT_EQ(fp8_fmha_select_bq(96, kBkv, kSm120Smem), 0);
}

TEST(Fp8FmhaTileSelect, SupportedHeadDims) {
    for (int hd : {64, 128, 256})
        EXPECT_TRUE(fp8_fmha_supports_head_dim(hd)) << "hd=" << hd;
    for (int hd : {32, 96, 160, 192, 512})
        EXPECT_FALSE(fp8_fmha_supports_head_dim(hd)) << "hd=" << hd;
}
