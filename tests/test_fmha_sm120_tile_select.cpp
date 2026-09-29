#include "compute/fmha_sm120_tile_select.h"

#include <gtest/gtest.h>

using namespace imp;

namespace {
constexpr size_t kSm120Smem = 101376;  // cudaDevAttrMaxSharedMemoryPerBlockOptin on sm_120
}  // namespace

// #2243: the old selector reached (Bq>=32, HD512) and (Bq16, HD256) off sm_120; neither has a kernel.
TEST(Sm120FmhaTileSelect, EverySelectedTileHasAnInstance) {
    int checked = 0;
    for (int hd : {32, 64, 96, 128, 160, 192, 256, 384, 512, 1024}) {
        for (size_t smem = 0; smem <= 512 * 1024; smem += 256) {
            const int bq = sm120_fmha_select_bq(hd, smem);
            if (bq == 0)
                continue;
            ++checked;
            EXPECT_TRUE(sm120_fmha_has_tile(bq, hd)) << "hd=" << hd << " smem=" << smem << " bq=" << bq;
            EXPECT_LE(sm120_fmha_smem_bytes(bq, sm120_fmha_bkv(hd), hd), smem) << "hd=" << hd << " bq=" << bq;
        }
    }
    EXPECT_GT(checked, 0);
}

TEST(Sm120FmhaTileSelect, TheTwoHolesOfTheOldSelector) {
    // HD512 with room for Bq=32 (135424 B) at occupancy 1: old -> Bq32 (no instance); now Bq16.
    EXPECT_EQ(sm120_fmha_select_bq(512, 160 * 1024), 16);
    // HD256 below the Bq=32 need (90368 B): old -> Bq16 (no instance); now 0 (caller falls back).
    EXPECT_EQ(sm120_fmha_select_bq(256, 90000), 0);
}

TEST(Sm120FmhaTileSelect, Sm120ChoicesUnchanged) {
    EXPECT_EQ(sm120_fmha_select_bq(64, kSm120Smem), 64);
    EXPECT_EQ(sm120_fmha_select_bq(96, kSm120Smem), 32);
    EXPECT_EQ(sm120_fmha_select_bq(128, kSm120Smem), 32);
    EXPECT_EQ(sm120_fmha_select_bq(256, kSm120Smem), 32);
    EXPECT_EQ(sm120_fmha_select_bq(512, kSm120Smem), 16);
    EXPECT_EQ(sm120_fmha_select_bq(160, kSm120Smem), 0);  // no instance at any Bq
}
