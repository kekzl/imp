#pragma once

#include <cstddef>
#include <initializer_list>

// Tile selection for fmha_sm120_prefill (FP16 WMMA). Host-only and constexpr so the CPU unit lane
// (tests/test_fmha_sm120_tile_select.cpp) covers it; the launcher switch instantiates exactly
// kSm120FmhaTiles.

namespace imp {

struct Sm120FmhaTile {
    int bq;
    int head_dim;
};

// Every (Bq, head_dim) with a fmha_sm120_kernel instance (the Bq switch in attention_fmha_sm120.cu).
inline constexpr Sm120FmhaTile kSm120FmhaTiles[] = {
    {128, 64}, {128, 96}, {128, 128}, {128, 256}, {64, 64},  {64, 96},  {64, 128},
    {64, 256}, {32, 64},  {32, 96},   {32, 128},  {32, 256}, {16, 512},
};

constexpr bool sm120_fmha_has_tile(int bq, int head_dim) {
    for (const Sm120FmhaTile& t : kSm120FmhaTiles)
        if (t.bq == bq && t.head_dim == head_dim)
            return true;
    return false;
}

// KV tile columns, must match the kernel's compile-time Bkv: 32 for hd >= 512, else SM120_Bkv (64).
constexpr int sm120_fmha_bkv(int head_dim) { return head_dim >= 512 ? 32 : 64; }

// Dynamic smem: Q (half) + shared K/V tile (half) + S (f32) + O_acc (f32) + row_m/row_l (f32).
constexpr size_t sm120_fmha_smem_bytes(int bq, int bkv, int head_dim) {
    return (size_t)bq * head_dim * 2 + (size_t)bkv * head_dim * 2 + (size_t)bq * bkv * sizeof(float) +
           (size_t)bq * head_dim * sizeof(float) + 2 * (size_t)bq * sizeof(float);
}

// Bq for head_dim under max_smem, 0 when no instanced tile fits. Occupancy 2 first (smem <= max/2,
// Bq 128 -> 64 -> 32), then occupancy 1 (Bq 32 -> 16). Only instanced pairs are candidates (#2243).
constexpr int sm120_fmha_select_bq(int head_dim, size_t max_smem) {
    const int bkv = sm120_fmha_bkv(head_dim);
    for (int bq : {128, 64, 32})
        if (sm120_fmha_has_tile(bq, head_dim) && sm120_fmha_smem_bytes(bq, bkv, head_dim) <= max_smem / 2)
            return bq;
    for (int bq : {32, 16})
        if (sm120_fmha_has_tile(bq, head_dim) && sm120_fmha_smem_bytes(bq, bkv, head_dim) <= max_smem)
            return bq;
    return 0;
}

// sm_120 opt-in smem 101376 B: HD64 Bq64, HD96/128 Bq32 (occupancy 2), HD256 Bq32, HD512 Bq16.
static_assert(sm120_fmha_select_bq(64, 101376) == 64, "HD64 on sm_120");
static_assert(sm120_fmha_select_bq(128, 101376) == 32, "HD128 on sm_120");
static_assert(sm120_fmha_select_bq(256, 101376) == 32, "HD256 on sm_120");
static_assert(sm120_fmha_select_bq(512, 101376) == 16, "HD512 on sm_120");

}  // namespace imp
