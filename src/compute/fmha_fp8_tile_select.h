#pragma once

#include <cstddef>
#include <cstdint>

// Tile selection for fmha_sm120_fp8_prefill. Host-only and constexpr so the CPU unit lane
// (tests/test_fmha_fp8_tile_select.cpp) covers it; the launcher instantiates exactly kFp8FmhaTiles.

namespace imp {

struct Fp8FmhaTile {
    int bq;
    int head_dim;
};

// Every (Bq, head_dim) with a fmha_sm120_fp8_kernel instance. The launcher is generated from this
// table, so the selector cannot return a pair that has no kernel.
inline constexpr Fp8FmhaTile kFp8FmhaTiles[] = {
    {128, 128}, {128, 256}, {64, 64}, {64, 128}, {64, 256}, {32, 64}, {32, 128}, {32, 256},
};

// Bq candidates, largest first.
inline constexpr int kFp8FmhaBqOrder[] = {128, 64, 32};

constexpr bool fp8_fmha_has_tile(int bq, int head_dim) {
    for (const Fp8FmhaTile& t : kFp8FmhaTiles)
        if (t.bq == bq && t.head_dim == head_dim)
            return true;
    return false;
}

constexpr bool fp8_fmha_supports_head_dim(int head_dim) {
    for (int bq : kFp8FmhaBqOrder)
        if (fp8_fmha_has_tile(bq, head_dim))
            return true;
    return false;
}

// Dynamic smem: Q_fp8 (1 B) + KV buffer (half) + S_tile (f32) + O_acc (f32) + row_m/row_l (f32).
constexpr size_t fp8_fmha_smem_bytes(int bq, int bkv, int head_dim) {
    return (size_t)bq * head_dim * sizeof(uint8_t) + (size_t)bkv * head_dim * 2 +
           (size_t)bq * bkv * sizeof(float) + (size_t)bq * head_dim * sizeof(float) +
           2 * (size_t)bq * sizeof(float);
}

// Largest Bq that has an instance for head_dim and fits max_smem; 0 when none does.
constexpr int fp8_fmha_select_bq(int head_dim, int bkv, size_t max_smem) {
    for (int bq : kFp8FmhaBqOrder)
        if (fp8_fmha_has_tile(bq, head_dim) && fp8_fmha_smem_bytes(bq, bkv, head_dim) <= max_smem)
            return bq;
    return 0;
}

namespace fp8_fmha_detail {
constexpr bool every_tile_is_a_candidate() {
    for (const Fp8FmhaTile& t : kFp8FmhaTiles) {
        bool found = false;
        for (int bq : kFp8FmhaBqOrder)
            found = found || bq == t.bq;
        if (!found)
            return false;
    }
    return true;
}
}  // namespace fp8_fmha_detail

static_assert(fp8_fmha_detail::every_tile_is_a_candidate(),
              "kFp8FmhaTiles has a Bq the selector never tries");
// sm_120 opt-in smem is 101376 B: HD64 has no Bq=128 instance, so it takes Bq=64 (#2195).
static_assert(fp8_fmha_select_bq(64, 64, 101376) == 64, "HD64 must select its Bq=64 instance");
static_assert(fp8_fmha_select_bq(96, 64, 101376) == 0, "HD96 has no FP8 instance");

}  // namespace imp
