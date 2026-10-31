#pragma once

// Host-side plan of the own dense tile GEMM (nvfp4_gemm_tile.cu, its only includer): which config and which
// stream-K plan a shape runs. Moved from nvfp4_gemm_tile.cu (inline, [[nodiscard]]).

#include "quant/nvfp4_gemm_tile_epilogue.cuh"

#include <algorithm>
#include <cmath>

namespace imp::nvfp4_tile {

// Data-parallel tile by wave cost per column of work (128x256 1.0, 128x128 1.12, isolated
// full-wave bench; 128x256 only from 3/4 of a wave). Stream-K only when that tile idles >= 15 % of its
// waves (M=2048 N=6144 132.5 vs 128.5 us data-parallel); then hybrid (4) vs uniform (3) by k-steps per CTA,
// uniform paying ~5.5 for its fixups: Gemma-4 2048x2816x2112 hybrid 29.0 vs uniform 30.5 us, flushed M=512
// N=17408 K=5120 110 vs 101.5. Below half a wave the split-only plan (4), else data-parallel (M=512 flushed:
// N=4096 K=4096 21.5 vs stream-K 31.5 us).
[[nodiscard]] inline int pick_variant(int M, int N, int kt, int sms, bool sk_fits) {
    const int m_tiles = (M + kBM - 1) / kBM;
    const int t128 = ((N + 127) / 128) * m_tiles;
    const int t256 = ((N + 255) / 256) * m_tiles;
    const auto waves = [&](int ctas) {
        return std::ceil(static_cast<float>(ctas) / static_cast<float>(sms));
    };
    const float c256 = waves(t256) * 256.0f;
    const float c128 = waves(t128) * 128.0f * 1.12f;
    const bool use256 = 4 * t256 >= 3 * sms && c256 <= c128;
    const int dp = use256 ? t256 : t128;
    const float idle = 1.0f - (static_cast<float>(dp) / (waves(dp) * static_cast<float>(sms)));
    if (4 * t128 <= sms)  // 128x64, grid <= half a wave: M=512 N=512 K=2048 flushed 9.5 vs split 11.3 us
        return 5;
    if (sk_fits && 2 * t128 <= sms)
        return 4;
    if (sk_fits && idle >= 0.15f && t128 >= sms) {
        const int hybrid = ((t128 / sms) * kt) + ((kt + 2) / 3);
        const float uniform = (static_cast<float>(t128 * kt) / static_cast<float>(sms)) + 5.5f;
        return m_tiles >= 16 && static_cast<float>(hybrid) <= uniform ? 4 : 3;  // M < 2048: uniform (in situ
        // Qwen3-8B 512x12288x4096 52.3 vs hybrid 54.2 us)
    }
    return use256 ? 1 : 0;
}

// Hybrid stream-K plan: whole waves of tiles data-parallel over grid = SMs, each tile of the partial last
// wave split into 3 k-ranges on as many CTAs; below one wave every tile splits in 2 (grid = tiles x 2;
// Qwen3-8B 512x1024x4096 in situ 10.9-11.0 us vs 4 pieces 11.5-11.6). Gemma-4 down_proj 2048x2816x2112 in
// situ, us: pieces 3 29.0 (cuBLASLt 29.1), 2 29.2-29.7, 4 30.0, 8 34.4, uniform stream-K 30.5-31.0.
[[nodiscard]] inline bool hybrid_plan(TileArgs& p, int sms, int kt) {
    const int tiles = ((p.N + 127) / 128) * ((p.M + kBM - 1) / kBM);
    const int dp = (tiles / sms) * sms;
    const int tail = tiles - dp;
    const int pieces = std::min(dp > 0 ? 3 : 2, kt);
    if (tail == 0 || pieces < 2 || tail * pieces > sms)
        return false;
    p.sk_dp = dp;
    p.sk_split = pieces;
    p.sk_per = dp > 0 ? sms : tail * pieces;  // the grid
    return true;
}

}  // namespace imp::nvfp4_tile
