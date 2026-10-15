#pragma once

#include <cstddef>

// Host mirror of the vendored APA / RA2 kernel geometry for exec workspace sizing.
// attention_apa.cu and attention_ra2.cu static_assert every value against the kernel headers.
namespace imp::fp4_attn_geom {

inline constexpr size_t kKvTileTokens = 64;                    // apa::BKV, ra2::BKV
inline constexpr size_t kApaStatsTiles = 16;                   // apa::STATS_TILES
inline constexpr size_t kApaWarpsHd128 = 12;                   // apa::Launch<128>::NW
inline constexpr size_t kApaQBlockRows = kApaWarpsHd128 * 16;  // 192: 16 q rows per warp
inline constexpr size_t kApaTileBytesPerD = 72;                // apa::Cfg<D>::TILE = 72 * D
inline constexpr size_t kRa2TileBytesPerD = 136;               // ra2::Cfg<D>::TILE = 136 * D + 16
inline constexpr size_t kRa2TileBytesExtra = 16;               // ra2 K scale slot

}  // namespace imp::fp4_attn_geom
