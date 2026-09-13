#pragma once

// Per-token grid math of the Qwen3-VL encoder (host-side): the (row,col) each token sits at
// (NOT raster position, since the patchifier groups tokens by spatial-merge block), and how
// to resample the learned square position-embedding table (48x48) onto this image's grid as
// four gather taps + weights per token. Kept on the host with its own oracle: a bilinear
// resample of an affine table must reproduce the affine function exactly, pinning taps and
// weights independently.

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace imp {

struct QwenVisionGrid {
    int tokens = 0;
    // Patch coordinates per token, in the order the patchifier emits tokens.
    // These are the position ids the encoder's 2-D RoPE rotates by.
    std::vector<int32_t> row;  // [tokens]
    std::vector<int32_t> col;  // [tokens]
    // Bilinear resample of the learned position table, as a gather: four flat
    // indices into a `pos_side * pos_side` table and four weights per token.
    std::vector<int32_t> pos_taps;   // [tokens * 4]
    std::vector<float> pos_weights;  // [tokens * 4]
};

inline constexpr int kQwenVisionPosTaps = 4;

// grid_h/grid_w count patches and must both be multiples of merge (smart_resize
// guarantees this; otherwise the tail of the last merge block would silently drop).
// pos_side is the learned table's side. Returns the error text instead of a grid, so a
// half-filled grid cannot exist as a value.
[[nodiscard]] std::expected<QwenVisionGrid, std::string> qwen3vl_build_vision_grid(int grid_h, int grid_w,
                                                                                   int merge, int pos_side);

}  // namespace imp
