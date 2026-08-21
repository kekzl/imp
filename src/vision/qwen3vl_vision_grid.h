#pragma once

// The per-token grid math of the Qwen3-VL encoder, on the host.
//
// Two things the encoder needs before it can run a single GEMM, both pure
// integer/float arithmetic over the patch grid, and both easy to get subtly
// wrong in a way that still produces a running encoder:
//
//   - the (row, col) each token sits at, which is NOT its raster position —
//     tokens come out of the patchifier grouped by spatial-merge block;
//   - how to resample the learned square position-embedding table (48x48 for
//     Qwen3-VL) onto this image's grid, expressed as four gather taps and four
//     weights per token.
//
// Kept on the host and in its own file because it is the part with a real
// oracle: a bilinear resample of an affine table must reproduce the affine
// function exactly, which pins taps and weights independently of each other.

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

// `grid_h`/`grid_w` count patches and must both be multiples of `merge`:
// smart_resize guarantees that, and a grid that is not would silently drop the
// tail of the last merge block. `pos_side` is the side of the learned table.
// Returns the error text instead of a grid, so a half-filled grid is not a
// value that exists.
[[nodiscard]] std::expected<QwenVisionGrid, std::string> qwen3vl_build_vision_grid(int grid_h, int grid_w,
                                                                                   int merge, int pos_side);

}  // namespace imp
