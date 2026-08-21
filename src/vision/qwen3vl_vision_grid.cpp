#include "vision/qwen3vl_vision_grid.h"

#include <algorithm>
#include <cmath>

namespace imp {

namespace {

// One axis of the bilinear resample, with align_corners=True — which is what
// Qwen3-VL uses, and which is why the endpoints land exactly on the table's
// first and last entry instead of half a texel inside them.
//
// `taps` are clamped into the table; `distance` deliberately is NOT computed
// from the clamped value. At the last index that leaves the second tap with
// weight 0, which is the whole point: it must not contribute.
struct AxisTaps {
    int32_t tap[2];
    float weight[2];
};

AxisTaps axis_taps(int index, int size, int side) {
    // float32 throughout, matching the reference. The floor below sits on an
    // exact integer whenever the grids line up, so the precision of this
    // division decides whether a token gets one tap or two.
    const float src = static_cast<float>(index) * static_cast<float>(side - 1) /
                      static_cast<float>(std::max(size - 1, 1));
    const float base = std::floor(src);
    AxisTaps a{};
    for (int o = 0; o < 2; ++o) {
        const int t = static_cast<int>(base) + o;
        a.tap[o] = static_cast<int32_t>(std::clamp(t, 0, side - 1));
        a.weight[o] = std::max(0.0f, 1.0f - std::fabs(src - base - static_cast<float>(o)));
    }
    return a;
}

}  // namespace

std::expected<QwenVisionGrid, std::string> qwen3vl_build_vision_grid(int grid_h, int grid_w, int merge,
                                                                     int pos_side) {
    if (grid_h <= 0 || grid_w <= 0)
        return std::unexpected("vision grid must be positive");
    if (merge <= 0)
        return std::unexpected("spatial merge size must be positive");
    if (grid_h % merge != 0 || grid_w % merge != 0)
        return std::unexpected("vision grid " + std::to_string(grid_h) + "x" + std::to_string(grid_w) +
                               " is not a multiple of the spatial merge size " + std::to_string(merge));
    if (pos_side <= 0)
        return std::unexpected("position-embedding grid side must be positive");

    const int tokens = grid_h * grid_w;
    QwenVisionGrid g;
    g.tokens = tokens;
    g.row.resize(static_cast<size_t>(tokens));
    g.col.resize(static_cast<size_t>(tokens));
    g.pos_taps.resize(static_cast<size_t>(tokens) * kQwenVisionPosTaps);
    g.pos_weights.resize(static_cast<size_t>(tokens) * kQwenVisionPosTaps);

    const int blocks_w = grid_w / merge;
    for (int i = 0; i < tokens; ++i) {
        // Undo the patchifier's merge-block grouping: tokens arrive as
        // (block_row, block_col, in_row, in_col), not as raster order.
        const int in_col = i % merge;
        const int in_row = (i / merge) % merge;
        const int block_col = (i / (merge * merge)) % blocks_w;
        const int block_row = i / (merge * merge * blocks_w);
        const int r = block_row * merge + in_row;
        const int c = block_col * merge + in_col;
        g.row[static_cast<size_t>(i)] = r;
        g.col[static_cast<size_t>(i)] = c;

        const AxisTaps h = axis_taps(r, grid_h, pos_side);
        const AxisTaps w = axis_taps(c, grid_w, pos_side);
        // Separable: the 2-D tap is the outer product of the two axes.
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                const size_t k = static_cast<size_t>(i) * kQwenVisionPosTaps + a * 2 + b;
                g.pos_taps[k] = h.tap[a] * pos_side + w.tap[b];
                g.pos_weights[k] = h.weight[a] * w.weight[b];
            }
        }
    }

    return g;
}

}  // namespace imp
