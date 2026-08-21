#include "model/mrope_positions.h"

#include <algorithm>

namespace imp {

std::expected<MRopePositions, std::string> qwen_build_mrope_positions(
    const std::vector<uint8_t>& is_image, const std::vector<MRopeImageGrid>& grids, int start_pos) {
    const size_t n = is_image.size();
    if (start_pos < 0)
        return std::unexpected("start position must not be negative");
    for (size_t g = 0; g < grids.size(); ++g)
        if (grids[g].rows <= 0 || grids[g].cols <= 0)
            return std::unexpected("image " + std::to_string(g) + " has an empty grid");

    std::vector<int32_t> pos(3 * n);
    auto set = [&](size_t token, int t, int h, int w) {
        pos[token] = t;
        pos[n + token] = h;
        pos[2 * n + token] = w;
    };

    size_t i = 0;
    size_t next_grid = 0;
    int cur = start_pos;
    while (i < n) {
        if (!is_image[i]) {
            // Text advances all three axes in lockstep.
            set(i, cur, cur, cur);
            ++cur;
            ++i;
            continue;
        }
        // One contiguous image run.
        size_t run = 0;
        while (i + run < n && is_image[i + run])
            ++run;
        if (next_grid >= grids.size())
            return std::unexpected("more image runs in the prompt than image grids (" +
                                   std::to_string(grids.size()) + ")");
        const MRopeImageGrid& g = grids[next_grid];
        if (static_cast<size_t>(g.tokens()) != run)
            return std::unexpected("image " + std::to_string(next_grid) + " has " +
                                   std::to_string(g.tokens()) + " tokens (" + std::to_string(g.rows) + "x" +
                                   std::to_string(g.cols) + ") but the prompt reserves " +
                                   std::to_string(run));
        // Raster order over the merged grid: the same order the vision merger
        // emits its tokens in, so token k sits at (k / cols, k % cols).
        for (int k = 0; k < g.tokens(); ++k)
            set(i + static_cast<size_t>(k), cur, cur + k / g.cols, cur + k % g.cols);
        // An image costs max(rows, cols) positions, not rows*cols: the axes
        // advance in parallel, so the longer side is what the next token has to
        // clear.
        cur += std::max(g.rows, g.cols);
        i += run;
        ++next_grid;
    }

    if (next_grid != grids.size())
        return std::unexpected("prompt contains " + std::to_string(next_grid) + " image runs but " +
                               std::to_string(grids.size()) + " grids were supplied");

    return MRopePositions{std::move(pos), cur};
}

}  // namespace imp
