#include "model/mrope_positions.h"

#include <algorithm>

namespace imp {

std::expected<MRopePositions, std::string> qwen_build_mrope_positions(
    const std::vector<uint8_t>& is_image, const std::vector<MRopeImageGrid>& grids, int start_pos) {
    std::vector<uint8_t> type(is_image.size());
    for (size_t i = 0; i < is_image.size(); ++i)
        type[i] = is_image[i] ? kMRopeImage : kMRopeText;
    return qwen_build_mrope_positions_mm(type, grids, {}, start_pos);
}

std::expected<MRopePositions, std::string> qwen_build_mrope_positions_mm(
    const std::vector<uint8_t>& token_type, const std::vector<MRopeImageGrid>& images,
    const std::vector<MRopeVideoGrid>& videos, int start_pos) {
    const size_t n = token_type.size();
    if (start_pos < 0)
        return std::unexpected("start position must not be negative");
    for (size_t g = 0; g < images.size(); ++g)
        if (images[g].rows <= 0 || images[g].cols <= 0)
            return std::unexpected("image " + std::to_string(g) + " has an empty grid");
    // HF repeat_interleave(video_grid_thw, t) with t := 1: one grid per frame pair.
    std::vector<MRopeImageGrid> frames;
    for (size_t v = 0; v < videos.size(); ++v) {
        if (videos[v].groups <= 0 || videos[v].rows <= 0 || videos[v].cols <= 0)
            return std::unexpected("video " + std::to_string(v) + " has an empty grid");
        frames.insert(frames.end(), static_cast<size_t>(videos[v].groups), {videos[v].rows, videos[v].cols});
    }

    std::vector<int32_t> pos(3 * n);
    auto set = [&](size_t token, int t, int h, int w) {
        pos[token] = t;
        pos[n + token] = h;
        pos[2 * n + token] = w;
    };

    size_t i = 0;
    size_t next_image = 0, next_frame = 0;
    int cur = start_pos;
    while (i < n) {
        const uint8_t type = token_type[i];
        if (type == kMRopeText) {
            // Text advances all three axes in lockstep.
            set(i, cur, cur, cur);
            ++cur;
            ++i;
            continue;
        }
        if (type != kMRopeImage && type != kMRopeVideo)
            return std::unexpected("token " + std::to_string(i) + " has unknown type " +
                                   std::to_string(type));
        const bool is_video = (type == kMRopeVideo);
        const char* what = is_video ? "video frame" : "image";
        // One contiguous run of this type.
        size_t run = 0;
        while (i + run < n && token_type[i + run] == type)
            ++run;
        const std::vector<MRopeImageGrid>& grids = is_video ? frames : images;
        size_t& next_grid = is_video ? next_frame : next_image;
        if (next_grid >= grids.size())
            return std::unexpected(std::string("more ") + what + " runs in the prompt than " + what +
                                   " grids (" + std::to_string(grids.size()) + ")");
        const MRopeImageGrid& g = grids[next_grid];
        if (static_cast<size_t>(g.tokens()) != run)
            return std::unexpected(std::string(what) + " " + std::to_string(next_grid) + " has " +
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

    if (next_image != images.size())
        return std::unexpected("prompt contains " + std::to_string(next_image) + " image runs but " +
                               std::to_string(images.size()) + " grids were supplied");
    if (next_frame != frames.size())
        return std::unexpected("prompt contains " + std::to_string(next_frame) + " video frame runs but " +
                               std::to_string(frames.size()) + " frame grids were supplied");

    return MRopePositions{std::move(pos), cur};
}

}  // namespace imp
