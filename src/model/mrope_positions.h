#pragma once

// The (t, h, w) position each token gets when a prompt contains images.
//
// Text tokens advance all three axes together, so a text-only prompt comes out
// identical to `0, 1, 2, ...` on every axis — which is what makes M-RoPE a no-op
// there. An image instead lays its tokens out on a grid: every token of the
// image shares one temporal position, and its height and width positions are its
// row and column within the image. The whole image then advances the running
// position by only `max(rows, cols)`, not by its token count, which is why a
// picture costs far fewer positions than tokens.
//
// Getting this wrong does not crash: the model reads the image as if its tokens
// were laid out somewhere else, and describes a different picture.

#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

namespace imp {

// One image's grid AFTER the vision merger, i.e. in LM tokens: `rows * cols`
// tokens, laid out in raster order.
struct MRopeImageGrid {
    int rows = 0;
    int cols = 0;
    int tokens() const { return rows * cols; }
};

// `is_image[i]` marks token i as belonging to an image. Image runs must be
// contiguous and their lengths must match `grids` in order — a mismatch means
// the placeholder expansion and the preprocessor disagree, and is refused here
// rather than silently mis-positioning the rest of the prompt.
//
// `start_pos` is the position the first token takes (0 for a fresh prompt, the
// continuation point when appending).
struct MRopePositions {
    // [3, n_tokens], axis-major, ready for MRopeParams.
    std::vector<int32_t> pos;
    // The position a token appended after this sequence would take. It travels
    // with `pos` because it is only meaningful together with it: the two used
    // to be separate out-parameters, and a caller that read one without the
    // other got a delta computed against the wrong sequence.
    int next_pos = 0;
};

[[nodiscard]] std::expected<MRopePositions, std::string> qwen_build_mrope_positions(
    const std::vector<uint8_t>& is_image, const std::vector<MRopeImageGrid>& grids, int start_pos);

}  // namespace imp
