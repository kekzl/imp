#pragma once

// (t,h,w) position per token when a prompt has images. Text tokens advance all 3 axes
// together (reduces to 0,1,2,... for text-only). An image shares one temporal position
// across its tokens; height/width = row/column. The image advances position by
// max(rows,cols), not its token count. Wrong values don't crash: the model describes
// a different picture than the one given.

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

// is_image[i] marks token i as image; image runs must be contiguous and match `grids` in
// order, or the placeholder expansion and preprocessor disagree - refused here rather
// than silently mis-positioning the rest. start_pos is the first token's position
// (0 fresh, or the continuation point).
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

// Token types as HF mm_token_type_ids: 0 text, 1 image, 2 video.
inline constexpr uint8_t kMRopeText = 0, kMRopeImage = 1, kMRopeVideo = 2;

// One video after the merger: `groups` frame pairs of rows*cols tokens each.
struct MRopeVideoGrid {
    int groups = 0;
    int rows = 0;
    int cols = 0;
};

// HF Qwen3VLModel.get_rope_index: each maximal run of equal non-text type takes the next grid of
// that type. A video is split into `groups` t=1 grids (timestamps separate them), so each frame
// pair is positioned like an image. Same refusals as above, per type.
[[nodiscard]] std::expected<MRopePositions, std::string> qwen_build_mrope_positions_mm(
    const std::vector<uint8_t>& token_type, const std::vector<MRopeImageGrid>& images,
    const std::vector<MRopeVideoGrid>& videos, int start_pos);

// Engine form: `items` holds one grid per encoded item (image or video frame pair) in prompt order,
// the k-th vision run of `token_type` takes items[k]. Refuses a run/item count mismatch.
[[nodiscard]] std::expected<MRopePositions, std::string> qwen_build_mrope_positions_items(
    const std::vector<uint8_t>& token_type, const std::vector<MRopeImageGrid>& items, int start_pos);

}  // namespace imp
