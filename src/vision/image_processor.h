#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_fp16.h>

namespace imp {

struct ImageData {
    std::vector<half> pixels;  // [3, H, W] CHW layout, normalized FP16
    int width = 0;
    int height = 0;
};

// Qwen-VL "smart resize": the target size for a dynamic-resolution vision
// encoder. Both dimensions come out divisible by `factor` (patch_size *
// merge_size — 32 for Qwen3-VL), the total pixel count is pulled into
// [min_pixels, max_pixels], and the aspect ratio is preserved as closely as
// those two allow.
//
// Ported from transformers' `smart_resize` (qwen2_vl image processing), which
// Qwen3-VL reuses. Two details are faithful to it on purpose:
//   - the initial rounding is Python's round(), i.e. BANKER'S rounding (ties to
//     even), not round-half-away-from-zero. They differ for exact .5 — a
//     16px side with factor 32 gives 0 in Python and 1 with std::round — and a
//     silent one-step difference here changes the token count for the whole
//     image;
//   - the max_pixels branch floors with a `max(factor, ...)` guard and the
//     min_pixels branch ceils WITHOUT one. Not symmetric, and not a typo.
//
// `ok` is false when the aspect ratio exceeds 200:1, which upstream raises on.
struct SmartResize {
    int height = 0;
    int width = 0;
    bool ok = false;
};
SmartResize qwen_smart_resize(int height, int width, int factor, int64_t min_pixels, int64_t max_pixels);

// Qwen-VL patchification: an image becomes [tokens, C*T*P*P] FP16, which is
// exactly what the flattened `patch_embed` matrix ([1024, 1536] for Qwen3-VL)
// multiplies. Ported from transformers' Qwen2VLImageProcessorFast — the
// ordering below is read off its reshape/permute, not inferred, because two
// parts of it are not what one would guess:
//
//   - TOKEN order is grouped by 2x2 MERGE BLOCK, not raster:
//     (grid_h/merge, grid_w/merge, merge_h, merge_w). The patch merger later
//     consumes four CONSECUTIVE tokens, which only works because of this.
//   - WITHIN a token the layout is (C, T, patch_h, patch_w) — channel-major,
//     then the temporal axis. For a still image the temporal axis is a plain
//     REPEAT of the same spatial patch (upstream `expand`), not a second frame.
//
// grid_h/grid_w count patches, so tokens == grid_h * grid_w. Both are even
// because smart_resize aligns to patch_size * merge_size.
struct QwenPatches {
    std::vector<half> data;  // [tokens, channels * temporal * patch * patch]
    int grid_h = 0;          // in patches
    int grid_w = 0;
    int tokens = 0;
    int features = 0;  // channels * temporal * patch * patch
};

struct QwenPatchifyConfig {
    int patch_size = 16;
    int merge_size = 2;
    int temporal_patch_size = 2;
    int64_t min_pixels = 65536;     // 256^2 — a PIXEL COUNT, despite upstream
    int64_t max_pixels = 16777216;  // 4096^2  spelling it shortest/longest_edge
    float mean[3] = {0.5f, 0.5f, 0.5f};
    float std[3] = {0.5f, 0.5f, 0.5f};
};

// Resize (smart_resize target, Catmull-Rom), rescale to [0,1], normalise, and
// patchify. Returns false on a decode/size failure. `rgb` is [h, w, 3] u8.
bool qwen_patchify(const uint8_t* rgb, int width, int height, const QwenPatchifyConfig& cfg,
                   QwenPatches& out);

// Load image from file, resize to target_size x target_size, normalize, convert to FP16 CHW.
bool load_and_preprocess_image(const std::string& path, int target_size, const float mean[3],
                               const float std[3], ImageData& out);

// Load image from memory buffer, resize + normalize + FP16 CHW.
bool load_and_preprocess_image_from_memory(const uint8_t* data, size_t len, int target_size,
                                           const float mean[3], const float std[3], ImageData& out);

}  // namespace imp
