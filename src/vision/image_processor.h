#pragma once

#include <string>
#include <vector>
#include <span>
#include <cstdint>
#include <cuda_fp16.h>

namespace imp {

struct ImageData {
    std::vector<half> pixels;  // [3, H, W] CHW layout, normalized FP16
    int width = 0;
    int height = 0;
};

// Qwen-VL smart resize: target size for a dynamic-resolution encoder. Both dims come out
// divisible by factor (patch_size*merge_size, 32 for Qwen3-VL); total pixels pulled into
// [min_pixels,max_pixels]; aspect ratio preserved as closely as those two allow. Ported
// from transformers' smart_resize; two details are faithful on purpose:
//   initial rounding is Python round() (banker's, ties to even), not round-half-away
//   (differs at exact .5, e.g. a 16px side/factor 32 gives 0 vs std::round's 1, changing
//   the whole image's token count);
//   max_pixels floors with a max(factor,...) guard, min_pixels ceils WITHOUT one - not
//   symmetric, not a typo.
// `ok` is false when aspect ratio exceeds 200:1 (upstream raises on this).
struct SmartResize {
    int height = 0;
    int width = 0;
    bool ok = false;
};
SmartResize qwen_smart_resize(int height, int width, int factor, int64_t min_pixels, int64_t max_pixels);

// Qwen-VL patchification: an image becomes [tokens, C*T*P*P] FP16, matching the flattened
// patch_embed matrix ([1024,1536] for Qwen3-VL). Ported from Qwen2VLImageProcessorFast;
// two non-obvious facts, read off its reshape/permute:
//   TOKEN order is grouped by 2x2 MERGE BLOCK, not raster: (grid_h/merge, grid_w/merge,
//   merge_h, merge_w) - the patch merger consumes four CONSECUTIVE tokens because of this;
//   WITHIN a token the layout is (C,T,patch_h,patch_w), channel-major then temporal; for a
//   still image the temporal axis is a plain REPEAT of the same spatial patch, not a 2nd frame.
// tokens == grid_h*grid_w; both even since smart_resize aligns to patch_size*merge_size.
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
bool load_and_preprocess_image_from_memory(std::span<const uint8_t> data, int target_size,
                                           const float mean[3], const float std[3], ImageData& out);

}  // namespace imp
