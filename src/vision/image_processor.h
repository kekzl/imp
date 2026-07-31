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

// Load image from file, resize to target_size x target_size, normalize, convert to FP16 CHW.
bool load_and_preprocess_image(const std::string& path, int target_size, const float mean[3],
                               const float std[3], ImageData& out);

// Load image from memory buffer, resize + normalize + FP16 CHW.
bool load_and_preprocess_image_from_memory(const uint8_t* data, size_t len, int target_size,
                                           const float mean[3], const float std[3], ImageData& out);

}  // namespace imp
