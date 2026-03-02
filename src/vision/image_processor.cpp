#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "vision/image_processor.h"
#include "core/logging.h"

#include <cmath>

namespace imp {

static bool preprocess_pixels(const uint8_t* rgb, int w, int h,
                               int target_size,
                               const float mean[3], const float std[3],
                               ImageData& out) {
    // Resize to target_size x target_size using bilinear interpolation
    std::vector<uint8_t> resized(target_size * target_size * 3);
    stbir_resize_uint8_linear(
        rgb, w, h, w * 3,
        resized.data(), target_size, target_size, target_size * 3,
        STBIR_RGB);

    // Convert to normalized FP16 in CHW layout
    out.width = target_size;
    out.height = target_size;
    int n_pixels = target_size * target_size;
    out.pixels.resize(3 * n_pixels);

    for (int c = 0; c < 3; c++) {
        float inv_std = 1.0f / std[c];
        for (int i = 0; i < n_pixels; i++) {
            float val = static_cast<float>(resized[i * 3 + c]) / 255.0f;
            val = (val - mean[c]) * inv_std;
            out.pixels[c * n_pixels + i] = __float2half(val);
        }
    }

    return true;
}

bool load_and_preprocess_image(const std::string& path, int target_size,
                                const float mean[3], const float std[3],
                                ImageData& out) {
    int w, h, channels;
    uint8_t* rgb = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!rgb) {
        IMP_LOG_ERROR("Vision: failed to load image: %s (%s)",
                      path.c_str(), stbi_failure_reason());
        return false;
    }

    IMP_LOG_INFO("Vision: loaded image %dx%d (%d channels) from %s",
                 w, h, channels, path.c_str());

    bool ok = preprocess_pixels(rgb, w, h, target_size, mean, std, out);
    stbi_image_free(rgb);
    return ok;
}

bool load_and_preprocess_image_from_memory(const uint8_t* data, size_t len,
                                            int target_size,
                                            const float mean[3], const float std[3],
                                            ImageData& out) {
    int w, h, channels;
    uint8_t* rgb = stbi_load_from_memory(data, static_cast<int>(len),
                                          &w, &h, &channels, 3);
    if (!rgb) {
        IMP_LOG_ERROR("Vision: failed to decode image from memory (%s)",
                      stbi_failure_reason());
        return false;
    }

    IMP_LOG_INFO("Vision: decoded image %dx%d from memory (%zu bytes)", w, h, len);

    bool ok = preprocess_pixels(rgb, w, h, target_size, mean, std, out);
    stbi_image_free(rgb);
    return ok;
}

} // namespace imp
