#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "vision/image_processor.h"
#include "core/logging.h"

#include <algorithm>
#include <cmath>

namespace imp {

namespace {

// Python's round(): ties go to the EVEN integer. std::round ties away from
// zero, which differs at exact .5 and would silently shift the token count.
int64_t round_half_to_even(double v) {
    const double r = std::nearbyint(v);  // honours the default FE_TONEAREST = ties-to-even
    return static_cast<int64_t>(r);
}

int64_t round_to_factor(double v, int factor) { return round_half_to_even(v / factor) * factor; }

}  // namespace

SmartResize qwen_smart_resize(int height, int width, int factor, int64_t min_pixels, int64_t max_pixels) {
    SmartResize out;
    if (height <= 0 || width <= 0 || factor <= 0)
        return out;
    const int lo = std::min(height, width), hi = std::max(height, width);
    if (static_cast<double>(hi) / static_cast<double>(lo) > 200.0) {
        IMP_LOG_WARN("smart_resize: aspect ratio %d:%d exceeds the 200:1 limit", hi, lo);
        return out;
    }

    int64_t h_bar = round_to_factor(height, factor);
    int64_t w_bar = round_to_factor(width, factor);
    const double area = static_cast<double>(height) * static_cast<double>(width);

    if (h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(area / static_cast<double>(max_pixels));
        h_bar = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(width / beta / factor)) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        // Deliberately no max(factor, ...) guard here — upstream has none, and
        // ceil() cannot land below one factor for a positive input anyway.
        const double beta = std::sqrt(static_cast<double>(min_pixels) / area);
        h_bar = static_cast<int64_t>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int64_t>(std::ceil(width * beta / factor)) * factor;
    }

    out.height = static_cast<int>(h_bar);
    out.width = static_cast<int>(w_bar);
    out.ok = (out.height > 0 && out.width > 0);
    return out;
}

static bool preprocess_pixels(const uint8_t* rgb, int w, int h, int target_size, const float mean[3],
                              const float std[3], ImageData& out) {
    // Resize to target_size x target_size using bilinear interpolation
    std::vector<uint8_t> resized(static_cast<size_t>(target_size) * target_size * 3);
    stbir_resize_uint8_linear(rgb, w, h, w * 3, resized.data(), target_size, target_size, target_size * 3,
                              STBIR_RGB);

    // Convert to normalized FP16 in CHW layout
    out.width = target_size;
    out.height = target_size;
    int n_pixels = target_size * target_size;
    out.pixels.resize(static_cast<size_t>(3) * n_pixels);

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

bool load_and_preprocess_image(const std::string& path, int target_size, const float mean[3],
                               const float std[3], ImageData& out) {
    int w, h, channels;
    uint8_t* rgb = stbi_load(path.c_str(), &w, &h, &channels, 3);
    if (!rgb) {
        IMP_LOG_ERROR("Vision: failed to load image: %s (%s)", path.c_str(), stbi_failure_reason());
        return false;
    }

    IMP_LOG_INFO("Vision: loaded image %dx%d (%d channels) from %s", w, h, channels, path.c_str());

    bool ok = preprocess_pixels(rgb, w, h, target_size, mean, std, out);
    stbi_image_free(rgb);
    return ok;
}

bool load_and_preprocess_image_from_memory(const uint8_t* data, size_t len, int target_size,
                                           const float mean[3], const float std[3], ImageData& out) {
    int w, h, channels;
    uint8_t* rgb = stbi_load_from_memory(data, static_cast<int>(len), &w, &h, &channels, 3);
    if (!rgb) {
        IMP_LOG_ERROR("Vision: failed to decode image from memory (%s)", stbi_failure_reason());
        return false;
    }

    IMP_LOG_INFO("Vision: decoded image %dx%d from memory (%zu bytes)", w, h, len);

    bool ok = preprocess_pixels(rgb, w, h, target_size, mean, std, out);
    stbi_image_free(rgb);
    return ok;
}

}  // namespace imp
