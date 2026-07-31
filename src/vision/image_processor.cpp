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

bool qwen_patchify(const uint8_t* rgb, int width, int height, const QwenPatchifyConfig& cfg,
                   QwenPatches& out) {
    if (!rgb || width <= 0 || height <= 0 || cfg.patch_size <= 0 || cfg.merge_size <= 0 ||
        cfg.temporal_patch_size <= 0)
        return false;

    const int factor = cfg.patch_size * cfg.merge_size;
    const SmartResize rs = qwen_smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels);
    if (!rs.ok)
        return false;

    // Upstream resamples with PIL BICUBIC. Catmull-Rom is the closest filter stb
    // offers; the two are not bit-identical and do not need to be — a resampling
    // difference of this size is far below what the encoder is sensitive to, and
    // claiming PIL parity would be false.
    std::vector<uint8_t> resized(static_cast<size_t>(rs.height) * rs.width * 3);
    if (!stbir_resize(rgb, width, height, width * 3, resized.data(), rs.width, rs.height, rs.width * 3,
                      STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM))
        return false;

    const int P = cfg.patch_size, M = cfg.merge_size, T = cfg.temporal_patch_size;
    const int gh = rs.height / P, gw = rs.width / P;
    const int C = 3;
    out.grid_h = gh;
    out.grid_w = gw;
    out.tokens = gh * gw;
    out.features = C * T * P * P;
    out.data.assign(static_cast<size_t>(out.tokens) * out.features, __float2half(0.0f));

    // Token index follows (gh/M, gw/M, M, M); inside a token, (C, T, ph, pw).
    size_t tok = 0;
    for (int bh = 0; bh < gh / M; ++bh) {
        for (int bw = 0; bw < gw / M; ++bw) {
            for (int mh = 0; mh < M; ++mh) {
                for (int mw = 0; mw < M; ++mw, ++tok) {
                    const int patch_row = bh * M + mh;
                    const int patch_col = bw * M + mw;
                    half* dst = out.data.data() + tok * out.features;
                    for (int c = 0; c < C; ++c) {
                        for (int ph = 0; ph < P; ++ph) {
                            const int y = patch_row * P + ph;
                            for (int pw = 0; pw < P; ++pw) {
                                const int x = patch_col * P + pw;
                                const uint8_t raw = resized[(static_cast<size_t>(y) * rs.width + x) * 3 + c];
                                const float v = (raw / 255.0f - cfg.mean[c]) / cfg.std[c];
                                // The temporal axis is a repeat for a still image.
                                for (int t = 0; t < T; ++t) {
                                    const size_t idx = ((static_cast<size_t>(c) * T + t) * P + ph) * P + pw;
                                    dst[idx] = __float2half(v);
                                }
                            }
                        }
                    }
                }
            }
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
