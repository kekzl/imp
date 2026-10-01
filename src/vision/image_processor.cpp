#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "vision/image_processor.h"
#include "vision/image_decode.h"
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

namespace {

// Upstream resamples with PIL BICUBIC; Catmull-Rom is the closest filter stb offers. Not
// bit-identical, and doesn't need to be: the resampling difference is far below what the
// encoder is sensitive to. A frame already at the target size passes through unchanged.
bool resize_rgb(const uint8_t* rgb, int width, int height, const SmartResize& rs, std::vector<uint8_t>& out) {
    out.resize(static_cast<size_t>(rs.height) * rs.width * 3);
    return stbir_resize(rgb, width, height, width * 3, out.data(), rs.width, rs.height, rs.width * 3,
                        STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM) != nullptr;
}

// temporal[t] is the resized [h, w, 3] frame on temporal slot t (cfg.temporal_patch_size slots).
void patchify_resized(std::span<const uint8_t* const> temporal, const SmartResize& rs,
                      const QwenPatchifyConfig& cfg, QwenPatches& out) {
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
                        for (int t = 0; t < T; ++t) {
                            const uint8_t* frame = temporal[static_cast<size_t>(t)];
                            for (int ph = 0; ph < P; ++ph) {
                                const int y = patch_row * P + ph;
                                for (int pw = 0; pw < P; ++pw) {
                                    const int x = patch_col * P + pw;
                                    const uint8_t raw =
                                        frame[(static_cast<size_t>(y) * rs.width + x) * 3 + c];
                                    const float v = (raw / 255.0f - cfg.mean[c]) / cfg.std[c];
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
}

bool patchify_config_ok(const QwenPatchifyConfig& cfg) {
    return cfg.patch_size > 0 && cfg.merge_size > 0 && cfg.temporal_patch_size > 0;
}

}  // namespace

bool qwen_patchify(const uint8_t* rgb, int width, int height, const QwenPatchifyConfig& cfg,
                   QwenPatches& out) {
    if (!rgb || width <= 0 || height <= 0 || !patchify_config_ok(cfg))
        return false;

    const int factor = cfg.patch_size * cfg.merge_size;
    const SmartResize rs = qwen_smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels);
    if (!rs.ok)
        return false;

    std::vector<uint8_t> resized;
    if (!resize_rgb(rgb, width, height, rs, resized))
        return false;
    // Still image: every temporal slot is the same frame.
    const std::vector<const uint8_t*> temporal(static_cast<size_t>(cfg.temporal_patch_size), resized.data());
    patchify_resized(temporal, rs, cfg, out);
    return true;
}

SmartResize qwen_video_smart_resize(int num_frames, int height, int width, int temporal_factor, int factor,
                                    int64_t min_pixels, int64_t max_pixels) {
    SmartResize out;
    if (num_frames <= 0 || height <= 0 || width <= 0 || factor <= 0 || temporal_factor <= 0 ||
        num_frames < temporal_factor)
        return out;
    double h = height, w = width;
    if (height < factor || width < factor) {
        const double scale = std::max(static_cast<double>(factor) / height,
                                      static_cast<double>(factor) / width);
        h = std::trunc(height * scale);
        w = std::trunc(width * scale);
    }
    if (std::max(h, w) / std::min(h, w) > 200.0) {
        IMP_LOG_WARN("video smart_resize: aspect ratio %.0f:%.0f exceeds the 200:1 limit", std::max(h, w),
                     std::min(h, w));
        return out;
    }

    int64_t h_bar = round_to_factor(h, factor);
    int64_t w_bar = round_to_factor(w, factor);
    const int64_t t_bar = round_to_factor(num_frames, temporal_factor);
    const double volume = static_cast<double>(num_frames) * h * w;

    if (t_bar * h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(volume / static_cast<double>(max_pixels));
        h_bar = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(h / beta / factor)) * factor);
        w_bar = std::max<int64_t>(factor, static_cast<int64_t>(std::floor(w / beta / factor)) * factor);
    } else if (t_bar * h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / volume);
        h_bar = static_cast<int64_t>(std::ceil(h * beta / factor)) * factor;
        w_bar = static_cast<int64_t>(std::ceil(w * beta / factor)) * factor;
    }

    out.height = static_cast<int>(h_bar);
    out.width = static_cast<int>(w_bar);
    out.ok = (out.height > 0 && out.width > 0);
    return out;
}

QwenPatchifyConfig qwen_video_patchify_config() {
    QwenPatchifyConfig cfg;
    cfg.min_pixels = 4096;
    cfg.max_pixels = 25165824;
    return cfg;
}

bool qwen_patchify_video(std::span<const uint8_t* const> frames, int width, int height,
                         const QwenPatchifyConfig& cfg, std::vector<QwenPatches>& out) {
    out.clear();
    if (frames.empty() || width <= 0 || height <= 0 || !patchify_config_ok(cfg))
        return false;
    for (const uint8_t* f : frames)
        if (!f)
            return false;

    const int T = cfg.temporal_patch_size;
    const int factor = cfg.patch_size * cfg.merge_size;
    const SmartResize rs = qwen_video_smart_resize(static_cast<int>(frames.size()), height, width, T, factor,
                                                   cfg.min_pixels, cfg.max_pixels);
    if (!rs.ok)
        return false;

    std::vector<std::vector<uint8_t>> resized(frames.size());
    for (size_t f = 0; f < frames.size(); ++f)
        if (!resize_rgb(frames[f], width, height, rs, resized[f]))
            return false;

    // Odd tail: HF pads with copies of the last frame up to a multiple of T.
    const size_t groups = (frames.size() + static_cast<size_t>(T) - 1) / static_cast<size_t>(T);
    out.resize(groups);
    std::vector<const uint8_t*> temporal(static_cast<size_t>(T));
    for (size_t g = 0; g < groups; ++g) {
        for (int t = 0; t < T; ++t)
            temporal[static_cast<size_t>(t)] =
                resized[std::min(g * static_cast<size_t>(T) + static_cast<size_t>(t), frames.size() - 1)]
                    .data();
        patchify_resized(temporal, rs, cfg, out[g]);
    }
    return true;
}

bool decode_rgb(std::span<const uint8_t> data, std::vector<uint8_t>& rgb, int& width, int& height) {
    DecodedImage img;  // JPEG via libjpeg-turbo, bit-identical to Pillow (#2381)
    if (!decode_image(data, img))
        return false;
    width = img.width;
    height = img.height;
    rgb = std::move(img.rgb);
    return true;
}

namespace {

// Pillow ImagingResample, BICUBIC (a = -0.5), 8 bpc: separable, horizontal pass first, the
// intermediate rounded and clamped to u8, 22-bit fixed-point weights. torchvision's uint8
// antialiased bicubic (what GotOcr2 runs) is a port of the same code; a float resampler differs
// by up to 18/255 at hard edges because it never clamps the intermediate.
struct ResampleTaps {
    std::vector<int> start, count;
    std::vector<int32_t> weights;  // [out, ksize]
    int ksize = 0;
};

double bicubic_filter(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

ResampleTaps resample_taps(int in_size, int out_size) {
    constexpr int kPrecisionBits = 32 - 8 - 2;
    ResampleTaps t;
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 2.0 * filterscale;
    t.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    t.start.resize(static_cast<size_t>(out_size));
    t.count.resize(static_cast<size_t>(out_size));
    t.weights.assign(static_cast<size_t>(out_size) * t.ksize, 0);
    std::vector<double> k(static_cast<size_t>(t.ksize));
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
        const int xmax = std::min(static_cast<int>(center + support + 0.5), in_size) - xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            k[x] = bicubic_filter((x + xmin - center + 0.5) / filterscale);
            ww += k[x];
        }
        for (int x = 0; x < xmax; ++x) {
            const double w = ww != 0.0 ? k[x] / ww : k[x];
            t.weights[static_cast<size_t>(xx) * t.ksize + x] = static_cast<int32_t>(
                w * (1 << kPrecisionBits) + (w < 0.0 ? -0.5 : 0.5));
        }
        t.start[xx] = xmin;
        t.count[xx] = xmax;
    }
    return t;
}

uint8_t clip8(int64_t ss) {
    constexpr int kPrecisionBits = 32 - 8 - 2;
    const int64_t v = ss >> kPrecisionBits;
    return static_cast<uint8_t>(std::clamp<int64_t>(v, 0, 255));
}

// [h, w, 3] u8 -> [oh, ow, 3] u8.
void pil_resize_bicubic_u8(const uint8_t* in, int w, int h, uint8_t* out, int ow, int oh) {
    constexpr int kPrecisionBits = 32 - 8 - 2;
    const int64_t half = int64_t{1} << (kPrecisionBits - 1);
    const ResampleTaps th = resample_taps(w, ow), tv = resample_taps(h, oh);
    std::vector<uint8_t> mid(static_cast<size_t>(h) * ow * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t ss = half;
                const int32_t* k = &th.weights[static_cast<size_t>(x) * th.ksize];
                for (int i = 0; i < th.count[x]; ++i)
                    ss += static_cast<int64_t>(in[(static_cast<size_t>(y) * w + th.start[x] + i) * 3 + c]) *
                          k[i];
                mid[(static_cast<size_t>(y) * ow + x) * 3 + c] = clip8(ss);
            }
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t ss = half;
                const int32_t* k = &tv.weights[static_cast<size_t>(y) * tv.ksize];
                for (int i = 0; i < tv.count[y]; ++i)
                    ss += static_cast<int64_t>(mid[(static_cast<size_t>(tv.start[y] + i) * ow + x) * 3 + c]) *
                          k[i];
                out[(static_cast<size_t>(y) * ow + x) * 3 + c] = clip8(ss);
            }
}

}  // namespace

bool internvl_preprocess(const uint8_t* rgb, int width, int height, const InternVLPreprocessConfig& cfg,
                         std::vector<half>& patches, std::vector<uint8_t>* resized) {
    const int S = cfg.image_size, P = cfg.patch_size;
    if (!rgb || width <= 0 || height <= 0 || S <= 0 || P <= 0 || S % P != 0)
        return false;
    std::vector<uint8_t> tile(static_cast<size_t>(S) * S * 3);
    pil_resize_bicubic_u8(rgb, width, height, tile.data(), S, S);
    const int side = S / P, features = 3 * P * P;
    patches.assign(static_cast<size_t>(side) * side * features, __float2half(0.0f));
    for (int pi = 0; pi < side * side; ++pi)
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < P; ++y)
                for (int x = 0; x < P; ++x) {
                    const int py = (pi / side) * P + y, px = (pi % side) * P + x;
                    const float v = (tile[(static_cast<size_t>(py) * S + px) * 3 + c] / 255.0f -
                                     cfg.mean[c]) /
                                    cfg.std[c];
                    patches[static_cast<size_t>(pi) * features + (c * P + y) * P + x] = __float2half(v);
                }
    if (resized)
        *resized = std::move(tile);
    return true;
}

bool load_and_preprocess_image(const std::string& path, int target_size, const float mean[3],
                               const float std[3], ImageData& out) {
    DecodedImage img;
    if (!decode_image_file(path, img)) {
        IMP_LOG_ERROR("Vision: failed to load image: %s", path.c_str());
        return false;
    }

    IMP_LOG_INFO("Vision: loaded image %dx%d from %s", img.width, img.height, path.c_str());

    return preprocess_pixels(img.rgb.data(), img.width, img.height, target_size, mean, std, out);
}

bool load_and_preprocess_image_from_memory(std::span<const uint8_t> data, int target_size,
                                           const float mean[3], const float std[3], ImageData& out) {
    DecodedImage img;
    if (!decode_image(data, img)) {
        IMP_LOG_ERROR("Vision: failed to decode image from memory (%zu bytes)", data.size());
        return false;
    }

    IMP_LOG_INFO("Vision: decoded image %dx%d from memory (%zu bytes)", img.width, img.height, data.size());

    return preprocess_pixels(img.rgb.data(), img.width, img.height, target_size, mean, std, out);
}

}  // namespace imp
