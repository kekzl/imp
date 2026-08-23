// =============================================================================
// Vision CPU preprocessing tests (issue #557 item 1).
//
// The 5 vision source files previously had zero unit coverage — validation was
// manual E2E only, requiring a local VL model. The CPU half (stb decode →
// resize → mean/std normalize → FP16 CHW) needs no GPU and no weights: this
// file synthesizes a 24-bit BMP in memory byte-by-byte (independent of stb's
// writers) and checks the produced tensor against hand-computed values.
//
// What is covered: CHW layout, channel order, normalization arithmetic,
// resize, the memory- and file-entry points, and clean failure on garbage
// input. The GPU half (SigLIP encoder, projector) stays E2E-only by design.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vision/image_processor.h"

namespace imp {
namespace {

// Build a minimal 24-bit uncompressed BMP (BITMAPINFOHEADER) from RGB rows.
// rows[y][x] = {r,g,b}, row 0 = TOP row (the builder flips to BMP's
// bottom-up storage). Row stride is padded to 4 bytes per the format.
std::vector<uint8_t> make_bmp24(const std::vector<std::vector<std::array<uint8_t, 3>>>& rows) {
    int h = static_cast<int>(rows.size());
    int w = static_cast<int>(rows[0].size());
    int stride = ((w * 3 + 3) / 4) * 4;
    int data_size = stride * h;
    int file_size = 54 + data_size;

    std::vector<uint8_t> b(static_cast<size_t>(file_size), 0);
    auto put16 = [&](size_t off, uint32_t v) {
        b[off] = v & 0xFF;
        b[off + 1] = (v >> 8) & 0xFF;
    };
    auto put32 = [&](size_t off, uint32_t v) {
        for (int i = 0; i < 4; i++)
            b[off + i] = (v >> (8 * i)) & 0xFF;
    };
    b[0] = 'B';
    b[1] = 'M';
    put32(2, file_size);
    put32(10, 54);  // pixel data offset
    put32(14, 40);  // BITMAPINFOHEADER size
    put32(18, static_cast<uint32_t>(w));
    put32(22, static_cast<uint32_t>(h));  // positive = bottom-up
    put16(26, 1);                         // planes
    put16(28, 24);                        // bpp
    put32(34, static_cast<uint32_t>(data_size));
    for (int y = 0; y < h; y++) {
        // BMP stores bottom-up; rows[0] is the top row.
        int src_y = h - 1 - y;
        for (int x = 0; x < w; x++) {
            size_t off = 54 + static_cast<size_t>(y) * stride + static_cast<size_t>(x) * 3;
            // BMP byte order is BGR.
            b[off] = rows[src_y][x][2];
            b[off + 1] = rows[src_y][x][1];
            b[off + 2] = rows[src_y][x][0];
        }
    }
    return b;
}

float px(const ImageData& img, int c, int y, int x) {
    size_t idx = static_cast<size_t>(c) * img.height * img.width + static_cast<size_t>(y) * img.width + x;
    return __half2float(img.pixels[idx]);
}

// Reference normalization: stb gives value/255, then (v - mean[c]) / std[c].
float norm_ref(uint8_t v, float mean, float stdv) { return (v / 255.0f - mean) / stdv; }

TEST(VisionPreprocess, ChwLayoutChannelOrderAndNormalization) {
    // 2x2 image with per-corner primary colors: layout errors (HWC-vs-CHW,
    // RGB-vs-BGR, row flip) each produce a distinct wrong tensor.
    //   top-left  RED    top-right  GREEN
    //   bot-left  BLUE   bot-right  WHITE
    auto bmp = make_bmp24({
        {{{255, 0, 0}}, {{0, 255, 0}}},
        {{{0, 0, 255}}, {{255, 255, 255}}},
    });
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float stdv[3] = {0.5f, 0.5f, 0.5f};

    ImageData img;
    ASSERT_TRUE(load_and_preprocess_image_from_memory(bmp, 2, mean, stdv, img));
    ASSERT_EQ(img.width, 2);
    ASSERT_EQ(img.height, 2);
    ASSERT_EQ(img.pixels.size(), 3u * 2 * 2);

    const float hi = norm_ref(255, 0.5f, 0.5f);  // +1.0
    const float lo = norm_ref(0, 0.5f, 0.5f);    // -1.0
    const float tol = 2e-3f;                     // one f16 step at |1.0| is ~5e-4

    // R channel
    EXPECT_NEAR(px(img, 0, 0, 0), hi, tol);  // red corner
    EXPECT_NEAR(px(img, 0, 0, 1), lo, tol);  // green corner
    EXPECT_NEAR(px(img, 0, 1, 0), lo, tol);  // blue corner
    EXPECT_NEAR(px(img, 0, 1, 1), hi, tol);  // white corner
    // G channel
    EXPECT_NEAR(px(img, 1, 0, 0), lo, tol);
    EXPECT_NEAR(px(img, 1, 0, 1), hi, tol);
    EXPECT_NEAR(px(img, 1, 1, 0), lo, tol);
    EXPECT_NEAR(px(img, 1, 1, 1), hi, tol);
    // B channel
    EXPECT_NEAR(px(img, 2, 0, 0), lo, tol);
    EXPECT_NEAR(px(img, 2, 0, 1), lo, tol);
    EXPECT_NEAR(px(img, 2, 1, 0), hi, tol);
    EXPECT_NEAR(px(img, 2, 1, 1), hi, tol);
}

TEST(VisionPreprocess, PerChannelMeanStdApplied) {
    // Uniform mid-gray input; distinct per-channel mean/std (SigLIP-style
    // configs use per-channel constants) must produce distinct channel values.
    auto bmp = make_bmp24({
        {{{128, 128, 128}}, {{128, 128, 128}}},
        {{{128, 128, 128}}, {{128, 128, 128}}},
    });
    const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};   // CLIP mean
    const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};  // CLIP std

    ImageData img;
    ASSERT_TRUE(load_and_preprocess_image_from_memory(bmp, 2, mean, stdv, img));
    for (int c = 0; c < 3; c++) {
        float expect = norm_ref(128, mean[c], stdv[c]);
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                EXPECT_NEAR(px(img, c, y, x), expect, 3e-3f) << "c=" << c << " y=" << y << " x=" << x;
    }
}

TEST(VisionPreprocess, ResizeToTargetSquare) {
    // 2x2 uniform image upscaled to 8x8: dimensions/buffer must follow
    // target_size and a uniform source must stay uniform after resampling.
    auto bmp = make_bmp24({
        {{{200, 100, 50}}, {{200, 100, 50}}},
        {{{200, 100, 50}}, {{200, 100, 50}}},
    });
    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float stdv[3] = {1.0f, 1.0f, 1.0f};

    ImageData img;
    ASSERT_TRUE(load_and_preprocess_image_from_memory(bmp, 8, mean, stdv, img));
    ASSERT_EQ(img.width, 8);
    ASSERT_EQ(img.height, 8);
    ASSERT_EQ(img.pixels.size(), 3u * 8 * 8);

    const float want[3] = {200 / 255.0f, 100 / 255.0f, 50 / 255.0f};
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                EXPECT_NEAR(px(img, c, y, x), want[c], 5e-3f) << "c=" << c << " y=" << y << " x=" << x;
}

TEST(VisionPreprocess, NonSquareInputResizedToSquare) {
    // 4x2 (landscape) → target 4: output must be exactly 4x4 (the model
    // expects square patches; aspect handling is resize-to-square).
    std::vector<std::vector<std::array<uint8_t, 3>>> rows(
        2, std::vector<std::array<uint8_t, 3>>(4, {{10, 20, 30}}));
    auto bmp = make_bmp24(rows);
    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float stdv[3] = {1.0f, 1.0f, 1.0f};

    ImageData img;
    ASSERT_TRUE(load_and_preprocess_image_from_memory(bmp, 4, mean, stdv, img));
    EXPECT_EQ(img.width, 4);
    EXPECT_EQ(img.height, 4);
    EXPECT_EQ(img.pixels.size(), 3u * 4 * 4);
}

TEST(VisionPreprocess, GarbageInputFailsCleanly) {
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float stdv[3] = {0.5f, 0.5f, 0.5f};
    ImageData img;

    std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33};
    EXPECT_FALSE(load_and_preprocess_image_from_memory(garbage, 4, mean, stdv, img));

    // Truncated BMP header: magic ok, data missing.
    std::vector<uint8_t> truncated = {'B', 'M', 0x36, 0x00, 0x00, 0x00};
    EXPECT_FALSE(load_and_preprocess_image_from_memory(truncated, 4, mean, stdv, img));
}

TEST(VisionPreprocess, FileEntryPointMatchesMemoryPath) {
    auto bmp = make_bmp24({
        {{{255, 0, 0}}, {{0, 255, 0}}},
        {{{0, 0, 255}}, {{255, 255, 255}}},
    });
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float stdv[3] = {0.5f, 0.5f, 0.5f};

    std::string path = testing::TempDir() + "imp_vision_preprocess_probe.bmp";
    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    fwrite(bmp.data(), 1, bmp.size(), f);
    fclose(f);

    ImageData from_file, from_mem;
    ASSERT_TRUE(load_and_preprocess_image(path, 2, mean, stdv, from_file));
    ASSERT_TRUE(load_and_preprocess_image_from_memory(bmp, 2, mean, stdv, from_mem));
    ASSERT_EQ(from_file.pixels.size(), from_mem.pixels.size());
    for (size_t i = 0; i < from_file.pixels.size(); i++)
        EXPECT_EQ(__half2float(from_file.pixels[i]), __half2float(from_mem.pixels[i])) << "i=" << i;

    remove(path.c_str());

    EXPECT_FALSE(load_and_preprocess_image("/nonexistent/imp_no_such_image.bmp", 2, mean, stdv, from_file));
}

// ---------------------------------------------------------------------------
// qwen_smart_resize — the dynamic-resolution target size (Qwen3-VL).
// Oracle: transformers' smart_resize (qwen2_vl image processing), reproduced by
// hand below for each case. factor 32 = patch_size 16 * merge_size 2; the pixel
// bounds are the staged Qwen3-VL-4B preprocessor_config.json (65536 / 16777216).
// ---------------------------------------------------------------------------
constexpr int kFactor = 32;
constexpr int64_t kMinPx = 65536;     // 256^2
constexpr int64_t kMaxPx = 16777216;  // 4096^2

TEST(QwenSmartResize, AlreadyAlignedAndInRangeIsUnchanged) {
    // 640x480: both divisible by 32, 307200 px sits inside the bounds.
    auto r = qwen_smart_resize(480, 640, kFactor, kMinPx, kMaxPx);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.height, 480);
    EXPECT_EQ(r.width, 640);
}

TEST(QwenSmartResize, BelowMinPixelsScalesUp) {
    // 100x100 -> round(3.125)*32 = 96 each; 9216 < 65536, so the min branch
    // runs: beta = sqrt(65536/10000) = 2.56, ceil(100*2.56/32)*32 = 256.
    auto r = qwen_smart_resize(100, 100, kFactor, kMinPx, kMaxPx);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.height, 256);
    EXPECT_EQ(r.width, 256);
    EXPECT_GE(static_cast<int64_t>(r.height) * r.width, kMinPx);
}

TEST(QwenSmartResize, AboveMaxPixelsScalesDown) {
    // 8000x8000 -> 64e6 px > max; beta = sqrt(64e6/16777216), which lands
    // exactly on 4096x4096 = max_pixels.
    auto r = qwen_smart_resize(8000, 8000, kFactor, kMinPx, kMaxPx);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.height, 4096);
    EXPECT_EQ(r.width, 4096);
    EXPECT_LE(static_cast<int64_t>(r.height) * r.width, kMaxPx);
}

// The one case where Python's round() and std::round() disagree and it CHANGES
// THE RESULT, not just an intermediate. 16/32 is exactly 0.5: ties-to-even
// gives 0, ties-away gives 1.
//   ties-to-even (correct): h_bar 0 -> product 0 < min -> min branch -> 32x2912
//   ties-away  (wrong):     h_bar 32 -> product 65536 == min, no branch -> 32x2048
// A silent one-step difference here changes the image's token count.
TEST(QwenSmartResize, TiesRoundToEvenLikePython) {
    auto r = qwen_smart_resize(16, 2048, kFactor, kMinPx, kMaxPx);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.height, 32);
    EXPECT_EQ(r.width, 2912) << "ties-away rounding would give 2048 here";
}

TEST(QwenSmartResize, RejectsExtremeAspectRatio) {
    // Upstream raises above 200:1; imp reports it instead of returning a size.
    auto r = qwen_smart_resize(1, 300, kFactor, kMinPx, kMaxPx);
    EXPECT_FALSE(r.ok);
}

TEST(QwenSmartResize, RejectsDegenerateInput) {
    EXPECT_FALSE(qwen_smart_resize(0, 100, kFactor, kMinPx, kMaxPx).ok);
    EXPECT_FALSE(qwen_smart_resize(100, 100, 0, kMinPx, kMaxPx).ok);
}

TEST(QwenSmartResize, OutputIsAlwaysFactorAligned) {
    for (int h : {37, 100, 255, 512, 1023, 4000})
        for (int w : {41, 100, 257, 640, 1919, 5000}) {
            auto r = qwen_smart_resize(h, w, kFactor, kMinPx, kMaxPx);
            ASSERT_TRUE(r.ok) << h << "x" << w;
            EXPECT_EQ(r.height % kFactor, 0) << h << "x" << w;
            EXPECT_EQ(r.width % kFactor, 0) << h << "x" << w;
        }
}

// ---------------------------------------------------------------------------
// qwen_patchify — layout, not values. The two orderings below are the ones the
// encoder silently depends on and that guessing gets wrong:
//   - tokens are grouped by 2x2 MERGE BLOCK, not raster;
//   - inside a token the layout is (C, T, ph, pw), and T is a REPEAT.
// Oracle: the reshape/permute in Qwen2VLImageProcessorFast.
// ---------------------------------------------------------------------------

// An RGB image where every patch-sized tile carries a distinct constant, so a
// token's content identifies which patch it came from.
std::vector<uint8_t> patch_id_image(int side, int patch) {
    std::vector<uint8_t> img(static_cast<size_t>(side) * side * 3);
    const int g = side / patch;
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            const int pid = (y / patch) * g + (x / patch);
            for (int c = 0; c < 3; ++c)
                img[(static_cast<size_t>(y) * side + x) * 3 + c] = static_cast<uint8_t>(pid * 16 + c);
        }
    return img;
}

QwenPatchifyConfig patchify_test_cfg(int side) {
    QwenPatchifyConfig c;
    // Pin the geometry so smart_resize is a no-op and the test controls the grid.
    c.min_pixels = 1;
    c.max_pixels = static_cast<int64_t>(side) * side;
    return c;
}

TEST(QwenPatchify, GridAndShape) {
    const int side = 64, P = 16;
    auto img = patch_id_image(side, P);
    QwenPatches out;
    ASSERT_TRUE(qwen_patchify(img.data(), side, side, patchify_test_cfg(side), out));
    EXPECT_EQ(out.grid_h, 4);
    EXPECT_EQ(out.grid_w, 4);
    EXPECT_EQ(out.tokens, 16);
    EXPECT_EQ(out.features, 3 * 2 * P * P);  // C*T*P*P = 1536
    EXPECT_EQ(out.data.size(), static_cast<size_t>(out.tokens) * out.features);
}

// Token k must be the k-th patch in MERGE-BLOCK order, not raster order. For a
// 4x4 patch grid with merge 2 the block order is:
//   block(0,0): patches 0,1,4,5    block(0,1): patches 2,3,6,7
//   block(1,0): patches 8,9,12,13  block(1,1): patches 10,11,14,15
TEST(QwenPatchify, TokensAreGroupedByMergeBlockNotRaster) {
    const int side = 64, P = 16;
    auto img = patch_id_image(side, P);
    QwenPatches out;
    ASSERT_TRUE(qwen_patchify(img.data(), side, side, patchify_test_cfg(side), out));

    const int expect[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
    for (int t = 0; t < 16; ++t) {
        // channel 0, temporal 0, first pixel of the patch.
        const float v = __half2float(out.data[static_cast<size_t>(t) * out.features]);
        const float want = (expect[t] * 16) / 255.0f * 2.0f - 1.0f;  // mean/std 0.5
        EXPECT_NEAR(v, want, 0.02f) << "token " << t << " came from the wrong patch";
    }
}

TEST(QwenPatchify, ChannelMajorAndTemporalIsARepeat) {
    const int side = 64, P = 16, T = 2;
    auto img = patch_id_image(side, P);
    QwenPatches out;
    ASSERT_TRUE(qwen_patchify(img.data(), side, side, patchify_test_cfg(side), out));

    const half* tok0 = out.data.data();
    for (int c = 0; c < 3; ++c) {
        const size_t base = (static_cast<size_t>(c) * T + 0) * P * P;
        const size_t rep = (static_cast<size_t>(c) * T + 1) * P * P;
        // The temporal axis repeats the same spatial patch...
        for (int i = 0; i < P * P; ++i)
            EXPECT_EQ(__half2float(tok0[base + i]), __half2float(tok0[rep + i]))
                << "temporal slot 1 must repeat slot 0 (channel " << c << ", elem " << i << ")";
        // ...and channels stay distinct, so the layout really is channel-major.
        if (c > 0) {
            const size_t prev = (static_cast<size_t>(c - 1) * T + 0) * P * P;
            EXPECT_NE(__half2float(tok0[base]), __half2float(tok0[prev]))
                << "channels " << c - 1 << " and " << c << " must not alias";
        }
    }
}

TEST(QwenPatchify, RejectsBadInput) {
    QwenPatches out;
    QwenPatchifyConfig c = patchify_test_cfg(64);
    std::vector<uint8_t> img(64 * 64 * 3, 0);
    EXPECT_FALSE(qwen_patchify(nullptr, 64, 64, c, out));
    EXPECT_FALSE(qwen_patchify(img.data(), 0, 64, c, out));
    c.patch_size = 0;
    EXPECT_FALSE(qwen_patchify(img.data(), 64, 64, c, out));
}

}  // namespace
}  // namespace imp
