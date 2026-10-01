// InternVL3.5 preprocessing and prompt layout vs the pinned HF processor
// (tests/fixtures/internvl, tools/internvl_fixture/run_preprocess.sh; crop_to_patches false).

#include "model/image_placeholders.h"
#include "vision/image_processor.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

const std::string kDir = std::string(IMP_TEST_FIXTURES_DIR) + "/internvl";

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::unordered_map<std::string, std::vector<int32_t>> read_ids(const std::string& path) {
    std::unordered_map<std::string, std::vector<int32_t>> kv;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string key, v;
        ss >> key;
        while (ss >> v)
            kv[key].push_back(static_cast<int32_t>(std::stol(v)));
    }
    return kv;
}

// Resized tile vs HF's within one u8 step (1/255), and the patches are that tile normalised.
TEST(InternVLPreprocess, TileMatchesHfWithinOneStep) {
    const auto png = read_bytes(kDir + "/synth_300x200.png");
    const auto want = read_bytes(kDir + "/pixels_448.u8");  // [3, 448, 448] CHW
    ASSERT_EQ(want.size(), 3u * 448 * 448) << "missing fixture " << kDir;
    std::vector<uint8_t> rgb;
    int w = 0, h = 0;
    ASSERT_TRUE(decode_rgb(png, rgb, w, h));
    ASSERT_EQ(w, 200);
    ASSERT_EQ(h, 300);

    InternVLPreprocessConfig cfg;
    std::vector<half> patches;
    std::vector<uint8_t> tile;
    ASSERT_TRUE(internvl_preprocess(rgb.data(), w, h, cfg, patches, &tile));
    ASSERT_EQ(patches.size(), 1024u * 588);

    int max_step = 0;
    size_t over = 0;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 448; ++y)
            for (int x = 0; x < 448; ++x) {
                const int d = std::abs(int(tile[(static_cast<size_t>(y) * 448 + x) * 3 + c]) -
                                       int(want[(static_cast<size_t>(c) * 448 + y) * 448 + x]));
                max_step = std::max(max_step, d);
                over += d > 1;
            }
    std::printf("[internvl-preprocess] max |imp - hf| = %d/255, %zu of %d values over 1/255\n", max_step,
                over, 3 * 448 * 448);
    EXPECT_LE(max_step, 1);

    // Patch (row 1, col 2), channel 2, pixel (3, 5) = tile(14 + 3, 28 + 5), normalised.
    const size_t pi = 1 * 32 + 2;
    const float got = __half2float(patches[pi * 588 + (2 * 14 + 3) * 14 + 5]);
    const float v = (tile[(static_cast<size_t>(17) * 448 + 33) * 3 + 2] / 255.0f - cfg.mean[2]) / cfg.std[2];
    EXPECT_NEAR(got, v, 2e-3);
}

TEST(InternVLPreprocess, PromptIdsMatchHfProcessor) {
    auto kv = read_ids(kDir + "/prompt.txt");
    ASSERT_EQ(kv["pad_ids"].size(), 3u) << "missing fixture " << kDir;
    std::vector<int32_t> tokens = kv["prompt_ids"];
    const int32_t ctx = kv["pad_ids"][0], start = kv["pad_ids"][1], end = kv["pad_ids"][2];
    ASSERT_TRUE(expand_internvl_image_placeholders(tokens, ctx, start, end, 1, 256));
    EXPECT_EQ(tokens, kv["input_ids"]);
    EXPECT_EQ(tokens.size(), 271u);

    std::vector<int32_t> two = kv["prompt_ids"];
    const auto orig = two;
    EXPECT_FALSE(expand_internvl_image_placeholders(two, ctx, start, end, 2, 256))
        << "count mismatch refused";
    EXPECT_EQ(two, orig);
}

}  // namespace
}  // namespace imp
