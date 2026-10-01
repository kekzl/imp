// decode_image() vs Pillow's Image.open(f).convert("RGB"): max abs diff 0/255 on every fixture in
// tests/fixtures/jpeg (baseline/progressive 4:2:0, 4:2:2, 4:4:4, gray, CMYK; #2381). stb was 3/255.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "vision/image_decode.h"

namespace imp {
namespace {

const std::string kDir = std::string(IMP_TEST_FIXTURES_DIR) + "/jpeg";

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

struct Ref {
    std::string name;
    int width = 0, height = 0;
};

std::vector<Ref> read_refs() {
    std::ifstream f(kDir + "/pillow_ref.txt");
    std::vector<Ref> refs;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream s(line);
        Ref r;
        s >> r.name >> r.width >> r.height;
        refs.push_back(r);
    }
    return refs;
}

TEST(ImageDecode, JpegMatchesPillowBitForBit) {
    const std::vector<Ref> refs = read_refs();
    ASSERT_EQ(refs.size(), 6u) << kDir << "/pillow_ref.txt";
    for (const Ref& r : refs) {
        const std::vector<uint8_t> jpg = read_bytes(kDir + "/" + r.name);
        const std::vector<uint8_t> ref = read_bytes(kDir + "/" + r.name.substr(0, r.name.size() - 4) +
                                                    ".rgb");
        ASSERT_FALSE(jpg.empty()) << r.name;
        ASSERT_EQ(ref.size(), static_cast<size_t>(r.width) * r.height * 3) << r.name;
        DecodedImage img;
        ASSERT_TRUE(decode_image(jpg, img)) << r.name;
        ASSERT_EQ(img.width, r.width) << r.name;
        ASSERT_EQ(img.height, r.height) << r.name;
        ASSERT_EQ(img.rgb.size(), ref.size()) << r.name;
        int max_diff = 0;
        for (size_t i = 0; i < ref.size(); ++i)
            max_diff = std::max(max_diff, std::abs(static_cast<int>(img.rgb[i]) - static_cast<int>(ref[i])));
        EXPECT_EQ(max_diff, 0) << r.name << ": max abs diff vs Pillow (/255)";
    }
}

TEST(ImageDecode, FileAndMemoryAgree) {
    DecodedImage a, b;
    ASSERT_TRUE(decode_image_file(kDir + "/progressive_420.jpg", a));
    ASSERT_TRUE(decode_image(read_bytes(kDir + "/progressive_420.jpg"), b));
    EXPECT_EQ(a.width, b.width);
    EXPECT_EQ(a.height, b.height);
    EXPECT_EQ(a.rgb, b.rgb);
    EXPECT_FALSE(decode_image_file(kDir + "/imp_no_such_image.jpg", a));
    EXPECT_TRUE(a.rgb.empty());
}

TEST(ImageDecode, PngStaysOnStb) {
    DecodedImage img;
    ASSERT_TRUE(decode_image_file(std::string(IMP_TEST_FIXTURES_DIR) + "/vision_test_64.png", img));
    EXPECT_EQ(img.width, 64);
    EXPECT_EQ(img.height, 64);
    EXPECT_EQ(img.rgb.size(), 64u * 64u * 3u);
}

TEST(ImageDecode, TruncatedJpegFails) {
    std::vector<uint8_t> jpg = read_bytes(kDir + "/baseline_444.jpg");
    ASSERT_GT(jpg.size(), 1000u);
    jpg.resize(jpg.size() / 2);
    DecodedImage img;
    EXPECT_FALSE(decode_image(jpg, img));
    EXPECT_TRUE(img.rgb.empty());
}

TEST(ImageDecode, CorruptJpegFailsWithoutExit) {
    std::vector<uint8_t> junk = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x02, 0x13, 0x37, 0x00, 0x00};
    DecodedImage img;
    EXPECT_FALSE(decode_image(junk, img));
    EXPECT_TRUE(img.rgb.empty());
}

// SOF0 patched to 20000 px wide: refused at the header, before any allocation.
TEST(ImageDecode, OversizeJpegRefused) {
    std::vector<uint8_t> jpg = read_bytes(kDir + "/baseline_420.jpg");
    size_t sof = 0;
    for (size_t i = 2; i + 9 < jpg.size(); ++i)
        if (jpg[i] == 0xFF && jpg[i + 1] == 0xC0) {
            sof = i;
            break;
        }
    ASSERT_NE(sof, 0u);
    jpg[sof + 7] = 20000 >> 8;
    jpg[sof + 8] = 20000 & 0xFF;
    DecodedImage img;
    EXPECT_FALSE(decode_image(jpg, img));
    EXPECT_TRUE(img.rgb.empty());
}

}  // namespace
}  // namespace imp
