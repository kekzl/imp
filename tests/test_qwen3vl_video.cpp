// Qwen3-VL video, CPU half: frame-pair patchify, video smart_resize, placeholder layout and
// M-RoPE positions against fixtures from transformers==5.17.0 (tools/qwen3vl_video_fixture/run.sh).

#include <gtest/gtest.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "model/image_placeholders.h"
#include "model/mrope_positions.h"
#include "vision/image_processor.h"

namespace imp {
namespace {

const std::string kDir = std::string(IMP_TEST_FIXTURES_DIR) + "/qwen3vl_video";

// Must match synth_frame() in tools/qwen3vl_video_fixture/gen.py bit for bit.
std::vector<uint8_t> synth_frame(int f, int h, int w) {
    std::vector<uint8_t> px(static_cast<size_t>(h) * w * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                const uint64_t v = static_cast<uint64_t>(f * 131 + y * 31 + x * 17 + c * 7 + 1) *
                                   2654435761ULL;
                px[(static_cast<size_t>(y) * w + x) * 3 + c] = static_cast<uint8_t>(
                    ((v & 0xFFFFFFFFULL) >> 13) & 0xFF);
            }
    return px;
}

// "key v0 v1 ..." per line; repeated keys (stamp, resize) keep every line.
std::multimap<std::string, std::vector<std::string>> read_kv(const std::string& path) {
    std::multimap<std::string, std::vector<std::string>> kv;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string key, v;
        ss >> key;
        std::vector<std::string> vals;
        while (ss >> v)
            vals.push_back(v);
        if (!key.empty())
            kv.emplace(key, std::move(vals));
    }
    return kv;
}

std::vector<int32_t> ints(const std::vector<std::string>& s) {
    std::vector<int32_t> out;
    for (const auto& v : s)
        out.push_back(static_cast<int32_t>(std::stol(v)));
    return out;
}

const std::vector<std::string>& one(const std::multimap<std::string, std::vector<std::string>>& kv,
                                    const std::string& key) {
    static const std::vector<std::string> empty;
    auto it = kv.find(key);
    return it == kv.end() ? empty : it->second;
}

std::vector<float> read_f32(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<size_t>(in.tellg());
    std::vector<float> out(bytes / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return out;
}

// Spacing of FP16 values at |ref|: 2^(e-10) for normals, 2^-24 below 2^-14.
float fp16_ulp(float ref) {
    const float a = std::fabs(ref);
    if (a < 6.103515625e-05f)
        return 5.9604644775390625e-08f;
    return std::ldexp(1.0f, std::ilogb(a) - 10);
}

struct UlpReport {
    size_t over = 0;  // elements more than 1 ulp off
    float max_abs = 0.0f;
};

UlpReport compare_video(const std::vector<QwenPatches>& groups, const std::vector<float>& ref) {
    UlpReport r;
    size_t k = 0;
    for (const auto& g : groups)
        for (const half h : g.data) {
            const float d = std::fabs(__half2float(h) - ref[k]);
            r.max_abs = std::max(r.max_abs, d);
            r.over += (d > fp16_ulp(ref[k]));
            ++k;
        }
    return r;
}

std::vector<const uint8_t*> ptrs(const std::vector<std::vector<uint8_t>>& frames) {
    std::vector<const uint8_t*> p;
    for (const auto& f : frames)
        p.push_back(f.data());
    return p;
}

// --- V1: frame-pair patchify -------------------------------------------------------------

TEST(QwenVideoPatchify, MatchesHfVideoProcessorWithinOneFp16Ulp) {
    const auto meta = read_kv(kDir + "/v1_meta.txt");
    const auto ref = read_f32(kDir + "/v1_pixels.f32");
    const auto thw = ints(one(meta, "grid_thw"));
    ASSERT_EQ(thw.size(), 3u) << "missing fixture " << kDir;
    ASSERT_EQ(ref.size(), static_cast<size_t>(thw[0] * thw[1] * thw[2]) * 1536);

    std::vector<std::vector<uint8_t>> frames;
    for (int f = 0; f < 4; ++f)
        frames.push_back(synth_frame(f, 32, 64));
    const auto p = ptrs(frames);
    std::vector<QwenPatches> groups;
    ASSERT_TRUE(qwen_patchify_video(p, 64, 32, qwen_video_patchify_config(), groups));

    ASSERT_EQ(static_cast<int>(groups.size()), thw[0]);
    for (const auto& g : groups) {
        EXPECT_EQ(g.grid_h, thw[1]);
        EXPECT_EQ(g.grid_w, thw[2]);
        EXPECT_EQ(g.features, 1536);
    }
    const UlpReport r = compare_video(groups, ref);
    EXPECT_EQ(r.over, 0u) << "elements more than 1 FP16 ulp from HF, max abs diff " << r.max_abs;
    RecordProperty("max_abs_diff", std::to_string(r.max_abs));
}

// Sensitivity of the fixture: a swapped pair must not pass, or the test above proves nothing
// about the temporal axis.
TEST(QwenVideoPatchify, FixtureRejectsFramesSwappedWithinAPair) {
    const auto ref = read_f32(kDir + "/v1_pixels.f32");
    std::vector<std::vector<uint8_t>> frames;
    for (int f : {1, 0, 2, 3})
        frames.push_back(synth_frame(f, 32, 64));
    const auto p = ptrs(frames);
    std::vector<QwenPatches> groups;
    ASSERT_TRUE(qwen_patchify_video(p, 64, 32, qwen_video_patchify_config(), groups));
    ASSERT_EQ(groups.size(), 2u);
    const UlpReport r = compare_video(groups, ref);
    EXPECT_GT(r.over, ref.size() / 4) << "swapped frames 0/1 still match the HF fixture";
}

TEST(QwenVideoPatchify, OddFrameCountRepeatsTheLastFrame) {
    std::vector<std::vector<uint8_t>> frames;
    for (int f = 0; f < 3; ++f)
        frames.push_back(synth_frame(f, 32, 64));
    const auto p = ptrs(frames);
    std::vector<QwenPatches> groups;
    ASSERT_TRUE(qwen_patchify_video(p, 64, 32, qwen_video_patchify_config(), groups));
    ASSERT_EQ(groups.size(), 2u);
    const int PP = 16 * 16, T = 2;
    const half* tok = groups[1].data.data();
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < PP; ++i)
            ASSERT_EQ(__half2float(tok[(c * T + 0) * PP + i]), __half2float(tok[(c * T + 1) * PP + i]))
                << "padded slot must repeat frame 2";
}

TEST(QwenVideoPatchify, RejectsBadInput) {
    std::vector<QwenPatches> groups;
    const auto f = synth_frame(0, 32, 64);
    const std::vector<const uint8_t*> single{f.data()};
    const std::vector<const uint8_t*> with_null{f.data(), nullptr};
    EXPECT_FALSE(qwen_patchify_video({}, 64, 32, qwen_video_patchify_config(), groups));
    EXPECT_FALSE(qwen_patchify_video(single, 64, 32, qwen_video_patchify_config(), groups))
        << "one frame is below temporal_patch_size; upstream raises";
    EXPECT_FALSE(qwen_patchify_video(with_null, 64, 32, qwen_video_patchify_config(), groups));
}

TEST(QwenVideoSmartResize, MatchesHfSmartResize) {
    const auto meta = read_kv(kDir + "/v1_meta.txt");
    const auto cfg = qwen_video_patchify_config();
    int cases = 0;
    for (auto [it, end] = meta.equal_range("resize"); it != end; ++it, ++cases) {
        const auto v = ints(it->second);
        ASSERT_EQ(v.size(), 5u);
        const SmartResize rs = qwen_video_smart_resize(v[0], v[1], v[2], 2, 32, cfg.min_pixels,
                                                       cfg.max_pixels);
        ASSERT_TRUE(rs.ok) << v[0] << "x" << v[1] << "x" << v[2];
        EXPECT_EQ(rs.height, v[3]) << "t=" << v[0] << " h=" << v[1] << " w=" << v[2];
        EXPECT_EQ(rs.width, v[4]) << "t=" << v[0] << " h=" << v[1] << " w=" << v[2];
    }
    EXPECT_EQ(cases, 10);
}

// --- V2: placeholder layout and M-RoPE ---------------------------------------------------

struct V2 {
    std::multimap<std::string, std::vector<std::string>> kv;
    int32_t image_pad = 0, video_pad = 0, vstart = 0, vend = 0;
    std::vector<int32_t> expanded;  // imp's expansion of the HF prompt ids
    std::vector<MRopeImageGrid> images;
    std::vector<MRopeVideoGrid> videos;
};

V2 build_v2() {
    V2 v;
    v.kv = read_kv(kDir + "/v2_layout.txt");
    const auto pads = ints(one(v.kv, "pad_ids"));
    if (pads.size() != 4)
        return v;
    v.image_pad = pads[0];
    v.video_pad = pads[1];
    v.vstart = pads[2];
    v.vend = pads[3];

    // Grids from imp's own resize, not the fixture's: the layout must follow from the pixels.
    const auto ihw = ints(one(v.kv, "image_hw"));
    const QwenPatchifyConfig icfg;
    const SmartResize irs = qwen_smart_resize(ihw[0], ihw[1], 32, icfg.min_pixels, icfg.max_pixels);
    v.images.push_back({irs.height / 32, irs.width / 32});
    const auto vthw = ints(one(v.kv, "video_thw"));
    const auto vcfg = qwen_video_patchify_config();
    const SmartResize vrs = qwen_video_smart_resize(vthw[0], vthw[1], vthw[2], 2, 32, vcfg.min_pixels,
                                                    vcfg.max_pixels);
    v.videos.push_back({(vthw[0] + 1) / 2, vrs.height / 32, vrs.width / 32});

    std::vector<double> secs;
    for (const auto& s : one(v.kv, "frame_seconds"))
        secs.push_back(std::strtod(s.c_str(), nullptr));
    std::map<std::string, std::vector<int32_t>> stamp_ids;
    for (auto [it, end] = v.kv.equal_range("stamp"); it != end; ++it) {
        std::string label = it->second.at(0);
        std::replace(label.begin(), label.end(), '_', ' ');
        stamp_ids[label] = ints(std::vector<std::string>(it->second.begin() + 1, it->second.end()));
    }
    VideoPlaceholderLayout layout;
    layout.tokens_per_group = v.videos[0].rows * v.videos[0].cols;
    for (double s : qwen_video_group_seconds(secs, 2)) {
        const auto found = stamp_ids.find(qwen_video_timestamp_text(s));
        // An unknown label leaves an empty stamp, so the sequence comparison fails loudly.
        layout.stamp_ids.push_back(found == stamp_ids.end() ? std::vector<int32_t>{} : found->second);
    }

    v.expanded = ints(one(v.kv, "prompt_ids"));
    if (!expand_image_placeholders(v.expanded, v.image_pad, {v.images[0].tokens()}))
        v.expanded.clear();
    else if (!expand_video_placeholders(v.expanded, v.video_pad, v.vstart, v.vend, {layout}))
        v.expanded.clear();
    return v;
}

TEST(QwenVideoLayout, GridsMatchHfProcessor) {
    const V2 v = build_v2();
    const auto ig = ints(one(v.kv, "image_grid_thw"));
    const auto vg = ints(one(v.kv, "video_grid_thw"));
    ASSERT_EQ(ig.size(), 3u) << "missing fixture " << kDir;
    ASSERT_EQ(vg.size(), 3u);
    EXPECT_EQ(v.images[0].rows * 2, ig[1]);
    EXPECT_EQ(v.images[0].cols * 2, ig[2]);
    EXPECT_EQ(v.videos[0].groups, vg[0]);
    EXPECT_EQ(v.videos[0].rows * 2, vg[1]);
    EXPECT_EQ(v.videos[0].cols * 2, vg[2]);
}

TEST(QwenVideoLayout, TimestampsFormatLikeHf) {
    const std::vector<double> secs{0.0, 7.0 / 30.0, 1.0, 1.5};
    const auto g = qwen_video_group_seconds(secs, 2);
    ASSERT_EQ(g.size(), 2u);
    EXPECT_EQ(qwen_video_timestamp_text(g[0]), "<0.1 seconds>");
    EXPECT_EQ(qwen_video_timestamp_text(g[1]), "<1.2 seconds>") << "1.25 rounds to even like Python";
    const std::vector<double> odd{0.0, 1.0, 2.0};
    const auto go = qwen_video_group_seconds(odd, 2);
    ASSERT_EQ(go.size(), 2u);
    EXPECT_EQ(go[1], 2.0) << "odd tail pads with the last frame";
}

TEST(QwenVideoLayout, PlaceholderIdsMatchHfProcessor) {
    const V2 v = build_v2();
    const auto want = ints(one(v.kv, "input_ids"));
    ASSERT_FALSE(want.empty()) << "missing fixture " << kDir;
    ASSERT_EQ(v.expanded.size(), want.size());
    size_t diffs = 0;
    for (size_t i = 0; i < want.size(); ++i)
        diffs += (v.expanded[i] != want[i]);
    EXPECT_EQ(diffs, 0u);
}

TEST(QwenVideoLayout, MRopePositionsMatchHfGetRopeIndex) {
    const V2 v = build_v2();
    const auto want_type = ints(one(v.kv, "mm_token_type_ids"));
    ASSERT_EQ(v.expanded.size(), want_type.size());
    std::vector<uint8_t> type(v.expanded.size(), kMRopeText);
    for (size_t i = 0; i < v.expanded.size(); ++i)
        type[i] = v.expanded[i] == v.image_pad   ? kMRopeImage
                  : v.expanded[i] == v.video_pad ? kMRopeVideo
                                                 : kMRopeText;
    for (size_t i = 0; i < type.size(); ++i)
        ASSERT_EQ(type[i], want_type[i]) << "token type at " << i;

    const auto r = qwen_build_mrope_positions_mm(type, v.images, v.videos, 0);
    ASSERT_TRUE(r) << r.error();
    const size_t n = type.size();
    size_t diffs = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const auto want = ints(one(v.kv, "pos" + std::to_string(axis)));
        ASSERT_EQ(want.size(), n);
        for (size_t i = 0; i < n; ++i)
            diffs += (r->pos[axis * n + i] != want[i]);
    }
    EXPECT_EQ(diffs, 0u);
    EXPECT_EQ(r->next_pos, std::stoi(one(v.kv, "next_pos").at(0)));
    EXPECT_EQ(r->next_pos - static_cast<int>(n), std::stoi(one(v.kv, "rope_delta").at(0)));
}

TEST(QwenVideoLayout, RefusesMismatchedVideoInput) {
    std::vector<int32_t> toks{1, 9, 2};
    const auto orig = toks;
    EXPECT_FALSE(expand_video_placeholders(toks, 9, 5, 6, {}));
    EXPECT_FALSE(expand_video_placeholders(toks, 9, 5, 6, {VideoPlaceholderLayout{0, {{7}}}}));
    EXPECT_FALSE(expand_video_placeholders(toks, 9, 5, 6, {VideoPlaceholderLayout{4, {}}}));
    EXPECT_EQ(toks, orig);

    // A 2-group video with one group's worth of tokens in the prompt.
    std::vector<uint8_t> type{kMRopeText, kMRopeVideo, kMRopeVideo, kMRopeText};
    EXPECT_FALSE(qwen_build_mrope_positions_mm(type, {}, {{2, 1, 2}}, 0));
    EXPECT_TRUE(qwen_build_mrope_positions_mm(type, {}, {{1, 1, 2}}, 0));
    EXPECT_FALSE(qwen_build_mrope_positions_mm(type, {}, {{0, 1, 2}}, 0));
}

}  // namespace
}  // namespace imp
