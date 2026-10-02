// Vision prompt layout golden: every registered vision family x (1 image, 3 images, video,
// image+video+text), rendered on a toy vocab. Golden strings recorded on main a8ad2086.

#include "model/image_placeholders.h"
#include "vision/vision_family.h"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace imp {
namespace {

const std::map<std::string, int32_t, std::less<>> kSpecials = {
    {"<|image_pad|>", 151655},  {"<|video_pad|>", 151656}, {"<|vision_start|>", 151652},
    {"<|vision_end|>", 151653}, {"<IMG_CONTEXT>", 151671}, {"<img>", 151669},
    {"</img>", 151670},         {"<|im_start|>", 151644},  {"<|im_end|>", 151645},
};

int32_t toy_find(std::string_view s) {
    const auto it = kSpecials.find(s);
    return it == kSpecials.end() ? -1 : it->second;
}

std::vector<int32_t> toy_encode(std::string_view s) {
    std::vector<int32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        bool hit = false;
        for (const auto& [text, id] : kSpecials)
            if (s.substr(i, text.size()) == text) {
                out.push_back(id);
                i += text.size();
                hit = true;
                break;
            }
        if (!hit)
            out.push_back(static_cast<unsigned char>(s[i++]));
    }
    return out;
}

std::string toy_decode(const std::vector<int32_t>& ids) {
    std::string out;
    for (int32_t id : ids) {
        if (id < 256) {
            out += static_cast<char>(id);
            continue;
        }
        for (const auto& [text, sid] : kSpecials)
            if (sid == id)
                out += text;
    }
    return out;
}

// Verbatim copy of tools/imp-server/vision_parts.cpp on main a8ad2086 (qwen_vision_blocks,
// qwen_expand_vision_placeholders), tokenizer lookups through toy_find.
std::string main_blocks(const std::string& order, bool internvl) {
    std::string blocks;
    for (const char kind : order) {
        if (internvl)
            blocks += "<IMG_CONTEXT>\n";
        else
            blocks += kind == 'v' ? "<|vision_start|><|video_pad|><|vision_end|>"
                                  : "<|vision_start|><|image_pad|><|vision_end|>";
    }
    return blocks;
}

std::expected<void, std::string> main_expand(bool internvl, std::vector<int32_t>& tokens,
                                             const std::vector<int>& image_tokens,
                                             const std::vector<VideoPlaceholderLayout>& videos) {
    if (internvl) {
        const int per_image = image_tokens.empty() ? 0 : image_tokens[0];
        return expand_internvl_image_placeholders(tokens, toy_find("<IMG_CONTEXT>"), toy_find("<img>"),
                                                  toy_find("</img>"), static_cast<int>(image_tokens.size()),
                                                  per_image);
    }
    const int32_t pad_id = toy_find("<|image_pad|>");
    if (pad_id < 0)
        return std::unexpected(std::string("tokenizer has no <|image_pad|>"));
    auto expanded = expand_image_placeholders(tokens, pad_id, image_tokens);
    if (expanded && !videos.empty())
        expanded = expand_video_placeholders(tokens, toy_find("<|video_pad|>"), toy_find("<|vision_start|>"),
                                             toy_find("<|vision_end|>"), videos);
    return expanded;
}

struct Case {
    const char* name;
    std::string order;  // 'i' image, 'v' video, prompt order
    std::vector<int> image_tokens;
    int video_groups;  // frame pairs per video
};

const std::vector<Case> kCases = {
    {"one_image", "i", {4}, 0},
    {"three_images", "iii", {4, 6, 2}, 0},
    {"video", "v", {}, 2},
    {"image_video_text", "ivi", {3, 5}, 2},
};

// Front-end flow: blocks before the user text, template render, then expansion.
std::string render(VisionFamily family, const Case& c) {
    const bool internvl = family == VisionFamily::InternVL;
    if (internvl && c.order.find('v') != std::string::npos)
        return "error: video parts need a Qwen3-VL model; this InternVL model takes images only";
    std::vector<VideoPlaceholderLayout> videos;
    for (const char k : c.order)
        if (k == 'v') {
            std::vector<double> seconds;
            for (int f = 0; f < 2 * c.video_groups; ++f)
                seconds.push_back(f * 0.5);
            videos.push_back(
                qwen_video_layout(3, seconds, 2, [](const std::string& s) { return toy_encode(s); }));
        }
    std::vector<int32_t> tokens = toy_encode("<|im_start|>user\n" + main_blocks(c.order, internvl) +
                                             "Describe.<|im_end|>");
    const auto r = main_expand(internvl, tokens, c.image_tokens, videos);
    return r ? toy_decode(tokens) : "error: " + r.error();
}

struct Golden {
    const char* model_type;
    const char* case_name;
    const char* text;
};

const std::vector<Golden> kGolden = {
#include "vision_prompt_layout_golden.inc"
};

std::string c_escape(const std::string& s) {
    std::string out;
    for (const char ch : s) {
        if (ch == '\n')
            out += "\\n";
        else if (ch == '"' || ch == '\\')
            out += std::string("\\") + ch;
        else
            out += ch;
    }
    return out;
}

const std::vector<const char*> kModelTypes = {"qwen3_vl", "qwen3_5_moe", "qwen3_5", "internvl_vision"};

TEST(VisionPromptLayout, MatchesGoldenRecordedOnMain) {
    size_t checked = 0;
    for (const char* mt : kModelTypes)
        for (const Case& c : kCases) {
            const std::string got = render(vision_family_of(mt), c);
            std::printf("GOLDEN {\"%s\", \"%s\", \"%s\"},\n", mt, c.name, c_escape(got).c_str());
            for (const Golden& g : kGolden)
                if (std::string_view(g.model_type) == mt && std::string_view(g.case_name) == c.name) {
                    EXPECT_EQ(got, g.text) << mt << " " << c.name;
                    ++checked;
                }
        }
    EXPECT_EQ(checked, kModelTypes.size() * kCases.size());
}

}  // namespace
}  // namespace imp
