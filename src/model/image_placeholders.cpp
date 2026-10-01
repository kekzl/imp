#include "model/image_placeholders.h"

#include <algorithm>
#include <cstdio>

namespace imp {

std::expected<void, std::string> expand_image_placeholders(std::vector<int32_t>& tokens, int32_t pad_id,
                                                           const std::vector<int>& counts) {
    size_t found = 0;
    for (int32_t t : tokens)
        if (t == pad_id)
            ++found;
    if (found != counts.size())
        return std::unexpected("prompt holds " + std::to_string(found) + " image placeholder(s) but " +
                               std::to_string(counts.size()) + " image(s) were encoded");
    for (size_t k = 0; k < counts.size(); ++k)
        if (counts[k] <= 0)
            return std::unexpected("image " + std::to_string(k) + " produced no tokens");
    if (found == 0)
        return {};

    size_t total = tokens.size();
    for (int c : counts)
        total += static_cast<size_t>(c) - 1;

    std::vector<int32_t> out;
    out.reserve(total);
    size_t k = 0;
    for (int32_t t : tokens) {
        if (t != pad_id) {
            out.push_back(t);
            continue;
        }
        out.insert(out.end(), static_cast<size_t>(counts[k]), pad_id);
        ++k;
    }
    tokens = std::move(out);
    return {};
}

std::expected<void, std::string> expand_video_placeholders(
    std::vector<int32_t>& tokens, int32_t video_pad_id, int32_t vision_start_id, int32_t vision_end_id,
    const std::vector<VideoPlaceholderLayout>& videos) {
    const size_t found = static_cast<size_t>(std::count(tokens.begin(), tokens.end(), video_pad_id));
    if (found != videos.size())
        return std::unexpected("prompt holds " + std::to_string(found) + " video placeholder(s) but " +
                               std::to_string(videos.size()) + " video(s) were encoded");
    for (size_t k = 0; k < videos.size(); ++k)
        if (videos[k].tokens_per_group <= 0 || videos[k].stamp_ids.empty())
            return std::unexpected("video " + std::to_string(k) + " produced no tokens");
    if (found == 0)
        return {};

    std::vector<int32_t> out;
    out.reserve(tokens.size());
    size_t k = 0;
    for (int32_t t : tokens) {
        if (t != video_pad_id) {
            out.push_back(t);
            continue;
        }
        const VideoPlaceholderLayout& v = videos[k++];
        for (const auto& stamp : v.stamp_ids) {
            out.insert(out.end(), stamp.begin(), stamp.end());
            out.push_back(vision_start_id);
            out.insert(out.end(), static_cast<size_t>(v.tokens_per_group), video_pad_id);
            out.push_back(vision_end_id);
        }
    }
    tokens = std::move(out);
    return {};
}

std::vector<double> qwen_video_group_seconds(std::span<const double> frame_seconds, int temporal_patch_size) {
    std::vector<double> out;
    if (frame_seconds.empty() || temporal_patch_size <= 0)
        return out;
    const size_t T = static_cast<size_t>(temporal_patch_size);
    const size_t n = frame_seconds.size();
    for (size_t i = 0; i < n; i += T)
        out.push_back((frame_seconds[i] + frame_seconds[std::min(i + T - 1, n - 1)]) / 2);
    return out;
}

std::string qwen_video_timestamp_text(double seconds) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "<%.1f seconds>", seconds);
    return buf;
}

size_t image_content_hash(std::span<const uint8_t> data) {
    size_t h = 0xcbf29ce484222325ULL;
    for (const uint8_t b : data) {
        h ^= b;
        h *= 0x100000001b3ULL;
    }
    return h ? h : 1;  // 0 is the cache's "no image" sentinel
}

size_t combine_image_hash(size_t running, size_t next) {
    if (running == 0)
        return next;
    // FNV-style mix, so swapping two images changes the result. Never returns
    // 0: that value means "no image" to the prefix cache.
    const size_t h = (running * 0x100000001b3ULL) ^ next;
    return h ? h : 1;
}

int image_tokens_before(const std::vector<int32_t>& tokens, int32_t pad_id, int upto) {
    if (upto <= 0)
        return 0;
    const size_t end = std::min(static_cast<size_t>(upto), tokens.size());
    int n = 0;
    for (size_t i = 0; i < end; ++i)
        n += (tokens[i] == pad_id);
    return n;
}

}  // namespace imp
