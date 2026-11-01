#pragma once

// EVS video token pruning (vLLM multimodal/video_prune/evs.py, PR #48912), host half.
// After the merger, token j of frame pair g > 0 scores 1 - cos(token j, same (h, w) token of
// pair g - 1); pair 0 scores 255. The top keep = max(tokens_per_group, int(total * (1 - q)))
// survive in their original order, all spliced into pair 0's prompt slot.

#include <cmath>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "model/mrope_positions.h"

namespace imp {

struct VideoPlaceholderLayout;

// One pruned video on a request; set by the server, read by the encode.
struct VideoEvsPlan {
    int first_item = 0;        // index of its pair 0 in Request::qwen_patches (encoded item order)
    int groups = 0;            // frame pairs
    int tokens_per_group = 0;  // merged rows * cols of one pair
    int keep = 0;              // retained tokens, < groups * tokens_per_group
    double rate = 0.0;         // pruning rate q the keep came from
};

// Score evs.py gives pair 0, so it always survives the top-k.
inline constexpr float kEvsFirstGroupScore = 255.0f;

// evs.py compute_retained_tokens_count: max(tokens_per_group, int(total * (1 - rate))), capped at
// total. 0 for an empty video.
int evs_retained_count(int groups, int tokens_per_group, double rate);

// evs.py compute_retention_mask on host scores: pair 0 forced to 255, stable descending sort
// (ties keep the lower index, NaN sorts first), top `keep` marked. dissim.size() = groups * tpg.
std::vector<uint8_t> evs_retention_mask(std::span<const float> dissim, int groups, int tokens_per_group,
                                        int keep);

// Ascending flat indices of the set entries of `mask`.
std::vector<int32_t> evs_kept_indices(std::span<const uint8_t> mask);

// True when the encoded items (token count each, prompt order) hold every plan: plans ascending
// and disjoint, each pair tokens_per_group tokens, 0 < keep < groups * tokens_per_group.
[[nodiscard]] bool evs_plans_match(std::span<const VideoEvsPlan> plans, std::span<const int> tokens_per_item);

// Rows of the full-size encoder output to keep, in prompt order: every row of an unpruned item,
// the kept[v] rows (indices within the video) of plans[v]. Plans must match (evs_plans_match).
std::vector<int32_t> evs_gather_rows(std::span<const VideoEvsPlan> plans,
                                     std::span<const int> tokens_per_item,
                                     std::span<const std::vector<int32_t>> kept);

// A float config rate as the decimal it was written as (0.1f -> 0.1), so keep matches Python's
// float64 arithmetic: 1000 tokens at 0.1f would keep 899, at 0.1 keeps 900.
inline double evs_rate_from_config(float rate) { return std::round(static_cast<double>(rate) * 1e6) / 1e6; }

// Prompt slot counts of a pruned video: {keep, 0, ..., 0}; timestamps and start/end stay.
void evs_prune_layout(VideoPlaceholderLayout& layout, int keep);

// M-RoPE for a prompt whose videos in `plans` are pruned. `token_type` / `tokens` are the pruned
// prompt (fold_video_pads), `items` one grid per encoded item as qwen_build_mrope_positions_items
// takes them, kept[v] = evs_kept_indices of plans[v]. Positions are those of the unpruned prompt:
// kept video tokens keep their (t, h, w), text keeps its own; next_pos is the unpruned one.
// An empty frame pair is located by the vision_end id that closes it.
[[nodiscard]] std::expected<MRopePositions, std::string> qwen_evs_mrope_positions(
    const std::vector<uint8_t>& token_type, std::span<const int32_t> tokens, int32_t vision_end_id,
    const std::vector<MRopeImageGrid>& items, std::span<const VideoEvsPlan> plans,
    std::span<const std::vector<int32_t>> kept, int start_pos);

}  // namespace imp
