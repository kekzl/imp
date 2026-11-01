#include "vision/video_evs.h"

#include "model/image_placeholders.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace imp {

int evs_retained_count(int groups, int tokens_per_group, double rate) {
    if (groups <= 0 || tokens_per_group <= 0)
        return 0;
    const int64_t total = static_cast<int64_t>(groups) * tokens_per_group;
    if (!(rate > 0.0))
        return static_cast<int>(total);
    // Python int(): truncation toward zero of the float64 product.
    const auto evs = static_cast<int64_t>(static_cast<double>(total) * (1.0 - rate));
    return static_cast<int>(std::min(total, std::max<int64_t>(tokens_per_group, evs)));
}

std::vector<uint8_t> evs_retention_mask(std::span<const float> dissim, int groups, int tokens_per_group,
                                        int keep) {
    const size_t total = groups > 0 && tokens_per_group > 0
                             ? static_cast<size_t>(groups) * static_cast<size_t>(tokens_per_group)
                             : 0;
    std::vector<uint8_t> mask(total, 0);
    if (total == 0 || dissim.size() != total)
        return mask;
    // torch.argsort(descending=True, stable=True); NaN sorts as the largest value.
    std::vector<float> key(dissim.begin(), dissim.end());
    for (size_t j = 0; j < total; ++j) {
        if (j < static_cast<size_t>(tokens_per_group))
            key[j] = kEvsFirstGroupScore;
        else if (std::isnan(key[j]))
            key[j] = std::numeric_limits<float>::infinity();
    }
    std::vector<int32_t> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return key[a] > key[b]; });
    const size_t k = std::min(total, static_cast<size_t>(std::max(keep, 0)));
    for (size_t r = 0; r < k; ++r)
        mask[static_cast<size_t>(order[r])] = 1;
    return mask;
}

std::vector<int32_t> evs_kept_indices(std::span<const uint8_t> mask) {
    std::vector<int32_t> out;
    for (size_t j = 0; j < mask.size(); ++j)
        if (mask[j])
            out.push_back(static_cast<int32_t>(j));
    return out;
}

bool evs_plans_match(std::span<const VideoEvsPlan> plans, std::span<const int> tokens_per_item) {
    int free_from = 0;  // first item a next plan may start at
    for (const VideoEvsPlan& p : plans) {
        if (p.groups <= 0 || p.tokens_per_group <= 0 || p.keep <= 0 ||
            p.keep >= p.groups * p.tokens_per_group || p.first_item < free_from ||
            static_cast<size_t>(p.first_item) + static_cast<size_t>(p.groups) > tokens_per_item.size())
            return false;
        for (int g = 0; g < p.groups; ++g)
            if (tokens_per_item[static_cast<size_t>(p.first_item) + static_cast<size_t>(g)] !=
                p.tokens_per_group)
                return false;
        free_from = p.first_item + p.groups;
    }
    return true;
}

std::vector<int32_t> evs_gather_rows(std::span<const VideoEvsPlan> plans,
                                     std::span<const int> tokens_per_item,
                                     std::span<const std::vector<int32_t>> kept) {
    std::vector<int32_t> rows;
    int32_t row = 0;  // first row of the current item in the full-size output
    size_t next = 0;
    for (size_t item = 0; item < tokens_per_item.size();) {
        if (next < plans.size() && next < kept.size() &&
            static_cast<size_t>(plans[next].first_item) == item) {
            for (const int32_t k : kept[next])
                rows.push_back(row + k);
            row += plans[next].groups * plans[next].tokens_per_group;
            item += static_cast<size_t>(plans[next].groups);
            ++next;
            continue;
        }
        for (int r = 0; r < tokens_per_item[item]; ++r)
            rows.push_back(row + r);
        row += tokens_per_item[item];
        ++item;
    }
    return rows;
}

void evs_prune_layout(VideoPlaceholderLayout& layout, int keep) {
    layout.group_tokens.assign(layout.stamp_ids.size(), 0);
    if (!layout.group_tokens.empty())
        layout.group_tokens[0] = keep;
}

namespace {

struct UnprunedPrompt {
    std::vector<uint8_t> type;  // token types of the unpruned prompt
    std::vector<size_t> src;    // pruned token i -> its unpruned index
};

std::expected<UnprunedPrompt, std::string> unprune(const std::vector<uint8_t>& token_type,
                                                   std::span<const int32_t> tokens, int32_t vision_end_id,
                                                   std::span<const VideoEvsPlan> plans,
                                                   std::span<const std::vector<int32_t>> kept) {
    const size_t n = token_type.size();
    UnprunedPrompt u;
    u.src.assign(n, 0);
    auto copy_text = [&](size_t i) {
        u.src[i] = u.type.size();
        u.type.push_back(kMRopeText);
    };
    size_t item = 0, next_plan = 0;
    size_t i = 0;
    while (i < n) {
        const uint8_t t = token_type[i];
        if (t == kMRopeText) {
            copy_text(i++);
            continue;
        }
        size_t run = 0;
        while (i + run < n && token_type[i + run] == t)
            ++run;
        const bool pruned = next_plan < plans.size() && plans[next_plan].first_item == static_cast<int>(item);
        if (!pruned) {
            for (size_t r = 0; r < run; ++r) {
                u.src[i + r] = u.type.size();
                u.type.push_back(t);
            }
            i += run;
            ++item;
            continue;
        }
        const VideoEvsPlan& p = plans[next_plan];
        const std::vector<int32_t>& k = kept[next_plan];
        if (t != kMRopeVideo || run != static_cast<size_t>(p.keep) || k.size() != run)
            return std::unexpected("pruned video " + std::to_string(next_plan) + " reserves " +
                                   std::to_string(run) + " tokens, plan keeps " + std::to_string(p.keep));
        const size_t tpg = static_cast<size_t>(p.tokens_per_group);
        std::vector<size_t> group_start(static_cast<size_t>(p.groups));
        group_start[0] = u.type.size();
        u.type.insert(u.type.end(), tpg, kMRopeVideo);
        size_t j = i + run;
        for (int g = 1; g < p.groups; ++g) {
            // Through the vision_end that closes pair g - 1, then pair g's stamp and start.
            for (int pass = 0; pass < 2; ++pass) {
                while (j < n && tokens[j] != vision_end_id) {
                    if (token_type[j] != kMRopeText)
                        return std::unexpected("pruned video " + std::to_string(next_plan) +
                                               " has vision tokens in an empty frame pair");
                    copy_text(j++);
                }
                if (j >= n)
                    return std::unexpected("pruned video " + std::to_string(next_plan) +
                                           " lacks frame pair " + std::to_string(g));
                if (pass == 0)
                    copy_text(j++);
            }
            group_start[static_cast<size_t>(g)] = u.type.size();
            u.type.insert(u.type.end(), tpg, kMRopeVideo);
        }
        for (size_t r = 0; r < run; ++r) {
            const auto flat = static_cast<size_t>(k[r]);
            if (flat >= group_start.size() * tpg || (r > 0 && k[r] <= k[r - 1]))
                return std::unexpected("pruned video " + std::to_string(next_plan) +
                                       " keeps an index out of order or range");
            u.src[i + r] = group_start[flat / tpg] + (flat % tpg);
        }
        i = j;
        item += static_cast<size_t>(p.groups);
        ++next_plan;
    }
    if (next_plan != plans.size())
        return std::unexpected("prompt holds " + std::to_string(next_plan) + " of " +
                               std::to_string(plans.size()) + " pruned videos");
    return u;
}

}  // namespace

std::expected<MRopePositions, std::string> qwen_evs_mrope_positions(
    const std::vector<uint8_t>& token_type, std::span<const int32_t> tokens, int32_t vision_end_id,
    const std::vector<MRopeImageGrid>& items, std::span<const VideoEvsPlan> plans,
    std::span<const std::vector<int32_t>> kept, int start_pos) {
    if (tokens.size() != token_type.size() || kept.size() != plans.size())
        return std::unexpected("pruned prompt: token, type or plan counts disagree");
    for (size_t v = 0; v < plans.size(); ++v) {
        const VideoEvsPlan& p = plans[v];
        if (p.groups <= 0 || p.tokens_per_group <= 0 || p.keep <= 0 || p.first_item < 0 ||
            (v > 0 && p.first_item < plans[v - 1].first_item + plans[v - 1].groups) ||
            static_cast<size_t>(p.first_item) + static_cast<size_t>(p.groups) > items.size())
            return std::unexpected("pruned video " + std::to_string(v) + " has an invalid plan");
        for (int g = 0; g < p.groups; ++g)
            if (items[static_cast<size_t>(p.first_item) + static_cast<size_t>(g)].tokens() !=
                p.tokens_per_group)
                return std::unexpected("pruned video " + std::to_string(v) + " pair " + std::to_string(g) +
                                       " does not match its tokens per pair");
    }
    auto u = unprune(token_type, tokens, vision_end_id, plans, kept);
    if (!u)
        return std::unexpected(u.error());
    auto full = qwen_build_mrope_positions_items(u->type, items, start_pos);
    if (!full)
        return std::unexpected(full.error());
    const size_t n = token_type.size(), m = u->type.size();
    MRopePositions out;
    out.pos.resize(3 * n);
    for (size_t a = 0; a < 3; ++a)
        for (size_t i = 0; i < n; ++i)
            out.pos[(a * n) + i] = full->pos[(a * m) + u->src[i]];
    out.next_pos = full->next_pos;
    return out;
}

}  // namespace imp
