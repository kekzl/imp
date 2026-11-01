#pragma once

// Group-limited top-k routing (DeepSeek-V3 noaux_tc, Ling/BailingMoeV3 group_limited_topk):
// experts split into n_group equal groups, a group scores the sum of its two best selection
// scores, only the topk_group best groups stay eligible for the final top-k.
// One implementation for the device kernel (moe_routing.cu, thread 0) and the CPU-lane test.

#ifdef __CUDACC__
#define IMP_MOE_HD __host__ __device__
#else
#define IMP_MOE_HD
#endif

namespace imp {

struct MoeGroupLimit {
    int n_group = 1;     // 1 = ungrouped
    int topk_group = 0;  // groups kept; <= 0 or >= n_group = all
    [[nodiscard]] IMP_MOE_HD bool active() const {
        return n_group > 1 && topk_group > 0 && topk_group < n_group;
    }
    // Expert cap for the fused M=1 gate GEMV + top-k kernel, which has no group limit: 0 when active.
    [[nodiscard]] IMP_MOE_HD int fused_expert_cap(int cap) const { return active() ? 0 : cap; }
};

inline constexpr int kMoeMaxGroups = 64;

// Writes keep[g] (1 = eligible) for g < n_group from selection scores sel[n_experts]. Ties keep the
// lower group index. Requires n_experts % n_group == 0, group size >= 2 and n_group <= kMoeMaxGroups;
// returns false (keep untouched) otherwise.
IMP_MOE_HD inline bool moe_group_limit_mask(const float* sel, int n_experts, MoeGroupLimit grp,
                                            unsigned char* keep) {
    if (!grp.active() || grp.n_group > kMoeMaxGroups || n_experts % grp.n_group != 0)
        return false;
    const int gs = n_experts / grp.n_group;
    if (gs < 2)
        return false;
    float score[kMoeMaxGroups];
    for (int g = 0; g < grp.n_group; ++g) {
        float b1 = -3.0e38f, b2 = -3.0e38f;
        for (int i = g * gs; i < (g + 1) * gs; ++i) {
            const float v = sel[i];
            if (v > b1) {
                b2 = b1;
                b1 = v;
            } else if (v > b2) {
                b2 = v;
            }
        }
        score[g] = b1 + b2;
        keep[g] = 0;
    }
    for (int r = 0; r < grp.topk_group; ++r) {
        int best = -1;
        for (int g = 0; g < grp.n_group; ++g)
            if (!keep[g] && (best < 0 || score[g] > score[best]))
                best = g;
        keep[best] = 1;
    }
    return true;
}

}  // namespace imp
