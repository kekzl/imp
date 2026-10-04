// AWQ fold-site table (which consumers share an activation, which producer absorbs 1/s, norm
// unit-offset flag): every field is silent-wrong if incorrect - missing group member, wrong
// offset flag, or hardcoded layer prefix all still load successfully. Pure function, no GPU.

#include "../tools/imp-quantize/awq_sites.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

using namespace imp;
using namespace imp::awq;

namespace {

const FoldSite* site_of(const std::vector<FoldSite>& sites, char group) {
    for (const auto& s : sites)
        if (s.group == group)
            return &s;
    return nullptr;
}

bool has_member(const FoldSite& s, const std::string& suffix) {
    return std::any_of(s.members.begin(), s.members.end(), [&](const std::string& m) {
        return m.size() >= suffix.size() && m.compare(m.size() - suffix.size(), suffix.size(), suffix) == 0;
    });
}

// A Qwen3.8-27B layer, both flavours. The names are the ones in the published
// model.safetensors.index.json, nested under model.language_model.
std::set<std::string> attention_layer(const std::string& base) {
    std::set<std::string> n;
    for (const char* p : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"})
        n.insert(base + p + ".weight");
    n.insert(base + "input_layernorm.weight");
    n.insert(base + "post_attention_layernorm.weight");
    n.insert(base + "self_attn.q_norm.weight");
    n.insert(base + "self_attn.k_norm.weight");
    return n;
}

std::set<std::string> gdn_layer(const std::string& base) {
    std::set<std::string> n;
    for (const char* p :
         {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a",
          "linear_attn.in_proj_b", "linear_attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"})
        n.insert(base + p + ".weight");
    n.insert(base + "linear_attn.norm.weight");
    n.insert(base + "input_layernorm.weight");
    n.insert(base + "post_attention_layernorm.weight");
    n.insert(base + "linear_attn.A_log");
    n.insert(base + "linear_attn.dt_bias");
    n.insert(base + "linear_attn.conv1d.weight");
    return n;
}

}  // namespace

// The refusal used to be a model_type allowlist that named four architectures,
// so qwen3_5 was rejected for "having a (1 + g) norm" while the tool already
// knew how to fold one. The table answers the convention question instead.
TEST(AwqArch, KnowsTheOffsetConventionPerArchitecture) {
    for (const char* plain : {"qwen2", "qwen3", "llama", "mistral"}) {
        const auto conv = arch_norm_convention(plain);
        ASSERT_TRUE(conv.has_value()) << plain;
        EXPECT_EQ(*conv, NormConvention::Plain) << plain;
    }
    // The qwen3_5 family, top-level and text_config spellings
    // (src/model/hf_config_loader.cpp), and the same-family qwen3_next.
    for (const char* offset : {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"}) {
        const auto conv = arch_norm_convention(offset);
        ASSERT_TRUE(conv.has_value()) << offset;
        EXPECT_EQ(*conv, NormConvention::UnitOffset) << offset;
    }
    // Unknown stays refused: the convention is read off imp's own loader table,
    // never guessed from the name.
    EXPECT_FALSE(arch_norm_convention("gemma3").has_value());
    EXPECT_FALSE(arch_norm_convention("deepseek_v3").has_value());
    EXPECT_FALSE(arch_norm_convention("").has_value());
}

TEST(AwqArch, ResolvesTheLayerPrefixFromTheCheckpoint) {
    EXPECT_EQ(resolve_layer_prefix(attention_layer("model.layers.0.")), "model.layers.");
    EXPECT_EQ(resolve_layer_prefix(gdn_layer("model.language_model.layers.7.")),
              "model.language_model.layers.");
    // A checkpoint with neither form must not be given one by default: the old
    // hardcoded "model.layers." found zero members and reported nothing wrong.
    EXPECT_EQ(resolve_layer_prefix({"encoder.block.0.attn.q.weight"}), "");
}

TEST(AwqSites, AttentionLayerCarriesTheFourClassicGroups) {
    const std::string base = "model.language_model.layers.3.";
    const auto sites = layer_fold_sites(attention_layer(base), base, NormConvention::UnitOffset, "ABCDEG");

    const FoldSite* a = site_of(sites, 'A');
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->members.size(), 3u);
    EXPECT_EQ(a->producer, base + "input_layernorm.weight");
    EXPECT_EQ(a->kind, FoldKind::NormVector);
    EXPECT_EQ(a->offset, NormOffset::Unit);
    EXPECT_EQ(a->scan_prefix, base + "self_attn.");

    const FoldSite* b = site_of(sites, 'B');
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->producer, base + "post_attention_layernorm.weight");
    EXPECT_EQ(b->offset, NormOffset::Unit);

    // C and D fold into rows, not into a norm, so the offset never enters.
    const FoldSite* c = site_of(sites, 'C');
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->kind, FoldKind::MatrixRows);
    EXPECT_EQ(c->producer, base + "self_attn.v_proj.weight");
    EXPECT_EQ(c->tie, TieMode::GqaValue);

    const FoldSite* d = site_of(sites, 'D');
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->kind, FoldKind::MatrixRows);
    EXPECT_EQ(d->producer, base + "mlp.up_proj.weight");
    EXPECT_EQ(d->tie, TieMode::Identity);

    // A GDN layer's groups must not be invented on an attention layer.
    EXPECT_EQ(site_of(sites, 'G'), nullptr);
    EXPECT_EQ(site_of(sites, 'E'), nullptr);
}

TEST(AwqSites, PlainArchitecturesKeepThePlainFold) {
    const std::string base = "model.layers.0.";
    const auto sites = layer_fold_sites(attention_layer(base), base, NormConvention::Plain, "ABCDEG");
    ASSERT_NE(site_of(sites, 'A'), nullptr);
    EXPECT_EQ(site_of(sites, 'A')->offset, NormOffset::Plain);
    EXPECT_EQ(site_of(sites, 'B')->offset, NormOffset::Plain);
}

TEST(AwqSites, GdnLayerFoldsTheFourInProjectionsAndOutProj) {
    const std::string base = "model.language_model.layers.1.";
    const auto sites = layer_fold_sites(gdn_layer(base), base, NormConvention::UnitOffset, "ABCDEG");

    const FoldSite* g = site_of(sites, 'G');
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(g->members.size(), 4u);
    for (const char* p : {"in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight"})
        EXPECT_TRUE(has_member(*g, p)) << p;
    EXPECT_EQ(g->producer, base + "input_layernorm.weight");
    EXPECT_EQ(g->offset, NormOffset::Unit);
    EXPECT_EQ(g->scan_prefix, base + "linear_attn.");
    // out_proj reads the scan output, not the pre-norm, so it is exempt from
    // the scan rather than a blocker.
    EXPECT_NE(std::find(g->scan_exempt.begin(), g->scan_exempt.end(), std::string("out_proj.weight")),
              g->scan_exempt.end());

    // E: the GDN gated norm is PLAIN multiplicative, [head_dim], shared across
    // the value heads, so the divisor must be tied across heads.
    const FoldSite* e = site_of(sites, 'E');
    ASSERT_NE(e, nullptr);
    ASSERT_EQ(e->members.size(), 1u);
    EXPECT_EQ(e->members[0], base + "linear_attn.out_proj.weight");
    EXPECT_EQ(e->producer, base + "linear_attn.norm.weight");
    EXPECT_EQ(e->kind, FoldKind::NormVector);
    EXPECT_EQ(e->offset, NormOffset::Plain);
    EXPECT_EQ(e->tie, TieMode::PerHeadDim);

    // The attention groups have no members here.
    EXPECT_EQ(site_of(sites, 'A'), nullptr);
    EXPECT_EQ(site_of(sites, 'C'), nullptr);
    // The FFN groups are the same on both layer flavours.
    EXPECT_NE(site_of(sites, 'B'), nullptr);
    EXPECT_NE(site_of(sites, 'D'), nullptr);
}

// Folding the norm divides its output for EVERY consumer. Three of the four
// in-projections scaled and one left alone is a checkpoint that loads and is
// wrong, so a missing member must drop the whole group.
TEST(AwqSites, GdnInProjGroupNeedsAllFourConsumers) {
    const std::string base = "model.language_model.layers.1.";
    std::set<std::string> names = gdn_layer(base);
    names.erase(base + "linear_attn.in_proj_b.weight");
    const auto sites = layer_fold_sites(names, base, NormConvention::UnitOffset, "ABCDEG");
    EXPECT_EQ(site_of(sites, 'G'), nullptr);
    // The rest of the layer is unaffected.
    EXPECT_NE(site_of(sites, 'E'), nullptr);
    EXPECT_NE(site_of(sites, 'B'), nullptr);
}

TEST(AwqSites, GroupSelectorDropsTheSitesItExcludes) {
    const std::string base = "model.language_model.layers.1.";
    const auto sites = layer_fold_sites(gdn_layer(base), base, NormConvention::UnitOffset, "BD");
    EXPECT_NE(site_of(sites, 'B'), nullptr);
    EXPECT_NE(site_of(sites, 'D'), nullptr);
    EXPECT_EQ(site_of(sites, 'G'), nullptr);
    EXPECT_EQ(site_of(sites, 'E'), nullptr);
}

// C, D and E fold into a tensor that groups A, B and G then quantize, so the
// row folds have to be decided first or the norm searches measure weights the
// writer will not write.
TEST(AwqSites, RowAndTiedFoldsComeBeforeTheNormFolds) {
    const std::string base = "model.language_model.layers.1.";
    const auto sites = layer_fold_sites(gdn_layer(base), base, NormConvention::UnitOffset, "ABCDEG");
    size_t first_norm = sites.size(), last_row = 0;
    for (size_t i = 0; i < sites.size(); i++) {
        const bool folds_a_norm_the_others_consume = sites[i].kind == FoldKind::NormVector &&
                                                     sites[i].group != 'E';
        if (folds_a_norm_the_others_consume)
            first_norm = std::min(first_norm, i);
        else
            last_row = std::max(last_row, i);
    }
    EXPECT_LT(last_row, first_norm) << "a norm group was searched before the fold into its members";
}

TEST(AwqSites, TieMapMatchesTheProducerItFoldsInto) {
    // GQA: 24 query heads over 4 KV heads, head_dim 128 (Qwen3.8-27B).
    const Geometry gqa{/*head_dim=*/128, /*n_rep=*/6};
    const auto m = tie_map(TieMode::GqaValue, 24 * 128, gqa);
    ASSERT_EQ(m.size(), static_cast<size_t>(24 * 128));
    EXPECT_EQ(m[0], 0);
    EXPECT_EQ(m[127], 127);
    EXPECT_EQ(m[128], 0) << "head 1 shares KV head 0";
    EXPECT_EQ(m[6 * 128], 128) << "head 6 is the first of KV head 1";
    EXPECT_EQ(tie_producer_len(TieMode::GqaValue, 24 * 128, gqa), 4 * 128);

    // The GDN norm is one row of head_dim shared by all 48 value heads.
    const Geometry heads{/*head_dim=*/128, /*n_rep=*/1};
    const auto p = tie_map(TieMode::PerHeadDim, 48 * 128, heads);
    ASSERT_EQ(p.size(), static_cast<size_t>(48 * 128));
    EXPECT_EQ(p[0], 0);
    EXPECT_EQ(p[128], 0);
    EXPECT_EQ(p[129], 1);
    EXPECT_EQ(tie_producer_len(TieMode::PerHeadDim, 48 * 128, heads), 128);

    const auto id = tie_map(TieMode::Identity, 8, heads);
    ASSERT_EQ(id.size(), 8u);
    EXPECT_EQ(id[5], 5);
    EXPECT_EQ(tie_producer_len(TieMode::Identity, 8, heads), 8);

    // A geometry that cannot express the tie yields nothing rather than a map
    // that silently folds the wrong channels together.
    EXPECT_TRUE(tie_map(TieMode::PerHeadDim, 100, heads).empty());
    EXPECT_EQ(tie_producer_len(TieMode::PerHeadDim, 100, heads), 0);
    EXPECT_TRUE(tie_map(TieMode::GqaValue, 24 * 128, Geometry{0, 6}).empty());
}

// Roadmap row 6: Qwen3-14B (dense, n_rep 5) ABCD 12.2634 vs BD 9.9068 PPL; Qwen3-0.6B (n_rep 2)
// ABCD wins; Qwen3.8-27B (GDN hybrid, n_rep 6) ABCD 4.5986 vs BDEG 4.6136, so hybrids keep all.
// #2474: n_rep 4 BDEG wins too (Qwen3-4B 14.1954 vs 14.9701, Qwen3-8B 11.4263 vs 11.7296).
TEST(AwqSites, DefaultGroupsDropAttentionOnDenseWideGqaOnly) {
    for (bool hybrid : {false, true}) {
        EXPECT_STREQ(default_groups(1, hybrid), kAwqDefaultGroups);
        EXPECT_STREQ(default_groups(2, hybrid), kAwqDefaultGroups);
        EXPECT_STREQ(default_groups(3, hybrid), kAwqDefaultGroups) << "n_rep 3 unmeasured";
    }
    EXPECT_STREQ(default_groups(4, false), "BDEG") << "Qwen3-4B, Qwen3-8B";
    EXPECT_STREQ(default_groups(4, true), kAwqDefaultGroups);
    EXPECT_STREQ(default_groups(5, false), "BDEG");
    EXPECT_STREQ(default_groups(6, false), "BDEG");
    EXPECT_STREQ(default_groups(6, true), kAwqDefaultGroups) << "Qwen3.8-27B";
    for (int64_t n_rep : {5, 6, 8}) {
        EXPECT_FALSE(attention_groups_on_wide_gqa(default_groups(n_rep, false), n_rep, false));
        EXPECT_FALSE(attention_groups_on_wide_gqa(default_groups(n_rep, true), n_rep, true));
    }
    EXPECT_TRUE(attention_groups_on_wide_gqa("ABCD", 5, false));
    EXPECT_TRUE(attention_groups_on_wide_gqa("BCD", 6, false));
    EXPECT_TRUE(attention_groups_on_wide_gqa("ABCD", 4, false));
    EXPECT_FALSE(attention_groups_on_wide_gqa("ABCD", 3, false));
    EXPECT_FALSE(attention_groups_on_wide_gqa("ABCD", 6, true));
}

// The wide-GQA default BDEG folds exactly the measured BD sites on a dense layer (Qwen3-14B).
TEST(AwqSites, WideGqaDefaultIsBdOnADenseLayer) {
    const std::string base = "model.layers.7.";
    const auto bdeg = layer_fold_sites(attention_layer(base), base, NormConvention::Plain, kAwqWideGqaGroups);
    const auto bd = layer_fold_sites(attention_layer(base), base, NormConvention::Plain, "BD");
    ASSERT_EQ(bdeg.size(), bd.size());
    for (size_t i = 0; i < bd.size(); i++) {
        EXPECT_EQ(bdeg[i].group, bd[i].group);
        EXPECT_EQ(bdeg[i].members, bd[i].members);
        EXPECT_EQ(bdeg[i].producer, bd[i].producer);
    }
    EXPECT_EQ(bd.size(), 2u);
}

namespace {

// A Gemma-4-26B-A4B layer after destacking: sandwich norms, dense MLP beside 128 experts, the
// router on its own norm. `full` = a full-attention layer (k_eq_v: no v_proj).
std::set<std::string> gemma4_layer(const std::string& base, bool full, int n_experts = 4) {
    std::set<std::string> n;
    for (const char* p : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj",
                          "mlp.up_proj", "mlp.down_proj", "router.proj"})
        n.insert(base + p + ".weight");
    if (!full)
        n.insert(base + "self_attn.v_proj.weight");
    for (const char* p :
         {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
          "post_feedforward_layernorm", "pre_feedforward_layernorm_2", "post_feedforward_layernorm_1",
          "post_feedforward_layernorm_2", "self_attn.q_norm", "self_attn.k_norm"})
        n.insert(base + p + ".weight");
    for (int e = 0; e < n_experts; e++)
        for (const char* p : {"gate_proj", "up_proj", "down_proj"})
            n.insert(base + "experts." + std::to_string(e) + "." + p + ".weight");
    return n;
}

}  // namespace

TEST(AwqSites, ExpertGroupsAreOptIn) {
    for (int64_t n_rep : {1, 4, 6})
        for (bool hybrid : {false, true})
            EXPECT_EQ(std::string(default_groups(n_rep, hybrid)).find_first_of("XY"), std::string::npos);
    EXPECT_NE(std::string(kAwqAllGroups).find('X'), std::string::npos);
    EXPECT_NE(std::string(kAwqAllGroups).find('Y'), std::string::npos);
}

TEST(AwqArch, Gemma4IsPlainAndNormalizesV) {
    EXPECT_EQ(arch_norm_convention("gemma4"), NormConvention::Plain);
    EXPECT_EQ(arch_norm_convention("gemma4_text"), NormConvention::Plain);
    EXPECT_TRUE(arch_normalizes_v("gemma4_text"));
    EXPECT_FALSE(arch_normalizes_v("qwen3"));
}

// post_attention_layernorm normalizes the attention OUTPUT on a sandwich block: folding the MLP
// scale into it would divide the residual branch, not the MLP input.
TEST(AwqSites, Gemma4MlpFoldsIntoPreFeedforwardNorm) {
    const std::string base = "model.language_model.layers.0.";
    const auto sites = layer_fold_sites(gemma4_layer(base, false), base, NormConvention::Plain, kAwqAllGroups,
                                        /*v_normed=*/true);
    const FoldSite* b = site_of(sites, 'B');
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->producer, base + "pre_feedforward_layernorm.weight");
    // v_norm divides a v_proj row scale back out per head.
    EXPECT_EQ(site_of(sites, 'C'), nullptr);
    ASSERT_NE(site_of(sites, 'A'), nullptr);
    EXPECT_EQ(site_of(sites, 'A')->members.size(), 3u);
}

TEST(AwqSites, Gemma4FullAttentionLayerFoldsQAndKOnly) {
    const std::string base = "model.language_model.layers.5.";
    const auto sites = layer_fold_sites(gemma4_layer(base, true), base, NormConvention::Plain, kAwqAllGroups,
                                        /*v_normed=*/true);
    const FoldSite* a = site_of(sites, 'A');
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->members.size(), 2u);
    EXPECT_FALSE(has_member(*a, "v_proj.weight"));
}

TEST(AwqSites, Gemma4ExpertsGetOneSharedInputGroupAndOneDownGroupEach) {
    const std::string base = "model.language_model.layers.2.";
    const auto sites = layer_fold_sites(gemma4_layer(base, false, 3), base, NormConvention::Plain,
                                        kAwqAllGroups, true);
    const FoldSite* x = site_of(sites, 'X');
    ASSERT_NE(x, nullptr);
    EXPECT_EQ(x->members.size(), 6u);  // gate + up of 3 experts
    EXPECT_EQ(x->producer, base + "pre_feedforward_layernorm_2.weight");
    EXPECT_EQ(x->kind, FoldKind::NormVector);
    EXPECT_FALSE(has_member(*x, "down_proj.weight"));
    EXPECT_FALSE(has_member(*x, "router.proj.weight"));
    EXPECT_EQ(x->calib_keys, std::vector<std::string>{"EXPERT_UP"});

    std::vector<const FoldSite*> ys;
    size_t x_pos = 0, last_y = 0;
    for (size_t i = 0; i < sites.size(); i++) {
        if (sites[i].group == 'Y') {
            ys.push_back(&sites[i]);
            last_y = i;
        }
        if (sites[i].group == 'X')
            x_pos = i;
    }
    ASSERT_EQ(ys.size(), 3u);
    // Y's producer (an expert up_proj) is an X member: row folds are searched first.
    EXPECT_LT(last_y, x_pos);
    EXPECT_EQ(ys[1]->members, std::vector<std::string>{base + "experts.1.down_proj.weight"});
    EXPECT_EQ(ys[1]->producer, base + "experts.1.up_proj.weight");
    EXPECT_EQ(ys[1]->kind, FoldKind::MatrixRows);
    EXPECT_EQ(ys[1]->calib_keys, std::vector<std::string>{"EXPERT_DOWN.1"});
}

// Qwen-MoE experts read post_attention_layernorm together with the router and the shared expert:
// no X there, the per-expert down groups still apply.
TEST(AwqSites, ExpertsWithoutAnExclusiveNormGetDownGroupsOnly) {
    const std::string base = "model.layers.1.";
    std::set<std::string> n = {base + "input_layernorm.weight", base + "post_attention_layernorm.weight",
                               base + "mlp.gate.weight"};
    for (int e = 0; e < 2; e++)
        for (const char* p : {"gate_proj", "up_proj", "down_proj"})
            n.insert(base + "mlp.experts." + std::to_string(e) + "." + p + ".weight");
    const auto sites = layer_fold_sites(n, base, NormConvention::UnitOffset, kAwqAllGroups);
    EXPECT_EQ(site_of(sites, 'X'), nullptr);
    EXPECT_EQ(site_of(sites, 'B'), nullptr);
    int ys = 0;
    for (const auto& s : sites)
        ys += s.group == 'Y';
    EXPECT_EQ(ys, 2);
}
