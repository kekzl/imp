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
