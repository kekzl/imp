#include "model/model.h"
#include "model/model_arch.h"
#include "model/model_config.h"
#include "model/model_profile.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using imp::ModelArch;
using imp::ModelConfig;
using imp::ModelProfile;
using Variant = imp::ModelProfile::AttnVariant;

namespace {

// GENERIC is the last ModelArch enumerator (model_arch.h); the loop below walks 0..GENERIC.
constexpr int kArchCount = static_cast<int>(ModelArch::GENERIC) + 1;
constexpr int kWindow = 128;
constexpr int kLayers = 4;

// One row per ModelArch: the arch-keyed ModelProfile fields, plus the attention variant and
// per-layer windows under a sliding_window=128 config with swa_layers {1,0,1,0}.
struct ProfileRow {
    ModelArch arch;
    bool gemma3, gemma4, gpt_oss, llama4, encoder;
    Variant swa_variant;
    std::array<int, kLayers> swa_windows;
};

constexpr std::array<int, kLayers> kMasked = {kWindow, 0, kWindow, 0};
constexpr std::array<int, kLayers> kAll = {kWindow, kWindow, kWindow, kWindow};

// clang-format off
constexpr ProfileRow kRows[] = {
    // arch                       gemma3 gemma4 gpt_oss llama4 encoder  variant              windows
    {ModelArch::LLAMA,            false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::MISTRAL,          false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::MIXTRAL,          false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::DEEPSEEK,         false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::NEMOTRON_H_MOE,   false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::QWEN3,            false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::QWEN3_MOE,        false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::QWEN35,           false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::QWEN35_MOE,       false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::QWEN36_MOE,       false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::QWEN4_EXP,        false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::GPT_OSS,          false, false, true,   false, false,   Variant::GPTOSS_SWA, kMasked},
    {ModelArch::GEMMA3,           true,  false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::GEMMA4,           false, true,  false,  false, false,   Variant::GEMMA4_SWA, kMasked},
    {ModelArch::LLAMA4,           false, false, false,  true,  false,   Variant::STANDARD,   kAll},
    {ModelArch::NOMIC_BERT,       false, false, false,  false, true,    Variant::STANDARD,   kAll},
    {ModelArch::GRANITE,          false, false, false,  false, false,   Variant::STANDARD,   kAll},
    {ModelArch::GENERIC,          false, false, false,  false, false,   Variant::STANDARD,   kAll},
};
// clang-format on

const ProfileRow* find_row(ModelArch arch) {
    for (const auto& r : kRows)
        if (r.arch == arch)
            return &r;
    return nullptr;
}

ModelProfile profile_for(ModelArch arch, const ModelConfig& base) {
    imp::Model m;
    m.config_ = base;
    m.config_.arch = arch;
    return imp::derive_model_profile(m, m.config_);
}

// Every enumerator has exactly one row; a new ModelArch fails here until its row is added.
TEST(ModelProfileTable, EveryArchHasOneRow) {
    EXPECT_EQ(std::size(kRows), static_cast<size_t>(kArchCount));
    for (int i = 0; i < kArchCount; i++) {
        const auto arch = static_cast<ModelArch>(i);
        int hits = 0;
        for (const auto& r : kRows)
            hits += r.arch == arch ? 1 : 0;
        EXPECT_EQ(hits, 1) << "ModelArch " << imp::model_arch_name(arch) << " (" << i << ")";
    }
}

TEST(ModelProfileTable, ArchKeyedFieldsAndSwaWindows) {
    ModelConfig cfg;
    cfg.sliding_window = kWindow;
    cfg.swa_layers = {1, 0, 1, 0};
    for (int i = 0; i < kArchCount; i++) {
        const auto arch = static_cast<ModelArch>(i);
        const ProfileRow* r = find_row(arch);
        if (r == nullptr)
            continue;  // EveryArchHasOneRow reports it
        SCOPED_TRACE(imp::model_arch_name(arch));
        const ModelProfile p = profile_for(arch, cfg);
        EXPECT_EQ(p.is_gemma3, r->gemma3);
        // Capability traits: the Gemma-4 and gpt-oss sets, each set on exactly its arch.
        for (bool t :
             {p.sandwich_norms, p.fp32_residual_norms, p.sanitize_ffn_fp16, p.scaled_router_norm,
              p.router_bias_is_expert_scale, p.expert_out_scale, p.per_layer_head_shapes, p.k_as_v_without_wv,
              p.v_rmsnorm, p.rope_full_head_dim, p.unit_softmax_scale, p.outlier_sensitive_logits})
            EXPECT_EQ(t, r->gemma4);
        for (bool t : {p.learned_attn_sinks, p.moe_expert_bias_glu, p.moe_router_logit_bias,
                       p.experts_convert_at_predequant, p.fp8_attn_proj_full, p.residual_rescale_in_scales})
            EXPECT_EQ(t, r->gpt_oss);
        EXPECT_EQ(p.deny_cublas_fp16_acc, r->gemma3 || r->gemma4 || r->gpt_oss);
        EXPECT_EQ(p.is_llama4, r->llama4);
        EXPECT_EQ(p.is_encoder, r->encoder);
        EXPECT_EQ(p.attn_variant, r->swa_variant);
        for (int l = 0; l < kLayers; l++)
            EXPECT_EQ(imp::layer_swa_window(cfg, p, l), r->swa_windows[l]) << "layer " << l;
    }
}

// Without swa_layers no arch takes a per-layer SWA variant; Gemma-3 pattern 2 globals odd layers.
TEST(ModelProfileTable, NoSwaMaskIsStandardForEveryArch) {
    ModelConfig cfg;
    cfg.sliding_window = kWindow;
    cfg.sliding_window_pattern = 2;
    for (int i = 0; i < kArchCount; i++) {
        const auto arch = static_cast<ModelArch>(i);
        SCOPED_TRACE(imp::model_arch_name(arch));
        const ModelProfile p = profile_for(arch, cfg);
        EXPECT_EQ(p.attn_variant, Variant::STANDARD);
        for (int l = 0; l < kLayers; l++)
            EXPECT_EQ(imp::layer_swa_window(cfg, p, l), kMasked[l]) << "layer " << l;
    }
}

}  // namespace
