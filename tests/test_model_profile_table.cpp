#include "exec/fp8_decode_sidecar.h"
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
    // Loader traits pinned from the pre-B-120 hand lists: 5 delta, 5 neox, 8 refusals.
    bool norm_delta, st_neox;
    const char* gguf_refusal;  // kLoads or the exact error text; null = row left short
};

constexpr const char* kLoads = "";

constexpr std::array<int, kLayers> kMasked = {kWindow, 0, kWindow, 0};
constexpr std::array<int, kLayers> kAll = {kWindow, kWindow, kWindow, kWindow};

// clang-format off
constexpr ProfileRow kRows[] = {
    // arch                       gemma3 gemma4 gpt_oss llama4 encoder  variant              windows  delta  neox   gguf refusal
    {ModelArch::LLAMA,            false, false, false,  false, false,   Variant::STANDARD,   kAll, false, true,  kLoads},
    {ModelArch::MISTRAL,          false, false, false,  false, false,   Variant::STANDARD,   kAll, false, true,  kLoads},
    {ModelArch::MIXTRAL,          false, false, false,  false, false,   Variant::STANDARD,   kAll, false, true,  kLoads},
    {ModelArch::DEEPSEEK,         false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, kLoads},
    {ModelArch::NEMOTRON_H_MOE,   false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, kLoads},
    {ModelArch::QWEN3,            false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, kLoads},
    {ModelArch::QWEN3_MOE,        false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, kLoads},
    {ModelArch::QWEN35,           false, false, false,  false, false,   Variant::STANDARD,   kAll, true,  false, kLoads},
    {ModelArch::QWEN35_MOE,       false, false, false,  false, false,   Variant::STANDARD,   kAll, true,  false, kLoads},
    {ModelArch::QWEN36_MOE,       false, false, false,  false, false,   Variant::STANDARD,   kAll, true,  false, kLoads},
    {ModelArch::QWEN4_EXP,        false, false, false,  false, false,   Variant::STANDARD,   kAll, true,  false, kLoads},
    {ModelArch::GPT_OSS,          false, false, true,   false, false,   Variant::GPTOSS_SWA, kMasked, false, false, kLoads},
    {ModelArch::GEMMA3,           true,  false, false,  false, false,   Variant::STANDARD,   kAll, false, false, kLoads},
    {ModelArch::GEMMA4,           false, true,  false,  false, false,   Variant::GEMMA4_SWA, kMasked, false, false, kLoads},
    {ModelArch::LLAMA4,           false, false, false,  true,  false,   Variant::STANDARD,   kAll, false, true,  kLoads},
    {ModelArch::NOMIC_BERT,       false, false, false,  false, true,    Variant::STANDARD,   kAll, false, false, kLoads},
    {ModelArch::GRANITE,          false, false, false,  false, false,   Variant::STANDARD,   kAll, false, true,  kLoads},
    {ModelArch::QWEN3_NEXT,       false, false, false,  false, false,   Variant::STANDARD,   kAll, true,  false, "qwen3next GGUF is not supported; load the SafeTensors checkpoint (#2410)"},
    {ModelArch::OLMO3,            false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, "olmo3 GGUF is not supported; load the SafeTensors checkpoint"},
    {ModelArch::LFM2,             false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, "lfm2 GGUF is not supported; load the SafeTensors checkpoint"},
    {ModelArch::COHERE2,          false, false, false,  false, false,   Variant::GPTOSS_SWA, kMasked, false, false, "cohere2moe GGUF is not supported; load the SafeTensors checkpoint"},
    {ModelArch::BAILING_HYBRID,   false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, "bailing_hybrid GGUF is not supported; load the SafeTensors checkpoint"},
    {ModelArch::KIMI_LINEAR,      false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, "kimi_linear GGUF is not supported; load the SafeTensors checkpoint"},
    {ModelArch::FLINT,            false, false, false,  false, false,   Variant::GPTOSS_SWA, kMasked, false, false, "flint GGUF is not supported; load the SafeTensors checkpoint"},
    {ModelArch::INKLING2,         false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, "inkling2_loop GGUF is not supported; load the SafeTensors checkpoint"},
    {ModelArch::GENERIC,          false, false, false,  false, false,   Variant::STANDARD,   kAll, false, false, kLoads},
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
        for (bool t : {p.moe_expert_bias_glu, p.moe_router_logit_bias, p.experts_convert_at_predequant,
                       p.residual_rescale_in_scales})
            EXPECT_EQ(t, r->gpt_oss);
        // Learned attention sinks: gpt-oss (self_attn.sinks) and Flint (self_attn.softmax_offset).
        EXPECT_EQ(p.learned_attn_sinks, r->gpt_oss || arch == ModelArch::FLINT);
        // Unset kv_cache.swa_snapshot_mb: only Flint takes a budget (SWA sizing beside prefix caching).
        EXPECT_EQ(p.swa_snapshot_default_mb, arch == ModelArch::FLINT ? 256 : 0);
        // FP16 attention projections decode from the FP8 sidecar: gpt-oss and Flash-Next.
        EXPECT_EQ(p.fp8_attn_proj_full, r->gpt_oss || arch == ModelArch::QWEN4_EXP);
        // gemm.fp8_ffn_proj=auto: the dense 16-bit FFN decodes from the FP8 sidecar on Gemma-4 NVFP4
        // checkpoints only (B-141: Qwen3 / Qwen3.8 / Llama off, the all-BF16 E4B off); =off removes it.
        EXPECT_FALSE(p.fp8_ffn_proj_auto) << "16-bit checkpoint";
        ModelConfig nvfp4 = cfg;
        nvfp4.is_nvfp4_prequant = true;
        const ModelProfile q = profile_for(arch, nvfp4);
        EXPECT_EQ(q.fp8_ffn_proj_auto, r->gemma4);
        EXPECT_EQ(imp::fp8_sidecar_scope("auto", q.fp8_attn_proj_full, "auto", q.fp8_ffn_proj_auto).ffn,
                  r->gemma4);
        EXPECT_FALSE(imp::fp8_sidecar_scope("auto", q.fp8_attn_proj_full, "off", q.fp8_ffn_proj_auto).ffn);
        EXPECT_EQ(p.deny_cublas_fp16_acc, r->gemma3 || r->gemma4 || r->gpt_oss);
        EXPECT_EQ(p.is_llama4, r->llama4);
        EXPECT_EQ(p.is_encoder, r->encoder);
        EXPECT_EQ(p.attn_variant, r->swa_variant);
        for (int l = 0; l < kLayers; l++)
            EXPECT_EQ(imp::layer_swa_window(cfg, p, l), r->swa_windows[l]) << "layer " << l;
    }
}

// B-120: the loaders read these three per-arch traits from kArchRegistry; rows pin the old lists.
TEST(ModelProfileTable, LoaderTraitsMatchThePinnedLists) {
    int n_delta = 0, n_neox = 0, n_refused = 0;
    for (int i = 0; i < kArchCount; i++) {
        const auto arch = static_cast<ModelArch>(i);
        const ProfileRow* r = find_row(arch);
        if (r == nullptr)
            continue;  // EveryArchHasOneRow reports it
        SCOPED_TRACE(imp::model_arch_name(arch));
        ASSERT_NE(r->gguf_refusal, nullptr) << "row must state norm_delta, st_neox and gguf_refusal";
        EXPECT_EQ(imp::norm_gamma_is_delta(arch), r->norm_delta);
        EXPECT_EQ(imp::safetensors_forces_neox_rope(arch), r->st_neox);
        const char* why = imp::gguf_refusal(arch);
        EXPECT_STREQ(why != nullptr ? why : kLoads, r->gguf_refusal);
        n_delta += r->norm_delta ? 1 : 0;
        n_neox += r->st_neox ? 1 : 0;
        n_refused += why != nullptr ? 1 : 0;
    }
    EXPECT_EQ(n_delta, 5);
    EXPECT_EQ(n_neox, 5);
    EXPECT_EQ(n_refused, 8);
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
