// #2462: imp-cli and imp-server resolve the same sampling defaults per arch.

#include <gtest/gtest.h>

#include "runtime/presets.h"
#include "sampling_fields.h"

#include <nlohmann/json.hpp>

namespace {

using imp::HFConfigLoader;
using imp::ModelArch;

// Server path: parse an empty body, then apply the loaded model's defaults (handlers_chat_core.cpp,
// handlers_completions.cpp).
SamplingFields server_defaults(ModelArch arch, const HFConfigLoader::GenerationConfig& gen,
                               const nlohmann::json& body = nlohmann::json::object()) {
    SamplingFields f;
    parse_sampling_fields(body, 0.5f, f);
    f.apply_model_defaults(imp::resolve_sampling_defaults(arch, gen));
    return f;
}

TEST(SamplingDefaultsParity, EveryArchCliEqualsServer) {
    HFConfigLoader::GenerationConfig none;
    HFConfigLoader::GenerationConfig shipped;
    shipped.temperature = 0.3f;
    shipped.top_p = 0.8f;
    shipped.top_k = 7;
    for (int a = 0; a <= static_cast<int>(ModelArch::GENERIC); ++a) {
        const auto arch = static_cast<ModelArch>(a);
        for (const auto* gen : {&none, &shipped}) {
            const imp::SamplingDefaults cli = imp::resolve_sampling_defaults(arch, *gen);
            const SamplingFields srv = server_defaults(arch, *gen);
            EXPECT_EQ(srv.temperature, cli.temperature) << imp::model_arch_name(arch);
            EXPECT_EQ(srv.top_p, cli.top_p) << imp::model_arch_name(arch);
            EXPECT_EQ(srv.top_k, cli.top_k) << imp::model_arch_name(arch);
        }
        const imp::SamplingDefaults preset = imp::get_sampling_defaults(arch);
        EXPECT_EQ(imp::resolve_sampling_defaults(arch, none).temperature, preset.temperature);
        EXPECT_EQ(imp::resolve_sampling_defaults(arch, shipped).top_k, 7);
    }
}

TEST(SamplingDefaultsParity, QwenPresetReplacesServerConstants) {
    const SamplingFields f = server_defaults(ModelArch::QWEN3, {});
    EXPECT_FLOAT_EQ(f.temperature, 0.6f);
    EXPECT_FLOAT_EQ(f.top_p, 0.95f);
    EXPECT_EQ(f.top_k, 20);
}

TEST(SamplingDefaultsParity, ExplicitRequestFieldsWin) {
    const nlohmann::json body = {{"temperature", 0.0}, {"top_p", 1.0}, {"top_k", 3}};
    const SamplingFields f = server_defaults(ModelArch::QWEN3, {}, body);
    EXPECT_FLOAT_EQ(f.temperature, 0.0f);
    EXPECT_FLOAT_EQ(f.top_p, 1.0f);
    EXPECT_EQ(f.top_k, 3);
}

}  // namespace
