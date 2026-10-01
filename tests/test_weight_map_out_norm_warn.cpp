// #2347: WeightMap warned "output norm (out_norm) was not found" on every Qwen3.8-Flash-Next load,
// whose checkpoint has no model.norm.weight by design (the hyper-connection mixer is the last
// step). The warning stays for a model that lacks the norm without a mixer.
#include "model/model.h"
#include "model/weight_map.h"
#include "core/logging.h"
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace imp {
namespace {

constexpr const char* kWarn = "output norm (out_norm) was not found";

Tensor fake_weight(void* backing) {
    Tensor t;
    t.data = backing;
    t.qtype = QType::F16;
    t.ndim = 2;
    t.shape[0] = 8;
    t.shape[1] = 8;
    return t;
}

std::string load_stderr(ModelArch arch, const std::vector<std::string>& names) {
    std::vector<uint16_t> backing(64, 0);
    Model model;
    model.config_.arch = arch;
    model.config_.n_layers = 1;
    model.config_.d_model = 8;
    model.config_.n_experts = 0;
    model.layers_.resize(1);
    std::unordered_map<std::string, Tensor> tensors;
    for (const auto& n : names)
        tensors[n] = fake_weight(backing.data());
    const LogLevel saved = log_get_level();  // another test may have raised it past WARN
    log_set_level(LogLevel::WARN);
    testing::internal::CaptureStderr();
    WeightMap wm(arch);
    (void)wm.apply_weights(model, tensors);
    std::string err = testing::internal::GetCapturedStderr();
    log_set_level(saved);
    return err;
}

const std::vector<std::string> kBase = {
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "lm_head.weight",
};

TEST(WeightMapOutNormWarn, MixerWithoutFinalNormIsNotWarned) {
    auto names = kBase;
    names.push_back("model.hyper_connection_mixer.hc_norm.weight");
    names.push_back("model.hyper_connection_mixer.input_mix_weight_down.weight");
    names.push_back("model.hyper_connection_mixer.input_mix_weight_up.weight");
    EXPECT_EQ(load_stderr(ModelArch::QWEN4_EXP, names).find(kWarn), std::string::npos);
}

TEST(WeightMapOutNormWarn, MissingNormWithoutMixerStillWarns) {
    EXPECT_NE(load_stderr(ModelArch::QWEN35, kBase).find(kWarn), std::string::npos);
}

TEST(WeightMapOutNormWarn, PresentNormIsNotWarned) {
    auto names = kBase;
    names.push_back("model.norm.weight");
    EXPECT_EQ(load_stderr(ModelArch::QWEN35, names).find(kWarn), std::string::npos);
}

}  // namespace
}  // namespace imp
