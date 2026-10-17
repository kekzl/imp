// SAE host side (roadmap row 116): feature summary and the Gemma Scope 2 directory loader.

#include "exec/sae.h"
#include "model/safetensors_writer.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace imp {
namespace {

TEST(Sae, TokenInfoTopByActivationThenId) {
    const std::vector<float> acts = {0.0f, 2.0f, 0.0f, 5.0f, 2.0f, 0.0f, 1.0f};
    const SaeTokenInfo info = sae_token_info(acts.data(), static_cast<int>(acts.size()), 3, 0.25f);
    EXPECT_EQ(info.l0, 4);
    EXPECT_FLOAT_EQ(info.rel_err, 0.25f);
    ASSERT_EQ(info.top.size(), 3u);
    EXPECT_EQ(info.top[0], (std::pair<int32_t, float>{3, 5.0f}));
    EXPECT_EQ(info.top[1], (std::pair<int32_t, float>{1, 2.0f})) << "a tie goes to the lower id";
    EXPECT_EQ(info.top[2], (std::pair<int32_t, float>{4, 2.0f}));
    EXPECT_EQ(sae_token_info(acts.data(), 7, 32, 0.0f).top.size(), 4u) << "never more than L0";
}

class SaeDir : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = (std::filesystem::temp_directory_path() / ("imp_sae_" + std::to_string(::getpid()))).string();
        std::filesystem::create_directories(dir_);
    }
    void TearDown() override { std::filesystem::remove_all(dir_); }
    void write(const std::string& hook, int d, int w, const std::string& arch = "jump_relu") {
        std::ofstream(dir_ + "/config.json") << "{\"hf_hook_point_in\": \"" << hook << "\", \"width\": " << w
                                             << ", \"architecture\": \"" << arch << "\"}";
        enc_.assign(static_cast<size_t>(d) * w, 0.5f);
        dec_.assign(static_cast<size_t>(w) * d, -0.25f);
        bw_.assign(static_cast<size_t>(w), 0.1f);
        bd_.assign(static_cast<size_t>(d), 0.2f);
        auto t = [](const char* n, std::vector<int64_t> s, const std::vector<float>& v) {
            return SafeTensorsOut{n, "F32", std::move(s), v.data(), v.size() * sizeof(float)};
        };
        ASSERT_EQ(write_safetensors(dir_ + "/params.safetensors",
                                    {t("w_enc", {d, w}, enc_), t("w_dec", {w, d}, dec_), t("b_enc", {w}, bw_),
                                     t("threshold", {w}, bw_), t("b_dec", {d}, bd_)}),
                  "");
    }
    std::string dir_;
    std::vector<float> enc_, dec_, bw_, bd_;
};

TEST_F(SaeDir, LoadsLayerAndTensors) {
    write("model.layers.3.output", 4, 8);
    SaeHost h;
    std::string err;
    ASSERT_TRUE(load_sae_dir(dir_, 4, 6, h, &err)) << err;
    EXPECT_EQ(h.layer, 3);
    EXPECT_EQ(h.width, 8);
    EXPECT_EQ(h.w_enc.size(), 32u);
    EXPECT_FLOAT_EQ(h.w_dec[5], -0.25f);
    EXPECT_FLOAT_EQ(h.b_dec[3], 0.2f);
}

TEST_F(SaeDir, RefusesMismatches) {
    SaeHost h;
    std::string err;
    write("model.layers.3.output", 4, 8);
    EXPECT_FALSE(load_sae_dir(dir_, 5, 6, h, &err)) << "d_model 5 vs 4";
    EXPECT_NE(err.find("w_enc"), std::string::npos) << err;
    EXPECT_FALSE(load_sae_dir(dir_, 4, 3, h, &err)) << "layer 3 of 3";
    EXPECT_NE(err.find("outside"), std::string::npos) << err;
    write("model.layers.3.mlp", 4, 8);
    EXPECT_FALSE(load_sae_dir(dir_, 4, 6, h, &err));
    write("model.layers.3.output", 4, 8, "topk");
    EXPECT_FALSE(load_sae_dir(dir_, 4, 6, h, &err));
    EXPECT_NE(err.find("jump_relu"), std::string::npos) << err;
}

}  // namespace
}  // namespace imp
