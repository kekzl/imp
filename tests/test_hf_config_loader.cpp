#include "model/hf_config_loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

using imp::HFConfigLoader;

namespace {

class GenerationConfigTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override {
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("imp_test_gencfg_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(tmp_dir_);
    }

    void write_gen_config(const std::string& json) {
        std::ofstream f(tmp_dir_ / "generation_config.json");
        f << json;
    }
};

// Mistral-3.2-style: ships only `temperature`. Other fields stay at sentinel
// so the CLI cascade falls through to the arch-family preset.
TEST_F(GenerationConfigTest, PartialFieldsLeaveSentinels) {
    write_gen_config(R"({
        "bos_token_id": 1,
        "do_sample": true,
        "eos_token_id": 2,
        "pad_token_id": 11,
        "temperature": 0.15
    })");

    HFConfigLoader::GenerationConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_generation_config(tmp_dir_.string(), cfg));

    EXPECT_FLOAT_EQ(cfg.temperature, 0.15f);
    EXPECT_LT(cfg.top_p, 0.0f);
    EXPECT_LT(cfg.top_k, 0);
    EXPECT_LT(cfg.repetition_penalty, 0.0f);
    ASSERT_EQ(cfg.eos_token_ids.size(), 1u);
    EXPECT_EQ(cfg.eos_token_ids[0], 2);
}

// Qwen3-Coder-30B-style: ships all four sampling fields plus a multi-EOS array.
TEST_F(GenerationConfigTest, AllSamplingFieldsAndEosArray) {
    write_gen_config(R"({
        "do_sample": true,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8
    })");

    HFConfigLoader::GenerationConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_generation_config(tmp_dir_.string(), cfg));

    EXPECT_FLOAT_EQ(cfg.temperature, 0.7f);
    EXPECT_FLOAT_EQ(cfg.top_p, 0.8f);
    EXPECT_EQ(cfg.top_k, 20);
    EXPECT_FLOAT_EQ(cfg.repetition_penalty, 1.05f);
    ASSERT_EQ(cfg.eos_token_ids.size(), 2u);
    EXPECT_EQ(cfg.eos_token_ids[0], 151645);
    EXPECT_EQ(cfg.eos_token_ids[1], 151643);
}

// do_sample=false overrides whatever temperature was specified — author wants
// deterministic greedy regardless of the per-model temperature default.
TEST_F(GenerationConfigTest, DoSampleFalseForcesGreedy) {
    write_gen_config(R"({
        "do_sample": false,
        "eos_token_id": 2,
        "temperature": 0.7
    })");

    HFConfigLoader::GenerationConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_generation_config(tmp_dir_.string(), cfg));

    EXPECT_FLOAT_EQ(cfg.temperature, 0.0f);
}

// Missing file is a soft failure — caller falls back to family preset.
TEST_F(GenerationConfigTest, MissingFileReturnsFalse) {
    HFConfigLoader::GenerationConfig cfg;
    EXPECT_FALSE(HFConfigLoader::load_generation_config(tmp_dir_.string(), cfg));
    EXPECT_LT(cfg.temperature, 0.0f);
    EXPECT_TRUE(cfg.eos_token_ids.empty());
}

} // namespace
