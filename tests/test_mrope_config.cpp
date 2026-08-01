// Reading `mrope_section` out of an HF config.
//
// The section split lives under `rope_scaling` in the Qwen2-VL generation and
// under `rope_parameters` in Qwen3-VL. Reading only one of them leaves a
// multimodal model silently on single-axis RoPE — which still generates text,
// just with every image token positioned as if it were a text token.

#include "model/hf_config_loader.h"
#include "model/json_util.h"
#include "model/model_config.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace imp {
namespace {

// The loader reads a directory, so each case is written to one.
class MRopeConfig : public ::testing::Test {
protected:
    void SetUp() override {
        // mkdtemp creates the directory atomically and reports the name it
        // actually used. tmpnam only guesses one — which is what the linker
        // warns about — and that guess then went into `system("rm -rf " + dir)`
        // on teardown: a shell taking apart a path nobody had validated.
        const std::string tmpl = (std::filesystem::temp_directory_path() / "imp_mrope_XXXXXX").string();
        std::vector<char> buf(tmpl.c_str(), tmpl.c_str() + tmpl.size() + 1);
        ASSERT_NE(::mkdtemp(buf.data()), nullptr) << "could not create a temp directory";
        dir_ = buf.data();
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);  // no shell involved
    }

    ModelConfig load(const std::string& json) {
        std::ofstream f(dir_ + "/config.json");
        f << json;
        f.close();
        ModelConfig cfg;
        EXPECT_TRUE(HFConfigLoader::load_config(dir_, cfg));
        return cfg;
    }

    std::string dir_;
};

const char* kBody = R"("hidden_size": 2560, "num_attention_heads": 32, "num_key_value_heads": 8,
    "num_hidden_layers": 36, "vocab_size": 151936, "head_dim": 128)";

TEST_F(MRopeConfig, ReadsTheSplitFromRopeParameters) {
    const ModelConfig cfg = load(std::string("{") + kBody + R"(,
        "rope_parameters": {"rope_theta": 5000000.0, "mrope_section": [24, 20, 20],
                            "mrope_interleaved": true}})");
    EXPECT_TRUE(cfg.has_mrope());
    EXPECT_EQ(cfg.mrope_section[0], 24);
    EXPECT_EQ(cfg.mrope_section[1], 20);
    EXPECT_EQ(cfg.mrope_section[2], 20);
    EXPECT_TRUE(cfg.mrope_interleaved);
}

TEST_F(MRopeConfig, ReadsTheSplitFromRopeScaling) {
    const ModelConfig cfg = load(std::string("{") + kBody + R"(,
        "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]}})");
    EXPECT_TRUE(cfg.has_mrope());
    EXPECT_EQ(cfg.mrope_section[0], 16);
    EXPECT_EQ(cfg.mrope_section[1], 24);
    EXPECT_EQ(cfg.mrope_section[2], 24);
    EXPECT_FALSE(cfg.mrope_interleaved) << "absent means the contiguous layout";
}

// Every text-only model must come out with M-RoPE off, so nothing in the hot
// path changes for them.
TEST_F(MRopeConfig, TextOnlyConfigLeavesMRopeOff) {
    const ModelConfig cfg = load(std::string("{") + kBody + R"(,
        "rope_theta": 1000000.0, "rope_scaling": {"type": "yarn", "factor": 4.0}})");
    EXPECT_FALSE(cfg.has_mrope());
    EXPECT_EQ(cfg.mrope_section[0], 0);
    EXPECT_EQ(cfg.mrope_section[1], 0);
    EXPECT_EQ(cfg.mrope_section[2], 0);
}

// A malformed split must not half-apply: two of three axes would put the H and
// W rotations on the wrong dimensions.
TEST_F(MRopeConfig, MalformedSplitIsIgnoredWholesale) {
    for (const char* bad :
         {R"([24, 20])", R"([24, 20, 20, 20])", R"([24, "20", 20])", R"([24, -1, 20])", R"("24,20,20")"}) {
        const ModelConfig cfg = load(std::string("{") + kBody + R"(, "rope_parameters": {"mrope_section": )" +
                                     bad + "}}");
        EXPECT_FALSE(cfg.has_mrope()) << bad;
        EXPECT_EQ(cfg.mrope_section[0], 0) << bad;
    }
}

}  // namespace
}  // namespace imp
