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

// ---------------------------------------------------------------------------
// special_tokens_map.json — authoritative additional_special_tokens list
// ---------------------------------------------------------------------------

class SpecialTokensMapTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override {
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("imp_test_stm_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(tmp_dir_);
    }

    void write_stm(const std::string& json) {
        std::ofstream f(tmp_dir_ / "special_tokens_map.json");
        f << json;
    }
};

// Mistral-3.2-style: object form for bos/eos/pad/unk + flat-string array for
// additional_special_tokens (with [INST], [TOOL_CALLS], etc.).
TEST_F(SpecialTokensMapTest, MistralObjectFormParsing) {
    write_stm(R"({
        "additional_special_tokens": [
            "<unk>", "<s>", "</s>", "[INST]", "[/INST]",
            "[AVAILABLE_TOOLS]", "[TOOL_CALLS]"
        ],
        "bos_token": {"content": "<s>", "lstrip": false},
        "eos_token": {"content": "</s>", "lstrip": false},
        "pad_token": {"content": "<pad>", "lstrip": false},
        "unk_token": {"content": "<unk>", "lstrip": false}
    })");

    HFConfigLoader::SpecialTokensMap stm;
    ASSERT_TRUE(HFConfigLoader::load_special_tokens_map(tmp_dir_.string(), stm));

    ASSERT_EQ(stm.additional_special_tokens.size(), 7u);
    EXPECT_EQ(stm.additional_special_tokens[3], "[INST]");
    EXPECT_EQ(stm.additional_special_tokens[6], "[TOOL_CALLS]");
    EXPECT_EQ(stm.bos_token, "<s>");
    EXPECT_EQ(stm.eos_token, "</s>");
    EXPECT_EQ(stm.pad_token, "<pad>");
    EXPECT_EQ(stm.unk_token, "<unk>");
}

// Qwen3-Coder-style: plain-string form for eos/pad, no bos/unk declared.
TEST_F(SpecialTokensMapTest, QwenPlainStringForm) {
    write_stm(R"({
        "additional_special_tokens": [
            "<|im_start|>", "<|im_end|>", "<|object_ref_start|>"
        ],
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>"
    })");

    HFConfigLoader::SpecialTokensMap stm;
    ASSERT_TRUE(HFConfigLoader::load_special_tokens_map(tmp_dir_.string(), stm));

    ASSERT_EQ(stm.additional_special_tokens.size(), 3u);
    EXPECT_EQ(stm.additional_special_tokens[0], "<|im_start|>");
    EXPECT_EQ(stm.eos_token, "<|endoftext|>");
    EXPECT_EQ(stm.pad_token, "<|endoftext|>");
    EXPECT_TRUE(stm.bos_token.empty());
    EXPECT_TRUE(stm.unk_token.empty());
}

TEST_F(SpecialTokensMapTest, MissingFileReturnsFalse) {
    HFConfigLoader::SpecialTokensMap stm;
    EXPECT_FALSE(HFConfigLoader::load_special_tokens_map(tmp_dir_.string(), stm));
    EXPECT_TRUE(stm.additional_special_tokens.empty());
}

// ---------------------------------------------------------------------------
// tokenizer_config.json — author-side flags (add_bos_token, etc.)
// ---------------------------------------------------------------------------

class TokenizerFlagsTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override {
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("imp_test_tflags_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(tmp_dir_);
    }

    void write_config(const std::string& json) {
        std::ofstream f(tmp_dir_ / "tokenizer_config.json");
        f << json;
    }
};

// Qwen3-Coder-style: BPE tokenizer that explicitly disables auto-BOS.
// Without this fix the SafeTensors path silently auto-prepended the
// pad/eos token to every prompt.
TEST_F(TokenizerFlagsTest, AddBosFalse) {
    write_config(R"({
        "add_bos_token": false,
        "add_prefix_space": false,
        "tokenizer_class": "Qwen2Tokenizer"
    })");

    HFConfigLoader::TokenizerFlags flags;
    ASSERT_TRUE(HFConfigLoader::load_tokenizer_flags(tmp_dir_.string(), flags));

    EXPECT_EQ(flags.add_bos_token, 0);
    EXPECT_EQ(flags.add_prefix_space, 0);
    EXPECT_LT(flags.add_eos_token, 0);  // unset
}

// Mistral-3.2-style: BOS yes, EOS no, prefix_space null (treated as unset).
TEST_F(TokenizerFlagsTest, MistralBosTrueEosFalse) {
    write_config(R"({
        "add_bos_token": true,
        "add_eos_token": false,
        "add_prefix_space": null,
        "tokenizer_class": "LlamaTokenizer"
    })");

    HFConfigLoader::TokenizerFlags flags;
    ASSERT_TRUE(HFConfigLoader::load_tokenizer_flags(tmp_dir_.string(), flags));

    EXPECT_EQ(flags.add_bos_token, 1);
    EXPECT_EQ(flags.add_eos_token, 0);
    EXPECT_LT(flags.add_prefix_space, 0);  // null → unset
}

// Mistral-3.2-style: author opts out of the chat_template.jinja's hardcoded
// default system prompt. Flag must propagate so apply() can suppress it.
TEST_F(TokenizerFlagsTest, UseDefaultSystemPromptFalse) {
    write_config(R"({
        "add_bos_token": true,
        "use_default_system_prompt": false
    })");

    HFConfigLoader::TokenizerFlags flags;
    ASSERT_TRUE(HFConfigLoader::load_tokenizer_flags(tmp_dir_.string(), flags));

    EXPECT_EQ(flags.use_default_system_prompt, 0);
}

// Gemma-4-style: tokenizer_config.json doesn't declare any of these flags;
// metadata lives in tokenizer.json instead. All fields stay at sentinel,
// caller falls back to its tokenizer-type-driven default.
TEST_F(TokenizerFlagsTest, AllUnsetFallsThrough) {
    write_config(R"({
        "tokenizer_class": "Gemma4Tokenizer",
        "padding_side": "left"
    })");

    HFConfigLoader::TokenizerFlags flags;
    ASSERT_TRUE(HFConfigLoader::load_tokenizer_flags(tmp_dir_.string(), flags));

    EXPECT_LT(flags.add_bos_token, 0);
    EXPECT_LT(flags.add_eos_token, 0);
    EXPECT_LT(flags.add_prefix_space, 0);
}

TEST_F(TokenizerFlagsTest, MissingFileReturnsFalse) {
    HFConfigLoader::TokenizerFlags flags;
    EXPECT_FALSE(HFConfigLoader::load_tokenizer_flags(tmp_dir_.string(), flags));
    EXPECT_LT(flags.add_bos_token, 0);
}

} // namespace
