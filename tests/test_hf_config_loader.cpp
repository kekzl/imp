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
        tmp_dir_ = std::filesystem::temp_directory_path() / ("imp_test_gencfg_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(tmp_dir_); }

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
        tmp_dir_ = std::filesystem::temp_directory_path() / ("imp_test_stm_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(tmp_dir_); }

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
        tmp_dir_ = std::filesystem::temp_directory_path() / ("imp_test_tflags_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(tmp_dir_); }

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

// ---- RoPE scaling ----

class RopeScalingConfigTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override {
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("imp_test_rope_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(tmp_dir_); }

    void write_config(const std::string& json) {
        std::ofstream f(tmp_dir_ / "config.json");
        f << json;
    }
};

// Llama-3.x rope_scaling.type=="llama3": per-frequency factor table.
// We feed a config matching meta-llama/Llama-3.1-8B-Instruct's published values
// and check that rope_short_factor / rope_long_factor get populated with one
// entry per rope-pair, that the highest-frequency dim survives unscaled
// (factor=1.0) and the lowest-frequency dim is fully scaled (factor=8.0).
TEST_F(RopeScalingConfigTest, Llama3PerFrequencyFactorTable) {
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 131072,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        },
        "vocab_size": 128256
    })");

    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));

    const int head_dim = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
    const int pairs = head_dim / 2;
    ASSERT_EQ(static_cast<int>(cfg.rope_short_factor.size()), pairs);
    ASSERT_EQ(static_cast<int>(cfg.rope_long_factor.size()), pairs);
    EXPECT_EQ(cfg.rope_scaling_orig_max_pos, 8192);

    // Llama-3 scaling is sequence-length independent, so short and long must match.
    for (int i = 0; i < pairs; i++) {
        EXPECT_FLOAT_EQ(cfg.rope_short_factor[i], cfg.rope_long_factor[i]);
    }

    // Highest-frequency pair (i=0) → wavelen = 2π → < high_wavelen (8192/4=2048).
    // No scaling: factor = 1.0.
    EXPECT_FLOAT_EQ(cfg.rope_short_factor[0], 1.0f);

    // Lowest-frequency pair (i=pairs-1) → wavelen >> low_wavelen (8192/1=8192).
    // Full scaling: factor = 8.0.
    EXPECT_FLOAT_EQ(cfg.rope_short_factor[pairs - 1], 8.0f);

    // Monotonically non-decreasing: shorter wavelengths get smaller factors.
    for (int i = 1; i < pairs; i++) {
        EXPECT_GE(cfg.rope_short_factor[i], cfg.rope_short_factor[i - 1] - 1e-5f);
    }

    // At least one pair sits in the smooth zone between low and high wavelen
    // boundaries — i.e., factor strictly between 1.0 and 8.0.
    bool saw_smooth = false;
    for (int i = 0; i < pairs; i++) {
        if (cfg.rope_short_factor[i] > 1.0f + 1e-3f &&
            cfg.rope_short_factor[i] < 8.0f - 1e-3f) {
            saw_smooth = true;
            break;
        }
    }
    EXPECT_TRUE(saw_smooth) << "expected a transition zone between high/low freq";
}

// tie_word_embeddings is now stored as tri-state so the SafeTensors loader
// can cross-check against actual lm_head.weight presence rather than
// silently tying on null.
TEST_F(RopeScalingConfigTest, TieWordEmbeddingsTriState) {
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "tie_word_embeddings": false
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.tie_word_embeddings, 0);

    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "tie_word_embeddings": true
    })");
    imp::ModelConfig cfg2;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg2));
    EXPECT_EQ(cfg2.tie_word_embeddings, 1);

    // Field absent → tri-state stays at -1 (unset); loader falls back to
    // null-detection.
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32
    })");
    imp::ModelConfig cfg3;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg3));
    EXPECT_EQ(cfg3.tie_word_embeddings, -1);
}

// Unknown architecture and unknown model_type both surface
// arch_inferred_fallback so callers can decide to warn loudly.
TEST_F(RopeScalingConfigTest, UnknownArchSetsFallbackFlag) {
    write_config(R"({
        "architectures": ["BogusForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_TRUE(cfg.arch_inferred_fallback);
    EXPECT_EQ(cfg.arch, imp::ModelArch::GENERIC);

    // Recognized arch → flag stays false.
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32
    })");
    imp::ModelConfig cfg2;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg2));
    EXPECT_FALSE(cfg2.arch_inferred_fallback);
    EXPECT_EQ(cfg2.arch, imp::ModelArch::LLAMA);
}

// Sanity: missing original_max_position_embeddings or factor<=1 → skip
// (warn-and-noop), don't populate the factor table.
TEST_F(RopeScalingConfigTest, Llama3DegenerateConfigSkipped) {
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 1.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        }
    })");

    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_TRUE(cfg.rope_short_factor.empty());
    EXPECT_TRUE(cfg.rope_long_factor.empty());
}

}  // namespace
