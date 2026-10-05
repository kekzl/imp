#include "model/hf_config_loader.h"
#include "model/model_config.h"
#include "model/model_limits.h"

#include <gtest/gtest.h>

#include <cmath>
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

// Llama-3.x rope_scaling.type=="llama3": per-frequency factor table. Feeds
// meta-llama/Llama-3.1-8B-Instruct's published config values and checks
// rope_short_factor/rope_long_factor get one entry per rope-pair, highest-frequency dim
// unscaled (factor=1.0), lowest-frequency dim fully scaled (factor=8.0).
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

// LFM2-MoE: conv layers ride the SSM pool (inner = hidden, kernel = conv_L_cache, short conv), attention
// layers from layer_types, num_dense_layers -> first_k_dense_replace, norm_eps read.
TEST_F(RopeScalingConfigTest, Lfm2ShortConvAndMoe) {
    write_config(R"({
        "architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe",
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 4,
        "conv_L_cache": 3, "norm_eps": 1e-05, "num_dense_layers": 2, "num_experts": 32,
        "num_experts_per_tok": 4, "moe_intermediate_size": 1792, "norm_topk_prob": true,
        "routed_scaling_factor": 1.0, "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "layer_types": ["conv", "conv", "full_attention", "conv"]
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, imp::ModelArch::LFM2);
    EXPECT_TRUE(cfg.ssm_short_conv);
    EXPECT_EQ(cfg.ssm_inner_size, 2048);
    EXPECT_EQ(cfg.ssm_conv_kernel, 3);
    EXPECT_EQ(cfg.ssm_proj_dim(), 3 * 2048);
    EXPECT_EQ(cfg.first_k_dense_replace, 2);
    EXPECT_FLOAT_EQ(cfg.rms_norm_eps, 1e-5f);
    EXPECT_TRUE(cfg.expert_weights_norm);
    EXPECT_EQ(cfg.n_kv_heads_per_layer, (std::vector<int>{0, 0, 8, 0}));
    EXPECT_FLOAT_EQ(cfg.rope_theta, 1e6f);
}

// Cohere2-MoE: sliding layers RoPE + window, full layers NoPE except the dense prefix (pattern 1),
// parallel block, sigmoid router without norm, prefix vs expert d_ff; shared experts refused.
TEST_F(RopeScalingConfigTest, Cohere2LayerMapAndParallelBlock) {
    write_config(R"({
        "architectures": ["Cohere2MoeForCausalLM"], "model_type": "cohere2_moe",
        "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4, "num_hidden_layers": 5,
        "head_dim": 128, "rope_theta": 50000, "sliding_window": 4096, "rms_norm_eps": 1e-06,
        "num_experts": 128, "num_experts_per_tok": 8, "intermediate_size": 768,
        "prefix_dense_intermediate_size": 3072, "first_k_dense_replace": 1,
        "prefix_dense_sliding_window_pattern": 1, "expert_selection_fn": "sigmoid", "norm_topk_prob": false,
        "num_shared_experts": 0, "use_parallel_block": true,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention",
                        "full_attention"]
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, imp::ModelArch::COHERE2);
    EXPECT_EQ(cfg.swa_layers, (std::vector<uint8_t>{0, 1, 1, 1, 0}));
    EXPECT_EQ(cfg.nope_layers, (std::vector<uint8_t>{0, 0, 0, 0, 1}));
    EXPECT_TRUE(cfg.parallel_block);
    EXPECT_TRUE(cfg.moe_sigmoid_gating);
    EXPECT_FALSE(cfg.expert_weights_norm);
    EXPECT_EQ(cfg.first_k_dense_replace, 1);
    EXPECT_EQ(cfg.d_ff, 3072);
    EXPECT_EQ(cfg.expert_d_ff, 768);
    EXPECT_EQ(cfg.sliding_window, 4096);
    EXPECT_FLOAT_EQ(cfg.rms_norm_eps, 1e-6f);

    write_config(R"({
        "architectures": ["Cohere2MoeForCausalLM"], "hidden_size": 64, "num_attention_heads": 4,
        "num_hidden_layers": 1, "num_shared_experts": 1, "layer_types": ["full_attention"]
    })");
    imp::ModelConfig bad;
    EXPECT_FALSE(HFConfigLoader::load_config(tmp_dir_.string(), bad));
}

// Olmo-3: layer_types 3 sliding + 1 full -> pattern 4; local layers keep rope_theta; YaRN parsed.
TEST_F(RopeScalingConfigTest, Olmo3LayerPatternAndYarn) {
    write_config(R"({
        "architectures": ["Olmo3ForCausalLM"], "model_type": "olmo3",
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 32,
        "num_hidden_layers": 8, "rope_theta": 500000, "sliding_window": 4096,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
                        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "rope_scaling": {"attention_factor": 1.2079441541679836, "beta_fast": 32.0, "beta_slow": 1.0,
                         "factor": 8.0, "original_max_position_embeddings": 8192, "rope_type": "yarn"}
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, imp::ModelArch::OLMO3);
    EXPECT_EQ(cfg.sliding_window_pattern, 4);
    EXPECT_EQ(cfg.sliding_window, 4096);
    EXPECT_FLOAT_EQ(cfg.rope_local_theta, 500000.0f);
    EXPECT_FLOAT_EQ(cfg.rope_freq_scale, 8.0f);
    EXPECT_FLOAT_EQ(cfg.yarn_ext_factor, 1.0f);
    EXPECT_EQ(cfg.rope_n_ctx_orig, 8192);

    // Not every-4th-full: refused rather than run with a wrong window map.
    write_config(R"({
        "architectures": ["Olmo3ForCausalLM"], "hidden_size": 64, "num_attention_heads": 4,
        "num_hidden_layers": 3, "layer_types": ["sliding_attention", "full_attention", "full_attention"]
    })");
    imp::ModelConfig bad;
    EXPECT_FALSE(HFConfigLoader::load_config(tmp_dir_.string(), bad));
}

// Granite (#2412): own arch, attention_multiplier replaces 1/sqrt(head_dim), the two residual-stream
// multipliers fold into embed_scale, logits_scaling != 1 is refused by the dimension validator.
TEST_F(RopeScalingConfigTest, GraniteMultipliers) {
    // ibm-granite/granite-4.2-8b config.json, the fields that matter here.
    write_config(R"({
        "architectures": ["GraniteForCausalLM"], "model_type": "granite",
        "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
        "num_hidden_layers": 40, "attention_multiplier": 0.0078125, "embedding_multiplier": 1.0,
        "residual_multiplier": 1.0, "logits_scaling": 1.0
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, imp::ModelArch::GRANITE);
    EXPECT_FALSE(cfg.arch_inferred_fallback);
    EXPECT_FLOAT_EQ(cfg.attn_scale, 0.0078125f);
    EXPECT_FLOAT_EQ(imp::attention_softmax_scale(cfg, false, 128), 0.0078125f);
    EXPECT_EQ(cfg.embed_scale, 0.0f);  // both multipliers 1.0: no embedding scale at all
    std::string err;
    EXPECT_TRUE(imp::validate_declared_dimensions(cfg, &err)) << err;

    // Granite 3.x shape: embedding 12, residual 0.22, logits 8.
    write_config(R"({
        "architectures": ["GraniteForCausalLM"], "hidden_size": 2048, "num_attention_heads": 32,
        "num_hidden_layers": 40, "attention_multiplier": 0.015625, "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22, "logits_scaling": 8.0
    })");
    imp::ModelConfig g3;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), g3));
    EXPECT_FLOAT_EQ(g3.embed_scale, 12.0f / 0.22f);
    EXPECT_FALSE(imp::validate_declared_dimensions(g3, &err));
    EXPECT_NE(err.find("logits_scaling"), std::string::npos) << err;

    // No multiplier keys: the default scale.
    write_config(R"({"architectures": ["LlamaForCausalLM"], "hidden_size": 4096,
        "num_attention_heads": 32, "num_hidden_layers": 32})");
    imp::ModelConfig llama;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), llama));
    EXPECT_FLOAT_EQ(imp::attention_softmax_scale(llama, false, 128), 1.0f / std::sqrt(128.0f));
    EXPECT_FLOAT_EQ(imp::attention_softmax_scale(llama, true, 256), 1.0f);
}

// Devstral-Small-2 / Ministral 3 (#2411): YaRN lives only in text_config.rope_parameters;
// mscale = mscale_all_dim = 1 makes HF's cos/sin factor 1.0; llama_4_scaling_beta 0.1 past 8192.
TEST_F(RopeScalingConfigTest, Ministral3YarnAndQueryTemperature) {
    write_config(R"({
        "architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
        "text_config": {
            "head_dim": 128, "hidden_size": 5120, "num_attention_heads": 32, "num_key_value_heads": 8,
            "num_hidden_layers": 40, "model_type": "ministral3",
            "rope_parameters": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 48.0,
                "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 8192, "rope_theta": 100000000.0,
                "rope_type": "yarn", "type": "yarn"}
        }
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, imp::ModelArch::MISTRAL);
    EXPECT_FLOAT_EQ(cfg.rope_theta, 1e8f);
    EXPECT_FLOAT_EQ(cfg.rope_freq_scale, 48.0f);
    EXPECT_FLOAT_EQ(cfg.yarn_ext_factor, 1.0f);
    EXPECT_EQ(cfg.rope_n_ctx_orig, 8192);
    // kernel applies 1 + 0.1 ln 48; HF wants get_mscale(48,1)/get_mscale(48,1) = 1.0
    EXPECT_NEAR(cfg.yarn_attn_factor * (1.0f + 0.1f * std::log(48.0f)), 1.0f, 1e-6f);
    EXPECT_FLOAT_EQ(cfg.attn_temp_scale, 0.1f);
    EXPECT_EQ(cfg.attn_temp_floor, 8192);

    // YaRN without mscale keys (gpt-oss shape): default get_mscale(factor), factor untouched.
    write_config(R"({"architectures": ["GptOssForCausalLM"], "hidden_size": 2880,
        "num_attention_heads": 64, "num_hidden_layers": 24,
        "rope_scaling": {"factor": 32.0, "original_max_position_embeddings": 4096, "rope_type": "yarn"}})");
    imp::ModelConfig oss;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), oss));
    EXPECT_FLOAT_EQ(oss.yarn_attn_factor, 1.0f);
    EXPECT_FLOAT_EQ(oss.attn_temp_scale, 0.0f);

    // GGUF spelling: yarn_log_multiplier 1 and attention.temperature_scale 0.1.
    imp::ModelConfig g;
    g.rope_freq_scale = 48.0f;
    g.yarn_ext_factor = 1.0f;
    g.rope_n_ctx_orig = 8192;
    imp::set_yarn_extras_gguf(g, [](const std::string& k, double def) {
        return k == "rope.scaling.yarn_log_multiplier" ? 1.0 : k == "attention.temperature_scale" ? 0.1 : def;
    });
    EXPECT_FLOAT_EQ(g.yarn_attn_factor, cfg.yarn_attn_factor);
    EXPECT_FLOAT_EQ(g.attn_temp_scale, 0.1f);
    EXPECT_EQ(g.attn_temp_floor, 8192);
}

// AWQ detection (audit gap #16). Both nested-under-quantization_config
// (HF standard) and standalone quant_config.json (older AutoAWQ) are
// recognised. Parsing only; the variant rule lives in load_safetensors (#2205).
TEST_F(RopeScalingConfigTest, AwqQuantConfigDetection) {
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "quantization_config": {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "zero_point": true,
            "version": "gemm"
        }
    })");
    HFConfigLoader::AWQConfig acfg;
    ASSERT_TRUE(HFConfigLoader::load_awq_config(tmp_dir_.string(), acfg));
    EXPECT_EQ(acfg.bits, 4);
    EXPECT_EQ(acfg.group_size, 128);
    EXPECT_TRUE(acfg.zero_point);
    EXPECT_EQ(acfg.version, "gemm");

    // Older AutoAWQ field names (w_bit, q_group_size) in standalone
    // quant_config.json (no nesting under quantization_config).
    write_config(R"({"hidden_size": 4096})");  // overwrite to avoid double-detect
    {
        std::ofstream f(tmp_dir_ / "quant_config.json");
        f << R"({"quant_method": "awq", "w_bit": 4, "q_group_size": 64, "zero_point": false})";
    }
    HFConfigLoader::AWQConfig acfg2;
    ASSERT_TRUE(HFConfigLoader::load_awq_config(tmp_dir_.string(), acfg2));
    EXPECT_EQ(acfg2.bits, 4);
    EXPECT_EQ(acfg2.group_size, 64);
    EXPECT_FALSE(acfg2.zero_point);
    std::filesystem::remove(tmp_dir_ / "quant_config.json");

    // Non-AWQ method → false.
    write_config(R"({
        "quantization_config": {"quant_method": "gptq", "bits": 4}
    })");
    HFConfigLoader::AWQConfig acfg3;
    EXPECT_FALSE(HFConfigLoader::load_awq_config(tmp_dir_.string(), acfg3));
}

// GPTQ config (#2249): config.json quantization_config (HF/optimum, e.g. Qwen *-GPTQ-Int4) or
// quantize_config.json; checkpoint_format from "checkpoint_format", "format" or is_marlin_format.
TEST_F(RopeScalingConfigTest, GptqQuantConfigDetection) {
    write_config(R"({"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128,
                     "desc_act": false, "sym": true}})");
    HFConfigLoader::GPTQConfig c1;
    ASSERT_TRUE(HFConfigLoader::load_gptq_config(tmp_dir_.string(), c1));
    EXPECT_EQ(c1.bits, 4);
    EXPECT_EQ(c1.group_size, 128);
    EXPECT_FALSE(c1.desc_act);
    EXPECT_EQ(c1.checkpoint_format, "");

    write_config(R"({"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": -1,
                     "desc_act": true, "checkpoint_format": "gptq_v2"}})");
    HFConfigLoader::GPTQConfig c2;
    ASSERT_TRUE(HFConfigLoader::load_gptq_config(tmp_dir_.string(), c2));
    EXPECT_EQ(c2.group_size, -1);
    EXPECT_TRUE(c2.desc_act);
    EXPECT_EQ(c2.checkpoint_format, "gptq_v2");

    // AWQ in config.json is not GPTQ.
    write_config(R"({"quantization_config": {"quant_method": "awq", "bits": 4}})");
    HFConfigLoader::GPTQConfig c3;
    EXPECT_FALSE(HFConfigLoader::load_gptq_config(tmp_dir_.string(), c3));

    // quantize_config.json wins over config.json; GPTQModel "format" key; is_marlin_format.
    {
        std::ofstream f(tmp_dir_ / "quantize_config.json");
        f << R"({"bits": 4, "group_size": 32, "format": "gptq_v2"})";
    }
    HFConfigLoader::GPTQConfig c4;
    ASSERT_TRUE(HFConfigLoader::load_gptq_config(tmp_dir_.string(), c4));
    EXPECT_EQ(c4.group_size, 32);
    EXPECT_EQ(c4.checkpoint_format, "gptq_v2");
    {
        std::ofstream f(tmp_dir_ / "quantize_config.json");
        f << R"({"bits": 4, "group_size": 128, "is_marlin_format": true})";
    }
    HFConfigLoader::GPTQConfig c5;
    ASSERT_TRUE(HFConfigLoader::load_gptq_config(tmp_dir_.string(), c5));
    EXPECT_EQ(c5.checkpoint_format, "marlin");
    std::filesystem::remove(tmp_dir_ / "quantize_config.json");
}

// DeepSeek V2/V3 MLA detection (audit gap #17). MLA-specific config
// fields trigger a load-time warning that imp's DEEPSEEK forward path
// uses standard MHA and will produce wrong outputs.
TEST_F(RopeScalingConfigTest, DeepseekMlaWarning) {
    write_config(R"({
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, imp::ModelArch::DEEPSEEK);
    // Detection only writes the WARN log; the test cannot easily inspect it,
    // but at least the load doesn't error out — caller continues with the
    // (incorrect) MHA path so no silent crash.

    // Non-MLA DeepSeek (no kv_lora_rank): no warn, normal load.
    write_config(R"({
        "architectures": ["DeepseekV2ForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32
    })");
    imp::ModelConfig cfg2;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg2));
    EXPECT_EQ(cfg2.arch, imp::ModelArch::DEEPSEEK);
}

// Multimodal model detection (audit gap #18). `vision_config` block
// presence triggers a warning that the vision tower will be skipped.
TEST_F(RopeScalingConfigTest, VisionConfigWarning) {
    write_config(R"({
        "architectures": ["Gemma3ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152
        }
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_EQ(cfg.arch, imp::ModelArch::GEMMA3);
    // vision_config triggers WARN; loader still succeeds.
}

// MXFP4 quantization config detection (audit gap #13): GPT-OSS and other MXFP4 SafeTensors
// exports declare quantization_config.quant_method=="mxfp4" at config.json top level; the
// loader sets a metadata flag so downstream code can warn the SafeTensors decode path isn't
// implemented (use GGUF for MXFP4 inference).
TEST_F(RopeScalingConfigTest, Mxfp4QuantConfigDetection) {
    write_config(R"({
        "architectures": ["GptOssForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "quantization_config": {
            "quant_method": "mxfp4",
            "block_size": 32
        }
    })");
    HFConfigLoader::MxFP4Config mcfg;
    ASSERT_TRUE(HFConfigLoader::load_mxfp4_config(tmp_dir_.string(), mcfg));
    EXPECT_EQ(mcfg.block_size, 32);

    // Default block_size when omitted: still 32.
    write_config(R"({
        "quantization_config": {
            "quant_method": "MXFP4"
        }
    })");
    HFConfigLoader::MxFP4Config mcfg2;
    ASSERT_TRUE(HFConfigLoader::load_mxfp4_config(tmp_dir_.string(), mcfg2));
    EXPECT_EQ(mcfg2.block_size, 32);

    // Non-MXFP4 quant_method → return false (no metadata to apply).
    write_config(R"({
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4
        }
    })");
    HFConfigLoader::MxFP4Config mcfg3;
    EXPECT_FALSE(HFConfigLoader::load_mxfp4_config(tmp_dir_.string(), mcfg3));

    // Missing quantization_config block → false.
    write_config(R"({"hidden_size": 4096})");
    HFConfigLoader::MxFP4Config mcfg4;
    EXPECT_FALSE(HFConfigLoader::load_mxfp4_config(tmp_dir_.string(), mcfg4));
}

// Llama-4 + Qwen3.5 non-MoE HF class names should now map to their
// existing imp enums (audit gap #15). Previously these silently
// downgraded to GENERIC.
TEST_F(RopeScalingConfigTest, NewlyMappedArchClassNames) {
    EXPECT_EQ(HFConfigLoader::map_architecture("Llama4ForCausalLM"),
              imp::ModelArch::LLAMA4);
    EXPECT_EQ(HFConfigLoader::map_architecture("Llama4ForConditionalGeneration"),
              imp::ModelArch::LLAMA4);
    EXPECT_EQ(HFConfigLoader::map_architecture("Qwen3_5ForCausalLM"),
              imp::ModelArch::QWEN35);
    EXPECT_EQ(HFConfigLoader::map_architecture("Qwen3_5ForConditionalGeneration"),
              imp::ModelArch::QWEN35);

    // Sanity: existing mappings unaffected.
    EXPECT_EQ(HFConfigLoader::map_architecture("LlamaForCausalLM"),
              imp::ModelArch::LLAMA);
    EXPECT_EQ(HFConfigLoader::map_architecture("Qwen3_5MoeForCausalLM"),
              imp::ModelArch::QWEN36_MOE);
    // Qwen3.8-Flash-Next ships model_type qwen4_exp under a ConditionalGeneration wrapper.
    EXPECT_EQ(HFConfigLoader::map_architecture("Qwen4ExpForConditionalGeneration"),
              imp::ModelArch::QWEN4_EXP);
    EXPECT_EQ(HFConfigLoader::map_architecture("Qwen4ExpForCausalLM"),
              imp::ModelArch::QWEN4_EXP);

    // Unknown arch still produces GENERIC (and emits a WARN).
    EXPECT_EQ(HFConfigLoader::map_architecture("BogusForCausalLM"),
              imp::ModelArch::GENERIC);
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

// audio_config: an unsupported modality must be detected from the object shape, not the key
// or the token id. Gemma-4-12B-NVFP4 writes audio_config as an object (model_type
// gemma4_unified_audio) and ships model.embed_audio.*; Gemma-4-26B-A4B-it-NVFP4 writes
// audio_config: null with no audio tensor. Both set audio_token_id 258881, so keying off
// presence or the token id would warn on both or neither incorrectly.

class AudioConfigTest : public ::testing::Test {
protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override {
        tmp_dir_ = std::filesystem::temp_directory_path() / ("imp_test_audio_" + std::to_string(::getpid()));
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(tmp_dir_); }

    void write_config(const std::string& json) {
        std::ofstream f(tmp_dir_ / "config.json");
        f << json;
    }
};

// Gemma-4-12B-NVFP4's shape.
TEST_F(AudioConfigTest, ObjectMarksTheModalityUnsupported) {
    write_config(R"({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "audio_token_id": 258881,
        "audio_config": {
            "model_type": "gemma4_unified_audio",
            "audio_embed_dim": 640
        },
        "text_config": {
            "hidden_size": 3840,
            "intermediate_size": 15360,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8
        }
    })");

    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_TRUE(cfg.has_audio_config);
}

// Gemma-4-26B-A4B-it-NVFP4's shape: the key is there and null. This is the
// false positive the object test exists to avoid.
TEST_F(AudioConfigTest, NullDoesNotMarkTheModality) {
    write_config(R"({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "audio_token_id": 258881,
        "audio_config": null,
        "text_config": {
            "hidden_size": 2816,
            "intermediate_size": 11264,
            "num_attention_heads": 16,
            "num_hidden_layers": 62,
            "num_key_value_heads": 4
        }
    })");

    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_FALSE(cfg.has_audio_config);
}

// rope_scaling: an unhandled spelling left rope_freq_scale at 1.0 with the model reporting
// its full declared context while rotating UNSCALED. Older Phi-3 exports spell LongRoPE `su`
// (longrope is the rename); dynamic_ntk appears in the wild too.
// Latent: the one local checkpoint that falls through is Qwen3-VL-4B with
// rope_type=="default", which means no scaling and wants exactly that - the silent exemption
// the third test pins.

TEST_F(RopeScalingConfigTest, UnhandledTypeIsFlagged) {
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "max_position_embeddings": 131072,
        "rope_theta": 10000.0,
        "rope_scaling": {"type": "su", "factor": 8.0}
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_TRUE(cfg.rope_scaling_unhandled);
    EXPECT_FLOAT_EQ(cfg.rope_freq_scale, 1.0f) << "nothing was applied, which is the point";
}

TEST_F(RopeScalingConfigTest, HandledTypeIsNotFlagged) {
    write_config(R"({
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "rope_scaling": {"type": "linear", "factor": 4.0}
    })");
    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_FALSE(cfg.rope_scaling_unhandled);
}

// Qwen3-VL-4B's shape. "default" and "none" mean no scaling, so falling through
// is correct and must stay silent; flagging it would cry wolf on a working
// model, the failure mode #1929's audio check was built to avoid.
TEST_F(RopeScalingConfigTest, DefaultAndNoneAreNotFlagged) {
    for (const char* t : {"default", "none"}) {
        write_config(std::string(R"({
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "hidden_size": 2560,
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "rope_scaling": {"rope_type": ")") +
                     t + R"("}
        })");
        imp::ModelConfig cfg;
        ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg)) << t;
        EXPECT_FALSE(cfg.rope_scaling_unhandled) << t;
    }
}

// Every text-only checkpoint: no key, no flag, no warning.
TEST_F(AudioConfigTest, AbsentKeyDoesNotMarkTheModality) {
    write_config(R"({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8
    })");

    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_FALSE(cfg.has_audio_config);
}

class Gemma4RopeConfigTest : public AudioConfigTest {};

// Gemma-4-26B-A4B full_attention: proportional RoPE, factor 0.25 at hd=512 rotates pairs
// 0..63 at theta^(-2p/512) and leaves 64..255 unrotated (GGUF rope_freqs: 64 x 1, 192 x 1e30).
TEST_F(Gemma4RopeConfigTest, FullAttentionProportionalRope) {
    write_config(R"({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2816, "intermediate_size": 2112, "num_attention_heads": 16,
            "num_hidden_layers": 6, "num_key_value_heads": 8, "num_global_key_value_heads": 2,
            "head_dim": 256, "global_head_dim": 512, "sliding_window": 1024,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                            "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                   "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        }
    })");

    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    ASSERT_EQ(cfg.rope_inv_freqs_global.size(), 256u);
    for (int p = 0; p < 256; ++p) {
        const float want = p < 64 ? static_cast<float>(std::pow(1e6, -2.0 * p / 512.0)) : 0.0f;
        EXPECT_FLOAT_EQ(cfg.rope_inv_freqs_global[p], want) << "pair " << p;
    }
    EXPECT_FLOAT_EQ(cfg.rope_theta, 1e6f);
    EXPECT_FLOAT_EQ(cfg.rope_theta_swa, 1e4f);
}

// No partial_rotary_factor: no table, global layers keep plain theta RoPE.
TEST_F(Gemma4RopeConfigTest, WithoutPartialFactorHasNoTable) {
    write_config(R"({
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 2816, "intermediate_size": 2112, "num_attention_heads": 16,
            "num_hidden_layers": 2, "num_key_value_heads": 8,
            "head_dim": 256, "global_head_dim": 512,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {"full_attention": {"rope_theta": 1000000.0}}
        }
    })");

    imp::ModelConfig cfg;
    ASSERT_TRUE(HFConfigLoader::load_config(tmp_dir_.string(), cfg));
    EXPECT_TRUE(cfg.rope_inv_freqs_global.empty());
}

}  // namespace
