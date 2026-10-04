// Golden ModelConfig dumps for every HF config.json the unit lane fed load_config, plus syn_*
// configs for branches no test reached (#2537). Each <name>.json pairs with <name>.golden.
// Mismatch: the actual dump lands in <tmp>/imp_hf_config_golden/<name>.golden.

#include "model/hf_config_loader.h"
#include "model/model_config.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using imp::HFConfigLoader;
using imp::ModelConfig;

namespace {

class Dump {
public:
    void i(const char* k, long long v) { line(k, std::to_string(v)); }
    void f(const char* k, double v) {
        char b[64];
        std::snprintf(b, sizeof b, "%a", v);
        line(k, b);
    }
    void s(const char* k, const std::string& v) { line(k, "\"" + v + "\""); }
    template <typename T>
    void vi(const char* k, const std::vector<T>& v) {
        std::string o = "[";
        for (size_t n = 0; n < v.size(); ++n)
            o += (n ? "," : "") + std::to_string(static_cast<long long>(v[n]));
        line(k, o + "]");
    }
    void vf(const char* k, const std::vector<float>& v) {
        std::string o = "[";
        for (size_t n = 0; n < v.size(); ++n) {
            char b[64];
            std::snprintf(b, sizeof b, "%s%a", n ? "," : "", static_cast<double>(v[n]));
            o += b;
        }
        line(k, o + "]");
    }
    void vs(const char* k, const std::vector<std::string>& v) {
        std::string o = "[";
        for (size_t n = 0; n < v.size(); ++n)
            o += (n ? ",\"" : "\"") + v[n] + "\"";
        line(k, o + "]");
    }
    void line(const char* k, const std::string& v) { out_ += std::string(k) + "=" + v + "\n"; }
    const std::string& str() const { return out_; }

private:
    std::string out_;
};

// Every ModelConfig data member, in declaration order (src/model/model_config.h).
std::string dump_config(const ModelConfig& c) {
    Dump d;
    d.i("arch", static_cast<int>(c.arch));
    d.i("n_layers", c.n_layers);
    d.i("n_heads", c.n_heads);
    d.i("n_kv_heads", c.n_kv_heads);
    d.i("d_model", c.d_model);
    d.i("d_ff", c.d_ff);
    d.i("vocab_size", c.vocab_size);
    d.i("max_seq_len", c.max_seq_len);
    d.i("head_dim", c.head_dim);
    d.f("rope_theta", c.rope_theta);
    d.f("rms_norm_eps", c.rms_norm_eps);
    d.f("rope_freq_scale", c.rope_freq_scale);
    d.f("embed_scale", c.embed_scale);
    d.f("attn_scale", c.attn_scale);
    d.f("logits_scaling", c.logits_scaling);
    d.f("norm_weight_offset", c.norm_weight_offset);
    d.i("n_experts", c.n_experts);
    d.i("n_experts_active", c.n_experts_active);
    d.i("expert_d_ff", c.expert_d_ff);
    d.vi("n_kv_heads_per_layer", c.n_kv_heads_per_layer);
    d.vi("d_ff_per_layer", c.d_ff_per_layer);
    d.vi("head_dim_per_layer", c.head_dim_per_layer);
    d.vi("n_heads_per_layer", c.n_heads_per_layer);
    d.vi("swa_layers", c.swa_layers);
    d.f("rope_theta_swa", c.rope_theta_swa);
    d.vf("rope_inv_freqs_global", c.rope_inv_freqs_global);
    d.i("ssm_conv_kernel", c.ssm_conv_kernel);
    d.i("ssm_state_size", c.ssm_state_size);
    d.i("ssm_group_count", c.ssm_group_count);
    d.i("ssm_inner_size", c.ssm_inner_size);
    d.i("ssm_dt_rank", c.ssm_dt_rank);
    d.i("gdn_grouped_head_layout", c.gdn_grouped_head_layout);
    d.i("hc_count", c.hc_count);
    d.i("hc_lowrank", c.hc_lowrank);
    d.i("ple_eos_token_id", c.ple_eos_token_id);
    d.i("gdn_gate_sigmoid", c.gdn_gate_sigmoid);
    d.i("rope_dim", c.rope_dim);
    d.i("qsa_budget", c.qsa_budget);
    d.i("qsa_ratio", c.qsa_ratio);
    d.i("rope_neox", c.rope_neox);
    d.vi("mrope_section", std::vector<int>(c.mrope_section, c.mrope_section + 3));
    d.i("mrope_interleaved", c.mrope_interleaved);
    d.i("kv_lora_rank", c.kv_lora_rank);
    d.i("q_lora_rank", c.q_lora_rank);
    d.i("qk_rope_head_dim", c.qk_rope_head_dim);
    d.i("qk_nope_head_dim", c.qk_nope_head_dim);
    d.i("v_head_dim", c.v_head_dim);
    d.f("mla_mscale", c.mla_mscale);
    d.f("mla_mscale_num", c.mla_mscale_num);
    d.i("first_k_dense_replace", c.first_k_dense_replace);
    d.i("rope_attn_disabled", c.rope_attn_disabled);
    d.f("yarn_ext_factor", c.yarn_ext_factor);
    d.f("yarn_attn_factor", c.yarn_attn_factor);
    d.f("yarn_beta_fast", c.yarn_beta_fast);
    d.f("yarn_beta_slow", c.yarn_beta_slow);
    d.i("rope_n_ctx_orig", c.rope_n_ctx_orig);
    d.f("attn_temp_scale", c.attn_temp_scale);
    d.i("attn_temp_floor", c.attn_temp_floor);
    d.vf("rope_short_factor", c.rope_short_factor);
    d.vf("rope_long_factor", c.rope_long_factor);
    d.i("rope_scaling_orig_max_pos", c.rope_scaling_orig_max_pos);
    d.i("sliding_window", c.sliding_window);
    d.i("sliding_window_pattern", c.sliding_window_pattern);
    d.f("rope_local_theta", c.rope_local_theta);
    d.i("ffn_activation", static_cast<int>(c.ffn_activation));
    d.i("norm_placement", static_cast<int>(c.norm_placement));
    d.i("n_experts_shared", c.n_experts_shared);
    d.i("expert_shared_d_ff", c.expert_shared_d_ff);
    d.f("expert_weights_scale", c.expert_weights_scale);
    d.i("expert_weights_norm", c.expert_weights_norm);
    d.i("moe_sigmoid_gating", c.moe_sigmoid_gating);
    d.f("attn_logit_softcap", c.attn_logit_softcap);
    d.f("final_logit_softcap", c.final_logit_softcap);
    d.i("mxfp4_hadamard_attn", c.mxfp4_hadamard_attn);
    d.i("mxfp4_hadamard_ffn", c.mxfp4_hadamard_ffn);
    d.i("multimodal_wrapper", c.multimodal_wrapper);
    d.i("has_audio_config", c.has_audio_config);
    d.i("rope_scaling_unhandled", c.rope_scaling_unhandled);
    d.i("is_nvfp4_prequant", c.is_nvfp4_prequant);
    d.i("nvfp4_group_size", c.nvfp4_group_size);
    d.i("is_llm_compressor_nvfp4", c.is_llm_compressor_nvfp4);
    d.vs("nvfp4_exclude_modules", c.nvfp4_exclude_modules);
    d.i("is_mxfp4_prequant", c.is_mxfp4_prequant);
    d.i("mxfp4_block_size", c.mxfp4_block_size);
    d.i("is_awq_prequant", c.is_awq_prequant);
    d.i("awq_group_size", c.awq_group_size);
    d.s("kv_cache_quant_hint", c.kv_cache_quant_hint);
    d.i("tie_word_embeddings", c.tie_word_embeddings);
    d.i("attention_bias", c.attention_bias);
    d.i("mlp_bias", c.mlp_bias);
    d.i("arch_inferred_fallback", c.arch_inferred_fallback);
    d.i("overrides.gemma4.force_mmvq", c.overrides.gemma4.force_mmvq);
    return d.str();
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// load_config on `json` in a scratch dir: outcome line plus the full dump.
std::string load_and_dump(const fs::path& json, const fs::path& scratch) {
    fs::create_directories(scratch);
    fs::copy_file(json, scratch / "config.json", fs::copy_options::overwrite_existing);
    ModelConfig cfg;
    std::string outcome;
    try {
        outcome = HFConfigLoader::load_config(scratch.string(), cfg) ? "result=true" : "result=false";
    } catch (const std::exception& e) {
        outcome = std::string("result=throw ") + e.what();
    }
    return outcome + "\n" + dump_config(cfg);
}

TEST(HFConfigGolden, EveryFixtureMatchesItsGolden) {
    const fs::path dir = fs::path(IMP_TEST_FIXTURES_DIR) / "hf_config_golden";
    std::vector<fs::path> jsons;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".json")
            jsons.push_back(e.path());
    std::sort(jsons.begin(), jsons.end());
    ASSERT_EQ(jsons.size(), 95u) << "corpus size changed: regenerate goldens on the base commit";

    const fs::path tmp = fs::temp_directory_path() / ("imp_hf_config_golden_" + std::to_string(::getpid()));
    const fs::path actual_dir = fs::temp_directory_path() / "imp_hf_config_golden";
    fs::create_directories(actual_dir);
    int mismatches = 0;
    for (const auto& j : jsons) {
        const std::string name = j.stem().string();
        const std::string actual = load_and_dump(j, tmp / name);
        std::ofstream(actual_dir / (name + ".golden"), std::ios::binary) << actual;
        const fs::path golden = dir / (name + ".golden");
        if (!fs::exists(golden) || read_file(golden) != actual) {
            ++mismatches;
            ADD_FAILURE() << name << ": dump differs from " << golden.string() << ", actual in "
                          << (actual_dir / (name + ".golden")).string();
        }
    }
    fs::remove_all(tmp);
    EXPECT_EQ(mismatches, 0);
}

}  // namespace
