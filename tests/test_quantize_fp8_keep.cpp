// Mixed NVFP4 + FP8 exports (roadmap row 101): which weights a unit names, how a budget picks
// units, and that the MIXED_PRECISION table the writer declares is what the loader counts. Each
// fails silently when wrong: a unit splitting q from k/v, or a table the loader reads as plain NVFP4.

#include "../tools/imp-quantize/checkpoint_out.h"
#include "../tools/imp-quantize/fp8_keep.h"
#include "../tools/imp-quantize/tensor_policy.h"

#include "model/hf_config_loader.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace imp::quantize {
namespace {

TEST(Fp8Keep, UnitOfNamesTheFusedGroup) {
    EXPECT_EQ(fp8_unit_of("model.layers.0.self_attn.q_proj.weight"), "0.qkv");
    EXPECT_EQ(fp8_unit_of("model.layers.0.self_attn.k_proj.weight"), "0.qkv");
    EXPECT_EQ(fp8_unit_of("model.layers.0.self_attn.v_proj.weight"), "0.qkv");
    EXPECT_EQ(fp8_unit_of("model.layers.12.self_attn.o_proj.weight"), "12.o");
    EXPECT_EQ(fp8_unit_of("model.layers.27.mlp.gate_proj.weight"), "27.gate_up");
    EXPECT_EQ(fp8_unit_of("model.layers.27.mlp.up_proj.weight"), "27.gate_up");
    EXPECT_EQ(fp8_unit_of("model.layers.27.mlp.down_proj.weight"), "27.down");
    EXPECT_EQ(fp8_unit_of("model.language_model.layers.3.mlp.down_proj.weight"), "3.down");
}

TEST(Fp8Keep, UnitOfRefusesEverythingElse) {
    for (const char* n :
         {"model.layers.3.mlp.experts.7.down_proj.weight",
          "model.layers.3.mlp.shared_expert.down_proj.weight", "model.layers.3.linear_attn.out_proj.weight",
          "model.layers.3.input_layernorm.weight", "model.layers.3.self_attn.q_norm.weight",
          "model.layers.3.mlp.down_proj.weight_scale", "model.embed_tokens.weight", "lm_head.weight",
          "model.layers.x.mlp.down_proj.weight"})
        EXPECT_EQ(fp8_unit_of(n), "") << n;
}

TEST(Fp8Keep, KeepListParsesAndMatches) {
    std::vector<std::string> list;
    std::string err;
    ASSERT_TRUE(parse_fp8_keep_list("*.down,27.gate_up", list, err)) << err;
    EXPECT_TRUE(fp8_keep_list_matches(list, "0.down"));
    EXPECT_TRUE(fp8_keep_list_matches(list, "27.gate_up"));
    EXPECT_FALSE(fp8_keep_list_matches(list, "26.gate_up"));
    EXPECT_FALSE(fp8_keep_list_matches(list, "27.qkv"));
    EXPECT_FALSE(fp8_keep_list_matches(list, ""));
    for (const char* bad : {"", "27", "27.ffn", "a.down", ".down", "27.down,", "27.q"}) {
        std::vector<std::string> l;
        EXPECT_FALSE(parse_fp8_keep_list(bad, l, err)) << bad;
    }
}

TEST(Fp8Keep, RankTextParses) {
    std::map<std::string, double> s;
    std::string err;
    ASSERT_TRUE(parse_fp8_rank_text("# unit score\n27.down 0.5\n\n3.qkv -1e-2  # worse\n", s, err)) << err;
    ASSERT_EQ(s.size(), 2u);
    EXPECT_DOUBLE_EQ(s["27.down"], 0.5);
    EXPECT_DOUBLE_EQ(s["3.qkv"], -0.01);
    for (const char* bad :
         {"27.down\n", "27.down x\n", "*.down 1\n", "27.down 1 2\n", "27.ffn 1\n", "# none\n"}) {
        std::map<std::string, double> t;
        EXPECT_FALSE(parse_fp8_rank_text(bad, t, err)) << bad;
    }
}

TEST(Fp8Keep, ByteMath) {
    EXPECT_EQ(fp8_output_bytes(2, 16), 2u * 16u + 4u);
    // NVFP4 [2,16] = 16 nibble bytes + 2 micro-scales + 4.
    EXPECT_EQ(nvfp4_output_bytes(2, 16), 22u);
    EXPECT_EQ(fp8_extra_bytes(2, 16), 14u);
    EXPECT_EQ(fp8_output_bytes(0, 16), 0u);
}

TEST(Fp8Keep, BudgetPicksByScorePerByte) {
    // density: a 10/100 = 0.1, b 30/200 = 0.15, c 1/10 = 0.1, d 0, e < 0
    const std::vector<Fp8Unit> u = {
        {"0.qkv", 10, 100}, {"0.down", 30, 200}, {"1.o", 1, 10}, {"1.down", 0, 5}, {"2.down", -1, 5}};
    EXPECT_EQ(select_fp8_units(u, 0), (std::set<std::string>{}));
    EXPECT_EQ(select_fp8_units(u, 200), (std::set<std::string>{"0.down"}));
    // 0.down (200) first, 0.qkv (100) does not fit in 250, 1.o (10) still does.
    EXPECT_EQ(select_fp8_units(u, 250), (std::set<std::string>{"0.down", "1.o"}));
    EXPECT_EQ(select_fp8_units(u, 1000), (std::set<std::string>{"0.down", "0.qkv", "1.o"}));
}

TEST(Fp8Keep, OutputsAreModeloptFp8) {
    const uint8_t codes[4] = {1, 2, 3, 4};
    const float scale = 0.5f;
    const auto o = fp8_weight_outputs("model.layers.1.mlp.down_proj.weight", 2, 2, codes, &scale);
    ASSERT_EQ(o.size(), 2u);
    EXPECT_EQ(o[0].name, "model.layers.1.mlp.down_proj.weight");
    EXPECT_EQ(o[0].dtype, "F8_E4M3");
    EXPECT_EQ(o[0].shape, (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(o[0].nbytes, 4u);
    EXPECT_EQ(o[1].name, "model.layers.1.mlp.down_proj.weight_scale");
    EXPECT_EQ(o[1].dtype, "F32");
    EXPECT_EQ(o[1].nbytes, sizeof(float));
}

TEST(Fp8Keep, InputScaleIsTheScaledActivationMax) {
    // x' = x / s: channel 1 has max 8 but s 4 -> 2; channel 0 max 3, s 1 -> 3 wins.
    EXPECT_FLOAT_EQ(fp8_input_scale({3.0f, 8.0f}, {1.0f, 4.0f}, 2), 3.0f / 448.0f);
    EXPECT_FLOAT_EQ(fp8_input_scale({3.0f, 8.0f}, {}, 2), 8.0f / 448.0f);
    EXPECT_EQ(fp8_input_scale({3.0f}, {}, 2), 0.0f);  // wrong K: none rather than a guess
    EXPECT_EQ(fp8_input_scale({0.0f, 0.0f}, {}, 2), 0.0f);
    const uint8_t codes[1] = {0};
    const float s = 1.0f, in_s = 0.25f;
    const auto o = fp8_weight_outputs("model.layers.0.self_attn.o_proj.weight", 1, 1, codes, &s, &in_s);
    ASSERT_EQ(o.size(), 3u);
    EXPECT_EQ(o[2].name, "model.layers.0.self_attn.o_proj.input_scale");
    EXPECT_EQ(o[2].data, &in_s);
}

struct TempDir {
    std::string path;
    TempDir() {
        path = (std::filesystem::temp_directory_path() / ("fp8keep_" + std::to_string(::getpid()))).string();
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::string read(const std::string& name) const {
        std::ifstream f(path + "/" + name);
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }
};

TEST(Fp8Keep, LoaderCountsTheMixedPrecisionTable) {
    TempDir d;
    const std::string layers = mixed_precision_layers_json({"model.layers.0.self_attn.q_proj",
                                                            "model.layers.0.mlp.down_proj"},
                                                           {"model.layers.1.mlp.down_proj"}, "    ");
    ASSERT_TRUE(write_modelopt_quant_config(d.path, {"lm_head"}, true, true, "{}", layers));
    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(d.path, cfg)) << d.read("hf_quant_config.json");
    EXPECT_TRUE(cfg.mixed_precision);
    EXPECT_EQ(cfg.n_nvfp4_tensors, 2);
    EXPECT_EQ(cfg.n_fp8_tensors, 1);
    EXPECT_EQ(cfg.n_other_tensors, 0);
    EXPECT_EQ(cfg.group_size, 16);
    EXPECT_EQ(cfg.kv_cache_quant_algo, "FP8");
    ASSERT_EQ(cfg.exclude_modules.size(), 1u);

    // No FP8 unit: the declaration stays plain NVFP4.
    imp::HFConfigLoader::NvFP4Config plain;
    ASSERT_TRUE(write_modelopt_quant_config(d.path, {}, true, false, "{}"));
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(d.path, plain));
    EXPECT_FALSE(plain.mixed_precision);
    EXPECT_EQ(plain.n_fp8_tensors, 0);
}

}  // namespace
}  // namespace imp::quantize
