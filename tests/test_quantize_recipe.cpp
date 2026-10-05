// imp-quantize export recipe (#2481): every field lands in hf_quant_config.json (modelopt) or
// imp_recipe.json (compressed-tensors), the same recipe writes the same bytes, and the loader
// still reads the declaration around it.

#include "../tools/imp-quantize/checkpoint_out.h"
#include "../tools/imp-quantize/options.h"
#include "../tools/imp-quantize/recipe.h"

#include "model/hf_config_loader.h"
#include "model/json_util.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

namespace imp::quantize {
namespace {

struct TempDir {
    std::string path;
    TempDir() {
        static int n = 0;
        path = (std::filesystem::temp_directory_path() /
                ("recipe_" + std::to_string(::getpid()) + "_" + std::to_string(n++)))
                   .string();
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::string read(const std::string& name) const {
        std::ifstream f(path + "/" + name, std::ios::binary);
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }
};

const JValue* member(const JValue& o, const std::string& key) {
    for (const auto& m : o.obj)
        if (m.key == key)
            return &m.value;
    return nullptr;
}

std::vector<std::string> keys(const JValue& o) {
    std::vector<std::string> out;
    for (const auto& m : o.obj)
        out.push_back(m.key);
    return out;
}

Recipe calibrated_recipe() {
    Recipe r;
    r.imp_version = "0.47.0";
    r.imp_tree = "0123abcd";
    r.format = "modelopt";
    r.lm_head = "fp8";
    r.kv_hint = "auto";
    r.kv_cache_fp8 = true;
    r.keep_gdn_proj = "all";
    r.calibrated = true;
    r.calib_sha256 = "ba7816bf";
    r.calib_model_id = "Qwen3-0.6B ppl_corpus_45k";
    r.calib_samples = 4096;
    r.calib_entries = 196;
    r.calib_groups = "BD";
    r.calib_weight = "abs";
    r.n_rep = 2;
    return r;
}

TEST(QuantizeRecipe, JsonCarriesEveryFieldInFixedOrder) {
    const std::string text = recipe_json(calibrated_recipe(), "");
    JsonParser p(text);
    const JValue v = p.parse();
    ASSERT_TRUE(p.ok()) << text;
    EXPECT_EQ(keys(v), (std::vector<std::string>{"recipe_version", "imp_version", "imp_tree", "format",
                                                 "lm_head", "kv_hint", "kv_cache_fp8", "keep_attn_gate",
                                                 "keep_gdn_proj", "gdn_proj_format", "calibration"}));
    EXPECT_EQ(member(v, "recipe_version")->as_int(), kRecipeVersion);
    EXPECT_EQ(member(v, "imp_tree")->str_val, "0123abcd");
    const JValue* c = member(v, "calibration");
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(keys(*c), (std::vector<std::string>{"method", "file_sha256", "model_id", "samples", "entries",
                                                  "groups", "weight", "n_rep", "hybrid"}));
    EXPECT_EQ(member(*c, "samples")->as_int(), 4096);
    EXPECT_EQ(member(*c, "groups")->str_val, "BD");
    EXPECT_EQ(member(*c, "n_rep")->as_int(), 2);
}

TEST(QuantizeRecipe, UncalibratedWritesNullCalibration) {
    Recipe r = calibrated_recipe();
    r.calibrated = false;
    const std::string text = recipe_json(r, "");
    JsonParser p(text);
    const JValue v = p.parse();
    ASSERT_TRUE(p.ok()) << text;
    EXPECT_EQ(member(v, "calibration")->type, JType::NUL);
}

TEST(QuantizeRecipe, HfQuantConfigCarriesTheRecipeAndStillLoads) {
    TempDir d;
    const auto wrote = write_modelopt_quant_config(d.path, {"lm_head"}, /*calibrated=*/true, /*kv_fp8=*/true,
                                                   recipe_json(calibrated_recipe(), "    "));
    ASSERT_TRUE(wrote) << wrote.error();
    const std::string text = d.read("hf_quant_config.json");
    JsonParser p(text);
    const JValue v = p.parse();
    ASSERT_TRUE(p.ok()) << text;
    const JValue* recipe = member(*member(v, "producer"), "recipe");
    ASSERT_NE(recipe, nullptr) << text;
    EXPECT_EQ(member(*recipe, "recipe_version")->as_int(), kRecipeVersion);
    EXPECT_EQ(member(*member(*recipe, "calibration"), "file_sha256")->str_val, "ba7816bf");
    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(d.path, cfg)) << text;
    EXPECT_EQ(cfg.kv_cache_quant_algo, "FP8");
}

TEST(QuantizeRecipe, SameRecipeSameBytesOneFieldApartDiffers) {
    TempDir a, b, c;
    const Recipe r = calibrated_recipe();
    Recipe other = r;
    other.calib_groups = "ABCD";
    ASSERT_TRUE(write_modelopt_quant_config(a.path, {"lm_head"}, true, true, recipe_json(r, "    ")));
    ASSERT_TRUE(write_modelopt_quant_config(b.path, {"lm_head"}, true, true, recipe_json(r, "    ")));
    ASSERT_TRUE(write_modelopt_quant_config(c.path, {"lm_head"}, true, true, recipe_json(other, "    ")));
    EXPECT_EQ(a.read("hf_quant_config.json"), b.read("hf_quant_config.json"));
    EXPECT_NE(a.read("hf_quant_config.json"), c.read("hf_quant_config.json"));
    ASSERT_TRUE(write_recipe_json(a.path, recipe_json(r, "")));
    const std::string ct = a.read("imp_recipe.json");  // JsonParser keeps a view
    JsonParser p(ct);
    p.parse();
    EXPECT_TRUE(p.ok()) << ct;
}

TEST(QuantizeRecipe, FillRecipeTakesTheCommandLine) {
    TempDir d;
    {
        std::ofstream f(d.path + "/calib.bin", std::ios::binary);
        f << "abc";
    }
    Options opt;
    opt.calib_file = d.path + "/calib.bin";
    opt.calib_weight_sq = true;
    opt.format = OutputFormat::CompressedTensors;
    opt.lm_head = LmHeadExport::Source;
    opt.kv_hint = KvHint::None;
    opt.keep_attn_gate = true;
    Recipe r;
    fill_recipe(r, opt, /*kv_cache_fp8=*/false);
    EXPECT_TRUE(r.calibrated);
    // SHA-256("abc"), FIPS 180-2 appendix B.1.
    EXPECT_EQ(r.calib_sha256, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(r.calib_weight, "sq");
    EXPECT_EQ(r.format, "compressed-tensors");
    EXPECT_EQ(r.lm_head, "source");
    EXPECT_EQ(r.kv_hint, "none");
    EXPECT_TRUE(r.keep_attn_gate);
    EXPECT_FALSE(r.imp_version.empty());
    EXPECT_FALSE(r.imp_tree.empty());
    // No calibration file: the calibration block stays empty.
    Options plain;
    Recipe u;
    fill_recipe(u, plain, false);
    EXPECT_FALSE(u.calibrated);
    EXPECT_TRUE(u.calib_sha256.empty());
}

}  // namespace
}  // namespace imp::quantize
