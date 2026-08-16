// imp-quantize's output-format rules.
//
// Every rule covered here fails SILENTLY: the checkpoint is written, loads, and
// generates. A reciprocal written the wrong way round scales every weight by
// amax²/36; a fused group that does not share a tensor scale leaves two of
// three matrices dequantized against the third's; a config.json that ends up
// with two `quantization_config` keys reads as unquantized because parsers keep
// the last one.

#include "../tools/imp-quantize/checkpoint_out.h"

#include "model/hf_config_loader.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>

using namespace imp::quantize;

// ---- format selection ----------------------------------------------------

TEST(QuantizeCheckpointOut, ParsesFormatNames) {
    OutputFormat f = OutputFormat::CompressedTensors;
    ASSERT_TRUE(parse_output_format("modelopt", f));
    EXPECT_EQ(f, OutputFormat::Modelopt);

    // All three spellings reach the same layout; `vllm` exists because that is
    // what a caller is actually after.
    for (const char* name : {"vllm", "compressed-tensors", "compressed_tensors"}) {
        f = OutputFormat::Modelopt;
        ASSERT_TRUE(parse_output_format(name, f)) << name;
        EXPECT_EQ(f, OutputFormat::CompressedTensors) << name;
    }
    EXPECT_FALSE(parse_output_format("nvfp4", f));
    EXPECT_FALSE(parse_output_format("", f));
}

// ---- tensor names --------------------------------------------------------

TEST(QuantizeCheckpointOut, TensorNamesPerFormat) {
    const auto mo = quant_tensor_names("model.layers.0.self_attn.q_proj", OutputFormat::Modelopt);
    EXPECT_EQ(mo.packed, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(mo.micro_scale, "model.layers.0.self_attn.q_proj.weight_scale");
    EXPECT_EQ(mo.global_scale, "model.layers.0.self_attn.q_proj.weight_scale_2");

    const auto ct = quant_tensor_names("model.layers.0.self_attn.q_proj", OutputFormat::CompressedTensors);
    EXPECT_EQ(ct.packed, "model.layers.0.self_attn.q_proj.weight_packed");
    EXPECT_EQ(ct.micro_scale, "model.layers.0.self_attn.q_proj.weight_scale");
    EXPECT_EQ(ct.global_scale, "model.layers.0.self_attn.q_proj.weight_global_scale");
}

// The direction of this is the whole point: Modelopt multiplies by the stored
// number, compressed-tensors divides by it (vLLM: "CT stores as divisors").
TEST(QuantizeCheckpointOut, GlobalScaleIsReciprocalOnlyForCompressedTensors) {
    EXPECT_FLOAT_EQ(global_scale_value(0.25f, OutputFormat::Modelopt), 0.25f);
    EXPECT_FLOAT_EQ(global_scale_value(0.25f, OutputFormat::CompressedTensors), 4.0f);

    // Round trip: what the reader computes must come back to the scale used.
    const float scale = 1.0f / 3.0f;
    EXPECT_NEAR(1.0f / global_scale_value(scale, OutputFormat::CompressedTensors), scale, 1e-9f);
}

TEST(QuantizeCheckpointOut, ZeroScaleDoesNotBecomeInfinity) {
    // An all-zero weight yields no scale. Writing 1/0 would put an infinity in
    // the checkpoint, which contaminates the layer's whole hidden state rather
    // than nulling it.
    EXPECT_FLOAT_EQ(global_scale_value(0.0f, OutputFormat::CompressedTensors), 0.0f);
    EXPECT_FLOAT_EQ(global_scale_value(0.0f, OutputFormat::Modelopt), 0.0f);
}

// ---- fused groups --------------------------------------------------------

TEST(QuantizeCheckpointOut, FusedGroupsMatchEngineMerges) {
    const std::string p = "model.layers.3.self_attn.";
    const std::string q = fusion_group_key(p + "q_proj.weight");
    EXPECT_FALSE(q.empty());
    EXPECT_EQ(q, fusion_group_key(p + "k_proj.weight"));
    EXPECT_EQ(q, fusion_group_key(p + "v_proj.weight"));
    // o_proj is its own linear in every engine — grouping it would force an
    // unrelated scale onto it.
    EXPECT_EQ(fusion_group_key(p + "o_proj.weight"), "");

    const std::string m = "model.layers.3.mlp.";
    const std::string g = fusion_group_key(m + "gate_proj.weight");
    EXPECT_FALSE(g.empty());
    EXPECT_EQ(g, fusion_group_key(m + "up_proj.weight"));
    EXPECT_EQ(fusion_group_key(m + "down_proj.weight"), "");
    EXPECT_NE(g, q);

    // GDN / linear attention, from the same packed_modules_mapping.
    const std::string l = "model.layers.4.linear_attn.";
    EXPECT_EQ(fusion_group_key(l + "in_proj_qkv.weight"), fusion_group_key(l + "in_proj_z.weight"));
    EXPECT_EQ(fusion_group_key(l + "in_proj_b.weight"), fusion_group_key(l + "in_proj_a.weight"));
    // The two GDN groups are separate merges and must not share a scale.
    EXPECT_NE(fusion_group_key(l + "in_proj_qkv.weight"), fusion_group_key(l + "in_proj_b.weight"));
}

TEST(QuantizeCheckpointOut, FusedGroupsAreKeyedPerLayerAndPerExpert) {
    EXPECT_NE(fusion_group_key("model.layers.0.self_attn.q_proj.weight"),
              fusion_group_key("model.layers.1.self_attn.q_proj.weight"));
    // Per-expert MoE: expert 0's gate/up pair is a different merge from
    // expert 1's, so one loud expert must not set the other's scale.
    EXPECT_NE(fusion_group_key("model.layers.0.mlp.experts.0.gate_proj.weight"),
              fusion_group_key("model.layers.0.mlp.experts.1.gate_proj.weight"));
    EXPECT_EQ(fusion_group_key("model.layers.0.mlp.experts.7.gate_proj.weight"),
              fusion_group_key("model.layers.0.mlp.experts.7.up_proj.weight"));
}

TEST(QuantizeCheckpointOut, NonWeightsAndBareNamesHaveNoGroup) {
    EXPECT_EQ(fusion_group_key("model.layers.0.self_attn.q_proj.weight_scale"), "");
    EXPECT_EQ(fusion_group_key("model.layers.0.input_layernorm.weight"), "");
    EXPECT_EQ(fusion_group_key("q_proj.weight"), "");  // no parent prefix to key on
    EXPECT_EQ(fusion_group_key(""), "");
}

// ---- scales --------------------------------------------------------------

TEST(QuantizeCheckpointOut, ExportScalePutsLoudestBlockAtOne) {
    // The micro-scale the kernel computes is local_absmax / (tensor_scale * 6),
    // so this convention puts the block holding the tensor peak at exactly 1.0
    // and every other block below it.
    //
    // Pinned because the obvious "improvement" — scaling by 448 so the
    // micro-scales fill FP8's range — measures 31.05 against 29.47 on
    // Qwen3-0.6B. See checkpoint_out.h.
    const float absmax = 3.5f;
    const float scale = export_tensor_scale(absmax);
    EXPECT_NEAR(absmax / (scale * 6.0f), 1.0f, 1e-5f);
}

TEST(QuantizeCheckpointOut, ExportScaleOfAnAllZeroTensorStaysUsable) {
    // Not 0: the kernel divides every value by this.
    EXPECT_GT(export_tensor_scale(0.0f), 0.0f);
    EXPECT_TRUE(std::isfinite(export_tensor_scale(0.0f)));
}

TEST(QuantizeCheckpointOut, Fp16AbsmaxIgnoresSignAndNonFinites) {
    // 1.0, -2.0, 0.5
    const uint16_t vals[] = {0x3C00, 0xC000, 0x3800};
    EXPECT_FLOAT_EQ(fp16_absmax(vals, 3), 2.0f);

    // A BF16 source can hold values FP16 cannot; the widened tensor then carries
    // an infinity. Letting it set the scale would zero the entire tensor.
    const uint16_t with_inf[] = {0x3C00, 0x7C00, 0xFC00, 0x7E00};  // 1.0, +inf, -inf, NaN
    EXPECT_FLOAT_EQ(fp16_absmax(with_inf, 4), 1.0f);

    const uint16_t zeros[] = {0x0000, 0x8000};
    EXPECT_FLOAT_EQ(fp16_absmax(zeros, 2), 0.0f);
    EXPECT_FLOAT_EQ(fp16_absmax(nullptr, 0), 0.0f);
}

TEST(QuantizeCheckpointOut, Fp16AbsmaxDecodesSubnormals) {
    const uint16_t sub[] = {0x0001};  // smallest positive subnormal, 2^-24
    EXPECT_FLOAT_EQ(fp16_absmax(sub, 1), std::ldexp(1.0f, -24));
}

// ---- the quantization_config ---------------------------------------------

TEST(QuantizeCheckpointOut, CompressedTensorsConfigCarriesWhatVllmTestsFor) {
    const std::string s = compressed_tensors_quant_config({"lm_head", "model.visual.merger.linear_fc1"},
                                                          /*calibrated=*/false);
    // vLLM's _is_nvfp4_format() reads exactly these five fields; any one of them
    // different and it silently selects another scheme.
    EXPECT_NE(s.find("\"strategy\": \"tensor_group\""), std::string::npos);
    EXPECT_NE(s.find("\"group_size\": 16"), std::string::npos);
    EXPECT_NE(s.find("\"num_bits\": 4"), std::string::npos);
    EXPECT_NE(s.find("\"type\": \"float\""), std::string::npos);
    EXPECT_NE(s.find("\"symmetric\": true"), std::string::npos);
    EXPECT_NE(s.find("\"format\": \"nvfp4-pack-quantized\""), std::string::npos);
    EXPECT_NE(s.find("\"quant_method\": \"compressed-tensors\""), std::string::npos);
    // Weight-only. Declaring activation quantization we never measured would
    // make vLLM quantize activations at runtime against absent scales.
    EXPECT_NE(s.find("\"input_activations\": null"), std::string::npos);
    EXPECT_NE(s.find("\"lm_head\""), std::string::npos);
    EXPECT_NE(s.find("\"model.visual.merger.linear_fc1\""), std::string::npos);
}

TEST(QuantizeCheckpointOut, EmptyIgnoreListStaysValidJson) {
    const std::string s = compressed_tensors_quant_config({}, false);
    EXPECT_NE(s.find("\"ignore\": []"), std::string::npos);
}

// ---- config.json patching ------------------------------------------------

TEST(QuantizeCheckpointOut, ReplacesAnExistingQuantizationConfig) {
    // Qwen3.8-27B ships `"quantization_config": null`. Prepending a second key
    // of the same name is legal JSON that most readers resolve to the LAST one,
    // so the checkpoint would announce itself as unquantized.
    const std::string src = R"({
  "model_type": "qwen3_5",
  "quantization_config": null,
  "vocab_size": 151936
})";
    std::string out, err;
    ASSERT_TRUE(patch_config_json(src, "{\"quant_method\": \"compressed-tensors\"}", out, err)) << err;
    EXPECT_EQ(out.find("quantization_config"), out.rfind("quantization_config"));
    EXPECT_NE(out.find("\"quant_method\": \"compressed-tensors\""), std::string::npos);
    EXPECT_EQ(out.find("null"), std::string::npos);
    // Every other byte is untouched.
    EXPECT_NE(out.find("\"model_type\": \"qwen3_5\""), std::string::npos);
    EXPECT_NE(out.find("\"vocab_size\": 151936"), std::string::npos);
}

TEST(QuantizeCheckpointOut, ReplacesAnObjectValuedQuantizationConfig) {
    const std::string src = R"({"a": 1, "quantization_config": {"nested": {"x": [1,2]}}, "b": 2})";
    std::string out, err;
    ASSERT_TRUE(patch_config_json(src, "{\"new\": true}", out, err)) << err;
    EXPECT_EQ(out, R"({"a": 1, "quantization_config": {"new": true}, "b": 2})");
}

TEST(QuantizeCheckpointOut, InsertsWhenAbsent) {
    const std::string src = "{\n  \"model_type\": \"qwen3\"\n}\n";
    std::string out, err;
    ASSERT_TRUE(patch_config_json(src, "{\"q\": 1}", out, err)) << err;
    EXPECT_NE(out.find("\"quantization_config\": {\"q\": 1},"), std::string::npos);
    EXPECT_NE(out.find("\"model_type\": \"qwen3\""), std::string::npos);
}

TEST(QuantizeCheckpointOut, InsertsIntoAnEmptyObjectWithoutATrailingComma) {
    std::string out, err;
    ASSERT_TRUE(patch_config_json("{}", "{\"q\": 1}", out, err)) << err;
    EXPECT_EQ(out.find(','), std::string::npos);
    ASSERT_TRUE(patch_config_json("{  \n }", "{\"q\": 1}", out, err)) << err;
    EXPECT_EQ(out.find(','), std::string::npos);
}

TEST(QuantizeCheckpointOut, DoesNotMatchTheKeyInsideANestedObjectOrAString) {
    // A nested `quantization_config` belongs to something else, and the same
    // word in a string value is not a key at all. Either one replaced instead
    // of the top-level key would leave the real declaration absent.
    const std::string src =
        R"({"text_config": {"quantization_config": "inner"}, "note": "quantization_config", "z": 0})";
    std::string out, err;
    ASSERT_TRUE(patch_config_json(src, "{\"q\": 1}", out, err)) << err;
    EXPECT_NE(out.find("\"text_config\": {\"quantization_config\": \"inner\"}"), std::string::npos);
    EXPECT_NE(out.find("\"note\": \"quantization_config\""), std::string::npos);
    EXPECT_NE(out.find("\"quantization_config\": {\"q\": 1},"), std::string::npos);
}

// ---- writer and reader, against each other -------------------------------
//
// The rest of this file checks the writer against the format on paper. This one
// checks it against the loader that has to read it back, which is the pair that
// was actually broken: imp detected compressed-tensors from recipe.yaml alone,
// so a checkpoint this tool wrote — correct, but without a recipe — was read as
// Modelopt and dequantized by the reciprocal of its own scales. Nothing failed;
// perplexity went from 31.05 to 1.2e47.
TEST(QuantizeCheckpointOut, WhatTheWriterDeclaresIsWhatTheLoaderDetects) {
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("ckpt_out_" + std::to_string(::getpid()))).string();
    std::filesystem::create_directories(dir);

    // A config.json shaped like a real source, patched exactly as the writer
    // patches it — not a hand-written stand-in, so the two cannot drift.
    const std::string src = R"({
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "quantization_config": null,
  "vocab_size": 151936
})";
    std::string patched, err;
    ASSERT_TRUE(patch_config_json(src,
                                  compressed_tensors_quant_config({"lm_head", "model.embed_tokens"}, false),
                                  patched, err))
        << err;
    std::ofstream(dir + "/config.json") << patched;

    imp::HFConfigLoader::NvFP4Config cfg;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(dir, cfg))
        << "loader did not recognise the checkpoint the writer just declared";
    // LLM_COMPRESSOR is what selects the divide-by-scale convention downstream.
    // MODELOPT here means every weight comes back scaled by amax^2/36.
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);
    EXPECT_EQ(cfg.group_size, 16);
    EXPECT_EQ(cfg.exclude_modules.size(), 2u);

    // The recipe.yaml written beside it must say the same thing, because a
    // reader that predates config.json detection consults only that file — and
    // reading this checkpoint as Modelopt inverts every scale in silence.
    std::string yerr;
    ASSERT_TRUE(write_recipe_yaml(dir, {"lm_head", "model.embed_tokens"}, yerr)) << yerr;
    std::filesystem::remove(dir + "/config.json");
    imp::HFConfigLoader::NvFP4Config from_recipe;
    ASSERT_TRUE(imp::HFConfigLoader::load_nvfp4_config(dir, from_recipe));
    EXPECT_EQ(from_recipe.format, cfg.format);
    EXPECT_EQ(from_recipe.group_size, cfg.group_size);
    EXPECT_EQ(from_recipe.exclude_modules, cfg.exclude_modules);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(QuantizeCheckpointOut, SurvivesEscapesAndRejectsNonObjects) {
    // A key whose string contains an escaped quote must not end the scan early.
    const std::string src = R"({"chat_template": "say \"hi\"", "quantization_config": 1})";
    std::string out, err;
    ASSERT_TRUE(patch_config_json(src, "{\"q\": 1}", out, err)) << err;
    EXPECT_NE(out.find("\"quantization_config\": {\"q\": 1}"), std::string::npos);

    EXPECT_FALSE(patch_config_json("[1,2]", "{}", out, err));
    EXPECT_FALSE(patch_config_json("", "{}", out, err));
    EXPECT_FALSE(patch_config_json("{\"a\"", "{}", out, err));
}

// The refusal has to be reachable before the conversion, not at the copy step
// where the file is finally needed — on a 27B source that is 25 minutes of work
// thrown away, and the modelopt path must not be caught by it at all.
TEST(QuantizeCheckpointOut, RefusesCompressedTensorsWithoutAConfigJson) {
    const std::string dir =
        (std::filesystem::temp_directory_path() / ("ckpt_noconf_" + std::to_string(::getpid()))).string();
    std::filesystem::create_directories(dir);

    std::string err;
    EXPECT_FALSE(can_declare_quantization(dir, OutputFormat::CompressedTensors, err));
    EXPECT_NE(err.find("config.json"), std::string::npos);
    // Modelopt declares itself in a file it writes itself, so it is unaffected.
    EXPECT_TRUE(can_declare_quantization(dir, OutputFormat::Modelopt, err));

    std::ofstream(dir + "/config.json") << "{}";
    EXPECT_TRUE(can_declare_quantization(dir, OutputFormat::CompressedTensors, err));

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}
