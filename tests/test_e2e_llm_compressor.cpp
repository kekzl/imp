#include "imp/imp.h"
#include "test_models.h"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

bool dir_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

}  // namespace

class LlmCompressorE2E : public ::testing::Test {
protected:
    static constexpr const char* kGemma4Dir = "/models/Gemma-4-26B-A4B-it-NVFP4";
    // Overridable, because a hardcoded default that does not exist on the host
    // reads as "this export is untested" when it means "the path is wrong".
    static std::string mistral_dir() {
        return imp_test::env_path_or(imp_test::kEnvModelMistral,
                                     "/models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4");
    }
    static std::string modelopt_coder_dir() {
        return imp_test::env_path_or(imp_test::kEnvModelModeloptCoder, "/models/Qwen3-Coder-30B-A3B-FP4");
    }
};

TEST_F(LlmCompressorE2E, Gemma4_LoadsWithoutIMA) {
    if (!dir_exists(kGemma4Dir)) {
        GTEST_SKIP() << "Model not present at " << kGemma4Dir;
    }

    ImpModel model = nullptr;
    ImpError rc = imp_model_load(kGemma4Dir, IMP_FORMAT_SAFETENSORS, &model);
    ASSERT_EQ(rc, IMP_SUCCESS) << "imp_model_load failed: " << imp_error_string(rc);
    ASSERT_NE(model, nullptr);

    imp_model_free(model);
}

// Gemma-4 MoE coherence gate: chat-template "capital of France?" must answer "Paris".
// Chat-template (not base completion) because Gemma-4-it degenerates into a repetition loop
// on raw completion prompts; Mistral-Small-3.2 holds together on raw prompts so its test
// uses that.
// Landed via PR #65 (90 Gemma-4 extras through translate_name->weight_map->existing GGUF
// forward path) plus the same PR's MoE per-expert NVFP4 prefill bypass in
// executor_forward_moe.cu (M>1 -> gemm_nvfp4 dequant->cuBLAS), which eliminated garbage output.
TEST_F(LlmCompressorE2E, Gemma4_LoadsAndGeneratesCoherent) {
    if (!dir_exists(kGemma4Dir)) {
        GTEST_SKIP() << "Model not present at " << kGemma4Dir;
    }

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(kGemma4Dir, IMP_FORMAT_SAFETENSORS, &model), IMP_SUCCESS);
    ASSERT_NE(model, nullptr);

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 512;
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = 0;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[2048];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx, "What is the capital of France?", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    std::string result(output, len);
    EXPECT_NE(result.find("Paris"), std::string::npos) << "Generated text: " << result;

    imp_context_free(ctx);
    imp_model_free(model);
}

// Mistral3 dense coherence gate: Mistral-Small-3.2 is multimodal, but with
// vision_tower/multi_modal_projector tensors skipped at load the language model alone runs
// as a standard dense LLM. Needed: (1) translate_name() strips the language_model. prefix
// and skips raw vision_tower.*/multi_modal_projector.* at the top level; (2)
// parse_recipe_yaml() recognizes the config_groups: group_0: weights schema as NVFP4,
// including the multi-line bracket-array ignore: [...] form.
TEST_F(LlmCompressorE2E, MistralSmall_LoadsAndGeneratesCoherent) {
    ASSERT_NO_FATAL_FAILURE(imp_test::require_readable_if_set(imp_test::kEnvModelMistral));
    const std::string mistral = mistral_dir();
    if (!dir_exists(mistral.c_str())) {
        GTEST_SKIP() << "Model not present at " << mistral;
    }

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(mistral.c_str(), IMP_FORMAT_SAFETENSORS, &model), IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 512;
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = 0;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;
    char output[2048];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx, "The capital of France is", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    std::string result(output, len);
    EXPECT_NE(result.find("Paris"), std::string::npos) << "Generated text: " << result;
    imp_context_free(ctx);
    imp_model_free(model);
}

// Modelopt NVFP4 (NVIDIA Model Optimizer SafeTensors + hf_quant_config.json) must keep
// working bit-identically after the Phase 1 dispatch reshuffle in load_nvfp4_config(). Loads
// Qwen3-Coder-30B-A3B-FP4 and checks coherent generation.
TEST_F(LlmCompressorE2E, Modelopt_QwenCoder30B_StillWorks) {
    ASSERT_NO_FATAL_FAILURE(imp_test::require_readable_if_set(imp_test::kEnvModelModeloptCoder));
    const std::string coder = modelopt_coder_dir();
    if (!dir_exists(coder.c_str())) {
        GTEST_SKIP() << "Model not present at " << coder;
    }

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(coder.c_str(), IMP_FORMAT_SAFETENSORS, &model), IMP_SUCCESS)
        << "Modelopt path regressed";
    ASSERT_NE(model, nullptr);

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 512;
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = 0;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);
    ASSERT_NE(ctx, nullptr);

    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[2048];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx, "def factorial(n):", &params, output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 5u) << "Output unexpectedly short";

    imp_context_free(ctx);
    imp_model_free(model);
}
