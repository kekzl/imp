#include "imp/imp.h"
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
    static constexpr const char* kMistralDir = "/models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4";
    static constexpr const char* kModeloptCoderDir = "/models/Qwen3-Coder-30B-A3B-FP4";
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

// Gemma-4 MoE coherence gate. Greedy "What is the capital of France?" with
// the model's own chat template applied must produce text containing "Paris".
//
// Why chat-template instead of base completion: Gemma-4-it is heavily
// instruction-tuned and degenerates into a repetition loop on raw
// completion prompts (e.g. "The capital of France is " →
// " the most of the most of the most..."). The Mistral test uses base
// completion because Mistral-Small-3.2 holds together on raw prompts; for
// Gemma-4-it we need the chat-template wrap to land on a sane decoding
// trajectory. The model dir ships a chat_template.jinja that imp picks up
// automatically when apply_chat_template=1.
//
// Coherence path landed in two pieces, both on main:
//   - PR #65 routes the 90 Gemma-4 extras (layer_scalar, router.scale,
//     router.per_expert_scale) through translate_name → weight_map →
//     existing GGUF Gemma-4 forward path (no new kernels).
//   - The same PR's MoE per-expert NVFP4 prefill bypass in
//     executor_forward_moe.cu (M>1 → gemm_nvfp4 dequant→cuBLAS instead of
//     per-row gemv_nvfp4_kpar) eliminated the corruption that produced
//     "way world ات set" pre-fix.
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

// Mistral3 dense coherence gate. Mistral-Small-3.2 is multimodal
// (Mistral3ForConditionalGeneration), but with vision_tower / multi_modal_projector
// tensors skipped at load time the language model alone runs as a standard
// dense LLM. Phase 2 added two pieces to make this work:
//   1. translate_name() now strips the Mistral3-style `language_model.` prefix
//      (`language_model.model.layers.0.q.weight_packed` →
//       `model.layers.0.q.weight_packed`), and skips raw `vision_tower.*` /
//      `multi_modal_projector.*` (no `model.` wrapper) at the top level.
//   2. parse_recipe_yaml() recognizes the elaborate
//      `config_groups: group_0: weights: {num_bits: 4, type: float}` schema as
//      NVFP4, and handles the multi-line bracket-array `ignore: [...]` form.
TEST_F(LlmCompressorE2E, MistralSmall_LoadsAndGeneratesCoherent) {
    if (!dir_exists(kMistralDir)) {
        GTEST_SKIP() << "Model not present at " << kMistralDir;
    }

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(kMistralDir, IMP_FORMAT_SAFETENSORS, &model), IMP_SUCCESS);
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

// Modelopt regression: the existing NVFP4 path (NVIDIA Model Optimizer
// SafeTensors with hf_quant_config.json) must keep working bit-identically
// after the Phase 1 dispatch reshuffle in load_nvfp4_config(). Loads
// Qwen3-Coder-30B-A3B-FP4 and verifies generation completes coherently.
TEST_F(LlmCompressorE2E, Modelopt_QwenCoder30B_StillWorks) {
    if (!dir_exists(kModeloptCoderDir)) {
        GTEST_SKIP() << "Model not present at " << kModeloptCoderDir;
    }

    ImpModel model = nullptr;
    ASSERT_EQ(imp_model_load(kModeloptCoderDir, IMP_FORMAT_SAFETENSORS, &model), IMP_SUCCESS)
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
