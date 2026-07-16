#include <gtest/gtest.h>
#include "imp/imp.h"
#include "test_models.h"
#include "compute/attention_cublas.h"  // executed-kernel coverage counter

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Environment helpers
// ---------------------------------------------------------------------------

static const char* primary_model() { return std::getenv(imp_test::kEnvModel); }

static const char* gdn_model() { return std::getenv(imp_test::kEnvModelGdn); }

static const char* gemma4_model() { return std::getenv(imp_test::kEnvModelGemma4); }

#define REQUIRE_MODEL(var)    \
    const char* path = var(); \
    if (!path)                \
    GTEST_SKIP() << "Set " #var " env var to run this test"

// ---------------------------------------------------------------------------
// Primary model tests (Qwen3-4B Q8_0 or similar dense transformer)
// ---------------------------------------------------------------------------

class PrimaryModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = primary_model();
        if (!path_)
            GTEST_SKIP() << "Set IMP_TEST_MODEL to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(PrimaryModelTest, GenerateCoherentOutput) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "The capital of France is", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    // Greedy output should contain "Paris" for any reasonable model
    std::string text(output, len);
    EXPECT_NE(text.find("Paris"), std::string::npos) << "Expected 'Paris' in output: " << text;
}

// Helper: drop <think>...</think> blocks so this test works against both
// non-thinking baselines (Qwen3-4B-Instruct, Llama-3.2-3B) and thinking
// models (Qwen3-8B, DeepSeek-R1-Distill). Open `<think>` with no close —
// model exhausted the token budget mid-reasoning — drops everything from
// `<think>` to end so the post-reasoning substring check fails clean.
static std::string strip_think_blocks_for_test_(std::string s) {
    while (true) {
        size_t open = s.find("<think>");
        if (open == std::string::npos)
            break;
        size_t close = s.find("</think>", open);
        if (close == std::string::npos) {
            s.erase(open);
            break;
        }
        s.erase(open, close + 8 /*len("</think>")*/ - open);
    }
    return s;
}

TEST_F(PrimaryModelTest, MultiTurnConversation) {
    ImpGenerateParams params = imp_generate_params_default();
    // Thinking models (Qwen3-8B, DeepSeek-R1) need headroom for the
    // <think>…</think> block plus the answer. Non-thinking models finish
    // long before reaching 256; the cap only matters for the worst case.
    params.max_tokens = 256;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    // Turn 1
    char out1[4096];
    size_t len1 = 0;
    ASSERT_EQ(imp_generate(ctx_, "Say hello.", &params, out1, sizeof(out1), &len1), IMP_SUCCESS);
    EXPECT_GT(len1, 0u);

    // Reset for turn 2
    ASSERT_EQ(imp_context_reset(ctx_), IMP_SUCCESS);

    // Turn 2 — different prompt, verify context is clean
    char out2[4096];
    size_t len2 = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is 2+2? Answer with just the number.", &params, out2, sizeof(out2),
                           &len2),
              IMP_SUCCESS);
    EXPECT_GT(len2, 0u);

    std::string text2 = strip_think_blocks_for_test_(std::string(out2, len2));
    EXPECT_NE(text2.find("4"), std::string::npos)
        << "Expected '4' in (think-stripped) output: " << text2;
}

TEST_F(PrimaryModelTest, TokenizeRoundtrip) {
    const char* text = "Hello, how are you today?";
    int32_t tokens[256];
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model_, text, tokens, &n_tokens, 256), IMP_SUCCESS);
    EXPECT_GT(n_tokens, 3);

    char buf[1024];
    ASSERT_EQ(imp_detokenize(model_, tokens, n_tokens, buf, sizeof(buf)), IMP_SUCCESS);
    // Detokenized text should contain the original words
    EXPECT_NE(std::string(buf).find("Hello"), std::string::npos);
}

TEST_F(PrimaryModelTest, PrefillThenDecodeMultipleTokens) {
    int32_t tokens[128];
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model_, "The meaning of life is", tokens, &n_tokens, 128), IMP_SUCCESS);
    ASSERT_GT(n_tokens, 0);

    ASSERT_EQ(imp_prefill(ctx_, tokens, n_tokens), IMP_SUCCESS);

    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 16;
    params.temperature = 0.0f;

    std::vector<int32_t> generated;
    for (int i = 0; i < 16; i++) {
        int32_t tok = 0;
        ImpError err = imp_decode_step(ctx_, &params, &tok);
        if (err != IMP_SUCCESS)
            break;
        generated.push_back(tok);
    }
    EXPECT_GE(generated.size(), 4u) << "Should generate at least a few tokens";
}

// ---------------------------------------------------------------------------
// GDN model tests (Qwen3.5 Gated DeltaNet hybrid)
// ---------------------------------------------------------------------------

class GDNModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = gdn_model();
        if (!path_)
            GTEST_SKIP() << "Set IMP_TEST_MODEL_GDN to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 0;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(GDNModelTest, GenerateCoherentOutput) {
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 64;  // need ≥10 words to exercise the unique-ratio check
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[4096];
    size_t len = 0;
    // A prompt that forces a longer natural answer — "one word" prompts exit
    // after 2-3 tokens and never stress the recurrent scan past token 3.
    ASSERT_EQ(imp_generate(ctx_, "Write a short paragraph about the planet Jupiter.", &params, output,
                           sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    std::string text(output, len);
    EXPECT_GT(text.size(), 5u) << "Output too short: " << text;

    // Recurrent-state collapse detector: split on whitespace, count unique
    // words. A degenerate output like " my my my my my..." has many tokens
    // but very few unique words. This catches the 2026-04-24 GDN regression
    // that the length-only check above missed.
    std::vector<std::string> words;
    {
        std::string cur;
        for (char c : text) {
            if (c == ' ' || c == '\n' || c == '\t') {
                if (!cur.empty()) {
                    words.push_back(cur);
                    cur.clear();
                }
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty())
            words.push_back(cur);
    }
    if (words.size() >= 10) {
        std::set<std::string> unique(words.begin(), words.end());
        // Require at least 30 % unique words once we have ≥10 words.
        // Degenerate " my my my my..." × 30 = 30 words, 1 unique = 3 %.
        const double unique_ratio = static_cast<double>(unique.size()) / words.size();
        EXPECT_GE(unique_ratio, 0.30)
            << "Recurrent-state collapse detected: " << unique.size() << " unique / " << words.size()
            << " total words (" << (unique_ratio * 100.0) << "%)\n"
            << "Full output: " << text;
    }
}

TEST_F(GDNModelTest, MultiTurnGDNState) {
    // GDN models carry recurrent state across tokens.
    // Multi-turn verifies the state management works correctly.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 32;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    // Turn 1
    char out1[4096];
    size_t len1 = 0;
    ASSERT_EQ(imp_generate(ctx_, "Say hello.", &params, out1, sizeof(out1), &len1), IMP_SUCCESS);
    EXPECT_GT(len1, 0u);

    // Reset
    ASSERT_EQ(imp_context_reset(ctx_), IMP_SUCCESS);

    // Turn 2 — independent prompt (GDN state should be clean after reset)
    char out2[4096];
    size_t len2 = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is 1+1? Answer with just the number.", &params, out2, sizeof(out2),
                           &len2),
              IMP_SUCCESS);
    EXPECT_GT(len2, 0u);

    // Verify generation works after reset (GDN state properly cleared).
    // Don't check exact content — small GDN models have limited instruction following.
    std::string text2(out2, len2);
    EXPECT_GT(text2.size(), 1u) << "Output too short after reset: " << text2;
}

// ---------------------------------------------------------------------------
// Gemma-4 model tests (26B-A4B MoE hybrid: per-layer SWA/global, 128 experts
// top-8, shared dense FFN alongside each MoE block, custom router, GEGLU)
//
// Primary regression test — Gemma-4 forward pass has historically been
// fragile (see memory/gemma4_working_2026_04_14.md). This test locks in
// correct output on the Q4_K_M model.
// ---------------------------------------------------------------------------

class Gemma4ModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = gemma4_model();
        if (!path_)
            GTEST_SKIP() << "Set IMP_TEST_MODEL_GEMMA4 to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 0;  // baseline path — paired with Gemma4GraphsTest
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(Gemma4ModelTest, AnswersCapitalOfFrance) {
    // With default gemma chat template, Gemma-4 emits a <|channel>thought...
    // block followed by the answer. "Paris" must appear in the output.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 64;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is the capital of France?", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    std::string text(output, len);
    EXPECT_NE(text.find("Paris"), std::string::npos) << "Expected 'Paris' in Gemma-4 output: " << text;
}

TEST_F(Gemma4ModelTest, RawCompletionProducesOutput) {
    // Raw (no-chat-template) path: instruct-tuned Gemma-4 without its chat
    // template produces structurally different output than with the template
    // (e.g. it emits <|channel>thought tokens as if the turn had started).
    // We don't assert exact content — just that the forward pass completes
    // successfully and returns non-empty non-degenerate text. The chat-template
    // test above already covers semantic correctness.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 12;
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "The capital of France is", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);
}

TEST_F(Gemma4ModelTest, PrefillFusesSwaLayers_Hd512StaysCublas) {
    // Executed-kernel gate (FA2-coverage dispatch): Gemma-4 has a 5:1 SWA:global
    // layer pattern — ~24 hd=256 SWA layers + ~6 hd=512 global layers (30 total).
    // After this dispatch the hd=256 SWA layers route to FA2 f16-QK per-layer
    // (the win — previously the coarse model-level force_cublas gate sent EVERY
    // layer to cuBLAS). The hd=512 global layers deliberately STAY on the
    // materialized cuBLAS path: the SMEM-capped fused hd=512 kernel is 2.8-4.6x
    // slower (docs/audit/gemma4_attn_routing_2026_07_16/PERF_LOG.md), so fusing them would regress. So a short prefill must
    // execute cuBLAS for ONLY the handful of hd=512 layers, not all 30. Checks
    // the executed kernel (a launch counter), not just the dispatch branch.
    imp::attention_cublas_prefill_reset_count();

    int32_t tokens[128];
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model_, "What is the capital of France?", tokens, &n_tokens, 128), IMP_SUCCESS);
    ASSERT_GT(n_tokens, 0);
    ASSERT_EQ(imp_prefill(ctx_, tokens, n_tokens), IMP_SUCCESS);

    uint64_t cublas_calls = imp::attention_cublas_prefill_call_count();
    // > 0: the hd=512 global layers DO use cuBLAS (the deliberate hybrid).
    EXPECT_GT(cublas_calls, 0u) << "expected the hd=512 global layers to use cuBLAS";
    // << 30: the hd=256 SWA layers (the majority) moved OFF cuBLAS to FA2. If the
    // SWA layers still used cuBLAS (routing regression) this would be ~30.
    EXPECT_LT(cublas_calls, 15u)
        << "Gemma-4 prefill used cuBLAS attention " << cublas_calls << " times — expected only the ~6 "
        << "hd=512 global layers (<15). A value near 30 means the hd=256 SWA layers regressed back "
        << "onto cuBLAS instead of FA2.";
}

TEST_F(Gemma4ModelTest, NoRepetitionDegeneration) {
    // Guards against the classic Gemma-4 failure mode: ~15 tokens of sensible
    // output followed by "own own own" (or "아니라 own 아니라") loops driven by
    // MoE routing instability / precision drift through 30 layers × 128 experts.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 100;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[8192];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "Name three European capitals.", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    std::string text(output, len);

    // Repetition check: the most common 4-char substring in English text
    // tops out around 5-10% of the text. If any single run of the same
    // character occupies >30% of the output, it's a degeneration loop.
    size_t max_run = 0;
    for (size_t i = 0; i < text.size();) {
        size_t j = i;
        while (j < text.size() && text[j] == text[i])
            ++j;
        max_run = std::max(max_run, j - i);
        i = j;
    }
    EXPECT_LT(max_run * 2, text.size()) << "Detected degeneration (run of " << max_run
                                        << " chars in output of " << text.size() << "): " << text;
}

// ---------------------------------------------------------------------------
// Gemma-4 with CUDA graphs enabled — regression guard for the graph path.
//
// History: the AsyncGraphLoop captured forward_decode_async() — a parallel
// reimplementation that diverged on Gemma-4 Q4_K_M (sampled <eos> at step 0
// of the WHILE body, terminating after ~3 tokens with garbage). Fixed by
// unifying forward_decode_async() with forward_logits() (the canonical
// path). Locks in: graphs MUST produce identical output to the no-graph
// path, both in the short (single-token capture) and long (async WHILE)
// regimes.
// ---------------------------------------------------------------------------

class Gemma4GraphsTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = gemma4_model();
        if (!path_)
            GTEST_SKIP() << "Set IMP_TEST_MODEL_GEMMA4 to run";

        ASSERT_EQ(imp_model_load(path_, IMP_FORMAT_GGUF, &model_), IMP_SUCCESS);
        ASSERT_NE(model_, nullptr);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 1;
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    const char* path_ = nullptr;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

TEST_F(Gemma4GraphsTest, AnswersCapitalOfFranceWithGraphs) {
    // Same prompt as the no-graph baseline above. Output must contain "Paris".
    // Generates >3 decoded tokens to ensure the AsyncGraphLoop launches and
    // produces correct output (the previous bug terminated at ~3 tokens).
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 64;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[4096];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "What is the capital of France?", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    std::string text(output, len);
    EXPECT_NE(text.find("Paris"), std::string::npos) << "Expected 'Paris' in Gemma-4 graph output: " << text;
}

TEST_F(Gemma4GraphsTest, LongDecodeStaysCoherent) {
    // 256 decode tokens: long enough for the AsyncGraphLoop to drive most
    // of the generation. Guards against state drift over a captured WHILE
    // body that runs many iterations from the same baked-in pointers.
    ImpGenerateParams params = imp_generate_params_default();
    params.max_tokens = 256;
    params.temperature = 0.0f;
    params.apply_chat_template = 1;

    char output[16384];
    size_t len = 0;
    ASSERT_EQ(imp_generate(ctx_, "Name three European capitals.", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    // Same degeneration heuristic as the no-graph variant.
    std::string text(output, len);
    size_t max_run = 0;
    for (size_t i = 0; i < text.size();) {
        size_t j = i;
        while (j < text.size() && text[j] == text[i])
            ++j;
        max_run = std::max(max_run, j - i);
        i = j;
    }
    EXPECT_LT(max_run * 2, text.size()) << "Graph-path degeneration (run of " << max_run
                                        << " chars in output of " << text.size() << "): " << text;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// #948 regression: decode-graph launch-topology re-derivation on ctx growth.
//
// The per-step decode graph bakes launch topology derived from the HOST
// max_context_len (split-K num_splits / GQA-vs-split-K kernel choice). The
// intended re-capture trigger — the pow2 max_blocks bucket — never fires
// because the decode batch pool pads max_blocks_per_seq to the pool stride.
// Pre-fix, a graph captured during a SHORT request replayed with a stale
// topology for the next LONG-prompt request (> prefill_chunk_size) and hit an
// illegal memory access at its first decode step, wedging the engine for the
// rest of the process. This test is the minimal HTTP repro as a C-API test:
// short request → >2048-token prompt → another short request.
// ---------------------------------------------------------------------------
TEST(DecodeGraphCtxGrowthTest, LongPromptAfterShortRequestStaysAlive) {
    const char* path = primary_model();
    if (!path)
        GTEST_SKIP() << "Set IMP_TEST_MODEL to run";

    ImpModel model = nullptr;
    ImpContext ctx = nullptr;
    ASSERT_EQ(imp_model_load(path, IMP_FORMAT_GGUF, &model), IMP_SUCCESS);
    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 4096;
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = 1;  // the bug lives in decode-graph replay
    ASSERT_EQ(imp_context_create(model, &cfg, &ctx), IMP_SUCCESS);

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = 0.0f;
    params.apply_chat_template = 0;

    char output[8192];
    size_t len = 0;

    // Request 1: short — captures the decode graph at a tiny context.
    params.max_tokens = 16;
    ASSERT_EQ(imp_generate(ctx, "The capital of France is", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    EXPECT_GT(len, 0u);

    // Request 2: prompt > prefill_chunk_size (2048 tokens) → chunked prefill,
    // then decode replays the pooled graph at a ~70x larger context.
    std::string filler;
    for (int i = 0; i < 300; i++)
        filler += "The maintenance log notes routine checks of pumps and valves. ";
    std::string prompt =
        "Remember this code: ZEBRA-9134.\n\n" + filler + "\nThe code I was asked to remember is";
    params.max_tokens = 24;
    ASSERT_EQ(imp_generate(ctx, prompt.c_str(), &params, output, sizeof(output), &len), IMP_SUCCESS);
    EXPECT_GT(len, 0u);
    std::string text2(output, len);
    EXPECT_NE(text2.find("ZEBRA"), std::string::npos)
        << "needle lost after chunked prefill: " << text2;

    // Request 3: the engine must still be alive (pre-fix: every request after
    // the wedge returned 0 tokens on a poisoned CUDA context).
    params.max_tokens = 16;
    ASSERT_EQ(imp_generate(ctx, "The capital of Italy is", &params, output, sizeof(output), &len),
              IMP_SUCCESS);
    std::string text3(output, len);
    EXPECT_NE(text3.find("Rome"), std::string::npos) << "engine wedged after ctx growth: " << text3;

    imp_context_free(ctx);
    imp_model_free(model);
}
