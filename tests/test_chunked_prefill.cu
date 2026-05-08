// E2E logits-equality battery for chunked prefill.
//
// Verifies that chunked prefill (prefill_chunk_size > 0) produces
// token-for-token identical greedy output compared to single-chunk prefill
// (prefill_chunk_size = 0) for FP16 KV, and at least 6/8 matching tokens
// for FP8 KV (small noise from FP8 dequant is expected).
//
// Tests skip cleanly when model files are absent — no crash, GTest reports SKIP.
//
// Run subset: build/test-e2e --gtest_filter="ChunkedPrefillTest.*"

#include <gtest/gtest.h>
#include "imp/imp.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace imp_test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool model_exists(const char* path) {
    FILE* f = fopen(path, "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

// Run greedy generation with a given prefill_chunk_size.
// Returns the generated text (greedy, temp=0, max_tokens tokens).
// Returns empty string on any API failure.
//
// Uses imp_generate (the high-level API) which uses the well-tested
// tokenize→prefill→decode path via imp_generate_streaming internally.
// This avoids having to manage the low-level imp_prefill_with_params
// + imp_decode_step loop manually in tests.
static std::string generate_greedy(const char* model_path, const std::string& prompt,
                                   int chunk_size, int max_tokens,
                                   bool use_fp8_kv = false) {
    ImpModel model = nullptr;
    if (imp_model_load(model_path, IMP_FORMAT_GGUF, &model) != IMP_SUCCESS || !model)
        return {};

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 4096;      // explicit: accommodates ~2000-token prompts + generation
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = 0;
    cfg.prefill_chunk_size = chunk_size;
    if (use_fp8_kv)
        cfg.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;

    ImpContext ctx = nullptr;
    if (imp_context_create(model, &cfg, &ctx) != IMP_SUCCESS || !ctx) {
        imp_model_free(model);
        return {};
    }

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = 0.0f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.max_tokens = max_tokens;
    params.apply_chat_template = 0;
    params.seed = 42;

    char output[65536];
    size_t output_len = 0;
    ImpError err = imp_generate(ctx, prompt.c_str(), &params, output, sizeof(output), &output_len);

    imp_context_free(ctx);
    imp_model_free(model);

    if (err != IMP_SUCCESS)
        return {};

    return std::string(output, output_len);
}

// ---------------------------------------------------------------------------
// Test class
// ---------------------------------------------------------------------------

class ChunkedPrefillTest : public ::testing::Test {
protected:
    // Model paths: env var override → /models absolute path (Docker bind-mount).
    // IMP_TEST_MODEL=path overrides Qwen3-4B path.
    // IMP_TEST_MODEL_LLAMA=path overrides Llama-3B path.
    static const char* qwen3_4b_path() {
        const char* p = std::getenv("IMP_TEST_MODEL");
        if (p) return p;
        return "/models/Qwen3-4B-Instruct-2507-Q8_0.gguf";
    }
    static const char* llama_3b_path() {
        const char* p = std::getenv("IMP_TEST_MODEL_LLAMA");
        if (p) return p;
        return "/models/Llama-3.2-3B-Instruct-Q8_0.gguf";
    }

    // ~1600-token prompt: 100 numbered items with 6 words each.
    // Forces >=4 chunks at chunk_size=512 and >=26 chunks at chunk_size=64.
    // Well within the 4096-token context window (leaves ~2400 tokens for generation).
    static std::string long_prompt() {
        std::string p = "Summarize the following list:\n";
        for (int i = 0; i < 100; i++) {
            p += "Item " + std::to_string(i) + ": ";
            for (int w = 0; w < 6; w++)
                p += "word" + std::to_string(w) + " ";
            p += "\n";
        }
        return p;
    }
};

// ---------------------------------------------------------------------------
// Test 1: FP16 KV — token-for-token equality across chunk sizes
//
// Greedy (temp=0) generation with FP16 KV must produce identical text for
// chunk=0 (single-chunk) vs chunk=64/128/512/1024 (chunked prefill).
// ---------------------------------------------------------------------------

TEST_F(ChunkedPrefillTest, Qwen3_4B_Q8_0_FP16_KV_LogitsEqual) {
    const char* path = qwen3_4b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = long_prompt();
    const int n = 8;  // decode tokens to compare

    std::string single      = generate_greedy(path, prompt, /*chunk=*/0,    n);
    std::string chunked512  = generate_greedy(path, prompt, /*chunk=*/512,  n);
    std::string chunked128  = generate_greedy(path, prompt, /*chunk=*/128,  n);
    std::string chunked64   = generate_greedy(path, prompt, /*chunk=*/64,   n);
    std::string chunked1024 = generate_greedy(path, prompt, /*chunk=*/1024, n);

    ASSERT_FALSE(single.empty())      << "single-chunk run produced no output";
    ASSERT_FALSE(chunked512.empty())  << "chunk=512 run produced no output";
    ASSERT_FALSE(chunked128.empty())  << "chunk=128 run produced no output";
    ASSERT_FALSE(chunked64.empty())   << "chunk=64 run produced no output";
    ASSERT_FALSE(chunked1024.empty()) << "chunk=1024 run produced no output";

    // Greedy generation with FP16 KV must be byte-identical across chunk sizes.
    EXPECT_EQ(single, chunked512)  << "chunk=512 diverges from single-chunk";
    EXPECT_EQ(single, chunked128)  << "chunk=128 diverges from single-chunk";
    EXPECT_EQ(single, chunked64)   << "chunk=64 diverges from single-chunk";
    EXPECT_EQ(single, chunked1024) << "chunk=1024 diverges from single-chunk";
}

// ---------------------------------------------------------------------------
// Test 2: FP8 KV — allow some character-level difference (FP8 dequant noise)
//
// FP8 KV introduces small rounding noise. The text must be non-empty and
// at least 75% similar (edit-distance proxy: common-prefix length >= 3/4 total).
// ---------------------------------------------------------------------------

TEST_F(ChunkedPrefillTest, Qwen3_4B_Q8_0_FP8_KV_LogitsEqual) {
    const char* path = qwen3_4b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = long_prompt();
    const int n = 8;

    std::string single  = generate_greedy(path, prompt, 0,   n, /*fp8=*/true);
    std::string chunked = generate_greedy(path, prompt, 512, n, /*fp8=*/true);

    ASSERT_FALSE(single.empty())  << "single-chunk FP8 run produced no output";
    ASSERT_FALSE(chunked.empty()) << "chunk=512 FP8 run produced no output";

    // FP8 KV noise: count matching characters from the start.
    // Require at least 6/8 of the shorter string to match character-for-character.
    int match = 0;
    int compare_len = static_cast<int>(std::min(single.size(), chunked.size()));
    for (int i = 0; i < compare_len; i++)
        if (single[i] == chunked[i])
            match++;
        else
            break;  // first divergence point

    int required = (compare_len * 6) / 8;
    EXPECT_GE(match, required) << "FP8 KV: only " << match << " of " << compare_len
                               << " chars matched from prefix (single='" << single
                               << "' vs chunked='" << chunked << "')";
}

// ---------------------------------------------------------------------------
// Test 3: Llama-3.2-3B, non-block-aligned chunk (64, block_size=16)
// ---------------------------------------------------------------------------

TEST_F(ChunkedPrefillTest, Llama_3_2_3B_Chunk_64_LogitsEqual) {
    const char* path = llama_3b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = long_prompt();
    const int n = 8;

    std::string single  = generate_greedy(path, prompt, 0,  n);
    std::string chunked = generate_greedy(path, prompt, 64, n);  // non-block-aligned (block=16)

    ASSERT_FALSE(single.empty())  << "single-chunk run produced no output";
    ASSERT_FALSE(chunked.empty()) << "chunk=64 run produced no output";

    EXPECT_EQ(single, chunked) << "chunk=64 (non-block-aligned) diverges from single-chunk"
                               << "\n  single:  '" << single << "'"
                               << "\n  chunked: '" << chunked << "'";
}

// ---------------------------------------------------------------------------
// Test 4: chunk > prompt length — must degrade to single-chunk silently
// ---------------------------------------------------------------------------

TEST_F(ChunkedPrefillTest, Qwen3_4B_ChunkLargerThanPrompt) {
    const char* path = qwen3_4b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string short_p = "What is 2+2?";
    const int n = 4;

    std::string single  = generate_greedy(path, short_p, 0,    n);
    std::string chunked = generate_greedy(path, short_p, 4096, n);  // chunk >> prompt

    // Both runs should produce output
    EXPECT_FALSE(single.empty())  << "single-chunk run produced no tokens";
    EXPECT_FALSE(chunked.empty()) << "chunk=4096 run produced no tokens";

    // When chunk >= prompt length the chunked path must be equivalent to single-chunk.
    EXPECT_EQ(single, chunked) << "chunk=4096 (larger than prompt) diverges from single-chunk";
}

// ---------------------------------------------------------------------------
// Test 5: coherence check — chunked prefill must not collapse to repetition
// ---------------------------------------------------------------------------

TEST_F(ChunkedPrefillTest, Qwen3_4B_GenerationCoherent) {
    const char* path = qwen3_4b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    std::string out = generate_greedy(path, long_prompt(), 512, 32);
    ASSERT_FALSE(out.empty()) << "chunk=512 run produced no output";

    // Count distinct words in the output.
    // A degeneration loop (e.g. "word word word ...") produces only 1-2 unique words.
    std::set<std::string> unique_words;
    std::string word;
    for (char c : out) {
        if (c == ' ' || c == '\n' || c == '\t') {
            if (!word.empty()) {
                unique_words.insert(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty())
        unique_words.insert(word);

    EXPECT_GE(static_cast<int>(unique_words.size()), 4)
        << "generation collapsed to repetition: only " << unique_words.size()
        << " unique words in output: '" << out << "'";
}

}  // namespace imp_test
