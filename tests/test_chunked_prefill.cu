// E2E logits-equality: chunked prefill (prefill_chunk_size>0) must match single-chunk via
// teacher-forced PPL (imp_perplexity hits every position, so positional/KV corruption shifts
// mean NLL massively).
// Greedy TEXT byte-equality is deliberately not asserted: chunk boundaries legitimately route
// through different attention kernels whose few-ULP logit diffs flip greedy argmax on
// near-tied prompts, making the suite a function of the exact quant file (#543) - a
// re-downloaded quant of the SAME model broke it while teacher-forced PPL stayed
// bit-identical. Skips cleanly when model files are absent.

#include <gtest/gtest.h>
#include "imp/imp.h"
#include "test_models.h"

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

// Wraps imp_generate (uses the well-tested tokenize->prefill->decode path via
// imp_generate_streaming) instead of manually driving imp_prefill_with_params+imp_decode_step.
static std::string generate_greedy(const char* model_path, const std::string& prompt,
                                   int chunk_size, int max_tokens,
                                   bool use_fp8_kv = false) {
    ImpModel model = nullptr;
    if (imp_model_load(model_path, IMP_FORMAT_GGUF, &model) != IMP_SUCCESS || !model)
        return {};

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 4096;      // explicit: accommodates ~3200-token prompts + generation
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

// Teacher-forced perplexity of `prompt` at a given prefill_chunk_size.
// Returns -1.0 on any API failure.
static double ppl_for_chunk(const char* model_path, const std::string& prompt,
                            int chunk_size, bool use_fp8_kv = false,
                            bool enable_graphs = false) {
    ImpModel model = nullptr;
    if (imp_model_load(model_path, IMP_FORMAT_GGUF, &model) != IMP_SUCCESS || !model)
        return -1.0;

    std::vector<int32_t> tokens(4096);
    int n_tokens = 0;
    if (imp_tokenize(model, prompt.c_str(), tokens.data(), &n_tokens,
                     static_cast<int>(tokens.size())) != IMP_SUCCESS ||
        n_tokens < 2) {
        imp_model_free(model);
        return -1.0;
    }

    ImpConfig cfg = imp_config_default();
    cfg.max_seq_len = 4096;
    cfg.max_batch_size = 1;
    cfg.enable_cuda_graphs = enable_graphs ? 1 : 0;
    cfg.prefill_chunk_size = chunk_size;
    if (use_fp8_kv)
        cfg.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;

    ImpContext ctx = nullptr;
    if (imp_context_create(model, &cfg, &ctx) != IMP_SUCCESS || !ctx) {
        imp_model_free(model);
        return -1.0;
    }

    double ppl = -1.0;
    ImpError err = imp_perplexity(ctx, tokens.data(), n_tokens, &ppl);

    imp_context_free(ctx);
    imp_model_free(model);

    return (err == IMP_SUCCESS) ? ppl : -1.0;
}

// Count whitespace-separated unique words (degeneration-loop detector).
static int count_unique_words(const std::string& text) {
    std::set<std::string> unique_words;
    std::string word;
    for (char c : text) {
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
    return static_cast<int>(unique_words.size());
}

// Chunk-invariance PPL tolerance vs chunk=0: cross-kernel-path noise measured at |dPPL| <=
// 0.15% relative (Qwen3-8B + Llama-3.2-3B, ~3.3-3.4k tokens); a positional/KV-corruption bug
// shifts PPL by >= 50%. 1% keeps 6x noise margin with >50x margin on real bugs.
static constexpr double kFp16RelTol = 0.01;
// FP8-KV adds dequant noise on the chunk-continuation KV reads.
static constexpr double kFp8RelTol = 0.03;

// ---------------------------------------------------------------------------
// Test class
// ---------------------------------------------------------------------------

class ChunkedPrefillTest : public ::testing::Test {
protected:
    // IMP_TEST_MODEL_QWEN4B / IMP_TEST_MODEL_LLAMA override the model paths, deliberately not the
    // generic IMP_TEST_MODEL (suite runs repurpose that for other gates): these chunk-equality
    // tolerances are calibrated for Qwen3-4B, where other models could cross the attention-kernel
    // threshold into a different kernel path with legitimately flippable greedy ties.
    static const char* qwen3_4b_path() {
        const char* p = std::getenv(imp_test::kEnvModelQwen4b);
        if (p) return p;
        return "/models/Qwen3-4B-Instruct-2507-Q8_0.gguf";
    }
    static const char* llama_3b_path() {
        const char* p = std::getenv(imp_test::kEnvModelLlama);
        if (p) return p;
        return "/models/Llama-3.2-3B-Instruct-Q8_0.gguf";
    }

    // 120 items (~27 tokens/item, ~3240 tokens) sits safely between two failure cliffs: below
    // ~154-155 items (~4096 tokens) the engine spills into a second prefill chunk needing 332 KV
    // blocks, overflowing the 256-block d_pf_block_tables_ buffer (cudaMemcpy invalid argument);
    // 120 items needs only ~200 KV blocks and stays above the 2049-token threshold needed for
    // >=2 chunks at chunk=1024.
    static std::string long_prompt() {
        std::string p = "Summarize the following list:\n";
        for (int i = 0; i < 120; i++) {
            p += "Item " + std::to_string(i) + ": ";
            for (int w = 0; w < 10; w++)
                p += "word" + std::to_string(w) + " ";
            p += "\n";
        }
        return p;
    }

    // ~1.4k-token probe, deliberately below the attn_scores s_cap clamp (~1984 @ max_seq_len
    // 4096) so chunk=0 is a genuine single-shot reference. Qwen3-4B is bit-identical across
    // chunk={64,128,512,1024}; Llama-3.2-3B within 0.01% (continuation chunks route cuBLAS, first
    // chunk FA2-f16qk). Above ~2.5k context late chunks route fp8/e4m3 FMHA and drift up to ~25%
    // NLL - open issue, no dedicated regression test here yet.
    static std::string probe_prompt() {
        std::string p = "Summarize the following list:\n";
        for (int i = 0; i < 54; i++) {
            p += "Item " + std::to_string(i) + ": ";
            for (int w = 0; w < 10; w++)
                p += "word" + std::to_string(w) + " ";
            p += "\n";
        }
        return p;
    }
};

// FP16 KV: greedy (temp=0) generation must produce identical text for chunk=0 vs
// chunk=64/128/512/1024.

TEST_F(ChunkedPrefillTest, Qwen3_4B_Q8_0_FP16_KV_LogitsEqual) {
    const char* path = qwen3_4b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = probe_prompt();

    const double base = ppl_for_chunk(path, prompt, /*chunk=*/0);
    ASSERT_GT(base, 0.0) << "single-chunk perplexity failed";

    for (int chunk : {64, 128, 512, 1024}) {
        const double ppl = ppl_for_chunk(path, prompt, chunk);
        ASSERT_GT(ppl, 0.0) << "chunk=" << chunk << " perplexity failed";
        EXPECT_NEAR(ppl, base, base * kFp16RelTol)
            << "chunk=" << chunk
            << " teacher-forced PPL diverges from single-chunk prefill";
    }
}

// FP8 KV: chunk continuation reads FP8-quantized KV of prior chunks; allow extra dequant
// noise on top of the FP16 tolerance.

TEST_F(ChunkedPrefillTest, Qwen3_4B_Q8_0_FP8_KV_LogitsEqual) {
    const char* path = qwen3_4b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = probe_prompt();

    const double base = ppl_for_chunk(path, prompt, /*chunk=*/0, /*fp8=*/true);
    ASSERT_GT(base, 0.0) << "single-chunk FP8 perplexity failed";

    const double chunked = ppl_for_chunk(path, prompt, /*chunk=*/512, /*fp8=*/true);
    ASSERT_GT(chunked, 0.0) << "chunk=512 FP8 perplexity failed";

    EXPECT_NEAR(chunked, base, base * kFp8RelTol)
        << "chunk=512 FP8-KV teacher-forced PPL diverges from single-chunk prefill";
}

// ---------------------------------------------------------------------------
// Test 3: Llama-3.2-3B, non-block-aligned chunk (64, block_size=16)
// ---------------------------------------------------------------------------

TEST_F(ChunkedPrefillTest, Llama_3_2_3B_Chunk_64_LogitsEqual) {
    const char* path = llama_3b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = probe_prompt();

    const double base = ppl_for_chunk(path, prompt, /*chunk=*/0);
    ASSERT_GT(base, 0.0) << "single-chunk perplexity failed";

    const double chunked = ppl_for_chunk(path, prompt, /*chunk=*/64);
    ASSERT_GT(chunked, 0.0) << "chunk=64 perplexity failed";

    EXPECT_NEAR(chunked, base, base * kFp16RelTol)
        << "chunk=64 (small odd chunk count) teacher-forced PPL diverges "
           "from single-chunk prefill";
}

// #548 gate: chunked continuations above fmha_prefill_threshold used to drift teacher-forced
// NLL up to ~25% (fp8/e4m3 FMHA) or CATASTROPHICALLY on Llama below it (0.29->7.13, f16-QK
// fast path). Root cause was a host race: pinned prefill staging
// (h_pf_token_ids_/h_pf_positions_) was rewritten while earlier chunks' H2D copies were still
// queued, fixed via pf_staging_evt_. Continuations now route FP16-QK FA2 at every ctx_len.

TEST_F(ChunkedPrefillTest, LongContext_Chunk_Invariance) {
    const char* path = llama_3b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = long_prompt();  // ~3.1k tokens, crosses the kernel threshold

    const double base = ppl_for_chunk(path, prompt, /*chunk=*/0);
    ASSERT_GT(base, 0.0) << "reference perplexity failed";

    for (int chunk : {64, 512}) {
        const double ppl = ppl_for_chunk(path, prompt, chunk);
        ASSERT_GT(ppl, 0.0) << "chunk=" << chunk << " perplexity failed";
        EXPECT_NEAR(ppl, base, base * kFp16RelTol)
            << "chunk=" << chunk << " long-context teacher-forced PPL diverges";
    }
}

// Same invariance with CUDA graphs ON (#981): the captured chunk forward bakes
// ctx_len/q_offset as host args, so replaying an earlier chunk's graph for a continuation
// chunk attends with stale geometry and silently truncates long context. Invariance must
// hold whether or not the model's capturability gates actually let this chunk capture.
TEST_F(ChunkedPrefillTest, LongContext_Chunk_Invariance_GraphsOn) {
    const char* path = llama_3b_path();
    if (!model_exists(path))
        GTEST_SKIP() << "model not present: " << path;

    const std::string prompt = long_prompt();  // ~3.1k tokens, >2 non-last chunks at chunk=512

    const double base = ppl_for_chunk(path, prompt, /*chunk=*/0);
    ASSERT_GT(base, 0.0) << "reference perplexity failed";

    for (int chunk : {512, 1024}) {
        const double ppl =
            ppl_for_chunk(path, prompt, chunk, /*use_fp8_kv=*/false, /*enable_graphs=*/true);
        ASSERT_GT(ppl, 0.0) << "chunk=" << chunk << " perplexity failed";
        EXPECT_NEAR(ppl, base, base * kFp16RelTol)
            << "chunk=" << chunk << " graphs-on long-context PPL diverges (#981 class)";
    }
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

    // Check baseline coherence first: greedy continuation quality on the synthetic list prompt
    // is a property of the model FILE, not chunking - some quants collapse to repetition even at
    // chunk=0. Only judge the chunked run against a baseline that is itself coherent.
    std::string single = generate_greedy(path, long_prompt(), 0, 32);
    ASSERT_FALSE(single.empty()) << "single-chunk run produced no output";
    if (count_unique_words(single) < 4)
        GTEST_SKIP() << "model file degenerates on the probe prompt even "
                        "unchunked — coherence probe not applicable: '"
                     << single << "'";

    std::string out = generate_greedy(path, long_prompt(), 512, 32);
    ASSERT_FALSE(out.empty()) << "chunk=512 run produced no output";

    // A degeneration loop (e.g. "word word word ...") produces 1-2 unique words.
    EXPECT_GE(count_unique_words(out), 4)
        << "generation collapsed to repetition under chunked prefill while "
           "single-chunk was coherent: '" << out << "'";
}

}  // namespace imp_test
