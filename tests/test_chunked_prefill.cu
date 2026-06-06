// E2E logits-equality battery for chunked prefill.
//
// Verifies that chunked prefill (prefill_chunk_size > 0) is logit-equivalent
// to single-chunk prefill (prefill_chunk_size = 0) via teacher-forced
// perplexity over the probe prompt (imp_perplexity applies the LM head to
// every position — any positional/KV-corruption bug in the chunked path
// shifts the mean NLL massively).
//
// Greedy TEXT byte-equality (the original form of these tests) is
// deliberately NOT asserted anymore: chunk=0 vs chunk>0 legitimately route
// through different attention kernels (cuBLAS S-matrix vs FA2, threshold-
// dependent per chunk), whose few-ULP logit differences flip greedy argmax
// on near-tied prompts. That made the suite a function of the exact quant
// file it was calibrated on (#543) — re-downloaded quants of the SAME model
// broke it while teacher-forced PPL was bit-identical to 0.15% relative.
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
                            int chunk_size, bool use_fp8_kv = false) {
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
    cfg.enable_cuda_graphs = 0;
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

// Chunk-invariance tolerances on teacher-forced PPL, relative to chunk=0.
// Measured cross-kernel-path noise (2026-06-06, Qwen3-8B + Llama-3.2-3B,
// 3.3-3.4k-token forcing): |dPPL| <= 0.15% relative. A positional or
// KV-corruption bug in the chunked path shifts PPL by >= 50%. 1% keeps a
// 6x noise margin while catching real bugs with > 50x margin.
static constexpr double kFp16RelTol = 0.01;
// FP8-KV adds dequant noise on the chunk-continuation KV reads.
static constexpr double kFp8RelTol = 0.03;

// ---------------------------------------------------------------------------
// Test class
// ---------------------------------------------------------------------------

class ChunkedPrefillTest : public ::testing::Test {
protected:
    // Model paths: env var override → /models absolute path (Docker bind-mount).
    // IMP_TEST_MODEL_QWEN4B=path overrides the Qwen3-4B path. Deliberately NOT
    // the generic IMP_TEST_MODEL: suite runs set that to whatever model is
    // under test (e.g. Qwen3-8B for the prefix-cache/greedy-lock gates), and
    // these chunk-equality expectations are calibrated for Qwen3-4B — on other
    // models chunk=0 vs chunk>0 cross the attention-kernel threshold into
    // DIFFERENT kernel paths, where greedy logit ties may legitimately flip.
    // IMP_TEST_MODEL_LLAMA=path overrides Llama-3B path.
    static const char* qwen3_4b_path() {
        const char* p = std::getenv("IMP_TEST_MODEL_QWEN4B");
        if (p) return p;
        return "/models/Qwen3-4B-Instruct-2507-Q8_0.gguf";
    }
    static const char* llama_3b_path() {
        const char* p = std::getenv("IMP_TEST_MODEL_LLAMA");
        if (p) return p;
        return "/models/Llama-3.2-3B-Instruct-Q8_0.gguf";
    }

    // Long prompt: 120 numbered items × ~27 tokens/item ≈ 3240 tokens total.
    //
    // Size rationale (bisected 2026-05-08):
    //   - Each item tokenizes to ~27 tokens with Qwen3 BPE (Llama similar).
    //   - max_seq_len=4096 → blocks_per_seq=256 (block_size=16).
    //     At 120 items: ~200 KV blocks needed, safely below the 256-block buffer
    //     allocated for d_pf_block_tables_.
    //   - 200 items → ~5312 tokens > max_seq_len=4096: engine spills into a
    //     second prefill chunk of 1216 tokens that needs 332 KV blocks, which
    //     overflows the 256-block d_pf_block_tables_ buffer (cudaMemcpy
    //     "invalid argument"). The cliff is between 154 and 155 items (~4096 tokens).
    //   - 120 items sits safely above 2049 tokens (≥4 chunks at chunk=512,
    //     ≥2 chunks at chunk=1024) and well below the 4096-token cliff.
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

    // Probe prompt for the NLL-equivalence tests: ~1.4k tokens, deliberately
    // BELOW the attn_scores s_cap clamp (~1984 @ max_seq_len 4096) so that
    // chunk=0 is a genuinely single-shot reference forward. Measured
    // 2026-06-06 (post fa2_fp16qk continuation-decline): Qwen3-4B is
    // BIT-IDENTICAL across chunk={64,128,512,1024}; Llama-3.2-3B is within
    // 0.01% (continuation chunks route through cuBLAS, first chunk through
    // FA2-f16qk). Above ~2.5k context the late chunks route through the
    // fp8/e4m3 FMHA family and drift up to ~25% NLL — that open issue is
    // tracked by the DISABLED_ long-context test below.
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

// ---------------------------------------------------------------------------
// Test 2: FP8 KV — chunk continuation reads FP8-quantized KV of previous
// chunks; allow the extra dequant noise on top of the FP16 tolerance.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Test 3b (DISABLED — issue #548): LONG-context chunk invariance.
//
// Once a chunk's ctx_len crosses fmha_prefill_threshold (~2.5-2.9k), the
// chunked path routes through the fp8/e4m3 FMHA family
// (attention_prefill_dispatch) whose accumulated e4m3 score noise drifts the
// teacher-forced NLL by up to ~25% vs the reference (measured 2026-06-06,
// 120-item long_prompt ~3.1k tokens, Llama-3.2-3B, post fa2_fp16qk
// continuation-decline):
//   chunk=0(→s_cap-clamped ~1984+rest) nll 0.80
//   chunk=64  nll 0.61   chunk=512 nll 0.97
// (Before the fa2_fp16qk continuation-decline this read nll 8.08 —
// catastrophic — at chunk=64; that part is fixed.) Same quality class as
// issue #511 (fp8-FMHA above threshold unvalidated); tracked in #548.
// Enable when the long-context chunk path is numerically reconciled.
// ---------------------------------------------------------------------------

TEST_F(ChunkedPrefillTest, DISABLED_LongContext_Chunk_Invariance) {
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

    // Baseline first: greedy continuation quality on the synthetic list
    // prompt is a property of the model FILE, not of chunking — some quants
    // of the same model collapse to repetition even with chunk=0 (observed
    // 2026-06-06 with a re-downloaded Qwen3-4B-Instruct-2507 Q8_0). Only
    // judge the chunked run against a baseline that is itself coherent.
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
