// E2E greedy regression locks — TEST_AUDIT.md risk #3 (Phase 2.4).
//
// The single highest-leverage test class in the audit: a fixed prompt run
// greedy (temp=0, top_k=1) through the FULL stack (tokenize → prefill →
// decode loop, CUDA graphs ON = the production path) must reproduce a
// frozen token sequence exactly. Locks would have caught: Nemotron NoPE
// (positionally blind since integration), Phi-4 RoPE-NeoX (#503), the FA2
// short-prefill regression (#512), and the gemma mode-2 NaN collapse
// (#514/#516) — all of which shipped while per-kernel parity suites were
// green.
//
// Lock lifecycle (tests/refs/e2e_greedy_locks.h):
//   1. Generate candidates: IMP_LOCK_PRINT=1 prints ready-to-paste lock
//      entries for the loaded model.
//   2. Verify EXTERNALLY before committing: GGUF locks are checked against
//      llama.cpp greedy output on the same raw prompt (semantic agreement —
//      bit-identical tokens across engines is not expected; coherent,
//      prompt-grounded continuation is). SafeTensors/NVFP4 models have no
//      external engine that loads them; their locks are verified by human
//      review + degen_suite and marked "internal" in the table.
//   3. Commit the entry. From then on ANY token drift fails loudly.
//
// A lock failure means: forward pass / tokenizer / sampler behavior changed.
// That is sometimes intentional (kernel rework with quality-neutral PPL) —
// regenerate via step 1+2 and say so in the PR; it is never noise: this test
// also asserts determinism (two fresh-context runs must match), so a flaky
// lock is itself a finding (atomics in the forward pass of a dense model).

#include <gtest/gtest.h>
#include "imp/imp.h"
#include "refs/e2e_greedy_locks.h"
#include "test_models.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char* model_path() { return std::getenv(imp_test::kEnvModel); }

std::string basename_of(const std::string& p) {
    size_t s = p.find_last_of('/');
    std::string b = (s == std::string::npos) ? p : p.substr(s + 1);
    if (!b.empty() && b.back() == '/')
        b.pop_back();
    return b;
}

bool is_safetensors_dir(const std::string& p) {
    // crude but sufficient: GGUF models are files ending in .gguf
    return p.size() < 5 || p.substr(p.size() - 5) != ".gguf";
}

class GreedyLockTest : public ::testing::Test {
protected:
    void SetUp() override {
        const char* path = model_path();
        if (!path)
            GTEST_SKIP() << "Set IMP_TEST_MODEL to run greedy locks";
        ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path, imp_test::kEnvModel));
        path_ = path;

        ImpModelFormat fmt = is_safetensors_dir(path_) ? IMP_FORMAT_SAFETENSORS : IMP_FORMAT_GGUF;
        ASSERT_EQ(imp_model_load(path, fmt, &model_), IMP_SUCCESS);

        ImpConfig cfg = imp_config_default();
        cfg.max_seq_len = 512;
        cfg.max_batch_size = 1;
        cfg.enable_cuda_graphs = 1;  // locks freeze the PRODUCTION path
        ASSERT_EQ(imp_context_create(model_, &cfg, &ctx_), IMP_SUCCESS);
    }

    void TearDown() override {
        if (ctx_)
            imp_context_free(ctx_);
        if (model_)
            imp_model_free(model_);
    }

    // Greedy generation via the token-level API (prefill + decode loop).
    std::vector<int32_t> generate_greedy(const char* prompt, int n_gen) {
        ImpGenerateParams params = imp_generate_params_default();
        params.temperature = 0.0f;
        params.top_k = 1;
        params.top_p = 1.0f;
        params.seed = 42;  // greedy ignores it; pinned for determinism anyway
        params.max_tokens = n_gen;

        std::vector<int32_t> prompt_tokens(512);
        int n_prompt = 0;
        EXPECT_EQ(imp_tokenize(model_, prompt, prompt_tokens.data(), &n_prompt, 512), IMP_SUCCESS);
        prompt_tokens.resize(n_prompt);

        EXPECT_EQ(imp_context_reset(ctx_), IMP_SUCCESS);
        EXPECT_EQ(imp_prefill_with_params(ctx_, prompt_tokens.data(), n_prompt, &params), IMP_SUCCESS);

        std::vector<int32_t> out;
        for (int i = 0; i < n_gen; i++) {
            int32_t tok = -1;
            if (imp_decode_step(ctx_, &params, &tok) != IMP_SUCCESS || tok < 0)
                break;
            out.push_back(tok);
        }
        return out;
    }

    std::string detok(const std::vector<int32_t>& toks) {
        std::string buf(4096, '\0');
        if (imp_detokenize(model_, toks.data(), (int)toks.size(), buf.data(), buf.size()) != IMP_SUCCESS)
            return "<detokenize failed>";
        buf.resize(strlen(buf.c_str()));
        return buf;
    }

    std::string path_;
    ImpModel model_ = nullptr;
    ImpContext ctx_ = nullptr;
};

// Candidate prompts for NEW locks (IMP_LOCK_PRINT=1). Raw completion
// prompts, no chat template: template churn must not invalidate locks.
// One factual cloze, one arithmetic (the prompt-blindness detector — Nemotron
// NoPE and the FA2 regression scrambled exactly this), one multi-sentence.
constexpr const char* kCandidatePrompts[] = {
    "The capital of France is",
    "Q: What is 17 + 25?\nA:",
    "The three primary colors are red, blue, and",
};
constexpr int kLockLen = 32;

TEST_F(GreedyLockTest, FrozenSequences) {
    const std::string base = basename_of(path_);
    const bool print_mode = std::getenv("IMP_LOCK_PRINT") != nullptr;

    if (print_mode) {
        for (const char* prompt : kCandidatePrompts) {
            std::vector<int32_t> run1 = generate_greedy(prompt, kLockLen);
            std::vector<int32_t> run2 = generate_greedy(prompt, kLockLen);
            ASSERT_EQ(run1, run2) << "NON-DETERMINISTIC at lock-generation time — do not lock";
            std::string esc;  // escape \n for the C literal
            for (char c : std::string(prompt))
                esc += (c == '\n') ? std::string("\\n") : std::string(1, c);
            printf("    // verified: <FILL: llama.cpp <image> <date> / internal <date>>\n");
            printf("    {\"%s\",\n     \"%s\",\n     %d,\n     {", base.c_str(), esc.c_str(),
                   (int)run1.size());
            for (size_t i = 0; i < run1.size(); i++)
                printf("%s%d", i ? ", " : "", run1[i]);
            printf("}},\n    // text: %s\n", detok(run1).c_str());
        }
        GTEST_SKIP() << "IMP_LOCK_PRINT mode: candidates printed, nothing asserted";
    }

    int n_locks = 0;
    for (const auto& lock : imp_refs::kGreedyLocks) {
        if (base != lock.model_basename)
            continue;
        n_locks++;
        SCOPED_TRACE(std::string(lock.model_basename) + " :: " + lock.prompt);

        // ALWAYS request kLockLen (same as lock generation): the engine
        // yields requested-1 tokens (prefill samples token 1, the graph loop
        // plans max_tokens-1 decode steps), so requesting lock.n_tokens
        // would generate one token short of the lock.
        std::vector<int32_t> run1 = generate_greedy(lock.prompt, kLockLen);

        // Determinism probe: a dense model MUST reproduce itself exactly on a
        // fresh context. A flaky lock is a forward-pass-atomics finding, not
        // test noise.
        std::vector<int32_t> run2 = generate_greedy(lock.prompt, kLockLen);
        ASSERT_EQ(run1, run2) << "NON-DETERMINISTIC greedy output (fresh contexts, temp=0):\n"
                              << "  run1: " << detok(run1) << "\n  run2: " << detok(run2);

        std::vector<int32_t> want(lock.tokens, lock.tokens + lock.n_tokens);
        // The model may stop early (EOS) — the lock stores exactly what the
        // engine produced at lock time, so lengths must match too.
        EXPECT_EQ(run1, want) << "GREEDY LOCK BROKEN for " << base << "\n  prompt: " << lock.prompt
                              << "\n  locked: " << detok(want) << "\n  now:    " << detok(run1)
                              << "\nIf this change is intentional (quality-neutral kernel rework), "
                                 "regenerate with IMP_LOCK_PRINT=1, re-verify externally "
                                 "(tests/refs/e2e_greedy_locks.h header comment), and say so in the PR.";
    }

    if (n_locks == 0)
        GTEST_SKIP() << "no greedy locks recorded for model '" << base
                     << "' — generate with IMP_LOCK_PRINT=1 and add to tests/refs/e2e_greedy_locks.h";
}

}  // namespace
