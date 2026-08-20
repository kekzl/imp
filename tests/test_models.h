// Central model-env registry for the test suite (TEST_AUDIT (retired) R6 / #581).
//
// Model-dependent tests are gated on a small set of IMP_TEST_MODEL* env vars.
// Before this header that gating was copy-pasted across ~25 files: each file
// re-spelled the env-var name, the std::getenv() call, and (for the GGUF
// suites) the /models/... container-fallback path. A typo in any copy silently
// disabled a test. This header is the single source of truth for:
//
//   * the env-var names (kEnv* constants), and
//   * how a path is resolved from them (env_path / env_path_or).
//
// The actual GTEST_SKIP() stays at the call site on purpose: GTEST_SKIP()
// expands to a `return` from the *enclosing* function, so it cannot be hidden
// inside a helper without skipping the helper instead of the test. Call sites
// therefore read:
//
//   const std::string path = imp_test::env_path(imp_test::kEnvModel);
//   if (path.empty())
//       GTEST_SKIP() << "Set " << imp_test::kEnvModel << " to run ...";
//
// or, for the GGUF suites that fall back to the Docker bind-mount and then
// skip-on-missing-file:
//
//   const std::string path = imp_test::env_path_or(imp_test::kEnvModel,
//                                                   "/models/Qwen3-8B-Q8_0.gguf");
//
// The hardcoded /models/... fallbacks match the Makefile container mount, which
// is `-v $(HOME)/models:/models`. It used to be `$(PWD)/models`, and the repo's
// models/ entries are absolute symlinks into $HOME/models, so under that mount
// every path missed, every model test skipped, and the battery reported green
// having loaded nothing. Skipping is only correct for an UNSET var - see
// require_readable() below. The fallbacks are passed by the caller so the
// model<->test mapping stays visible at the call site; this header only owns
// the env-var names and the getenv mechanics.

#ifndef IMP_TESTS_TEST_MODELS_H
#define IMP_TESTS_TEST_MODELS_H

#include <gtest/gtest.h>

#include <sys/stat.h>

#include <cstdlib>
#include <string>

namespace imp_test {

// --- Model env-var names (single source of truth) -------------------------
//
// Generic primary model. Suite runs point this at whatever model is under test
// (Qwen3-8B Q8_0 for the greedy-lock / prefix-cache gates, a SafeTensors dir
// for NVFP4 locks, ...). GGUF vs SafeTensors is sniffed from the path.
inline constexpr const char* kEnvModel = "IMP_TEST_MODEL";

// Generic GGUF model for loader/tensor-kind coverage.
inline constexpr const char* kEnvGguf = "IMP_TEST_GGUF";

// Tokenizer-compat golden (paired with kEnvModel).
inline constexpr const char* kEnvGolden = "IMP_TEST_GOLDEN";

// Architecture-specific overrides for the multi-model E2E suite.
inline constexpr const char* kEnvModelGdn = "IMP_TEST_MODEL_GDN";
inline constexpr const char* kEnvModelGemma4 = "IMP_TEST_MODEL_GEMMA4";

// Chunked-prefill calibration models. Deliberately distinct from kEnvModel:
// the chunk-equality expectations are calibrated for these specific models.
inline constexpr const char* kEnvModelQwen4b = "IMP_TEST_MODEL_QWEN4B";
inline constexpr const char* kEnvModelLlama = "IMP_TEST_MODEL_LLAMA";

// MoE/hybrid model for the deterministic-mode E2E gate.
inline constexpr const char* kEnvMoeModel = "IMP_TEST_MOE_MODEL";

// Vision mmproj GGUFs for the SigLIP / gemma4v golden tests.
inline constexpr const char* kEnvMmproj = "IMP_TEST_MMPROJ";
inline constexpr const char* kEnvMmprojGemma4 = "IMP_TEST_MMPROJ_GEMMA4";

// LLM-Compressor / ModelOpt export directories for the loader E2E gate. They
// were hardcoded in the test, and one of the two defaults named a directory
// that does not exist on this machine (`Qwen3-Coder-30B-A3B-FP4` against the
// `-Instruct-FP4` that is actually there), so the case skipped for a reason
// nobody could see from the test output.
inline constexpr const char* kEnvModelModeloptCoder = "IMP_TEST_MODEL_MODELOPT_CODER";
inline constexpr const char* kEnvModelMistral = "IMP_TEST_MODEL_MISTRAL";

// DeepSeek-V2/V3 directory for MLA config tests.
// Expected: a HF model directory containing config.json.
// Default: /models/DeepSeek-V2-Lite (Docker bind-mount path).
inline constexpr const char* kEnvModelDeepSeek = "IMP_TEST_MODEL_DEEPSEEK";

// --- Accessors ------------------------------------------------------------

// Value of env var `name`, or "" if unset. Caller decides whether "" means
// skip (most suites) or use a fallback (env_path_or).
inline std::string env_path(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// Value of env var `name`, or `fallback` if unset. `fallback` is the
// documented /models/... Docker-mount path; the caller still checks the file
// exists and skips if not.
inline std::string env_path_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string(fallback);
}

// const char* variant for the suites that pass the result straight to
// fopen()/the C API. Both the getenv pointer and the literal `fallback` have
// static lifetime, so the returned pointer is safe to hold. The `fallback`
// stays at the call site (model<->test mapping visible); only the
// getenv-or-default mechanic is shared.
inline const char* env_cstr_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// --- Guards ---------------------------------------------------------------

// A var that is UNSET means "nobody asked for this model" and skipping is
// right. A var that is SET but names a path that is not there is a
// CONFIGURATION error, and skipping it is the trap this suite fell into:
// `make test-e2e` mounted the repo's models/ directory, whose entries are
// absolute symlinks into $HOME/models and therefore dangle inside the
// container, so every model path missed, every model test skipped, and the
// whole battery reported green without loading a single checkpoint.
//
// Letting the bad path fall through to imp_model_load() is not the fix either:
// that reports a bare IMP_ERROR against a coherence assertion and never names
// the path, so a wrong mount reads as a product failure.
//
// It has to be a void helper invoked through ASSERT_NO_FATAL_FAILURE, for the
// same reason GTEST_SKIP() stays at the call site above: ASSERT_* returns from
// its *enclosing* function.
//
//   ASSERT_NO_FATAL_FAILURE(imp_test::require_readable(path, imp_test::kEnvModel));
//
// stat() rather than ifstream, because half these vars name a SafeTensors
// directory and the other half a .gguf file, and both must pass.
inline void require_readable(const char* path, const char* var) {
    ASSERT_NE(path, nullptr) << var << " is unset";
    struct stat st {};
    ASSERT_EQ(::stat(path, &st), 0)
        << var << " is set to '" << path << "', which does not exist or cannot be read.\n"
        << "Point it at a checkpoint that is there, or unset it to skip these tests.";
}

inline void require_readable(const std::string& path, const char* var) {
    ASSERT_NO_FATAL_FAILURE(require_readable(path.c_str(), var));
}

// The env_path_or() form, where an unset var falls back to the documented
// /models/... path. Here a missing path is genuinely ambiguous: the default is
// a convenience, so "not installed" must stay a skip. Only an EXPLICITLY set
// var is a promise that the checkpoint is there, so only that is enforced.
// Call this before the exists()-check that skips.
inline void require_readable_if_set(const char* var) {
    const char* v = std::getenv(var);
    if (!v || !*v)
        return;
    ASSERT_NO_FATAL_FAILURE(require_readable(v, var));
}

}  // namespace imp_test

#endif  // IMP_TESTS_TEST_MODELS_H
