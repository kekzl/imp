// Central model-env registry for the test suite (TEST_AUDIT.md R6 / #581).
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
// The hardcoded /models/... fallbacks are NOT a defect (TEST_AUDIT.md §8): they
// match the Makefile container mount `-v $(PWD)/models:/models` and skip
// cleanly when the file is absent. They are passed by the caller so the
// model<->test mapping stays visible at the call site; this header only owns
// the env-var names and the getenv mechanics.

#ifndef IMP_TESTS_TEST_MODELS_H
#define IMP_TESTS_TEST_MODELS_H

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

}  // namespace imp_test

#endif  // IMP_TESTS_TEST_MODELS_H
