// Central IMP_TEST_MODEL* env registry: single source of truth for env names and path
// resolution, replacing copy-pasted getenv gating that silently disabled tests on a typo.
// GTEST_SKIP() stays at call sites: it returns from the enclosing function, not a helper.

#ifndef IMP_TESTS_TEST_MODELS_H
#define IMP_TESTS_TEST_MODELS_H

#include <gtest/gtest.h>

#include <sys/stat.h>

#include <cstdlib>
#include <string>

namespace imp_test {

// Primary model env var (IMP_TEST_MODEL); GGUF vs SafeTensors is sniffed from the path,
// target model set per suite (e.g. Qwen3-8B Q8_0 for greedy-lock/prefix-cache gates).
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

// LLM-Compressor/ModelOpt loader E2E export dirs. Previously hardcoded with one default
// naming a nonexistent dir (Qwen3-Coder-30B-A3B-FP4 vs the actual -Instruct-FP4), which
// skipped silently.
inline constexpr const char* kEnvModelModeloptCoder = "IMP_TEST_MODEL_MODELOPT_CODER";
inline constexpr const char* kEnvModelMistral = "IMP_TEST_MODEL_MISTRAL";

// Dense native-NVFP4 checkpoint for the batch-invariance instrument (AUDIT_arch_2026 D-2).
// Dense on purpose: MoE routing flips would sit on top of the numerics under test.
// Default: /models/Qwen3-14B-NVFP4.
inline constexpr const char* kEnvModelNvfp4 = "IMP_TEST_MODEL_NVFP4";

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

// const char* variant for callers passing straight to fopen()/the C API; getenv pointer and
// literal fallback both have static lifetime, so the returned pointer stays valid.
inline const char* env_cstr_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// --- Guards ---------------------------------------------------------------

// Unset var: skip is right. A set var naming an unreadable path is a config error, not a
// skip (a wrong Docker mount with dangling models/ symlinks once made every model test skip
// silently). Void helper + ASSERT_NO_FATAL_FAILURE: ASSERT_* returns from the enclosing function.
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

// env_path_or()'s fallback to /models/... is a convenience, so an unset var stays a skip.
// Only an explicitly set var promises the checkpoint exists; call before the exists()-skip check.
inline void require_readable_if_set(const char* var) {
    const char* v = std::getenv(var);
    if (!v || !*v)
        return;
    ASSERT_NO_FATAL_FAILURE(require_readable(v, var));
}

}  // namespace imp_test

#endif  // IMP_TESTS_TEST_MODELS_H
