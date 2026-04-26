# llm-compressor NVFP4 SafeTensors Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a translation-layer-based loader to imp that lets the existing NVFP4 compute pipeline consume models quantized with `vllm-project/llm-compressor` (RedHatAI, Mistral official, vLLM community).

**Architecture:** Pure naming translation at SafeTensors enumeration time. llm-compressor tensor names (`*.weight_packed`, `*.weight_global_scale`, `*.input_global_scale`) are renamed to modelopt names (`*.weight`, `*.weight_scale_2`, `*.input_scale`) before `weight_map.cpp` sees them. Multimodal prefix `model.language_model.*` is stripped silently with a one-time INFO log. Vision-tower tensors and Gemma-4-specific extra scales are skipped with WARN. Compute kernels and weight upload are unchanged.

**Tech Stack:** C++20, custom mini-parser for `recipe.yaml` (no new YAML library), GoogleTest for unit + e2e tests, Docker for build/run.

**Source spec:** `docs/superpowers/specs/2026-04-26-llm-compressor-nvfp4-loader-design.md`

**Build/test environment:** All builds run in Docker — host has no nvcc/cuda toolchain. Use `docker build -t imp:test ... .` (full rebuild ~3-15 min depending on cache state) + `docker run --rm --gpus all -v /home/kekz/github.com/kekzl/imp/models:/models imp:test imp-tests --gtest_filter="..."`. Unit tests (CPU-only) can run via `imp-tests` without `--gpus all`.

---

## File Structure

| File | Purpose |
|---|---|
| `src/model/llm_compressor_loader.h` (NEW) | Public API: `NameTranslation`, `TranslationCounters`, `translate_name()`, `log_summary()`, `parse_llm_compressor_recipe()` |
| `src/model/llm_compressor_loader.cpp` (NEW) | Implementation: rename table, prefix strip, skip patterns, mini-YAML parser |
| `src/model/hf_config_loader.h` (EDIT) | Add `NvFP4Format` enum + extend `NvFP4Config` struct |
| `src/model/hf_config_loader.cpp` (EDIT) | Add `detect_nvfp4_format()`, `probe_first_tensor_name()`; dispatch from `load_nvfp4_config()` |
| `src/model/safetensors_loader.cpp` (EDIT) | Hook translation into `load_shard()` tensor enumeration loop |
| `tests/test_llm_compressor_loader.cpp` (NEW) | Unit tests for translation, parser, format detection |
| `tests/test_e2e_llm_compressor.cpp` (NEW) | E2E load + coherence + modelopt-regression tests |
| `CMakeLists.txt` (EDIT) | Register new source + test files |

---

## Task 1: Add `NvFP4Format` enum to `NvFP4Config`

**Files:**
- Modify: `src/model/hf_config_loader.h:44-49`

- [ ] **Step 1: Edit the header to add the enum and extend the struct**

In `src/model/hf_config_loader.h`, replace the existing `NvFP4Config` struct (around line 44) with:

```cpp
    // Source format of NVFP4 quantization metadata.
    enum class NvFP4Format {
        MODELOPT,         // hf_quant_config.json from NVIDIA Model Optimizer
        LLM_COMPRESSOR,   // recipe.yaml from vllm-project/llm-compressor
    };

    // NVFP4 quantization config. Sourced from hf_quant_config.json (modelopt)
    // or recipe.yaml (llm-compressor) — see `format` field for which.
    struct NvFP4Config {
        int group_size = 16;                       // micro-scale group (default: 16 for NVFP4)
        std::string kv_cache_quant_algo;           // "FP8" or empty (modelopt only)
        std::vector<std::string> exclude_modules;  // e.g. ["lm_head"]
        NvFP4Format format = NvFP4Format::MODELOPT;
    };
    static bool load_nvfp4_config(const std::string& model_dir, NvFP4Config& cfg);
```

- [ ] **Step 2: Verify the header compiles**

Run:

```bash
cd /home/kekz/github.com/kekzl/imp-perf-fp4
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -5
```

Expected: build succeeds (or fails downstream — header alone shouldn't break anything yet because no code references the new field).

- [ ] **Step 3: Commit**

```bash
git add src/model/hf_config_loader.h
git commit -m "feat(loader): add NvFP4Format enum to NvFP4Config

Preparatory change for llm-compressor NVFP4 loader. format defaults
to MODELOPT so existing call sites are unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Create `llm_compressor_loader.h` skeleton

**Files:**
- Create: `src/model/llm_compressor_loader.h`

- [ ] **Step 1: Write the new header**

Create `src/model/llm_compressor_loader.h`:

```cpp
#pragma once

#include "model/hf_config_loader.h"

#include <string>
#include <unordered_map>

namespace imp::llm_compressor {

// Counts of translation actions taken across one shard load. Used to emit
// a single summary log line at end of load instead of one log per tensor.
struct TranslationCounters {
    int suffix_renames = 0;       // .weight_packed → .weight, etc.
    int prefix_strips = 0;        // model.language_model. → model.
    int vision_skipped = 0;       // model.vision_tower.* / model.visual.*
    int gemma4_extra_skipped = 0; // .layer_scalar / .per_expert_scale / Gemma-4 .scale
    int passed_through = 0;       // unknown patterns, returned unchanged
};

// Result of translating one tensor name. SKIP means do not emit this tensor.
struct NameTranslation {
    enum Action { EMIT, SKIP };
    Action action;
    std::string out_name;  // populated when action == EMIT
};

// Apply rename + prefix-strip + skip rules deterministically. Increments
// the matching counter. Pure apart from counter mutation.
NameTranslation translate_name(const std::string& in,
                               TranslationCounters& counters);

// Emit one INFO log summarizing what translate_name() did across a shard.
// Call once at the end of the enumerate-tensors loop in load_shard().
void log_summary(const TranslationCounters& counters);

// Parse a recipe.yaml file and populate cfg. Returns true if the file is
// a NVFP4 recipe in the expected QuantizationModifier shape; false on
// missing file, parse error, or unsupported scheme.
bool parse_recipe_yaml(const std::string& model_dir,
                       imp::HFConfigLoader::NvFP4Config& cfg);

} // namespace imp::llm_compressor
```

- [ ] **Step 2: Verify it compiles standalone**

Run:

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -5
```

Expected: build still succeeds (header is not yet referenced anywhere).

- [ ] **Step 3: Commit**

```bash
git add src/model/llm_compressor_loader.h
git commit -m "feat(loader): add llm_compressor_loader.h skeleton

Public API surface for the translation-layer loader. No implementation
yet; tasks follow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Implement `translate_name()` suffix renames + first unit tests

**Files:**
- Create: `src/model/llm_compressor_loader.cpp`
- Create: `tests/test_llm_compressor_loader.cpp`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_compressor_loader.cpp`:

```cpp
#include "model/llm_compressor_loader.h"
#include <gtest/gtest.h>

using namespace imp::llm_compressor;

TEST(LlmCompressorTranslate, RenamesWeightPacked) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.self_attn.q_proj.weight_packed", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(c.suffix_renames, 1);
}

TEST(LlmCompressorTranslate, RenamesWeightGlobalScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.mlp.gate_proj.weight_global_scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.mlp.gate_proj.weight_scale_2");
    EXPECT_EQ(c.suffix_renames, 1);
}

TEST(LlmCompressorTranslate, RenamesInputGlobalScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.5.self_attn.k_proj.input_global_scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.5.self_attn.k_proj.input_scale");
    EXPECT_EQ(c.suffix_renames, 1);
}

TEST(LlmCompressorTranslate, WeightScaleUnchanged) {
    // .weight_scale exists in BOTH formats with identical layout, no rename.
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.mlp.up_proj.weight_scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.mlp.up_proj.weight_scale");
    EXPECT_EQ(c.suffix_renames, 0);
    EXPECT_EQ(c.passed_through, 1);
}

TEST(LlmCompressorTranslate, UnknownPassesThrough) {
    TranslationCounters c{};
    auto t = translate_name("model.embed_tokens.weight", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.embed_tokens.weight");
    EXPECT_EQ(c.passed_through, 1);
}
```

- [ ] **Step 2: Add the test file to CMakeLists.txt (test-core block)**

Edit `CMakeLists.txt`. Find the `imp_add_test_module(test-core SOURCES ...)` block and append `tests/test_llm_compressor_loader.cpp`:

```cmake
    imp_add_test_module(test-core SOURCES
        tests/test_tensor.cpp
        tests/test_tensor_kind_table.cpp
        tests/test_tensor_kind_matcher.cpp
        tests/test_tensor_kind_coverage.cpp
        tests/test_kv_cache.cpp
        tests/test_gguf_loader.cpp
        tests/test_llm_compressor_loader.cpp
    )
```

- [ ] **Step 3: Run the test (will fail to link — function not implemented)**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -10
```

Expected: link error referring to undefined `imp::llm_compressor::translate_name`.

- [ ] **Step 4: Implement minimal `translate_name()` for suffix renames**

Create `src/model/llm_compressor_loader.cpp`:

```cpp
#include "model/llm_compressor_loader.h"

#include "core/logging.h"

#include <fstream>
#include <sstream>
#include <string_view>

namespace imp::llm_compressor {

namespace {

// Return true and update `name` in place if it ends with `from`. Replaces
// the suffix with `to`.
bool try_rename_suffix(std::string& name, std::string_view from, std::string_view to) {
    if (name.size() < from.size()) return false;
    if (name.compare(name.size() - from.size(), from.size(), from) != 0) return false;
    name.replace(name.size() - from.size(), from.size(), to);
    return true;
}

} // namespace

NameTranslation translate_name(const std::string& in, TranslationCounters& counters) {
    std::string out = in;

    // Step 1: suffix renames (mutually exclusive — try in order, stop at first match)
    if (try_rename_suffix(out, ".weight_packed", ".weight")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }
    if (try_rename_suffix(out, ".weight_global_scale", ".weight_scale_2")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }
    if (try_rename_suffix(out, ".input_global_scale", ".input_scale")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }

    // No rule matched — pass through unchanged
    counters.passed_through++;
    return {NameTranslation::EMIT, std::move(out)};
}

void log_summary(const TranslationCounters& c) {
    IMP_LOG_INFO("llm-compressor format: %d suffix renames, %d prefix strips, "
                 "%d vision tensors skipped, %d Gemma-4 extras skipped, "
                 "%d pass-through",
                 c.suffix_renames, c.prefix_strips,
                 c.vision_skipped, c.gemma4_extra_skipped,
                 c.passed_through);
}

bool parse_recipe_yaml(const std::string& /*model_dir*/,
                       imp::HFConfigLoader::NvFP4Config& /*cfg*/) {
    // Implemented in a later task.
    return false;
}

} // namespace imp::llm_compressor
```

- [ ] **Step 5: Add the new source to CMakeLists imp library target**

Edit `CMakeLists.txt`. Find the `add_library(imp ...)` block (search for `src/model/safetensors_loader.cpp` to find the model-sources area) and add `src/model/llm_compressor_loader.cpp` to the same source list:

```bash
grep -n "src/model/safetensors_loader.cpp\|src/model/hf_config_loader.cpp" CMakeLists.txt
```

Expected: shows the line(s) where existing model loader sources are listed. Add a line `src/model/llm_compressor_loader.cpp` immediately after the safetensors_loader entry.

- [ ] **Step 6: Build + run tests**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressorTranslate.*"
```

Expected: 5 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/model/llm_compressor_loader.cpp tests/test_llm_compressor_loader.cpp CMakeLists.txt
git commit -m "feat(loader): translate_name() suffix renames + tests

Maps llm-compressor weight tensor naming to modelopt naming:
- .weight_packed → .weight
- .weight_global_scale → .weight_scale_2
- .input_global_scale → .input_scale

Pass-through for unknown patterns. 5 unit tests cover the four named
suffixes plus pass-through behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add prefix strip + skip patterns to `translate_name()`

**Files:**
- Modify: `src/model/llm_compressor_loader.cpp`
- Modify: `tests/test_llm_compressor_loader.cpp`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_llm_compressor_loader.cpp`:

```cpp
TEST(LlmCompressorTranslate, StripsMultimodalPrefix) {
    TranslationCounters c{};
    auto t = translate_name(
        "model.language_model.layers.0.self_attn.q_proj.weight_packed", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);
    EXPECT_EQ(t.out_name, "model.layers.0.self_attn.q_proj.weight");
    EXPECT_EQ(c.suffix_renames, 1);
    EXPECT_EQ(c.prefix_strips, 1);
}

TEST(LlmCompressorTranslate, SkipsVisionTower) {
    TranslationCounters c{};
    auto t = translate_name(
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.vision_skipped, 1);
}

TEST(LlmCompressorTranslate, SkipsVisualPrefix) {
    // Qwen3-VL naming uses model.visual.* instead of model.vision_tower.*
    TranslationCounters c{};
    auto t = translate_name("model.visual.blocks.0.attn.q_proj.weight", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.vision_skipped, 1);
}

TEST(LlmCompressorTranslate, SkipsLayerScalar) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.layer_scalar", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.gemma4_extra_skipped, 1);
}

TEST(LlmCompressorTranslate, SkipsPerExpertScale) {
    TranslationCounters c{};
    auto t = translate_name("model.layers.5.experts.per_expert_scale", c);
    EXPECT_EQ(t.action, NameTranslation::SKIP);
    EXPECT_EQ(c.gemma4_extra_skipped, 1);
}

TEST(LlmCompressorTranslate, DoesNotSkipProjScale) {
    // .scale suffix on a recognized projection name is NOT a Gemma-4 extra.
    // (Defensive against false-positive blanket .scale skip.)
    TranslationCounters c{};
    auto t = translate_name("model.layers.0.self_attn.q_proj.scale", c);
    EXPECT_EQ(t.action, NameTranslation::EMIT);  // pass through
    EXPECT_EQ(c.gemma4_extra_skipped, 0);
}
```

- [ ] **Step 2: Run tests — expect 6 failures**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressorTranslate.*"
```

Expected: 5 pass (from Task 3), 6 fail (the new ones — strip and skip not implemented).

- [ ] **Step 3: Add prefix strip and skip patterns to `translate_name()`**

In `src/model/llm_compressor_loader.cpp`, replace the existing `translate_name()` body with the following (keeps the suffix-rename block, adds prefix strip and skip patterns BEFORE the suffix-rename step so prefix is normalized first):

```cpp
namespace {

bool try_rename_suffix(std::string& name, std::string_view from, std::string_view to) {
    if (name.size() < from.size()) return false;
    if (name.compare(name.size() - from.size(), from.size(), from) != 0) return false;
    name.replace(name.size() - from.size(), from.size(), to);
    return true;
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Recognized projection names whose `.scale` is NOT a Gemma-4 extra.
// If a tensor name segment immediately before `.scale` matches one of these,
// the tensor passes through (handled later by weight_map).
bool is_proj_segment(std::string_view name_before_dot_scale) {
    // Last segment between dots — find last '.' in the substring.
    auto pos = name_before_dot_scale.rfind('.');
    std::string_view last = (pos == std::string_view::npos)
                                ? name_before_dot_scale
                                : name_before_dot_scale.substr(pos + 1);
    return last == "q_proj" || last == "k_proj" || last == "v_proj" ||
           last == "o_proj" || last == "gate_proj" || last == "up_proj" ||
           last == "down_proj";
}

} // namespace

NameTranslation translate_name(const std::string& in, TranslationCounters& counters) {
    std::string out = in;

    // Step 1: skip patterns (vision tower) — check raw input before any mutation.
    if (starts_with(out, "model.vision_tower.") || starts_with(out, "model.visual.")) {
        counters.vision_skipped++;
        return {NameTranslation::SKIP, ""};
    }

    // Step 2: skip Gemma-4 extras.
    if (ends_with(out, ".layer_scalar") || ends_with(out, ".per_expert_scale")) {
        counters.gemma4_extra_skipped++;
        return {NameTranslation::SKIP, ""};
    }
    if (ends_with(out, ".scale")) {
        // Only skip if the segment immediately before .scale is NOT a known proj.
        std::string_view before_scale(out.data(), out.size() - 6); // strip ".scale"
        if (!is_proj_segment(before_scale)) {
            counters.gemma4_extra_skipped++;
            return {NameTranslation::SKIP, ""};
        }
        // else fall through (pass-through handles it).
    }

    // Step 3: prefix strip (multimodal language_model wrapper).
    static constexpr const char kMultimodalPrefix[] = "model.language_model.";
    static constexpr size_t kMultimodalPrefixLen = sizeof(kMultimodalPrefix) - 1;
    if (starts_with(out, kMultimodalPrefix)) {
        out = "model." + out.substr(kMultimodalPrefixLen);
        counters.prefix_strips++;
        // Continue to suffix-rename step below.
    }

    // Step 4: suffix renames (mutually exclusive).
    if (try_rename_suffix(out, ".weight_packed", ".weight")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }
    if (try_rename_suffix(out, ".weight_global_scale", ".weight_scale_2")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }
    if (try_rename_suffix(out, ".input_global_scale", ".input_scale")) {
        counters.suffix_renames++;
        return {NameTranslation::EMIT, std::move(out)};
    }

    // Step 5: pass through (still increments prefix_strips counter from above
    // if we did strip; suffix_renames stays 0 in that case).
    counters.passed_through++;
    return {NameTranslation::EMIT, std::move(out)};
}
```

- [ ] **Step 4: Run tests — expect all 11 pass**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressorTranslate.*"
```

Expected: 11 pass, 0 fail.

- [ ] **Step 5: Commit**

```bash
git add src/model/llm_compressor_loader.cpp tests/test_llm_compressor_loader.cpp
git commit -m "feat(loader): translate_name() prefix strip + skip patterns

- Strip 'model.language_model.' multimodal prefix
- Skip 'model.vision_tower.*' / 'model.visual.*' (vision encoder)
- Skip Gemma-4 extras (.layer_scalar, .per_expert_scale, .scale)
- Defensive: .scale on q_proj/k_proj/v_proj/o_proj/{gate,up,down}_proj
  passes through (not a Gemma-4 extra)

6 new tests covering each path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Implement `parse_recipe_yaml()` mini-parser

**Files:**
- Modify: `src/model/llm_compressor_loader.cpp`
- Modify: `tests/test_llm_compressor_loader.cpp`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_llm_compressor_loader.cpp`:

```cpp
#include <fstream>
#include <cstdlib>

namespace {

std::string write_temp_recipe(const std::string& content) {
    std::string path = std::string(std::getenv("TMPDIR") ?: "/tmp")
                       + "/recipe_" + std::to_string(::getpid()) + ".yaml";
    // Caller must wrap path so that a "model_dir" gets the file as model_dir/recipe.yaml.
    // We'll create a temp dir and place recipe.yaml inside it.
    std::string dir = path + ".d";
    std::string mkdir = "mkdir -p '" + dir + "'";
    std::system(mkdir.c_str());
    std::ofstream out(dir + "/recipe.yaml");
    out << content;
    out.close();
    return dir;
}

void cleanup_temp_recipe(const std::string& dir) {
    std::string rm = "rm -rf '" + dir + "'";
    std::system(rm.c_str());
}

} // namespace

TEST(LlmCompressorRecipe, ParsesGemma4Recipe) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, 're:.*embed.*', 're:.*router', 're:.*vision_tower.*']
      scheme: NVFP4
      bypass_divisibility_checks: false
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.group_size, 16);
    ASSERT_EQ(cfg.exclude_modules.size(), 4u);
    EXPECT_EQ(cfg.exclude_modules[0], "lm_head");
    EXPECT_EQ(cfg.exclude_modules[1], "re:.*embed.*");
    cleanup_temp_recipe(dir);
}

TEST(LlmCompressorRecipe, ParsesQwen36Recipe) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: ['re:.*lm_head', 're:visual.*', 're:model.visual.*', 're:.*mlp.gate$', 're:.*embed_tokens$', 're:.*shared_expert_gate$', 're:.*linear_attn.*']
      scheme: NVFP4
      bypass_divisibility_checks: false
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.exclude_modules.size(), 7u);
    EXPECT_EQ(cfg.exclude_modules[3], "re:.*mlp.gate$");
    cleanup_temp_recipe(dir);
}

TEST(LlmCompressorRecipe, RejectsNonNVFP4Scheme) {
    std::string dir = write_temp_recipe(R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: W8A8
)");
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml(dir, cfg);
    EXPECT_FALSE(ok);
    cleanup_temp_recipe(dir);
}

TEST(LlmCompressorRecipe, ReturnsFalseOnMissingFile) {
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::llm_compressor::parse_recipe_yaml("/tmp/nonexistent_dir_xyz", cfg);
    EXPECT_FALSE(ok);
}
```

- [ ] **Step 2: Run — expect 4 failures**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressorRecipe.*"
```

Expected: 4 fail (parse_recipe_yaml returns false stub).

- [ ] **Step 3: Implement `parse_recipe_yaml()`**

In `src/model/llm_compressor_loader.cpp`, replace the stub `parse_recipe_yaml()` with the full implementation:

```cpp
namespace {

// Strip leading whitespace and quotes from a value substring.
std::string trim_value(std::string_view sv) {
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) sv.remove_prefix(1);
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' ||
                           sv.back() == '\r' || sv.back() == '\n')) sv.remove_suffix(1);
    if (sv.size() >= 2 && (sv.front() == '\'' || sv.front() == '"') &&
        sv.front() == sv.back()) {
        sv.remove_prefix(1);
        sv.remove_suffix(1);
    }
    return std::string(sv);
}

// Parse a bracket-array `[a, b, c]` (single-line). Returns vector of values.
std::vector<std::string> parse_bracket_array(std::string_view body) {
    std::vector<std::string> out;
    // Find content inside brackets.
    auto lb = body.find('[');
    auto rb = body.rfind(']');
    if (lb == std::string_view::npos || rb == std::string_view::npos || rb <= lb) return out;
    std::string_view inner = body.substr(lb + 1, rb - lb - 1);

    size_t start = 0;
    bool in_quote = false;
    char quote_char = 0;
    for (size_t i = 0; i <= inner.size(); i++) {
        bool at_end = (i == inner.size());
        char c = at_end ? ',' : inner[i];
        if (!at_end && in_quote) {
            if (c == quote_char) in_quote = false;
            continue;
        }
        if (!at_end && (c == '\'' || c == '"')) {
            in_quote = true;
            quote_char = c;
            continue;
        }
        if (c == ',') {
            std::string item = trim_value(inner.substr(start, i - start));
            if (!item.empty()) out.push_back(std::move(item));
            start = i + 1;
        }
    }
    return out;
}

} // namespace

bool parse_recipe_yaml(const std::string& model_dir,
                       imp::HFConfigLoader::NvFP4Config& cfg) {
    std::ifstream in(model_dir + "/recipe.yaml");
    if (!in.good()) return false;

    std::string scheme;
    std::vector<std::string> ignore_list;
    bool seen_quant_mod = false;

    std::string line;
    while (std::getline(in, line)) {
        // Look for the QuantizationModifier subkeys: scheme, ignore, group_size.
        // Strip leading whitespace for keyword matching.
        std::string_view sv(line);
        while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) sv.remove_prefix(1);

        if (sv.find("QuantizationModifier:") == 0) { seen_quant_mod = true; continue; }
        if (!seen_quant_mod) continue;

        if (sv.find("scheme:") == 0) {
            scheme = trim_value(sv.substr(7));
        } else if (sv.find("ignore:") == 0) {
            ignore_list = parse_bracket_array(sv.substr(7));
        } else if (sv.find("group_size:") == 0) {
            try { cfg.group_size = std::stoi(trim_value(sv.substr(11))); }
            catch (...) { /* keep default */ }
        }
    }

    if (!seen_quant_mod) {
        IMP_LOG_ERROR("recipe.yaml has no QuantizationModifier block");
        return false;
    }
    if (scheme != "NVFP4" && scheme != "NVFP4_W4A16") {
        IMP_LOG_ERROR("recipe.yaml scheme '%s' not supported (need NVFP4 or NVFP4_W4A16)",
                      scheme.c_str());
        return false;
    }

    cfg.exclude_modules = std::move(ignore_list);
    cfg.format = imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
    IMP_LOG_INFO("NVFP4 model (llm-compressor): scheme=%s, group_size=%d, exclude=%zu modules",
                 scheme.c_str(), cfg.group_size, cfg.exclude_modules.size());
    return true;
}
```

- [ ] **Step 4: Run tests — expect 4 pass**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressorRecipe.*"
```

Expected: 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/model/llm_compressor_loader.cpp tests/test_llm_compressor_loader.cpp
git commit -m "feat(loader): recipe.yaml mini-parser

Parses the QuantizationModifier subset of llm-compressor recipes:
scheme (NVFP4/NVFP4_W4A16), ignore patterns (bracket-array), group_size.
Sets cfg.format = LLM_COMPRESSOR. Rejects non-NVFP4 schemes with clear
error.

4 unit tests cover Gemma-4 + Qwen3.6 real recipes plus rejection +
missing-file paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement format detection

**Files:**
- Modify: `src/model/hf_config_loader.cpp`
- Modify: `src/model/hf_config_loader.h` (add helper signature)
- Modify: `tests/test_llm_compressor_loader.cpp`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_llm_compressor_loader.cpp`:

```cpp
TEST(LlmCompressorFormatDetect, PrefersModeloptWhenBothPresent) {
    std::string dir = std::string(std::getenv("TMPDIR") ?: "/tmp")
                      + "/fmt_both_" + std::to_string(::getpid());
    std::system(("mkdir -p '" + dir + "'").c_str());
    std::ofstream(dir + "/hf_quant_config.json") << R"({"quantization":{"quant_algo":"NVFP4"}})";
    std::ofstream(dir + "/recipe.yaml") << "default_stage:\n  default_modifiers:\n    QuantizationModifier:\n      scheme: NVFP4\n";

    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::MODELOPT);

    std::system(("rm -rf '" + dir + "'").c_str());
}

TEST(LlmCompressorFormatDetect, DetectsLlmCompressorByRecipeYaml) {
    std::string dir = std::string(std::getenv("TMPDIR") ?: "/tmp")
                      + "/fmt_lc_" + std::to_string(::getpid());
    std::system(("mkdir -p '" + dir + "'").c_str());
    std::ofstream(dir + "/recipe.yaml") << R"(default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: NVFP4
)";
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_TRUE(ok);
    EXPECT_EQ(cfg.format, imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR);

    std::system(("rm -rf '" + dir + "'").c_str());
}

TEST(LlmCompressorFormatDetect, ReturnsFalseWhenNoConfigPresent) {
    std::string dir = std::string(std::getenv("TMPDIR") ?: "/tmp")
                      + "/fmt_none_" + std::to_string(::getpid());
    std::system(("mkdir -p '" + dir + "'").c_str());
    // Empty dir, no config files.
    imp::HFConfigLoader::NvFP4Config cfg;
    bool ok = imp::HFConfigLoader::load_nvfp4_config(dir, cfg);
    EXPECT_FALSE(ok);
    std::system(("rm -rf '" + dir + "'").c_str());
}
```

- [ ] **Step 2: Run — expect 3 failures (load_nvfp4_config doesn't dispatch yet)**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressorFormatDetect.*"
```

Expected: 3 fail.

- [ ] **Step 3: Wire format detection into `load_nvfp4_config()`**

In `src/model/hf_config_loader.cpp`, find the existing `load_nvfp4_config` function (around line 392). Add the include at the top:

```cpp
#include "model/llm_compressor_loader.h"
```

Then REPLACE the entire `load_nvfp4_config` function with:

```cpp
namespace {

bool file_exists_at(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

} // namespace

bool HFConfigLoader::load_nvfp4_config(const std::string& model_dir, NvFP4Config& cfg) {
    bool has_modelopt = file_exists_at(model_dir + "/hf_quant_config.json");
    bool has_compressor = file_exists_at(model_dir + "/recipe.yaml");

    if (has_modelopt && has_compressor) {
        IMP_LOG_WARN("Both quant config files present in %s — preferring modelopt",
                     model_dir.c_str());
    }

    if (has_modelopt) {
        // Existing modelopt parsing (unchanged from before — reproduce here).
        std::string path = model_dir + "/hf_quant_config.json";
        JValue root;
        if (!parse_json_file(path, root)) return false;

        const JValue* quant = jobj_find(root, "quantization");
        if (!quant || quant->type != JType::OBJECT) return false;

        const JValue* algo = jobj_find(*quant, "quant_algo");
        if (!algo || algo->type != JType::STRING) return false;
        if (algo->str_val != "NVFP4" && algo->str_val != "nvfp4") return false;

        jobj_get_int(*quant, "group_size", cfg.group_size);

        const JValue* kv_algo = jobj_find(*quant, "kv_cache_quant_algo");
        if (kv_algo && kv_algo->type == JType::STRING)
            cfg.kv_cache_quant_algo = kv_algo->str_val;

        const JValue* exclude = jobj_find(*quant, "exclude_modules");
        if (exclude && exclude->type == JType::ARRAY) {
            for (const auto& v : exclude->arr) {
                if (v.type == JType::STRING)
                    cfg.exclude_modules.push_back(v.str_val);
            }
        }

        cfg.format = NvFP4Format::MODELOPT;
        IMP_LOG_INFO("NVFP4 model (Model Optimizer): group_size=%d, kv_cache=%s, exclude=%zu modules",
                     cfg.group_size, cfg.kv_cache_quant_algo.c_str(), cfg.exclude_modules.size());
        return true;
    }

    if (has_compressor) {
        return imp::llm_compressor::parse_recipe_yaml(model_dir, cfg);
    }

    // Neither file present.
    return false;
}
```

- [ ] **Step 4: Run tests — expect 3 pass + all prior tests still pass**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressor*"
```

Expected: all 18 LlmCompressor* tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/model/hf_config_loader.cpp tests/test_llm_compressor_loader.cpp
git commit -m "feat(loader): format detection via file presence

load_nvfp4_config dispatches on hf_quant_config.json (modelopt) vs
recipe.yaml (llm-compressor). Both present → prefer modelopt + WARN.
Neither present → return false (no NVFP4 detected).

Sets cfg.format consistently. 3 unit tests cover all three cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Hook translation into `safetensors_loader.cpp`

**Files:**
- Modify: `src/model/safetensors_loader.cpp`

- [ ] **Step 1: Inspect current load_shard tensor enumeration**

Run:

```bash
sed -n '385,460p' /home/kekz/github.com/kekzl/imp-perf-fp4/src/model/safetensors_loader.cpp
```

Find the line `tensor_map.emplace(tensor_name, t);` (around line 450). The translation hook goes immediately before this `emplace`.

- [ ] **Step 2: Add include + extend load_shard signature**

At the top of `src/model/safetensors_loader.cpp`, add:

```cpp
#include "model/llm_compressor_loader.h"
```

Then change the static function signature from:

```cpp
static bool load_shard(const std::string& path,
                       std::unordered_map<std::string, Tensor>& tensor_map,
                       ShardInfo& shard) {
```

to:

```cpp
static bool load_shard(const std::string& path,
                       std::unordered_map<std::string, Tensor>& tensor_map,
                       ShardInfo& shard,
                       bool llm_compressor_format,
                       imp::llm_compressor::TranslationCounters& counters) {
```

- [ ] **Step 3: Insert translation call into the enumerate loop**

Inside `load_shard`, find the line `for (const auto& kv : root.obj) {` (around line 418). Replace the body up to and including `tensor_map.emplace(tensor_name, t);` with:

```cpp
    for (const auto& kv : root.obj) {
        std::string tensor_name = kv.first;  // copy — may be mutated by translation
        const JValue& tensor_meta = kv.second;

        if (tensor_name == "__metadata__") continue;
        if (tensor_meta.type != JType::OBJECT) continue;

        // Translate llm-compressor names → modelopt names if applicable.
        if (llm_compressor_format) {
            auto t = imp::llm_compressor::translate_name(tensor_name, counters);
            if (t.action == imp::llm_compressor::NameTranslation::SKIP) continue;
            tensor_name = std::move(t.out_name);
        }

        const JValue* dtype_val = jobj_find(tensor_meta, "dtype");
        if (!dtype_val || dtype_val->type != JType::STRING) continue;
        DType dtype = safetensors_dtype(dtype_val->str_val);

        const JValue* shape_val = jobj_find(tensor_meta, "shape");
        if (!shape_val || shape_val->type != JType::ARRAY) continue;

        int ndim = static_cast<int>(shape_val->arr.size());
        if (ndim > kMaxDims) continue;

        int64_t shape[kMaxDims] = {};
        for (int d = 0; d < ndim; d++) {
            shape[d] = shape_val->arr[d].as_int();
        }

        const JValue* offsets_val = jobj_find(tensor_meta, "data_offsets");
        if (!offsets_val || offsets_val->type != JType::ARRAY || offsets_val->arr.size() != 2) continue;

        uint64_t offset_start = static_cast<uint64_t>(offsets_val->arr[0].as_int());
        uint64_t offset_end = static_cast<uint64_t>(offsets_val->arr[1].as_int());

        if (tensor_data_offset + offset_end > file_size) continue;

        void* tensor_ptr = tensor_data_base + offset_start;
        Tensor t(tensor_ptr, dtype, ndim, shape, /*on_device=*/false);
        tensor_map.emplace(tensor_name, t);

        IMP_LOG_DEBUG("Tensor: %s dtype=%s shape=[%ld%s%s%s%s] offsets=[%lu,%lu]",
                      tensor_name.c_str(), dtype_val->str_val.c_str(),
                      (long)shape[0],
                      ndim > 1 ? "," : "", ndim > 1 ? std::to_string(shape[1]).c_str() : "",
                      ndim > 2 ? "," : "", ndim > 2 ? std::to_string(shape[2]).c_str() : "",
                      (unsigned long)offset_start, (unsigned long)offset_end);
    }
```

- [ ] **Step 4: Update load_shard call sites to pass the new args**

Find all call sites of `load_shard(`. Run:

```bash
grep -n "load_shard(" /home/kekz/github.com/kekzl/imp-perf-fp4/src/model/safetensors_loader.cpp
```

Expected: ~1-2 call sites in the same file. For each, you need:

1. Detect format BEFORE the loop over shards (do `load_nvfp4_config` first to know format).
2. Pass the format flag and counters into `load_shard`.

The cleanest way: at the top of the function that loops shards (likely `SafetensorsLoader::load` or similar), add:

```cpp
imp::HFConfigLoader::NvFP4Config probe_cfg;
bool probe_ok = imp::HFConfigLoader::load_nvfp4_config(model_dir, probe_cfg);
bool llm_compressor_format =
    probe_ok && probe_cfg.format == imp::HFConfigLoader::NvFP4Format::LLM_COMPRESSOR;
imp::llm_compressor::TranslationCounters tcounters{};
```

Then change each `load_shard(path, tensor_map, shard)` call to `load_shard(path, tensor_map, shard, llm_compressor_format, tcounters)`.

After the shards loop completes, add:

```cpp
if (llm_compressor_format) {
    imp::llm_compressor::log_summary(tcounters);
}
```

- [ ] **Step 5: Build + run all loader-related unit tests**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm imp:test test-core --gtest_filter="LlmCompressor*"
```

Expected: all 18 tests still pass (no regression in unit tests; integration verified in next task).

- [ ] **Step 6: Commit**

```bash
git add src/model/safetensors_loader.cpp
git commit -m "feat(loader): hook llm-compressor translation into load_shard

When format == LLM_COMPRESSOR, each tensor name passes through
translate_name() before emplace. SKIP is honored. Counters aggregated
across all shards and emitted as one summary log line at end of load.

modelopt path unchanged: llm_compressor_format flag stays false.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: E2E load smoke test for Gemma-4-NVFP4

**Files:**
- Create: `tests/test_e2e_llm_compressor.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/test_e2e_llm_compressor.cpp`:

```cpp
#include "imp/imp.h"
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <cstdlib>
#include <string>

namespace {

bool dir_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

} // namespace

class LlmCompressorE2E : public ::testing::Test {
protected:
    static constexpr const char* kGemma4Dir =
        "/models/Gemma-4-26B-A4B-it-NVFP4";
};

TEST_F(LlmCompressorE2E, Gemma4_LoadsWithoutIMA) {
    if (!dir_exists(kGemma4Dir)) {
        GTEST_SKIP() << "Model not present at " << kGemma4Dir;
    }

    ImpConfig cfg = imp_default_config();
    ImpModel* model = nullptr;
    ImpError rc = imp_model_load(kGemma4Dir, &cfg, &model);
    ASSERT_EQ(rc, IMP_OK) << "imp_model_load failed: " << imp_error_string(rc);
    ASSERT_NE(model, nullptr);

    imp_model_free(model);
}

TEST_F(LlmCompressorE2E, Gemma4_GreedyGeneratesCoherent) {
    if (!dir_exists(kGemma4Dir)) {
        GTEST_SKIP() << "Model not present at " << kGemma4Dir;
    }

    ImpConfig cfg = imp_default_config();
    ImpModel* model = nullptr;
    ASSERT_EQ(imp_model_load(kGemma4Dir, &cfg, &model), IMP_OK);

    ImpContext* ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &ctx), IMP_OK);

    ImpGenerateParams params = imp_default_generate_params();
    params.max_tokens = 16;
    params.temperature = 0.0f;  // greedy

    char output[512] = {};
    ImpError rc = imp_generate(ctx, "What is 2+2?", &params, output, sizeof(output));
    ASSERT_EQ(rc, IMP_OK);

    std::string result(output);
    EXPECT_NE(result.find("4"), std::string::npos)
        << "Output should contain '4'. Got: " << result;

    imp_context_free(ctx);
    imp_model_free(model);
}
```

- [ ] **Step 2: Register the test in CMakeLists**

Edit `CMakeLists.txt`. Find the `imp_add_test_module(test-e2e WITH_STUB SOURCES ...)` block (around line 432) and append:

```cmake
        tests/test_e2e_llm_compressor.cpp
```

- [ ] **Step 3: Build + run (model must be present)**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
    test-e2e --gtest_filter="LlmCompressorE2E.*"
```

Expected: both tests pass. If `LoadsWithoutIMA` passes but `GreedyGeneratesCoherent` fails on the assertion (output doesn't contain "4"), this is the **R1 quality concern** from the spec — Gemma-4 extras may be necessary. Document the actual output and decide whether to ship Phase 1 or escalate to Phase 2 (custom Gemma-4 multiplier kernel).

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_llm_compressor.cpp CMakeLists.txt
git commit -m "test(e2e): smoke + coherence test for Gemma-4-NVFP4

LoadsWithoutIMA: model loads via llm-compressor format without illegal
memory access (was the headline failure before this loader existed).

GreedyGeneratesCoherent: 'What is 2+2?' contains '4'. Validates the
R1 risk from the spec (Gemma-4 extras skipped — output should still
be coherent).

Both auto-skip when /models/Gemma-4-26B-A4B-it-NVFP4 absent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: E2E test for Mistral-Small-3.2-NVFP4 (dense)

**Files:**
- Modify: `tests/test_e2e_llm_compressor.cpp`

- [ ] **Step 1: Download the model (one-time)**

Run on the host:

```bash
mkdir -p /home/kekz/models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4
docker run --rm \
  -v /home/kekz/models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4:/out \
  python:3.12-slim bash -c "
    pip install --quiet huggingface_hub[hf_transfer]
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
        RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4 \
        --local-dir /out --max-workers 8
  "
docker run --rm -v /home/kekz/models:/m alpine \
    chown -R 1000:1000 /m/Mistral-Small-3.2-24B-Instruct-2506-NVFP4
```

Expected: ~14 GB download.

- [ ] **Step 2: Add the test**

Append to `tests/test_e2e_llm_compressor.cpp`:

```cpp
TEST_F(LlmCompressorE2E, MistralSmall_LoadsAndGeneratesCoherent) {
    static constexpr const char* kDir =
        "/models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4";
    if (!dir_exists(kDir)) {
        GTEST_SKIP() << "Model not present at " << kDir;
    }

    ImpConfig cfg = imp_default_config();
    ImpModel* model = nullptr;
    ASSERT_EQ(imp_model_load(kDir, &cfg, &model), IMP_OK);

    ImpContext* ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &ctx), IMP_OK);

    ImpGenerateParams params = imp_default_generate_params();
    params.max_tokens = 16;
    params.temperature = 0.0f;

    char output[512] = {};
    ASSERT_EQ(imp_generate(ctx, "What is the capital of France?",
                           &params, output, sizeof(output)), IMP_OK);

    std::string result(output);
    EXPECT_NE(result.find("Paris"), std::string::npos)
        << "Output should contain 'Paris'. Got: " << result;

    imp_context_free(ctx);
    imp_model_free(model);
}
```

- [ ] **Step 3: Run the test**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
    test-e2e --gtest_filter="LlmCompressorE2E.MistralSmall_*"
```

Expected: passes. If the load fails, dense Mistral may be hitting a different llm-compressor naming wrinkle — capture the failure log + add a focused test in test-core for the specific tensor name.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_llm_compressor.cpp
git commit -m "test(e2e): Mistral-Small-3.2-NVFP4 dense load + coherence

Dense (non-MoE) llm-compressor model. Validates that the translation
layer works for both architecture classes (MoE via Gemma-4, dense via
Mistral-Small). Auto-skips when model absent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Modelopt regression test

**Files:**
- Modify: `tests/test_e2e_llm_compressor.cpp`

- [ ] **Step 1: Add the regression test**

Append to `tests/test_e2e_llm_compressor.cpp`:

```cpp
TEST_F(LlmCompressorE2E, Modelopt_QwenCoder30B_StillWorks) {
    static constexpr const char* kDir =
        "/models/Qwen3-Coder-30B-A3B-FP4";
    if (!dir_exists(kDir)) {
        GTEST_SKIP() << "Model not present at " << kDir;
    }

    ImpConfig cfg = imp_default_config();
    ImpModel* model = nullptr;
    ASSERT_EQ(imp_model_load(kDir, &cfg, &model), IMP_OK)
        << "Modelopt path regressed";

    ImpContext* ctx = nullptr;
    ASSERT_EQ(imp_context_create(model, &ctx), IMP_OK);

    ImpGenerateParams params = imp_default_generate_params();
    params.max_tokens = 32;
    params.temperature = 0.0f;

    char output[512] = {};
    ASSERT_EQ(imp_generate(ctx, "def factorial(n):", &params,
                           output, sizeof(output)), IMP_OK);

    std::string result(output);
    // Loose check — just verify generation produced something non-empty.
    EXPECT_GT(result.size(), 5u) << "Output unexpectedly short: " << result;

    imp_context_free(ctx);
    imp_model_free(model);
}
```

- [ ] **Step 2: Run with Qwen3-Coder-FP4 mounted**

```bash
docker build -t imp:test --build-arg IMP_BUILD_TESTS=ON . 2>&1 | tail -3
docker run --rm --gpus all \
  -v /home/kekz/github.com/kekzl/imp/models:/models \
  imp:test test-e2e --gtest_filter="LlmCompressorE2E.Modelopt_*"
```

Expected: passes. If it fails, the new code regressed the modelopt path — investigate the file_exists check and config dispatch in `load_nvfp4_config`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_llm_compressor.cpp
git commit -m "test(e2e): modelopt regression check (Qwen3-Coder-30B-A3B-FP4)

The new format-detection dispatch in load_nvfp4_config() must not
break the existing modelopt path. This test loads + generates with the
existing modelopt-format model. Auto-skips when model absent.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Manual perf benchmark + validation summary

**Files:**
- Create: `docs/llm-compressor-validation-results.md`

- [ ] **Step 1: Run baseline benchmark — Gemma-4 Q4_K_M GGUF**

```bash
docker run --rm --gpus all \
  -v /home/kekz/github.com/kekzl/imp/models:/models \
  imp:test imp-cli \
    --model /models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
    --bench --bench-pp 512 --bench-reps 5 \
    --max-tokens 256 --temperature 0 2>&1 | grep -E "^pp|^tg"
```

Capture the `pp 512` and `tg 256` numbers.

- [ ] **Step 2: Run new benchmark — Gemma-4 NVFP4 (llm-compressor)**

```bash
docker run --rm --gpus all \
  -v /home/kekz/models:/models \
  imp:test imp-cli \
    --model /models/Gemma-4-26B-A4B-it-NVFP4 \
    --bench --bench-pp 512 --bench-reps 5 \
    --max-tokens 256 --temperature 0 2>&1 | grep -E "^pp|^tg"
```

Capture both numbers.

- [ ] **Step 3: Repeat for Mistral-Small (dense, only NVFP4 — no GGUF baseline needed)**

```bash
docker run --rm --gpus all \
  -v /home/kekz/models:/models \
  imp:test imp-cli \
    --model /models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4 \
    --bench --bench-pp 512 --bench-reps 5 \
    --max-tokens 256 --temperature 0 2>&1 | grep -E "^pp|^tg"
```

- [ ] **Step 4: Run modelopt regression bench (must match prior baseline)**

```bash
docker run --rm --gpus all \
  -v /home/kekz/github.com/kekzl/imp/models:/models \
  imp:test imp-cli \
    --model /models/Qwen3-Coder-30B-A3B-FP4 \
    --bench --bench-pp 512 --bench-reps 5 \
    --max-tokens 256 --temperature 0 2>&1 | grep -E "^pp|^tg"
```

Compare: must be within 5% of pre-change Qwen3-Coder-30B-A3B-FP4 baseline (~13000 pp, ~48 tg per recent measurements).

- [ ] **Step 5: Write validation summary**

Create `docs/llm-compressor-validation-results.md`:

```markdown
# llm-compressor NVFP4 Loader — Validation Results

**Date:** YYYY-MM-DD
**Commit:** <run `git rev-parse HEAD`>
**Hardware:** RTX 5090 (sm_120f), CUDA 13.2.1, Docker `imp:test`

## E2E Tests (gtest)

| Test | Result | Notes |
|---|---|---|
| `Gemma4_LoadsWithoutIMA` | pass / fail | |
| `Gemma4_GreedyGeneratesCoherent` | pass / fail | actual output: "..." |
| `MistralSmall_LoadsAndGeneratesCoherent` | pass / fail | actual output: "..." |
| `Modelopt_QwenCoder30B_StillWorks` | pass / fail | (regression check) |

## Performance — Gemma-4-26B-A4B-it

| Variant | pp512 (tok/s) | tg256 (tok/s) |
|---|---|---|
| Q4_K_M GGUF (baseline) | _fill_in_ | _fill_in_ |
| NVFP4 (llm-compressor, new) | _fill_in_ | _fill_in_ |
| Delta | _fill_in_ % | _fill_in_ % |

## Performance — Mistral-Small-3.2-24B (NVFP4 only)

| Variant | pp512 (tok/s) | tg256 (tok/s) |
|---|---|---|
| NVFP4 (llm-compressor) | _fill_in_ | _fill_in_ |

## Performance — Qwen3-Coder-30B-A3B-FP4 (modelopt regression)

| Variant | pp512 (tok/s) | tg256 (tok/s) |
|---|---|---|
| modelopt NVFP4 (pre-change baseline, reference) | ~13000 | ~48 |
| modelopt NVFP4 (post-change, current) | _fill_in_ | _fill_in_ |
| Delta | _fill_in_ % | _fill_in_ % (must be < 5%) |

## R1 Quality Risk — Disposition

- Gemma-4 extras skipped: _N_ tensors (per loader summary log)
- Greedy "What is 2+2?" output: _quote actual output_
- Decision: ship as Phase 1 / block on Phase 2 — _fill_in_

## R3 W4A4/W4A16 — Status

- Mistral-Small-3.2 has `input_global_scale` tensors: yes/no
- Gemma-4 has `input_global_scale` tensors: yes/no
- W4A16 fallback path needed: yes/no — _decision_

## Followups Discovered During Validation

- (list any that came up)
```

Fill in actual measurements from steps 1-4. Replace placeholders with real numbers.

- [ ] **Step 6: Commit**

```bash
git add docs/llm-compressor-validation-results.md
git commit -m "docs: llm-compressor NVFP4 loader validation results

Records E2E test outcomes + perf comparison vs Q4_K_M GGUF baseline +
modelopt regression check. R1 quality disposition + R3 W4A4/W4A16
follow-up notes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Checklist (run before handoff)

After completing all tasks, verify:

1. **Spec coverage**:
   - Translation rules from spec §"Translation Layer" → Tasks 3, 4 ✓
   - recipe.yaml parser from spec §"Recipe.yaml Parser" → Task 5 ✓
   - Format detection from spec §"Format Detection" → Task 6 ✓
   - Integration from spec §"Integration Points" → Tasks 6, 7 ✓
   - Unit tests from spec §"Validation: Unit Tests" → Tasks 3-6 ✓
   - E2E tests from spec §"Validation: End-to-End Tests" → Tasks 8, 9, 10 ✓
   - Manual perf from spec §"Validation: Manual Performance" → Task 11 ✓

2. **Risks documented in spec are addressed**:
   - R1 (Gemma-4 extras) — Task 8 step 3 documents quality outcome; Task 11 disposition
   - R2 (parser robustness) — Task 5 returns clear error on malformed input
   - R3 (W4A4 vs W4A16) — Task 11 step 5 records which models had `input_global_scale`

3. **TODO backlog from spec is preserved** (not silently absorbed into this plan):
   - GDN+NVFP4 → out of scope, mentioned in spec
   - Multimodal → out of scope (vision tower SKIPped, not LOADed)
   - convert_scales_sfatom fusion → out of scope (separate perf spec)
   - W4A16 first-class → tracked via R3 outcome

---

## Estimated Effort

| Task | Effort |
|---|---|
| 1-2: Setup (header + skeleton) | 30 min |
| 3-4: Translation layer + 11 unit tests | 2 hrs |
| 5: Recipe parser + 4 unit tests | 1.5 hrs |
| 6: Format detection + 3 unit tests | 1 hr |
| 7: Hook into safetensors_loader | 1 hr |
| 8-10: E2E tests + Mistral download | 3 hrs |
| 11: Validation runs + write doc | 2 hrs |
| **Total** | **~11 hours** (~1.5 days) |

Plus Docker rebuild time (~3-5 min per iteration when source changes, totals ~1-2 hrs across the whole plan).
