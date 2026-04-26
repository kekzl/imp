# llm-compressor NVFP4 SafeTensors Loader for imp

**Date**: 2026-04-26
**Status**: Approved design, ready for implementation plan
**Scope**: Phase 1 — text-only Dense + MoE NVFP4 models from llm-compressor format

## Problem

imp's existing NVFP4 loader supports only NVIDIA Model Optimizer (modelopt) output. The de-facto open-source NVFP4 standard is `vllm-project/llm-compressor`, used by:

- **Mistral AI** for official releases (Mistral-Large-3-NVFP4, Mistral-Small-4-NVFP4)
- **Red Hat AI** for the entire RedHatAI/* NVFP4 catalog (~30 models including Qwen3.6, Gemma-4 MoE, Llama-4)
- **Community quantizers** for nearly all post-2026 NVFP4 community uploads
- **vLLM itself** as its first-party quantization toolchain (under the `vllm-project/` GitHub org)

Today, attempting to load any llm-compressor NVFP4 model into imp fails with illegal memory access during the first forward pass — the loader reads tensors with the wrong names/dtypes and writes garbage to GPU memory.

This spec covers a translation-layer-based loader that maps llm-compressor on-disk naming → modelopt naming so the existing NVFP4 compute pipeline (kernels, dispatch, weight upload) handles them unchanged.

## Goals

- Load any well-formed llm-compressor NVFP4 SafeTensors model where the underlying architecture is text-only Dense or text-only MoE
- Specifically validate against: `RedHatAI/gemma-4-26B-A4B-it-NVFP4`, `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`
- Maintain bit-identical regression for existing modelopt path (validated on `Qwen3-Coder-30B-A3B-FP4`)
- No changes to NVFP4 compute kernels, weight_map.cpp, weight_upload.cu, or executor code

## Non-Goals (Explicit Out-of-Scope)

| Out of Phase 1 | Tracker | Why |
|---|---|---|
| GDN+NVFP4 hybrid models (Qwen3.5/3.6/3.7) | TODO-gdn-nvfp4 | Needs new compute path in GDN scan kernel; ~5-7 days separate work |
| Multimodal/vision support (Gemma-4-Vision, Qwen3-VL) | TODO-multimodal-nvfp4 | Vision encoder is its own subsystem; ~2-3 weeks |
| Performance optimization of NVFP4-MoE decode path (`convert_scales_sfatom` fusion, SFA pointer cache) | Existing perf-investigation | Loader correctness first; perf is separate spec |
| Other llm-compressor schemes (MXFP4, FP8, INT8, GPTQ) | Future scope | NVFP4 first; prove the pattern, then extend |
| Self-quantization tooling | Out of imp's scope | Users download pre-quantized models |

## Architecture

### Format Difference Map (modelopt vs llm-compressor)

The on-disk byte layout is **identical**: packed FP4 weights (uint8, 2 nibbles per byte), F8_E4M3 per-block micro-scales, FP32 per-tensor scalars. Only naming and config metadata differ.

| Role | modelopt name | llm-compressor name |
|---|---|---|
| Packed FP4 weights | `*.weight` | `*.weight_packed` |
| F8_E4M3 per-16-block micro-scale | `*.weight_scale` | `*.weight_scale` (identical) |
| FP32 per-tensor weight scalar | `*.weight_scale_2` | `*.weight_global_scale` |
| FP32 per-tensor activation scalar | `*.input_scale` | `*.input_global_scale` |
| Path prefix (multimodal models) | `model.layers.*` | `model.language_model.layers.*` |
| Quantization config file | `hf_quant_config.json` | `recipe.yaml` |
| Vision tower tensors | not present | `model.vision_tower.*` (present, BF16, must be skipped) |

### Component Map

```
NEW   src/model/llm_compressor_loader.h       (~40 LOC public API)
NEW   src/model/llm_compressor_loader.cpp     (~150 LOC parser + translation table)
EDIT  src/model/safetensors_loader.cpp        (~30 LOC: hook into translation at enumerate-tensors)
EDIT  src/model/hf_config_loader.cpp          (~20 LOC: dispatch to recipe.yaml parser when modelopt config absent)
EDIT  src/model/hf_config_loader.h            (~5 LOC: add format enum to NvFP4Config)
NEW   tests/test_llm_compressor_loader.cpp    (~120 LOC unit tests)
NEW   tests/test_e2e_llm_compressor.cpp       (~80 LOC end-to-end load tests)
```

**Unchanged** (intentional scope boundary):
- `src/model/weight_map.cpp` — sees only post-translation modelopt-style names
- `src/quant/nvfp4_quant.cu`, `nvfp4_gemm.cu` — kernels are format-agnostic
- `src/model/weight_upload.cu` — upload is format-agnostic
- `src/graph/executor_*.cu` — dispatch is format-agnostic

### Data Flow

```
SafeTensors directory on disk
        ↓
safetensors_loader.cpp::enumerate_tensors()
        ↓
[NEW]  llm_compressor_loader::translate_name(orig)
        → returns {EMIT, renamed} or {SKIP, ""}
        → emits one-time INFO/WARN log when prefix stripped or skip applied
        ↓
weight_map.cpp::map_to_role(translated_name)   ← unchanged
        ↓
NvFP4PreQuantWeight struct populated         ← unchanged
        ↓
weight_upload.cu uploads compressed bytes + scales to GPU   ← unchanged
        ↓
nvfp4_quant.cu / nvfp4_gemm.cu compute path   ← unchanged
```

## Detailed Design

### 1. Translation Layer (`llm_compressor_loader.cpp`)

Single function with deterministic static lookup:

```cpp
namespace imp::llm_compressor {

struct NameTranslation {
    enum Action { EMIT, SKIP };
    Action action;
    std::string out_name;  // populated when action == EMIT
};

// Pure function: deterministic, side-effect free apart from counter increments
NameTranslation translate_name(const std::string& in,
                               TranslationCounters& counters);

// Aggregate counters for end-of-load summary log
struct TranslationCounters {
    int suffix_renames = 0;
    int prefix_strips = 0;
    int vision_skipped = 0;
    int gemma4_extra_skipped = 0;
    int passed_through = 0;
};

void log_summary(const TranslationCounters& c);

} // namespace imp::llm_compressor
```

Translation rules applied in order:

**Step 1 — Suffix rename** (replace last suffix if it matches):

| Input ends in | Replace with |
|---|---|
| `.weight_packed` | `.weight` |
| `.weight_global_scale` | `.weight_scale_2` |
| `.input_global_scale` | `.input_scale` |

**Step 2 — Prefix strip** (one rule):

| Input starts with | Action |
|---|---|
| `model.language_model.` | strip to `model.` (counter+=1; log INFO once at end) |

**Step 3 — Skip patterns** (return SKIP):

| Match | Counter | Log behavior |
|---|---|---|
| starts with `model.vision_tower.` | `vision_skipped` | INFO once: "skipped N vision_tower tensors (text-only mode)" |
| starts with `model.visual.` | `vision_skipped` | same (Qwen3-VL naming) |
| ends with `.layer_scalar` or `.per_expert_scale` (Gemma-4 extras) | `gemma4_extra_skipped` | WARN once at end: "skipped N Gemma-4 extra scaling tensors — output quality may be reduced" |
| ends with `.scale` AND not preceded by a recognized projection name (e.g. `q_proj.scale`, `gate_proj.scale` would NOT skip) — implementation must verify the exact path pattern from real Gemma-4 file before applying | `gemma4_extra_skipped` | same WARN |

**Step 4 — Pass through** (no match):

Return `{EMIT, in}` unchanged. Counter `passed_through++`. This handles standard non-quantized tensors (token_embd, layernorms, etc.) which use the same names in both formats.

**End-of-load summary log** (from `safetensors_loader` after `enumerate_tensors` finishes):

```
[INFO] llm-compressor format: 11725 tensors translated (weight_packed → weight),
       11725 scales remapped, 90 Gemma-4 extras skipped, 656 vision tensors skipped,
       multimodal prefix stripped on 12325 tensors.
```

### 2. Recipe.yaml Parser

llm-compressor recipes are intentionally simple and follow a documented structure. We parse only the subset we need without adding a YAML library dependency.

Required structure:
```yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [literal_name, 're:regex_pattern']
      scheme: NVFP4              # or NVFP4_W4A16
      bypass_divisibility_checks: false
```

We extract:
- `scheme` — must equal `NVFP4` or `NVFP4_W4A16`. Anything else → return `false` (decline)
- `ignore` — list of strings; populate `cfg.exclude_modules`
- `group_size` — optional, default 16

Implementation: ~50 LOC line-based parser, looking for indented keys. Bracket-array `[a, b, c]` and block-array (dash-prefixed lines) both supported. On structural mismatch, log clear error: `recipe.yaml does not match expected QuantizationModifier structure; please report this model`.

### 3. Format Detection (`hf_config_loader.cpp`)

Extend `NvFP4Config` struct with a format enum:

```cpp
enum class NvFP4Format { MODELOPT, LLM_COMPRESSOR };
struct NvFP4Config {
    int group_size = 16;
    std::string kv_cache_quant_algo;
    std::vector<std::string> exclude_modules;
    NvFP4Format format = NvFP4Format::MODELOPT;  // new
};
```

Detection logic invoked from `safetensors_loader` init:

```cpp
NvFP4Format detect_nvfp4_format(const std::string& dir) {
    bool has_modelopt = file_exists(dir + "/hf_quant_config.json");
    bool has_compressor = file_exists(dir + "/recipe.yaml");

    if (has_modelopt && !has_compressor) return MODELOPT;
    if (has_compressor && !has_modelopt) return LLM_COMPRESSOR;
    if (has_modelopt && has_compressor) {
        IMP_LOG_WARN("Both quant config files present in %s — preferring modelopt",
                     dir.c_str());
        return MODELOPT;
    }
    // Neither file present — probe first quantized tensor name
    return probe_first_tensor_name(dir);
}

// Reads safetensors header of first model-*.safetensors file, looks for any
// tensor name ending in .weight_packed (LLM_COMPRESSOR) or .weight_scale_2
// (MODELOPT). If neither present, returns MODELOPT (no NVFP4 detected anyway).
NvFP4Format probe_first_tensor_name(const std::string& dir);
```

When format == LLM_COMPRESSOR but no recipe.yaml present, log WARN and use defaults: `group_size=16, exclude_modules empty`.

### 4. Integration Points

`safetensors_loader.cpp::enumerate_tensors`:

```cpp
TranslationCounters counters{};
for (auto& [name, info] : header_tensors) {
    if (config.nvfp4 && config.nvfp4->format == LLM_COMPRESSOR) {
        auto t = llm_compressor::translate_name(name, counters);
        if (t.action == NameTranslation::SKIP) continue;
        emit_tensor(t.out_name, info);
    } else {
        emit_tensor(name, info);  // existing modelopt / non-NVFP4 path
    }
}
if (config.nvfp4 && config.nvfp4->format == LLM_COMPRESSOR) {
    llm_compressor::log_summary(counters);
}
```

`hf_config_loader.cpp::load_nvfp4_config`:

```cpp
bool HFConfigLoader::load_nvfp4_config(const std::string& model_dir, NvFP4Config& cfg) {
    cfg.format = detect_nvfp4_format(model_dir);
    switch (cfg.format) {
        case NvFP4Format::MODELOPT:
            return parse_hf_quant_config(model_dir, cfg);  // existing
        case NvFP4Format::LLM_COMPRESSOR:
            return parse_llm_compressor_recipe(model_dir, cfg);  // new
    }
}
```

## Error Handling

| Error scenario | Behavior |
|---|---|
| recipe.yaml present but malformed | ERROR: clear message naming the structural problem; load fails |
| recipe.yaml has `scheme: W8A8` (non-NVFP4) | ERROR: "this NVFP4 loader does not support scheme=W8A8; use a different model"; load fails |
| LLM_COMPRESSOR format detected but mandatory tensors missing post-translation | ERROR: weight_map flags missing tensors; load fails with clear list |
| Both config files present | WARN once, prefer modelopt (deterministic) |
| Vision tower / Gemma-4 extras encountered | WARN/INFO once (counters in summary); skip silently after that, load proceeds |
| Unknown tensor name (no rule matches) | Pass through unchanged; weight_map handles or ignores per existing rules |

## Validation

### Unit Tests (`tests/test_llm_compressor_loader.cpp`, CPU-only)

| Test | Verifies |
|---|---|
| `TranslateName_PackedWeight` | `.weight_packed` → `.weight` |
| `TranslateName_GlobalScales` | both global scales remapped |
| `TranslateName_MultimodalPrefix` | `model.language_model.X` → `model.X` |
| `TranslateName_VisionTowerSkip` | `model.vision_tower.X` → SKIP |
| `TranslateName_Gemma4Extras_Skip` | `*.layer_scalar`, `*.per_expert_scale`, `*.scale` → SKIP |
| `TranslateName_PassThrough` | unknown patterns unchanged |
| `RecipeYaml_ParseGemma4` | parses real Gemma-4 recipe correctly |
| `RecipeYaml_ParseQwen36` | parses real Qwen3.6 recipe (more `re:` patterns) |
| `RecipeYaml_RejectsNonNVFP4` | `scheme: W8A8` returns false |
| `FormatDetection_BothFiles` | MODELOPT preferred + WARN logged |
| `FormatDetection_NeitherFile_Probe` | tensor-name probe falls back correctly |
| `LoaderCounters_Summary` | summary log content matches counter state |

### End-to-End Tests (`tests/test_e2e_llm_compressor.cpp`, GPU + models required)

| Test | Model | Verification |
|---|---|---|
| `Gemma4_Loads` | Gemma-4-26B-A4B-it-NVFP4 | loads without IMA, generates ≥ 8 tokens |
| `Gemma4_Coherent` | same | greedy "What is 2+2?" output contains "4" |
| `Gemma4_LayerCount` | same | loaded layer count == config.num_hidden_layers |
| `MistralSmall_Loads` | Mistral-Small-3.2-24B-NVFP4 | loads, generates tokens |
| `MistralSmall_Coherent` | same | greedy "Capital of France?" contains "Paris" |
| `Modelopt_Regression_Coder30B` | Qwen3-Coder-30B-A3B-FP4 | existing path still works (decode > 30 tok/s) |

Tests gate on `IMP_LLM_COMPRESSOR_TEST_DIR` env var; auto-skip when models absent.

### Manual Performance Validation (not in CI)

A/B benchmark for documentation purposes:
- Gemma-4-26B-A4B-it-Q4_K_M GGUF (current) vs Gemma-4-26B-A4B-it-NVFP4 (new path)
- Measure both pp512 and tg256
- Document baseline numbers in `docs/comparison-llama-cpp.md` for future reference

Expected: NVFP4 prefill significantly faster (CUTLASS direct vs dequant detour); decode unclear (may hit `convert_scales_sfatom` MoE bottleneck — separate optimization tracked elsewhere).

## Open Risks

### R1: Gemma-4 extra scaling tensors

Gemma-4-NVFP4 includes 90 extra tensors (`layer_scalar`, `per_expert_scale`, `scale` — 30 each across 30 MoE layers) absent from modelopt-format models. RedHatAI's reported GSM8K recovery (95.6%) may depend on these scales being applied.

**Mitigation plan**:
1. Phase 1 ships with SKIP + WARN
2. Validation includes greedy-coherence smoke tests (e.g., "What is 2+2?" → "4")
3. If validation passes → ship
4. If quality is visibly degenerate → block on Phase 2 (separate spec for custom Gemma-4 multiplier kernel that applies these scales)

**Decision**: ship Phase 1 with skip-and-warn. Quality concerns gate on smoke-test results, not on hypothetical loss.

### R2: Recipe.yaml parser robustness

Mini-parser covers ~99% of real llm-compressor recipes. Multi-stage recipes, custom modifiers, or exotic YAML constructs not supported.

**Mitigation**: Explicit error message with "report this recipe.yaml as an issue" guidance. No silent fallback.

### R3: W4A4 vs W4A16 detection

llm-compressor produces both:
- W4A4 (default `scheme: NVFP4`) — has `input_global_scale` per layer
- W4A16 (`scheme: NVFP4_W4A16`) — no `input_global_scale`, uses FP16 activations

imp's NVFP4 path today expects activation quantization (matches W4A4). For W4A16 models, we need to dispatch through an FP16-activation path.

**Mitigation**: Phase 1 detects W4A16 via missing `input_global_scale` tensors and either (a) falls back to FP16-activation path if it exists in current code, or (b) declines load with clear "W4A16 not yet supported, use W4A4 variant" error. Decision deferred to implementation: check whether the FP16-activation NVFP4 path already exists.

## TODO Backlog (out-of-scope for this spec)

| Item | Trigger to start |
|---|---|
| **GDN+NVFP4 integration** for Qwen3.5/3.6/3.7 | User wants to load Qwen3.6-NVFP4 |
| **Multimodal/vision support** for Gemma-4-Vision, Qwen3-VL | Vision use-case exists |
| **`convert_scales_sfatom` fusion + SFA pointer cache** (NVFP4-MoE decode perf) | NVFP4-MoE decode performance critical |
| **W4A16 first-class support** (if not handled in R3 fallback) | User-targeted W4A16 model |
| **Custom Gemma-4 layer_scalar / per_expert_scale kernel** | If R1 quality validation fails |
| **Other llm-compressor schemes** (MXFP4, FP8, INT8, GPTQ) | Coverage need beyond NVFP4 |

## Estimated Effort

- Translation layer + parser + format detection: ~250 LOC, ~1 day
- Integration into safetensors_loader + hf_config_loader: ~50 LOC, ~0.5 day
- Unit tests: ~120 LOC, ~0.5 day
- E2E tests + smoke validation: ~80 LOC + benchmarking, ~0.5 day

Total: **~2-3 days** for a single engineer, including validation runs.
