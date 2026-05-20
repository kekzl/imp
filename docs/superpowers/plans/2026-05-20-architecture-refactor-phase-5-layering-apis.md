# Architecture Refactor Phase 5 — Schichten und APIs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the four highest-risk structural wounds documented in `docs/architecture.md` "Known structural wounds": (1) VRAM ownership scattered across 5 modules, (2) `RuntimeConfig::current()` global singleton with 105 call sites, (3) duplicated public API doors (`imp_generate*` parallel with `imp_prefill+imp_decode_step`), (4) the 1 GiB cuBLAS S-matrix workspace. Plus one small alignment task — `gemma4` config section migrating to per-model overrides.

**Architecture:** Five independent **Tracks**, each runnable on its own merge cadence. Tracks share the Phase-5 umbrella but have minimal cross-coupling — Track A can be done while Track D is in-flight, etc. Each Track is internally a sequence of TU-split or class-introduction PRs in the Phase 1-4 style.

**Tech Stack:** C++20, CUDA 13.2, CMake, Docker build (`make build`), GTest suite (`make verify-fast`).

---

## Reference: Source spec

`docs/superpowers/specs/2026-05-20-architecture-refactor-roadmap-design.md` §3 Phase 5.

## Honest scope sizing

Phase 5 is the most invasive phase of the roadmap. Realistic estimates per Track at single-developer-with-Claude pace:

| Track | Scope | Risk | Estimated effort |
|---|---|---|---|
| **A. `gemma4` → `ModelOverrides`** | 26 refs, 6 files | Low (mechanical) | **~1 day** |
| **B. Public API dedupe** | 4 entry points, 875-LOC `imp_api.cpp` | Medium (ABI-stable surface) | **~2 days** |
| **C. VRAM owner consolidation** | 1218 LOC, 5 modules → 1 `MemoryManager` | Medium (touch every allocator caller) | **~3-5 days** |
| **D. `RuntimeConfig` de-globalize** | 105 call sites, 30 files | Large (per-Engine plumbing through subsystems) | **~5-10 days** |
| **E. Tiled streaming softmax** (soft) | Kernel rewrite + benchmark gates | High (perf-sensitive, kernel correctness) | **~10-15 days** |

**Recommended order:** A → B → C → D → E. Easiest and most independent first; D has the largest blast radius and benefits from C landing first (the new `MemoryManager` is one fewer global to wire through); E is opt-in soft.

This plan documents all five Tracks. Pick a Track, execute its tasks, land its PRs, move to the next. There is no requirement that all five Tracks land in one session.

---

## Reference: Pre-flight inventory

| Subsystem | Files | LOC | Call sites |
|---|---|---|---|
| VRAM owner | `src/memory/{vram_allocator,device_allocator,pinned_allocator}.{cu,cpp,h}`, `src/runtime/{vram_budget,storage_planner}.{cpp,h}` | 1218 (10 files) | ~30 |
| `RuntimeConfig::current()` | `src/runtime/config.{h,cpp}` + 30 reader files | 324 + spread | 105 calls |
| Public API | `include/imp/imp.h`, `src/api/imp_api.cpp` | 142 + 875 | 4 entry pairs |
| S-matrix wound | `src/exec/executor_workspace_buffers.cu:200`, `src/compute/attention_cublas.cu` | 1 GiB device buf + cuBLAS path | 1 default-path consumer |
| `gemma4` config | `src/runtime/config.h:185-193`, 6 caller files | 8 fields | 26 refs |

---

## Track A — `gemma4` → `ModelOverrides`

**Goal:** Move the model-specific `gemma4` section out of the global `RuntimeConfig` into a per-model `ModelConfig::overrides` (or `ModelOverrides`) structure. Solves the "spec smell" that a model name appears in a runtime singleton.

**Files:**
- Modify: `src/runtime/config.h` — remove `Gemma4` section
- Modify: `src/runtime/config.cpp` — remove `gemma4.*` parser entries
- Modify: `src/model/model_config.h` — add `ModelOverrides` struct (or extend existing `ModelConfig`)
- Modify: `src/model/gguf_loader.cpp` — populate `ModelOverrides` from GGUF metadata when arch=gemma4
- Modify: 6 caller files to read from `model_.config.overrides.*` instead of `RuntimeConfig::current().gemma4.*`
- Modify: `imp.conf.example` — drop `[gemma4]` section
- Modify: `docs/architecture.md` — remove the gemma4-in-singleton smell mention

### Task A.1: Pre-flight inventory

- [ ] **Step 1: Confirm 26 references across 6 files**

```bash
grep -rn 'gemma4\.\|\.gemma4\b' src/
```

Expected files: `src/exec/executor_forward_moe.cu`, `src/exec/executor_attention.cu`, `src/exec/gemm_kernel_gguf.cu`, `src/model/gguf_loader.cpp`, `src/runtime/engine_init_resolver.cpp`, `src/runtime/config.cpp`.

- [ ] **Step 2: List the 8 gemma4 fields**

```bash
sed -n '/^    struct Gemma4 {$/,/^    } gemma4;$/p' src/runtime/config.h
```

Fields (verify): `fp32_gemm_out`, `no_graphs`, `force_mmvq`, `fp32_expert_down`, `no_decode_fast`, `no_post_ffw_1`, `ggml_prefill`, (and one more — confirm from the file).

### Task A.2: Add `ModelOverrides` to `ModelConfig`

- [ ] **Step 1: Edit `src/model/model_config.h`**

Add inside `ModelConfig`:

```cpp
struct Overrides {
    // Gemma-4-specific runtime overrides. Populated from GGUF metadata
    // when arch == ModelArch::GEMMA4; otherwise default-constructed.
    // Previously lived in RuntimeConfig::current().gemma4; moved to
    // ModelConfig in Phase 5 of the architecture refactor.
    struct Gemma4 {
        bool fp32_gemm_out = false;
        bool no_graphs = false;
        bool force_mmvq = false;
        bool fp32_expert_down = false;
        bool no_decode_fast = false;
        bool no_post_ffw_1 = false;
        bool ggml_prefill = false;
    } gemma4;
};

Overrides overrides;
```

Adjust field list to match Step 2 of Task A.1 exactly.

### Task A.3: Migrate callers

- [ ] For each of the 6 caller files: replace `RuntimeConfig::current().gemma4.<field>` with `model_.config.overrides.gemma4.<field>` (or equivalent path — verify by reading each call site).
- [ ] In `src/runtime/engine_init_resolver.cpp`, the seeding from `RuntimeConfig` into `EngineConfig::gemma4` is removed; the Gemma-4 overrides now live in `ModelConfig`.
- [ ] In `src/model/gguf_loader.cpp`, populate `model.config.overrides.gemma4` from GGUF metadata key-value pairs (the existing seeding logic moves here).

### Task A.4: Remove `Gemma4` from `RuntimeConfig`

- [ ] Edit `src/runtime/config.h`: delete the `struct Gemma4 { ... } gemma4;` block.
- [ ] Edit `src/runtime/config.cpp`: delete all `gemma4.*` parser branches (lines ~209-220).
- [ ] Edit `imp.conf.example`: delete the `[gemma4]` section.

### Task A.5: Verify

- [ ] `grep -rn 'RuntimeConfig::current()\.gemma4\|gemma4\.fp32\|gemma4\.no_graphs\|gemma4\.force_mmvq\|gemma4\.fp32_expert_down\|gemma4\.no_decode_fast\|gemma4\.no_post_ffw_1\|gemma4\.ggml_prefill' src/ tests/ tools/` returns zero.
- [ ] `make build` clean.
- [ ] `make verify-fast` green.
- [ ] Commit with subject `refactor(config): move gemma4 section to ModelConfig::overrides`.

**Track A complete. ~1 day.**

---

## Track B — Public API dedupe

**Goal:** `imp_generate` and `imp_generate_streaming` become thin wrappers around `imp_prefill_with_params` + a `imp_decode_step` loop, eliminating the parallel implementation that today must be kept in sync.

**Public surface impact:** ABI-stable. Symbols stay; semantics stay; implementation collapses.

**Files:**
- Modify: `src/api/imp_api.cpp` — rewrite `imp_generate*` to call `imp_prefill_with_params` + `imp_decode_step` loop
- Modify: `include/imp/imp.h` — update docstrings only (no signature changes)
- Add tests if missing: parity test that `imp_generate` and the manual `prefill+decode` loop produce identical output for the same seed.

### Task B.1: Map the duplication

- [ ] **Step 1: Read the current implementations**

```bash
grep -n 'imp_generate\|imp_generate_streaming\|imp_prefill_with_params\|imp_decode_step' src/api/imp_api.cpp
```

For each function, note its line range and what it does.

- [ ] **Step 2: Identify the duplicated logic**

Both paths share: sampling params translation, EOS detection, stop strings, max_tokens, MTP draft acceptance, repeat-penalty buffers, think-token tracking. Note which functions in `engine_sampling_stop.cpp` / `engine_scheduler.cpp` each path calls.

### Task B.2: Refactor `imp_generate` to wrap `imp_prefill+imp_decode_step`

- [ ] **Step 1: Write the new `imp_generate` implementation**

```cpp
ImpError imp_generate(ImpContext ctx, const char* prompt, const ImpGenerateParams* params,
                     char* output_buf, int output_buf_size, int* out_n_tokens) {
    if (!ctx || !prompt || !output_buf || !out_n_tokens) return IMP_ERROR_INVALID_ARG;

    // Tokenize
    std::vector<int32_t> tokens;
    if (auto e = tokenize_prompt(ctx, prompt, tokens); e != IMP_OK) return e;

    // Prefill (delegates to imp_prefill_with_params)
    if (auto e = imp_prefill_with_params(ctx, tokens.data(),
                                          static_cast<int>(tokens.size()),
                                          params); e != IMP_OK) return e;

    // Decode loop (delegates to imp_decode_step)
    std::vector<int32_t> output;
    int max_tokens = params ? params->max_tokens : kDefaultMaxTokens;
    for (int i = 0; i < max_tokens; ++i) {
        int32_t tok;
        auto e = imp_decode_step(ctx, params, &tok);
        if (e != IMP_OK) return e;
        if (is_eos(ctx, tok)) break;
        output.push_back(tok);
    }

    // Detokenize
    return detokenize_to_buf(ctx, output, output_buf, output_buf_size, out_n_tokens);
}
```

Adjust signatures to actual `imp.h` types.

- [ ] **Step 2: Same for `imp_generate_streaming`** — the difference is the per-token callback after `imp_decode_step` returns.

### Task B.3: Add parity test

- [ ] **Step 1: Create `tests/test_api_generate_parity.cpp`**

```cpp
TEST(ApiGenerateParity, GenerateMatchesManualPrefillDecode) {
    // Same seed, same params, same prompt → identical output via both paths.
    const char* prompt = "Hello, world.";
    ImpGenerateParams params = imp_generate_params_default();
    params.seed = 42;
    params.max_tokens = 16;

    // Path 1: imp_generate
    char buf1[1024];
    int n1 = 0;
    ASSERT_EQ(imp_generate(ctx, prompt, &params, buf1, sizeof(buf1), &n1), IMP_OK);

    // Reset context
    ASSERT_EQ(imp_context_reset(ctx), IMP_OK);

    // Path 2: manual prefill + decode loop
    std::vector<int32_t> tokens(256);
    int n_tokens = 0;
    ASSERT_EQ(imp_tokenize(model, prompt, tokens.data(), &n_tokens, tokens.size()), IMP_OK);
    ASSERT_EQ(imp_prefill_with_params(ctx, tokens.data(), n_tokens, &params), IMP_OK);

    std::vector<int32_t> out_tokens;
    for (int i = 0; i < params.max_tokens; ++i) {
        int32_t tok;
        ASSERT_EQ(imp_decode_step(ctx, &params, &tok), IMP_OK);
        if (is_eos(tok)) break;
        out_tokens.push_back(tok);
    }
    char buf2[1024];
    int detok_n = 0;
    ASSERT_EQ(imp_detokenize(model, out_tokens.data(), out_tokens.size(), buf2, sizeof(buf2), &detok_n), IMP_OK);

    ASSERT_STREQ(buf1, buf2);
}
```

### Task B.4: Verify

- [ ] `make build` clean.
- [ ] `make verify-fast` green; new parity test passes.
- [ ] `wc -l src/api/imp_api.cpp` drops by ~150-200 LOC (the duplicated decode loop is gone).
- [ ] Commit.

**Track B complete. ~2 days.**

---

## Track C — VRAM owner consolidation

**Goal:** Consolidate 5 module pairs into one `MemoryManager` class. Today's split:

| File | LOC | Concern |
|---|---|---|
| `src/memory/vram_allocator.{cu,h}` | 233 | Device VRAM allocation (`vram_alloc`, `vram_free`) |
| `src/memory/device_allocator.{cu,h}` | 206 | Lower-level cudaMalloc wrappers + pool |
| `src/memory/pinned_allocator.{cpp,h}` | 261 | Pinned-host allocation |
| `src/runtime/vram_budget.{cpp,h}` | 261 | VRAM budget tracking (free/used/reserved) |
| `src/runtime/storage_planner.{cpp,h}` | 257 | Storage-tier planning (which tensor lives where) |

Five concerns currently spread across `src/memory/` and `src/runtime/`. Phase 5 consolidates into one `MemoryManager` in `src/memory/`.

**Files:**
- Create: `src/memory/memory_manager.{h,cu}` — unified class
- Modify: ~30 call sites across `src/` (verified by grep)
- Eventually delete (after consolidation): the 5 module pairs

### Task C.1: Design the `MemoryManager` interface

- [ ] **Step 1: Read each module's public surface**

```bash
cat src/memory/vram_allocator.h
cat src/memory/device_allocator.h
cat src/memory/pinned_allocator.h
cat src/runtime/vram_budget.h
cat src/runtime/storage_planner.h
```

Map the public API of each. The union becomes the `MemoryManager` interface.

- [ ] **Step 2: Draft the consolidated interface**

```cpp
class MemoryManager {
public:
    // Construction — one per Engine.
    MemoryManager();
    ~MemoryManager();

    // Device VRAM (formerly vram_allocator + device_allocator)
    void* alloc_device(size_t bytes, const char* tag);
    void  free_device(void* ptr);
    size_t free_vram() const;
    size_t used_vram() const;

    // Pinned host (formerly pinned_allocator)
    void* alloc_pinned(size_t bytes);
    void  free_pinned(void* ptr);

    // Budget (formerly vram_budget)
    VRAMBudget compute_budget(size_t weight_bytes, /* ... */) const;
    void reserve(size_t bytes, const char* tag);

    // Storage planning (formerly storage_planner)
    StoragePlan plan_storage(const Model& model, /* ... */) const;

private:
    // Aggregated state from the 5 modules.
};
```

Refine the interface based on actual call patterns.

### Task C.2: Migrate one module at a time

Each module gets its own PR. Order: smallest → largest. Pattern per module:

- [ ] Move source into `memory_manager.{h,cu}` as private methods or aggregated state.
- [ ] Replace each call site with the equivalent `mem_mgr_.<method>()` call (or `engine.memory_manager().<method>()`).
- [ ] Delete the source module after the last call site is migrated.
- [ ] Build + tests green per migration.

Suggested order: `vram_budget` → `storage_planner` → `pinned_allocator` → `device_allocator` → `vram_allocator`.

### Task C.3: Verify

- [ ] All 5 source modules deleted.
- [ ] `MemoryManager` covers all use cases.
- [ ] `grep -rn 'vram_alloc(\|device_alloc(\|pinned_alloc(\|compute_vram_budget(\|plan_storage(' src/` returns only matches via `mem_mgr_` accessors.
- [ ] `make verify` (full) green.
- [ ] Each migration commit is one concern.

**Track C complete. ~3-5 days, ~5-7 PRs.**

---

## Track D — `RuntimeConfig` de-globalize

**Goal:** Eliminate `RuntimeConfig::current()` singleton. Each `Engine` owns its own `RuntimeConfig` instance and passes it (by `const&`) into subsystems.

**Scale:** 105 call sites across 30 files. This is the largest single refactor in the entire roadmap.

**Files (per Phase-4 split, the readers are organized by subsystem):**
- `src/runtime/engine.cpp` (10 calls)
- `src/runtime/engine_init_resolver.cpp` (15)
- `src/exec/executor_forward.cu` (9), `executor_forward_moe.cu` (14), `executor_attention.cu` (8), `executor_ssm_gdn.cu` (6), `executor_workspace_buffers.cu` (3), `executor_pre_dequant.cu` (3)
- `src/model/chat_template.cpp` (4), `model/weight_upload.cu` (3)
- Plus ~20 other files with 1-2 calls each

### Task D.1: Add per-Engine `RuntimeConfig` member

- [ ] **Step 1: Edit `src/runtime/engine.h`**

Add to `Engine`'s private section:

```cpp
private:
    RuntimeConfig runtime_config_;
    // ...

public:
    const RuntimeConfig& runtime_config() const { return runtime_config_; }
```

- [ ] **Step 2: Edit `src/runtime/engine.cpp::Engine::init`**

Replace the `RuntimeConfig::current()` access with `runtime_config_ = RuntimeConfig::load(...)`. The singleton is no longer the source of truth.

- [ ] **Step 3: Keep `RuntimeConfig::current()` working** — it becomes a thin wrapper returning the last-created Engine's config, or a debug-warning fallback. Provides backwards-compat during the migration.

### Task D.2: Migrate readers, subsystem by subsystem

Each subsystem gets its own PR. Pattern:

- [ ] Add a `const RuntimeConfig& cfg` parameter (or take it from `engine_.runtime_config()`).
- [ ] Replace each `RuntimeConfig::current()` call site with the param/member access.
- [ ] Build + tests green per subsystem.

Suggested order (by call-site count, smallest first):
1. `weight_upload.cu` (3) + `workspace_buffers.cu` (3) + `pre_dequant.cu` (3) + `chat_template.cpp` (4) — small TUs
2. `executor_ssm_gdn.cu` (6) + `executor_attention.cu` (8) + `executor_forward.cu` (9)
3. `executor_forward_moe.cu` (14) + `engine_init_resolver.cpp` (15)
4. `engine.cpp` (10)
5. The remaining ~20 files

### Task D.3: Remove `RuntimeConfig::current()`

- [ ] After all call sites are migrated, delete the singleton accessor from `src/runtime/config.{h,cpp}`.
- [ ] Grep verifies zero remaining references.
- [ ] `make verify` (full) green.

**Track D complete. ~5-10 days, ~10-15 PRs.**

---

## Track E (SOFT — may slip indefinitely) — Tiled streaming softmax

**Goal:** Replace the ~1 GiB cuBLAS S-matrix workspace with a tiled streaming softmax implementation. Removes the static ceiling on max context length and frees ~1 GiB of VRAM for KV cache.

**Why soft:** Perf-sensitive kernel rewrite. The cuBLAS S-matrix path is the DEFAULT prefill attention path post-Phase-2; any regression hits every Qwen3/Gemma-4 user.

**Two paths the spec mentions:**

(a) Tiled streaming softmax IN the cuBLAS path — keep cuBLAS QK^T and PV, but tile the softmax stage so only one row-block of S exists at a time.

(b) FMHA default-switch — after Phase 2's dispatcher simplification, evaluate whether the single remaining FMHA variant is perf-competitive enough to become the default, demoting cuBLAS-attention to fallback.

### Task E.1: Decision — path (a) vs (b)

- [ ] **Step 1: Benchmark current cuBLAS vs FMHA**

```bash
scripts/gen_perf_baseline.sh
# Then explicitly force FMHA-only via attention.force_cublas_attn=false
# and re-run for the same model set.
```

- [ ] **Step 2: Decide** based on the perf gap. Document in a memo (`docs/plans/<date>-tiled-softmax-vs-fmha-decision.md`).

### Task E.2 (path a — tiled softmax in cuBLAS)

- [ ] Design tile geometry: row-block size, column-block size, register pressure.
- [ ] Implement `tiled_causal_softmax_kernel<>` in `src/compute/`.
- [ ] Wire into `attention_cublas_prefill` to call the tiled kernel instead of the monolithic `causal_softmax`.
- [ ] Remove the `attn_scores_buf_` allocation from `executor_workspace_buffers.cu:200`.
- [ ] Benchmark against the pre-change baseline; documented perf delta in commit body.

### Task E.2 (path b — FMHA default-switch)

- [ ] Flip `force_cublas_attn` semantics in `executor_attention.cu` — FMHA becomes default.
- [ ] Remove the `attn_scores_buf_` allocation.
- [ ] Add a Gemma-4 hd=512 special case if FMHA can't handle it (use cuBLAS as fallback).
- [ ] Benchmark.

### Task E.3: Update wound documentation

- [ ] Edit `docs/architecture.md` to remove the "1 GiB S-matrix" wound.
- [ ] Add a closing memo describing what shipped + perf numbers.

**Track E complete (if pursued). ~10-15 days.**

---

## Phase 5 closeout

After Tracks A-D land (Track E is soft and may slip):

- [ ] **Step 1: Full verification suite**

```bash
make verify
```

- [ ] **Step 2: Final perf snapshot**

```bash
scripts/gen_perf_baseline.sh
git diff tests/perf_baseline.json
```

Tracks A-D are structural — no perf change expected. Document any surprise.

- [ ] **Step 3: Write MEMORY.md entry**

Write `architecture_refactor_phase_5_closed_2026_MM_DD.md` in `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/`. Update the Architecture section of MEMORY.md to point at the new memo.

- [ ] **Step 4: Update `docs/architecture.md`**

Three of the four "Known structural wounds" get rewritten or removed:
- `engine.cpp` 3112 LOC → done in Phase 4 (already updated)
- 1 GiB S-matrix → either resolved by Track E or marked "deferred, see roadmap"
- `RuntimeConfig::current()` singleton → resolved by Track D
- `src/exec/` directory was previously `src/graph/` → keep as historical note

- [ ] **Step 5: Mark Phase 5 closed in roadmap spec**

Add Status line at top of Phase 5 section with PR numbers per Track.

- [ ] **Step 6: Mark the entire roadmap as closed**

The roadmap spec gets a final "**All five phases closed 2026-MM-DD**" header banner at the top of §3.

- [ ] **Step 7: Commit closeout on `docs/phase-5-closeout` branch + PR.**

**Phase 5 closes the architecture refactor roadmap.** No further phases planned.
