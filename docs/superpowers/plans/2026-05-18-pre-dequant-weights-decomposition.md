# `pre_dequant_weights` Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the 2415-LOC `GraphExecutor::pre_dequant_weights()` god-function into 7 phase-extracted private methods, preserving exact behavior.

**Architecture:** Replace one mega-function with a thin orchestrator that calls 7 named phase methods. Each phase already has a `// --- Phase N: ... ---` section comment, so boundaries are unambiguous. Cross-phase mutable state (`remaining_budget`) flows by reference; per-phase locals (lambdas like `cache_weight`, `register_prequant`) move into their owning method. No behavior change — the only test is "existing test suite + smoke prompt produces identical results."

**Tech Stack:** C++20, CUDA, existing `GraphExecutor` class in `src/graph/executor.h` + `src/graph/executor_pre_dequant.cu`. No new files; just method extractions within the existing TU.

---

## Phase boundaries (verified by depth-1 brace scan of current `main`, 2026-05-18)

| Phase | Lines | Approx LOC | Contents |
|---|---|---|---|
| Entry + StoragePlanner diagnostic | 175–209 | 35 | Function header, budget compute, Phase 4.2 plan_storage() diagnostic |
| **Phase 0** | 211–436 | 226 | NVFP4 prequant promote to Tensor sidecars (inside `if (cfg.is_nvfp4_prequant)`) |
| **Phase 0b** | 438–538 | 101 | Register prequant-promoted NVFP4 in CUTLASS cache (same `if`) |
| **Phase 1** | 539–659 | 121 | FP16 weight cache + fused KV + fused gate+up (with FP8/NVFP4-decode skip-guards) |
| **Phase 2** | 660–806 | 147 | FP8 cache for uncached weights |
| **Phase 3** | 807–2070 | 1264 | NVFP4 decode cache + Phase 3b CUTLASS NVFP4 + Phase 3c-native MXFP4 GGUF |
| **Phase 3c** standalone | 2071–2218 | 148 | Native MXFP4 GGUF when NVFP4 decode disabled |
| **Phase 4 tail** | 2219–2589 | 371 | Tensor registry, overlay diagnostic, NVFP4 device-args caching |

**Cross-phase mutable state:**
- `size_t remaining_budget` — reduced by Phase 1, 2, 3 (each phase deducts what it cached). Pass by `&` to each phase method.
- `const ModelConfig& cfg` — pass by `const&`.
- `cudaStream_t stream` — pass through.
- `total_cache_bytes`, `cached_count`, `budget_exhausted` — Phase 1 local only; stay private to extracted method.

**Cross-phase reads (member access via `this->`):**
- `wcache_.fp16`, `.fp8`, `.nvfp4`, `.cutlass_nvfp4`, `.cutlass_mxfp4`, `.fused_kv`, `.fused_gate_up`, `.use_fp8`, `.dual_path_quant`, `.nvfp4_bytes`, `.nvfp4_decode_mode`
- `model_->layer(i)`, `model_->out_proj_`, `model_->nvfp4_scratch_`, `model_->output_proj()`
- `hints_`, `qscratch_`, `vram_alloc_`, `gpu_allocations_`

All extracted methods are private members of `GraphExecutor`, so they have full member access — no need to pass `this`-state explicitly.

---

## File structure

**Modify:**
- `src/graph/executor.h:477+` — add 7 private method declarations to `GraphExecutor` class
- `src/graph/executor_pre_dequant.cu` — extract 7 phases out of `pre_dequant_weights`; orchestrator becomes ~50 LOC

**No new files.**

---

## Task 1: Read full function + dep audit

**Files:**
- Read: `src/graph/executor_pre_dequant.cu:175–2589`

- [ ] **Step 1: Read pre_dequant_weights in 5 chunks**

```bash
# Chunks of ~500 lines for context window safety
Read src/graph/executor_pre_dequant.cu offset=175 limit=500   # lines 175–674
Read src/graph/executor_pre_dequant.cu offset=675 limit=500   # lines 675–1174
Read src/graph/executor_pre_dequant.cu offset=1175 limit=500  # lines 1175–1674
Read src/graph/executor_pre_dequant.cu offset=1675 limit=500  # lines 1675–2174
Read src/graph/executor_pre_dequant.cu offset=2175 limit=415  # lines 2175–2589 (function end)
```

- [ ] **Step 2: Confirm cross-phase variables list**

After reading, verify the cross-phase state matches the table above. If a phase consumes a variable defined in a different phase that's not in `remaining_budget`, flag it and revise the method signatures.

- [ ] **Step 3: No commit** (read-only audit)

---

## Task 2: Add private method declarations to executor.h

**Files:**
- Modify: `src/graph/executor.h` — add 7 declarations under the `pre_dequant_weights` public declaration

- [ ] **Step 1: Find the line after `pre_dequant_weights` declaration in executor.h**

Look for the line in `executor.h` matching `pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget)`. The new private declarations go in the `private:` section (search for `private:` near the end of the class).

- [ ] **Step 2: Add 7 private method declarations**

```cpp
private:
    // ... existing private members ...

    // Phases of pre_dequant_weights(), extracted for readability.
    // Cross-phase state: remaining_budget is reduced by each FP16/FP8/NVFP4
    // pass; cfg is const reference to model config.
    void pre_dequant_phase0_promote_nvfp4_sidecars_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase0b_register_cutlass_nvfp4_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase1_fp16_cache_(const ModelConfig& cfg, const VRAMBudget& budget,
                                        size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase2_fp8_cache_(const ModelConfig& cfg,
                                       size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase3_nvfp4_decode_(const ModelConfig& cfg, const VRAMBudget& budget,
                                          size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase3c_standalone_mxfp4_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase4_tensor_registry_(const ModelConfig& cfg, cudaStream_t stream);
```

- [ ] **Step 3: Build to confirm header compiles**

```bash
make build 2>&1 | tail -10
```

Expected: clean build (declarations alone don't change semantics; impl provided per phase below).

- [ ] **Step 4: NO commit yet** (committing header without impl breaks build for anyone else; we'll commit Phase 0 + header together)

---

## Task 3: Extract Phase 0 (NVFP4 prequant promotion)

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:211–436` → moved into `pre_dequant_phase0_promote_nvfp4_sidecars_`

- [ ] **Step 1: Define the new method body**

At end of `src/graph/executor_pre_dequant.cu` (before `}  // namespace imp`), add:

```cpp
void GraphExecutor::pre_dequant_phase0_promote_nvfp4_sidecars_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!cfg.is_nvfp4_prequant)
        return;
    // ... paste lines 226–436 contents here (the body of `if (cfg.is_nvfp4_prequant) { ... }`) ...
}
```

Note: the original `if (cfg.is_nvfp4_prequant)` wrapper at line 226 becomes a guard at the top of the new method, so the body indentation reduces by one level. The inner `for`/`auto`/etc. all stay.

- [ ] **Step 2: Replace lines 211–436 in `pre_dequant_weights` with a single call**

```cpp
    // --- Phase 0: Promote NVFP4 pre-quantized weights to Tensor sidecars ---
    // (extracted, see pre_dequant_phase0_promote_nvfp4_sidecars_)
    pre_dequant_phase0_promote_nvfp4_sidecars_(cfg, stream);
```

- [ ] **Step 3: Build**

```bash
make build 2>&1 | tail -10
```

Expected: clean build. If errors, fix immediately — most likely: missing `model_->` / `wcache_.` qualifier on a member access (no — those work fine in a member method), OR a lambda captured a local that we forgot to move. Read the error, find the missing piece, fix.

- [ ] **Step 4: Smoke test on Qwen3-8B Q8_0**

```bash
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3-8B-Q8_0/ --prompt "Hi" --max-tokens 5 2>&1 | tail -10
```

Expected: 5 tokens generated, no errors, output looks coherent.

- [ ] **Step 5: Smoke test on Mistral-3.2-NVFP4 (Phase 0 exerciser)**

```bash
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Mistral-Small-3.2-24B-Instruct-2506-NVFP4/ \
  --set diagnostics.audit_nvfp4_scales=true \
  --prompt "Hi" --max-tokens 1 2>&1 | grep -E "(NVFP4 audit|NVFP4 prequant)" | head -5
```

Expected: same audit lines we saw in PR #271's session — "NVFP4 audit: input_scale present in 280/280 Linears", "NVFP4 prequant: 280 Linears carry input_scale". If the Phase 0 extraction works, this is byte-identical output.

- [ ] **Step 6: Commit**

```bash
git add src/graph/executor.h src/graph/executor_pre_dequant.cu
git commit -m "refactor(pre-dequant): extract Phase 0 (NVFP4 sidecar promotion)

Move 226 LOC out of pre_dequant_weights() into private method
pre_dequant_phase0_promote_nvfp4_sidecars_. Pure structural — same
member access, same lambdas, same behavior. Verified smoke on
Qwen3-8B Q8_0 + Mistral-3.2-NVFP4 audit log byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Extract Phase 0b (CUTLASS NVFP4 cache register)

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:438–538` → moved into `pre_dequant_phase0b_register_cutlass_nvfp4_`

- [ ] **Step 1: Move the body**

Phase 0b is currently inside the same `if (cfg.is_nvfp4_prequant)` as Phase 0. After Phase 0 extraction (Task 3), Phase 0b is now at the top level of `pre_dequant_weights` but conditionally — wrap it the same way:

```cpp
void GraphExecutor::pre_dequant_phase0b_register_cutlass_nvfp4_(
    const ModelConfig& cfg, cudaStream_t stream) {
    if (!cfg.is_nvfp4_prequant)
        return;
    // ... paste original lines 438–538 body ...
}
```

- [ ] **Step 2: Replace call site**

```cpp
    pre_dequant_phase0b_register_cutlass_nvfp4_(cfg, stream);
```

- [ ] **Step 3: Build**

```bash
make build 2>&1 | tail -10
```

- [ ] **Step 4: Smoke test (Q8_0 + NVFP4)**

```bash
# Same as Task 3 Step 4 and Step 5.
```

- [ ] **Step 5: Commit**

```bash
git add src/graph/executor_pre_dequant.cu
git commit -m "refactor(pre-dequant): extract Phase 0b (CUTLASS NVFP4 cache register)

Move 101 LOC out of pre_dequant_weights() into
pre_dequant_phase0b_register_cutlass_nvfp4_. Pure structural.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Extract Phase 1 (FP16 weight cache)

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:539–659` → `pre_dequant_phase1_fp16_cache_`

**Note:** Phase 1 has a skip-guard structure at lines 539–553 (skip if FP8 prefill or NVFP4-decode-only). The skip+early-log lines need to live inside the method, returning early.

- [ ] **Step 1: Move the body, preserving the skip-guards**

```cpp
void GraphExecutor::pre_dequant_phase1_fp16_cache_(
    const ModelConfig& cfg, const VRAMBudget& budget,
    size_t& remaining_budget, cudaStream_t stream) {
    if (wcache_.use_fp8) {
        IMP_LOG_INFO(
            "FP8 prefill: skipping FP16 cache (Phase 1), "
            "all dense weights → FP8 cache (Phase 2)");
        return;
    }
    if (budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY) {
        IMP_LOG_INFO(
            "NVFP4 decode only: skipping FP16 cache (Phase 1), "
            "VRAM reserved for NVFP4 decode cache");
        return;
    }
    // ... lines 554–659 body (FP16 cache_weight lambda + per-layer loops) ...
    // remaining_budget update at end stays
}
```

- [ ] **Step 2: Replace call site**

```cpp
    pre_dequant_phase1_fp16_cache_(cfg, budget, remaining_budget, stream);
```

- [ ] **Step 3: Build + smoke test (Q8_0 — biggest FP16-cache exerciser)**

```bash
make build 2>&1 | tail -10
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3-8B-Q8_0/ --prompt "Hi" --max-tokens 5 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add src/graph/executor_pre_dequant.cu
git commit -m "refactor(pre-dequant): extract Phase 1 (FP16 weight cache)

Move 121 LOC out of pre_dequant_weights() into
pre_dequant_phase1_fp16_cache_. remaining_budget threaded by reference.
Pure structural.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Extract Phase 2 (FP8 cache)

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:660–806` → `pre_dequant_phase2_fp8_cache_`

- [ ] **Step 1: Move the body**

```cpp
void GraphExecutor::pre_dequant_phase2_fp8_cache_(
    const ModelConfig& cfg, size_t& remaining_budget, cudaStream_t stream) {
    if (!wcache_.use_fp8)
        return;
    // ... lines 665–806 body ...
}
```

Phase 2 starts with `if (wcache_.use_fp8)` at line 665 (verify), so wrap is same idiom.

- [ ] **Step 2: Replace call site + build + smoke test**

```bash
# Use Qwen3-8B Q8_0 + --kv-fp8 + --prefill-fp8 to exercise Phase 2
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3-8B-Q8_0/ --kv-fp8 --prefill-fp8 \
  --prompt "Hi" --max-tokens 5 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(pre-dequant): extract Phase 2 (FP8 cache)
..."
```

---

## Task 7: Extract Phase 3 (NVFP4 decode + 3b + 3c-native) — the big one

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:807–2070` → `pre_dequant_phase3_nvfp4_decode_`

**Risk:** This is 1264 LOC, the biggest chunk. Several local lambdas (`replace_weight` at 1493, `check`/`collect`/`replace`/`register_mx` at 1913/1947/1974/1990). All are local to Phase 3 so they move with it.

- [ ] **Step 1: Move the body**

```cpp
void GraphExecutor::pre_dequant_phase3_nvfp4_decode_(
    const ModelConfig& cfg, const VRAMBudget& budget,
    size_t& remaining_budget, cudaStream_t stream) {
    // ... entire Phase 3 + 3b + 3c-native body ...
}
```

The phase has its own internal sub-conditions (NVFP4 mode 1/2, MXFP4 sub-branch). All lambdas (`replace_weight`, etc.) are at higher indent inside `if (...)` blocks — they stay where they are when we move the whole region.

- [ ] **Step 2: Replace call site + build + smoke test on three model classes**

```bash
make build 2>&1 | tail -10

# (a) Q8_0 baseline (NVFP4 decode = OFF)
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3-8B-Q8_0/ --prompt "Hi" --max-tokens 5 2>&1 | tail -3

# (b) NVFP4 model (NVFP4 decode mode 2)
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3-8B-NVFP4-cortecs/ --prompt "Hi" --max-tokens 5 2>&1 | tail -3

# (c) MXFP4 GGUF (Phase 3c-native sub-path)
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3.5-4B-mxfp4.gguf --prompt "Hi" --max-tokens 5 2>&1 | tail -3
```

Expected: all three produce coherent output.

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(pre-dequant): extract Phase 3 (NVFP4 decode + 3b + 3c-native)
..."
```

---

## Task 8: Extract Phase 3c standalone MXFP4

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:2071–2218` → `pre_dequant_phase3c_standalone_mxfp4_`

- [ ] **Step 1: Move the body**

This block runs when NVFP4 decode is OFF but MXFP4 weights are present. Wrap with the original guard condition (verify against the existing `if (...)` at line 2071).

- [ ] **Step 2: Build + smoke test (MXFP4 with no NVFP4 decode)**

```bash
docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
  imp-cli --model /models/Qwen3.5-4B-mxfp4.gguf --no-nvfp4 \
  --prompt "Hi" --max-tokens 5 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

---

## Task 9: Extract Phase 4 tail (tensor registry + overlay + device-args)

**Files:**
- Modify: `src/graph/executor_pre_dequant.cu:2219–2589` → `pre_dequant_phase4_tensor_registry_`

- [ ] **Step 1: Move the body**

Phase 4 is a sequence of bookkeeping passes that run unconditionally. The `register_tensor` lambda at line 2047 is right before this block — verify whether it's used by Phase 4 (it likely is). If so, move it into the method too.

- [ ] **Step 2: Build + full smoke (all four model types)**

```bash
# Run all four to confirm tensor registry works for every quant
for m in Qwen3-8B-Q8_0 Qwen3-8B-NVFP4-cortecs; do
  docker run --rm --gpus all -v /home/kekz/models:/models imp:test \
    imp-cli --model /models/$m/ --prompt "Hi" --max-tokens 5 2>&1 | tail -3
done
```

- [ ] **Step 3: Commit**

---

## Task 10: Final verify + PR

**Files:**
- Verify: full pre_dequant_weights is now ~50 LOC orchestrator

- [ ] **Step 1: Inspect the new pre_dequant_weights**

Should look approximately like:

```cpp
void GraphExecutor::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
    if (!initialized_ || !model_)
        return;

    const auto& cfg = model_->config();
    size_t free_vram = 0, total_vram = 0;
    IMP_CUDA_CHECK_LOG(cudaMemGetInfo(&free_vram, &total_vram));
    size_t min_reserve = std::max(budget.reserve_bytes, total_vram / 10);
    size_t remaining_budget = (free_vram > min_reserve) ? (free_vram - min_reserve) : 0;

    {  // Phase 4.2 storage planner diagnostic (keep inline — 10 LOC)
        hints_.vram_budget_bytes = remaining_budget;
        StoragePlan diag_plan = plan_storage(*model_, cfg, hints_);
        if (diag_plan.failed) {
            IMP_LOG_WARN("StoragePlanner (diagnostic): plan failed — %s", diag_plan.failure_reason.c_str());
        } else {
            IMP_LOG_INFO("StoragePlanner (diagnostic): %zu entries, projected VRAM %.2f MiB",
                         diag_plan.entries.size(), diag_plan.projected_vram_bytes / (1024.0 * 1024.0));
        }
    }

    pre_dequant_phase0_promote_nvfp4_sidecars_(cfg, stream);
    pre_dequant_phase0b_register_cutlass_nvfp4_(cfg, stream);
    pre_dequant_phase1_fp16_cache_(cfg, budget, remaining_budget, stream);
    pre_dequant_phase2_fp8_cache_(cfg, remaining_budget, stream);
    pre_dequant_phase3_nvfp4_decode_(cfg, budget, remaining_budget, stream);
    pre_dequant_phase3c_standalone_mxfp4_(cfg, stream);
    pre_dequant_phase4_tensor_registry_(cfg, stream);
}
```

- [ ] **Step 2: Run `make verify-fast` for the full pre-merge gate**

```bash
make verify-fast 2>&1 | tail -20
```

Expected: PASS on fast gtest filter. Perf/smoke gates may SKIP if baseline GGUFs aren't in `$REPO/models/`.

- [ ] **Step 3: Push + create PR**

```bash
git push -u origin refactor/pre-dequant-decompose
gh pr create --base main --title "refactor(pre-dequant): decompose 2415-LOC pre_dequant_weights into 7 phase methods" --body "..."
```

PR body should explain:
- Motivation: god-file analysis identified pre_dequant_weights as a mega-function with 6+ concerns
- Approach: pure structural extraction, zero behavior change
- Verification: smoke test on Q8_0/NVFP4/MXFP4/MXFP4-no-NVFP4 paths after each phase + verify-fast at end
- Net: 2415 → ~50 LOC orchestrator + 7 private methods averaging ~340 LOC each

---

## Risks + mitigations

- **Lambda capture corruption**: a lambda inside Phase N captures a local from Phase N-1. Mitigation: each phase's lambdas are visibly local (they're declared inside the phase block already). Read carefully in Task 1.
- **Compile error from missing `this->`**: extracted method is a member of `GraphExecutor` and has full member access. No risk unless we accidentally moved the body outside the class definition.
- **Behavior change from reordering**: not possible — we're not reordering, just wrapping each existing block in a method call at the same place.
- **Build break partway through**: between Task 3 and Task 4 we have a half-extracted state on the working tree. Mitigation: each task commits after build+test pass. If a task's build fails, fix before committing; revert and re-think if it's structural.

## Self-Review checklist

- [x] **Spec coverage:** Every phase boundary identified in the depth-1 scan has a corresponding task. Phase 4.2 inline (10 LOC) intentionally kept inline.
- [x] **Placeholder scan:** No "TBD" / "handle edge cases" / "implement later". Code blocks for the new method bodies are precise (paste original content).
- [x] **Type consistency:** All 7 methods follow `pre_dequant_phaseN_<verb>_` naming and `void` return. Parameter list consistent across methods that need `remaining_budget` vs those that don't.
