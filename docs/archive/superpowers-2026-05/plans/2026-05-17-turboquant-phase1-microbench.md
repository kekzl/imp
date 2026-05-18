# TurboQuant Phase 1 — Bottleneck Microbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Definitively confirm or refute that the per-token QJL XNOR+popcount + Q-side sketch precompute is the dominant component of TurboQuant's ~23% decode-rate gap vs FP8 on Qwen3-8B Q8_0, before committing to the 2-3 week kernel rewrite Phase 2-5 of the design memo describes.

**Architecture:** Add a debug-only `diagnostics.tq_skip_qjl` runtime flag that compile-time-strips the QJL XNOR+popcount path from `paged_attention_decode_turboquant_kernel` via a new `SKIP_QJL` template parameter. With the flag flipped, the kernel runs the PolarQuant FP4-dequant dot only and combines `dot = dot_polar` (skipping the per-token QJL byte-loads, popcounts, and the Q-side sketch precompute). A bench script captures nsys timeline + ncu `ComputeWorkloadAnalysis` for {full-TQ, TQ-stripped, FP8, NVFP4} on Qwen3-8B Q8_0 at pp={512, 4096}, tg=256. Per-token kernel time delta between full-TQ and TQ-stripped is the QJL cost in isolation; NVFP4 serves as a proxy for the post-Path-A perf ceiling (NVFP4's per-token compute structure is identical to MXFP4-K-with-no-QJL except for scale dtype, which is the same FP16 register load on both).

**Tech Stack:** CUDA 13.2 sm_120a, RuntimeConfig (`src/runtime/config.{h,cpp}`), `src/compute/attention_paged_turboquant.cu` (1108 LOC, four kernels: decode + splitk for std-TQ + decode + splitk for TQ-Lite), `tests/test_turboquant.cu` for parity check, nsys + ncu inside the `imp:test` Docker image.

**Acceptance criteria from design memo §5:**
- QJL-stripped kernel runs ≥ 15% faster per token than full TQ → Path A bottleneck-targeted, proceed to Phase 2.
- NVFP4 microbench within 5% of FP8 per-token cost → Path A perf ceiling confirmed.
- If both criteria fail: write a "Path A refuted" memo and shelve.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/runtime/config.h` | Modify (~5 lines) | Add `diagnostics.tq_skip_qjl` field |
| `src/runtime/config.cpp` | Modify (~6 lines) | Parse `diagnostics.tq_skip_qjl`; legacy env `IMP_TQ_SKIP_QJL` |
| `src/compute/attention_paged_turboquant.cu` | Modify (~40 lines) | Add `bool SKIP_QJL` template param to four kernels; gate XNOR+popcount + Q-sketch precompute |
| `src/compute/attention_paged.h` | Modify (~2 lines) | No signature change — `skip_qjl` resolved inside the .cu via `RuntimeConfig::current()` |
| `src/graph/executor_attention.cu` | No change | Dispatch site keeps current signature — flag read inside kernel launcher |
| `tests/test_turboquant.cu` | Modify (~30 lines) | Add `TurboQuantStripQJLMatchesPolarQuantOnly` parity test |
| `tools/analysis/bench_turboquant_components.sh` | Create (~120 lines) | Capture nsys + ncu for {TQ-full, TQ-stripped, FP8, NVFP4} on Qwen3-8B Q8_0 |
| `docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md` | Create at end of phase | Findings memo + ship/no-ship decision |

---

## Task 1: Worktree setup (skill-delegated)

**Files:** none (uses `superpowers:using-git-worktrees`)

- [ ] **Step 1: Invoke the worktree skill**

Use `superpowers:using-git-worktrees` to create an isolated worktree off `main` named `perf/turboquant-phase1-microbench`. Phase 1's code changes (kernel template param + runtime flag + one bench script + one parity test) need isolation from any unrelated in-flight work on the host branch.

- [ ] **Step 2: Confirm starting state**

Run: `git status && git log --oneline -3`
Expected: clean tree, on the new `perf/turboquant-phase1-microbench` branch, HEAD is the merge-base with `main`.

---

## Task 2: Add `diagnostics.tq_skip_qjl` runtime flag

**Files:**
- Modify: `src/runtime/config.h:220-246`
- Modify: `src/runtime/config.cpp:251-282` (apply_overrides section) and `:330-360` (seed_from_env section)

- [ ] **Step 1: Add the flag to RuntimeConfig::Diagnostics**

In `src/runtime/config.h`, add inside `struct { ... } diagnostics;` (after the `audit_nvfp4_scales` field at line 245):

```cpp
        // Bench-only: when true, the TurboQuant decode kernels skip the QJL
        // XNOR+popcount correction and the Q-side QJL sketch precompute; the
        // per-token dot collapses to dot_polar (PolarQuant FP4 dequant dot
        // alone, with kQJLLambda forced to 0). Used by Phase 1 microbench to
        // isolate per-token QJL cost from total kernel time. Output is NOT
        // bit-equivalent to the QJL-on path; this is a perf-isolation tool,
        // not a quality flag. Legacy env: IMP_TQ_SKIP_QJL=1.
        bool tq_skip_qjl = false;
```

- [ ] **Step 2: Wire the override parser**

In `src/runtime/config.cpp`, find the block around line 281-282 that parses `diagnostics.audit_nvfp4_scales`. Add immediately after it:

```cpp
    else if (eq("diagnostics.tq_skip_qjl"))
        cfg.diagnostics.tq_skip_qjl = parse_bool(val, cfg.diagnostics.tq_skip_qjl);
```

- [ ] **Step 3: Wire the legacy env seed**

In `src/runtime/config.cpp` `seed_from_env()`, find the `IMP_AUDIT_NVFP4_SCALES` block around line 358-359. Add immediately after it (follow the surrounding pattern — `'1' only` semantics for one-shot toggles):

```cpp
    // diagnostics.tq_skip_qjl — IMP_TQ_SKIP_QJL: '1' only.
    if (const char* e = std::getenv("IMP_TQ_SKIP_QJL"))
        cfg.diagnostics.tq_skip_qjl = (e[0] == '1');
```

- [ ] **Step 4: Build to verify the flag compiles**

Run: `make build 2>&1 | tail -20`
Expected: build succeeds. The flag is declared but not yet consumed — no behavior change.

- [ ] **Step 5: Commit**

```bash
git add src/runtime/config.h src/runtime/config.cpp
git commit -m "feat(runtime): add diagnostics.tq_skip_qjl flag for TurboQuant Phase 1 bench

Adds a bench-only RuntimeConfig flag that the TurboQuant decode kernels
(wired in the next commit) will read to compile-out the QJL XNOR+popcount
correction and Q-side sketch precompute. Used by Phase 1 microbench to
isolate per-token QJL cost vs total kernel time.

Output of the QJL-stripped path is NOT bit-equivalent to QJL-on. This is
a perf-isolation tool, not a quality flag."
```

---

## Task 3: Add `SKIP_QJL` template parameter to TurboQuant decode kernels

**Files:**
- Modify: `src/compute/attention_paged_turboquant.cu:43-279` (decode kernel)
- Modify: `src/compute/attention_paged_turboquant.cu:285-510` (splitk kernel)
- Modify: `src/compute/attention_paged_turboquant.cu:515-622` (launcher block — add a second `SKIP_QJL=true` dispatch path)

Skip the TurboQuant-Lite kernels for Phase 1 — Lite is QJL-only, stripping it removes the kernel's reason to exist, and Lite is not part of the Path A scope per the design memo §3.1.4 (Lite is slated for retirement, not optimisation).

- [ ] **Step 1: Add a parity test for the QJL-on path to lock in current behavior**

Edit `tests/test_turboquant.cu`. Find an existing TurboQuant decode test (e.g., a `paged_attention_decode_turboquant` invocation) and copy its setup into a new test that runs the kernel twice with the same inputs — once before any code change, once after — and asserts bit-equivalent output. This test prevents accidental regression of the QJL-on path while we add the template parameter.

Add at the bottom of the file:

```cpp
TEST(TurboQuantDecode, QJLOnPathBitIdenticalAfterTemplateRefactor) {
    // Anchor test: with SKIP_QJL=false (current behavior), the kernel output
    // must remain bit-identical to the pre-refactor baseline. Synthetic
    // small case: batch=1, n_heads=4, n_kv_heads=4, head_dim=128, ctx=64,
    // block_size=16. Run on fixed-seed RNG inputs.
    constexpr int batch = 1, n_heads = 4, n_kv_heads = 4, head_dim = 128;
    constexpr int ctx_len = 64, block_size = 16;
    // ... (setup mirrors the existing TurboQuant decode test in this file;
    // reuse its helpers — find_or_alloc_qjl_matrix, write_tq_kv_cache_synthetic,
    // etc.). Compare against a checked-in golden FP32 reference computed
    // off-line with kQJLLambda=0.1 — or, if no golden exists, snapshot the
    // first run's output and assert all subsequent runs match it.
    GTEST_SKIP() << "Anchor — fill in once the existing TurboQuant test "
                    "helpers are inspected; bit-identity check guards against "
                    "the template-param refactor in this PR.";
}
```

Then **scan the existing tests in `test_turboquant.cu`** for any test that already exercises `paged_attention_decode_turboquant` end-to-end. If one exists, the parity test is unnecessary — the existing test will catch any regression. Document the finding in the commit message.

- [ ] **Step 2: Run the existing TurboQuant test suite as a baseline**

Run: `make test-gpu 2>&1 | grep -i turboquant | head -20`
Expected: existing TurboQuant tests pass. Record the count for comparison after Step 5.

- [ ] **Step 3: Add `SKIP_QJL` template parameter to `paged_attention_decode_turboquant_kernel`**

In `src/compute/attention_paged_turboquant.cu`, change the kernel template signature at line 43:

```cpp
template <int HEAD_DIM, bool USE_MXFP4 = false, bool SKIP_QJL = false>
__global__ void __launch_bounds__(256, 2) paged_attention_decode_turboquant_kernel(
    /* ... existing args unchanged ... */) {
```

Inside the kernel, gate the Q-side QJL sketch precompute (lines 95-136). Wrap the entire `// Compute Q's QJL sketch` block + the trailing `__syncthreads()` at line 136 in:

```cpp
    if constexpr (!SKIP_QJL) {
        // ... existing Q-sketch precompute block (lines 95-135) ...
    }
    __syncthreads();
```

The `__syncthreads()` outside the `if constexpr` is required to keep all warps synchronized before entering the inner loop regardless of the SKIP_QJL choice (otherwise some warps may race against the smem layout used later).

Inside the inner loop, gate the per-token QJL correction (lines 233-251). Replace:

```cpp
            // QJL correction: warp-parallel XNOR+popcount
            float dot_qjl;
            {
                /* ... existing XNOR+popcount block ... */
                dot_qjl = q_norm * k_norm * static_cast<float>(2 * match_count - sketch_dim) * inv_sketch_dim;
            }

            // Combined estimate with QJL correction
            float dot = (1.0f - kQJLLambda) * dot_polar + kQJLLambda * dot_qjl;
```

with:

```cpp
            float dot;
            if constexpr (SKIP_QJL) {
                dot = dot_polar;
            } else {
                // QJL correction: warp-parallel XNOR+popcount
                float dot_qjl;
                {
                    const uint8_t* k_sketch = K_sk_block + t * sketch_slot_stride + kv_head * sketch_head_bytes;
                    const uint32_t* q_sketch32 = reinterpret_cast<const uint32_t*>(q_sketch);
                    const int n_words = sketch_bytes / 4;
                    int local_match = 0;
                    for (int sb = lane_id; sb < n_words; sb += WARP_SIZE) {
                        uint32_t k_word;
                        memcpy(&k_word, k_sketch + sb * 4, sizeof(uint32_t));
                        local_match += __popc(~(q_sketch32[sb] ^ k_word));
                    }
                    int match_count = static_cast<int>(warp_reduce_sum(static_cast<float>(local_match)));
                    dot_qjl = q_norm * k_norm * static_cast<float>(2 * match_count - sketch_dim) * inv_sketch_dim;
                }
                dot = (1.0f - kQJLLambda) * dot_polar + kQJLLambda * dot_qjl;
            }
```

Compiler will dead-code-eliminate the `if constexpr (SKIP_QJL)` branch entirely — the resulting SASS for the SKIP_QJL=true instantiation will not contain the popc / sketch byte loads. This is what makes the Phase 1 measurement clean.

- [ ] **Step 4: Repeat the gating in `paged_attention_splitk_turboquant_kernel`**

In `src/compute/attention_paged_turboquant.cu:285`, apply the identical change pattern:
1. Add `bool SKIP_QJL = false` template parameter (line 285).
2. Wrap the Q-side QJL sketch precompute (lines 340-377 region) in `if constexpr (!SKIP_QJL)`.
3. Wrap the per-token QJL correction (lines 468-485 region) in the same `if constexpr (SKIP_QJL) { dot = dot_polar; } else { /* existing XNOR+popcount + combine */ }` pattern.

The line numbers shift after Step 3's edits — re-grep for `// QJL correction:` and `// Compute Q's QJL sketch` inside the splitk kernel body to find the exact insertion points.

- [ ] **Step 5: Build and run the TurboQuant test suite to confirm QJL-on path is unchanged**

Run: `make build && make test-gpu 2>&1 | grep -i turboquant | head -20`
Expected: same count of passing tests as Step 2's baseline (no regressions). Since both new instantiations default `SKIP_QJL=false` and no dispatch site has been updated yet, this is purely a refactor-safety check.

- [ ] **Step 6: Commit**

```bash
git add src/compute/attention_paged_turboquant.cu tests/test_turboquant.cu
git commit -m "perf(turboquant): add SKIP_QJL template param to decode kernels (no-op by default)

Adds a SKIP_QJL=false template parameter to paged_attention_decode_turboquant_kernel
and paged_attention_splitk_turboquant_kernel. Default behavior unchanged —
both kernels still emit the QJL XNOR+popcount + Q-side sketch precompute
and combine dot_polar + dot_qjl with kQJLLambda=0.1.

The SKIP_QJL=true instantiation compiles out the QJL paths entirely via
if constexpr, producing a PolarQuant-only kernel. Used by the next commit
to wire the diagnostics.tq_skip_qjl runtime flag for Phase 1 microbench
QJL-cost isolation.

TurboQuant-Lite is intentionally not touched — it's QJL-only and slated
for retirement per the Phase 1 design memo §3.1.4."
```

---

## Task 4: Wire the `tq_skip_qjl` flag into the TurboQuant launcher

**Files:**
- Modify: `src/compute/attention_paged_turboquant.cu:515-622` (the `paged_attention_decode_turboquant` host launcher)

- [ ] **Step 1: Add a SKIP_QJL dispatch arm to the launcher**

In `src/compute/attention_paged_turboquant.cu`, find the host launcher `paged_attention_decode_turboquant(...)` (around line 515-622). It currently dispatches on `(head_dim, USE_MXFP4)`. Add an outer branch on `RuntimeConfig::current().diagnostics.tq_skip_qjl`:

```cpp
void paged_attention_decode_turboquant(/* args */) {
    /* ... existing setup ... */
    const bool skip_qjl = RuntimeConfig::current().diagnostics.tq_skip_qjl;
    if (skip_qjl) {
        // Mirror the existing dispatch macro, but with SKIP_QJL=true.
#define LAUNCH_TQ_SKIP_QJL(HD, MXFP4_FLAG)                                                                  \
        paged_attention_decode_turboquant_kernel<HD, MXFP4_FLAG, /*SKIP_QJL=*/true>                         \
            <<<grid, block, smem_bytes, stream>>>(/* same arg list as the existing launcher */)
        /* head_dim x use_mxfp4 dispatch mirroring the existing block */
#undef LAUNCH_TQ_SKIP_QJL
    } else {
        /* ... existing dispatch block unchanged ... */
    }
}
```

Don't refactor the existing dispatch into a shared macro — keep both branches' macros local. Phase 5 of the design memo retires this whole file anyway, so the duplication is intentional throwaway scaffolding.

Add `#include "runtime/config.h"` at the top of the file if not already present.

- [ ] **Step 2: Repeat for the splitk launcher**

Find the splitk dispatch (in the same file, typically immediately after the non-splitk launcher) and add the same `skip_qjl ? SKIP_QJL=true : SKIP_QJL=false` outer branch with identical pattern.

- [ ] **Step 3: Build and run the TurboQuant test suite**

Run: `make build && make test-gpu 2>&1 | grep -i turboquant | head -20`
Expected: all existing TurboQuant tests pass — they don't set `diagnostics.tq_skip_qjl`, so they hit the unchanged SKIP_QJL=false dispatch arm.

- [ ] **Step 4: Manual smoke test with the flag flipped**

Run:
```bash
docker run --rm --gpus all -v $(pwd)/models:/m imp:test bash -c "
  IMP_TQ_SKIP_QJL=1 imp-cli --model /m/Qwen3-8B-Q8_0.gguf --kv-turboquant \
    --bench --bench-pp 128 --bench-reps 1 --max-tokens 16 --temperature 0 2>&1 | tail -5
"
```
Expected: the run completes (no IMA, no crash). The output text will likely be degraded (QJL correction is doing real work for quality at long context), but for a 16-token bench prompt that's irrelevant — we want the kernel to run without crashing.

If it crashes: the most likely cause is the `if constexpr (!SKIP_QJL)` Q-sketch block reading uninitialized `q_sketch` smem in the inner loop. Audit Step 3 of Task 3 — the smem zero-init at lines 105-108 should run regardless of SKIP_QJL because subsequent code (the smem reduction at `crosswarp_reduce_and_write`) reuses the same smem region.

- [ ] **Step 5: Commit**

```bash
git add src/compute/attention_paged_turboquant.cu
git commit -m "perf(turboquant): dispatch SKIP_QJL=true when diagnostics.tq_skip_qjl is set

Wires the diagnostics.tq_skip_qjl RuntimeConfig flag (added in the previous
commit) into both TurboQuant decode launchers. Adds a second dispatch arm
that instantiates SKIP_QJL=true; the default arm is unchanged.

This is the Phase 1 bench knob: with IMP_TQ_SKIP_QJL=1, decode produces
incoherent text (QJL is doing real work on retrieval-heavy traffic) but
the kernel time isolates the per-token PolarQuant cost from the per-token
QJL cost. Difference between the two run modes is what Phase 1 measures."
```

---

## Task 5: Write the Phase 1 bench script

**Files:**
- Create: `tools/analysis/bench_turboquant_components.sh`

- [ ] **Step 1: Write the bench script**

Create `tools/analysis/bench_turboquant_components.sh` with this content:

```bash
#!/bin/bash
# Phase 1 of the TurboQuant–FP8 gap design memo.
# Captures nsys + ncu for {TQ-full, TQ-stripped, FP8, NVFP4} on Qwen3-8B Q8_0
# at pp=512 and pp=4096 with tg=256, and reports the per-token kernel-time
# fraction attributable to QJL XNOR+popcount + Q-side sketch precompute.
#
# Acceptance per design memo §5:
#   - (TQ-full − TQ-stripped) / TQ-full >= 15%  → Path A bottleneck-targeted.
#   - (NVFP4 − FP8) / FP8 <= 5%                 → Path A perf ceiling confirmed.
#
# Usage:
#   tools/analysis/bench_turboquant_components.sh [model_path]
# Default: models/Qwen3-8B-Q8_0.gguf
#
# Requires: imp:test docker image (built via `make build`), nsight-systems
# on host at /opt/nvidia/nsight-systems, nsight-compute on host at
# /opt/nvidia/nsight-compute or inside the container.

set -e
MODEL_PATH="${1:-/m/Qwen3-8B-Q8_0.gguf}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
OUT_DIR="${OUT_DIR:-/tmp/tq_phase1}"

mkdir -p "$OUT_DIR"
chmod 777 "$OUT_DIR"

run_nsys() {
    local label="$1"
    local kv_flag="$2"
    local env_extra="$3"
    local pp="$4"
    echo "=== nsys: $label  kv=$kv_flag  env=$env_extra  pp=$pp ==="
    docker run --rm --gpus all \
        -v "$MODELS_DIR":/m \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
        -v "$OUT_DIR":/out \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        $env_extra \
        imp:test bash -c "
            /usr/local/cuda/bin/nsys profile -t cuda,nvtx \
                -o /out/${label}_pp${pp} --force-overwrite=true \
                imp-cli --model '$MODEL_PATH' $kv_flag \
                --bench --bench-pp $pp --bench-reps 3 --max-tokens 256 \
                --temperature 0 --no-cuda-graphs 2>&1 | tail -5
        "
    echo ""
    echo "    Top kernels for $label pp=$pp:"
    docker run --rm \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v "$OUT_DIR":/out \
        imp:test \
        /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum \
            --format csv --force-export=true \
            "/out/${label}_pp${pp}.nsys-rep" 2>/dev/null \
        | grep -iE "paged_attention|cublas" | head -8
    echo ""
}

run_ncu() {
    local label="$1"
    local kv_flag="$2"
    local env_extra="$3"
    echo "=== ncu ComputeWorkloadAnalysis: $label  kv=$kv_flag ==="
    docker run --rm --gpus all \
        -v "$MODELS_DIR":/m \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v "$OUT_DIR":/out \
        -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        $env_extra \
        imp:test bash -c "
            /usr/local/cuda/bin/ncu \
                --section ComputeWorkloadAnalysis \
                --section MemoryWorkloadAnalysis \
                --kernel-name 'regex:paged_attention_(decode|splitk)_turboquant' \
                --kernel-name 'regex:paged_attention_(decode|splitk)_(fp8|nvfp4)' \
                --launch-skip 5 --launch-count 3 \
                --csv --log-file /out/${label}_ncu.log \
                imp-cli --model '$MODEL_PATH' $kv_flag \
                --bench --bench-pp 512 --bench-reps 1 --max-tokens 32 \
                --temperature 0 --no-cuda-graphs 2>/dev/null | tail -3
        "
    echo "    ncu log: $OUT_DIR/${label}_ncu.log"
    echo ""
}

echo "============================================================="
echo "TurboQuant Phase 1 bench — model: $MODEL_PATH"
echo "============================================================="
echo ""

# 1. Full TurboQuant
run_nsys tq_full "--kv-turboquant" "" 512
run_nsys tq_full "--kv-turboquant" "" 4096
run_ncu  tq_full "--kv-turboquant" ""

# 2. TurboQuant with QJL stripped
run_nsys tq_stripped "--kv-turboquant" "-e IMP_TQ_SKIP_QJL=1" 512
run_nsys tq_stripped "--kv-turboquant" "-e IMP_TQ_SKIP_QJL=1" 4096
run_ncu  tq_stripped "--kv-turboquant" "-e IMP_TQ_SKIP_QJL=1"

# 3. FP8 (the perf target we're trying to match)
run_nsys fp8 "--kv-fp8" "" 512
run_nsys fp8 "--kv-fp8" "" 4096
run_ncu  fp8 "--kv-fp8" ""

# 4. NVFP4 (proxy for the post-Path-A MXFP4-K decode cost — same per-token
#    structure modulo scale dtype, which is a register-resident FP16 load
#    either way).
run_nsys nvfp4 "--kv-nvfp4" "" 512
run_nsys nvfp4 "--kv-nvfp4" "" 4096
run_ncu  nvfp4 "--kv-nvfp4" ""

echo ""
echo "============================================================="
echo "Summary"
echo "============================================================="
echo ""
echo "nsys reports: $OUT_DIR/*.nsys-rep"
echo "ncu CSV logs: $OUT_DIR/*_ncu.log"
echo ""
echo "To compute acceptance criteria, pick the paged_attention_decode_*"
echo "row from each report's cuda_gpu_kern_sum and compare 'Avg (ns)':"
echo "  qjl_fraction = (avg_tq_full - avg_tq_stripped) / avg_tq_full"
echo "  ceiling_gap  = (avg_nvfp4 - avg_fp8) / avg_fp8"
echo ""
echo "Acceptance per design memo §5:"
echo "  qjl_fraction >= 0.15  → Path A bottleneck-targeted (PROCEED)"
echo "  ceiling_gap  <= 0.05  → Path A perf ceiling confirmed (PROCEED)"
echo ""
echo "Write findings to:"
echo "  docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md"
```

- [ ] **Step 2: Make the script executable and dry-run it for help/syntax**

Run: `chmod +x tools/analysis/bench_turboquant_components.sh && bash -n tools/analysis/bench_turboquant_components.sh && echo OK`
Expected: `OK`. Bash `-n` is parser-only; it doesn't execute anything.

- [ ] **Step 3: Commit**

```bash
git add tools/analysis/bench_turboquant_components.sh
git commit -m "tools: add bench_turboquant_components.sh for TurboQuant Phase 1

Captures nsys timeline + ncu ComputeWorkloadAnalysis for {TQ-full,
TQ-stripped (IMP_TQ_SKIP_QJL=1), FP8, NVFP4} on Qwen3-8B Q8_0 at
pp={512, 4096}, tg=256.

Used by Phase 1 of docs/plans/turboquant_fp8_gap_design_2026_05_17.md
to isolate per-token QJL kernel cost and check the design memo's two
acceptance criteria before committing to Path A's kernel rewrite."
```

---

## Task 6: Run the Phase 1 bench (measurement, not code)

**Files:** none (produces `/tmp/tq_phase1/*.nsys-rep` + `*_ncu.log`)

- [ ] **Step 1: Confirm Qwen3-8B Q8_0 is present**

Run: `ls -la models/Qwen3-8B-Q8_0.gguf`
Expected: file exists. If not: download or copy from `/home/kekz/models/`.

- [ ] **Step 2: Run the bench script**

Run: `tools/analysis/bench_turboquant_components.sh 2>&1 | tee /tmp/tq_phase1/run.log`
Expected: completes in ~5-10 minutes. Three nsys runs + three ncu runs per config × four configs = ~12 minutes wall clock.

- [ ] **Step 3: Verify all four configs ran cleanly**

Run: `ls -la /tmp/tq_phase1/*.nsys-rep`
Expected: 8 .nsys-rep files (4 configs × 2 pp lengths). If any are missing, check `/tmp/tq_phase1/run.log` for IMA / OOM / config errors and re-run that config.

- [ ] **Step 4: Extract the headline numbers**

For each of the 8 nsys reports, extract the `Avg (ns)` of the relevant `paged_attention_decode_*` kernel. Use:

```bash
for f in /tmp/tq_phase1/*.nsys-rep; do
    echo "=== $f ==="
    docker run --rm \
        -v /usr/local/cuda:/usr/local/cuda:ro \
        -v /tmp/tq_phase1:/out \
        imp:test \
        /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum \
            --format csv --force-export=true "/out/$(basename $f)" 2>/dev/null \
        | grep -E "paged_attention_decode|paged_attention_splitk" | head -3
done | tee /tmp/tq_phase1/summary.txt
```

This is data collection — no code change.

---

## Task 7: Write the Phase 1 findings memo + ship/no-ship decision

**Files:**
- Create: `docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md`

- [ ] **Step 1: Draft the findings memo from `/tmp/tq_phase1/summary.txt`**

Create the file with this structure (the numbers are the data; fill them in from Step 4 of Task 6):

```markdown
# TurboQuant Phase 1 findings

**Date:** 2026-05-17
**Branch:** `perf/turboquant-phase1-microbench`
**Scope:** Verify the design memo §1.2 "QJL is the bottleneck" hypothesis.
**Decision gate:** docs/plans/turboquant_fp8_gap_design_2026_05_17.md §5

## Measurements

Qwen3-8B Q8_0, RTX 5090, CUDA 13.2, 3 reps per config, --no-cuda-graphs.

| Config            | Avg (ns) pp=512 | Avg (ns) pp=4096 | Δ vs full-TQ |
|---                |---:             |---:              |---:          |
| TQ full           | (fill)          | (fill)           | 0% (anchor)  |
| TQ stripped       | (fill)          | (fill)           | -X.Y%        |
| FP8               | (fill)          | (fill)           | -A.B%        |
| NVFP4 (Path A ceiling proxy) | (fill) | (fill)         | -C.D%        |

QJL fraction = (TQ_full − TQ_stripped) / TQ_full = **X.Y%**
Ceiling gap   = (NVFP4 − FP8) / FP8                = **C.D%**

## Acceptance criteria (per design memo §5)

- [ ] / [x] QJL fraction ≥ 15% → Path A bottleneck-targeted.
- [ ] / [x] Ceiling gap ≤ 5%   → Path A perf ceiling confirmed.

## Decision

(One of three outcomes — fill in based on the table above.)

**A. Both criteria met → PROCEED to Phase 2 (NIAH quality A/B).**
    Open PR scoping Phase 2; reference this memo.

**B. QJL fraction met but ceiling gap fails → PROCEED with caveat.**
    Path A closes the gap to FP8 but leaves a residual N% NVFP4-vs-FP8
    cost — likely the K-norm extra FP16 load + INT4 V dequant. Document
    the residual; Path A still recovers most of the 23%. Phase 2 still
    needed for quality A/B.

**C. QJL fraction fails (<15%) → SHELVE.**
    The 23% gap is not dominated by QJL — Path A's kernel rewrite won't
    close it. Roadmap line `docs/roadmap.md:65-67` should be updated to
    reflect the refuted hypothesis. Next steps: defer; focus on `pp=512`
    large dense or other roadmap items.

## Surface findings

(Anything surprising in the nsys / ncu traces that wasn't in the design
memo's predictions: K-sketch byte load latency, smem bank conflicts in
the Q-sketch precompute, divergent dispatch overhead between SKIP_QJL=true
and SKIP_QJL=false instantiations, etc.)

## Next steps

(If A or B: outline Phase 2 — NIAH harness, model set, threshold; reference
design memo §5 Phase 2.)
(If C: outline the shelving — what gets removed from the roadmap, whether
the `tq_skip_qjl` flag stays as a debug knob or gets reverted.)
```

- [ ] **Step 2: Update the project memory file with the Phase 1 outcome**

Add a one-line pointer to `MEMORY.md` and a per-finding file in `memory/`. Example:

In `/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/MEMORY.md`, under the "Shipped root-cause fixes / Open issues" section (whichever is closer to "in-progress investigations"):

```markdown
- [TurboQuant Phase 1 findings](turboquant_phase1_findings_2026_05_17.md) — (one-line outcome: PROCEED / PROCEED-WITH-CAVEAT / SHELVE)
```

And create `memory/turboquant_phase1_findings_2026_05_17.md` mirroring the findings memo, distilled to a body of ≤ 30 lines (frontmatter per memory format).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md
git commit -m "docs(plans): TurboQuant Phase 1 findings + ship/no-ship decision

Captures the nsys + ncu measurements from
tools/analysis/bench_turboquant_components.sh on Qwen3-8B Q8_0:
  - QJL fraction:  X.Y%
  - Ceiling gap:   C.D%

Decision: <A | B | C> per the design memo §5 acceptance criteria.

Next step: <Phase 2 NIAH | shelve | …>"
```

---

## Task 8: Update the roadmap to reflect Phase 1 outcome

**Files:**
- Modify: `docs/roadmap.md:65-67`

- [ ] **Step 1: Update the TurboQuant roadmap entry**

Replace the existing roadmap entry's recommendation paragraph with the Phase 1 outcome. Pick the matching template:

**If PROCEED:**
```markdown
### Closing the TurboQuant–FP8 gap

TurboQuant currently runs ~23% behind FP8 on Qwen3-8B Q8_0 decode (191
vs 248 tok/s). **Phase 1 confirmed (2026-05-17): the QJL XNOR+popcount
correction accounts for X.Y% of decode-kernel time; the NVFP4 ceiling
proxy lands C.D% from FP8.** Both acceptance criteria from
`docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5 are met. Proceeding
to Phase 2 (NIAH retrieval-quality A/B at 4K + 16K context). Findings:
`docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md`.
```

**If SHELVE:**
```markdown
### ~~Closing the TurboQuant–FP8 gap~~ — Phase 1 refuted

TurboQuant currently runs ~23% behind FP8 on Qwen3-8B Q8_0 decode (191
vs 248 tok/s). **Phase 1 (2026-05-17) measured the QJL XNOR+popcount
correction at only X.Y% of decode-kernel time, below the 15% threshold
in `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5.** The
"algorithm-inherent QJL overhead" diagnosis is refuted; Path A's kernel
rewrite would not close the gap. TurboQuant stays as-is (opt-in, 23%
caveat documented). Findings:
`docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): update TurboQuant–FP8 gap entry with Phase 1 outcome

References docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md
for the measurements + decision."
```

---

## Task 9: Open PR for review

- [ ] **Step 1: Verify everything builds + tests pass**

Run: `make verify-fast 2>&1 | tail -20`
Expected: green. (verify-fast: build + filtered tests + perf gate + smoke prompt, ~90s.)

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin perf/turboquant-phase1-microbench
gh pr create --base main --title "perf(turboquant): Phase 1 microbench + bottleneck verification" \
    --body "$(cat <<'EOF'
## Summary

Implements Phase 1 of `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` — a 3-5 day standalone microbench gate before any of the Path A kernel-rewrite work.

- Adds `diagnostics.tq_skip_qjl` runtime flag (also `IMP_TQ_SKIP_QJL=1`).
- Adds `SKIP_QJL` template parameter to `paged_attention_decode_turboquant_kernel` and `paged_attention_splitk_turboquant_kernel`. Defaults to `false`; the SKIP_QJL=true instantiation compiles out the QJL XNOR+popcount + Q-side sketch precompute via `if constexpr`.
- Wires the flag through both launchers' dispatch sites.
- Adds `tools/analysis/bench_turboquant_components.sh` to capture nsys + ncu for {TQ-full, TQ-stripped, FP8, NVFP4} on Qwen3-8B Q8_0 at pp={512, 4096}, tg=256.
- Phase 1 findings: `docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md`.
- Roadmap entry `docs/roadmap.md:65-67` updated with the outcome.

## Test plan

- [ ] `make test-gpu 2>&1 | grep -i turboquant` matches pre-PR pass count (QJL-on path unchanged).
- [ ] `make verify-fast` green.
- [ ] `IMP_TQ_SKIP_QJL=1 imp-cli --model models/Qwen3-8B-Q8_0.gguf --kv-turboquant --bench --bench-pp 128` runs without crash.
- [ ] `tools/analysis/bench_turboquant_components.sh` ran end-to-end and produced 8 .nsys-rep files + 4 ncu CSVs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Capture the PR URL**

The `gh pr create` output line ending in `/pull/N` — drop it into the findings memo's "Next steps" section for traceability.

---

## Self-review

**1. Spec coverage check** (against design memo §5 Phase 1's four tasks):

- ✅ Task 1 ("nsys + ncu for `paged_attention_decode_turboquant_kernel<128,true>` across pp={512, 4096} tg=256") → Task 5/6 (bench script + run).
- ✅ Task 2 ("QJL-stripped kernel variant, debug-only, behind `RuntimeConfig::diagnostics`, short-circuit XNOR+popcount + `kQJLLambda=0`") → Tasks 2, 3, 4.
- ✅ Task 3 ("compute per-token QJL cost as fraction of total kernel time") → Task 6 Step 4 + Task 7's `qjl_fraction` field.
- ✅ Task 4 ("microbench MXFP4-K-only kernel using `paged_attention_decode_nvfp4_kernel`") → Task 5's NVFP4 dispatch arm — the design memo notes NVFP4's per-token compute is structurally identical to MXFP4-K's, so the NVFP4 KV path is a faithful proxy for the post-Path-A perf ceiling.

**2. Placeholder scan:**

- The Step 1 of Task 3's `GTEST_SKIP` placeholder is intentional — it's a documented "skip + audit existing tests first" gate, not a hidden TODO. Resolves to either a real test or a deletion in Step 1's audit step. Re-evaluate after the audit.
- All other steps have full content.

**3. Type consistency:**

- `SKIP_QJL` is consistently the third template parameter (`<int HEAD_DIM, bool USE_MXFP4, bool SKIP_QJL>`) in Tasks 3 and 4.
- `diagnostics.tq_skip_qjl` (snake_case field) matches the project's `RuntimeConfig` style.
- `IMP_TQ_SKIP_QJL` env-var follows the existing `IMP_*` legacy seed pattern (e.g., `IMP_NVFP4_FORCE_DEQUANT`).

---

## Notes

- **Time budget:** 3-5 days per design memo. Tasks 1-5 are ~1-2 days of code; Task 6 is ~1 hour wall clock once the script is in place; Tasks 7-9 are ~1 day to write up + open PR.
- **Worktree:** isolation matters because Task 3 modifies a 1108-line CUDA file that's part of the KV cache hot path. A bad edit can break TurboQuant entirely; verifying via `make test-gpu` after each task minimises blast radius.
- **CUDA Graphs disabled in bench script:** `--no-cuda-graphs` is mandatory for kernel-level nsys timing (per CLAUDE.md "nsys needs --no-cuda-graphs"). The bench measures kernel time in isolation, not end-to-end tok/s — that's a different number and is not the Phase 1 acceptance metric.
- **Why NVFP4 as the MXFP4-K proxy:** the design memo §3.1.1 establishes that "MXFP4 K with UE8M0 micro-scales" and "NVFP4 K with E4M3 per-16 scales" have structurally identical per-token decode compute: `cvt.rn.f16x2.e2m1x2` + scale FMA + dot accumulate. The only kernel difference is one scale-decode instruction (UE8M0 → fp32 vs e4m3 → half), which is a single FFMA — negligible compared to the 32 FMAs in the K dequant + dot. NVFP4 KV in current main = the cheapest faithful proxy for Path A's post-rewrite perf.
- **Findings memo location:** the design memo (`docs/plans/`) and the findings memo (`docs/superpowers/plans/`) intentionally live in different directories. Design memos document multi-phase plans; findings memos document per-phase outcomes. The Phase 1 findings memo is also mirrored to `memory/` for cross-session persistence.
