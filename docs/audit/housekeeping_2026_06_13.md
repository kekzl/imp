# Housekeeping Sprint — 2026-06-13

Cleanup-only pass: dead/unreachable code, documentation reconciliation against
ground truth, comment/note hygiene. **Behavior-preserving only.**

Branch: `housekeeping/docs-truth-pass` (off `main`).

---

## Executive summary

This codebase was already audited for structural debt and dead code **three times**
(`docs/audit/structural_consistency_2026_06_06.md`,
`structural_debt_2026_06_08.md`, `structural_debt_2026_06_09.md`). The dead-code
seam was largely mined in PR #635 (5 dead kernels, 116 LOC) and the structural
big-ticket findings (C1/C2/C4) were **refuted on verification**. A fresh full-tree
sweep here confirms that: there is no large pool of removable dead code left.

The actionable surface this pass found is **documentation/comment truth**, not dead
code. The single material finding is a kernel banner that claimed a **WGMMA**
implementation that does not (and cannot) exist on `sm_120a` — contradicting both
ground truth and the file's own header. Fixed, plus two smaller stale arch claims.

**No code was deleted.** Every change is a comment or an informational log string;
codegen and dispatch are unchanged.

### ⚠️ Conflict: mission "Ground Truth" benchmarks are *older* than the repo

The mission's Ground Truth cites May-2026 figures (`~0.65–0.72× vLLM`,
`pp512/2048/4096 ≈ 16.5k/17.2k/18.2k`, `tg256 ≈ 290–295`). The repo's benchmark
docs (`README.md`, `BENCHMARKS.md`) were **refreshed later** (PR #676, 2026-06-12)
with freshly measured numbers in which imp now **wins** MoE pp2048 vs vLLM (+21%)
and TTFT broadly. Overwriting the current numbers with the mission's older snapshot
would *regress* the docs — the opposite of housekeeping. **I did not change current
benchmark numbers.** The only genuinely stale benchmark assertions the mission names
— `1258 tok/s` and `20× behind vLLM` — are already refuted everywhere they appear in
the repo (see Phase 1). No live doc asserts them as current; nothing to fix there.

---

## Phase 0 — Inventory

- Tree: ~166k LOC across `src/ include/ tools/ tests/`
  (`*.cpp/*.cu/*.cuh/*.h/*.hpp`).
- Build: `make build` → `docker build -t imp:test .` (CUDA 13.3, raw-gencode
  `sm_120a` + `compute_120f` PTX fallback). Tests: `make test-unit` (host),
  `make test-gpu` (RTX 5090). Docker 29.5.3 present; host has no CUDA toolkit by
  design. `build/` is root-owned and empty.
- Candidate sources swept: arch `#ifdef`/comment markers
  (`wgmma|hopper|sm_90a|sm_100|b200|tcgen05|tmem|fa4`), stale benchmark numbers
  (`1258|20x`), TODO/FIXME/XXX, unbridged config fields, dead include paths,
  prior-audit "candidate dead kernel" residue.

### Candidate classification

| Item | Tag | Note |
|---|---|---|
| `attention_fmha_sm120.cu` banner: "Native WGMMA Flash Attention", "WGMMA for both QK^T and PV GEMMs", "compiler can fuse WMMA into WGMMA on sm_120" | **DEAD (doc)** | False capability claim; file actually uses WMMA/`mma.sync` per its own `.h`. **Fixed.** |
| `attention_paged_common.cuh:319` "Used by both Hopper and Blackwell WMMA prefill kernels" | **DEAD (doc)** | No Hopper kernels exist in this single-target tree. **Fixed.** |
| `executor_workspace_buffers.cu:274` log "WMMA/TCGEN05 fallback" | **DEAD (doc)** | `tcgen05` does not exist on `sm_120`; real fallback is tiled WMMA FMHA. **Fixed.** |
| `attention_tc.h:9` "Requires sm_90+ (Hopper/Blackwell). Falls back to scalar on older GPUs." | **FALLBACK** | `tc_attention_available()` (`attention_tc.cu:407`) is a *live* runtime capability gate; `sm_120 ≥ sm_90` so the statement is accurate. Kept. |
| `attention_fmha_sm120.cu:579` "compiles cleanly for sm_90/sm_100 too" | **FALLBACK** | Explains a `__CUDA_ARCH__ >= 1200` *compile* guard, not a runtime claim — true and load-bearing for host-side compilation. Kept. |
| sm_120f PTX fallback path | **FALLBACK** | Live fallback for non-5090 SKUs. Kept. |
| legacy materialized causal-softmax + cuBLAS attention path | **FALLBACK** | ~18% prefill coverage (hd≠128, per-layer, sinks); reached at runtime. Kept. |
| `CMakeLists.txt:341` `-I .../examples/88_hopper_fmha` | **UNCERTAIN** | No `#include` in `src/include` resolves through it (grep-clean), so it *looks* like a dead include path. But proving a `-I` truly unused requires ruling out transitive CUTLASS includes; removing it is low value and could break the build. Flagged, not touched. |
| `generation.think_budget` global config field (`config.h:388`, parsed `config.cpp:208`) | **UNCERTAIN** | Already flagged by the 2026-06-09 audit: parsed but no `cfg.generation.think_budget` value-read; the live path is per-request `req.think_budget`. Bridge-or-retire is a user decision, not a safe delete. Kept. |
| `softmax_sum_kernel`, `softmax_to_pairs_kernel`, etc. (2026-06-09 C8 list) | **(already removed)** | Removed in PR #635. Confirmed gone. |

---

## Phase 1 — Documentation truth-pass

### Stale benchmark numbers (`1258`, `20×`)

Every hit is in **refutation context** — the docs already state these numbers are
wrong:

- `GOAL.md:86` — "the '20× grouped-GEMM gap' premise is refuted".
- `docs/audit/performance_agent_readiness_2026_05_31.md:13–17,39,208` — "The central
  audit premise is outdated… '1258' value… conclusively refuted".
- `CHANGELOG.md:694` — historical record ("CUTLASS 3.x… scaffold (#22) — path for
  sm_100+"); changelog is an append-only archive, left untouched.

**No edit needed** — no live doc asserts 1258/20× as a current figure.

### Arch-capability markers

The vast majority of `wgmma|hopper|sm_100|tcgen05|tmem` hits are **legitimate
negative statements** ("no `tcgen05`/TMEM/`wgmma` — those are datacenter Blackwell
only") in `README.md`, `GOAL.md`, `docs/sm120.md`, `Dockerfile`, the `ptx-status-*`
survey docs, and `tools/analysis/*.sh`. Per the mission these **must stay** and were
left intact.

Three hits were genuine false-capability assertions and were corrected (see Phase 3).

---

## Phase 2 — Dead code removal

**None removed.** The candidate scan surfaced no provably-dead code that wasn't
already removed by prior audits (PR #635). The two arguable items
(`88_hopper_fmha` include path, `generation.think_budget`) are flagged UNCERTAIN
above and deliberately left per the "never delete a path you can't prove
unreachable" guardrail.

---

## Phase 3 — Comment & note hygiene

### Edits (all behavior-preserving — comment / log-string text only)

1. **`src/compute/attention_fmha_sm120.cu`** (banner, lines 1–16; tile comment
   ~57–60) — rewrote the "Native WGMMA Flash Attention … WGMMA for both QK^T and
   PV GEMMs … 2 warp groups of 128 … compiler can fuse WMMA into WGMMA on sm_120"
   block to accurately describe the WMMA/`mma.sync` HMMA implementation (matching
   the authoritative `.h`). The old banner also self-contradicted ("no shared
   memory materialization of S" while listing `S_tile` in the smem layout). Thread
   org corrected to the actual "256 threads (8 warps of 32)" (`SM120_NUM_WARPS=8`).

2. **`src/compute/attention_paged_common.cuh:319`** — "Used by both Hopper and
   Blackwell WMMA prefill kernels" → "Used by the sm_120 WMMA prefill kernels"
   (no Hopper kernels exist in this tree).

3. **`src/exec/executor_workspace_buffers.cu:274`** — log string
   "using WMMA/TCGEN05 fallback" → "using tiled WMMA FMHA fallback" (`tcgen05`
   is not present on `sm_120`).

### TODO/FIXME (kept — all live)

Only 5 in `src/`+`include/`, all legitimate and current:
`json_schema.{h,cpp}` (NFA→recursive-grammar backend plan), `sampling.cu` ×2
(determinism caveats), `gemm_grouped.cu:147` (true grouped/batched GEMM idea).
None reference shipped/abandoned work. No commented-out code blocks found in `src/`.

---

## Phase 4 — Verify

- Build: `make build` (Docker, CUDA 13.3, `sm_120a` raw-gencode) — **PASS** (exit 0).
- Tests: `make test-unit` (host) — **PASS**, 37/37 across 5 suites (exit 0).
- `make test-gpu` not run: requires the RTX 5090 and is unaffected here — the only
  changes are comment text and one informational log string, which cannot alter
  codegen, dispatch, or output. No behavioral diff.
- No pre-existing failures to attribute: build clean, host suite green.

---

## Phase 5 — Summary

- **LOC delta:** 0 lines of code removed; comment/string text only (net ~−2 lines).
- **Removed items:** none (dead code already mined by prior audits / PR #635).
- **FALLBACK kept:** sm_120f PTX fallback; legacy cuBLAS+causal-softmax attention
  (~18% prefill coverage); `tc_attention_available()` capability gate;
  `attention_fmha_sm120.cu` host-compile guard.
- **UNCERTAIN flagged:** `88_hopper_fmha` `-I` path; `generation.think_budget`
  global field — both for human bridge-or-retire decisions, not blind deletion.
- **Doc corrections (old → new):** 3 stale arch-capability comments/log-strings
  (WGMMA banner; "Hopper and Blackwell"; "WMMA/TCGEN05"). No benchmark numbers
  changed — the repo's are newer than the mission's snapshot, and the named stale
  figures (1258/20×) are already refuted in-repo.
- **Commit list:**
  - `e72c9146` housekeeping: fix stale arch-capability comments (no WGMMA/tcgen05 on sm_120a)
  - _(this commit)_ housekeeping: add sprint report (dead-code/doc/comment pass)
