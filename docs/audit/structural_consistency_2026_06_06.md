# Structural Consistency Audit — 2026-06-06

Cross-level consistency audit over five layers: **docs ↔ code**, **build/CI**, **source structure**,
**tests**, and the **config surface**. Five independent audit passes (one per layer), with all
high-impact findings re-verified by hand against the tree at `main` (post #543).

**Verdict: structurally sound.** No high-severity inconsistencies. The architecture boundaries the
project claims (single exception-translation point, centralized config, downward-only module
dependencies, single-sourced CI toolchain version) all hold under inspection. Findings below are
drift items, mostly documentation lagging code.

---

## 1. Verified findings

### M1 — `imp.conf.example` is missing ~35 `RuntimeConfig` keys, including flipped defaults (MEDIUM)

Spot-verified absent from the example (present in `src/runtime/config.h`):

| Key | Code default | Why it matters |
|---|---|---|
| `attention.fmha_fa2` | on | FA2 default flipped on in PR #478 |
| `gemm.nvfp4_lm_head` / `nvfp4_lm_head_gdn` | true | default-on since #483, +2.2% PPL trade-off |
| `runtime.prefill_graph` | true | flipped 2026-05-17 |
| `runtime.graph_capture_mode` | "relaxed" | flipped from "global" |
| `gdn.chunkwise_scan` | true | +16.7% perf flip |
| `server.green_contexts` | — | undocumented |

The entire `[ffn]` section (`sparsity_probe`, `sparsity_threshold`) is absent — example has 11 of the
12 code sections. Full missing-key list spans `attention.{attn_scores_mib, fa2_fp16qk, fmha_fa2,
force_cublas_decode, gate_concat, no_qknorm_fused, splitk_pipe}`, `diagnostics.*` (11 keys),
`gemm.{nvfp4_attn_proj, nvfp4_ssm_proj, nvfp4_moe_decode, q4k_imma_enabled, q4k_hmma_enabled, …}`,
`kv_cache.bitdecoding_*`, `moe.{mr_nr, nvfp4_smallM, prefetch_top_k, reserve_mib, …}`,
`runtime.{cuda_graphs, debug_raw, deterministic_gemm, max_batch_size, …}`, `bench.generate`.

Functionally harmless (parser warns on unknown keys; code defaults are authoritative), but users
cannot discover/override these from the example. Diagnostics keys arguably belong out of the
example on purpose — the perf-relevant flipped defaults do not.

### M2 — README pins CUTLASS v4.5.0; the build uses v4.5.1 (MEDIUM)

`README.md:32`, `README.md:116`, `README.md:163` say **v4.5.0**. `CMakeLists.txt:74` (`GIT_TAG
v4.5.1`), `Dockerfile:33` (`--branch v4.5.1`), and `docs/performance.md:8` say **v4.5.1** (bumped in
#447; README never followed).

### M3 — `docs/performance.md` is two baseline refreshes stale (MEDIUM)

`performance.md` says "Last refreshed: 2026-05-27" and "CUDA 13.2.1". Since then: full zoo re-bench
2026-05-30 (`BENCHMARKS.md`, commit `bebafd5`, decode +16–168% on NVFP4), CUDA 13.3 toolchain
(PR #485), and the Q8_0 baseline refresh 2026-06-05 (PR #540, tg128≈268). `BENCHMARKS.md` is
current; `performance.md` points readers at superseded numbers.

### L1 — CUDA version messaging is split 13.2-minimum / 13.3-canonical (LOW)

`CMakeLists.txt:15` requires `CUDAToolkit 13.2`; `CONTRIBUTING.md` says "13.2 or newer"; README
badge, CLAUDE.md and the Docker/CI images all say 13.3. Not wrong (13.3 ≥ 13.2), but no doc states
the actual policy: *minimum 13.2, canonical/tested 13.3*.

### L2 — `mtp_retest/` is untracked root clutter with no `.gitignore` entry (LOW)

`build-ciq/` and `core.1` are covered by existing patterns (`build-*/`, `core.*`); `mtp_retest/`
(~768 KB of local MTP retest artifacts) matches nothing. One-line `.gitignore` fix — or delete the
directory (the MTP diagnosis is closed as a dead end).

### L3 — CLI flag documentation gaps + one naming inconsistency (LOW)

Undocumented in `docs/usage.md`: `--perplexity` (the determinism-proof tool from #481),
`--revision`, `--mtp-spec-decode`, `--streaming-kv` / `--no-streaming-kv-auto` / `--stream-sinks` /
`--stream-window`, `--dual-path-quant`, `--no-fp8-prefill`. Deprecated aliases
(`--kv-turboquant*`) are correctly undocumented. Naming drift: imp-cli takes `--prefix-caching`
(`tools/imp-cli/args.cpp:202`) while imp-server takes `--prefix-cache`
(`tools/imp-server/args.cpp:128`).

### L4 — `.env.example` covers 3 of the 14 compose-consumed `IMP_*` variables (LOW)

`docker-compose.yml` wires 14 `IMP_*` env vars through to the entrypoint; `.env.example` documents
only `IMP_MODEL`, `IMP_MODELS_DIR`, `IMP_PORT`.

### L5 — Cosmetic doc/code drift (LOW)

- `docs/architecture.md:56` names the pre-dequant phase TUs without their `pre_dequant_` filename prefix.
- LOC claims disagree: README "~97k", CLAUDE.md "~93k" (different scopes, both estimates).
- `src/compute/gemm_grouped_nvfp4_smallM.{h,cu}` is the lone camelCase filename in a snake_case tree (intentional: smallM threshold).
- Architecture diagrams (`.dot/.svg/.png`) mutually consistent (all 2026-05-22) but predate the 05-27 `architecture.md` edits.

---

## 2. Refuted during verification

> **"docker-compose `IMP_*` env vars are NOT implemented"** — one audit pass flagged this CRITICAL
> because `imp-server`'s arg parser has no `getenv()` fallback. **Wrong:**
> `docker-entrypoint.sh:24–115` translates all 14 compose-wired `IMP_*` vars to CLI flags, the
> Dockerfile sets it as `ENTRYPOINT` (line 115), and the imp-server compose service has no
> `command:` override (the nearby `command:` belongs to the prometheus service). The env path works
> end-to-end.

---

## 3. Clean under inspection

**Source structure**
- C-API boundary: exception→`ImpError` translation exists *only* in `src/api/imp_api.cpp` (11
  try/catch sites covering every exported function). No leaks, no out-of-band `ImpError` construction.
- Module dependency direction is strictly downward (compute/quant/memory/exec → runtime); no cycles found.
- No ad-hoc env reads: `getenv` only in `src/runtime/config.cpp` (seed_from_env), `src/model/hf_hub.cpp`
  (HOME/HF caches), `src/runtime/engine_init_resolver.cpp` (CUBLAS_WORKSPACE_CONFIG).
- 100% `#pragma once`; no orphaned `.cpp`/`.cu` (every implementation file is in `CMakeLists.txt`).

**Build / CI**
- All documented Make targets exist and resolve to real scripts/CMake targets; all scripts referenced
  by Makefile/CI exist.
- CUDA toolchain single-sourced in CI (`CUDA_VERSION` env drives image check, nvcc verify, cache keys);
  job name `Build` matches the required-check ruleset.
- No `--mount=type=cache` anywhere (project ban honored).
- sm_120a raw-gencode workaround + `compute_120f` PTX fallback present (`CMakeLists.txt:28–39`).
- Tracked top-level dirs (`bench/`, `monitoring/`, `review/`, `profiles/`, `prompts/`) all referenced
  or self-explanatory; no orphans.

**Tests**
- 102/102 test files registered across the 8 test binaries; removed tests (turboquant) documented in CMake comments.
- Exactly 3 `DISABLED_` tests, all with documented root causes (2× DetEval cross-context — known
  GDN-hybrid layout-sensitivity boundary; 1× FMHA-MXFP4 HD=256 smem limit). No zombie disables.
- Ship-gates: `PrefixCacheE2ETest` active (model-gated via `IMP_TEST_MODEL`), `DetEvalE2ETest`
  same-context variants active.
- `perf_baseline.json` schema matches `gen_perf_baseline.sh` output and the verify.sh gate
  (3% decode / 5% prefill). Note: the server-side `test_perf_regression.py` lane measures TTFT/p50 —
  a separate metric system by design, not a schema clash.
- `GTEST_SKIP` + `IMP_TEST_MODEL*` conventions uniform (incl. the new `IMP_TEST_MODEL_QWEN4B` from #543).
- Coverage gaps (vision untested, attention cross-path numerics, server streaming pipeline) are
  *known and ranked* in `docs/TEST_AUDIT.md` — Phase-2 program shipped #1–#8/#10; the audit doc still matches reality.

**Config surface**
- Section names 3-way consistent (config.h ↔ imp.conf.example ↔ docs) for the 11 sections the example has.
- Parser warns on unknown keys (doesn't silently drop); loading precedence documented and correct.
- Spot-checked defaults match code (`prefix_cache = true` per #541, `deterministic = false`).
- All 28 legacy `IMP_*` mappings in `seed_from_env()` present and commented.

---

## 4. Recommended fixes (smallest first)

1. `.gitignore` += `mtp_retest/` (or delete the dir). — L2
2. README ×3: CUTLASS v4.5.0 → v4.5.1. — M2
3. `docs/performance.md`: refresh header (2026-06-05 baseline, CUDA 13.3) or stamp it
   "superseded by BENCHMARKS.md". — M3
4. `imp.conf.example`: add the missing perf-relevant keys (at minimum `fmha_fa2`, `fa2_fp16qk`,
   `nvfp4_lm_head*`, `prefill_graph`, `graph_capture_mode`, `chunkwise_scan`, `[ffn]`); decide
   explicitly whether `diagnostics.*` stays out. — M1
5. `docs/usage.md`: document `--perplexity` and the streaming-KV flag family; align
   `--prefix-caching`/`--prefix-cache` naming. — L3
6. One-line CUDA policy statement in CONTRIBUTING/README: "minimum 13.2, canonical 13.3". — L1
7. `.env.example`: list the remaining compose-consumed `IMP_*` vars. — L4
