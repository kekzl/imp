# Structural debt audit — 2026-06-08

Scope: codebase-wide structural debt beyond the VRAM-cache layer (which was
rebuilt in PR #621, see `vram_cache_structure_2026_06_07.md`). Verified against
the tree with grep/wc; counts are evidence, not estimates. Diagnostic only — no
code changed. Ranked by pain × risk × how-well-bounded-the-fix-is.

## TL;DR

The VRAM cache was one instance of a recurring shape: **a decision recomputed in
many places instead of decided once and read.** The same shape appears in three
more places. The highest-value next target is the same move we just made for
tiers — a single **ModelProfile** for architecture facts. The rest is a god-object
split (big, risky) and a dead-code/flag sweep (cheap, do anytime).

---

## D1 — No central `ModelProfile` (highest priority)

Architecture-derived facts are recomputed inline across the codebase instead of
classified once into a struct the rest of the code reads. This is **the same class
of bug** that the rejected FP16-gate diff hit and that caused #514/#516 (a
scattered `if (arch==…)` one path forgets).

**Evidence:**
- **GDN/SSM-hybrid detection** — "loop the layers, check `gdn_gate.data`/
  `ssm_in.data != nullptr`" is recomputed at ≥6 sites:
  `engine_init_resolver.cpp:128`, `engine_kv_cache_init.cpp:155/174/210`,
  `engine_weight_upload.cpp:142`, `vram_budget.cpp:60`. Each decides something
  (SSM state dtype, FP8 disable, VRAM footprint, graph compatibility) off the
  same fact, independently.
- **Hot-path arch branching** — per-forward/per-layer `cfg.arch == ModelArch::*`:
  `executor_forward_moe.cu` (15), `executor_attention.cu` (15),
  `executor_forward_moe_batch.cu` (7). These pick FFN-norm variant, attn scale,
  SWA routing, expert-offset layout *every layer* instead of from a pre-decided
  enum (e.g. `AttentionVariant`, `FfnNormVariant`).
- **RoPE / FP8-eligibility / CUDA-graph-eligibility** — each decided in one place
  but then re-read from `model_config_` downstream instead of from a profile.

**Nuance (not all arch-checks are debt):** `model.cpp` (72) and
`hf_config_loader.cpp` (44) are *load-time classification* — the correct place,
run once. The debt is the **hot-path re-derivation**, not the loader.

**What exists vs what's missing:** `ModelConfig` (static metadata) + `EngineConfig`
(decided runtime flags) + a global `process_diag` singleton (holds
`cublas_fp16_acc` — bad pattern). **Missing:** a `ModelProfile` that computes the
*derived* facts (is_gdn, is_moe, is_hybrid, attn_variant, ffn_norm_variant,
rope_variant, fp8_eligible, graphs_eligible) once at init and is read everywhere.

**Why it's the right next target:** real bug history (#514/#516), exactly the
"scattered → one truth" move just proven on tiers, and well-bounded (one struct,
filled once, call-sites migrated individually behind the existing canaries:
coherence across dense/MoE/GGUF/gemma, verify-fast).

---

## D2 — `GraphExecutor` god-object

`src/exec/executor.h` declares `GraphExecutor` with **53 methods + 69 data members
+ 10 supporting structs** in one ~1160-line header. It owns: weight caches, KV
cache, MoE routing + expert LRU, attention, FFN, SSM, sampling, RoPE (3 variants),
LoRA, workspace allocation, and the whole pre-dequant quantization pipeline.

**Why it's hard to work with:** no component is testable in isolation (KV
calibration needs the whole workspace + model); adding a quant strategy touches
the executor core; the forward DAG hides behind `init()`/`forward()`/`*_workspace()`.

**Why it's NOT first:** large blast radius, many PRs, needs its own design pass.
Best done *after* D1 (a ModelProfile removes a chunk of the executor's arch
branching, shrinking the split surface).

---

## D3 — Oversized files / god-functions

| File | Lines | Issue |
|------|-------|-------|
| `pre_dequant_phase3_nvfp4_decode.cu` | 1906 | 5 sub-phases sharing `Nvfp4DecodeContext`; a **329-line lambda** (`cache_moe_native_nvfp4`, lines ~1528–1856) capturing 20+ vars |
| `executor_kernels.cu` | 1932 | conflates dp4a GEMV dispatch + attention utils (RoPE, KV index) + sampling preprocessing |
| `weight_upload.cu` | 2174 | qtype byte-pattern detection + GPU alloc + per-format decompress |

**Not debt (don't touch):** `jinja.cpp` (2629) is a self-contained lexer→parser→
evaluator, single responsibility; `gdn.cu` (2061) is domain-cohesive. Size alone
isn't the smell — *conflated responsibilities* are.

---

## D4 — Flag sprawl + dead/half-migrated paths (cheap sweep, do anytime)

`RuntimeConfig` has 100+ fields; `gemm` alone has 18. Confirmed leftovers:
- **Dead/retired flags:** `diagnostics.tq_skip_qjl` (comment: "TurboQuant retired
  2026-05-17"); `kv_cache.fp8_auto_legacy` (env-var compat shim).
- **`gemm.q4k_imma_enabled`** — suspected dead *gate*: grep finds it only in
  comments + the kernel file's own comment claiming "the dispatch site checks
  it", but the actual prefill dispatch (`executor_kernels.cu:1875`) checks the
  *separate* `q4k_imma_prefill`. **Verify before removing** (the ~1200 lines of
  q4k_imma kernel infra are NOT all dead — `q4k_imma_prefill` is live; only the
  `_enabled` gate looks orphaned).
- **Two config systems** half-separated: `RuntimeConfig.gemm` (global) vs
  `ModelConfig::Overrides::Gemma4` (per-model). config.h:352 itself notes
  "model-specific knobs do not belong on a global runtime singleton" while Q4_K
  flags remain global.
- **Overlapping pairs:** `nvfp4_lm_head` + `nvfp4_lm_head_gdn`; the q4k_imma gate
  pair above.

Low risk, immediately visible, and shrinks the surface for D1/D2. Good filler
between the bigger pieces.

---

## Recommended order

1. **D1 ModelProfile** — highest pain×risk, well-bounded, same proven move.
2. **D4 dead-code/flag sweep** — cheap, do opportunistically (even before/between).
3. **D2 GraphExecutor split** — biggest structural win, highest risk; own design
   round, after D1 thins its arch branching.
4. **D3 god-functions** — extract the 329-line MoE lambda + split
   executor_kernels by domain; can ride along with D1/D2 where they overlap.

Each follows the same discipline as the VRAM rebuild: one decision moved to one
place, call-sites migrated individually, gated by coherence (dense/MoE/GGUF/
gemma-3) + `EngineRelaunchTest` + `verify-fast`, strictly behaviour-neutral.

---

## Progress

- **D1 — DONE** (PR #622 init-path classification + PR #623 hot-path arch
  identity). All 55 hot-path `cfg.arch == ModelArch::X` reads + the ≥6
  GDN/SSM/MoE detection loops now read one `ModelProfile`. The scaffolded
  `attn_variant`/eligibility fields were dropped, not wired: the real code keys
  off arch identity + per-layer tensor presence.
- **D4 — partial** (this PR): removed three confirmed-dead flags —
  `gemm.no_dp4a` (split long ago into `no_dp4a_gemv`/`no_dp4a_lm`),
  `gemm.q4k_imma_enabled` (the orphaned gate this audit flagged: parsed +
  mirrored into `GemmContext` but never read in any dispatch), and
  `diagnostics.debug_gemm_dispatch` (declared + parsed, never read). Misleading
  comments referencing the removed flag fixed.

### New findings (during the D4 sweep)

- **Dead `WeightCaches::q4k_imma` cache** — the OLD Phase-2C Q4_K IMMA cache
  (`Q4kImmaCacheEntry` map, `use_q4k_imma`, `mmq_q4k_imma_tile`/`_reorder`) is
  only *declared + freed*, never populated (its load-time gate was the now-dead
  `q4k_imma_enabled`). Removing the ~1200-line tile/reorder stack is a D3-class
  change (kernel infra), deferred — left an INACTIVE note on the cache decl.
- **Unbridged `server.*` / `paths.*` config keys** — `server.prefix_cache`,
  `server.green_contexts`, `paths.mmproj` are parsed into `RuntimeConfig` but
  never read; the live enablement flows through the C-API
  (`use_prefix_caching` / `enable_green_contexts` / CLI `--mmproj`). This is the
  D4 "two config systems half-separated" item — these keys are silently inert
  from `imp.conf`. NOT removed in this sweep: they are user-facing surface that
  looks like incomplete wiring, not internal dead code — needs a decision to
  either bridge them or retire them (own change).
