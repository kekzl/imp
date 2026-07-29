---
name: codebase-audit
description: Use when auditing the imp codebase for structural debt, dead code, god-objects/files, duplication, flag sprawl, or deciding whether a cleanup is worth shipping — "structure audit", "tech debt", "is this still used", "dead code", "refactor for clarity", "should we split this file", "remove this flag". Do NOT use for build/test mechanics (building-and-testing), perf/kernel work (benchmark-cuda / sm120-cuda-expert), or output quality (check-degeneration).
---

# Codebase audit & structural debt — imp

## The one hard rule

**Never act on a raw finding. Verify every finding against the real code first.**
Fan-out audits (Explore agents, grep sweeps) systematically **over-flag**. On the
2026-06-09 pass, the big-ticket findings C1/C2/C4 were all REFUTED on inspection,
and several "dead flags/files" were live. Treat a finding as a *candidate*, not a
fact, until a targeted check confirms it. Most of an audit's value is in the
verification, not the sweep.

## Verification recipes (the load-bearing checks)

| Claim | How to actually verify | Trap that fooled a past audit |
|---|---|---|
| "flag/symbol X is dead" | grep across **`src/ tests/ tools/`** — `tools/` IS in scope | `dump_tokens`/`force_bos`/`bench.generate` looked dead in `src/`, but are read in `tools/imp-cli/main.cpp` |
| "config field never read" | grep the leaf field for a **value-read in a conditional**, excluding decl/parse/copy | a local var sharing the field name hides the read (`fp8_auto_legacy`) |
| "kernel/function is dead" | trace the **call graph**, not just `<<<` sites — reachable from a live dispatch = live | `q4k_imma` *cache* was dead, but `mmq_q4k_imma_tile`/`_reorder` are LIVE via `q4k_imma_prefill → mmq_q4k_imma_gemm` (called on the fly). Only the ~30-LOC cache struct was dead, not the "~1200 LOC stack" |
| "this file is a god-file, split it" | **split on conflation, not size**; check if the shared part is already factored | `gdn.cu`/`gemm.cu`/`sampling.cu`/`json_schema.cpp` are each one cohesive domain. The 6 `attention_paged*` variants already share `attention_paged_common.cuh` (online-softmax loop) — only dequant differs |
| "rewrite this into a registry/template" | **no speculative abstraction** — need a concrete bug, not a smell | the `weight_map.cpp` `if (!matched)` ladder works and is readable; a mis-mapped tensor name = garbage output. Risk > cosmetic gain |
| "imp.conf key is inert" | is it read in `src/` AND `tools/`? Is the live path the **C-API** instead? | `server.prefix_cache` etc. were parsed but unread — live enablement bridges at engine init (PR #636) |

## Workflow

1. **Fan out** — Explore agents / grep to surface candidates (breadth is fine here).
2. **Verify each** with the recipe above. Demote refuted ones immediately.
3. **Write it up** — `docs/audit/structural_debt_YYYY_MM_DD.md`. Counts are evidence,
   not estimates. Keep a **"Refuted (do not re-chase)"** section so the next pass
   doesn't re-flag the same false positives. Exception: the memory subsystem has a
   running findings log at repo root, **`AUDIT.md`** (CONFIRMED / REFUTED / OPEN per
   finding, negative results included) — append there rather than opening a parallel
   dated report, and read it before sweeping `src/memory/`.
4. **Ship cleanups** as small PRs (REQUIRED SUB-SKILL: building-and-testing) — branch
   off `main`, never stack, Docker `make build` + `make test-unit`, prefer batching
   related nits. Behavior-sensitive removals: also a coherence check (check-degeneration).

## File-size gate (god-file findings go through this, not ad-hoc splits)

`tools/check_filesize.py` (config: `tools/filesize_thresholds.toml`) measures **code LOC** (comments/blanks stripped) per category — kernel `.cu` warn>500/hard>600, normal TU warn>600/hard>800, header warn>500/hard>700. CI job `File size`: advisory warn step + **blocking** hard step. The real cost metric is *recompile blast radius* (one `.cu` = one `ptxas` TU), not line count — split on **compile-time isolation** (kernel def / host wrapper / explicit instantiations), never on size alone. A legitimately monolithic file goes into `[allow]` in the toml **with a reason** (empty reason is rejected). Baseline + per-file rationale: `docs/audit/AUDIT_FILESIZE.md`.

## Allocation-site gate (invariant I1: one module talks to the driver)

`tools/check_alloc_sites.py` counts direct `cudaMalloc*`/`cudaFree*`/`cudaHostAlloc` sites outside `src/memory/` and diffs them against `tools/alloc_allowlist.txt`. CI job `Alloc sites`: advisory `--stats` step + **blocking** allowlist gate — it fails on a new direct site *and* on a stale entry, so removing a site means updating the allowlist in the same PR. Note what it does and does not measure: it counts **source sites**, not executed allocations. A change can remove ~96% of runtime allocation traffic and leave the site count flat (the T2 slot pool did exactly that, keeping the old path as a fallback — `AUDIT.md` B34). For the runtime number, build with `-DIMP_ALLOC_INTERPOSE=ON` and read `steady_state_allocations()`; never benchmark that build (`AUDIT.md` G16).

## Priors — settled, do NOT re-flag

From the structural-debt audit chain (D1–D4 / C1–C8, archived in `docs/archive/README.md`;
current verdicts in `docs/audit/housekeeping_2026_06_13.md`):
- **exec/ god-object** (`GraphExecutor`) is intrinsically forward-pass-coupled; the
  D2 runner-extraction track ENDED (cosmetic friend-backdoor only). Header split done
  (1188→584 lines). Don't re-attempt runner classes.
- **ModelProfile (D1)** centralized arch facts — don't reintroduce scattered `cfg.arch==`.
- `gdn.cu` is domain-cohesive ("don't touch"); the compute/ files are not god-files.
- The two-config-systems split (`RuntimeConfig` global vs `ModelConfig::Overrides`) is
  known and deliberate; config keys use the typed-binder `B/I/F/S(...)` ladder (PR #626).
- **Audit #5 (2026-07-07, `docs/audit/structural_debt_2026_07_07.md`)**: server layer is
  the dominant debt source → issues #888–#897 filed. Before a new pass, read that
  report's **NOT-flagged list** — re-flagging its verified negatives wastes the sweep.
  Companion: `docs/audit/vram_audit_2026_07_07.md` (note: a `VramOwned` type does NOT
  exist — past audits hallucinated it).
- **Memory-subsystem priors (2026-07-29, `AUDIT.md`)**: the "engine teardowns leak
  ~15 GiB" finding is **REFUTED as a leak** — every CUDA-level release works
  (`mempool trim: reserved 8320->0 used 0->0`), but WSL2/WDDM never returns a
  process's peak VRAM commitment, so free VRAM only ever decreases within a process.
  Do not re-flag it, and do not "fix" it in the allocator. Two measurement traps that
  come with it: a successful `cudaMalloc` proves nothing about free VRAM here (28 GiB
  succeeds with 22.6 GiB reported free — time it instead, ~1530 vs ~237 GB/s), and
  `cudaMemPoolAttrReservedMemCurrent` drops to 0 on a trim even when nothing was
  returned, so retention checks must assert on `UsedMemCurrent`.

## Red flags — you are about to over-flag

- About to delete code because a sweep said "no references" → did you grep `tools/`?
- About to split a file because it's big → is it *conflated* or just one large domain?
- About to rewrite a working ladder/loader for elegance → where's the concrete bug?
- Reporting LOC of "dead" kernels → did you trace the call graph, or just `<<<`?
