# imp Agent Skills — Index

Project-scoped skills for agentic work on imp (`.claude/skills/*/SKILL.md`). Each
description states when to fire AND when not to — keep that property when editing.

| Skill | Covers | Pairs with |
|---|---|---|
| [building-and-testing](building-and-testing/SKILL.md) | Docker build, test suite, verify gates, CI reality (no GPU runner), determinism/PPL caveats, PR conventions | benchmark-cuda (perf), check-degeneration (quality) |
| [benchmark-cuda](benchmark-cuda/SKILL.md) | Benchmarking & profiling (cudaEvent/ncu/nsys/roofline), measurement artifacts on this box (clock ramp, host drift), baseline refresh + publishing numbers | sm120-cuda-expert |
| [sm120-cuda-expert](sm120-cuda-expert/SKILL.md) | Writing/optimizing CUDA kernels for sm_120a; PTX templates + dead-ends ledgers in `references/` | benchmark-cuda, check-degeneration |
| [check-degeneration](check-degeneration/SKILL.md) | Output-coherence battery after hot-path changes (degen_suite, GTest battery, graphs parity) | — |
| [server-api](server-api/SKILL.md) | imp-server endpoints (OpenAI + Anthropic), streaming, json_schema, tool calling, cache_control, validation tools | check-degeneration |
| [add-model-arch](add-model-arch/SKILL.md) | New-architecture integration checklist + wrong-output diagnostic fingerprints | quant-formats, check-degeneration |
| [quant-formats](quant-formats/SKILL.md) | GGUF/NVFP4/FP8 formats, StorageTier dispatch contract, decode cache, KV dtypes | sm120-cuda-expert |
| [codebase-audit](codebase-audit/SKILL.md) | Structural-debt / dead-code / god-file / flag audits + the verification discipline that stops fan-out over-flagging; `docs/audit/` convention | building-and-testing, check-degeneration |
| [docs-sync](docs-sync/SKILL.md) | Keeping architecture.md / README / GOAL.md / supported-models.md / imp.conf.example / CHANGELOG coherent after a change; English-only rule | benchmark-cuda (perf), codebase-audit |
| [shipping-prs](shipping-prs/SKILL.md) | PR/merge/release mechanics — branch off main, no stacking, squash + auto-merge race (`Build` required check, ruleset 14716423), version bump + CHANGELOG + tag flow | building-and-testing, docs-sync (CHANGELOG prose) |

Boundaries (to avoid trigger collisions):

- *Measure* perf → benchmark-cuda · *write* the kernel → sm120-cuda-expert · *is the output still sane* → check-degeneration.
- *Build/test mechanics & CI* → building-and-testing · *model output via HTTP* → server-api · *model loads but is wrong* → add-model-arch · *bytes/scales/tiers* → quant-formats.
- *Is this code dead / should we refactor* → codebase-audit · *keep the docs consistent with the code* → docs-sync. Perf-number measurement + `perf_baseline.json` refresh stay with benchmark-cuda (docs-sync only reconciles the surrounding prose).
- *Open/merge/release a PR, auto-merge, tag a version* → shipping-prs · *build/test mechanics behind that PR* → building-and-testing (it cross-refs shipping-prs for the merge flow).

Audit history: [AUDIT_skills_2026_06_07.md](AUDIT_skills_2026_06_07.md). Content refreshes:
2026-06-10 (post-audit sprint, PRs #608–#651); 2026-07-09 (PRs #652–#939 — single-sourced
dep pins in `cmake/imp-deps.cmake`, C++23/Ubuntu 26.04 toolchain, auto-armed auto-merge
workflow, spec-ngram default-on bench confound, FA2 hd=256 arc + FP8-tile attention,
thinking-state reconcile, MXFP4/VRAM-reserve lessons, file-size gate). Descriptions/triggers
unchanged.
