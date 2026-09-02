# imp Agent Skills - Index

Project-scoped skills for agentic work on imp (`.claude/skills/*/SKILL.md`). Each description states when to fire AND when not to; keep that property when editing. Form rule for every skill: fact, number, path, decision; tables over paragraphs; no em dashes; a count or runtime only next to the command that prints it.

| Skill | Covers | Pairs with |
|---|---|---|
| [building-and-testing](building-and-testing/SKILL.md) | Docker build, test binaries and lanes, the gate groups and hooks, CI reality (no GPU runner), stale-object and container-env traps, determinism/PPL caveats | benchmark-cuda (perf), check-degeneration (quality), shipping-prs (merge flow) |
| [benchmark-cuda](benchmark-cuda/SKILL.md) | The 13 STOP facts of this box, methodology, harness table (`tools/analysis/*`), ncu/nsys recipes and traps, roofline cells, aggregate-throughput method, baseline refresh, verdict expiry | sm120-cuda-expert |
| [sm120-cuda-expert](sm120-cuda-expert/SKILL.md) | sm_120a laws, numerics rules for tensor-core rewrites, chunk-parallel GDN scan ledger, paths that must stay active, closed classes; PTX templates + dead-end ledger in `references/` | benchmark-cuda, check-degeneration |
| [check-degeneration](check-degeneration/SKILL.md) | Failure-mode table, server suite, GTest equivalence gates, parity arms (graphs, ragged, chunkpar, sparse), NIAH, PPL judge rules | server-api |
| [server-api](server-api/SKILL.md) | Source map, flags, endpoints, semantics (thinking, constraints, prefix cache, speculation, long context, priority, request ids, OTLP tracing, model swap), validation, fingerprints | check-degeneration |
| [add-model-arch](add-model-arch/SKILL.md) | "Is it a new arch" check, integration checklist, wrong-output fingerprints, VRAM arithmetic | quant-formats, check-degeneration |
| [quant-formats](quant-formats/SKILL.md) | GGUF vs NVFP4 worlds, StorageTier contract, KV dtypes (NVFP4 default on Qwen3.5 family), quality judging, the two NVFP4 layouts, imp-quantize | sm120-cuda-expert |
| [code-graph](code-graph/SKILL.md) | `codegraph` commands, sync and control-symbol rules, the missing `launches` edges, three traps | codebase-audit, sm120-cuda-expert |
| [find-stubs](find-stubs/SKILL.md) | Six rungs with current baselines, gate-based features that silently no-op, accepted-but-ignored fields, tests that assert nothing | codebase-audit, code-graph |
| [codebase-audit](codebase-audit/SKILL.md) | Verification recipes, ledger-first workflow, the static gates that shape findings, settled priors | building-and-testing, check-degeneration |
| [docs-sync](docs-sync/SKILL.md) | Doc ownership table, what to sync per change class, env-var rule, roadmap ledger rows | benchmark-cuda, docs-layers, shipping-prs |
| [docs-layers](docs-layers/SKILL.md) | Reader layers L0-L3, metadata header, which numbers may appear, PROV, SSoT map, generated blocks, plan closure, `STALE.md` | docs-sync, shipping-prs |
| [shipping-prs](shipping-prs/SKILL.md) | Branch/merge/release mechanics, auto-merge race, PR body and CHANGELOG form, triage table, release cut and notes | building-and-testing, docs-sync |

Boundaries:

- Measure perf: benchmark-cuda. Write the kernel: sm120-cuda-expert. Is the output still sane: check-degeneration.
- Build/test/CI: building-and-testing. Model output via HTTP: server-api. Model loads but is wrong: add-model-arch. Bytes/scales/tiers/KV dtypes: quant-formats.
- Who calls this: code-graph (one query). Is it dead / worth refactoring: codebase-audit (the graph produces the candidate, the audit decides). Docs consistent with code: docs-sync; layer/lint/provenance: docs-layers.
- Open/merge/release: shipping-prs; the build behind it: building-and-testing.

## Audit and refresh history

| Date | Scope | Record |
|---|---|---|
| 2026-06-07 | audit of 3 skills, 4 created, README added | [AUDIT_skills_2026_06_07.md](AUDIT_skills_2026_06_07.md) |
| 2026-06-10 | post-audit sprint, PRs #608-#651 | descriptions unchanged |
| 2026-07-09 | PRs #652-#939: dep pins in `cmake/imp-deps.cmake`, C++23/Ubuntu 26.04, auto-armed auto-merge, spec-ngram bench confound, FA2 hd=256, thinking reconcile, file-size gate | descriptions unchanged |
| 2026-08-27 | full 13-skill audit vs PRs #1479-#1786 (#1787): blocking static gates, batched-decode regime, scheduler split, CHANGELOG cycle, ccg enrich breakage, aggregate methodology; docs-layers added | descriptions unchanged |
| 2026-08-31 | building-and-testing, docs-layers: hook filter, numbers rule 3b (#1825, #1827, #1828) | |
| 2026-09-02 | all 13 skills rewritten to the no-prose form (every paragraph carries a number, path or decision), refreshed vs PRs #1787-#1856: mtp auto default + bench pins, sparse decode attention, NVFP4 KV default and word loads, chunk-parallel GDN scan (3xTF32/3xFP16 rules), FA2 2-CTA + softmax, stream-K, PDL device half, OTLP tracing, priority/X-Request-Id, recurrent snapshot host tier, container `IMP_SET`, `entrypoint`/`kernels` gate groups, stale-object rebuild trap; find-stubs baselines re-measured at `c3d9689e`; every cited path, config key, make target and flag checked against the tree (`pathcheck.py`, 0 real misses); anchor diff old vs new per skill (0 facts dropped) | this table |
