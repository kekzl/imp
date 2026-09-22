<!--
layer: L3
audience: agents
verified: 2026-09-22
commit: 23c88fb8
-->

# Docs target structure

Phase 1 of the docs cleanup. Input: [`DOCS_INVENTORY.md`](DOCS_INVENTORY.md). Status: proposed, awaiting confirmation.

## Rules

| Rule | Value |
|---|---|
| `README.md` | <= 150 lines, L0 |
| `docs/*.md`, `docs/internals/*.md` | <= 300 lines each |
| Layer | frontmatter `layer:`, path-enforced by `scripts/docs_lint.py` (unchanged) |
| Fact removed or moved | one row in `DISPATCH_docs_v3.md` (file, section, where the fact went) |
| Dated measurement not current | moves to `docs/archive/` (record), not deleted |
| Untouched | `CHANGELOG.md`, `docs/MISSION_JOURNAL.md`, `docs/vram_audit.md`, `docs/roadmap.md`, `docs/audit/**`, `docs/plans/**`, `docs/archive/**`, `AUDIT.md`, `THIRD_PARTY_LICENSES.md`, `.claude/**` |

## Canonical owner per fact class

| Fact class | Owner | Everywhere else |
|---|---|---|
| Perf numbers (pinned gate, competitive) | `docs/PERF.md` | link |
| Per-model dated sweeps | `docs/BENCHMARKS.md` (latest sweep per section) | link |
| Feature status | `docs/FEATURES.md` | link |
| Supported checkpoints, quants | `docs/MODELS.md` | link |
| Limitations, untested paths | `docs/LIMITATIONS.md` | link |
| Deliberate absences | `docs/DESIGN_DECISIONS.md` | link |
| HTTP endpoints, fields, errors | `docs/API.md` | link |
| CLI flags, `imp.conf` keys, env vars | `docs/CONFIG.md` | link |
| Build, test commands | `CONTRIBUTING.md` | link |
| `sm_120a` has / lacks | `docs/internals/SM120.md` | link |

## Target tree

| Target | Layer | Budget | Source | Action |
|---|---|---|---|---|
| `README.md` | L0 | 150 | `README.md` 329 | keep: 1-line what, hardware, 3-command quickstart, feature table, pinned gate row, links; out: "Is imp for you", "How it works", "Where imp loses", "Build from source", "How this was built" -> links to owners |
| `docs/README.md` | L1 | 80 | same 76 | index tables only; update for new/split files |
| `docs/QUICKSTART.md` | L1 | 150 | same 189 | commands + tables |
| `docs/DEPLOYMENT.md` | L1 | 250 | same 254 | compose, auth, health, capacity as tables |
| `docs/API.md` | L1 | 300 | `API.md` 473 (endpoints, fields, prompt caching, tracing, errors) | tables |
| `docs/API_FEATURES.md` | L1 | 250 | `API.md` (constrained decoding, tool calling, thinking, images) | new: split on reference boundary "request field -> generation behaviour" |
| `docs/CONFIG.md` | L1 | 300 | `usage.md` 487 (config, CLI, `--json`, server flags, LoRA, C API) | rename + table rewrite |
| `CONTRIBUTING.md` | L1 | 150 | same 143 + `usage.md` requirements/build/project structure | merge build into existing Build section |
| `docs/MODELS.md` | L1 | 200 | same 183 | tables |
| `docs/FEATURES.md` | L1 | 150 | same 111 | "Model architectures" -> link `MODELS.md` |
| `docs/quantization.md` | L1 | 200 | `quantization.md` 552 (formats, MXFP4, KV dtype, choosing, `imp-quantize` usage) | tables |
| `docs/internals/QUANT_PIPELINE.md` | L2 | 250 | same 67 + `quantization.md` "NVFP4 internal pipeline" | merge |
| `docs/archive/quantization_awq_findings.md` | record | - | `quantization.md` AWQ / refuted sections | move |
| `docs/LIMITATIONS.md` | L1 | 300 | same 792 | table: limitation, affected, workaround, issue; untested paths + missing gates as table |
| `docs/DESIGN_DECISIONS.md` | L1 | 150 | same 232 | table: decision, measurement, source |
| `docs/TROUBLESHOOTING.md` | L1 | 200 | same 234 | table: symptom, cause, fix, issue; build section -> link `CONTRIBUTING.md` |
| `docs/determinism.md` | L1 | 200 | same 295 | guarantees table, limits table, recipe block |
| `docs/GOAL.md` | L1 | 120 | same 207 | hero set, release bar, north star as tables; "What imp is NOT" -> link `DESIGN_DECISIONS.md` |
| `docs/PERF.md` | L1 | 250 | same 246 | owner of every current number |
| `docs/BENCHMARKS.md` | L1 | 300 | same 882 | latest sweep per section, each row date + commit + model + quant + command |
| `docs/archive/benchmarks_pre_v0.44.md` | record | - | `BENCHMARKS.md` superseded sweeps | move |
| `docs/internals/ARCHITECTURE.md` | L2 | 200 | same 154 | Mermaid + component table (component, file, responsibility) |
| `docs/internals/MEMORY.md` | L2 | 300 | same 949 (A2-A6, invariants) | tiers, owners, invariants, planner as tables |
| `docs/archive/memory_census_2026.md` | record | - | `MEMORY.md` A0, A1, A7, B0 log | move |
| `docs/internals/KERNELS.md` | L2 | 150 | same 139 | table: kernel, file, role, status; "REFUTED" narrative -> archive |
| `docs/internals/PROFILING.md` | L2 | 100 | same 193 | nsys / ncu commands only; phase plan -> `docs/plans/` |
| `docs/internals/SM120.md` | L2 | 105 | same | tables |
| `docs/internals/ATTENTION_DISPATCH.md` | L2 | 90 | same 83 | dispatch table |
| `docs/internals/BENCHMARKING.md` | L2 | 141 | same | protocol as numbered commands |
| `docs/internals/CPP23.md` | L2 | 72 | same | rules table |
| `docs/internals/vision_gemma4v_spec.md` | L2 | 66 | same | unchanged |
| `docs/usage.md` | - | - | - | deleted after split (`CONFIG.md`, `CONTRIBUTING.md`) |
| L3 `CLAUDE.md` tree, `AGENTS.md`, `tests/README.md`, `fuzz/README.md`, `tools/**/README.md` | L2/L3 | as now | same | filler and paragraph lint only |

Net: 1 file deleted (`usage.md`), 2 L1 files new (`API_FEATURES.md`, `CONFIG.md`), 3 archive records new.

## Tooling (phase 3)

| Item | Where |
|---|---|
| Filler / marketing word list | `tools/docs_lint.toml`, read by `scripts/docs_lint.py` |
| Paragraph > 2 sentences | `docs_lint.py` rule, finding per paragraph |
| `README.md` <= 150, docs <= 300 lines | `docs_lint.py` budgets; `tools/filesize_thresholds.toml` gets a `[thresholds.docs]` entry only if `tools/check_filesize.py` is taught `.md` |
| Internal links + anchors | extend the relative-link check in `scripts/check-release.sh` (`hygiene` gate) with `#anchor` resolution |
| Command execution | CPU commands in a container in phase 3; GPU commands after `gpu-busy-check.sh` exit 0 |
