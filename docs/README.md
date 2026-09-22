<!--
layer: L1
audience: operators
verified: 2026-09-23
commit: 9cbb8004
-->

# docs

| Layer | Reader | Where |
|---|---|---|
| **L0** | first contact, knows LLMs, not CUDA | [`../README.md`](../README.md) |
| **L1** | operators: deploy, configure, diagnose | `docs/*.md` |
| **L2** | kernel work: PTX, MMA, occupancy, roofline | [`internals/`](internals/) |
| **L3** | AI agents working on the tree | `CLAUDE.md`, per directory |

Gate: `scripts/docs_lint.py` (frontmatter, links, anchors, provenance, prose rules in `tools/docs_lint.toml`, line limits in `tools/filesize_thresholds.toml`). A doc links downward; it does not repeat.

## Start here (L1)

| Doc | Answers |
|---|---|
| [`QUICKSTART.md`](QUICKSTART.md) | from nothing to an answered completion |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | compose, auth, reverse proxy, health, capacity |
| [`CONFIG.md`](CONFIG.md) | `imp.conf` keys, `imp-cli` and `imp-server` flags, C API |
| [`API.md`](API.md) | HTTP endpoints, request fields, errors |
| [`API_FEATURES.md`](API_FEATURES.md) | constrained decoding, tool calling, thinking, images |
| [`MODELS.md`](MODELS.md) | which checkpoints and quants load, and what each needs |
| [`quantization.md`](quantization.md) | formats, KV cache dtype, choosing a quant, `imp-quantize` |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | symptom, cause, fix |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | build, test, PR rules |

## Single sources of truth

Nothing else in the tree states these; everything else links here.

| Doc | Owns |
|---|---|
| [`PERF.md`](PERF.md) | every current number and the methodology that makes one admissible |
| [`BENCHMARKS.md`](BENCHMARKS.md) | latest per-model sweep per section, each row with date, commit and command |
| [`FEATURES.md`](FEATURES.md) | what exists, with ✅ / 🟡 / ⚪ status |
| [`LIMITATIONS.md`](LIMITATIONS.md) | what does not exist, or exists untested |
| [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) | what is absent on purpose, with the measurement |
| [`determinism.md`](determinism.md) | reproducibility guarantees and limits |
| [`GOAL.md`](GOAL.md) | mission, hero models, release bars |

## Internals (L2)

| Doc | Content |
|---|---|
| [`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md) | component diagram and table; the statement of what `sm_120a` has and lacks |
| [`internals/SM120.md`](internals/SM120.md) | hardware notes, MMA shapes, measured ceilings |
| [`internals/KERNELS.md`](internals/KERNELS.md) | kernel catalogue |
| [`internals/ATTENTION_DISPATCH.md`](internals/ATTENTION_DISPATCH.md) | attention kernel per phase × dtype × layer |
| [`internals/MEMORY.md`](internals/MEMORY.md) | tiers, allocators, invariants I1-I7; read before anything about VRAM |
| [`internals/QUANT_PIPELINE.md`](internals/QUANT_PIPELINE.md) | quantized-weight layers, NVFP4 pipeline, GEMM dispatch |
| [`internals/BENCHMARKING.md`](internals/BENCHMARKING.md) | the measurement contract |
| [`internals/CPP23.md`](internals/CPP23.md) | C++23 in use, host/device line |
| [`internals/PROFILING.md`](internals/PROFILING.md) | nsys and ncu commands on this host |
| [`internals/vision_gemma4v_spec.md`](internals/vision_gemma4v_spec.md) | the Gemma-4 vision encoder, as implemented |

## Records (append-only, not linted)

| Record | Content |
|---|---|
| [`roadmap.md`](roadmap.md) | gap list with how each gap was measured or refuted |
| [`MISSION_JOURNAL.md`](MISSION_JOURNAL.md), [`vram_audit.md`](vram_audit.md) | dated journals |
| [`plans/`](plans/README.md) | one record per campaign, each ending in its verdict |
| [`audit/`](audit/README.md) | audit records; the index says which are live ledgers |
| [`archive/`](archive/) | superseded sweeps and investigations (`benchmarks_pre_v0.44.md`, `memory_census_2026.md`, `limitations_detail_2026.md`, `quantization_awq_findings.md`, `kernels_refuted_2026.md`) |
