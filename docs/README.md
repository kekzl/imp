<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# docs

Documentation is layered by **who is reading**, and every file declares its layer
in frontmatter. `scripts/docs_lint.py` gates the contract in CI.

| layer | reader | where |
|---|---|---|
| **L0** | first contact, knows LLMs, not CUDA | [`../README.md`](../README.md) |
| **L1** | operators: deploy, configure, diagnose | `docs/*.md`, this directory |
| **L2** | kernel work: PTX, MMA, occupancy, roofline | [`internals/`](internals/) |
| **L3** | AI agents working on the tree | `CLAUDE.md`, per directory |

A document links downward; it does not repeat. If an L1 page starts explaining
`mma.sync`, it is in the wrong layer.

## Start here (L1)

| doc | what it answers |
|---|---|
| [`QUICKSTART.md`](QUICKSTART.md) | from nothing to an answered completion |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | config, auth, reverse proxy, health, capacity |
| [`API.md`](API.md) | which endpoints and fields actually work, with status |
| [`MODELS.md`](MODELS.md) | which checkpoints and quants load, and what each needs |
| [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) | symptom → cause → fix |

## The four single sources of truth

Nothing else in the tree states these. Everything else links here.

| doc | owns |
|---|---|
| [`PERF.md`](PERF.md) | every number, and the methodology that makes one admissible |
| [`FEATURES.md`](FEATURES.md) | what exists, with ✅ / 🟡 / ⚪ status |
| [`LIMITATIONS.md`](LIMITATIONS.md) | what does not exist, or exists untested |
| [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) | what is absent *on purpose*, with the measurement |

## Contracts

- [`determinism.md`](determinism.md) — what reproducibility guarantees, and the two batch-invariance properties that hold instead of the one that does not
- [`GOAL.md`](GOAL.md) — the mission and the release bars
- [`quantization.md`](quantization.md) — formats, the NVFP4 path, and the AWQ findings including the refuted ones
- [`usage.md`](usage.md) — full CLI, `imp.conf` and C API reference

## Internals (L2)

| doc | what it is |
|---|---|
| [`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md) | the narrative, and **the** statement of what `sm_120a` has and lacks |
| [`internals/SM120.md`](internals/SM120.md) | hardware notes, MMA shapes, measured ceilings |
| [`internals/KERNELS.md`](internals/KERNELS.md) | kernel catalogue and design reference |
| [`internals/ATTENTION_DISPATCH.md`](internals/ATTENTION_DISPATCH.md) | which attention kernel runs for each phase × dtype × layer |
| [`internals/MEMORY.md`](internals/MEMORY.md) | tiers, allocators, invariants I1-I7. Read before anything about VRAM |
| [`internals/QUANT_PIPELINE.md`](internals/QUANT_PIPELINE.md) | the two layers handling quantized weights |
| [`internals/BENCHMARKING.md`](internals/BENCHMARKING.md) | the measurement contract |
| [`internals/PROFILING.md`](internals/PROFILING.md) | nsys and ncu on this host |

## Records, not documentation

These are append-only and are deliberately **not** linted or refreshed: a record
is a statement about one dated afternoon.

- [`roadmap.md`](roadmap.md) — the gap list, with how each gap was measured or refuted
- [`MISSION_JOURNAL.md`](MISSION_JOURNAL.md), [`vram_audit.md`](vram_audit.md)
- [`BENCHMARKS.md`](BENCHMARKS.md) — per-model competitive figures, each row carrying its own date, commit and command
- [`archive/`](archive/), [`audit/`](audit/), [`plans/`](plans/)
