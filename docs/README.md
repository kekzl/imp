# docs

Nineteen more markdown files sit next to this one and GitHub lists them
alphabetically, which tells you nothing about which is the map and which is a
snapshot of one afternoon. This is the index.

**If you are here to understand the engine, read [`architecture.md`](architecture.md)
first.** It is the canonical narrative; everything else is either a companion to
it, a contract, or a record.

## The canonical four

`CLAUDE.md` routes agents to these, and they are the ones kept current.

| doc | what it is |
|---|---|
| [`architecture.md`](architecture.md) | the narrative — how a request becomes tokens |
| [`sm120.md`](sm120.md) | the hardware imp targets, and what consumer Blackwell does *not* have |
| [`MEMORY_ARCHITECTURE.md`](MEMORY_ARCHITECTURE.md) | the memory subsystem: tiers, allocators, invariants I1–I7. Read before anything about VRAM, ownership or lifetime |
| [`BENCHMARKING.md`](BENCHMARKING.md) | the measurement contract — what counts as a number here, and what does not |

## Companions to the architecture

Narrower than `architecture.md`, same standing.

- [`attention-dispatch.md`](attention-dispatch.md) — which attention kernel runs for each (phase × dtype × layer)
- [`quant-pipeline.md`](quant-pipeline.md) — the two parallel layers handling quantized weights, and the boundary between them
- [`quantization.md`](quantization.md) — formats, the NVFP4 path, and the AWQ calibration findings (including the ones that were refuted)
- [`sm120_optimal_kernel.md`](sm120_optimal_kernel.md) — design reference for the hot-path attention kernel
- [`vision_gemma4v_spec.md`](vision_gemma4v_spec.md) — the gemma4v encoder spec

## Contracts and promises

What imp guarantees, and where the edges are.

- [`GOAL.md`](GOAL.md) — the mission and the release bars, including the non-goals
- [`determinism.md`](determinism.md) — reproducibility guarantees, the documented limits, and the two batch-invariance properties that hold instead of the one that does not
- [`supported-models.md`](supported-models.md) — architectures that load, and what each needs
- [`usage.md`](usage.md) — running the CLI and the server

## Numbers

- [`BENCHMARKS.md`](BENCHMARKS.md) — published competitive numbers, tied to a release tag
- [`performance.md`](performance.md) — methodology behind them
- [`vram_audit.md`](vram_audit.md) — append-only per-component VRAM accounting
- [`nsys_profiling.md`](nsys_profiling.md) — how to profile with Nsight Systems on this box

## Where the work goes

- [`roadmap.md`](roadmap.md) — current focus and the open gaps. **"What is still open?" is answered here**, not in the audit ledger
- [`MISSION_JOURNAL.md`](MISSION_JOURNAL.md) — the investigation record: what was tried, what it cost, what it disproved
- [`plans/`](plans/) — design documents for work large enough to need one before code

## Audit and test ledgers

[`audit/`](audit/) holds two different kinds of file, and mixing them up wastes time.

**Living ledgers** — consult these:

- [`audit/SETTLED.md`](audit/SETTLED.md) — read *before* forming audit hypotheses. Every finding with a terminal state
- [`audit/ARCHMAP.md`](audit/ARCHMAP.md) — structure derived from source, not from prose
- [`audit/AUDIT_FILESIZE.md`](audit/AUDIT_FILESIZE.md) — per-file rationale for the size gate
- [`audit/TEST_INVENTORY.md`](audit/TEST_INVENTORY.md), [`audit/ESCAPE_ANALYSIS.md`](audit/ESCAPE_ANALYSIS.md), [`audit/MUTATION_BASELINE.md`](audit/MUTATION_BASELINE.md), [`audit/TEST_HARDENING_LOG.md`](audit/TEST_HARDENING_LOG.md) — what the suite covers, what escaped it, and how well it kills mutants

**Dated campaign reports** — snapshots of one investigation, kept because their
numbers are still cited. They are history, not current state: `roofline_*`,
`structural_debt_*`, `prefill_gap_*`, `housekeeping_*`, `vram_audit_*`,
`cpp23_migration_*`, `ppl_parity_*`, `PERF_AUDIT_*`, `DISPATCH_BASELINE_*`, and
[`audit/AUDIT_ARCH_2026_07_29.md`](audit/AUDIT_ARCH_2026_07_29.md) with its
evidence directory.

[`audit/decisions/`](audit/decisions/) holds ADRs — a decision outliving its
context is the point of one.

[`archive/`](archive/) is where a document goes when its subject was shelved and
the reasoning is still worth having.

## A note on deleting things here

A refuted experiment is not clutter — this repo spends real GPU time
re-discovering things it already disproved. The bar for removing a doc is that
its conclusion is carried somewhere maintained, not that its subject is over.
