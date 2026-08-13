<!--
layer: L3
audience: agents
verified: 2026-08-13
commit: 1e4fad60
-->

# Open questions from the docs rewrite

Collected rather than guessed, per dispatch §10. Each entry states what is
undecided, what evidence exists on each side, and what would settle it.

## Q1. Is "TMA warp-specialized grouped GEMM" available on sm_120a?

Two documents in this repo say opposite things, and the docs rewrite cannot
state either as fact until one is retired.

- `CLAUDE.md:87` and `AGENTS.md:11`: "No `tcgen05` / TMEM / wgmma / **TMA-WS
  grouped GEMM**", listed among datacenter-Blackwell-only features.
- `docs/sm120.md:31`: the `compute_120f` family-feature suffix "enables FP8 MMA
  `kind::f8f6f4` and **TMA warp-specialized grouped GEMM tactics**".

Not in dispute: imp emits plain TMA bulk-tensor loads in its own kernel.
`src/compute/gemm_grouped_nvfp4_smallM.cu:65` wraps
`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes`
and its own comment says "Emits UTMALDG on SM120". So the dispatch's rule
"statements that TMA is not used are forbidden" is correct for TMA as a copy
mechanism.

The disagreement is narrower: whether the *CUTLASS warp-specialized grouped
GEMM tactic* is selectable on this arch. **Settles it:** a CUTLASS tactic dump
for the grouped NVFP4 path on `compute_120f`, or a `cuobjdump` of the built
kernel showing whether a WS mainloop was instantiated. Until then the rewritten
docs say only what is measured: TMA loads yes, WS grouped-GEMM tactic unknown.

## Q2. Where do the audit artifacts belong?

The dispatch specifies `docs/_audit/`. `.gitignore:102` excludes `_audit/`
deliberately ("Release-readiness audit (local-only triage notes)"), so
committing there is impossible without reversing a standing decision, and the
dispatch's own Definition of Done requires `PURGE_LOG.md`, `ONBOARDING_RUN.md`
and `AGENT_EVAL.md` to be readable by someone else.

Resolved in favour of `docs/audit/docs-rewrite/`, which is tracked and is where
this repo already keeps ledgers of exactly this kind (`SETTLED.md`,
`PERF_LOG.md`, `AUDIT_FILESIZE.md`). Flagged rather than silently redirected
because it is a deviation from the written instruction.

## Q3. Which decode figure is the README headline?

The dispatch (§2.7) names "~200 tok/s on Qwen3.6-35B-A3B-NVFP4". Three
candidates exist in-tree with different meanings and the choice is editorial,
not factual:

- `tests/perf_baseline.json` pins **Qwen3-8B-Q8_0 at 287.19 tok/s** — the gate,
  and the only figure re-measured on every push.
- The 2026-07-12 hero sweep reads Qwen3-Coder-30B-A3B-NVFP4 at 389 tok/s and
  Qwen3.6-35B-A3B at 311 — a marketing-strength number measured once.
- The 2026-08-12 Nemotron work reads 362-386 tok/s on three checkpoints.

A headline needs one figure with one referent. Recommended: the gate figure,
because it is the one a reader can reproduce and the one CI defends. Left open
because it is the maintainer's call which model represents the project.

## Q4. Does `docs/GOAL.md` survive as a separate document?

The dispatch's target tree has no `GOAL.md`. Its non-goals belong in
`DESIGN_DECISIONS.md`, but its release bars do not obviously belong anywhere in
the new structure, and `check-release.sh` reads it. Kept for now, its non-goals
duplicated into `DESIGN_DECISIONS.md` as the SSoT, with `GOAL.md` linking there.
Merging it away needs a decision about the release bars first.
