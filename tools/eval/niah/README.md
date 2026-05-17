# TurboQuant Phase 2 NIAH harness

Runs a 4-config × 2-context × 5-depth × 3-seed Needle-in-a-Haystack
retrieval test on Qwen3-8B Q8_0, comparing FP16 / FP8 / TurboQuant
(QJL on) / TurboQuant (QJL off via `IMP_TQ_SKIP_QJL=1`, shipped in
PR #246).

Output: `sample_results/niah_results.json` (raw) + `niah_summary.md`
(aggregate accuracy + Phase 2 verdict).

## Usage

```bash
# Full matrix (~6-8 min wall clock):
tools/eval/niah/niah_bench.py

# Smoke test (1 prompt):
tools/eval/niah/niah_bench.py --smoke

# Subset (e.g. only TQ configs, only 4K context):
tools/eval/niah/niah_bench.py --config tq_qjl_on --config tq_qjl_off --ctx 4096
```

## Caveat

`IMP_TQ_SKIP_QJL=1` proxies "post-Path-A storage" at the **quality** level
by stripping QJL while keeping PolarQuant FP4 K + INT4 V. It does NOT
swap the underlying storage to straight MXFP4-K. So this harness answers
"does the QJL correction matter for retrieval quality" — not the strictly
broader "does PolarQuant→straight-MXFP4 storage transition matter".

See `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5 Phase 2.
