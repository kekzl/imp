# KV-cache NIAH harness

Runs a 4-config × 2-context × 5-depth × 3-seed Needle-in-a-Haystack
retrieval test on Qwen3-8B Q8_0, comparing the KV-cache dtypes
FP16 / FP8 / NVFP4 / MXFP4-KV. (The TurboQuant `tq_qjl_on` / `tq_qjl_off`
configs were retired in Phase 5, 2026-05-17.)

Output: `sample_results/niah_results.json` (raw) + `niah_summary.md`
(aggregate accuracy + verdict).

## Usage

```bash
# Full matrix (~6-8 min wall clock):
tools/eval/niah/niah_bench.py

# Smoke test (1 prompt):
tools/eval/niah/niah_bench.py --smoke

# Subset (e.g. only the FP4 KV configs, only 4K context):
tools/eval/niah/niah_bench.py --config nvfp4 --config mxfp4_kv --ctx 4096
```

Available `--config` names: `fp16`, `fp8`, `nvfp4`, `mxfp4_kv`.

See `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5 Phase 2 for
the historical TurboQuant context.
