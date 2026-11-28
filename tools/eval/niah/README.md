<!--
layer: L2
audience: kernel-devs
verified: 2026-11-28
commit: 89516523
-->

# KV-cache NIAH harness

Runs an 8-config × 2-context × 5-depth × 3-seed Needle-in-a-Haystack
retrieval test on Qwen3-8B Q8_0, comparing the KV-cache dtypes
FP16 / FP8 / NVFP4 / MXFP4-KV plus four FP8 sparse-attention arms. (The TurboQuant `tq_qjl_on` / `tq_qjl_off`
configs were retired in Phase 5, 2026-05-17.)

Output: `sample_results/niah_results.json` (raw) + `niah_summary.md`
(aggregate accuracy + verdict).

## Usage

```bash
# Full matrix (~6-8 min wall clock):
tools/eval/niah/niah_bench.py

# Smoke test (1 prompt):
tools/eval/niah/niah_bench.py --smoke

# Long contexts: prompts are passed via --prompt-file (a bind-mounted temp
# file), so --ctx 32768 and above work; a bare argv caps at ~128 KiB
# (~32k tokens) per exec argument.

# Subset (e.g. only the FP4 KV configs, only 4K context):
tools/eval/niah/niah_bench.py --config nvfp4 --config mxfp4_kv --ctx 4096
```

Available `--config` names: `fp16`, `fp8`, `nvfp4`, `mxfp4_kv`, `fp8_ng`, `fp8_sparse4k`, `fp8_sparse2k`, `fp8_sparse4k_spec`.

See `git show e58d9d9c9^:docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5 Phase 2 (removed in #273) for
the historical TurboQuant context.
