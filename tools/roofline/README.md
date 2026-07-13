# tools/roofline — reproducible roofline & coverage pipeline

Measures roofline proximity (%-roofline, AI, achieved FLOPS/BW) and kernel
coverage (legacy-fallback shares) of the imp hot-path kernels on the RTX 5090
(sm_120a), with append-only history per commit.

## Quickstart

```bash
tools/roofline/roofline measure                 # full sweep, 3 restarts (~hours, GPU exclusive)
tools/roofline/roofline measure --models q8-dense --shapes tg256 --restarts 1   # smoke
tools/roofline/roofline plot --latest           # roofline.png + roofline_trend.png
tools/roofline/roofline plot --compare RUN_A RUN_B
tools/roofline/roofline report --run latest -o audit/roofline_$(date +%Y_%m_%d).md
tools/roofline/roofline regress --baseline <run_id|sha> --threshold 5
tools/roofline/roofline issues --run latest     # dry-run; --create files GitHub issues
tools/roofline/roofline ab --knob fa2           # unprofiled A/B (FA2 on vs never)
```

## Architecture

- **Measurement** (`measure`): per cell (model × shape × restart), two passes in
  fresh Docker containers (= restart variance by construction):
  1. **nsys** (full timeline): kernel time shares, cuBLAS API attribution
     (batched GEMM ⇒ legacy attention QK^T/PV), phase split (init/prefill/
     decode), and calibration of the ncu `--launch-skip` (steady-state window).
  2. **ncu** (pinned counter set, `--clock-control base`): per launch
     time, `dram__bytes`, tensor-pipe FLOP counters, SM/DRAM %, occupancy.
- **History** (`history/`): append-only.
  - `runs/<run_id>.json` — parsed run (committed). `run_id = <shortSHA>[-dirty]_<timestamp>`.
  - `index.jsonl` — one line per run for trend queries (committed).
  - `raw/<run_id>/*.ncu_raw.csv.gz` + `*.nsys_extract.json` — raw exports,
    re-parse without re-measure (committed).
  - `raw/<run_id>/*.ncu-rep|*.nsys-rep|*.sqlite` — binary originals,
    **local only** (gitignored; the release check forbids them in the repo).
- **Plots** (`plot`): matplotlib in the `imp-roofline-plot` container
  (Dockerfile.plot, host stays clean) — renders exclusively from history.
- **Report** (`report`): Markdown with the Module-1 table, the Module-2 coverage
  matrix and a prioritized lever list; every number references the run (commit+ts).
  **Lever-list caveat:** shares are measured with `--no-cuda-graphs`; under the
  shipped graphs+PDL decode loop, launch-latency classes (moe_routing, rmsnorm,
  rope, kv_write, elementwise, split-K reduce) largely overlap away (2026-07-13:
  no-graphs kernel-time sum ≈1.8× the real graphs-ON step on Qwen3-30B; router
  fusion 0% e2e, split-K cap −21…−35%). Validate those levers graphs-ON first.
- **Gate** (`regress`): exit≠0 when a kernel class (time share ≥0.5%) drops in
  the median by more than the threshold below the baseline AND the restart
  ranges are disjoint (otherwise variance, no fail).

## Determinism / methodology

- The counter set, shapes, peaks and classification regexes are versioned in
  `config.json` (`config_version` — only compare runs within the same version).
- ncu locks clocks to base (`--clock-control base`); compute peaks are
  normalized to the **measured** SM clock (`gpc__cycles_elapsed.avg.per_second`),
  ridge points to the boost clock (2.407 GHz) — both in config.json.
- AI = FLOPs / `dram__bytes.sum` (measured DRAM traffic, not estimated).
- **FLOP counting**: tensor-core FLOPs from the `sm__ops_path_tensor_src_*`
  counters (calibrated against known GEMM shapes, see report methodology). Non-TC:
  SASS thread instructions (ffma=2 FLOP; hfma counted as packed HFMA2=4 FLOP
  — an upper bound, flagged as such in the report).
- Profile runs use `--no-cuda-graphs` (graph replay hides kernels, the kernel
  mix is identical — see docs/MISSION_JOURNAL/memory).
- Prefill cells measure `--bench-reps 3 --max-tokens 1`; decode cells
  `--bench-pp 64 --max-tokens 256`. pp restart variance (known up to 2.6×) is
  reported as min/med/max, never averaged away.

## CI

GPU measurement is LOCAL (CI has no GPU runner). The workflow
`.github/workflows/roofline.yml` checks only parser/math against the checked-in
history (re-parse) on each PR and renders plots as an artifact. Baseline pinning:
after merge to main, run `roofline measure` locally + commit the history;
`regress` runs in the pre-push hook when a baseline is pinned (`history/BASELINE`).
