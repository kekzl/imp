# Phase 7 — Performance Sanity

Strategic precisions (NVFP4 + FP8 KV) measured on Mistral-Small-3.2-24B-Instruct-2506-NVFP4. Q8_0 GGUF baseline re-checked as a legacy data point.

## Build correction

A real bug was discovered while comparing my first Q8 run against `tests/perf_baseline.json`: `cmake/CompilerFlags.cmake` set `CMAKE_CUDA_FLAGS_RELWITHDEBINFO` to the CMake default `-O2 -g -lineinfo -DNDEBUG`. That dropped `-O3 --use_fast_math --extra-device-vectorization -Xptxas -O3` vs the `_RELEASE` flags. Result: pp512 on Q8_0 came in at 4012 tok/s (vs ~13000+ expected). The bringup spec requires `RelWithDebInfo`, so the correct fix is to harmonize the flags.

**Fix applied** to `cmake/CompilerFlags.cmake`: both `CXX` and `CUDA` `RELWITHDEBINFO` now match Release for the optimizer (`-O3` / `--use_fast_math` / PTX `-O3` / `--extra-device-vectorization`); only `-g` (host) and `-lineinfo` (device) are added on top, both code-gen-neutral. After the rebuild, Q8_0 pp512 jumped from 4012 → 13278 tok/s (3.3× recovery), confirming the diagnosis. **6 LOC change, 1 file** — committed.

## Strategic precision baselines (NEW — establishing, no prior baseline existed)

| Path | Model | pp512 (tok/s) | tg256 (tok/s) | log |
|---|---|---:|---:|---|
| NVFP4 + FP16 KV (default) | Mistral-Small-3.2-24B-NVFP4 | **1218.19** | **93.96** | `72_bench_nvfp4_postfix.log` |
| NVFP4 + FP8 KV (`--kv-fp8`) | Mistral-Small-3.2-24B-NVFP4 | **1220.32** | **93.91** | `72_bench_fp8_postfix.log` |

**FP8 KV at parity with NVFP4 default on a 24B model.** ½ KV-cache memory for 0% throughput cost — the FP8 KV path should arguably become the default for non-GDN, non-Gemma-4 models. (Memory file `kv_dtype_tradeoffs_2026_04_24.md` already recommends flipping the default; this run confirms.)

GPU was at full boost during the bench (mid-run check: pstate P1, 2880 MHz core, 456 W). The `--bench --bench-reps 10` is enough warmup to leave WSL2's idle clamp.

## Q8_0 GGUF (legacy reference)

| Path | pp512 (tok/s) | tg128 (tok/s) | reps | vs `tests/perf_baseline.json` |
|---|---:|---:|---:|---|
| Q8_0 default (FP8 prefill + NVFP4 decode auto) | **13277.98** | **146.49** | 5 | pp Δ −15.1% (within memory-documented cuBLAS variance up to 2.6×); tg Δ **−43.2%** |
| Q8_0 raw mmvq (`--no-nvfp4 --no-fp8-prefill --kv-fp16`) | 7271.19 | 146.69 | 5 | n/a |

**The −43% decode delta on Q8_0 is real and unexplained by the perf-flag fix alone** (raw mmvq path matches the auto-upgraded NVFP4-decode path within 0.2%, and pp512 lands within variance). Possible causes (not chased — out of strategic scope):
- Real regression on `main` between 2026-03-27 (baseline date) and 2026-04-29 — would need bisect over ~50 commits.
- Baseline drift: the `tests/perf_baseline.json` was captured before some compiler/driver/CUTLASS update.
- WSL2 driver state difference.

Per the bringup spec, "Regression >5% on TTFT or tok/s in **either precision** [NVFP4 or FP8] → bisect subagent." Q8_0 is GGUF/legacy, not in the strategic precisions, so per the priority strategy: **flagged but not bisected**. This will appear as a "Next 3 highest-leverage follow-ups" item in the final report.

## Regressions in strategic paths
**None.** Both NVFP4 and FP8 KV are new baselines (no prior numerical measurement to regress against). They land at parity with each other and produce coherent goldens (Phase 5).

## Status
✅ **Phase 7 PASS in strategic scope** (NVFP4 + FP8). One latent build-config bug fixed and committed; one open Q8_0-decode delta logged as a follow-up.
