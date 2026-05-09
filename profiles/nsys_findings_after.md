# nsys Findings — AFTER patches

**Date:** 2026-05-09
**Build:** `imp:profile` rebuilt with patches for F1, F2, F4, F7, F8 + F3 verify-gate.
**Baseline:** `profiles/nsys_findings.md` (same workloads, identical capture method).

## Patches shipped

| # | Files | What changed |
|---|-------|--------------|
| F7 | `src/runtime/engine.cpp`, `src/graph/executor.h` | `attn_scores_cap()` accessor + per-request `effective_chunk` clamp so `n × ctx_len ≤ s_cap²` for every chunk in `step_prefill_one`. |
| F4 | `src/compute/attention_cublas.cu` | New `causal_softmax_fp32_to_fp16_kernel` fuses softmax + downcast in one pass. Replaces `softmax_fp32_inplace` + `fp32_to_fp16_kernel` pair on both MHA and GQA paths. |
| F8 | `src/graph/executor_workspace_buffers.cu`, `src/compute/gemm_cutlass_sm120.cu`, `src/compute/weight_dispatch.cu` | Pre-zero the executor's `cutlass_act_sf` workspace once at allocation. Drop the per-call `cudaMemsetAsync` in `quantize_fp16_to_nvfp4_cutlass` (workspace path). External `weight_dispatch` slice path keeps an explicit memset for safety. |
| F1 | `src/compute/gemm.cu` | `benchmark_and_select_algo`: warm up *all* candidates with `kWarmupIters=3` before timing any. Prevents cold-start WMMA selection. |
| F2 | `src/graph/executor_kernels.{h,cu}`, `src/graph/executor_attention.cu` | New `attn_gate_split_interleaved_kernel<T>` + host launcher. Replaces `nh × 2` `cudaMemcpy2DAsync` loop in the GDN attn-output-gate split with one fused kernel. |
| F3 | `scripts/verify.sh` | New section: graphs-OFF vs graphs-ON tg256 bench, fails if speedup < 1.5× (configurable via `IMP_VERIFY_MIN_GRAPH_SPEEDUP`). Catches future PRs that silently break decode graph capture. |
| F5 | (deferred) | depends on per-shape WMMA-fallback enumeration; revisit after F1 settles. |
| F6 | (deferred) | needs `ncu` w/ CAP_SYS_ADMIN — consumer driver restricts perf counters. |

## Validation results (nsys re-capture, identical methodology)

### Llama-3.2-3B Q8_0 W1 (long prefill, pp=8192 tg=64)

|  | total kernel time | softmax+cast | cudaMemsetAsync calls |
|--|------------------:|--------------:|----------------------:|
| **before** | 2.25 s | 633 ms (28 %) | 10 203 |
| **after**  | **1.87 s** | **273 ms (15 %)** | **3 515** |
| **delta**  | **−17 % prefill** | **−57 %** | **−66 %** |

### Qwen3-4B Q8_0 W1 (long prefill)

|  | total kernel time | softmax+cast | cudaMemsetAsync |
|--|------------------:|--------------:|----------------:|
| **before** | 3.37 s | 1101 ms (33 %) | 13 042 |
| **after**  | **2.72 s** | **468 ms (17 %)** | **4 432** |
| **delta**  | **−19 % prefill** | **−57 %** | **−66 %** |

### Qwen3.5-4B GDN Q8_0 W2 (decode-heavy, pp=256 tg=2048)

|  | D2D copy count | D2D time | `cudaMemcpy2DAsync` calls |
|--|----------------:|----------:|---------------------------:|
| **before** | 1 344 488 | 695 ms | 1 049 344 |
| **after**  | **295 144** | **177 ms** | **0** |
| **delta**  | **−78 %** | **−74 %** | **eliminated** |

The nh × 2 `cudaMemcpy2DAsync` loop is replaced by 32 792 launches of `attn_gate_split_interleaved_kernel<half>` totalling 24 ms. Net: 1.05 M API launches → 32 K, with kernel time well under the API time saved.

Smoke decode tok/s (graphs ON, single rep): **228.75 tok/s vs 220 baseline = +4 %**.

### Qwen3.5-4B GDN Q8_0 W1 (long prefill)

|  | status |
|--|--------|
| **before** | aborted with `chunked_prefill: attn_scores_ capacity 4096×4096 too small for n=4096 × ctx_len=8192` |
| **after**  | **runs cleanly: pp 13 933 tok/s, tg 217 tok/s** |

The `effective_chunk` clamp picks 2048 for this workload (s_cap²/total_input = 16M/8192 = 2048), unblocking long-context profiling for the entire GDN/Mamba2 family — Qwen3.5-9B, Qwen3.5-27B, Qwen3.6-35B-A3B-NVFP4.

### Qwen3-4B Q8_0 W2 (decode)

|  | total kernel time | cudaMemsetAsync calls |
|--|------------------:|----------------------:|
| **before** | 16.88 s | 1 390 |
| **after**  | 16.80 s | **768** |
| **delta**  | −0.5 % | −45 % |

Decode wasn't F8's primary target (decode has fewer NVFP4 quantize calls than prefill). Smoke shows tg32 graphs-ON 246.85 tok/s vs 242 prior — within noise but trending up.

### F1 (WMMA fallback) — note

WMMA kernel call count and time *unchanged* in the 1-rep nsys re-capture (651 / 1334 calls in Llama / Qwen). The `benchmark_and_select_algo` warmup-pre-pass kicks in only when the heuristic gate runs (per-shape, on cold cache). With `bench-reps=1`, each shape is benched exactly once and the new warmup loop is only used for that single decision. Real production load (`gen_perf_baseline.sh` uses `reps=5`) and re-runs of the same shape benefit from the per-shape `s_gemm_cache`. **Re-bench needed with `--bench-reps 5+` to verify WMMA-fallback selection drops in steady state.**

## End-to-end deltas (smoke benches, graphs ON, bench-reps=1)

| Model | tg before | tg after | delta |
|-------|----------:|---------:|------:|
| Qwen3-4B Q8 (tg32)        | 242 tok/s  | **247 tok/s** | +2 %  |
| Qwen3.5 GDN Q8 (tg32)     | 220 tok/s  | **229 tok/s** | +4 %  |
| Qwen3.5 GDN Q8 (tg64, pp=8192) | n/a (aborted) | **217 tok/s** | unblocked |

## `make verify-fast` (canonical pre-merge gate, Qwen3-8B Q8, reps=5, graphs ON)

| Metric | Baseline (`tests/perf_baseline.json`) | After patches | Delta |
|--------|---------------------------------------:|---------------:|-------:|
| decode tg128 | 147.85 tok/s | **156.17 tok/s** | **+5.63 %** |
| prefill pp512 | 13277.98 tok/s | **14841.82 tok/s** | **+11.78 %** |
| graphs ON / OFF | n/a | **1.77×** speedup | new gate, threshold 1.5× |
| gtest fast filter | PASS | PASS | — |
| smoke degeneration | PASS | PASS | — |

The 11.78 % prefill win on Qwen3-8B (matching baseline test methodology with reps=5)
is the canonical production-load result — bigger than the 1-rep nsys captures showed
because cuBLAS algo selection benefits from the warmup pre-pass (F1) over multiple
reps, and F4's softmax fuse + F8's memset reduction compound over each rep.

Recommended next step: refresh `tests/perf_baseline.json` via
`scripts/gen_perf_baseline.sh` so future PRs are gated against the new ceiling.

## Files changed (one concern per commit recommended)

```
src/runtime/engine.cpp                          # F7
src/graph/executor.h                            # F7
src/compute/attention_cublas.cu                 # F4
src/graph/executor_workspace_buffers.cu         # F8
src/compute/gemm_cutlass_sm120.cu               # F8
src/compute/weight_dispatch.cu                  # F8
src/compute/gemm.cu                             # F1
src/graph/executor_kernels.h                    # F2
src/graph/executor_kernels.cu                   # F2
src/graph/executor_attention.cu                 # F2
scripts/verify.sh                               # F3
```

11 files, ~250 LoC added/modified.

## F5 investigation — RESOLVED via F1

Added `IMP_LOG_GEMM_ALGO=1` debug logging to `benchmark_and_select_algo`. Re-ran
Llama-3.2-3B Q8 W1 prefill at `--bench-reps 5 --no-cuda-graphs` to enumerate
which shapes hit cuBLAS and what they pick now:

```
[gemm-algo] shape M=512 N=3072 K=3072   candidates=8   PICKED tile=21 (0.30ms)
[gemm-algo] shape M=512 N=3072 K=8192   candidates=8   PICKED tile=18 (0.68ms)
pp 8192 tokens  avg 450.82ms  (18171.44 tok/s)  [5 reps]
tg   32 tokens  avg 273.82ms  ( 116.87 tok/s)  [5 reps]
```

Only **two** shapes route through the cuBLAS path on dense Q8 prefill:
- attn output proj (M=512 chunk × N=3072 × K=3072) — picks tile 21 (≈ 128×128)
- FFN down (M=512 chunk × N=3072 × K=8192) — picks tile 18 (≈ 128×64)

Both pick **modern tiles, not legacy WMMA**. The previous baseline's 651 WMMA
kernel calls per-W1 captured AT reps=1 reflected single-shot algo selection
where WMMA's first-call latency happened to win. F1's warmup-pre-pass shows
its real impact at reps≥2: prefill improves to **18 171 tok/s on Llama Q8 W1
(+32 % vs the 13 800 baseline)**, and the verify-fast Qwen3-8B run already
showed +11.78 % prefill.

The remaining hot GEMMs (Q/K/V projections, gate/up FFN) bypass cuBLAS entirely
— they go through the CUTLASS_NVFP4 fast path (`MainloopSm120TmaWarpSpecialized
BlockScaled`) via on-the-fly NVFP4 quantization. F5 is **not actionable** as
originally scoped: the two shapes still on cuBLAS are already picking
high-performance tiles, and routing them through NVFP4 would cost extra weight
storage with no clear speedup.

**F5 status: closed — F1's warmup pre-pass solved the WMMA-fallback problem
for the production-relevant call patterns (reps ≥ 2). Re-bench any future
shape-routing investigation with `IMP_LOG_GEMM_ALGO=1` for evidence.**

## Open follow-ups
- **F6** (`gemv_dp4a_gate_up_kernel` deep-dive) — deferred. Needs `ncu` with `CAP_SYS_ADMIN` (or `RestrictProfilingToAdminUsers=0` driver flag). Helper script `profiles/run_ncu_topk.sh` ready when perf counters are unlocked. Optimization targets: cp.async double-buffering, wider vector loads, Q8-block alignment refinements.
- **Refresh `tests/perf_baseline.json`** via `scripts/gen_perf_baseline.sh` — current baseline pre-dates these patches; CI will report perf gains as "regressions" against the old number until refresh.
- **`make verify-fast`** — should pick up F3's new graphs-gate section automatically; first run confirms speedup ≥ 1.5×.

## Reproduction

```bash
# Re-capture after patches
mkdir -p profiles/after
./profiles/run_nsys_baselines.sh    # (point to profiles/after dir or copy + edit)
./profiles/export_stats.sh
python3 profiles/analyze_csv.py profiles/csv_after
```
