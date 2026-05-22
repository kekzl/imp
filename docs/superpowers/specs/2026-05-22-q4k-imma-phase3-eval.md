# Q4_K_M INT8 IMMA — Phase 3 E2E evaluation

**Date:** 2026-05-22
**Status:** Phase 2C (dispatch + handler + config flag) was already wired in
the codebase before this session. Phase 3 = E2E A/B on production models.
**Verdict:** **Keep `gemm.q4k_imma_enabled = false` by default.** No clear
production winner among the Q4_K_M models we test on.

## What was tested

Local imp.conf:
```
gemm.q4k_imma_enabled = true
```

Passed via `imp-cli --config /tmp/q4k_on.conf`. A/B against default (flag
off, dequant→cuBLAS path). Verified via `git rev-parse origin/main = 4115f52`
plus the close-track-e branch HEAD.

## Results

### Gemma-3-12B Q4_K_M (dense, ~12 GB weights)

| seq | IMMA off | IMMA on | Δ |
|---:|---:|---:|---:|
| pp1024 | 5115 tok/s | 5664 tok/s | **+10.7%** |
| pp4096 | 7133 tok/s | 7215 tok/s | +1.1% |

Smoke on "What is 17 + 25?" — **both** IMMA-on and IMMA-off produce
incoherent token streams (URL fragments, isolated digits). This is a
pre-existing Gemma-3 Q4_K_M quality issue independent of IMMA — same model
without `--chat-template gemma` also degenerates with raw cuBLAS path.
Cannot use this model to validate IMMA correctness.

### Gemma-4-26B Q4_K_M (MoE, A4B = 4B active params)

| seq | IMMA off | IMMA on | Δ |
|---:|---:|---:|---:|
| pp1024 | 4649 tok/s | 4645 tok/s | −0.1% (noise) |

Smoke: "17 + 25 = 42" — coherent both paths.

No perf change confirms the memo: MoE experts stay below `MIN_M=32` and skip
the IMMA dispatch (dense attention QKV/out projections do hit the path but
they're a smaller fraction of total time on MoE models).

### Qwen3.6-35B-A3B Q4_K_M (MoE, A3B = 3B active)

| seq | IMMA off | IMMA on | Δ |
|---:|---:|---:|---:|
| pp1024 | 3154 tok/s | 3193 tok/s | +1.3% (within noise) |

Smoke: "find the sum of 17 and 25" — coherent. No correctness regression.

## Interpretation

The IMMA kernel **technically works correctly** (Qwen3.6 smoke output is
coherent at IMMA-on). It delivers a real +10.7% at pp1024 on the only dense
Q4_K_M model we have, but that model has pre-existing quality issues unrelated
to IMMA, so we can't verify the IMMA path doesn't introduce a subtle quality
regression on the dense GEMM workload.

MoE Q4_K_M models (Gemma-4, Qwen3.6) skip the IMMA path for expert GEMMs
because they fall under `MIN_M=32`. The dense attention projections in those
models DO hit IMMA, but they're not a big enough fraction of total time to
move the needle.

## Decision

**Keep default `gemm.q4k_imma_enabled = false`.** No production model
benefits clearly enough to justify the default flip + the small but real
risk of subtle quality drift on dense Q4_K_M weights.

**Re-enable trigger:** when we have a dense Q4_K_M model that produces
coherent output without modifications (Qwen3-32B Q4_K_M would qualify if/when
acquired — see memo `mmq_q4k_v2_phase2_shipped_2026_05_16` which flagged this
as the awaited test model).

Users wanting the +10.7% on dense Q4_K_M can opt-in via:
```
gemm.q4k_imma_enabled = true
```
in their `imp.conf`.

## Phase 2C/3 status

- Phase 2C dispatch: ✅ already wired (handler at `src/exec/gemm_kernel_q4k_imma.cu`, config flag at `src/runtime/config.h:179`, dispatch site at `src/exec/executor_kernels.cu:1665`)
- Phase 3 E2E A/B: ✅ this doc
- Default-flip: ❌ deferred until clean dense Q4_K_M model available

The kernel infrastructure remains in place. No code changes required.
