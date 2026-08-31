# FP8 prefill for the GDN/SSM projections (gemm.fp8_ssm_prefill)

Record, 2026-08-31. Follow-up to the prefill roofline pin (run `1d5b9230`,
PR #1835): on the hybrid cell (`Qwen3.6-35B-A3B-NVFP4`) the `gemm_cublas`
class runs at 24.8% roofline with 21.5% of the pp512 window - the FP16 GDN
projections (`ssm_in`/`ssm_out`, BF16 producer recipe), est. +13.9% window.

## Kernel-level validation (microbench, sm_120a, CUDA 13.3.1)

cuBLASLt FP8xFP8 (per-tensor scales, COMPUTE_32F) vs the FP16 path at the
GDN projection shapes, 50 reps after >1 s warmup:

| shape (M, N, K) | FP16 | FP8 | speedup |
|---|---|---|---|
| 512, 10240, 5120 (ssm_in pp512) | 250.7 us / 214 TFLOPS | 120.9 us / 444 | 2.07x |
| 512, 5120, 6144 (ssm_out pp512) | 147.4 us / 219 | 67.5 us / 478 | 2.18x |
| 4096, 10240, 5120 | 1847.7 us / 233 | 513.8 us / 836 | 3.60x |
| 15, 10240, 5120 (probe M) | 71.2 us | 37.1 us | 1.92x |

FP16 caps at ~215-230 TFLOPS (the FP32-accumulate quarter-rate); FP8
reaches 836 TFLOPS = 50% of FP8 peak. Status OK at every M including 15
(the historical cuBLAS-13.4-NOT_SUPPORTED canary).

## Design as shipped

- Flag `gemm.fp8_ssm_prefill` (default OFF), requires `gemm.fp8_ssm_proj`.
- Reuses the existing decode-sidecar FP8 bytes + per-row scales
  (zero extra weight VRAM); adds the FP8 activation scratch (~24 MiB on
  the 35B) and a device unit scale.
- Dispatch (`executor_gemm_dispatch.cu`, prefill FP16 branch): per-tensor
  E4M3 act quant (async reduction path) + `gemm_cublaslt` FP8xFP8 +
  `scale_cols_fp16` epilogue folding the per-row weight scales back in.
  `beta == 0` only (a residual add would be rescaled too).
- `cudaStreamIsCapturing` guard: spec-verify chunks (M=2-9) and batched
  decode (M<=32) share this dispatch under graph capture - a first-call
  cuBLASLt algo benchmark inside a capture is illegal, and captured paths
  keep FP16 numerics. Captured graphs therefore never contain the FP8
  path; only uncaptured prefill takes it.
- **Scope: `SSM_OUT` only.** See below.

## The SSM_IN arm is closed (root cause unisolated)

| arm | PPL (200-token corpus, Qwen3.6-35B) |
|---|---|
| baseline (flag off) | 4.0947 |
| SSM_OUT only | 4.1348 (+1.0%) |
| SSM_IN only | 248320 = vocab size, uniform logits |
| both | 243743 |

- The identical building-block pipeline reproduces FP16 within 3.6% RMS at
  the exact engine shapes (M=15/200, N=8192, K=2048) in
  `FP8GemmTest.SsmPrefillFp8RowscaleMatchesFp16` - the math is not the bug.
- The obvious hypothesis (activation channel outliers collapse the
  per-tensor act scale) is REFUTED: a 400x single-channel outlier degrades
  RMS not at all (0.035 vs 0.036 healthy) -
  `FP8GemmTest.PerTensorActQuantSurvivesChannelOutlier`.
- Engine act absmax measured 0.3-57 (ssm_out arm) - no pathological scale.
- With the separate `ssm_in` sidecar entries removed and the hook widened
  to SSM_IN via debug env, PPL stayed sane (4.23) because no entry exists;
  the corruption appeared only with the phase2b `in_prefill_side` entries
  present AND the SSM_IN hook firing. Not bisected further (entry content
  vs phase-4 tier side effect). Do not re-enable SSM_IN without isolating
  it.

## Affected models

Only hybrids whose producer recipe leaves the recurrent projections at
BF16/F16 (Qwen3.6-35B-A3B-NVFP4, GGUF Q8_0-source hybrids). On
`Qwen3.8-27B-NVFP4` the projections are checkpoint-NVFP4, already served
by CUTLASS at prefill; the sidecar builds no F16 entries there and the
flag no-ops (verified: no sidecar log line, output unchanged).

## Measurement traps hit (this session)

- The roofline "hybrid" cell is Qwen3.6-35B, NOT Qwen3.8-27B - half a day
  of analysis targeted the wrong model's prefill before checking
  `tools/roofline/config.json`.
- A stray gitignored `./imp.conf` in the repo root pins `kv_cache.dtype=fp8`
  into every dev run started from `/src` (the #1784 fix covered the build
  context, not the dev-run cwd). Neutralized with `--config <empty file>`.
- `nsys` in `imp:toolchain` still needs `apt-get install libcap2 libdw1t64`
  + explicit QdstrmImporter (recorded 2026-08-28, hit again).
- On the 35B the 45k/13.8k-token PPL corpus no longer fits: weight caches +
  vision arena + library reserve leave <900 MiB, KV falls to the 16-block
  floor (512 tokens) regardless of `kv_cache.max_blocks`; the July sweeps
  that ran it predate today's cache set. Verdict via ~1000-token corpus
  slices, both arms per slice.
- `--perplexity` on an MTP-shipping checkpoint silently loads the MTP head
  (`speculative.mtp_k` auto resolves to 2 on single-stream): +0.79 GiB that
  tipped this model's KV pool onto the floor. `--bench` pins spec off and
  therefore fit where `--perplexity` failed. Pin `speculative.mtp_k=0` in
  PPL harnesses.

## Gates

- `FP8GemmTest.*` 6/6 green (test-compute).
- `make dev-test` 8/8; `ci_static_gates.sh` all green (filesize/lanes/alloc
  pins re-pinned: +4 LOC workspace_buffers, 2 alloc sites, 2 GPU tests).
- Sliced-corpus PPL A/B (13 814 tokens of `ppl_corpus_45k.txt` in ~1k-token
  slices, `speculative.mtp_k=0` pinned, both arms per slice): 12.0875 ->
  12.0835, **-0.03%**, ON worse in 7/14 pairs - flat. (The 200-token corpus
  read +1.0% - the known small-corpus verdict inversion.)
- pp512/pp4096 bench A/B (Qwen3.6-35B, 3 alternating rounds, fresh process
  per run, --bench = spec pinned off):

  | pair | pp512 | pp4096 | tg |
  |---|---|---|---|
  | round 1 | 10677 -> 9886 (-7.4%) | 12709 -> 12474 (-1.9%) | flat |
  | round 2 | 10494 -> 10250 (-2.3%) | 12398 -> 12378 (-0.2%) | flat |
  | round 3 | 10251 -> 10182 (-0.7%) | 12487 -> 12468 (-0.2%) | flat |

  6/6 pairs negative. **VERDICT: REFUTED e2e.** ssm_out is 16 of the 48
  FP16 MB per GDN layer (~1% window ceiling alone), and the per-chunk
  overhead (act-quant reduction pair + quantize + scale_cols + FP8 algo
  entries at every chunk M) eats it. The arm that would pay - ssm_in,
  32 MB/layer - is the one closed for the uniform-logits failure.
- degen_suite: skipped - nothing ships that changes execution
  (flag stays default-off on a refuted, unmerged branch).

## ROADMAP CLOSED (2026-08-31)

Closed unmerged per the #1774 convention; branch `perf/fp8-ssm-prefill`
is the record. Re-opening this lever requires (1) isolating the SSM_IN
corruption and (2) an act-quant scheme cheap enough to beat FP16 cuBLAS
at a 16-32 MB/layer weight-byte win - per-chunk per-tensor E4M3 was not.
