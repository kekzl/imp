---
layer: L3
audience: agents
verified: 2026-08-13
commit: 81ffa573
---

# src/compute — CUDA kernels

Attention, GEMM/GEMV, normalisation, sampling, SSM/GDN scans. Every file here
compiles for `sm_120a` only. 144 files; this is the hot path.

## Invariants

- **One logical unit per file.** Each `.cu` is one translation unit, so touching
  a kernel in a 1.5k-LOC file re-`ptxas`es the whole thing with no intra-file
  parallelism. Split kernel / launch wrapper / explicit instantiations when
  recompiles bite. `tools/check_filesize.py` gates this in CI.
- **No portability branches.** No other architecture, no FP16 dequant fallback
  in the decode hot path.
- **Every `<<<>>>` carries `IMP_CUDA_CHECK_LAUNCH()`.** CI enforces it (`Launch
  guards`); a launch-config failure must surface where it happens, not at the
  next synchronising call.
- **Numeric code is bit-sensitive.** Moving a kernel between files must be
  verbatim; state so in the commit.
- **A kernel that cannot serve its input must fail loud**, not return a
  plausible buffer. The pattern to copy is `gemm()`'s scale-less-weight guard.

## Entry points

- `attention_dispatch.cu` — picks the FMHA chain per (phase × dtype × layer)
- `attention_fmha_sm120.cu` — register-resident FA2, the default prefill path
- `attention_paged_*.cu` — paged decode attention, one per KV dtype
- `gemm.cu` — the generic dispatch, and the guard that refuses packed weights
- `gemm_cutlass_grouped_3x.cu` — MoE prefill, the primary NVFP4 GEMM path
- `gdn_scan.cu`, `gdn_scan_tc.cu` — Gated DeltaNet recurrent scans

## Build & test

```
make dev                  # incremental, 2-14 s — iterate here
make dev-test             # the real CI lane
make build                # full image; required for anything you MEASURE
make verify-fast          # ~90 s gate, the only thing that runs a kernel
                          # against a correctness or perf check
```

CI has **no GPU**. Green in GitHub Actions says nothing about a kernel.

## Pitfalls

- `build-dev/` carries whichever branch was last compiled in it. `git checkout`
  does not rebuild.
- `--gtest_filter` on a `TYPED_TEST`/`TEST_P` suite without wildcards matches
  zero tests and reports `PASSED`.
- `__launch_bounds__` overrides measured **-4.5 % to -20 %** on GEMV and
  attention paths here. Do not add one without an A/B.
- `compute-sanitizer` does not work on this WSL2 host. `make asan` covers host
  code only.
- A green test proves little in this directory: mutation-validate the test by
  breaking the kernel and checking the test notices.

## Do not touch

`third_party/`, and anything CUTLASS generates. Dependency pins are
single-sourced in `cmake/imp-deps.cmake` — bump only that file.

## See also

[`docs/internals/ARCHITECTURE.md`](../../docs/internals/ARCHITECTURE.md) for what
`sm_120a` has and lacks, [`KERNELS.md`](../../docs/internals/KERNELS.md) for the
kernel catalogue, [`BENCHMARKING.md`](../../docs/internals/BENCHMARKING.md)
before quoting any number. Skill `sm120-cuda-expert` carries the playbook.
