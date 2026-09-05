<!--
layer: L3
audience: agents
verified: 2026-09-06
commit: b5de0dd7
-->

# src/compute - CUDA kernels

Attention, GEMM/GEMV, norms, sampling, SSM/GDN scans. `sm_120a` only.
This is the hot path.

## Invariants

- **One logical unit per file.** Each `.cu` is one translation unit: touching a
  kernel in a 1.5k-LOC file re-`ptxas`es all of it. Split kernel / wrapper /
  instantiations when recompiles bite. `tools/check_filesize.py` gates it.
- **No portability branches.** No other arch, no FP16 dequant in decode.
- **Every `<<<>>>` carries `IMP_CUDA_CHECK_LAUNCH()`** (CI job `Launch guards`).
- **Numeric code is bit-sensitive:** move a kernel verbatim, and say so.
- **A kernel that cannot serve its input fails loud.** Copy `gemm()`'s
  scale-less-weight guard.

## Entry points

- `attention_dispatch.cu`: picks the **prefill** FMHA chain per (dtype x layer)
- `attention_fmha_sm120.cu`: register-resident FA2, the default prefill path
- `attention_paged_*.cu`: the paged **decode** kernels, one per KV dtype
- `gemm.cu`: the generic dispatch, and the guard that refuses packed weights
- `gemm_cutlass_grouped_3x.cu`: MoE prefill, the primary NVFP4 GEMM path
- `gdn_scan.cu`, `gdn_scan_tc.cu`: Gated DeltaNet recurrent scans

**Which decode-attention variant runs is decided one level up**, in
`src/exec/executor_attention_decode.cu` (the `dispatch_record::set_attn_decode`
calls); this directory holds the kernels it selects between. The prefill gate is
`src/exec/executor_attention.cu`.

## Test

Generic targets (`make dev`, `make build`, `make verify-fast`): root `CLAUDE.md`.
After `make build`, the per-module binaries in `build/` skip the suite:
`test-compute` and `test-attention` cover `src/compute/` (`test-quant`,
`test-kv`, `test-moe-gdn` cover theirs); all need a GPU.

```
docker run --rm --gpus all -v $PWD:/src -w /src imp:test \
    ./build/test-attention --gtest_filter='*Paged*'
```

`ctest` registers only the `unit`/`gpu`/`perf` aggregates: filter inside the
binary, not with `ctest -R`. Filter and lane pitfalls: `tests/CLAUDE.md`.

## Pitfalls

- `build-dev/` carries the branch last compiled in it; `git checkout` does not
  rebuild.
- `__launch_bounds__` measured **-4.5 % to -20 %** here. Never add one blind;
  pair-measure it (`make verify-ab`).
- `compute-sanitizer` is dead on this WSL2 host; `make asan` is host-code only.

## Do not touch

`third_party/` and CUTLASS-generated code. Dependency pins live only in
`cmake/imp-deps.cmake`.

## See also

[`ARCHITECTURE.md`](../../docs/internals/ARCHITECTURE.md) (what `sm_120a` has and
lacks), [`KERNELS.md`](../../docs/internals/KERNELS.md) (catalogue),
[`BENCHMARKING.md`](../../docs/internals/BENCHMARKING.md) before quoting a
number. Skill `sm120-cuda-expert`.
