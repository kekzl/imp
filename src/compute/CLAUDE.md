<!--
layer: L3
audience: agents
verified: 2026-10-30
commit: 28185525
-->

# src/compute - CUDA kernels

Attention, GEMM/GEMV, norms, sampling, SSM/GDN scans. `sm_120a` only.

## Invariants

- One logical unit per `.cu`: a kernel edit re-`ptxas`es the whole TU. Split kernel / wrapper / instantiations when recompiles bite (`tools/check_filesize.py`).
- Every `<<<>>>` carries `IMP_CUDA_CHECK_LAUNCH()` (CI job `Launch guards`).
- Numerics are bit-sensitive: move kernels verbatim and say so.
- A kernel that cannot serve its input fails loud: copy `gemm()`'s scale-less-weight guard.

## Entry points

- `attention_dispatch.cpp`: prefill FMHA chain per (dtype x layer)
- `attention_fmha_sm120.cu`: register-resident FA2, default prefill
- `attention_paged_*.cu`: paged decode, one per KV dtype
- `gemm.cpp`: generic dispatch, packed-weight guard
- `gemm_grouped_nvfp4_tile.cu`: MoE prefill, grouped NVFP4 GEMM (device- and host-args)
- `gdn_scan.cu`, `gdn_scan_tc.cu`: Gated DeltaNet scans
- Decode-attention selection: `src/exec/executor_attention_decode.cpp` (`dispatch_record::set_attn_decode`); prefill gate: `src/exec/executor_attention_prefill.cpp`

## Test

After `make build`, GPU only: `test-compute`, `test-attention` (also `test-quant`, `test-kv`, `test-moe-gdn`).

```
docker run --rm --gpus all --entrypoint test-attention imp:test --gtest_filter='*Paged*'
```

`ctest` has only `unit`/`gpu`/`perf` aggregates: filter in the binary, not `ctest -R`. More: `tests/CLAUDE.md`.

## Pitfalls

- `build-dev/` holds the last-compiled branch; `git checkout` does not rebuild.
- `__launch_bounds__` moves perf both ways: add/remove only with `make verify-ab`.
- `compute-sanitizer` fails on this WSL2 host; `make asan` = host code only.
- `paged_attention_decode` split-K: `total_blocks_nosplit` counts the kernel that runs unsplit; GQA ratio > 8 without multitok = `paged_attention_gqa_kernel`, batch x n_kv CTAs (#2074).

## See also

[`ARCHITECTURE.md`](../../docs/internals/ARCHITECTURE.md), [`KERNELS.md`](../../docs/internals/KERNELS.md), [`BENCHMARKING.md`](../../docs/internals/BENCHMARKING.md) before quoting a number. Skill `sm120-cuda-expert`.
