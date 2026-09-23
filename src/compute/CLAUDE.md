<!--
layer: L3
audience: agents
verified: 2026-09-23
commit: 9b4a84e5
-->

# src/compute - CUDA kernels

Attention, GEMM/GEMV, norms, sampling, SSM/GDN scans. `sm_120a` only; the hot path.

## Invariants

- One logical unit per `.cu`: touching a kernel re-`ptxas`es the whole translation unit. Split kernel / wrapper / instantiations when recompiles bite (`tools/check_filesize.py`).
- Every `<<<>>>` carries `IMP_CUDA_CHECK_LAUNCH()` (CI job `Launch guards`).
- Numeric code is bit-sensitive: move a kernel verbatim, and say so.
- A kernel that cannot serve its input fails loud: copy `gemm()`'s scale-less-weight guard.

## Entry points

- `attention_dispatch.cu`: prefill FMHA chain per (dtype x layer)
- `attention_fmha_sm120.cu`: register-resident FA2, the default prefill path
- `attention_paged_*.cu`: paged decode kernels, one per KV dtype
- `gemm.cu`: generic dispatch and the packed-weight guard
- `gemm_cutlass_grouped_3x.cu`: MoE prefill, the primary NVFP4 GEMM path
- `gdn_scan.cu`, `gdn_scan_tc.cu`: Gated DeltaNet recurrent scans

Decode-attention variant selection lives one level up: `src/exec/executor_attention_decode.cu` (`dispatch_record::set_attn_decode`). Prefill gate: `src/exec/executor_attention.cu`.

## Test

After `make build`: `test-compute`, `test-attention` cover this directory (`test-quant`, `test-kv`, `test-moe-gdn` cover theirs); all need a GPU.

```
docker run --rm --gpus all --entrypoint test-attention imp:test --gtest_filter='*Paged*'
```

`ctest` registers only the `unit`/`gpu`/`perf` aggregates: filter inside the binary, not with `ctest -R`. Filter and lane pitfalls: `tests/CLAUDE.md`.

## Pitfalls

- `build-dev/` holds the branch last compiled there; `git checkout` does not rebuild.
- `__launch_bounds__` moves perf in both directions: never add or remove one without `make verify-ab`.
- `compute-sanitizer` does not work on this WSL2 host; `make asan` covers host code only.
- `paged_attention_decode` split-K: `total_blocks_nosplit` must count the kernel that runs without splits; GQA ratio > 8 without multitok = `paged_attention_gqa_kernel`, batch x n_kv CTAs (#2074).

## See also

[`ARCHITECTURE.md`](../../docs/internals/ARCHITECTURE.md) (what `sm_120a` has and lacks), [`KERNELS.md`](../../docs/internals/KERNELS.md) (catalogue), [`BENCHMARKING.md`](../../docs/internals/BENCHMARKING.md) before quoting a number. Skill `sm120-cuda-expert`.
