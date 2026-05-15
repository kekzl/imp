# SF native layout audit — 2026-05-10

## cache_moe_native_nvfp4 produces (`src/graph/executor_pre_dequant.cu` ~line 1776):

- **Buffer name**: `nvfp4_moe_ms_native` (allocated at line 1828; also `nvfp4_moe_packed_native` and `nvfp4_moe_ts_native` in the same call)
- **Allocated by**: `vram_alloc_force(vram_alloc_, total_ms, "nvfp4_moe_ms_native")` at line 1828
- **Layout (by dimension)**: `[n_experts, N, K/16]` tightly packed.
  - Each expert's block occupies `expert_ms_bytes = N * (K / 16)` bytes (line 1800).
  - Within an expert, scales are row-major: row `r` of expert `e` starts at `d_ms + e * expert_ms_bytes + r * (K/16)` bytes.
  - No padding between rows or between experts.
- **Stride pattern**:
  - stride along expert: `expert_stride_ms = N * (K / 16)` bytes (line 1879, set from `expert_ms_bytes`)
  - stride along N-row within expert: `K / 16` bytes (1 UE4M3 byte per 16 FP4 elements)
  - stride along K-block: 1 byte (contiguous, no gaps)
- **M-padding**: none — `expert_ms_bytes` is exactly `N * (K / 16)`, no alignment rounding applied
- **Bytes per element**: 1 byte (UE4M3 = FP8 E4M3 stored as uint8_t)

## gemv_nvfp4_moe_decode reads (`src/quant/nvfp4_gemm.cu`):

- **Function**: `gemv_nvfp4_moe_decode_kernel` at line 855; `gemv_nvfp4_moe_decode_mr_kernel<NR>` at line 973; host launcher `gemv_nvfp4_moe_decode` at line 1012. Also used identically in `gemv_nvfp4_moe_gate_up_fused_kernel` (line 887), `gemv_nvfp4_moe_gate_up_mr_kernel<NR>` (line 932), `gemv_nvfp4_moe_swiglu_decode_kernel` (line 1061), and `gemv_nvfp4_moe_swiglu_mr_kernel<NR>` (line 1097).
- **SF input parameter**: `const uint8_t* micro_scales` passed through `NvFP4MoEQuantResult::micro_scales` + `expert_stride_ms`
- **Stride assumption** (concrete indexing from code):
  - `gemv_nvfp4_moe_decode_kernel` line 870: `MS = micro_scales + (size_t)expert_id * expert_stride_ms`; row offset applied inside `gemv_nvfp4_row` at `MS + (size_t)row * n_mb` (line 876), where `n_mb = K / kMicroBlockSize` = `K / 16`.
  - `gemv_nvfp4_moe_decode_mr_kernel` line 991: `MS = micro_scales + expert_id * expert_stride_ms + row * n_mb`.
  - All six kernels use the same two-level index: `base + expert_id * expert_stride_ms + row * (K/16)`.
  - `expert_stride_ms` is taken directly from `NvFP4MoEQuantResult::expert_stride_ms` (set by cache_moe_native_nvfp4 at line 1879 to `N * (K/16)`).
- **M-alignment expected**: 1 — no alignment constraint on N; every row starts at `row * (K/16)` bytes past the expert base, no padding.

Key constant: `kMicroBlockSize = 16` (line 31 of `nvfp4_gemm.cu`), so `n_mb = K / 16`.

## Compatible?

**Yes** — layouts are identical by construction.

`cache_moe_native_nvfp4` sets `r.expert_stride_ms = expert_ms_bytes = N * (K / 16)` (lines 1800, 1879). The decode kernels index scales as `expert_id * expert_stride_ms + row * (K / 16)`. This is exactly `[expert][row][k-block]` row-major with no padding, which is also how HF SafeTensors NVFP4 per-expert scale tensors are stored (they are copied verbatim at line 1864-1866 with `cudaMemcpyAsync` of exactly `expert_ms_bytes` bytes per expert, no reformat).

File:line evidence:
- Producer stride set: `executor_pre_dequant.cu:1800, 1879`
- Consumer stride read: `nvfp4_gemm.cu:870, 876, 991` (all kernels identical pattern)
- No intermediate convert_scales call on the `nvfp4_moe_ms_native` buffer anywhere in the decode path (the SfAtom convert only runs on the separate CUTLASS prefill buffer)

## Implications for smallM kernel:

- **Re-use existing nvfp4_moe_ms_native pointer directly**: Yes — the buffer is already in `NvFP4MoEQuantResult::micro_scales` with `expert_stride_ms` encoding the exact stride the kernel needs. No additional pointer arithmetic.

- **Need additional layout transform**: No — the native row-major UE4M3 layout `[expert][row][k-block]` is what the spec's custom kernel will consume directly (spec section "Native scale layout", line 192-210 of the spec). This avoids the SfAtom layout conversion required by the CUTLASS path.

- **New activation-quantize kernel (quantize_fp16_to_nvfp4_moe_native) must produce**: `[M_active, K/2]` packed FP4 + `[M_active, K/16]` row-major UE4M3 micro-scales — identical element ordering to the weight SF buffer, just with M_active rows instead of N rows per expert. The per-expert stride convention is the same: stride = `M_e * (K/16)` bytes, no row padding. This matches the existing decode-side convention exactly.

- **Anything that would break the spec's "drop-in" claim**: No. The spec states (line 208-210): "it reuses the layout the load-time path already produces and the decode-time path already consumes. Zero ABI risk." The audit confirms this claim: the `NvFP4MoEQuantResult` struct fields `micro_scales` + `expert_stride_ms` are the direct passthrough from `cache_moe_native_nvfp4` through to every decode GEMV kernel. The smallM kernel can receive the same `NvFP4MoEQuantResult*` pointer and read the same fields without any glue layer.

## Confidence

**High** — the layout is fully determined by two arithmetic expressions (`N * (K/16)` for expert_stride_ms, `row * (K/16)` for in-expert row offset) that appear verbatim in both the producer (executor_pre_dequant.cu:1800, 1879) and all six consumer kernels (nvfp4_gemm.cu:870, 876, 904-960, 991, 1075-1115). There is no indirection, no runtime format flag, and no conditional branch that could diverge the layout path. The HF SafeTensors per-expert scale tensors are copied byte-for-byte with no reformat (line 1864-1866).

## Open questions

None — the layout is unambiguous from the code. The only future constraint to remember: if a model's per-expert scale tensors ever arrive in a non-row-major format from the loader, `cache_moe_native_nvfp4` would need a reformat step before copying. Currently all supported HF NVFP4 models (Qwen3-Coder, Qwen3.6, Gemma-4, Nemotron-H) use the same row-major UE4M3 scale layout from llm-compressor / modelopt, so this is not a near-term concern.
