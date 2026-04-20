# CUTLASS 3.x NVFP4 Grouped GEMM — MoE Status

## Context

CUDA 13.2 Update 1 (installed 2026-04-16) improves Blackwell Grouped GEMM throughput by ~20% on large problem sizes and extends `cublasLtMatmulGrouped` to NVFP4 inputs. The NVFP4 prequant MoE prefill path previously dequanted NVFP4 weights → FP16 → standard grouped GEMM, losing the NVFP4 tensor-core throughput advantage. The per-expert decode path used `gemv_nvfp4_kpar` which turned out to produce incorrect output on Qwen3-Coder-30B-A3B-FP4 (colons / newlines / semantic garbage).

This work replaces both with CUTLASS 3.x `GroupProblemShape` + `PtrArray` grouped GEMM: NVFP4 × NVFP4 → FP16 directly, per-expert variable M with shared N/K, per-group alpha via `fusion_args.alpha_ptr_array`.

Primary beneficiary: **Qwen3-Coder-30B-A3B-FP4** (128 experts × 48 layers × 3 projections, 18,624 expert weight tensors).

## Status (2026-04-20)

### Shipped

**Kernel dispatch** — `src/compute/gemm_cutlass_grouped_3x.{cu,h}`:
- Full `gemm_grouped_cutlass_3x_nvfp4` body, not a stub
- Per-expert variable M, shared N/K, per-group alpha through `fusion_args.alpha_ptr_array`
- Persistent workspace + staging via grow-only device buffers (`s_staging`, `s_workspace`)
- Unit test `tests/test_cutlass_grouped_3x_nvfp4.cu::GroupedMatchesPerExpertSingle` — passes on RTX 5090, validates grouped output against per-expert single-GEMM reference (1% relative tolerance)

**Executor wiring** — `src/graph/executor_forward_moe.cu`:
- New branch in MoE prefill before the legacy fallback (gated by `IMP_CUTLASS3X_MOE=1`)
- Predicate: `wcache_.cutlass_nvfp4` covers all experts for gate (gated-only) + up + down projections of this layer AND persistent staging buffers are allocated
- Persistent staging: `moe_.cutlass3x_packed` (max_expanded × max_K / 2 bytes) + `moe_.cutlass3x_sf` (worst-case SfAtom with 128-row padding per expert)
- Per-projection `do_grouped` helper: quantizes activations to per-expert NVFP4 slabs, builds host pointer arrays from `wcache_.cutlass_nvfp4`, dispatches grouped GEMM
- Active-expert filter: M=0 experts are skipped (saves tile-scheduling on decode where only top_k of n_experts receive tokens)

**Upstream coverage** (already present, no new work needed):
- SafeTensors loader: `weight_map.cpp:517-534`, `safetensors_loader.cpp:638-679` — populates `expert_nvfp4_{gate,up,down}` from Model Optimizer files
- Weight upload: `weight_upload.cu:1614-1635` — uploads per-expert weight_scale / weight_scale_2 / input_scale
- CUTLASS SfAtom conversion: `executor_pre_dequant.cu:887-935` — converts per-expert micro-scales to SfAtom layout at init

### Measured (Qwen3-Coder-30B-A3B-FP4, `--no-cuda-graphs --chat-template qwen`, GPU free)

| Path | Quality | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| Legacy BEFORE E4M3 denorm fix | Garbage (`":\n\n):`) | ~68 | ~37 (garbage) |
| Legacy AFTER denorm fix | Coherent | ~77 | ~36 |
| 3.x per-expert quantize (pre-fused) | Coherent | ~353 | ~19 |
| 3.x + fused MoE quantize, forced | Coherent | ~1655 | ~24 |
| 3.x + fused quantize + 15-memcpy dispatch | Coherent | ~2028 (n>1→3.x) | ~38 (n=1→legacy) |
| 3.x + single-memcpy dispatch (auto) | Coherent | ~2753 | ~42 |
| **3.x + shared-quantize gate+up (default, current)** | Coherent | **~2891** | **~51** |

The auto-dispatch heuristic in `executor_forward_moe.cu:~1376` sends prefill chunks (n > 1) through CUTLASS 3.x grouped and decode steps (n == 1) through the legacy per-expert GEMV. Key optimizations layered in this session:

1. **Fused MoE quantize** — `quantize_fp16_to_nvfp4_cutlass_moe` in `gemm_cutlass_sm120.cu`: one kernel launch handles all expert rows at once, using a device array of per-expert SFA base pointers and binary-searching the offsets array for expert lookup. Replaced 24 per-expert launches per projection, giving +5.5× on prefill (371 → 2028).
2. **Single-memcpy dispatch** — `gemm_cutlass_grouped_3x.cu`: all 15 per-expert host arrays are assembled into one host buffer and copied with a single `cudaMemcpyAsync`, instead of 15 separate memcpy launches per grouped GEMM. Gave +36% prefill (2028 → 2753).
3. **Shared FP8 helpers** — `fp8_utils.cuh::fp8_e4m3_to_float_fast` + `float_to_fp8_e4m3` are now the single source of truth; removed 5 duplicated implementations of E4M3 decode and 2 of UE4M3 encode.

`IMP_CUTLASS3X_MOE=1` forces the 3.x path on decode too (useful for A/B testing).

Net speedup vs broken pre-fix baseline: prefill **40×** (68 → 2753), decode exceeds original pre-fix throughput (37 → 42 tok/s) with correct output.

**Root cause of garbage**: `fp8_e4m3_to_float_fast` in `src/quant/nvfp4_gemm.cu` approximated E4M3 denormal values (`exp==0`) as `2^-7 * (1 + man/8)` instead of the correct `man * 2^-9` — a **~50× inflation** on any micro-scale with exp=0. imp's own NVFP4 quantizer clamps scales to `>= 2^-9` so self-quantized models were unaffected, but Model Optimizer prequant scales can hit the denormal range, corrupting the per-expert GEMV. Fix is a single-branch addition and is a pure improvement for any NVFP4-prequant model.

**Result**: Legacy path is now both correct and ~2× faster than the 3.x path. The 3.x wiring remains shipped as an opt-in scaffold (`IMP_CUTLASS3X_MOE=1`) — useful as a validated reference for zero-sync grouped MoE work, and as a backup should future models hit a failure mode the legacy path doesn't handle.

### Performance notes

The 2× decode regression is dominated by kernel-launch overhead:
- 48 layers × 3 projections × up-to-8 active experts = **~1,150 per-expert `quantize_fp16_to_nvfp4_cutlass` kernels per token** on decode
- Each quantize is tiny (M_e=1 row) so launch overhead dwarfs compute

Follow-up optimizations (not in this ship):
1. **Fuse quantize + grouped GEMM launch** — one kernel per projection that routes rows to per-expert SFA slabs via a `row_to_expert[]` lookup, eliminates 8 separate launches
2. **CUDA Graph capture on the 3.x path** — current path does D2H sync for `h_offsets` (copied from 2.x pattern); need device-side problem shape construction to unblock graphs
3. **Prefill-only fallback** — for decode, the legacy per-expert GEMV is 2× faster when correct; the bug is likely a scale-handling issue that could be fixed directly in `gemv_nvfp4_kpar` by ensuring the per-expert `tensor_scale` correctly reaches the kernel

### Dead ends confirmed
- FP16 Grouped GEMM on SM120: not supported (`SM120 TmaWarpSpecialized only F8F6F4 MMA`)
- FP16 PtrArray on SM90: CUTLASS 4.4.2 API bug in `sm90_gemm_array_tma_warpspecialized_cooperative.hpp:291`
- `cvt .e2m1x2`: blocked in CUDA 13.2 ptxas, retry with 13.3+

### Preexisting blocker (unchanged)
CUDA Graph capture on Qwen3-Coder-30B-A3B-FP4: `cudaMemcpyAsync at executor_forward_moe.cu:194 — operation failed due to a previous error during capture`. This work runs with `--no-cuda-graphs`. Fixing graph capture is a separate investigation — the MoE path has multiple D2H copies that must be removed for graph-friendly dispatch.

## Out of scope
- Hopper BF16/FP16/FP8 Grouped GEMM: only sm_100, not sm_120 (per 13.2 Update 1 release notes)
- MXFP4 grouped GEMM: not prioritized — needs MR-GPTQ calibration first (see `docs/MXFP4_QUANTIZATION.md`)
- sm_120 FP4 cuBLASLt: NVIDIA hasn't compiled FP4 kernels for consumer Blackwell (probe returns status=7). Re-test after each libcublas patch bump.

## References
- Memory: `cutlass_3x_grouped_gemm.md`, `cuda_13_2_update_1.md`, `nvfp4_prequant_status.md`
- CUTLASS Example 79d: Blackwell GeForce NVFP4 Grouped GEMM
- Single-expert NVFP4 GEMM: `src/compute/gemm_cutlass_sm120.cu`
- This work: `src/compute/gemm_cutlass_grouped_3x.cu`, `src/graph/executor_forward_moe.cu` (3.x branch), `tests/test_cutlass_grouped_3x_nvfp4.cu`. The CUTLASS 2.x GemmGrouped reference (`gemm_cutlass_grouped_sm120.cu`) was retired after cuBLAS `cublasLtMatmulGrouped` in CUDA 13.2+ became faster on Gemma-4 Q5_K_M.
