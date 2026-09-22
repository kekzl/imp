<!--
layer: L2
audience: kernel-devs
verified: 2026-09-22
commit: 9cbb8004
-->

# Quant / dequant pipeline

Companion doc to [`ARCHITECTURE.md`](ARCHITECTURE.md): the two parallel layers that handle
quantized weights, the boundary between them, and a concrete NVFP4 walkthrough.
User-facing format choices: [`quantization.md`](../quantization.md).

## Two layers, one direction

| Layer | Where | When | What |
|---|---|---|---|
| **Hot-path dequant** | `src/quant/dequant_*.cu` | Per-forward-pass, per-token | In-place dequant for kernels that take FP16/FP8 inputs and need to materialize from a Q*_K block on demand. Tiny scratch buffers, no persistent state. |
| **Init-time pre-dequant** | `src/exec/pre_dequant_phase*.cu` | Once at `Engine::init` | Multi-phase orchestration that decides which weights live in which device tier (FP16 cache, FP8 cache, NVFP4 decode cache, MXFP4 standalone) and runs the actual format conversions. |

Both layers ultimately call kernels from `src/quant/` (`dequant_q4k_to_fp16`, `quantize_fp16_nvfp4_cutlass`, `calibrate_quantize_fp8_async`, etc.). The difference is **lifetime**: hot-path dequant is per-call ephemeral; pre-dequant produces device tensors that live as long as the engine.

## Files

### `src/quant/` - kernels + hot-path orchestration
- `dequant_fp16.cu` - generic Q*_K -> FP16 row-block kernels
- `dequant_int8.cu` - INT8 quant/dequant
- `dequant_gptq.cu` - GPTQ-specific paths
- `dequant_gpu.cu` - small device-side dequant helpers
- `fp8_quant.cu`, `fp8_utils.cu`, `fp8_utils.cuh` - FP8 (E4M3) quant/dequant + calibration
- `nvfp4_quant.cu`, `nvfp4_gemm.cu` - NVFP4 quant + GEMM
- `mxfp4_gemm.cu` - MXFP4 dense GEMV/GEMM
- `quant_gemm.cu`, `quant_gemm.h`, `quant_types.h` - shared types + dispatch shims
- `turboquant_fp4.cuh` - UE8M0/FP4 helpers, kept for MXFP4-KV after TurboQuant was retired (PR #251)

### `src/exec/pre_dequant_*.cu` - init-time pipeline (Phase 3 of refactor)
Pure orchestration; calls `src/quant/` kernels for the actual format work.

- `executor_pre_dequant.cu` - 76 LOC orchestrator that calls each phase in order
- `pre_dequant_internal.h` - 6 shared helpers (`borrow_payload_from_wcache`, `for_each_dense_weight`, etc.)
- `pre_dequant_phase0_nvfp4_loader.cu` - Phase 0 + 0b: NVFP4 sidecar promotion + CUTLASS-NVFP4 registration. The fused-projection scale split is gated on provenance recorded by `weight_map.cpp` and asserted afterwards (`nvfp4_merged_scale_guard.h`, pure); the checkpoint's `quantization_config.ignore` partition is reconstructed and enforced in `safetensors_loader.cpp` via `model/nvfp4_module_policy.h` (also the role list `imp-quantize` writes against). A compressed-tensors W4A16 checkpoint (`input_activations: null`, never read) is served A16 at `n == 1` and W4A4 from M >= 2: `gemm.nvfp4_smallm` up to M = 32, CUTLASS NVFP4 x NVFP4 above.
- `pre_dequant_phase1_fp16_cache.cu` - Phase 1: GGUF Q*_K -> FP16 device cache (used by Q4_K_M, Q5_K_M, Q6_K, Q8_0, ...)
- `pre_dequant_phase2_fp8_cache.cu` - Phase 2: FP16 -> FP8 device tensors for the `fp8_prefill` path
- `pre_dequant_phase3_nvfp4_decode.cu` - Phase 3: NVFP4 decode-cache quantization (the bulk, 10 helpers), split further into `pre_dequant_phase3_fp8.cu`, `pre_dequant_phase3_cutlass.cu` and `pre_dequant_phase3_moe.cu` (the MoE expert stacks, where #1106 gave the `nvfp4_moe_sfatom` scale-factor slabs an owner)
- `pre_dequant_phase3c_mxfp4.cu` - Phase 3c: standalone MXFP4 (separate from NVFP4 pipeline)
- `pre_dequant_phase4_tensor_registry.cu` - Phase 4: WeightMap -> role/tier registration

## Loader enforcement (NVFP4 checkpoints)

- Compressed-tensors checkpoints declare `targets: ["Linear"]` plus an `ignore` list; the
  `config_groups` they ship set `input_activations: null` (W4A16), a field imp does not read.
- `safetensors_loader.cpp` reconstructs the `ignore` partition and logs one `NVFP4 inventory:`
  line; a Linear neither packed nor in `ignore`, or a packed Linear with no `weight_global_scale`,
  refuses to load. Modelopt checkpoints are exempt, no `ignore` list to reconstruct.
- `exclude_modules` is a hint, not a partition: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` has
  46 packed Linears with no `weight_scale_2` and loads anyway.
- Fused-projection scale splits (`qkv_proj`, `gate_up_proj`) require provenance from
  `weight_map.cpp`'s tensor-name mapper; without it the repair is indistinguishable from a
  sibling that merely failed to promote.

## Producing NVFP4 outside `imp-quantize`

NVIDIA Model Optimizer is the primary external producer (user-facing usage:
[`quantization.md`](../quantization.md)):

```bash
pip install nvidia-modelopt
python -m modelopt.llm.ptq --model Qwen/Qwen3-8B --quant nvfp4 --output ./Qwen3-8B-nvfp4/
```

| Mode | What's quantized |
|---|---|
| `nvfp4` | all linear layers |
| `nvfp4_mlp_only` | MLP / FFN layers only |
| `nvfp4_experts_only` | MoE expert layers only |
| `nvfp4_omlp_only` | MLP + output projection |

Modelopt exports declare their layout in `hf_quant_config.json` (`--format modelopt`, read by
imp only) or, via `imp-quantize --format vllm`, `quantization_config` in `config.json` (repeated
in `recipe.yaml`, read by imp **and** vLLM); compressed-tensors stores the tensor scale as a
divisor (`1 / weight_global_scale`), Modelopt as the multiplier directly. PR #88 lit up the
CUTLASS NVFP4 x NVFP4 prefill cache for Modelopt exports (Qwen3-Coder-30B, Mistral-3.2,
Qwen3.6, Gemma-4).

## GEMM dispatch: the registry pattern

`src/exec/gemm_kernel_registry.{cu,h}` is a (strategy key -> function pointer) table that
`GraphExecutor::gemm_via_handle_` (`src/exec/executor_gemm_dispatch.cu`) consults at three
sites. The FP8, NVFP4 GEMV/GEMM, MXFP4 and FP16 arms stay inside `gemm_via_handle_` (the R5
migration stopped there; the 9 registrations without a dispatch site were retired in
AUDIT_arch_2026 dispatch #8, decision (b), `docs/audit/SETTLED.md` section H).

A new key lands together with its dispatch site; `GemmKernelRegistryTest.RegistryHoldsExactlyTheProducedKeys`
pins the count at 10. Registered tiers:

- `gemm_kernel_cutlass_nvfp4.cu`: `{CUTLASS_NVFP4, F16, M>1}`, CUTLASS NVFP4 prefill GEMM (with the dual-cache MXFP4 hand-off)
- `gemm_kernel_gguf.cu`: `{FP16, <qtype>, M==1}` for Q4_K, Q5_K, Q5_1, Q8_0, Q6_K, Q4_0, Q2_K, Q3_K; GGUF small-M (dp4a + mmvq + fused-gemv fallback)
- `gemm_kernel_generic_dequant.cu`: `{FP16, NONE, M>1}`, dequant -> cuBLAS catch-all for uncached weights

## Boundary rules

When adding a new quant format:

- If the format needs **per-call decode** during forward pass: add the kernel to `src/quant/` and a dispatch entry to `src/exec/gemm_kernel_<format>.cu`.
- If the format needs **per-engine setup** (allocating tiered device tensors, computing a quant cache at load time): add a new `src/exec/pre_dequant_phase<N>_<name>.cu` file and one call from `executor_pre_dequant.cu`.
- The boundary stays clean as long as `src/quant/` stays kernels-and-helpers and `src/exec/pre_dequant_*.cu` stays orchestration.

## Where the two layers meet at runtime

1. `Engine::init_kv_cache()` (`src/runtime/engine_kv_cache_init.cpp`) -> `executor_pre_dequant.cu::pre_dequant_weights()` runs all phases sequentially, producing device tensors in the right tiers. **The call site is load-bearing, not incidental**: since #1106 the whole pipeline runs *before* the KV pool is sized, so the pool takes the measured residual instead of the caches being sized against an estimate (#1103, the reverse order left the card at 0 MiB free and cost ~7x decode).
2. Per-forward-pass: `gemm_kernel_registry` dispatches to the right `gemm_kernel_<format>.cu`, which reads the pre-dequant tier or calls `src/quant/dequant_*.cu` for an on-demand decode.

Both paths share kernels in `src/quant/`. The split between `src/quant/` and `src/exec/pre_dequant_*.cu` is **when the work happens**, not what work it is.

## NVFP4 walkthrough

Dense layers, Phase 0 through decode (`pre_dequant_phase0_nvfp4_loader.cu`, `pre_dequant_phase3_cutlass.cu`):

```
SafeTensors NVFP4 packed weights + scales
  -> loader (BF16 norms / router -> FP16, packed FP4 stays packed)
  -> Phase 0: register in NVFP4 decode cache (no re-quant)
  -> Phase 3b: CUTLASS scale-factor layout (SfAtom) for prefill
  -> prefill: CUTLASS NVFP4 GEMM via gemm_dispatch() (sm_120 tensor cores)
  -> decode:  NVFP4 GEMV (prmt register LUT, K-parallel)
```

MoE layers (Modelopt SafeTensors, per-expert; `pre_dequant_phase3_moe.cu`):

```
SafeTensors per-expert weights
  -> cache_moe_native_nvfp4 builds one contiguous [ne, N, K_packed]
     buffer per layer per projection (D2D-memcpy from per-expert tensors)
  -> per-expert tensors freed inline (32 GiB VRAM ceiling on 35B-A3B)
  -> CUDA Graphs capture cleanly via the decode fast-path
```

Without `cache_moe_native_nvfp4` the legacy FP16 dequant + cuBLAS sm_80 WMMA fallback fires per
layer per token, killing CUDA Graphs and dropping decode 5-17x.

NVFP4 KV cache (`--kv-nvfp4`) supports chunked prefill since PR #149, MXFP4 KV (`--kv-mxfp4`)
since PR #249: past chunks' K/V are gathered from the paged cache via
`paged_kv_gather_nvfp4_to_fp16` (PTX `cvt.rn.f16x2.e2m1x2` inner loop + UE4M3 scale fold) and
concatenated with the current chunk before rectangular cuBLAS attention. Hybrid GDN+MoE /
Mamba2+MoE archs (Qwen3.5/3.6, Nemotron-H) in scope since PR #156.

TurboQuant's deprecated `--kv-turboquant{,-lite}` alias flags were removed 2026-07-07 (PR #251).
