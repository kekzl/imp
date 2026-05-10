# Prefill optimization baseline — 2026-05-10

## Reproduced numbers

```
Model:    /home/kekz/models/Qwen3-Coder-30B-A3B-Instruct-FP4 (Modelopt NVFP4 SafeTensors)
Build:    imp:test (commit f415f87, branch docs/audit-cleanup-2026-05-10)
HW:       RTX 5090, sm_120a, CUDA 13.2.1
Reps:     3 (after 1 warmup), greedy decode (T=0)

Command:
  docker run --rm --gpus all -v /home/kekz/models:/models:ro imp:test \
    imp-cli --model /models/Qwen3-Coder-30B-A3B-Instruct-FP4 \
            --bench --bench-pp 512 --max-tokens 256 --bench-reps 3

Result:
  pp 512 tokens  avg   412.47 ms  (1241.29 tok/s)
  tg 256 tokens  avg   985.87 ms  ( 259.67 tok/s)
```

Reference (memory/vllm_comparison_2026_05_10.md):
- imp:    pp512=1258, tg256=261  (re-bench: 1241 / 260 — within noise)
- vLLM:   pp512=25513, tg256=189
- gap:    20.5× on prefill, +37% imp on decode

## Init log key lines

```
NVFP4 model (Model Optimizer): group_size=16, kv_cache=FP8, exclude=49 modules
NVFP4 prequant: uploaded 18624 scale tensors to GPU
CUTLASS 3.x MoE staging: 11.00 MiB (packed=8.00, sf=3.00) max_expanded=8192   ← activation staging present
CUTLASS sm_120 NVFP4 cache (prequant): 192 tensors, 54.00 MiB                  ← attention NVFP4 → CUTLASS_NVFP4 OK
NVFP4 MoE cache: 144 tensors, 15552.07 MiB                                      ← experts on native row-major (NOT CUTLASS_NVFP4)
Phase-4 plan-ideal tiers: fp16=2 fp8=0 nvfp4=192 cutlass_nvfp4=0 mxfp4=0 fp32=0
Phase-4 wcache actual:    fp16=0 fp8=0 nvfp4=0 cutlass_nvfp4=192 cutlass_mxfp4=0 nvfp4_moe=144 ...

VRAM ledger:
  nvfp4_moe_packed_native     13824.0 MiB  (144 allocs)   ← 48 layers × 3 projections
  nvfp4_moe_ms_native          1728.0 MiB  (144 allocs)   ← native row-major UE4M3 micro-scales
  moe_3x_packed (activations)     8.0 MiB
  moe_3x_sf     (activations)     3.0 MiB
```

## Active dispatch (smoking gun)

Every prefill call logs:
```
MoE prefill: NVFP4→FP16 batch path (n=512, expanded=4096)
```

This is `executor_forward_moe.cu:1256` — the **slow** dequant→cuBLAS fallback:
1. dequant 96 MiB packed NVFP4 weights → 384 MiB FP16 per projection (3× per layer, 48 layers)
2. cuBLAS FP16 grouped batched GEMM (cannot use FP4 tensor cores)

The **fast** CUTLASS 3.x NVFP4 grouped path at `executor_forward_moe.cu:1340` is gated by:
```
covers_ids(expert_*_ids) → handle.primary_tier == StorageTier::CUTLASS_NVFP4
```
But MoE expert tensors have:
- Their per-expert `Tensor::data` set to nullptr after `cache_moe_native_nvfp4` frees the
  per-expert source allocations (`executor_pre_dequant.cu:1907-1922`).
- `register_tensor` then bails at `if (!t.data) return kInvalidTensorID`
  (`executor_pre_dequant.cu:2126`) → `expert_up_ids[e] = kInvalidTensorID`
- Even if reassigned, the slices wouldn't be in `wcache_.cutlass_nvfp4` (only the first
  192 attention NVFP4 tensors were `convert_nvfp4_to_cutlass`'d).
- And the SF buffer is in native row-major layout, not CUTLASS SfAtom layout.

So the predicate's `covers_ids` immediately fails. Slow path is the only candidate.

## Initial hypothesis (to verify with profiler)

Switching MoE prefill from dequant→FP16-cuBLAS to CUTLASS 3.x NVFP4 grouped should:
- Skip ~30 ms of dequant memory writes (55 GiB blowup → 0)
- Use 4× peak FP4 TC (3354 TOPS) instead of FP16 TC (838 TFLOPS)
- Theoretical upside: 3-5× on MoE prefill

But the comparison-memo gap is 20×, so this alone won't close it. Profile next to verify and find the rest.
