# Q4_K_M MoE Prefill Gap Analysis

**Date:** 2026-05-25
**Status:** Analysis complete. Three levers identified, none sprint-scoped.

## Problem

Q4_K_M prefill is -48 to -59% vs llama.cpp across three models:

| Model | imp pp512 | llama.cpp pp512 | Gap |
|-------|-----------|-----------------|-----|
| Gemma-3-12B Q4_K_M | 4,059 | 7,762 | -48% |
| Gemma-4-26B-A4B Q4_K_M | 4,734 | 9,932 | -52% |
| Qwen3.6-35B-A3B Q4_K_M | 3,151 | 7,644 | -59% |

Source: cross-engine bench 2026-05-24.

## Root Cause

imp's Q4_K_M MoE prefill path:
1. `dequant_gpu()`: read Q4_K (0.55 B/elem) → write FP16 (2.0 B/elem) = 2.55 B/elem
2. `gemm_moe_batched()`: read FP16 (2.0 B/elem) + compute = 2.0 B/elem
3. **Total: 4.55 B/elem round-trip**

llama.cpp's MMQ path:
1. Load Q4_K to SMEM (0.55 B/elem)
2. Decode nibbles to int8 in-register
3. dp4a/IMMA MMA
4. **Total: 0.55 B/elem**

**8.3x bandwidth overhead** explains the gap. At M=32/expert (typical for pp512 with 128 experts, top_k=8), the workload is memory-bandwidth-bound.

## What FP8 doesn't help

FP8 prefill cache is auto-disabled for Q4_K_M (sub-8-bit) since PR #219. Even when manually enabled, Q4_K attention weights are promoted to Q6_K by the quantization recipe (they pass the sub-8-bit check as Q6_K). Only MoE expert weights remain Q4_K.

A/B test: FP8 ON vs OFF = 4,767 vs 4,741 tok/s = **0.5% delta** (noise).

## Approach explored: NVFP4 decode cache → CUTLASS 3.x

With `nvfp4_decode_all=true`, Phase 3 builds an NVFP4 MoE cache from Q4_K data. CUTLASS SfAtom promotion converts the cache to per-expert CUTLASS_NVFP4 entries, enabling the CUTLASS 3.x grouped GEMM path (0.5 B/elem).

**Results:**
- Promotion works for gate+up projections (16 of 27 nvfp4_moe entries promoted)
- Down projection NOT in nvfp4_moe for Gemma-4 (GGUF layout stores down separately)
- VRAM budget limits to ~9/30 layers
- da_cache abort kills partial coverage (designed for native NVFP4 where partial = bug)

**Verdict:** Integration complexity exceeds sprint scope. Needs:
1. Down projection NVFP4 caching for Gemma-4 GGUF layout
2. da_cache partial-coverage tolerance for budget-limited GGUF models
3. Full-budget NVFP4 MoE allocation (~12 GiB for all 30 layers)

## Three levers

### A: Profile cuBLAS dequant→GEMM path (1-2 days)
Profile with ncu to check if L2 cache keeps the FP16 intermediate hot. If L2 hit rate is >90%, the effective bandwidth overhead is smaller than theoretical 8.3x. Might reveal the gap is in kernel launch overhead or gather/scatter, not bandwidth.

### B: Custom Q4_K GEMM kernel (2-3 weeks)
Port llama.cpp's MMQ approach:
- Load Q4_K blocks to SMEM
- Decode nibbles to unsigned int8 (no symmetric shift)
- dp4a or IMMA int8 MMA
- Stream-K scheduling for load balancing across SMs

This is the only lever that eliminates the bandwidth round-trip. Architectural considerations:
- dp4a path: straightforward, works on SM120
- IMMA path: refuted in Phase 3 (3.3x slower at pp1024 for dense Q4_K)
- Must integrate with MoE gather/scatter + expert routing

### C: Accept the gap
Q4_K_M is a specific quant tier. Users who need fast prefill should use NVFP4 SafeTensors instead (zero gap vs vLLM on NVFP4). Document the limitation.

## Bug found during analysis

`--no-fp8-prefill` CLI flag was broken: env var `IMP_NO_FP8_PREFILL` was set but never read by `seed_from_env`. Fixed in this PR.
