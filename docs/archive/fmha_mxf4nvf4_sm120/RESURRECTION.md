# Resurrection: FMHA sm_120 mxf4nvf4 block-scale prefill

**Archived 2026-05-20** (Phase 2 of architecture refactor roadmap).

## What this was

An FP4 Flash Attention prefill kernel using the
`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` MMA variant with per-16-element
UE4M3 scales. Branched in `attention_dispatch.cu`: when
`IMP_FMHA_BLOCKSCALE=1` (default), routed through this kernel; otherwise
fell to the legacy `fmha_sm120_mxfp4_prefill`.

## Why it was archived

The path was never default-on for any production NVFP4 model in current
benchmarks. The legacy `fmha_sm120_mxfp4_prefill` path handles all
shapes including head_dim=96 (which the block-scale variant explicitly
falls back from). The +1.8% HD=128 win documented in
`sm120_mma_variants_2026_04_25.md` did not survive end-to-end measurement
on real NVFP4 models — it was a kernel-microbench artifact.

The Phase 2 architecture refactor removed it as part of collapsing the
attention dispatch chain to "Default cuBLAS / Sliding-Fallback / one
FMHA variant".

## How to resurrect

1. `git mv` both files back to `src/compute/`.
2. Restore the include in `attention_dispatch.cu`.
3. Restore the `if (use_blockscale) { if (fmha_sm120_mxf4nvf4_prefill(...)) return; }`
   branch.
4. Restore the `IMP_FMHA_BLOCKSCALE` config field.
5. Re-benchmark against the legacy MXFP4 path on a current NVFP4
   production model before flipping the default.

## Original source

Frozen at this PR's HEAD.
