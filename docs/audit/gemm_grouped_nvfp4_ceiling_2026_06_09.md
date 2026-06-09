# `gemm_grouped_nvfp4` MoE-prefill 41% roofline — structural-ceiling analysis (#601)

**Date:** 2026-06-09 · **Resolves:** [#601](https://github.com/kekzl/imp/issues/601)
(`gemm_grouped_nvfp4`: 41% roofline on `nvfp4-moe/pp2048`) · **Verdict:**
documented structural ceiling (the acceptance-criteria alternative). The default
path is already the best available; the one alternative kernel regresses.

## The finding

The CUTLASS 3.x NVFP4 block-scaled grouped GEMM (MoE expert FFN, prefill) reaches
41.3% of the DRAM roofline on `nvfp4-moe/pp2048` (Qwen3-30B-A3B-NVFP4), 52% on
pp512, 44% on pp4096 — at 30–48% of the prefill window.

## Why — small per-expert M → 1-wave, 23%-occupancy grid

Qwen3-30B-A3B: `top_k=8`, 128 experts, `moe_intermediate=768`, hidden=2048. At
chunk=512 each expert sees only `512·8/128 ≈ 32` tokens. The grouped GEMM
(N=768 gate/up resp. 2048 down, K=2048/768) therefore launches a small,
shallow grid:

| kernel (committed ncu, pp2048) | grid | occupancy |
|---|---|---|
| `cutlass::GemmUniversal<…BlockScaled…>` (gemm_grouped_nvfp4) | **170 CTAs** | **23%** |

Grid = 170 = exactly the SM count → **one wave**, and the block-scaled CUTLASS
collective (128×128×128 tile, `OpClassBlockScaledTensorOp`, StageCount auto-
carveout) caps occupancy at 23%. A 1-wave, 23%-occupancy kernel cannot saturate
HBM → 41% roofline. AI≈575 FLOP/B sits below the (recalibrated ½-rate) FP4
ridge, so the classifier's "memory-bound" label is right, but the limiter is
occupancy/wave-fill, not raw bandwidth demand.

## The one alternative kernel regresses (fresh 3-arm A/B, 2026-06-09)

imp ships a hand-rolled persistent **small-M grouped GEMM** (`moe.nvfp4_smallM`,
default off) with M-aware tile selection (16/32/64/128) — exactly aimed at the
M≈32 underfill. It lives in the non-device-args tier, so testing it means
disabling the default device-args dispatch. Best-of-3 isolated trials,
Qwen3-30B-A3B-NVFP4:

| Arm | pp512 (best tok/s) | pp2048 (best tok/s) |
|---|---|---|
| **A — default (device-args CUTLASS grouped)** | **19051** | **17185** |
| B — device-args off + smallM (thr=128) | 14180 | 13009 |
| C — device-args off + host-args grouped | 13964 | 12963 |

The default wins by **+25–32%**. The smallM kernel (B) is within noise of the
plain host-args grouped (C) — its better M-tiling does **not** overcome the
per-layer host dispatch overhead it is stuck behind. This independently
reproduces the 2026-05-14 result baked into the code (device-args default-ON
comment: "+11–39% pp512 vs the legacy host-args + smallM dispatch on
Qwen3-Coder / Qwen3.6 / Qwen3-30B-Modelopt / Gemma-4"). Tool:
`tools/analysis/moe_smallm_ab.sh`.

**The device-args advantage is dispatch, not GEMM math:** both tiers run the same
class of CUTLASS block-scaled grouped kernel; device-args wins by keeping the
per-expert problem shapes device-resident (no D2H sync / host relaunch per
layer). So the 23%-occupancy grouped kernel is what every path runs — and the
M-aware smallM kernel does not beat it. The tile-M underfill is therefore **not**
the dominant limiter; the occupancy/wave-fill structural limit of the sm_120
block-scaled collective at these MoE shapes is.

## Why >70% is not reachable on the existing path

- The default device-args path is already the fastest available (+25–32% over
  both alternatives), and it is the 41%-roofline kernel itself.
- Raising occupancy past 23% would require retuning the CUTLASS block-scaled
  collective (smaller tiles / more stages / cluster) for the small-M-per-expert
  regime — but the hand-rolled M-aware kernel that does exactly that (smallM)
  already fails to beat the 128-tile path, so the expected payoff is low and the
  risk (a from-scratch device-args small-M collective) is high.
- pp512 already reads 52% (higher than pp2048's 41%), i.e. the kernel is near
  its achievable envelope for these shapes; the variation across pp512/2048/4096
  (52/41/44%) is within restart/window-attribution noise.

This is the batch-prefill counterpart of the batch-1 MoE wall (#600): MoE expert
GEMMs are intrinsically small per expert, so the grouped GEMM runs shallow grids
at low occupancy. Consistent with the authoritative finding that the NVFP4
prefill lever is **attention** (FA2, #597), not the grouped GEMM (already
fused, near its structural envelope) — see
`memory/moe_prefill_20x_premise_refuted_2026_05_31`.

## Conclusion

41% roofline on `gemm_grouped_nvfp4` is a structural ceiling of small-M-per-expert
MoE prefill on the sm_120 block-scaled CUTLASS collective. The default device-args
path is already optimal among available implementations; `moe.nvfp4_smallM`
regresses and stays default-off. No code change; this analysis is the documented
resolution per the issue's acceptance criteria.
