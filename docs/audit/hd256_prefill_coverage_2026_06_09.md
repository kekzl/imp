# hd=256 prefill attention "coverage" — blocked on #566, not a wiring gap (#603)

**Date:** 2026-06-09 · **Issue:** [#603](https://github.com/kekzl/imp/issues/603)
(hd≠128 models still run materialized cuBLAS attention, 6.9% of
`q4k-dense-hd256/pp2048`) · **Verdict:** the premise is a mischaracterization —
hd=256 is fully supported by the tiled kernels; the cuBLAS path is a deliberate
**correctness anchor** for hd=256 + sliding-window (#566). Closing #603 to <2%
requires fixing that correctness bug, not wiring. No code change recommended.

## The premise is inverted: the tiled kernels already support hd=256

| Prefill kernel | hd=256? |
|---|---|
| `fmha_sm120_fp8_prefill` | **yes** (instantiated hd ∈ {64,128,256}) |
| `fmha_sm120_prefill` (FP16 WMMA) | **yes** (hd ∈ {64,96,128,256}) |
| `flash_attention_blackwell` | **yes** (Br=64, hd ∈ {64,96,128,256}) |
| `attention_mxfp4_prefill` | **yes** (any hd % 32 == 0) |
| `fmha_sm120_fa2_prefill` (register-resident FA2) | no (hd=128 only) |

So the FA2 *family* declines hd≠128, but three other tiled prefill kernels
accept hd=256. The gap is **not** missing kernel coverage.

## Why gemma-3 (hd=256) stays on cuBLAS: deliberate #566 correctness anchor

`src/exec/executor_attention.cu:954-962` (verbatim):

> NOTE (#566): … The WMMA chain's hd=256 + window combination computes
> catastrophically wrong attention (gemma-3-12b at n>1024: teacher-forced
> PPL 42 vs llama.cpp 1.0; long-prompt recall answered '77' instead of '477').
> The cuBLAS masked softmax is the correctness reference and handles the
> window — keep SWA prefill here whenever the S-matrix fits.

gemma-3 has `sliding_window_pattern = 6` (`model_config.h:67`) → **5 of every 6
layers are sliding-window (SWA)**, 1 is global. The tiled hd=256 + sliding-window
path is numerically broken (#566 unresolved), so the SWA layers — the bulk of the
6.9 % — are intentionally routed to the cuBLAS masked softmax, which is correct.
The materialized path is the **correctness reference**, not dead coverage.

## What would it take to reach <2 %, and is it worth it

- The 6.9 % is dominated by the **5/6 SWA layers**. Moving them off cuBLAS
  **requires fixing the #566 hd=256+sliding-window masking bug** in a tiled kernel
  (FP8-FMHA / FP16-WMMA) and validating teacher-forced PPL back to ~1.0 vs the
  cuBLAS reference. That is a correctness-debugging project on a path that today
  produces PPL 42 — not a dispatch change.
- The only **correctness-safe wiring** win is routing the **1/6 global** (no-window)
  hd=256 layers to the tiled FP8/FP16 FMHA (which is correct without a window).
  That removes ~1/6 of 6.9 % ≈ **1.1 pp** → ~5.8 %, still above the 2 % target, and
  adds per-layer dispatch complexity for a sliver of a legacy model.
- **Cost/benefit:** the affected model is gemma-3-12b **Q4_K (GGUF)** — the legacy,
  non-priority quant family (NVFP4 + SafeTensors is the priority). The lever is
  ~3 % of one model's prefill window, against the risk of re-enabling a path that
  measured PPL 42. Trading correctness headroom for ~3 % on a legacy model is a
  poor trade.

## Recommendation

Keep the cuBLAS correctness anchor. #603 is **blocked on #566** (hd=256 +
sliding-window tiled-attention correctness), not a coverage/wiring gap — re-scope
it as such or close it as won't-fix pending a priority bump for GGUF gemma-3. If
hd=256 SWA prefill ever becomes priority, the work is: reproduce #566 (gemma-3-12b
teacher-forced PPL at n>1024 on SWA layers), fix the window-mask in the FP16-WMMA
or FP8 FMHA kernel, verify parity vs `attention_cublas_prefill`, then prefer the
tiled path in the `s_matrix_fits && !prefer_fmha` branch for non-`per_layer_shapes`
SWA models.

No code change in this PR — documentation + premise correction only.
