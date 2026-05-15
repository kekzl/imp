# MoE NVFP4 prefill optimization — 2026-05-10

## Result

| Model | pp512 before | pp512 after | Δ | tg256 before | tg256 after | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Coder-30B-A3B-NVFP4 | 1241 | **13046** | **×10.5** | 260 | 268 | +3.1% |
| Qwen3.6-35B-A3B-NVFP4     | 1092 | **8011**  | **×7.3**  | 225 | 232 | +3.1% |
| Gemma-4-26B-A4B-NVFP4     | ≈1000 | **9271**  | **×9.3**  | 202 | 210 | +4.0% |

Cross-engine on Qwen3-Coder NVFP4 (vLLM 0.20.2, same RTX 5090):

| | pp512 | tg256 |
|---|---:|---:|
| imp before | 1241 | 261 |
| imp after  | **13046** | **268** |
| vLLM       | 25513 | 189 |
| imp/vLLM ratio | **0.51×** | **1.42×** |

Target was 2× of vLLM (≥12750 tok/s prefill, decode preserved). **Met on the first iteration.**

The 4k-token prefill (`pp4096`) on Qwen3-Coder also lands at 10774 tok/s — the speedup holds across prompt sizes.

## What changed

The slow path on the prefill MoE dispatch (`executor_forward_moe.cu:1246` "NVFP4→FP16 batch path") was firing exclusively on every NVFP4-prequant MoE model because the fast path's predicate (`covers_ids` requiring `StorageTier::CUTLASS_NVFP4` on per-expert weights) could never be satisfied. Two reasons:

1. `cache_moe_native_nvfp4` (`executor_pre_dequant.cu:1776`) packs all per-expert NVFP4 weights into one contiguous buffer, then **frees the per-expert source allocations**, leaving each `expert_w_*[e].data == nullptr`. `register_tensor` bails on null data → all expert TensorIDs become `kInvalidTensorID`.
2. Even if `data` were valid, MoE expert weights weren't in the `wcache_.cutlass_nvfp4` map — only the 192 attention NVFP4 weights were registered there. The fast path's `infer_tier_from_wcache` lookup returned `Undefined`.

The slow path dequants 96 MiB of NVFP4 → 384 MiB of FP16 per expert per projection (3 projections × 48 layers = 144 calls per prefill, 88 % of GPU time at 2.47 ms each), then runs cuBLAS FP16 grouped batched GEMM. The fast path runs CUTLASS Sm120 NVFP4 grouped GEMM directly — no dequant, FP4 tensor cores at 4× FP16 peak.

### Fix (4 files, ~180 lines)

1. **`src/compute/gemm_cutlass_sm120.{h,cu}`** — added `convert_nvfp4_moe_scales_to_sfatom()`. Single-launch kernel that converts a contiguous `[ne, N, K/16]` row-major UE4M3 SF buffer into a contiguous `[ne, cutlass_nvfp4_sf_size(N, K)]` SfAtom-layout UE4M3 buffer. Uses `blockIdx.y` for expert id (no 128× launch overhead). Added `bool sf_borrowed` to `CutlassNvFP4Weight` so per-expert wcache entries can borrow into a shared layer-projection SfAtom buffer instead of owning per-entry `cudaMalloc` allocations; `free_cutlass_nvfp4_weight` skips `cudaFree` when `sf_borrowed`.

2. **`src/graph/executor_pre_dequant.cu`** — after `cache_moe_native_nvfp4` finishes packing one layer-projection (gate/up/down) and freeing per-expert source allocations, allocate a contiguous `nvfp4_moe_sfatom` buffer (sized `ne × cutlass_nvfp4_sf_size(N, K)`), run the conversion kernel, **re-stamp `experts[e].data` to slice into the contiguous packed buffer**, set `experts[e].on_device = true` and `tensor_scale = h_ts[e]`, and register a per-expert `wcache_.cutlass_nvfp4` entry with `sf_borrowed = true`. The follow-on `register_tensor` (no other change) now finds the entry, sets `primary_tier = StorageTier::CUTLASS_NVFP4`, and populates the handle's `payload.cutlass_nvfp4` so the fast-path `covers_ids` predicate passes.

3. **`src/graph/executor_forward_moe.cu`** — physical reorder of two adjacent `else if` branches in the MoE prefill dispatch chain. The `CUTLASS 3.x NVFP4 grouped` branch now sits **above** the `NVFP4→FP16 batch path`, so when its predicate passes (now true on every NVFP4 MoE prequant model with sm_120 + adequate VRAM) the slow dequant→cuBLAS path doesn't fire. The slow path stays as a fallback for: `IMP_NO_CUTLASS3X_MOE=1`, llm-compressor format models that need the dequant path for correctness, allocation failures, or non-sm_120 hardware. Renamed the slow path's `down_act` local to `slow_down_act` since the fast-path block above it already declares one.

### VRAM cost

Per layer-projection: one extra `nvfp4_moe_sfatom` allocation sized `ne × cutlass_nvfp4_sf_size(N, K)` bytes — for Qwen3-Coder (N=768, K=2048) that's 96 KiB × 128 experts = 12 MiB per projection × 144 projections = **+1728 MiB** total. Same total as the existing native row-major SF buffer (kept in place because the slow fallback path and the decode GEMV both consume native layout). Fits comfortably in the existing budget on a 32 GiB 5090.

### Why decode didn't regress

The change is gated entirely on the prefill `n > 1` MoE dispatch chain in `executor_forward_moe.cu`. Decode (`n == 1`) flows through `gemv_nvfp4_moe_decode` / `gemv_nvfp4_moe_swiglu_decode` which still consume `NvFP4MoEQuantResult` directly with native row-major scales — not touched. The MoE decode CUDA-graph fast-path (which is what gives this model its +234% over `--no-cuda-graphs`) is unaffected because it bypasses this code path entirely. Across three NVFP4 MoE models tg256 actually moved up 3-4 % (likely from cleaner cache state during measurement); the perf-baseline gate's Qwen3-4B Q8_0 dense decode came in at +2.4 % over baseline.

## What was NOT touched (and why)

- **CUTLASS templates** — no change. The Sm120 `MainloopSm120TmaWarpSpecializedBlockScaled` mainloop and `mxf4nvf4` block-scaled MMA are already correctly instantiated and were already firing for the 192 attention NVFP4 weights. The 0.5 % "Sm120 NVFP4 grouped" line in the original profile was that path running on attention only.
- **Decode kernels** — `paged_attention_decode_nvfp4_kernel`, `gemv_nvfp4_moe_*` are imp's moat (vLLM is 30-40 % behind on NVFP4 decode). Untouched.
- **CUDA Graph capture** — the prefill change is outside the captured region. Graphs ON vs OFF still gives 1.49× decode (threshold 1.3×) per `verify-fast`.

## Verification

- `make test-gpu` — 78/78 pass, 18 skipped (model-dependent), no regressions.
- `make verify-fast`:
  - Q8_0 dense decode: 154.75 tok/s vs baseline 151.16 (+2.4 %, within 3 % threshold).
  - Q8_0 dense prefill: 15080 tok/s vs baseline 14548 (+3.7 %, within 5 % threshold).
  - Graphs ON vs OFF decode ratio: 1.49× (threshold 1.3×).
  - Smoke prompt: "What is the capital of France?" → "Paris" (Qwen3-4B Q8_0).
- Coherence probe on Qwen3-Coder NVFP4 (the changed model) with prompt "Write a Python function that computes the factorial of n using recursion" — produced a syntactically valid `def factorial(n):` recursion implementation. No repetition loops, no NaN, no graph fallback in stderr.
- Qwen3.6-35B-A3B-NVFP4 and Gemma-4-26B-A4B-NVFP4 — all three NVFP4 MoE models the change applies to log `MoE prefill: CUTLASS 3.x NVFP4 grouped` (instead of `NVFP4→FP16 batch path`) and bench at +7-10× prefill.

## Did NOT need

- Profile-guided ncu metrics — the nsys top-kernel summary already showed `dequantize_nvfp4_moe_kernel` at 88 % of GPU time; the optimization is structural, not a kernel rewrite. Future work that wants to close the remaining ~2× gap to vLLM (e.g. writing a custom prefill GEMM with TRT-LLM-style autotune, or fusing routing into the activation quantize) WOULD need ncu. None of that is needed to pass the 2× target.
- New CUTLASS tile configs / different epilogues — the existing `<128,128,128>` tile is what's already running for attention NVFP4 GEMMs and per `memory/lever4_tile_tuning_baseline_wins_2026_05_06.md` it was already A/B-validated as the winner on this workload.
- vLLM kernel diff — the bottleneck (88 % dequant) was so dominant that the path forward was obvious without a side-by-side profile. Useful as future work if pursuing the remaining 2× headroom.

## Remaining gap to vLLM

imp now sits at 0.51× of vLLM on Qwen3-Coder NVFP4 prefill (13046 vs 25513). The path to closing the rest:

1. **Activation quantize fusion** — `quantize_fp16_to_nvfp4_cutlass_moe` is a separate launch from the GEMM. vLLM's TRT-LLM `fp4_gemm` auto-tuner picks fused variants. Estimated upside +20-40 %.
2. **Larger per-expert tiles** — Modelopt models put 4096 tokens through 128 experts at top_k=8, so per-expert M is small (avg 32). vLLM's flashinfer auto-tunes among multiple tile sizes per shape; CUTLASS Sm120 currently uses one fixed `<128,128,128>` tile. A persistent kernel with M-aware tile selection could be +30-50 %.
3. **Routing/gather fusion** — there's a chain of small ops (top-K, fused permute, gather, RMSNorm) running between transformer layers that aren't graph-captured for prefill. Capturing prefill in a CUDA graph would close hundreds of µs of launch latency per prefill.

All three are independent, pursue-if-needed work. Closing 2× → 1× of vLLM on prefill via this path is feasible but is a multi-PR investigation.
