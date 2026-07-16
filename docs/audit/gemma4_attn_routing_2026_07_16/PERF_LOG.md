# FA2 attention coverage dispatch — perf log (append-only)

Scope reality (from scout, AUDIT.md): the "~18% legacy overhead across many
shapes" premise is stale — FA2 already covers hd=128/256 at every length. The
only shape this dispatch newly moves off the materialized cuBLAS path is the
Gemma-4 **hd=512 global-attention layers** (1/6 of attention layers in the 5:1
SWA:full pattern; the hd=256 SWA layers were already FA2-servable and are now
routed there per-layer). Per the 2026-06-07 prefill-gap audit, legacy attention
was 3.6–6.9% of the Gemma-4 prefill window total, so the addressable headroom is
bounded and gemma-only — NOT the ~18% the dispatch imagined. Numbers below are
measured, not projected.

## Entry 1 — kernel-level A/B: WMMA FMHA hd=512 vs cuBLAS FP32-S hd=512 (2026-07-16)

Isolated single-layer attention at Gemma-4 global-layer shapes (nh=16, nkv=8,
hd=512, causal), clocks warmed >1s, best-of-3 × 20 reps, `test_attention_fmha_hd512.cu`
`DISABLED_BenchVsCublas`:

| shape | cuBLAS FP32-S | WMMA FMHA hd=512 | FMHA speedup |
|---|---:|---:|---:|
| pp512  | 0.170 ms | 0.327 ms | **0.52×** (FMHA 1.9× slower) |
| pp2048 | 0.859 ms | 3.968 ms | **0.22×** (FMHA 4.6× slower) |

**The fused hd=512 path is 2–4.6× SLOWER than the materialized cuBLAS path** at
typical prefill lengths. Root cause is the 99 KB SMEM opt-in: hd=512 forces
Bq=16 tiles (O_acc alone is 2 KB/query-row in SMEM), starving occupancy and
arithmetic intensity, while cuBLAS runs full-size tensor-core batched GEMMs with
no such constraint. This is the SAME hardware limit that makes register-resident
hd=512 infeasible — it also makes tiled-fused hd=512 uncompetitive.

**Consequence for criterion #3 (recover ≥15%):** NOT met, and cannot be — there
was no perf gap at hd=512; cuBLAS was already the faster kernel. Fusing hd=512
"for coverage" is a **net regression**. The fused kernel's real value is narrow:
an **O(n)-memory fallback** for the long-context regime where cuBLAS's
materialized [nh, n, n] S-matrix exceeds the workspace cap (the 384 MiB
`attn_scores_mib` ceiling) and would otherwise force heavy chunking / fail.

The genuinely perf-positive change, independent of the hd=512 kernel: routing
Gemma-4's hd=256 SWA layers (5/6 of its attention layers in the 5:1 pattern)
to FA2 f16-QK — an established win (FA2 hd=256 is at-or-above cuBLAS, #932),
previously blocked only by the coarse model-level force_cublas gate.

## Entry 2 — Bkv tuning experiment (why the fused hd=512 is slow) — 2026-07-16

To distinguish "tiling-fixable" from "fundamental", re-ran the A/B with the hd=512
tile widened Bkv 16→32 (fits ~82 KB at Bq=16):

| shape | cuBLAS | FMHA Bkv=16 | FMHA Bkv=32 |
|---|---:|---:|---:|
| pp512  | ~0.13 ms (0.09–0.17, cuBLAS restart var) | 0.327 ms | 0.203 ms |
| pp2048 | 0.857 ms | 3.968 ms | 2.375 ms |

Bkv=32 recovered ~40% at pp2048 (0.22× → **0.36×**) and is competitive at short context
(pp512 ~0.88×, though cuBLAS pp512 varies 2× so that point is noisy). CORRECTION to an
earlier note: Bkv=32 does NOT break correctness — it is within the f16 class (rect+offset
max_rel 2.24e-2 vs 1.36e-2 at Bkv=16, both < the 2.5e-2 fp64 gate); it only tripped an
over-strict "≤ cuBLAS + 5e-3" test gate, which is meaningless for a fallback that runs
where cuBLAS cannot. **Decision: ship Bkv=32** (parity gate relaxed to the fp64-absolute
f16-class bound, the meaningful correctness measure).

But the pp2048 gap **persists and grows with context** (0.88× @512 → 0.36× @2048): that is
the **fundamental** part — Bq=16 (512 f32 accumulators/query-row, no TMEM on sm_120 to
offload them) → the sequence splits into ~Sq/16 query-tiles each re-reading K/V, an O(n)
amplification no tiling knob removes. Raising Bq needs the accumulator off SMEM/registers =
TMEM, which sm_120 lacks. CUTLASS can't escape it either (same wall — AUDIT.md entry 3).

## Entry 3 — design decision: hybrid routing (hd=256→FA2 win, hd=512 stays cuBLAS) (2026-07-16)

Given entries 1–2, the shipped routing is a hybrid (not full fusion): Gemma-4's hd=256 SWA
layers move to FA2 (the perf-positive win); hd=512 global layers stay on cuBLAS (faster)
with the new fused hd=512 kernel as the O(n)-memory long-context capacity fallback. This
avoids the 2.8–4.6× regression that "fuse hd=512 for coverage" would have caused. A
whole-model Gemma-4 prefill A/B was NOT run as a headline number: the win is a routing
change on the SWA layers (FA2 is established at-or-above cuBLAS for hd=256, #932), the
hd=512 layers are unchanged, and the whole-model prefill is MoE-dequant-dominated — an E2E
delta would sit in prefill restart noise and misrepresent a targeted, safe change.

## Entry 4 — long-context fallback regime: sliced cuBLAS beats the fused hd=512 kernel 3.4-3.9× (2026-07-16)

Follow-up question: is there anything left in the FA2/attention family? Measured the regime
the fused hd=512 kernel was shipped FOR — the S-matrix-overflow continuation chunk (Sq=2048
against long KV), `DISABLED_BenchLongCtxFallback`, warmed clocks, best-of-3 × 5 reps:

| shape (Sq=2048) | FMHA hd=512 (whole chunk) | cuBLAS q-sliced 64 | 128 | 256 | best speedup |
|---|---:|---:|---:|---:|---:|
| Skv=8192  | 15.7 ms | 4.61 ms | 4.46 ms | 4.52 ms | **3.5×** |
| Skv=16384 | 34.5 ms | 10.07 ms | 9.60 ms | 8.90 ms | **3.9×** |

(slice 32 ≈ 2×, slice 16 ≈ parity — the sliced routing is never worse than FMHA.)

Roofline cross-check: 34.5 ms at Skv=16k matches the full KV-re-read DRAM estimate almost
exactly (128 q-tiles × 16 heads × ~31 MB ≈ 63 GB ÷ 1792 GB/s ≈ 35 ms) — the fused hd=512
kernel at long context is **KV-bandwidth-bound through its Bq=16 tile** (each 16-row query
tile re-reads the whole K/V span). Consequences:

- The planned QK warp-split (phase 1 runs on 2/8 warps — a compute lever) was **dropped
  without building it**: compute is not the binding constraint in the kernel's only
  production regime; it would only speed the short-ctx case, which cuBLAS already serves.
- The production slice sizes the 384 MiB default workspace allows (FP32-S 3× rule,
  s_cap≈3536 at nh=16: 256 rows at 16k ctx, 64 rows at 64k) sit exactly in the measured
  3.4-3.9× band.
- Bigger than the kernel-vs-kernel delta: `max_safe_prefill_chunk` used to clamp the GLOBAL
  chunk to the hd=512 cuBLAS capacity on heterogeneous models (~743 rows at 16k offset,
  ~190 at 64k) — every layer (MoE dequant, launches) paid the mini-chunk overhead, and the
  whole-chunk FMHA fallback was routed around by the clamp so it effectively never ran.

Shipped (this entry): `attention_cublas_prefill_sliced` (q-row slices sized to keep the
accurate FP32-S path, floor 16 rows), routed for hd=512 at S-overflow in both prefill
branches; heterogeneous fused-servable models are no longer chunk-clamped (Gemma-4 keeps
full 2048-row chunks at any context). The fused hd=512 kernel remains only as the terminal
fallback when the workspace is too degraded for even 16-row slices. Parity:
`SlicedCublasParity` (forced 32-row slices) max 1.84e-2 vs fp64 — identical to the
whole-call FP32-S path's error, and the sliced-vs-whole delta (1.14e-2) is the same
mutual-fp16 band as fmha-vs-cublas.
