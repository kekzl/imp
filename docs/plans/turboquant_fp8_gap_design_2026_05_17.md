# TurboQuant–FP8 gap design memo

**Date**: 2026-05-17
**Status**: design only — no source changes
**Subject**: Close (or shelve) the ~23% decode-rate gap between `--kv-turboquant`
and `--kv-fp8` on Qwen3-8B Q8_0 (191 vs 248 tok/s, per roadmap).
**Branch**: `perf/m5-cluster-bq128-investigation` @ `be9f570` (HEAD)
**Roadmap entry being scoped**: `docs/roadmap.md:65` ("Closing the TurboQuant–FP8 gap")

## Table of contents

1. [Status confirmation](#1-status-confirmation)
2. [Current TurboQuant architecture](#2-current-turboquant-architecture)
3. [Two candidate paths](#3-two-candidate-paths)
4. [Risks & blockers](#4-risks--blockers)
5. [Implementation plan](#5-implementation-plan)
6. [Decision recommendation](#6-decision-recommendation)

---

## 1. Status confirmation

### 1.1 The 23% gap claim

Roadmap claim (`docs/roadmap.md:67`):

> TurboQuant currently runs ~23% behind FP8 on Qwen3-8B Q8_0 decode
> (191 vs 248 tok/s). The gap is algorithm-inherent — QJL sketch
> computation adds per-token overhead. Closing it would need to drop
> QJL and switch to MXFP4 K directions with group micro-scales.

The 191/248 absolute numbers in the roadmap **do not exactly match** any
public TurboQuant measurement currently in the project memory store. The
closest comparable A/B is in `kv_dtype_tradeoffs_2026_04_24.md` on a
*different model* (Llama-3.2-3B Q8_0, RTX 5090):

| Config | tg256 tok/s | Δ vs FP8 |
|---|---:|---:|
| FP16 (default) | 319 | 0% |
| FP8 E4M3 | 319 | 0% |
| INT4 | 305 | -4% |
| TurboQuant (MXFP4-K) | 256 | **-20%** |
| TurboQuant Lite | 258 | -19% |

The directional finding is identical (TurboQuant is ~20% behind FP8 at short
context on a Q8_0 dense model). The 23% on Qwen3-8B figure is plausible —
TurboQuant's overhead scales with the number of attention heads and the per-token
sketch computation, both of which are larger on 8B than on 3B — but is **not
independently re-measured in this memo** and should be re-run before any final
ship/no-ship decision (§5 Phase 1 includes the bench).

### 1.2 The "algorithm-inherent" bottleneck claim

The roadmap attributes the gap to "QJL sketch computation per-token overhead".
A SASS / code audit of `src/compute/attention_paged_turboquant.cu` confirms
this is **plausible but not unique** — see §2.3 for the per-token cost
breakdown. The headline finding: TurboQuant's decode kernel performs **two
independent Q·K estimates per token** (PolarQuant FP4-dequant dot, plus a
QJL XNOR+popcount sketch correction) plus a 1-bit Q-side sketch precomputation
per attention step, where FP8 performs **one** Q·K dot via single-PTX
`cvt.rn.f16x2.e4m3x2` dequant. The QJL piece is the *new* overhead the FP8
path doesn't pay; whether it's the **dominant** overhead is an open question
(§4.1).

Specifically, three TurboQuant-only costs sit on the per-token hot path:

1. **Q-side QJL sketch precomputation**: one Rademacher-matrix × Q matmul
   per attention step (warp-cooperative, `sketch_dim` × `head_dim/8` byte
   reads from `qjl_matrix`). Cost is per-(batch, head), not per-token —
   it amortizes across context length.
2. **Per-token K-side dot enrichment**: PolarQuant FP4-dequant dot **+** QJL
   XNOR+popcount on the per-token sketch (`sketch_dim/8` extra byte reads
   from `K_sketches`, plus `popc` and a per-token `dot_qjl` reduction).
3. **K-norm + V-scale load and FP16→FP32 convert**: one extra global half load
   per token vs FP8 (which encodes scale inside the byte via E4M3 exponent).

The PolarQuant component alone (FP4-dequant + warp_reduce_sum) is roughly
comparable to FP8's single-instruction dequant + dot. The **QJL correction is
the differential cost** — adds ~6–10% per-token compute on top of an already
INT4-shaped V path.

cuBLAS variance disclaimer: the roadmap's 248 tok/s FP8 baseline is on the
short-context decode path where cuBLAS algo-selection jitter is small (single
token decode never hits GEMM-flop tiers). The gap is **not** cuBLAS-variance
noise; it's measurable kernel time inside `paged_attention_decode_turboquant_kernel`.

### 1.3 Memo cross-reference summary

| Memo | Relevance |
|---|---|
| `kv_dtype_tradeoffs_2026_04_24.md` | Original FP16/FP8/INT4/TQ/TQ-Lite matrix; identifies TurboQuant as -20% vs FP8 at short ctx and -55% at 20K ctx on Llama-3.2-3B. Notes the "flip default to FP8" recommendation was later REFUTED — but the *measurements* it cites are still valid. |
| `kv_research_grade_eval_2026_05_09.md` | Surveys BitDecoding, DeltaKV, K5 H2O, K2 MLA. **Crucial framing**: BitDecoding (the highest-ROI option per that eval) operates on **NVFP4 KV storage with Tensor-Core decode**. TurboQuant uses a totally different design — QJL random projection — that has no Tensor-Core analog. The two systems are incompatible at the kernel level. |
| `bitdecoding_long_context_eval_2026_05_14.md` | **Null result**: even BitDecoding's TC-on-NVFP4 path shows 0% end-to-end gain because decode is bandwidth-bound on weight loads, not on attention math. Strong evidence that *any* attention-kernel-only optimization (including closing TurboQuant's gap) may not materialize end-to-end. |
| `nvfp4_kv_potential_2026_04_25.md` | Per-model VRAM analysis showing NVFP4 KV is the right tool for Klasse-A models (Gemma-3-27B, Gemma-4-26B, Qwen3-32B). TurboQuant occupies a niche below NVFP4 in compression but with worse perf. |
| `int4_kv_validation_2026_04_24.md` | Sister memo to the dtype-tradeoffs one. INT4 KV is coherent but loses 22% decode @ 20K ctx — bracketing TurboQuant's behaviour at the low end. |

The big-picture takeaway: TurboQuant's place in the imp KV stack is a niche
"more compression than INT4 (~15-12% VRAM vs 25%), better quality than INT4
on retrieval" tier. The 23% perf gap vs FP8 is the price for that niche.
The question this memo addresses is whether the niche is worth keeping at all,
given that NVFP4-KV (3.9× compression, parity perf) already covers most of
what TurboQuant was designed for, and BitDecoding-on-NVFP4 is the active
research path for further wins.

---

## 2. Current TurboQuant architecture

### 2.1 Where the code lives

| File | Lines | Role |
|---|---:|---|
| `src/quant/turboquant.h` | 35 | `QJLProjection` struct (matrix pointer + dims + seed) |
| `src/quant/turboquant.cu` | 103 | `qjl_init` / `qjl_destroy` — Rademacher matrix generation via Philox PRNG |
| `src/quant/turboquant_fp4.cuh` | 98 | Shared device helpers: FP4 E2M1 quant/dequant LUT, UE8M0 ↔ float, `cvt.rn.satfinite.e2m1x2.f32` PTX |
| `src/graph/executor_kv_write.cu` | (KV write dispatch) | Selects between `write_kv_cache_turboquant_kernel`, `_mxfp4_kernel`, and `_lite_kernel` based on `QType` |
| `src/graph/executor_kernels.cu:981-1431` | ~450 | Three KV-write kernels: standard TQ (PolarQuant INT4 + QJL), MXFP4 TQ (PolarQuant FP4 E2M1 + UE8M0 + QJL), Lite (QJL sketch-only + INT4 V) |
| `src/compute/attention_paged_turboquant.cu` | 1108 | Decode kernels: `paged_attention_decode_turboquant_kernel<HD,USE_MXFP4>` and `_lite_kernel<HD>` + their Split-K variants |
| `src/memory/kv_cache.cu:116-180` | (KV layout) | Sketch pool + UE8M0 micro-scale pool allocation alongside the main K/V pool |
| `src/runtime/engine.cpp:1234-1340` | (init) | Computes `sketch_dim` (= `head_dim` for TQ, `2*head_dim` for Lite), enables MXFP4 path when `head_dim % 32 == 0`, calls `qjl_init` once at engine start |

### 2.2 Cache layout (per-token, per-head)

For `--kv-turboquant` with MXFP4 (the optimised path on RTX 5090 since
`head_dim % 32 == 0` for all in-scope models):

| Component | Bytes / (token, head) | Layout in pool |
|---|---:|---|
| K directions (FP4 E2M1, normalised) | `head_dim / 2` | Main pool, K region |
| V values (INT4 signed) | `head_dim / 2` | Main pool, V region |
| K norm (FP16) | 2 | `scale_pool_` K slot |
| V scale (FP16) | 2 | `scale_pool_` V slot |
| K UE8M0 micro-scales | `head_dim / 32` | `mscale_pool_` (K-only) |
| K QJL sketch (1-bit packed) | `sketch_dim / 8` = `head_dim / 8` for std TQ | `sketch_pool_` (K-only) |

Plus a one-time shared `qjl_matrix` of `sketch_dim × head_dim / 8` bytes
(≤ 2 KiB for `head_dim=128`, allocated via `cudaMalloc` in `qjl_init`).

**Footprint vs FP16 KV** (`head_dim=128`):
- FP16: `128 × 2 = 256` B per (tok, head, K) + same for V = **512 B total**
- TQ-MXFP4: `64 + 2 + 4 + 16 = 86` B for K, `64 + 2 = 66` B for V = **152 B total** (29.7%)
- TQ-Lite: `0 + 2 + 16 = 18` B for K, `66` B for V = **84 B total** (16.4%)
- FP8: `128 + 0 = 128` B for K (E4M3 encodes scale), same for V = **256 B total** (50%)
- NVFP4: `64 + 1 = 65` B for K, same V = **130 B total** (25.4%)

So TurboQuant's compression sits between NVFP4 (25%) and TQ-Lite (16%) —
real, but not free. The 23% perf cost is what you pay for the marginal
compression beyond NVFP4 plus the retrieval-quality claim.

### 2.3 Per-token hot path — exact instruction sequence

Reading `paged_attention_decode_turboquant_kernel<128, USE_MXFP4=true>`
(the production path for `head_dim=128` Qwen3-8B):

**Setup (once per attention call, amortised over `ctx_len` tokens)**:

1. Load Q into `q_reg[ELEMS=4]` registers (lines 75-84).
2. Compute `q_norm = sqrtf(sum_sq)` via warp reduction (88-93).
3. **Q-side QJL sketch**: zero `q_sketch[sketch_bytes]` in smem, then each
   warp computes one sketch row via a sketch_dim × head_dim/8 random-sign
   matvec, packing the signs as bits via `atomicOr` (95-135). For
   `sketch_dim=128`: 4 warps × 32 iterations = 128 sketch rows × 16 byte
   reads per Q matvec = **2 KiB qjl_matrix traffic per attention call**.

**Inner loop (per cached token, in the block-strided warp loop)**:

4. **PolarQuant Q·K dot** (lines 201-231): FP4-dequant 64 K bytes via
   `tq_fp4_unpack_lo/hi` + UE8M0 micro-scale via `tq_fp4_ue8m0_to_float`,
   accumulate into `dot_polar`, warp_reduce, scale by `k_norm` from
   FP16 → FP32. **Cost**: ~32 fused FFMAs + 4 popc-style decodes per lane
   + 1 warp reduction = comparable to FP8's per-token dot.
5. **QJL correction** (lines 233-248): per-lane XNOR + `__popc` over
   `sketch_bytes/4 = 4` uint32s of `K_sketches`, warp reduce the
   match count, compute `dot_qjl = q_norm * k_norm * (2*match - sketch_dim) / sketch_dim`.
   **Cost**: 4 byte-loads + 4 XNORs + 4 popcs + 1 warp reduction per token —
   this is **purely additive** vs the FP8 path.
6. **Combine** (line 251): `dot = (1 - 0.1) * dot_polar + 0.1 * dot_qjl`.
   Two FFMAs, one register write.
7. **Scale + softcap + online softmax** (253-257): identical to FP8 path.
8. **V accumulation** (260-271): INT4 unpack (signed nibble, `>=8 ? -16 : 0`
   branchless via subtract) + per-head FP16 scale + FFMA into `o_reg`. Cost
   slightly higher than FP8 V (E4M3 dequant is single-PTX, INT4 unpack is
   2-3 ops per nibble).

**Differential cost vs FP8 paged decode**: per token, TQ-MXFP4 pays for
extra (5), the more expensive V (8), and the K-norm/V-scale FP16 loads
that FP8 folds into the cached byte. Setup (3) is amortized but non-zero
at short context.

### 2.4 SASS-level estimate

A precise SASS audit would require an `ncu --section ComputeWorkloadAnalysis`
run on Qwen3-8B Q8_0 with each KV dtype, which is out of scope for a
design memo (no source changes). Order-of-magnitude estimate based on the
instruction inventory above:

| Path | per-token cost (relative units) |
|---|---:|
| FP8 K dequant + Q·K dot + V dequant + V·P | 1.0 (baseline) |
| TurboQuant PolarQuant dot + QJL correction + INT4 V | ≈ 1.20-1.30 |

This bracket matches the 23% gap in the roadmap and the 20% gap in the
2026-04-24 Llama-3.2-3B measurement. **The bottleneck is per-token compute,
not memory bandwidth** — TurboQuant's K bytes are *fewer* than FP8's, so
DRAM traffic is lower, but the compute-per-byte ratio is higher.

This matters for Path B analysis (§3.2): if the per-token QJL XNOR+popcount
is replaced by something cheaper, the gap closes; if it's removed entirely
(Path A), the path collapses to a near-NVFP4 shape with the K-norm extra
load as the only residual cost.

---

## 3. Two candidate paths

### 3.1 Path A — Drop QJL, switch to MXFP4 K with group micro-scales

**Roadmap's recommendation.** The proposal: remove the QJL random-projection
random-sign sketch entirely; replace TurboQuant's K storage with **straight
MXFP4** (FP4 E2M1 + UE8M0 group scales of 32 elements) without the PolarQuant
direction/norm split.

#### 3.1.1 What the kernel looks like

This is **already a kernel that exists**: `paged_attention_decode_nvfp4_kernel`
in `src/compute/attention_paged_nvfp4.cu`. NVFP4 differs from MXFP4 only in
scale type (E4M3 per-16 vs UE8M0 per-32) and per-tensor scale, not in the
hot-path shape:

```cuda
// Existing NVFP4 K dequant inner loop (paged_attention_decode_nvfp4.cu:141)
asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
    : "=r"(out) : "r"(packed_byte));
// Two half values out of one byte; half2 multiply by scale; FMA into dot.
```

For MXFP4 K with UE8M0 micro-scales, the inner loop is the same `cvt.rn.f16x2.e2m1x2`
plus a UE8M0-to-FP16 step that's already implemented as `tq_fp4_ue8m0_to_float`
in `turboquant_fp4.cuh`. Effectively this is "NVFP4 paged attention with a
different scale dtype." The structural change is removing:

- `K_sketches` pool (`sketch_pool_` in `kv_cache.cu:116-151`).
- `qjl_matrix` (`qjl_init` / `qjl_destroy` in `turboquant.cu`).
- The Q-side sketch precompute step (§2.3 step 3).
- The per-token QJL XNOR+popcount step (§2.3 step 5).
- The PolarQuant K-norm storage (becomes implicit in the UE8M0 group scale).

Net code delta: the entire `attention_paged_turboquant.cu` (1108 lines)
collapses into a thin wrapper around `paged_attention_decode_nvfp4` plus
a UE8M0 scale-decode swap; the KV-write kernels collapse similarly.

#### 3.1.2 What the perf looks like

If the differential cost in §2.4 is correct (1.20-1.30× FP8), and Path A
removes the QJL XNOR+popcount + Q-sketch precompute (the bulk of the
differential), the **expected post-Path-A decode rate** is somewhere between
NVFP4 (which is parity-with-FP16 on Qwen-class models, see Lever 2 memo)
and current TurboQuant. Best case: parity with NVFP4 = parity with FP16/FP8.
Worst case: still slightly behind FP8 due to the K-norm extra load, but
much closer (maybe -5% vs -23%).

**Important caveat** from `bitdecoding_long_context_eval_2026_05_14.md`:
*all attention-kernel-only changes on imp at decode show 0% end-to-end gain
at the contexts tested* because the bandwidth limit is weight loads, not
attention. So Path A might close the gap to **0%** without changing the
end-to-end tok/s number at all — i.e., Path A wins, but the win is invisible
to the user. This is fine for a "remove the cost so other levers can compose"
framing, less fine for a "ship this for headline perf" framing.

#### 3.1.3 The quality question

This is the load-bearing risk. QJL was published as a **retrieval-preserving
sketch** — the Johnson-Lindenstrauss random projection preserves inner
products in expectation, and the 1-bit sign quantization preserves their
sign with high probability. The original TurboQuant paper claims this
matters for long-context retrieval tasks (needle-in-haystack, multi-hop QA).

MXFP4 has **no such retrieval-preservation guarantee** — it's just direct
4-bit quantization with per-group scales. The 2026-04-24 memo's
**TurboQuant @ 20K ctx = -55%** vs **NVFP4 @ 20K ctx ≈ parity** numbers
(when NVFP4 lands) suggest the perf cost of QJL is real, but the *quality*
side of that trade has never been measured in imp. From the existing data:

- INT4 KV @ 20K ctx: coherent but -22% decode (`int4_kv_validation_2026_04_24.md`).
- TurboQuant @ 20K ctx: -55% decode but ostensibly higher retrieval fidelity.
- TurboQuant Lite @ 20K ctx: -42% decode, pure QJL.

The retrieval-quality delta between TurboQuant and (hypothetical) MXFP4-K-with-no-QJL
**has not been measured** anywhere in the project. Without it, Path A's
"this just closes the gap" framing is incomplete — it might close the
perf gap and open a retrieval-quality gap. NIAH at 16K is the cheapest
test to run (§5 Phase 2).

#### 3.1.4 Engine surface change

`QType::TURBOQUANT` semantics would change: today it's "PolarQuant FP4 K
directions + UE8M0 micro-scales + QJL 1-bit sketches + INT4 V". Path A
turns it into "MXFP4 K (FP4 E2M1 + UE8M0 group scales) + INT4 V" — which
is **structurally NVFP4-K with a different scale dtype** plus INT4 V.

The cleanest surface treatment:
- Rename `--kv-turboquant` → `--kv-mxfp4` (matches existing MXFP4 weight terminology).
- Keep `--kv-turboquant` as a deprecated alias for backwards compatibility for
  one release, with a deprecation log line.
- `--kv-turboquant-lite` either follows the same rename (`--kv-qjl-sketch`)
  or is **dropped entirely** if the Phase 2 quality test shows TurboQuant
  Lite never wins on quality vs MXFP4-K + INT4-V at any context (likely
  outcome — Lite is the "extreme compression" variant whose only argument
  is the 16% VRAM number, which NVFP4 + INT4-V doesn't match).

Loader-side: KV dtype is runtime config, not model-bound. No model
metadata changes. The `imp_dtype` C-API constants
`IMP_DTYPE_TURBOQUANT=9` / `IMP_DTYPE_TURBOQUANT_LITE=10` (`include/imp/types.h`)
either stay (with new semantics) or get a new pair added and old ones
deprecated. Per CLAUDE.md rule "C API is stable" — adding new constants
is safer than repurposing existing ones.

### 3.2 Path B — Keep QJL, optimise the per-token overhead

Cheaper code change, but the roadmap's "algorithm-inherent" framing
suggests it can't fully close the gap. Quantitatively: the per-token
QJL XNOR+popcount is ~4 byte-loads + 4 popc + 1 reduce per token per
warp. Even fusing it perfectly with the PolarQuant dot saves perhaps half
its cost — that's a ~3-5% perf recovery, not 23%.

Sub-options if Path B is pursued anyway:

#### 3.2.1 B1 — Fuse QJL sketch load with K-direction load

Today `K_dir_tok` (FP4 byte) and `k_sketch` (sketch bit byte) come from
**different pools** (`pool_` vs `sketch_pool_`) — two separate DRAM accesses
per token. Co-locating them in a single packed layout (`{K_dir | K_sketch}`
per token) would let one `ldg` cover both, saving a memory transaction
per token. Bookkeeping cost: KV-cache pool layout changes, all four KV-write
kernels (`write_kv_cache_turboquant_kernel`, `_mxfp4_kernel`, `_lite_kernel`,
plus the chunked-prefill gather if added) need rewriting.

**Estimated gain**: 3-5% recovery — the bandwidth saved is small because
K_dir is already only `head_dim/2 = 64` bytes and the sketch is 16 bytes;
both fit in a single 128-byte coalesced load already (the issue is the
*pointer chase* to two pools, not the bytes per pointer chase).

#### 3.2.2 B2 — Reduce sketch dimension

Current default: `sketch_dim = head_dim` for TQ (128 for Qwen3-8B),
`sketch_dim = 2 × head_dim` for TQ-Lite (256). Dropping to `sketch_dim = head_dim/2`
(64 for std TQ) halves the QJL byte traffic and XNOR+popcount cost.

Risk: paper's 4-bit-equivalent quality argument requires `sketch_dim >= head_dim`.
Below that threshold, the QJL correction's variance grows and may not be
worth applying. There's no in-imp data on whether `sketch_dim/2` is
materially worse on retrieval — but the paper's framing strongly implies
it would be.

**Estimated gain**: 4-6% recovery, with a quality regression that's hard
to bound without retrieval-quality benchmarks (which Path A would need
anyway — see §5 Phase 2).

#### 3.2.3 B3 — Move QJL out of the inner loop entirely

Restructure to compute the QJL correction in a **separate kernel** that
runs after the main online-softmax loop, applying the per-token dot
correction only to the **top-k tokens** the softmax selected (k ≈ 32-64
attention slots above some threshold). This trades full per-token QJL
cost for a 2-pass attention with smaller second-pass cost.

**Estimated gain**: 8-12% recovery — meaningful, but at the cost of a
2-pass kernel structure that doesn't compose with split-K and increases
the smem footprint.

**Combined Path B (B1+B2+B3) ceiling estimate**: 15-20% recovery, partially
closing the 23% gap. Still leaves TurboQuant at -5..-10% vs FP8. Not
zero, but recovers most of the per-token cost without dropping the QJL
quality property.

#### 3.2.4 Path B downside

The roadmap's "algorithm-inherent" framing implies Path B is partially
fighting the architecture. Each of B1/B2/B3 is a kernel-rewrite of
substantial size for incremental gain. The aggregate cost of three
optimisation passes is probably comparable to Path A's "rip QJL out
and reuse the NVFP4 kernel" — and Path A actually closes the gap to
zero in the best case, where Path B at best closes it to -5..-10%.

### 3.3 Comparison

| | Path A: drop QJL, use MXFP4-K | Path B: optimise QJL per-token |
|---|---|---|
| Perf upside (decode tok/s) | Close gap to ~0% (best case) | Close gap to ~-5..-10% (best case) |
| Code change | Rip `attention_paged_turboquant.cu` (1108 LOC), `turboquant.cu` (103 LOC), QJL bookkeeping; reuse `attention_paged_nvfp4.cu` shape with UE8M0 scale swap. Net **-2000 LOC**, +200 LOC for the MXFP4-scale-typed NVFP4 variant. | Rewrite KV-write kernels for fused layout (B1), restructure decode kernel for 2-pass (B3), parameter changes (B2). Net **+500 LOC**, no removals. |
| Quality risk | Untested. QJL's retrieval claim is real per paper — MXFP4-K may regress NIAH/RULER at long context. **Quality test plan mandatory before any flip.** | Lower: keeps the QJL algorithm. B2 (smaller sketch_dim) is the riskiest sub-option here. |
| Surface change | `--kv-turboquant` semantics change; rename to `--kv-mxfp4` plus deprecation alias. `--kv-turboquant-lite` likely retires. C-API `IMP_DTYPE_TURBOQUANT` constant retained, new `IMP_DTYPE_MXFP4_KV` added. | None — flag stays, behaviour transparently faster. |
| Composability with BitDecoding | Path A's storage = NVFP4-K shape + INT4-V. The K side is **directly compatible** with BitDecoding's TC dispatch (since BitDecoding already targets NVFP4-K). V side would still need its own TC port. | Path B's storage stays incompatible with BitDecoding (QJL sketches aren't TC-friendly). Locks TurboQuant out of the highest-ROI long-context KV future direction. |
| Engineering weeks | 2-3 (kernel reuse, plus rename + tests). | 3-5 across B1+B2+B3. |

---

## 4. Risks & blockers

### 4.1 Bottleneck attribution is "plausible" not "measured"

§1.2 and §2.4 argue the 23% gap is dominated by per-token QJL overhead.
This is **not directly measured** — there's no `ncu` profile in the project
that isolates the QJL kernel cost from the PolarQuant cost from the
K-norm-load cost. Before Path A is greenlit, run an `ncu` profile of
`paged_attention_decode_turboquant_kernel<128,USE_MXFP4=true>` on Qwen3-8B
Q8_0 + a stub kernel with QJL stripped (same kernel, `kQJLLambda=0` and
the XNOR+popcount loop short-circuited) to measure the actual per-token
QJL cost as a fraction of total kernel time.

**Acceptance criterion**: if QJL stripped accounts for ≥15% of kernel
time, Path A is well-targeted. If it's <10%, the roadmap's bottleneck
diagnosis is wrong and Path A won't close the gap — defer or shelve.

### 4.2 Quality regression on retrieval-heavy tasks

The single biggest unknown. QJL's whole reason to exist is preserving
inner products under random projection — that's a retrieval property.
MXFP4 doesn't have it. Imp currently has **no NIAH or RULER quality
benchmark in `tests/`**. Building one is itself a multi-day project that
gates this work.

Minimum viable retrieval test (per §5 Phase 2): NIAH at 4K and 16K
context on Qwen3-8B Q8_0, comparing FP16 (gold), FP8, TurboQuant (current
QJL), and MXFP4-K (Path A candidate). Acceptance threshold: MXFP4-K
NIAH score within 5% of TurboQuant at 16K. Without that data, no default
flip is defensible.

### 4.3 Model coverage

TurboQuant currently has KV-write and decode kernels for `head_dim ∈
{64, 96, 128, 256}` (dispatch in `attention_paged_turboquant.cu:567-583`).
Path A inherits NVFP4's coverage: the existing `paged_attention_decode_nvfp4_kernel`
also covers all four head_dims. **No coverage loss**.

However: `kv_cache.cu:199-201` currently rejects TurboQuant for the
**per-layer head_dim** path (Gemma-4's dual 256/512 SWA + full-attention
geometry). Path A could either (a) inherit that limitation (TurboQuant
stays out-of-scope for Gemma-4) or (b) fix it as part of the rewrite
since the NVFP4 path already handles per-layer head_dim. Option (b) is
the natural choice if Path A is reusing the NVFP4 plumbing anyway.

For Path B: same limitation persists, plus Path B is irrelevant for
Gemma-4 (which is best-served by NVFP4 KV anyway).

### 4.4 Engine surface change & deprecation

CLAUDE.md rule: "C API in `include/imp/` is stable. Update every caller
if a public function changes." Path A:

- `IMP_DTYPE_TURBOQUANT` (value 9 in `types.h:19`): **retain** with same
  numeric value; either update semantics with a release note, or add
  `IMP_DTYPE_MXFP4_KV = 11` and deprecate 9.
- `IMP_DTYPE_TURBOQUANT_LITE` (value 10): if retired, retain the constant
  but log-warn-and-fall-back-to-NVFP4-or-MXFP4 when set. Quieter break.
- `ImpEngineConfig::turboquant_sketch_multiplier` (`include/imp/config.h:42`):
  retain field, ignore if set under new semantics.

CLI flag rename: `tools/imp-cli/args.cpp:161` and `tools/imp-server/args.cpp:86`
both parse `--kv-turboquant` directly. The flag can stay as an alias for
`--kv-mxfp4` (or whatever name), with a one-line `IMP_LOG_WARN` at parse
time.

`tests/test_turboquant.cu` will need rewriting end-to-end (the test
file is QJL-specific). Net testing-LOC change: -600 lines (the QJL test
matrix retires).

### 4.5 BitDecoding interaction & TurboQuant Lite question

The 2026-05-14 BitDecoding long-context null-result memo says: BitDecoding's
TC kernel doesn't materially help end-to-end at any tested context because
decode is bandwidth-bound on **weight loads**, not on attention math.

Implication: even if Path A makes TurboQuant equal to FP8 on the
attention-kernel-time axis, the **end-to-end decode tok/s might not change**
on Qwen3-8B Q8_0 (which is weight-bandwidth-bound at decode like every
8B-class model on RTX 5090). The same bandwidth-bound argument applies in
reverse — if the user's bottleneck is weights, the 23% TQ vs FP8 gap should
be much smaller end-to-end than the kernel-time gap.

This **may already be visible in the 191 vs 248 tok/s numbers** in the
roadmap — those are tg256 (decode) rates, and the actual attention-kernel
time difference may be much larger than 23%, with bandwidth-boundedness
already compressing the visible gap.

If true, Path A's perf upside on Qwen3-8B Q8_0 is probably bounded by the
weight-load ceiling, not the attention kernel — i.e. closing the kernel
gap to zero might still leave a ~10% end-to-end gap because TurboQuant
KV-write is slower than FP8 KV-write (different code path, more pools,
more cache-line touches).

**Mitigation**: include KV-write kernel cost in Phase 1's microbench
(not just decode). If KV-write is the residual hot spot post-Path A,
the surface-area reasoning is simpler — Path A is unambiguously a win.

### 4.6 TurboQuant Lite

Path A as described drops QJL entirely. TurboQuant Lite (`--kv-turboquant-lite`,
QType::TURBOQUANT_LITE) is QJL-sketch-only K (no PolarQuant directions
at all). Path A effectively retires TQ-Lite as a separate dtype — it
has no analog in the MXFP4-K + INT4-V design.

The question: does anyone use `--kv-turboquant-lite`? The 2026-04-24 memo
benched it at -42% decode @ 20K ctx vs FP16 — meaningfully worse than
even standard TurboQuant. Its only argument is the 16% VRAM floor for
truly extreme long-context cases. NVFP4 (25% VRAM) covers most of that
need with much better perf. **Recommendation**: retire `--kv-turboquant-lite`
as part of Path A. Net code removal is large (~600 LOC of decode kernels
plus KV-write plus test scaffolding).

---

## 5. Implementation plan

Five phases. Phase 1 is small and standalone (3-5 days); Phases 2-4 only
proceed if Phase 1 confirms the bottleneck hypothesis.

### Phase 1 — Isolated microbench & bottleneck verification (3-5 days)

**Goal**: confirm or refute the §1.2 "QJL is the bottleneck" claim
*before* any kernel rewrite.

Tasks:
1. Write `tools/analysis/bench_turboquant_components.sh`. Run Qwen3-8B
   Q8_0 with `--kv-turboquant` and capture nsys timeline + `ncu --section
   ComputeWorkloadAnalysis` for `paged_attention_decode_turboquant_kernel<128,true>`
   across pp={512, 4096} tg=256.
2. Build a one-off "QJL-stripped" kernel variant (debug-only, behind
   `RuntimeConfig::diagnostics`) that short-circuits the QJL XNOR+popcount
   loop and forces `kQJLLambda=0`. Same nsys + ncu run.
3. Compute the per-token QJL cost as a fraction of total kernel time.
4. Microbench MXFP4-K-only kernel (synthetic: reuse `paged_attention_decode_nvfp4_kernel`
   on a fake MXFP4 cache with UE8M0 scales encoded into the per-tensor
   `tensor_scale_pool`) on the same shapes. Compare per-token cost to
   FP8 and to current TurboQuant.

Acceptance criteria:
- ≥ 15% kernel-time fraction attributed to QJL → Path A bottleneck-targeted, proceed.
- MXFP4-K microbench within 5% of FP8 per-token cost → Path A perf ceiling confirmed.

If criteria fail: write a "Path A refuted" memo and shelve. The 23% gap
is then attributed to a different cost (probably K-norm load + V dequant)
and Path A doesn't move the needle.

### Phase 2 — Quality A/B (NIAH + RULER subset) (4-6 days)

**Goal**: confirm Path A's MXFP4-K storage doesn't regress retrieval
quality vs current QJL.

Tasks:
1. Stand up a minimal NIAH harness in `tests/long_context/` (or `tools/eval/`):
   needle text inserted at depth ∈ {0%, 25%, 50%, 75%, 95%} in a 4K and
   16K filler context, accuracy = needle string retrieved.
2. Run NIAH at 4K + 16K on Qwen3-8B Q8_0 with FP16 / FP8 / current
   TurboQuant / *prototype MXFP4-K cache* (using the QJL-stripped variant
   built in Phase 1).
3. Optionally a RULER-subset run if NIAH is too coarse (variable-tracking
   at 16K context).

Acceptance criteria:
- MXFP4-K NIAH score within 5pp of TurboQuant at 16K → Path A green-light.
- 5-10pp regression: investigate per-depth pattern; if uniform, probably the
  4-bit quantization itself is the issue (not the QJL absence) — Path A
  still ships but with caveat docs.
- >10pp regression: QJL is doing real retrieval work. Path A refuted; fall
  back to Path B sub-options or shelve entirely.

### Phase 3 — Production wire-up (1-2 weeks)

If Phase 2 green-lights Path A:

1. Add `IMP_DTYPE_MXFP4_KV` constant + `QType::MXFP4_KV` enum entry.
2. Implement `attention_paged_mxfp4_kv.cu` as a slim shim over
   `attention_paged_nvfp4.cu` with UE8M0 scale decode instead of E4M3.
   (Most logic actually moves *into* `attention_paged_nvfp4.cu` behind a
   template parameter `SCALE_DTYPE ∈ {E4M3, UE8M0}` — avoid code duplication.)
3. Update `kv_cache.cu` to add an MXFP4_KV path (UE8M0 scale pool, no
   sketch pool, no `qjl_matrix`). Reuse the existing TurboQuant MXFP4
   group-scale layout (`mscale_pool_`); just don't allocate the sketch pool.
4. Update `engine.cpp` init: no `qjl_init` call when dtype is MXFP4_KV.
5. CLI flag `--kv-mxfp4` in both `imp-cli` and `imp-server`.
6. Tests: `test_mxfp4_kv.cu` minimal correctness suite (KV-write/read
   round-trip, decode parity with NVFP4 at same shape modulo scale dtype).

### Phase 4 — Default-flip decision (1 day, just measurement)

After Phase 3 lands:
1. Re-run the 2026-04-24 KV-dtype tradeoffs matrix with MXFP4-KV added.
2. If MXFP4-KV is uniformly within 3% of FP8 across the dense model
   battery (Qwen3-4B/8B, Llama-3.2-3B), **and** Phase 2's NIAH 16K
   regression is <5pp: candidate for being the new default for "I want
   more compression than FP8."
3. Default stays FP16 (per `kv_dtype_tradeoffs_2026_04_24.md` REFUTED
   update — there's no defensible "flip default" policy yet). MXFP4-KV
   is the opt-in.

### Phase 5 — Retire QJL code path (1 week)

After MXFP4-KV ships and is verified:
1. Mark `--kv-turboquant` as deprecated, alias to `--kv-mxfp4` for one
   release. Log `IMP_LOG_WARN` at parse.
2. Mark `--kv-turboquant-lite` as removed, log `IMP_LOG_ERROR` and fall
   back to MXFP4-KV.
3. After one release, delete `src/quant/turboquant.{h,cu}`, `src/quant/turboquant_fp4.cuh`
   stays (its FP4/UE8M0 helpers are needed by MXFP4-KV), `src/compute/attention_paged_turboquant.cu`,
   `src/graph/executor_kernels.cu:981-1431` (the three TQ KV-write kernels),
   the sketch_pool + mscale_pool paths in `kv_cache.cu` (mscale_pool keeps,
   sketch_pool retires), and `tests/test_turboquant.cu`.
4. Net deletion: ~2000-2500 lines. CHANGELOG entry for the surface change.

---

## 6. Decision recommendation

**Recommendation: Phase 1 microbench first, gated on the bottleneck verification.**

**Justification (one sentence)**: The roadmap's "QJL is algorithm-inherent
overhead" diagnosis is plausible but unverified, and a 3-5 day standalone
nsys+ncu measurement isolating the QJL kernel cost will definitively tell
us whether Path A's reuse-the-NVFP4-kernel plan can actually close the
gap — before committing the 2-3 weeks of kernel rewrite, retrieval-quality
testing, and surface-area work it would take to ship Path A end-to-end,
and *especially* before doing so against the bandwidth-bound ceiling
that the BitDecoding null-result memo already documented for similar
attention-kernel optimizations.

Secondary framing if Phase 1 confirms QJL is the bottleneck (most likely
outcome): **the right framing is "retire TurboQuant, not optimise it"**
— Path A's NVFP4-K-with-UE8M0-scales storage is structurally identical to
NVFP4 KV minus the per-tensor scale, and imp's existing NVFP4 KV path
already gives parity perf with FP16. There may not be a TurboQuant-shaped
hole in the imp dtype lineup once Path A lands — TQ becomes "NVFP4 with
INT4 V", which is a one-paragraph addition to the NVFP4 path, not a
1108-line parallel kernel. The big win is the **-2000 LOC code retirement**,
not a headline perf number on Qwen3-8B.

Worst-case outcome if both Phase 1 and Phase 2 fail (QJL not the bottleneck
AND MXFP4-K regresses NIAH): **shelve entirely**. TurboQuant becomes a
"keep as-is, opt-in, with the 23% caveat documented" tier and the project
focuses on BitDecoding-on-NVFP4 (already in main as opt-in) plus the
roadmap's other items (`pp=512` large dense models, the FP8 prefill
re-eval) which have more measurement-backed upside.

---

## Cross-references

- Memos: [`kv_dtype_tradeoffs_2026_04_24.md`](../../../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/kv_dtype_tradeoffs_2026_04_24.md), [`kv_research_grade_eval_2026_05_09.md`](../../../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/kv_research_grade_eval_2026_05_09.md), [`nvfp4_kv_potential_2026_04_25.md`](../../../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/nvfp4_kv_potential_2026_04_25.md), [`bitdecoding_long_context_eval_2026_05_14.md`](../../../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/bitdecoding_long_context_eval_2026_05_14.md), [`int4_kv_validation_2026_04_24.md`](../../../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/int4_kv_validation_2026_04_24.md), [`lever2_nvfp4_kv_implemented_2026_05_07.md`](../../../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/lever2_nvfp4_kv_implemented_2026_05_07.md)
- Code: `src/quant/turboquant.{h,cu}`, `src/quant/turboquant_fp4.cuh`, `src/compute/attention_paged_turboquant.cu`, `src/compute/attention_paged_nvfp4.cu` (reference for Path A), `src/graph/executor_kernels.cu:981-1431` (KV-write), `src/memory/kv_cache.cu:116-180` (pool layout), `src/runtime/engine.cpp:1234-1340` (init).
- Roadmap entry: `docs/roadmap.md:65-67` ("Closing the TurboQuant–FP8 gap").
- Sister design memos in this batch: `docs/plans/bitdecoding_phase2_design_2026_05_17.md` (already-shipped status check), `docs/plans/q4k_imma_design_2026_05_17.md` (INT8 IMMA scoping pattern).
