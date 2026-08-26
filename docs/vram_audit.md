# VRAM Audit — append-only measured breakdown

Per-component VRAM accounting for imp on the single target chip (RTX 5090,
sm_120a, 32 607 MiB total). **Append-only**: never overwrite a prior run's
numbers — add a new dated section so before/after deltas stay auditable.

All numbers are **measured**, not estimated:
- `cudaMemGetInfo` device free/used at lifecycle checkpoints (`MemAccount`,
  `src/memory/mem_account.{h,cu}`), gated by `diagnostics.vram_audit`.
- A 2 ms device-used **peak sampler** runs for the whole workload, so any
  transient prefill activation / score-matrix spike is captured (not just
  steady state).
- Per-pool `note()` counters + the existing `VRAMAllocator::report()` tag table
  + the loaders' own size logs itemise what bypasses the tracker.

### Reproduce

```bash
make build
# starts imp-server with the harness, drives the fixed workload, prints tables:
bash tools/analysis/vram_audit_run.sh
# harness only, any model:
imp-server --model <m> --max-batch 8 --set runtime.max_seq_len=4096 \
  --set diagnostics.vram_audit=true --set diagnostics.vram_audit_dump=/tmp/a.txt
```

Fixed workload: **Qwen3-Coder-30B-A3B-Instruct NVFP4, ctx=4096, continuous
batching, 8 concurrent requests** (`--max-batch 8`).

---

## 2026-06-12 — baseline (main @ 13ec1f6e + audit harness)

Model: `Qwen3-Coder-30B-A3B-Instruct-FP4` (qwen3moe, 48 layers, d_model=2048,
heads=32, kv_heads=4, d_ff=5472, vocab=151936, NVFP4 Model-Optimizer, group=16,
author kv_cache_quant_algo=FP8). Workload ran clean: 24 reqs, 0 errors, 5106
stream chunks, 124.9 tok/s aggregate across 8 streams, GPU 491 W / 2670 MHz.

### Totals (device truth)

| metric | MiB | note |
|---|---:|---|
| total VRAM | 32607 | |
| **resident at steady state** | **28289** | free 4318 |
| **peak under 8-concurrent load** | **28309** | free 4298 |
| peak − resident | **+20** | **no transient prefill spike — workspaces are statically pre-allocated** |

### Lifecycle phase deltas (measured, full coverage incl. raw cudaMalloc)

| checkpoint | used MiB | free MiB | Δ MiB (phase cost) |
|---|---:|---:|---:|
| 00 pre_init (driver + CUDA primary context) | 1679.6 | 30927.0 | — |
| 01 prewarm_gemm (cuBLAS/CUTLASS prewarm) | 2355.6 | 30251.0 | +676.0 |
| 02 weights + NVFP4 decode caches | 20697.6 | 11909.0 | +18342.0 |
| 03 kv_cache + executor workspaces | 28288.7 | 4317.9 | +7591.1 |
| 04 features (prefix/graph/residual) | 28288.7 | 4317.9 | +0.0 |
| 05 post_warmup | 28288.7 | 4317.9 | +0.0 |

### Per-component breakdown (reserved vs used vs peak)

Sources: `[note]` MemAccount pool counter; `[vram_alloc]` VRAMAllocator tag
report; `[loader]` loader size log; `[budget]` vram_budget.cpp.

| component | resident MiB | peak MiB | source / notes |
|---|---:|---:|---|
| NVFP4 packed weights (18625 tensors) | 15467 | 15467 | `[note WEIGHTS]` engine "weights ~17280" incl. phase0 cache |
| CUTLASS NVFP4 SF cache — prefill prequant | 1782 | 1782 | `[loader]` phase0 (18624 tensors) |
| CUTLASS NVFP4 SF cache — decode overlay | 1800 | 1800 | `[loader]` phase3 weight cache (18625 tensors) |
| MoE micro-scales `nvfp4_moe_ms_ref` (144 allocs) | 1728 | 1728 | `[vram_alloc]` contiguous ms_ref copies (#679 already freed 1728 dup) |
| NVFP4 LM-head decode cache | 167 | 167 | `[loader]` phase3 |
| **KV block pool** (F16, 512 blk = 16384 tok, block_size=32) | **1536** | 1536 | `[note KV_BLOCK_POOL]`+`[loader]` **under-provisioned — see findings** |
| GEMM prewarm workspace (cuBLAS + CUTLASS) | 676 | 676 | `[budget]` checkpoint 01 |
| shared_workspace (max attn/ffn/moe/ssm) | 434 | 434 | `[vram_alloc]` saved 232 vs separate alloc |
| **attn_scores — legacy cuBLAS S-matrix** (32h×2496×2496 FP16) | **380** | 380 | `[vram_alloc]` materialized scores path |
| persistent_workspace (hidden+resid+norm+logits) | 53 | 53 | `[vram_alloc]` incl. logits 8×vocab |
| moe_3x_packed/sf, moe_dequant, cutlass_act data/sf | 50 | 50 | `[vram_alloc]` |
| CUDA context / driver baseline (WSL2/WDDM) | 1679 | 1679 | checkpoint 00 |
| reconciliation residual (cudaMallocAsync pool reserve + cuBLAS/CUTLASS internal + fragmentation) | ~2537 | — | device_used − Σ above; see note below |
| **TOTAL** | **28289** | **28309** | |

> Note: the per-pool `note()` counters are write-only — they do not see a raw
> `cudaFree` that follows (e.g. #679 frees the scattered ms_ref scales after the
> contiguous copy), so the `WEIGHTS` note can over-read while `nvfp4_moe_ms_ref`
> double-books the same logical bytes. The lifecycle **checkpoints** and the
> **device totals** are the ground truth; the residual absorbs both genuine
> untracked memory and this note skew.
>
> **Both halves of this note have since been measured (2026-07-29).**
>
> The double-booking is **confirmed**, and by a stronger signal than expected:
> once `--mem-report` names the charges the notes cannot see, the MoE config
> reports **102.0 % accounted — a *negative* residual of −552 MiB**. Accounted
> exceeding device-used is only possible if some bytes are counted twice, so a
> residual that can go negative proves what this note suspected. AUDIT B32.
>
> The pool-reserve suspicion is **refuted**. It read: "imp sets the release
> threshold to UINT64_MAX, so init-time frees are retained in the pool, not
> returned to the OS." Measured across three load→generate→free cycles: the
> teardown trim logs `mempool trim: reserved 8320->0 MiB used 0->0 MiB`, graph
> memory reads zero, and setting the release threshold to 0 and re-trimming
> recovers **0 MiB**. Every CUDA-level release works. The memory is nevertheless
> gone — `cudaMemGetInfo` drops by the model's footprint once and never recovers
> — because **WSL2/WDDM does not return a process's peak VRAM commitment**.
> That is a platform property, not a pool-tuning matter, and no allocator change
> can address it. AUDIT B36. Careful with the obvious check: a `cudaMalloc`
> succeeding proves nothing here (28 GiB succeeds with 22.6 GiB reported free —
> the driver oversubscribes into host memory); time it instead, 1531 GB/s
> resident against 237 GB/s spilled.

### Not consumers (verified, so we don't chase phantoms)

- **FP8 prefill cache budgeted at 9861 MiB but NOT allocated** (`Phase-4 wcache
  actual: fp8=0`; "FP8 prefill disabled for native NVFP4"). The `fp8=9861` line
  in the budget log is a reservation ceiling, never realised on sm_120.
- **CUDA-graph buffers, prefix cache, SSM/GDN state: ~0 MiB** for this model
  (checkpoint 04 = +0; dense qwen3moe has no GDN/SSM, residual buffer off).

### Key measured findings

1. **Peak ≈ resident (+20 MiB).** The activation / materialized-scores spike the
   audit worried about is **already statically allocated** (attn_scores 380 +
   shared_workspace 434). There is no transient prefill peak to cap; reducing it
   means shrinking the *resident* workspaces, not bounding a spike.
2. **Weights + NVFP4 overlay dominate: ~20 944 MiB (74%)** = packed 15467 +
   SF caches 3582 + ms_ref 1728 + LM-head 167.
3. **KV is min-sized, not over-sized: 1536 MiB / 16384 tokens total.** At
   batch=8 that is only ~2048 tok/seq — the requested 8×4096 does **not** fit;
   the budget floor (`min(16384, max_seq_len×4)`) drove it. KV is a *capacity*
   lever (raise to use the free 4.3 GB), not a VRAM-reduction target. Model
   declares FP8 KV; imp keeps F16 by default (`--kv-fp8` halves it, correctness
   varies by family).
4. **7721 MiB of weights are "decode-redundant"** (overlay covers decode) but
   kept resident for M>1 prefill (`phase4` upper bound, not freeable as-is).
5. **Only 4318 MiB free.** Headroom is thin; the levers below are about buying
   back that headroom (for more KV/context or larger models), since peak does
   not exceed resident.

### Follow-up: cudaMallocAsync pool reserve — MEASURED, refuted as a lever

Hypothesis: the ~2.4 GiB residual is freed-but-retained pool memory (the #679
ms_ref `cudaFree`s held by the UINT64_MAX release threshold), reclaimable with
`cudaMemPoolTrimTo`. **Measured on the same workload:**

| | reserved MiB | used MiB | trimmable MiB |
|---|---:|---:|---:|
| default mempool at init_complete | 17280 | 17249 | **31** |
| after `cudaMemPoolTrimTo(0)` | 17248 | 17249 | ~0 |

The pool is ~fully *used* (live NVFP4 weights 15467 + ms_ref copy 1728 ≈ pool
used); only **31 MiB** is slack and the trim reclaimed **32 MiB**. **Refuted.**
The #679-freed scatter scales did NOT accumulate as pool slack. The ~2.4 GiB
residual is therefore **CUDA primary context + cuBLAS/CUTLASS internal
reservations + WSL2/WDDM driver overhead** (baseline checkpoint 00 alone is
1679 MiB), which is driver-owned and not cleanly reclaimable.

### Follow-up: Lever-A (`nvfp4_moe_ms_ref` slab) — already shipped, no resident win

The 1728 MiB `nvfp4_moe_ms_ref` is **not a duplicate** — #679 already frees the
scattered per-expert source scales after the contiguous copy
(`pre_dequant_phase3_nvfp4_decode.cu:512`, confirmed in the run log: "freed
1728.00 MiB duplicated per-expert micro-scales"). Making the SafeTensors loader
emit a contiguous scale slab would set `scales_contig=true` and skip the *copy*,
but the scales must be resident for NVFP4 decode either way — it only removes a
**transient init-time 2× peak**, not steady-state resident VRAM. **Not a
footprint-reduction lever.**

### Net: which levers actually reduce resident VRAM (measured)

- **attn_scores 380 MiB** — genuinely resident, removable by lowering the FA2
  prefill threshold (needs FA2 short-prefill PPL+perf parity). The one clean
  no-accuracy/no-throughput win the data supports.
- **KV FP8 ~768 MiB** — halves the 1536 MiB F16 KV; author declares FP8 KV.
  Accuracy trade-off (per-family long-context PPL gate). Better framed as a
  *context-capacity* lever (KV is min-sized, not over-sized).
- **CUTLASS NVFP4 SF caches 3582 MiB** (phase0 prefill 1782 + phase3 decode
  1800) — the only large remaining prize; needs a probe to confirm whether both
  are independently required or dedupable. Higher risk (touches prefill + decode
  GEMM correctness).
- Refuted/out-of-scope: ms_ref slab (no win), pool trim (32 MiB), decode-
  redundant prefill weights 7721 MiB (not freeable without a decode-only mode →
  prefill throughput regression).

---

## 2026-06-12 — Lever: drop Phase-0b redundant CUTLASS SF build (−1810 MiB)

Investigation of the CUTLASS NVFP4 SF caches found a **duplicate build**, not two
independent caches:
- Phase 0b (`pre_dequant_phase0_nvfp4_loader.cu`) built a CUTLASS SfAtom buffer
  for every prequant weight (18624 tensors, 1782 MiB) and stored it in
  `wcache_->cutlass_nvfp4`.
- Phase 3b (`nvfp4_decode_convert_cutlass_`) then iterates the **same**
  `wcache_->nvfp4` map and unconditionally rebuilds `cutlass_nvfp4[ptr]` for
  every entry (18625 tensors, 1800 MiB), **overwriting** Phase 0b's map entries
  without freeing them (`CutlassNvFP4Weight` has a raw `scale_factors` pointer
  and no destructor). Phase 4 snapshots the final map (Phase 3b's entries) into
  the weight handles the GEMM kernels read, so **Phase 0b's 1782 MiB was
  allocated, orphaned, and resident-dead until process exit** — never wired to
  any consumer.

Fix: Phase 0b now only seeds `wcache_->nvfp4` (the decode-GEMV registration that
Phase 3b iterates); the redundant `convert_nvfp4_to_cutlass` build is removed.
Phase 3b remains the sole, authoritative, budget-aware builder.

### Measured before/after (same fixed workload, deterministic PPL)

| metric | before (fb22c234) | after | Δ |
|---|---:|---:|---:|
| device resident MiB | 28218 | **26408** | **−1810** |
| device free MiB | 4388 | **6198** | **+1810 (+41%)** |
| Phase 3b cutlass cache | 18625 T / 1800.55 MiB | 18625 T / 1800.55 MiB | identical (live cache unchanged) |
| Phase 0b | "1782 MiB" SfAtom | "registered 18624" (no alloc) | dedup |
| **PPL (deterministic, 199 tok)** | **5.7878** (nll 1.7558) | **5.7878** (nll 1.7558) | **0 — byte-identical** |
| throughput, 8 concurrent | 124.9 tok/s | 136.8 tok/s | no regression |

The PPL match (identical mean_nll to 4 dp on a separate before/after build) plus
the unchanged Phase-3b cache confirm the saving is pure dead-memory removal with
**no correctness and no throughput regression**. New peak under load: 26408 MiB
resident, free headroom 4318 → 6198 MiB.

---

## 2026-06-12 — Lever B (attn_scores → FA2) REFUTED by measurement

Hypothesis: lower `attention.attn_scores_mib` (default 384 → the 380 MiB cuBLAS
materialized-scores buffer) so the FA2 path covers prefill, freeing 380 MiB. The
stale note in `executor_workspace_buffers.cu:268` said cuBLAS was only "~30%
faster than FMHA at n==cap"; FA2 gained ~25% since (#653/#673/#674), so parity
seemed plausible. Tested via `--set attention.attn_scores_mib=1` (threshold=129,
buffer → 1 MiB, FA2 handles all prefill); no rebuild (runtime knob).

VRAM + correctness checked out — but the **throughput gate failed catastrophically
at short prefill** (`imp-cli --bench`, 12 reps, 2-3 trials, healthy host 13801 MHz):

| metric | cuBLAS (default) | FA2 (mib=1) | Δ |
|---|---:|---:|---:|
| VRAM (attn_scores buffer) | 380 MiB | 1 MiB | −379 |
| PPL (deterministic) | 5.7878 | 5.7673 | −0.36% (FA2 marginally better) |
| **pp512** | **19600 tok/s** | **1480 tok/s** | **−92% (12× slower)** |
| pp2048 | 43004 tok/s | 42820 tok/s | −0.43% (parity) |
| tg256 (decode) | 319.3 | 319.2 | ~0% |

**Refuted.** FA2 is at parity only at the larger chunk (pp2048); at short prefill
(pp512) it is ~12× slower than cuBLAS GEMM+softmax on the small materialized
matrix — short prefill is exactly where cuBLAS wins most, and it is the common
TTFT case. The 380 MiB attn_scores buffer is **load-bearing for short-prefill
throughput** (and, separately, for hd=256 / gemma-3 correctness —
`executor_attention_prefill.cu`, the FA2 hd=256 dispatch, where FA2 mis-serves hd=256). Not reclaimable.

Lesson: pp2048 alone (−0.43%) would have green-lit a −92% pp512 regression. The
gated metric (pp512) and the short-prefill regime must be measured explicitly.

---

## Net result of this audit pass

- **Shipped: −1810 MiB** (Phase-0b SF dedup), correctness- and throughput-neutral,
  applies to every NVFP4-prequant model. Free headroom 4388 → 6198 MiB.
- **Refuted by measurement: ms_ref slab** (0, #679 done), **pool trim** (32 MiB),
  **attn_scores→FA2** (−92% pp512). Each would have been a plausible-looking lever
  on estimate alone.
- **Remaining, accuracy-gated:** KV FP8 ~768 MiB (per-family long-context PPL
  gate; better as a context-capacity lever — KV is min-sized at 1536 MiB).
- The dominant footprint (weights 14.6 GiB + NVFP4 SF/overlay + scales) and the
  ~1.7 GiB CUDA-context/cuBLAS reservation are structural / irreducible without
  an accuracy or prefill-throughput trade.

---

## 2026-06-12 — Lever C (KV-FP8 storage): viable, accuracy-gated

`--kv-fp8` switches the KV cache to FP8 E4M3 storage (f16 compute, dequant on
read — the vLLM-style path, NOT the refuted fp8-QK). The model declares
`kv_cache_quant_algo=FP8`; imp keeps F16 by default. Measured (deterministic PPL
A/B for accuracy; `imp-cli --bench` 12 reps × 2 trials for throughput):

| metric | F16 | FP8 E4M3 | Δ |
|---|---:|---:|---:|
| KV per-token | 1536 MiB / 16384 tok | 768 MiB / 16384 tok (or 1140 MiB / 24320 tok) | **−768 MiB** (or +48% context) |
| PPL — 199-tok corpus | 5.7878 | 5.7598 | −0.48% (noise, FP8 better) |
| **PPL — 3766-tok natural prose** | **16.7244** | **16.8627** | **+0.83%** (real, small) |
| pp2048 | ~43155 tok/s | ~42982 tok/s | −0.4% (parity) |
| **tg256 (decode)** | 319.3 | **321.9** | **+0.8% (faster)** |

Unlike the three refuted levers, KV-FP8 **delivers**: −768 MiB at fixed context
(or it unlocks the full 8×4096 KV that F16 cannot fit — F16 caps at 16384 total
tokens = 2048/seq at batch 8), decode is marginally *faster* (halved KV-read
bandwidth outweighs the forced `deterministic_gemm` on the FP8 path), prefill at
parity. The cost is **+0.83% PPL** on real multi-thousand-token context (the
short corpus is too short to show KV-quant compounding — filler/repetitive text
gives PPL→1.0 and must not be used to gate this).

Verdict: **viable opt-in** (`--kv-fp8`). Flipping it to default needs the same
+0.83%-class long-context gate per model family (correctness varies by family —
the engine's F16 default is deliberate); this run validates only Qwen3-Coder-30B.
It is best understood as a **context-capacity** lever here, since the F16 KV is
already small (1536 MiB) and min-sized — FP8 lets the requested 8×4096 actually
fit rather than reclaiming idle VRAM.
