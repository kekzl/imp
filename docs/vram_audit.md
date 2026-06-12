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
> untracked memory and this note skew. The cudaMallocAsync **pool reserve** is a
> prime residual suspect: imp sets the pool release threshold to UINT64_MAX
> (`engine_weight_upload.cpp:92`), so init-time `cudaFree`s (incl. #679's ms_ref)
> are **retained in the pool, not returned to the OS** — still counted as
> device-used. Quantified in the follow-up section below.

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
(`pre_dequant_phase3_nvfp4_decode.cu:1676`, confirmed in the run log: "freed
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
