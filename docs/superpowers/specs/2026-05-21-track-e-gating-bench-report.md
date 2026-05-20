# Track E gating bench — decision report

**Date:** 2026-05-21
**Branch:** main (verify on `main` HEAD = ba2f9bb)
**Hardware:** RTX 5090 sm_120a, CUDA 13.2, container `imp:test`
**Methodology:** `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 3 warmup + 10 reps, median reported

Decides whether to commit ~10-15 dev days to Track E (tiled streaming softmax
attention kernel) or accept the 1 GiB cuBLAS S-matrix workspace as a wound.

## Results summary

| Säule | Outcome |
|---|---|
| **1** cuBLAS prefill perf | 22-50 TFLOPS effective across production shapes; S-matrix grows to 8-16 GiB at seq=8192 (well past 1 GiB cap) |
| **2** FMHA prefill perf | 1.3-1.7× **slower than cuBLAS** at production seq (≥512). Bails entirely on hd=512 (Gemma-4 global). |
| **3** HW ceiling | Pipeline upper-bound: **2862 ns / 64×64 tile**. Implies ~3.5× cuBLAS at seq=2048-4096, ~3.3× at seq=8192 |
| **4** Workspace impact | Median +5.0% ctx (range +3.9% to +9.5%) by freeing 1 GiB |

**Decision: PROCEED with Track E.**

## Säule 1+2 — cuBLAS vs FMHA on production shapes

Selected rows (median of 10 reps). Full 42-row table in `/tmp/attn_full.log` from
`docker run imp:test imp-tests --gtest_filter='Matrix/AttnPrefillBench.*'`.

| Model class | seq | cuBLAS ms | cuBLAS GFLOPS | S MiB | FMHA ms | FMHA GFLOPS | FMHA/cuBLAS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-dense (nh=32 nkv=8 hd=128) | 1024 | 0.284 | 30284 | 128 | 0.453 | 18973 | **1.60×** |
| Qwen3-dense | 2048 | 1.023 | 33601 | 512 | 1.465 | 23462 | **1.43×** |
| Qwen3-dense | 4096 | 3.667 | 37476 | 2048 | 5.317 | 25850 | **1.45×** |
| Qwen3-dense | 8192 | 15.398 | 35702 | **8192** | 20.647 | 26627 | **1.34×** |
| Gemma-4-SWA (hd=256) | 2048 | 1.400 | 49095 | 512 | 2.150 | 31963 | **1.54×** |
| Gemma-4-SWA | 4096 | 5.371 | 51174 | 2048 | 8.158 | 33694 | **1.52×** |
| Gemma-4-global (hd=512) | 2048 | 0.485 | 70846 | 128 | **NaN** (bails) | — | — |
| Gemma-4-global | 8192 | 9.863 | 55741 | 2048 | **NaN** | — | — |
| Llama-3-70B-style (nh=64) | 4096 | 7.600 | 36168 | 4096 | 10.831 | 25378 | **1.43×** |

Observations:

1. **Path (b) is dead.** FMHA is consistently 30-60% slower than cuBLAS at
   production seq lengths. Dispatch flip to FMHA would regress every model
   on every production shape.
2. **FMHA can't handle hd=512.** Gemma-4 global layers require cuBLAS unconditionally,
   reinforcing the dispatch-flip is impossible.
3. **S-matrix bloat is real.** At seq=8192, S = 8-16 GiB depending on nh.
   Production caps seq at ≤~2900 (the 1 GiB constraint). Track E removes
   this cap.

## Säule 3 — HW ceiling microbench

`TILED_CEILING_BENCH | Br=64 Bkv=64 HD=128 (FP16)`:

```
Stage A cp.async K+V :   850 ns/tile  (39 GB/s per CTA, 32 KB/tile, L2-bound)
Stage B mma.sync 512 :  1555 ns/tile  (1348 TFLOPS per CTA — misleading, see below)
Stage C softmax+resc :  2862 ns/tile  (PESSIMISTIC — see below)
Pipeline serial sum  :  5267 ns/tile  ← lower bound
Pipeline max overlap :  2862 ns/tile  ← upper bound
```

### Interpretation

**Stage A (cp.async K+V load):** 32 KB per CTA in 850 ns = 39 GB/s per CTA × 170
SMs = 6.6 TB/s aggregate. Far past DRAM peak (1792 GB/s) → KV tiles hit L2.
Realistic production with unique KV tiles per CTA per iter would be
**DRAM-bound: ~3050 ns/tile** (32 KB ÷ (1792 GB/s ÷ 170 SMs)). This is the
real ceiling A.

**Stage B (mma.sync.m16n8k16 × 512):** Per-CTA mma issue rate with full
accumulator dependency chain. Reported 1348 TFLOPS is per-CTA × 170 SMs and
overcounts; effective grid-wide TFLOPS ≈ **227 TFLOPS** (well below the 838
TFLOPS Tensor-Core peak — dependency chain prevents back-to-back issue).
Real attention kernel accumulates into different tiles in parallel and can
close the gap.

**Stage C (online softmax + O rescale):** 2862 ns is **pessimistic** —
microbench has O accumulator in SMEM (smem RMW dominates). Production tiled-
streaming kernel keeps O in registers (cheap multiply), so realistic Stage
C is ~200-500 ns.

### Track E projected throughput

Per-tile cost (realistic): `max(DRAM-A=3050, B=1555, C=500) = 3050 ns/tile`.

For seq=2048 prefill, nh=32, causal:
- Tile count: 32 × (2048/64) × (2048/64) / 2 = **16,384 tiles**
- 170 SMs running concurrently: tile rate = 170/3050ns = **55.7 G tiles/sec**
- Track E time = 16384 / 55.7e9 = **0.294 ms**
- cuBLAS measured: 1.023 ms → **Track E projected speedup: 3.5×**

For seq=4096:
- Tile count: 65536
- Track E time = 1.18 ms vs cuBLAS 3.67 ms → **3.1× speedup**

For seq=8192:
- Tile count: 262144
- Track E time = 4.71 ms vs cuBLAS 15.4 ms → **3.3× speedup**

If Stage A stays L2-bound in practice (cached KV across heads/batches), the
ceiling rises to `max(B=1555, C=500) = 1555 ns/tile` → **6-7× speedup**.

## Säule 4 — Workspace freeing impact

`scripts/analyze_attention_workspace_savings.py` output (full 27 rows in
script stdout):

| Model | KV dtype | ctx before | ctx after | Δ ctx | Δ % |
|---|---|---:|---:|---:|---:|
| Qwen3-8B Q8_0 | FP16 | 151,460 | 158,742 | +7,282 | +4.8% |
| Qwen3-8B NVFP4 | FP16 | 166,024 | 173,306 | +7,282 | +4.4% |
| Qwen3.6-35B Q4_K_M | FP16 | 57,344 | 62,805 | +5,461 | +9.5% |
| Gemma-4-26B Q4_K_M | FP16 | 29,081 | 31,129 | +2,048 | +7.0% |

Aggregate (27 model × dtype configs):
- **median Δ context: +5.0%**
- range: +3.9% to +9.5%

**Workspace saving alone is borderline** — at the 5% decision threshold. Not a
strong reason for Track E on its own. But it's a real secondary win once the
perf reason carries the decision.

## Decision matrix outcome

| Bench-Outcome | Action | Match? |
|---|---|---|
| FMHA ≤5% hinter cuBLAS auf ≥3 dtype-classes | Drop cuBLAS-attention, dispatch flip | **NO** (FMHA 1.3-1.7× slower) |
| Ceiling-microbench ≥2× besser als FMHA UND cuBLAS wins ≥10% | **Track E proceed** | **YES** (ceiling ~3-7× cuBLAS) |
| Ceiling ≤1.5× FMHA UND cuBLAS wins ≥10% | Defer Track E | NO |
| Workspace-saving ≤5% extra ctx on all models | Defer Track E | borderline (median = 5%) |

**Verdict: PROCEED with Track E.** Headroom is large (3-7× cuBLAS at prefill),
path (b) is impossible (FMHA too slow + bails on hd=512), and the 1 GiB
workspace cap is a real (if secondary) wound.

## Open questions for the Track E spec

1. **Tile geometry final:** Br=64 Bkv=64 microbench is reasonable. Tune at
   implementation time vs Bq=128 / Bkv=64 for HD=128 (existing FMHA default).
2. **NVFP4/FP8-KV inner-loop:** ceiling for FP4 m16n8k64.block_scale is 268
   TOPS per `sm120_mma_variants_2026_04_25`. ~3.3× the FP16 ceiling. Worth
   benching as Säule 3b before committing — but tracks with the FP16 result.
3. **HW feature menu** (from user's table, all available + useful):
   mma.sync m16n8k16/k32/k64, ldmatrix/stmatrix.sync, cp.async double-buffer,
   L2 persisting cache for Q tile, redux.sync.max/add, optional warp-spec
   (1-2 loader + 6-7 compute warps). Stage C in microbench was the bottleneck
   — production should put O in registers.
4. **Defer or include hd=512 (Gemma-4 global)** in Track E scope? FMHA bails
   on it. cuBLAS will remain the fallback for hd=512 regardless. Recommend
   **scope Track E to hd≤256** for v1.

## Reproduce

```bash
make build
docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 imp:test \
    imp-tests --gtest_filter='Matrix/AttnPrefillBench.*:TiledAttentionCeilingBench.*'
python3 scripts/analyze_attention_workspace_savings.py
```

## Artifacts (this commit)

- `tests/bench/attention_prefill_paths_bench.{cu,h}` — Säule 1+2 harness
- `tests/test_attention_prefill_paths_bench.cu` — gtest sweep (42 configs)
- `tests/bench/tiled_attention_ceiling_bench.{cu,h}` — Säule 3 microbench
- `tests/test_tiled_attention_ceiling_bench.cu` — gtest harness
- `scripts/analyze_attention_workspace_savings.py` — Säule 4 analytical
- `docs/superpowers/specs/2026-05-21-track-e-gating-bench-report.md` — this report

Total: ~700 LOC, 1 docker rebuild cycle (~5 min), 1 GPU bench run (~30 s).
