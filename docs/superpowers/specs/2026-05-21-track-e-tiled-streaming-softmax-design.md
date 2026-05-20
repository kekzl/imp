# Track E — tiled streaming softmax attention kernel (design)

**Date:** 2026-05-21
**Author:** Raphael Friedmann (with Claude Code)
**Status:** design — awaiting user approval, then writing-plans skill
**Gating bench:** `docs/superpowers/specs/2026-05-21-track-e-gating-bench-report.md`
**Roadmap:** completes Track E of the Phase-5 architecture refactor closeout (see `MEMORY.md` → "Refactor ALL 5 PHASES closed 2026-05-20").

## Why

The current prefill attention dispatch is split between cuBLAS (default, fast, but materialises a [n_heads × seq × seq] S-matrix capped at 1 GiB) and the FMHA fallback (chosen only for long-context / non-Gemma-4 sliding). The Säule-3 microbench shows the sm_120a hardware ceiling for tiled streaming attention is **3-7× faster than cuBLAS at production seq lengths**, while FMHA today sits at 0.5-0.7× cuBLAS — i.e. the existing tiled kernel leaves most of the hardware on the floor.

Track E builds **one new hand-written kernel** that replaces cuBLAS as the default for all prefill, with cuBLAS retained only for the rare bail-out shapes (FP8/INT8 KV, hd=48, etc.). It eliminates the 1 GiB S-matrix workspace and unlocks the projected 3-5× prefill speedup that the bench identified.

## Decisions log

| Q | Decision |
|---|---|
| 1. Kernel structure | **1 producer + 7 consumer warps** (warp-specialized, mbarrier coordination). |
| 2. KV dtypes for v1 | **FP16 + NVFP4** (runtime dispatch on `kv_dtype`). |
| 3. Head dims supported | **64, 96, 128, 256, 512** (hd=512 via HD-chunking). |
| 4. Dispatch strategy | **Default for all prefill**. cuBLAS only when Track E bails. |

## §1 Architecture overview

**Kernel name:** `attention_tiled_streaming_sm120` in `src/compute/attention_tiled_streaming.cu`.

**Goal:** Replace the cuBLAS materialised-S-matrix prefill path with a single hand-written FA2-style streaming kernel that:
1. Eliminates the 1 GiB S-matrix workspace.
2. Beats cuBLAS on production seq (projected 3-5× per Säule-3 ceiling bench).
3. Becomes the default for *all* prefill attention, with cuBLAS as fallback only for shapes the kernel does not support.

**Kernel structure: 1 producer + 7 consumer warps (256 threads total).**

```
┌─────────────────┐                  ┌──────────────────────────────────┐
│ warp 0          │                  │ warps 1..7 (consumers)           │
│ (producer)      │   mbarrier       │                                  │
│                 │   ───────►       │  - ldmatrix Q, K, V              │
│  cp.async K     │   K ready        │  - mma.sync m16n8k16 (QKᵀ + PV) │
│  cp.async V     │                  │  - online softmax (FP32 m, l)    │
│  ↓              │   V ready        │  - O accumulator in REGISTERS    │
│  double-buf →   │   ───────►       │  - per-warp owns row-tile        │
└─────────────────┘                  └──────────────────────────────────┘
```

**Supported attention features (parity with cuBLAS + FMHA combined):**

- causal masking — required for every prefill
- sliding window — Gemma-4 SWA
- soft-cap (logit cap) — Gemma-3 / Gemma-4
- GQA arbitrary ratios where `n_heads % n_kv_heads == 0`
- chunked prefill via `q_offset` parameter (same contract as cuBLAS path)
- CUDA Graphs capture — no `cudaMalloc` in hot path, no host-pointer indirection

**Unsupported in v1 (cuBLAS fallback fires):**

- `n_heads % n_kv_heads != 0`
- `head_dim < 32` or `head_dim ∉ {64, 96, 128, 256, 512}`
- KV dtype ∉ {F16, NVFP4}

**File layout:**

- `src/compute/attention_tiled_streaming.h` — public host launcher
- `src/compute/attention_tiled_streaming.cu` — kernel + launcher (~600-800 LOC)
- `src/exec/executor_attention.cu` — dispatch gate updated to prefer Track E
- `tests/test_attention_tiled_streaming.cu` — correctness vs cuBLAS reference
- `tests/perf_baseline.json` — adds Track E entries, retains cuBLAS baselines for bail-out paths

## §2 Tile geometry + SMEM layout

**Constraint:** sm_120 dynamic smem cap = 100 KiB per CTA. 8 warps (1 producer + 7 consumers).

**KV buffering:** K is double-buffered (producer loads K[i+1] while consumers compute QKᵀ on K[i]). V is single-buffered (loaded after QKᵀ, consumed by PV, reused for K[i+2] of next iter).

| hd | Br | Bkv | K dbuf | V buf | Q smem | Total smem | Row-tiles | Warp util at QKᵀ |
|---|---:|---:|---:|---:|---:|---:|---:|---|
|  64 | 128 | 64 | 16 KB |  8 KB | 16 KB | 40 KB | 8 | **7+1**: 6 warps × 1 tile, 1 warp × 2 tiles |
|  96 |  96 | 64 | 12 KB |  6 KB | 18 KB | 36 KB | 6 | **6+1**: 6 warps × 1 tile, 1 warp helper |
| 128 |  64 | 64 | 32 KB | 16 KB | 16 KB | 64 KB | 4 | **4+3**: 4 mma + 3 softmax/O-rescale helpers |
| 256 |  32 | 32 |  8 KB |  4 KB | 16 KB | 28 KB | 2 | **2+5**: 2 mma + 5 helpers (Bkv shrunk to fit) |
| 512 |  32 | 32 | HD-chunked | HD-chunked | 32 KB | 64 KB | 2 | **2+5**, HD_chunk=128 (4 sub-iters per KV tile) |

(hd=256 has room to grow Bkv back to 64 → 56 KB if perf-bench shows benefit; deferred to tuning phase.)

**HD-chunking for hd=512 (FA3-style):**

1. Q loaded fully (32 × 512 × 2 B = 32 KB) — stays in smem through all KV iters
2. For each KV tile (Bkv=32 KV positions):
   - Loop chunk `c` in 0..3:
     - Producer cp.async loads K[c] (32 × 128 × 2 B = 8 KB)
     - Consumers accumulate QKᵀ partial into FP32 S_frag (regs)
   - Softmax on complete S
   - Loop chunk `c` in 0..3:
     - Producer cp.async loads V[c] (8 KB)
     - Consumers accumulate PV partial into O_chunk[c] (regs)
3. Store full O row-tile (32 × 512 = all 4 chunks) at end

**O accumulator: REGISTERS only.** Lesson from Säule 3 — SMEM RMW was the softmax bottleneck.

- Per consumer warp owns one row-tile (16 rows × HD cols, FP32).
- HD=128: 16 × 128 / 32 lanes = 64 FP32 regs/lane.
- HD=256: 128 FP32 regs/lane (tight, may spill some).
- HD=512: per HD-chunk 16 × 128 = 64 FP32 regs/lane × hold one chunk active at a time.

**Online softmax state: REGISTERS.** `row_m[16]`, `row_l[16]` FP32 per warp → 1 FP32 reg/lane. Cheap.

**Q tile L2-persist hint.** Q is read once at iter 0, reused for all KV tiles. Mark Q's gmem region as L2-persisting via `cudaStreamSetAttribute(accessPolicyWindow)` so Q stays cached across KV-tile iterations. Säule-3 already saw L2 hit empirically — this just makes it explicit.

## §3 Warp roles + pipeline

### Warp 0 — Producer

Sole responsibility: feed K and V tiles to the consumers.

```
loop iter = 0..n_kv_tiles-1:
    # K-load for iter+1 (or iter 0 on warm-up)
    cp.async K[next_iter, :, :] into K_smem[k_slot]
    cp.async.commit_group
    mbarrier.arrive K_ready[k_slot]
    k_slot ^= 1

    # Wait for consumers to finish PV on iter-1's V before reusing V buffer
    mbarrier.wait V_consumed[1 - v_slot]

    # V-load for current iter (after consumers signal QKᵀ done)
    mbarrier.wait QKt_done[iter]
    cp.async V[iter, :, :] into V_smem
    cp.async.commit_group
    mbarrier.arrive V_ready[iter]
```

cp.async chunk size: 16 bytes (8 halves) — matches FMHA pattern. 32 threads of warp 0 issue cp.async in a tight loop.

### Warps 1..7 — Consumers

Role splits by hd:

| hd | Mma-warps | Helper-warps | Helper duty |
|---|---|---|---|
| 64 | 7 (all) | 0 | full row-parallel |
| 96 | 6 | 1 | softmax reduction + O-rescale slave |
| 128 | 4 | 3 | softmax + O-rescale slave (warps 5-7 mirror 1-3's row-tile during rescale) |
| 256 | 2 | 5 | softmax + O-rescale + S-fragment shuffle |
| 512 | 2 | 5 | same as 256, plus drive HD-chunk loop counter |

Helpers are pinned to specific row-tiles via a static mapping table in shared memory (initialised once at kernel start by warp 0).

### Pipeline (per KV tile, steady state)

```
                Producer                       Consumers
─────────────────────────────────────────────────────────────────────
T0:  cp.async K[i+1] ──┐
                       │
T1:  ─── wait QKt[i-1] │   ldmatrix K[i] (K_smem[curr])
                       │   ldmatrix Q (from Q_smem, cached)
                       │   mma.sync.m16n8k16 × N (QKᵀ) → S_frag in regs
                       │   warp-reduce row_max  (redux.sync.max.f32)
                       │   subtract + __expf → P_frag in regs
                       │   warp-reduce row_sum (redux.sync.add.f32)
                       │   rescale O_frag *= exp(prev_m - new_m)
                       │   mbarrier.arrive QKt_done[i]
T2:  cp.async V[i] ────┘
                            ldmatrix V[i] (V_smem)
                            mma.sync.m16n8k16 × M (PV) → O_frag += P · V[i]
                            mbarrier.arrive V_consumed[i]
T3:  cp.async K[i+2]
       (depends on consumers ldmatrix'ing K[i] first)
```

Steady-state overlap: producer's K[i+1] load runs concurrently with consumers' QKᵀ on K[i]. Producer's V[i] load overlaps with consumers' softmax. Total iter dominated by `max(load_K + load_V, QKᵀ + softmax + PV)`. Per Säule-3 ceiling: load ≈ 850 ns L2 / 3050 ns DRAM; compute ≈ 1500-2000 ns → expect compute-bound on L2-hit, load-bound on DRAM-miss.

### mbarrier inventory

5 mbarriers per CTA in dedicated smem region (40 bytes total):

| mbar | Purpose | Arrive | Wait |
|---|---|---|---|
| `Q_ready` | Q loaded once at start | 1 (producer) | 7 (consumers) |
| `K_ready[0..1]` | Phase-flip K slots | 1 | 7 |
| `V_ready` | Phase-flip single V | 1 | 7 |
| `QKt_done` | Consumers signal QKᵀ + softmax complete | 7 | 1 (producer) |
| `V_consumed` | Consumers signal PV complete (V slot reusable) | 7 | 1 |

Use `mbarrier.try_wait.parity` with explicit phase counters, matching the pattern in `tests/bench/fmha_v_load_bench.cu:76-87`.

### Online softmax (in-registers, all FP32)

```
S_row[0..Bkv-1]  ← QKᵀ output (FP32, regs split across 4 lanes per row)
r_max  ← redux.sync.max.f32(S_row)
new_m  ← max(prev_m, r_max)
scale  ← __expf(prev_m - new_m)
P_row  ← __expf(S_row - new_m)
r_sum  ← redux.sync.add.f32(P_row)
new_l  ← scale * prev_l + r_sum
O_row *= scale                                  # FP32 elementwise on O_frag regs
prev_m, prev_l ← new_m, new_l
```

`redux.sync.max.f32` and `redux.sync.add.f32` are 1-instruction warp reductions on sm_90+ — lower latency than `__shfl_xor_sync` trees. No smem touch for softmax state.

### Epilogue (after last iter)

```
For each row owned by consumer warp w:
    O_row *= 1.0f / prev_l                       # final normalise
stmatrix.sync.aligned O → O_smem                # FP16 quantise O_frag
cp.async-style 16-byte stores O_smem → gmem
```

`stmatrix.sync.aligned.m8n8.x4.shared.b16` writes 16×8 FP16 tiles in one warp instruction (matches the m16n8k16 D-frag layout when downcast to FP16).

## §4 NVFP4-KV path + dispatch integration + testing

### NVFP4-KV inner-loop

When `kv_dtype = NVFP4` the inner mma swaps to:

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

- Twice the K-loop coverage per mma (k=64 vs k=16) — 4× fewer mma instructions.
- Throughput ceiling: 268 TOPS per `sm120_mma_variants_2026_04_25` (3.3× FP16).

Both operands must be e2m1 (FP4):

- **K, V** read directly from the NVFP4 paged cache. No dequant.
- **Q** is FP16 (qkv-proj output) → quantised per row-tile via `cvt.rn.satfinite.e2m1x2.f32` PTX inline, FP4 + per-block UE8M0 scale stored in smem next to Q. One-time cost at kernel start.
- **P** (post-softmax probabilities) → quantised FP16→FP4 per consumer warp before PV mma. UE8M0 scale derived from `row_l`.

UE8M0 scale propagation: each mma carries SF_A and SF_B (16 × b8 each). Stored in dedicated smem scale-buffer alongside Q/K/V tiles. Matches existing pattern in `src/quant/nvfp4_quant.cu`.

Numerical precision: post-mma accumulator stays FP32 (same as FP16 path). Online softmax stays FP32. Only the operands round to FP4 → empirically parity with decode-path NVFP4 (`lever2_nvfp4_kv_implemented_2026_05_07`).

Kernel template branches on dtype at compile time: `template <int Bq, int HD, KvDtype kv_dt>` — two specialisations compiled. No runtime branch inside hot loop.

### Dispatch integration (`src/exec/executor_attention.cu`)

Replace the post-Phase-2 2-branch gate with a 3-branch gate that prefers Track E:

```cpp
if (attention_tiled_streaming_prefill(qv, kk, vv, ao, nh, nkv, hd, scale,
                                       causal, sliding_window, softcap,
                                       q_offset, stream)) {
    // Track E handled it.
} else if (s_matrix_fits && !non_gemma4_sliding) {
    attention_cublas_prefill(qv, kk, vv, ao, attn_scores_, ...);
} else {
    attention_prefill_dispatch(...);  // FMHA chain — now rarely fires
}
```

`attention_tiled_streaming_prefill` returns false when:

- `n_heads % n_kv_heads != 0`
- `head_dim ∉ {64, 96, 128, 256, 512}`
- KV dtype ∉ {F16, NVFP4}
- KV dtype is NVFP4 but `K.scales` is null (sidecar missing)

cuBLAS S-matrix workspace stays allocated but is only used for the rare bail-out cases (FP8-KV, INT8-KV, hd=48, etc.). A future PR may shrink or remove the workspace once those paths migrate; not in scope for v1.

Chunked prefill: `q_offset` parameter flows in unchanged. Track E applies the causal mask using `abs_q_pos = q_offset + i` for row `i` against `j ≤ abs_q_pos`. Matches cuBLAS semantics exactly.

### Testing strategy

**Correctness:** `tests/test_attention_tiled_streaming.cu` replicates the 42-config sweep from Säule 1 and compares output vs `attention_cublas_prefill` reference.

- Tolerance: max abs error < 5e-3, max rel error < 1e-2 on FP16 output (matches existing FMHA test gate).
- NVFP4-KV: compare vs existing `paged_attention_decode_nvfp4` driven over the prefill range.
- Edge cases: causal-only, causal+softcap, causal+sliding, full attention, `q_offset>0` (chunked-prefill simulation), GQA ratios {1, 2, 4, 8, 16}.

**Regression coverage:** existing `test_attention_chunked.cu`, `test_attention_fmha_sm120.cu`, `test_attention_paged_nvfp4_tc.cu`, the e2e `test-e2e` suite — all must pass unchanged. Dispatcher prefers Track E so these tests now indirectly exercise it.

**Perf gate updates (`tests/perf_baseline.json`):**

| Metric | Threshold | Note |
|---|---|---|
| Qwen3-8B Q8_0 tg256 | ≥ 255 tok/s | unchanged (decode untouched) |
| Qwen3-8B Q8_0 pp512 | ≥ 22000 tok/s | **new gate, +25% vs 17636** |
| Gemma-4-26B Q4_K_M tg256 | ≥ 183 tok/s | unchanged |
| Qwen3.6-35B Q4_K_M tg256 | ≥ 143 tok/s | unchanged |
| Qwen3-8B NVFP4 pp512 | ≥ 25000 tok/s | **new gate, +33% vs 18802 (NVFP4-KV inner-loop win)** |
| Track E unit ms vs cuBLAS | Track E ≤ 0.50 × cuBLAS @ seq=2048 | **hard gate — ≥2× speedup or revert** |

Refresh baselines via `scripts/gen_perf_baseline.sh` after merge. 3% decode / 5% prefill regression thresholds (existing policy).

### CUDA Graphs

Track E captures cleanly. No host allocations in the hot path, no host-pointer indirection. The kernel uses `cp.async` and `mbarrier` — both capture-safe (matches the existing FMHA kernel which already captures). NVFP4-KV path additionally uses cvt PTX and UE8M0 scale smem regions — none touch global memory outside the input K/V/Q/O pointers — also capture-safe.

### Out-of-scope (v2 follow-ups)

- FP8 KV inner-loop (`mma.sync.m16n8k32 kind::f8f6f4`) — rare in production, deferred.
- Warp-spec ratio tuning (1+7 → 2+6 if compute-bound shows up in profiling).
- Dedicated paged-KV-cache reader inside kernel (skip the `paged_kv_gather` materialisation step). Currently we accept already-gathered contiguous K/V.
- hd ∈ {48, 160, ...} other niche head_dims.

### Dev-day estimate

| Phase | Days |
|---|---|
| FP16 kernel skeleton (1+7 warp-spec, no NVFP4, hd=128 only) | 4 |
| HD generalisation (64, 96, 256) | 2 |
| HD=512 chunking | 3 |
| Causal + sliding + softcap | 2 |
| GQA + chunked-prefill (q_offset) | 2 |
| NVFP4-KV path | 5 |
| Test suite + correctness fuzzing | 3 |
| Perf tuning + baseline refresh | 4 |
| Integration + CI green | 2 |
| **Total** | **27 days** |

## Risks + mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| 1+7 warp-spec underutilises warps at hd=128 (4 mma-active) | medium | Helpers do O-rescale + softmax-reduction overlap. Profile after v1; consider 4+4 hybrid if compute-bound. |
| NVFP4-Q quantise overhead negates 3.3× mma win | medium | Q-quant runs once per kernel invocation, amortised across all KV tiles. Measure isolation in micro-bench before commit. |
| hd=512 HD-chunking is slower than cuBLAS at small seq | low | Fall back to cuBLAS at seq < 256 for hd=512 via dispatch gate check. |
| Register spill at hd=256 O-accumulator | medium | If `ptxas -v` shows spills, drop to Br=16 at hd=256 (4 row-tiles → still 2+5 warp split). |
| FP4-P quantisation hurts numerical stability | high | Verified by `lever2_nvfp4_kv_implemented_2026_05_07` for decode; add NIAH 16K eval gate before flipping NVFP4-KV default to Track E. |
| 27-day estimate slips | medium | Phase 1-5 (FP16 only) is 13 days and gives 3× speedup standalone. Ship FP16 alone if NVFP4 work runs long. |

## Reproduce the gating bench

```bash
make build
docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 imp:test \
    imp-tests --gtest_filter='Matrix/AttnPrefillBench.*:TiledAttentionCeilingBench.*'
python3 scripts/analyze_attention_workspace_savings.py
```

## Next step

After user approval of this design, invoke the **writing-plans** skill to produce an implementation plan with task-by-task breakdown (TDD-style steps, exact file paths, expected test output per step). Plan is saved to `docs/superpowers/plans/2026-05-21-track-e-tiled-streaming-softmax.md`.
