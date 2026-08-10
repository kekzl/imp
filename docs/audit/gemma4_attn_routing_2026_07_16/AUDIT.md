# FA2 Attention Coverage Audit — sm_120a

Append-only. Scout phase of the "close FA2 coverage / kill legacy causal_softmax +
cuBLAS" dispatch. Read-only investigation, zero edits to source, zero kernel launches.

---

## Entry 1 — Scout: coverage matrix + gap list (2026-07-16)

### TL;DR — the dispatch premise is stale

The dispatch is built on the **May-2026 audit conclusion** that FA2-uncovered
attention shapes fall back to a materialized `causal_softmax + cuBLAS` path carrying
**~18 % prefill overhead across many shapes**, and asks to extend FA2 to close it.

**That work has already shipped.** The repo's own prefill-gap audit states it
explicitly (`docs/archive/prefill_gap_2026_06_07.md:142-144`):

> "Legacy materialized attention is **0.0 % on every hd=128 model** (the 2026-05-31
> "~18 % materialized attention" figure is dead — fixed by #525/#478). Only hd≠128
> (gemma-3/4) still runs it, at 3.6–6.9 % of window."

And since that audit, hd=256 (gemma-3) also moved to FA2 (#930/#932, `attention.fa2_hd256`
default-on). As of `main` @ today, **the legacy cuBLAS prefill path is not the broad
fallback the dispatch assumes** — it serves a deliberately-retained narrow tail.

There is no ~18 % overhead left to recover. The acceptance criterion "recover ≥15 %
prefill on previously-uncovered shapes" **has no target shapes to apply to** — every
shape that carried that overhead was migrated to FA2 across #478/#493/#525/#932.

### Dispatch pin (file:line)

| Component | Location |
|---|---|
| Prefill gate (cuBLAS-vs-FA2-vs-FMHA decision) | `src/exec/executor_attention_prefill.cu:347-402` (non-chunked), `:278-318` (chunked q_offset>0) |
| FMHA chain (fused ladder) | `src/compute/attention_dispatch.cu:38-128`; host model `src/compute/attention_dispatch_decision.h` |
| FA2 f16-QK entry | `try_fa2_fp16qk_prefill(...)` → `src/compute/attention_fmha_sm120.cu` |
| FA2 hd=256 instance | `attention.fa2_hd256` (default on, #932) |
| Legacy materialized path | `attention_cublas_prefill(...)` → `src/compute/attention_cublas.cu:387` |
| Final fused tier | `flash_attention_blackwell(...)` → `src/compute/attention_blackwell.cu:383` (declines hd ∉ {64,96,128,256}) |
| Chain-exhausted guard | `attention_dispatch.cu:127` throws `std::runtime_error` (#654) — no silent garbage |

### Coverage matrix (BEFORE — i.e. current shipped state)

Every mainstream target shape already routes to a **fused** kernel (FA2 or WMMA-FMHA),
not to legacy cuBLAS. Verified against the executor gate + `attention-dispatch.md`.

| head_dim | causal | GQA | KV dtype (F16/FP8/NVFP4/MXFP4/INT4) | SWA | softcap | Path | Route condition |
|---|---|---|---|---|---|---|---|
| 128 | yes | any | any | any | any | **FA2 f16-QK** | `try_fa2_fp16qk_prefill` succeeds at every length (#493/#525) |
| 256 (uniform, incl. GDN/Mamba2 hybrids) | yes | any | any | any | any | **FA2 f16-QK** (Bq=64/TWOSLOT) | `fa2_hd256` default-on (#930/#932) |
| 64 / 96 | yes | any | any | any | any | **WMMA-FMHA** → Blackwell | FA2 declines; `fmha_sm120_prefill` / `flash_attention_blackwell` accept |
| 128/256, long seq, FA2 declined edge | yes | any | any | any | any | **WMMA-FMHA chain** | tiled O(n), no S-matrix |

### Gap list — what still executes legacy `attention_cublas_prefill`

Sorted by prefill-time contribution (measured shares from `prefill_gap_2026_06_07.md`
and `attention-dispatch.md`, not guessed):

| # | Cell | Prefill share | Why it's on cuBLAS | Reachable via dispatch chain? |
|---|---|---|---|---|
| 1 | **Gemma-4 heterogeneous per-layer head_dim (256 / 512)** | small, gemma-only (was 3.6–6.9 % of window in 06-07; hd=256 layers since moved to FA2, so effectively only the hd=512 layers) | `force_cublas_attn = per_layer_shapes && !attn_shapes_uniform()` (`executor_attention_prefill.cu:347-348`). **hd=512 has NO fused kernel** — see infeasible cell below | No (force_cublas bypasses chain) |
| 2 | **gpt-oss learned sinks, hd∉{128,256}, seq < `fmha_prefill_threshold`** | below-threshold short prompts only | cuBLAS FP16 is the **accuracy reference**; FA2 itself declines sinks. Above threshold, sinks ride WMMA-FMHA since #992 | Above threshold: yes (WMMA). Below: cuBLAS by design |
| 3 | **Chunked continuation `q_offset > 0`, seq < threshold, FA2-declined** | tiny; #847 already routes small growing chunks to tiled FMHA to kill cuBLAS algo churn | FA2 conservatively declines `q_offset > 0` (blanket gate post-#548) | Above threshold: yes (tiled FMHA) |
| 4 | **Vision encoder attention** (`encoder_forward.cu:209`) | n/a — **non-causal bidirectional** vision (CLIP/SigLIP-style), `causal=false` | Outside the causal-LLM prefill target set entirely | No (separate code path) |
| 5 | **`attention.force_cublas_decode`** (`executor_attention_decode.cu:65-125`) | 0 — **debug flag, default off** | Explicit "isolate paged-attention bugs" reference toggle | No (debug-only) |
| 6 | **Parity test references** (`test_attention_chunked.cu`, `test_attention_crosspath.cu`, `test_gpt_oss_sinks_ref.cu`) | 0 — tests | cuBLAS FP16 is the trusted numerical reference the fused kernels are validated against | No (tests) |

### Infeasible cell (hard sm_120 constraint) — written per anti-cheat gate

**head_dim = 512 (Gemma-4 wide layers) has no register-resident / SMEM-tiled fused
kernel on sm_120a, and cannot get one.** A flash-style kernel at hd=512 needs
~176 KB of shared memory at Br=64 (Q/K/V/S tiles), versus the **99 KB opt-in SMEM
limit** on GB202 (`cudaDeviceProp::sharedMemPerBlockOptin`; `attention-dispatch.md:52`).
sm_120 has no TMEM / `tcgen05` / cluster-MMA to offload that state. Splitting hd=512
across two CTAs with a cross-CTA reduction is a different kernel class (2-CTA cooperative),
not "extend FA2". Therefore Gemma-4's hd=512 layers are a **legitimate, documented
cuBLAS-exclusive cell** — not a silent laziness fallback. This is exactly the
"infeasible cell + reason" the dispatch's anti-cheat gate asks to be recorded here
rather than papered over.

### Why "kill the legacy path entirely" is the wrong move (correctness)

`attention_cublas_prefill` (FP16 materialized QK^T → softmax → PV) is deliberately kept
as the **short-prompt accuracy reference**. The fused e4m3 alternative was demoted to
opt-in (#511/#656) because raw e4m3 Q/K conversion compounds per-layer score error into
prompt-blind / degenerate output: teacher-forced PPL **gemma-3-12b 16.6→549**,
**Qwen3-8B 40.5→4506** when the fp8-QK kernel served prefill
(`attention_dispatch_decision.h:76-79`, `attention-dispatch.md:50`). Deleting cuBLAS
would remove (a) the only kernel that serves hd=512, (b) the sink-capable below-threshold
reference, (c) the non-causal vision encoder path, and (d) the numerical reference the
parity tests assert against. That fails the dispatch's own "no lowering existing
coverage / no silent legacy fallback" gates in the opposite direction.

### Proposed target-shape set

Given the above, the honest target-shape set for a *builder* phase is **empty of the
originally-imagined work**. Two candidate residual items exist, both small and both
optional:

- **R1 (small, real):** extend FA2 f16-QK to accept `q_offset > 0` chunked continuations
  (cell #3), removing the conservative post-#548 blanket decline. Upside is sub-1 %
  (audit shows legacy attn 0.0–1.9 % of window); #847 already neutralized the churn cost.
- **R2 (infeasible):** hd=512 fused attention — blocked by the 99 KB SMEM ceiling
  (see infeasible cell). Not pursuable as "extend FA2".

**Recommendation:** do NOT run builder/validator/profiler against the original premise —
there is no ~18 % overhead and no broad FA2 gap to close; the work shipped in
#478/#493/#525/#932/#992. If a residual is desired, scope R1 as a standalone small
optimization with its own before/after (expect sub-1 %), and record R2 as permanently
infeasible on sm_120. Freezing this as the target-shape set per the orchestrator gate.

### Evidence trail

- `docs/attention-dispatch.md` — canonical routing table, lines 7-14 (0.0 % on hd=128),
  39-42 (gate), 52 (hd=256/512 SMEM), 50 (e4m3 PPL catastrophe).
- `docs/archive/prefill_gap_2026_06_07.md:142-144` — "~18 % figure is dead, fixed by #525/#478".
- `git log src/compute/attention_dispatch.cu src/exec/executor_attention_prefill.cu`:
  #525 "FP16-QK FA2 for short prefill — replace materialized cuBLAS path",
  #493 default-on hd=128, #932 fa2_hd256 default-on, #992 sink-capable WMMA FMHA,
  #1025 dead-path removal.
- Callers of `attention_cublas_prefill`: prefill executor (below-threshold reference),
  vision encoder (non-causal), decode debug flag, parity tests, engine prewarm. No
  production causal-LLM prefill shape routes to it that a fused kernel could take instead.

---

## Entry 2 — Builder/validator: hd=512 is NOT infeasible — correction + resolution (2026-07-16)

Entry 1 recorded hd=512 as a permanently-infeasible cell. **That was wrong**, and the
correction is the substance of this dispatch:

- The infeasibility argument (≈176 KB SMEM > 99 KB opt-in) applies only to the
  **register-resident flash** kernels (FA2 / `flash_attention_blackwell`), which hold the
  O accumulator in registers. The tiled **WMMA FMHA** (`fmha_sm120_prefill`) holds Q / KV /
  O_acc in **shared memory** and streams KV tiles — so its footprint is set by the tile,
  not the full head_dim. At `Bq=16, Bkv=16, HD=512` the block needs **~65 KB** (`compute_smem_sm120`),
  comfortably under the 99 KB opt-in. The kernel body is already parametric in `Bkv`/HD;
  the only missing piece was the instantiation and a Bq=16 launcher tier.

Resolution shipped:
- **Kernel** (`attention_fmha_sm120.cu`): derive `Bkv = (HD>=512)?16:64` at compile time,
  add a `Bq=16` launcher tier and a `case 512` instantiation. `__launch_bounds__(256,2)`
  runs at occupancy 1 here (same as hd=256 today — 2 blocks never fit at these SMEM sizes),
  so no launch-config failure. Applies to **Gemma-4 global layers AND Qwen3.5-27B**
  (both hd=512; see `attention_cublas.cu:423`).
- **Routing** (`executor_attention_prefill.cu`): the coarse MODEL-level `force_cublas_attn`
  gate is replaced by per-layer fused-servability in both the chunked and non-chunked
  branches. Heterogeneous (Gemma-4) models now route FA2 for their hd=256 SWA layers and
  the WMMA FMHA dispatch for their hd=512 global layers; the materialized cuBLAS branch is
  gated `!hetero_shapes`, making it **unreachable for the target set**. Uniform/mainstream
  and gpt-oss-sinks routing is byte-for-byte unchanged (cuBLAS stays their below-threshold
  reference).

### Parity finding (validator)

`test_attention_fmha_hd512.cu` (6 configs: causal/GQA/MHA, softcap, rect+q_offset,
tile-edge, sliding-window) vs an fp64 eager reference AND the cuBLAS FP32-S legacy path:
FMHA hd=512 tracks fp64 at **max 1.2–1.87e-2 / mean ~3e-4** (clean f16 class for a 512-long
reduction) and is **at-or-below the cuBLAS error on every config**. Note: the materialized
cuBLAS path is only accurate at hd=512 on its **FP32-S** branch (needs a ≥3× score buffer,
#677); on FP16-S it truncates the large hd=512 scores (max_rel 5.8e-2…1.07e-1). The fused
kernel has no such precision cliff — a latent robustness win, not just a routing change.

### Final target-shape set (frozen)

| Shape | Path after this dispatch | cuBLAS reachable? |
|---|---|---|
| Gemma-4 hd=256 SWA layers (causal+SWA+softcap, GQA) | FA2 f16-QK (per-layer) | No |
| Gemma-4 hd=512 global layers (causal, GQA, softcap) | WMMA FMHA hd=512 (new) | No |
| Qwen3.5-27B hd=512 (if uniform) | WMMA FMHA hd=512 (new) | No |
| everything already covered (hd 128/256 uniform, gpt-oss sinks) | unchanged | unchanged |

No infeasible cells remain for the causal-LLM prefill target set. cuBLAS retains only its
legitimate non-target roles: the non-causal **vision encoder**, the `force_cublas_decode`
**debug flag**, engine **prewarm**, and **parity-test** references.

---

## Entry 3 — Profiler: measurement OVERRIDES "make hd=512 unreachable" (2026-07-16)

Entry 2 planned to route hd=512 fully off cuBLAS ("unreachable for the target set"). **The
perf measurement kills that plan** and the design was revised to a hybrid. Full data in
`PERF_LOG.md` entry 1; summary:

- **The fused WMMA FMHA hd=512 is 2.8–4.6× SLOWER than the materialized cuBLAS path**
  (pp512 0.52×, pp2048 0.22× — isolated kernel A/B, warmed clocks). There was never a perf
  gap at hd=512; cuBLAS was already the faster kernel.
- **Root cause is the same hardware wall as the "infeasible" one.** hd=512 needs 512 f32
  accumulators per query row. sm_120 has **no TMEM/`tcgen05`** to hold them off-register/
  off-SMEM, so O_acc lives in SMEM → the 99 KB opt-in caps the query tile at **Bq=16** →
  the sequence splits into ~128 query-tiles at Sq=2048, each re-reading K/V, and the QK
  WMMA runs on ≤2 of 8 warps. cuBLAS sidesteps this by materializing the S-matrix and
  running full-size tensor-core GEMMs. Tuning Bkv 16→32 recovered ~40% (still 2.8× slower)
  AND broke correctness at rectangular/`q_offset` shapes — reverted.
- **CUTLASS does not change this** (asked during review). imp uses CUTLASS only for
  GEMM/grouped-GEMM (no FP4 cuBLASLt on sm_120); **all attention in imp is hand-written
  `mma.sync`, zero CUTLASS FMHA** (grep-confirmed). CUTLASS's fast fused-attention
  collectives target sm_90 (`wgmma`+TMA-WS) and sm_100 (`tcgen05`+TMEM); on sm_120 CUTLASS
  falls back to the same register/SMEM `mma.sync` tiling and hits the identical Bq wall. It
  is a hardware-capability gap, not a library choice.

### Revised design (shipped) — hybrid, perf-positive, no regression

| Gemma-4 layer | Path | vs before |
|---|---|---|
| hd=256 SWA (≈24 of 30, the 5:1 majority) | **FA2 f16-QK** (per-layer) | was cuBLAS → **now FA2 (faster, the real win)** |
| hd=512 global (≈6 of 30) | **cuBLAS while S-matrix fits** | unchanged (cuBLAS is the faster kernel here) |
| hd=512 global, S-matrix overflow (long ctx) | **WMMA FMHA hd=512** (O(n) fallback) | was heavily chunked / capacity-limited → **now has an O(n) path** |

Net: the coarse model-level `force_cublas` gate (which sent EVERY Gemma-4 layer to cuBLAS)
is replaced by per-layer routing. The **win is moving the majority SWA layers to FA2**; the
new hd=512 kernel earns its keep only as the long-context O(n) capacity fallback, not as a
speed play. The literal dispatch criterion "0 target shapes route to cuBLAS" is **not met
for hd=512 short/medium, by design and by measurement** — forcing it would regress Gemma-4
prefill. This is recorded here per the anti-cheat gate ("infeasible/counter-productive cell
→ write the reason, don't silently force it").

Coverage test updated accordingly: `Gemma4ModelTest.PrefillFusesSwaLayers_Hd512StaysCublas`
asserts a short Gemma-4 prefill uses cuBLAS for <15 layers (only the ~6 hd=512 globals), not
all 30 — proving the SWA majority moved to FA2 while hd=512 stays on the faster cuBLAS path.

---

## Entry 4 — S-overflow regime re-measured: sliced cuBLAS replaces the FMHA fallback; chunk clamp lifted (2026-07-16)

Correction to entry 3 first: "Bkv=32 … broke correctness … reverted" is stale — the break
was an over-strict test gate, not a parity failure, and **Bkv=32 shipped** (PERF_LOG entry 2).

Substance (PERF_LOG entry 4 has the numbers): the fused hd=512 kernel's one production
regime — the S-matrix-overflow long-context chunk — was measured, and **cuBLAS in
workspace-sized q-row slices is 3.4-3.9× faster** there (the fused kernel is
KV-bandwidth-bound through its Bq=16 tile: ~Sq/16 full K/V re-reads). Two things followed:

1. **Routing**: new `attention_cublas_prefill_sliced` (slices sized to the FP32-S 3× rule,
   floor 16 rows) serves hd=512 at S-overflow in both prefill branches. The fused hd=512
   kernel is now only the terminal fallback for a too-degraded workspace (<16-row slices).
   This also covers uniform hd=512 models (Qwen3.5-27B), whose overflow chunks previously
   took the slow whole-chunk FMHA.
2. **Chunk clamp**: `max_safe_prefill_chunk` no longer clamps heterogeneous fused-servable
   models (Gemma-4) to the hd=512 S-capacity quadratic. The clamp used to shrink EVERY
   layer's chunk at long context (~190 rows at 64k) — MoE dequant and per-chunk launch
   overhead were multiplied model-wide just to keep the hd=512 layers' whole-call cuBLAS
   footprint inside the workspace. With per-layer slicing, Gemma-4 keeps full-size chunks
   at any context.

The FA2 kernels proper (hd=128/256) were re-checked against the refuted-lever list and the
roofline pins: no open lever — the remaining attention-prefill headroom was in this routing,
not in kernels.
