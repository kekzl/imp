# Qwen3.5-27B MXFP4 — host-dequant + storage planner design memo

**Date**: 2026-05-17
**Status**: design only — no source changes
**Subject**: Make Qwen3.5-27B MXFP4 (12 GiB raw + 48 GiB FP16 fallback)
load on 32 GiB VRAM. Current behavior: PR #60's diagnostic refuses the
FP16-fallback alloc and surfaces a clear error. Workarounds: 9B Q8_0,
35B-A3B Q4_K_M.
**Branch**: `docs/moe-host-offload-graphs-memo` @ `b217769` (HEAD)

## Table of contents

1. [The IMA root cause](#1-the-ima-root-cause)
2. [The "host-dequant + storage planner" plan](#2-the-host-dequant--storage-planner-plan)
   1. [Storage planner](#21-storage-planner)
   2. [Host-dequant on miss](#22-host-dequant-on-miss)
3. [Alternative — smarter FP16 cache policy](#3-alternative--smarter-fp16-cache-policy)
4. [Implementation phases](#4-implementation-phases)
5. [Risks](#5-risks)
6. [Decision recommendation](#6-decision-recommendation)

---

## 1. The IMA root cause

### 1.1 What the original investigation actually found

`qwen35_27b_mxfp4_ima_2026_04_25.md` is point-in-time and partially
superseded — re-read carefully before citing.

The dynamic-debug session pinned the *visible* IMA on Qwen3.5-27B MXFP4
to a NULL `A_log` pointer in the GDN scan kernel, not to VRAM
oversubscription. The MXFP4 converter packs `A_log` under
`ssm_dt.weight` while the Q8_0 path uses the canonical `ssm_a`; the
loader's `else if (field == "ssm_dt") layer.ssm_dt_b = tensor;` ignored
the suffix and silently overwrote `ssm_a` with `ssm_dt_b`.

- **PR #61** fixed the `A_log` loader differentiation.
- **PR #60** (commit `f5738d3`, 2026-04-26) added the
  oversubscription pre-flight (`src/graph/executor_pre_dequant.cu:1532
  – 1574`) so the failure mode is now a clear `IMP_LOG_ERROR` instead
  of an IMA cascade.
- **Open**: with the FP16 fallback bypassed, alpha/beta MXFP4 GEMV
  at N=48 produces NaN logits (`tok=-1`). This is a separate kernel
  bug, not a load-time issue, and is *not* the subject of this memo.

So the "IMA at load" framing in the roadmap entry is slightly stale.
What's left after PR #60+#61 is:

- A model that **cannot load** on 32 GiB because the FP16 fallback
  needs ~48 GiB and the planner refuses to oversubscribe.
- A diagnostic that explains *why* the model is rejected — but
  doesn't make it run.

### 1.2 The flow today (verified against current HEAD)

`src/graph/executor_pre_dequant.cu:1499 – 1646`:

1. All MXFP4 weights are uploaded to VRAM by
   `src/model/weight_upload.cu` (~12 GiB on Qwen3.5-27B).
2. For each MXFP4 tensor the executor optionally also populates a
   per-tensor FP16 fallback in `wcache_.fp16` (used when the prefill
   path can't fire native MXFP4 GEMM/GEMV).
3. The FP16 fallback is forced ON for GDN models (`has_gdn` branch,
   line 1509): the MXFP4 prefill dispatch hits `cuBLAS-INTERNAL_ERROR`
   on GDN-shape weights (notably `ssm_out` K=6144 N=5120 and FFN
   K=17408 N=5120) that's still not root-caused.
4. FP16 expansion at 4× over MXFP4 = ~48 GiB on the 27B model.
5. 12 + 48 = 60 GiB > 32 GiB → pre-flight check refuses the alloc and
   sets `fp16_total = 0`; downstream weight pointers stay raw MXFP4 and
   the first attention prefill bails (or, with `=force`, eventually
   IMAs at the first decode forward).

The fundamental tension is that the FP16-cache policy was written
assuming the *whole* set of MXFP4 tensors fits comfortably in residual
VRAM after raw upload. On dense MXFP4 models ≤ 8B that holds. On
GDN+MoE 27B it doesn't, and the executor has no story other than
refuse-to-run.

### 1.3 Where the FP16 cache is *actually* read

Grep for `wcache_.fp16` in the executor turns up these consumers
(`src/graph/executor_pre_dequant.cu`):

- Line 558 — Phase 1 cache_weight (initial dequant of *non-MXFP4*
  qtypes; MXFP4 path is line 1531-1602).
- Line 691 — fused KV / gate+up pre-dequant fast-path.
- Line 883 — generic-dequant catch-all (`from_scratch` flag).
- Line 950 / 975 / 1052 — NVFP4 pre-dequant migration.
- Line 1190 — "free FP16 if NVFP4/FP8 alternative exists" reclaim.
- Line 1527 / 1587 / 1620 / 2101 — MXFP4 → FP16 replace-pointer pass
  (line 1626-1641 enumerates every weight slot replaced: `wq`/`wk`/
  `wv`/`wo`/`w_up`/`w_gate`/`w_down`/`ssm_in`/`ssm_out`/`gdn_gate`/
  `gdn_alpha`/`gdn_beta`/`out_proj_`).
- Line 2236-2245 — MoE expert *packed* gate/up/down lookup.

The MXFP4 path **rewrites the model's weight tensor pointers** (lines
1619-1645) — i.e. once the FP16 fallback is populated, the model's
`L.wq.data` etc. point at FP16 and `L.wq.qtype == F16`. Every
downstream consumer transparently sees FP16. That's why the fallback
exists: it's a sledgehammer that converts the entire MXFP4 model to
FP16 in-place at startup so the rest of the executor doesn't need to
know.

Removing this sledgehammer means **every** consumer in the list above
must either grow an MXFP4 path or get its weight via on-demand dequant.

---

## 2. The "host-dequant + storage planner" plan

### 2.1 Storage planner

Pre-walk the model architecture before the FP16 cache pass and compute
the full VRAM footprint:

```
footprint = raw_mxfp4_bytes
          + sum(fp16_cache_candidate_bytes)
          + reserved(activations, workspaces, cuBLASLt scratch)
          + reserved(KV cache @ user-or-default max_seq_len)
```

The third and fourth terms are already estimated in
`src/runtime/vram_budget.cpp` (`compute_vram_budget` returns a
strategy + per-segment byte budget). The planner extends this to a
**per-tensor decision** rather than a global on/off flag.

If `footprint > free_vram - headroom`, the planner trims the FP16-cache
candidate set by priority:

1. **Always cache**: small attention output projections, RMSNorm
   weights, embeddings (already implicitly cached as FP16 by virtue of
   being non-quant).
2. **Cache if budget allows**: `wq`/`wk`/`wv`/`wo`, dense FFN
   `w_gate`/`w_up`/`w_down`.
3. **Skip by default**: MoE per-expert weights (handled natively by
   `gemm_grouped_nvfp4_smallM` / dp4a paths without going through
   FP16).
4. **Skip by default**: LM head (`out_proj_`; uses a separate dequant
   path on the prefill side and tiles into a smaller workspace).
5. **Skip by default**: ssm_in / ssm_out / gdn_* — but only once the
   alpha/beta N=48 NaN bug (memo `qwen35_27b_mxfp4_ima_2026_04_25.md`
   §"What remains") is fixed, otherwise GDN forward bails on garbage
   logits.

Heuristic for the priority-2 group: cache a tensor's FP16 expansion
only if it's actually consumed by a small-M code path (M ≤ 16, where
the dequant-then-FP16-cuBLAS path beats native MXFP4 GEMV by enough to
justify the 4× VRAM cost). Above M ≈ 16, native MXFP4 GEMM should win
on raw weights — see `q4k_mmvq_crossover_2026_05_15.md` for the
analogous Q4_K crossover at M ≈ 16.

The planner's job is to make the **skip / cache** decision *before* any
allocation, so the executor never enters the oversubscribed branch.

### 2.2 Host-dequant on miss

When a runtime code path requests a tensor's FP16 representation and
the planner decided to skip caching it:

1. Trigger host-side `dequant_mxfp4_to_fp16` (CPU code; the existing
   GPU kernel in `src/quant/mxfp4_gemm.cu` is straightforward to port —
   it's a block-of-32 nibble-unpack + ue4m3-to-fp32-scale multiply).
2. Stage the FP16 bytes into pinned host memory (allocated once at
   engine init; reused across misses).
3. `cudaMemcpyAsync` H2D to a fixed device scratch buffer (per-stream,
   reused).
4. GEMM proceeds on the scratch buffer as if it were a cached weight.

Bookkeeping-equivalent to the K8 CPU offload memo's cold/hot KV model,
but applied to **weights** rather than KV pages. The cost model differs
in one important way: KV pages get evicted by LRU and re-read; a
weight is read **every forward pass**. If host-dequant misses are
hot-path (per-token) the engine dies on PCIe bandwidth — see §5.

---

## 3. Alternative — smarter FP16 cache policy

A much cheaper option avoids host-dequant entirely.

The current FP16 cache pass (line 1525-1530, current HEAD) caches
**every** MXFP4 tensor by default. Reality check:

- MoE expert weights are handled by
  `gemm_kernel_cutlass_nvfp4.cu` /
  `gemm_kernel_nvfp4_gemm.cu` / dp4a paths that never read from
  `wcache_.fp16` — they read raw `L.expert_*_packed.data`. The only
  exception is the `L.fp16_packed_*_cache` pointer set at lines
  2236-2245, which is read by a specific MoE prefill fallback path
  (`executor_forward_moe.cu`); but for MXFP4 MoE that fallback isn't
  the primary route on Qwen3.5-27B.
- LM head (`out_proj_`) prefill uses `gemm_kernel_generic_dequant.cu`
  with on-the-fly dequant and a small batched workspace.
- ssm_in / ssm_out / gdn_* on GDN models go through `gemm_dispatch`
  (which **does** check `wcache_.fp16` on the prefill side per memo
  comment at line 1499-1507).

Pruned candidate set on Qwen3.5-27B:

| Group | tensor count × layers | FP16 bytes | keep? |
|---|---|---|---|
| `wq`/`wk`/`wv`/`wo` (attn) | 4 × 64 | ~1.5 GiB | **yes** |
| `w_gate`/`w_up`/`w_down` (dense FFN, *if* any non-MoE layer) | varies | small | yes |
| MoE expert gate/up/down packed | 3 × 128 × N_experts | ~8-10 GiB | **skip** |
| `ssm_in`/`ssm_out`/`gdn_*` | 5 × 16 GDN layers | ~1-2 GiB | yes (until GDN alpha/beta NaN bug fixed) |
| `out_proj_` (LM head) | 1 | ~1 GiB | skip |

Expected post-pruning FP16-cache footprint: roughly **8-12 GiB** vs
the current 48 GiB.

12 GiB MXFP4 raw + ~10 GiB FP16 cache + ~2 GiB activations/workspaces
+ ~6 GiB KV @ 16K context = **~30 GiB**, which fits with thin margin.
Going to ~8K KV gives more headroom.

If profiling confirms the MoE/LM-head FP16-cache entries are never
read on Qwen3.5-27B's actual decode/prefill paths, this fixes the
load failure with **no host-dequant machinery at all**.

---

## 4. Implementation phases

### Path A — smart cache policy (recommended first, ~3-5 days)

- **Phase A1** — instrument & profile (1-2 days).
  Add a debug counter to every `wcache_.fp16.find(p)` site (lines
  691 / 883 / 950 / 975 / 1052 / 1587 / 1620 / 2101 / 2236-2245). Run
  the production-bench fleet:
  - Qwen3-4B Q8_0, Qwen3-8B Q8_0 — confirm baseline FP16-cache hit
    pattern.
  - Qwen3-Coder-30B-A3B NVFP4 (MoE).
  - Gemma-4-26B Q4_K_M / NVFP4.
  - Qwen3.5-9B GDN Q8_0 (smaller GDN model that *fits* — proxy for
    27B's GDN read pattern).

  For each tensor slot, record `(hit_count, on_prefill, on_decode)`.
  The MoE expert / LM-head hypothesis (§3) is validated iff hit_count
  is zero or near-zero across all 5 models for those slots.

- **Phase A2** — policy in pre-dequant (1 day).
  Add a tensor-class predicate at line 1525-1530's bulk-alloc loop:
  skip MXFP4 tensors whose owning slot is in the "skip" list from §3.
  Gate behind a `RuntimeConfig::attention.mxfp4_fp16_cache_policy`
  enum (`legacy` / `pruned`) defaulting to `legacy` until validated.

- **Phase A3** — verify (1 day).
  - Qwen3.5-27B MXFP4 loads cleanly on 32 GiB.
  - Bench decode tok/s and prefill pp512 vs the smaller Qwen3.5 GDN
    models (regression check — confirm the skipped slots really
    aren't on the hot path).
  - Run `make verify` to confirm no regression on the production
    fleet.
  - If Qwen3.5-27B decodes garbage, the alpha/beta N=48 NaN bug
    (separate memo) is exposed; that's a follow-up, not a blocker for
    A2's policy flip.

### Path B — full host-dequant + storage planner (multi-week, ~3-4 weeks)

- **Phase B1** — storage planner (1 week).
  New `src/runtime/storage_planner.{h,cpp}`. Inputs: model arch +
  qtypes + free VRAM + KV budget. Output: per-tensor decision
  `{cache_fp16, raw_only, defer_to_host}`. Unit-tested with the
  fleet models.

- **Phase B2** — CPU MXFP4→FP16 (~2 days).
  Port `dequant_mxfp4_to_fp16` device kernel to host code in
  `src/quant/mxfp4_quant.cpp`. Trivial — block-of-32 nibble unpack +
  ue4m3 scale; the math is identical to the GPU path.

- **Phase B3** — pinned-host staging buffer (~3 days).
  Per-stream pinned-host scratch (`cudaMallocHost`), one bucket per
  max-shape class to avoid per-miss alloc. WSL2 has a hard pinned-mem
  ceiling (~32 GiB on 64 GiB hosts; lower with other apps holding
  pinned mem) — see §5.

- **Phase B4** — integration in `weight_upload.cu` /
  `executor_pre_dequant.cu` (1 week).
  Cache miss → enqueue host-dequant → pinned stage → H2D → GEMM
  proceeds. Must be capture-safe (CUDA graphs path must either skip
  host-dequant entirely or capture only the H2D copy from a
  pre-prepared host buffer).

- **Phase B5** — test + bench (1 week).
  Qwen3.5-27B MXFP4 end-to-end (gated on the N=48 NaN fix from the
  separate memo). Regression on smaller models. Sweep prefill chunk
  sizes to confirm hot-path misses don't blow up PCIe latency.

---

## 5. Risks

### Path A — smart cache policy

- **Profiling assumption fails**. If A1 reveals Qwen3.5-27B has
  hot FP16-cache reads on tensors §3 marked "skip", policy A2 would
  break decode quality or perf rather than fix loading. The
  prerequisite is empirical — A1 must complete before A2 is even
  designed.
- **The N=48 alpha/beta NaN bug**. Even with the FP16 cache pruned to
  fit, GDN decode goes through alpha/beta MXFP4 GEMV which currently
  produces NaN at N=48. Path A doesn't fix this; Qwen3.5-27B still
  needs the kernel fix (separate scope). Path A's value is that it
  removes the *load* blocker so the kernel bug becomes investigable
  at all.
- **PR #60 diagnostic still fires**. The pre-flight check at lines
  1532-1574 is keyed on `fp16_total`. Path A reduces `fp16_total`
  enough to pass it, but if any future model expands MoE/LM-head
  footprint, the check will refuse again — desired behavior.

### Path B — host-dequant

- **PCIe is per-token death**. A single H2D copy of a 5120×17408 FP16
  weight is ~180 MiB at ~16 GiB/s PCIe5 effective → ~11 ms. If that
  fires per token, decode goes from 200 → ~5 tok/s. Path B is **only
  viable if host-dequant is genuinely cold** — i.e. the planner skips
  caching only for tensors that the runtime almost never asks for via
  the FP16 codepath. If they're cold by accident (Path A's
  assumption), they're cold by design here too — same hypothesis,
  just heavier machinery to handle the cold case correctly.
- **WSL2 pinned-memory ceiling**. On a 64 GiB host, pinned-host
  allocation tops out around 32 GiB (WSL2's host-physical reservation
  for the VM) — and that's the *upper* bound; in practice with
  Chrome / VS Code / docker running, useful pinned headroom is closer
  to 16-24 GiB. The 48 GiB nominal FP16 expansion **cannot** live
  fully pinned even if we wanted to. B3 must shard: pin only the
  scratch buffer (one tensor at a time), not the full materialized
  set.
- **Graph capture incompatibility**. CUDA graphs can't capture host
  callbacks cleanly. The decode fast-path uses graphs (see
  `cuda_graphs_moe_works_2026_05_07.md`). If host-dequant fires
  during a captured region, graph capture either aborts or the
  captured graph is broken. The planner must guarantee zero host
  misses inside any captured region — same constraint Path A imposes
  naturally because Path A doesn't introduce host-dequant at all.
- **Implementation cost vs payoff**. The "Day-30-60" stack
  (`review_implementation_2026_05_16.md`) has 6+ open multi-week
  items already in flight (M5 Slice 2, M3 Phase 4, B1, R5 tails).
  Adding a 3-4 week host-dequant project for a single model with a
  separate kernel-NaN blocker is poor schedule allocation.

### Shared risk

- **Qwen3.5-27B MXFP4 may not matter to anyone**. The roadmap entry
  exists because we tried to load it. No user has filed a bug. The
  working alternatives (Qwen3.5-9B Q8_0, Qwen3.6-35B-A3B Q4_K_M)
  cover the same use cases at the same or better quality.

---

## 6. Decision recommendation

**Ship Path A's Phase A1 profiling (1-2 days). Decide from there.**

Justification: the entire host-dequant plan is only worth funding if
the FP16 cache is genuinely hot for Qwen3.5-27B; one day of
instrumentation answers that question definitively, and if the
hypothesis holds (MoE / LM-head / large-FFN entries are cold), Phase
A2+A3 ships the fix in another 2-3 days without any of the
multi-week host-dequant machinery. If the hypothesis fails, defer
entirely — workarounds exist and no user has asked.

## Cross-references

- Memo: `qwen35_27b_mxfp4_ima_2026_04_25.md`
  (`/home/kekz/.claude/projects/-home-kekz-github-com-kekzl-imp/memory/`)
- Code:
  - `/home/kekz/github.com/kekzl/imp/src/model/weight_upload.cu`
  - `/home/kekz/github.com/kekzl/imp/src/graph/executor_pre_dequant.cu`
    (lines 540-660 Phase 1 FP16 cache; lines 1499-1646 MXFP4 →
    FP16 path; lines 2236-2245 MoE FP16 cache lookup)
  - `/home/kekz/github.com/kekzl/imp/src/quant/mxfp4_gemm.cu`
    (GPU dequant kernel; port target for Path B's host dequant)
  - `/home/kekz/github.com/kekzl/imp/src/runtime/vram_budget.cpp`
    (existing footprint estimator; planner extension point)
- PR: `f5738d3` (PR #60) — VRAM-oversubscription diagnostic
- Related plan: `k8_cpu_offload_design_2026_05_17.md`
  (analogous cold/hot bookkeeping for KV pages)
