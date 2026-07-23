# Adversarial Performance Audit — 2026-07-23

Append-only. Fresh measurement pass over every prior "at the ceiling" conclusion,
questioning all previously-refuted levers. Host verified healthy before every
number (Q8 gate tg128 = 287.2 vs baseline 288.02, band [275, 290]; clocks
2890 MHz SM / 13801 MHz mem / ~500 W sampled DURING load, spread 0.05%).
Commit: f95c84aa (+ this audit). Method per docs/BENCHMARKING.md; spec-OFF via
`--set speculative.ngram=false`; nsys with `--cuda-graph-trace=node` (graphs ON).

## Fresh time-budget maps (measured this session)

### Decode, dense NVFP4 (Qwen3-14B-NVFP4, tg256@512 spec-off = 170.6 tok/s)

nsys graphs-ON kernel shares: `gemv_nvfp4_gate_up_fused` 42.1%,
`gemv_nvfp4_residual` 29.9%, `gemv_nvfp4_qkv_fused` 10.5%, lm_head
(`gemv_nvfp4_multirow_fp32`) 4.5% → **~87% of the step is the NVFP4 weight
sweep**. Attention (splitk_fp8 + reduce) ~3.6%, rmsnorm 3.1%, everything else
<2% each. Step time 5.88 ms vs 4.6 ms pure weight-stream minimum (8.3 GB @
1792 GB/s) → **e2e decode runs at ~80% of the absolute weight-bandwidth wall.**
The remaining 20% has 6 refuted kernel levers against it (2026-05→07). No
factor-level gain exists inside the step. CONFIRMS the 2026-07-22 ceiling.

### Decode, MoE NVFP4 (Qwen3-Coder-30B-A3B-FP4, tg256@128 spec-off = 385 under nsys)

Expert GEMVs (`moe_gate_up` 21.5% + `moe_decode_mr` 13.8%) = 35%; dense-side
GEMVs (qkv 10.2% + residual 9.8%) = 20%; attention 10%; routing
(topk_gating 6.1% + gemv_gate_fp32 5.1% + weighted_sum 1.9%) = 13%; norms/rope/
kv-write/swiglu ≈ 14%; lm_head 4.4%. Byte math: expert gate_up streams
~14.2 MB in 11.2 µs ≈ 1266 GB/s = **71% HBM**; expert down ~7.1 MB in 7.2 µs ≈
55%. Kernel-sum ≈ 2.32 ms vs measured step 2.6 ms. The nominal "2.4× headroom
vs pure active-byte minimum" lives in scatter structure + many small kernels —
the launch/latency class was refuted under graphs+PDL twice (2026-07-13
`qwen30b_launch_lever_class_refuted`, mr_nr sweep). No factor here either.

### Prefill (Coder-30B pp4096 single-chunk = 59.8k tok/s; Q8 pp512 = 13.6k; 14B-NVFP4 pp512 = 26.3k)

- **The "legacy materialized causal_softmax + cuBLAS ≈ 18% of prefill" claim is
  dead — 0 launches of `causal_softmax` in any fresh profile** (also confirmed
  by audits 2026-06-07 and DISPATCH_BASELINE_2026-07-17). FA2 family carries
  31% of pp4096 time, grouped-GEMM CUTLASS 20%, quant/permute/scatter ~10%.
- MoE prefill leads vLLM single-seq (#558 closed 2026-07-22; today's pp4096
  number is far above the bar). Prefill is not gated (2.6× cuBLAS restart
  variance) and not the mission bottleneck. Not a lever.

## The re-derived conclusion

Single-stream decode is comprehensively bandwidth/structure-bound; prefill
leads. **The only factor-level lever on this chip is more accepted tokens per
weight sweep — speculation quality.** Fresh evidence that the machinery is
ready and the drafter is the sole gap:

1. **Verify is cheap and lossless.** On the self-repetitive bench prompt the
   existing suffix/n-gram drafter emits 3-token drafts, 100% accepted, lifting
   Qwen3-14B-NVFP4 tg 170.6 → 220–237 (**+29–39%** at k=4..32; chunk-4 verify
   step ≈ 2.2× a decode step, so 4 emitted tokens / 2.2 steps ≈ 1.8×). The
   k-sweep also shows verify cost is flat once the chunk fits the capture
   bucket (tg 216–236 for k = 4..32, all noise-band).
2. **The drafter never fires on reasoning text.** Qwen3-14B-NVFP4, real
   reasoning prompt, 1024 tokens: `drafted=0 accepted=0`, spec-on == spec-off
   (170 vs 177 tok/s, loop-mechanics noise). Coder-30B-A3B on the same prompt:
   **only 19 of 101 probe steps produce any draft** (miss rate 81%); when it
   fires it pays 10 tok/verify — net just +5% (389→414). The entire
   speculation upside on reasoning/agentic prose — the GOAL's core workload —
   is untapped.
3. Scoreboard context: the "14B tg 225" hero number is spec-ON on repetitive
   bench text; the true spec-off decode is 170.6. Speculation already delivers
   +32% where it engages. Reasoning gets 0%.

## Ranked levers

### #1 Token-Recycling adjacency drafter + multi-candidate verify (BUILD — plan docs/plans/2026-07-22-token-recycling-spec-tree.md)

- **Measured now:** 0% spec engagement on reasoning (dense), 19% engagement
  (MoE). Verify economics measured: chunk-4 ≈ 2.2× decode step ⇒ break-even
  accept ≈ 2.2 tok/verify; every accepted token beyond that is ~pure gain.
- **Expected:** +20–50% decode on reasoning-heavy agentic output (Token
  Recycling, ACL 2025: ~2× lossless on general text; conservative here because
  code/structured text already benefits). Roofline-clean: speculation is the
  only mechanism that amortizes the 8.3 GB weight sweep over >1 token.
- **Risk:** moderate — lossless-by-construction (greedy argmax verify is the
  safety net); worst case is wasted verify compute, gated per-milestone.
  Blast radius: `engine_spec_ngram.cpp` draft-fallback hook + new drafter
  files + `greedy_argmax_all` top-K extension. Flag-gated
  (`speculative.token_recycling`, default off) for clean A/B.
- **Acceptance criteria (hard):** on the reasoning-prompt battery,
  spec-ON(TR) wall-clock decode ≥ +10% vs spec-OFF on Qwen3-14B-NVFP4 with
  byte-identical greedy output; no regression on the bench-prompt path or
  `verify-fast`; kill per plan if accept stays < 1.3 tok/verify at milestone 3.

### #2 Cheapen the verify step itself (supporting lever for #1)

Chunk-4 verify at 2.2× a decode step is more than the GEMM bytes justify
(weights are read once either way; M=4 vs M=1 is ≤1.2× on the GEMV side).
The overhead sits in the ctx-tier attention + eager-loop host work (#964
measured the tier effect; #1001 already cut the GEMMs). Every 0.1× shaved off
verify cost lowers the break-even accept for #1 linearly. Measure-first via
nsys on a verify-heavy run; only build if a ≥0.3× reduction is visible.

### #3 MoE decode expert-GEMV streaming (LOW confidence — do not build now)

Expert gate_up at 71% / down at 55% HBM leaves nominally ~1.3–1.8× inside the
MoE window (35% of step) ⇒ ≤ +15% e2e best case. Adjacent levers were refuted
(mr_nr, launch fusion, splitk cap). Revisit only with a concrete ncu stall
analysis showing a fixable structural stall, not before.

Everything else touched this pass (prefill coverage, legacy attention path,
launch classes, cuBLAS variance as a mean-shift lever, routing overhead) is
either already closed, refuted under graphs, or not on the gated metric.

---

## Implementation results (same session, lever #1 milestones 1–3)

Shipped flag-gated (`speculative.token_recycling`, default OFF):
adjacency table (`token_recycle_draft.{h,cpp}`), top-M logit harvest in the
verify lm-head pass (`rowwise_topm.cu` + `greedy_argmax_all` extension), and
the **route-(a) multi-candidate verify** — per-candidate private KV blocks
via the existing per-row block tables (`kv_get_block_id` resolves reads AND
writes per row at n_sequences>1), each candidate re-forwards t0 into its own
partial-block copy (`KVCache::copy_blocks_device`), winner block copied back
before rollback. No token-level mask needed; hybrid/MoE/MLA/SWA/penalties/MTP
excluded (linear fallback). 15 new host/GPU tests; full GPU suite green;
verify-fast green; greedy output byte-identical to spec-off (both linear and
multi-candidate arms).

**Measured (Qwen3-14B-NVFP4, reasoning prompt, healthy host):**

| arm | tg tok/s | engagement | emitted/verify |
|---|---|---|---|
| spec-off | 170.7–171.4 | — | — |
| default (suffix) | 171–178 | 4 verifies/1024 tok | 13.75 when firing |
| TR linear (m1+2) | 168–170 | doomed after 8–13 | 1.55–1.88 |
| TR multi-candidate | 168–170 (doom-protected) | 12 verifies | 2.00 |
| TR mc, suffix off, no doom | 116–129 | 370 verifies/1024 | 1.74–1.86 |

**Verdict: the drafter works (0 → ~2.0 emitted/verify on reasoning) but the
verify step is too expensive for it to pay.** Break-even accept = verify
cost / decode-step cost ≈ 1.9–2.2 today. Fresh decomposition of the
captured verify step (nsys, bucket 17, ctx tier 4096): wall ≈ 18 ms vs
decode step 5.9 ms, split into

1. **~8.2 ms M=17 GEMMs running as CUTLASS NVFP4 kernels** (≈200 launches ×
   39–51 µs) — on native-NVFP4 ST models the `verify_nvfp4_gemm` overlay
   (#1001) does not engage, so the chunk pays the prefill GEMM path at
   ~55–60% of the GEMV-equivalent effective bandwidth (GEMV sum ≈ 4.8 ms
   for the same weights).
2. **~7 ms host overhead** — `cudaStreamSynchronize` is 45% of total API
   time (avg 3.9 ms/call on WSL2/WDDM), plus per-step staging H2D and the
   mc copy-back sync.

**Follow-up (the actual unlock, = lever #2):** route native-NVFP4 M≤17
verify chunks through the multirow GEMV kernels (`gemv_nvfp4_*_mr`) instead
of CUTLASS, and cut the per-verify sync/staging overhead. Getting the
verify step to ~1.2× a decode step drops break-even accept to ~1.2 — the
measured TR accept of 1.7–2.0 then yields **+40–65% decode on reasoning**
without any drafter improvement. Token-Recycling stays default-off until
that lands.

---

## TTFT addendum (same session): the short-prompt floor

Prefill wall by prompt size, warm engine, 5 reps (healthy host):

| pp | 14B-NVFP4 | Coder-30B-FP4 |
|---|---|---|
| 32 | 18.8 ms (1.7k tok/s) | 14.7 ms |
| 128 | 12.0 ms (10.6k) | **24.3 ms** (5.3k) |
| 512 | 19.8 ms (25.8k) | 21.5 ms (23.8k) |
| 2048 | 62.0 ms (33.0k) | 40.3 ms (50.8k) |

Short prompts — the agentic incremental-turn case — sit on a **~12–25 ms
floor** while the same GEMM work at pp2048 streams at 33–50k tok/s; two
anomalies (14B pp32 > pp128; Coder pp128 > pp512) mark path-selection
effects, not noise. Decomposition at 14B pp128 (nsys): ~9 ms GPU in M=128
CUTLASS NVFP4 GEMMs at ~51% of the weight-sweep bandwidth bound (4.6 ms
ideal) + ~2.5–10 ms host/launch gaps — **prefill graph capture is disabled
on every ST-native NVFP4 hero** because the workspace gate treats the
~1.5 GiB fp16 dequant target of the **lm_head** as capture-blocking
(`executor_workspace_buffers.cu` kCap=512 MiB check sets
`nvfp4_dequant_uncapturable_`), even though prefill only ever runs the
lm_head at M=1 (last position) and never through the M>1 dequant fallback.

TTFT levers, ranked (short-prompt TTFT ≈ prefill + 1 decode step ≈ 18–30 ms
today, ~11–13 ms reachable):

1. **Small-M NVFP4 GEMM efficiency — same root as issue #1055.** The
   M≤128 CUTLASS class runs at ~half effective bandwidth; fixing it serves
   the verify chunk AND the short-prompt prefill (~3–4 ms at pp128).
2. ~~Exempt head/embed weights from the prefill-capture dequant gate~~ —
   **BUILT AND REFUTED (same day, reverted).** The gate's stated reason IS
   wrong (the ~1.5 GiB lm_head dequant target blocks capture although the
   captured prefill runs the lm_head at M=1 only — the exemption worked and
   capture engaged cleanly), but enabling capture on ST-native NVFP4 models
   measured net-NEGATIVE to neutral: interleaved A/B on 14B-NVFP4, 8–30
   reps, `runtime.prefill_graph` on/off — pp128 15.3–23.6 ms (on) vs
   12.6–15.3 ms (off); with a capture hysteresis added (capture only on
   repeated geometry) still 13.3 vs 12.8 ms at 30 reps. Root cause: the
   NVFP4-ST eager prefill issues only ~240 CUTLASS launches, so graph
   replay saves nothing, while capture/instantiate costs ~3–9 ms whenever
   the geometry or engine resets (every CLI/bench rep; 13 instantiates in a
   10-rep run). GGUF (Q8) keeps a marginal capture win (25.0 vs 26.3 ms at
   pp128) — its default stays as-is. The misleading WARN is cosmetic; the
   real short-prompt lever is the small-M GEMM path (item 1 / issue #1055).
3. Coder-30B pp128 anomaly (24.3 ms > pp512's 21.5 ms): grouped-GEMM
   tokens-per-expert collapse at small batch — needs its own scoping.
4. Multi-turn TTFT is already served by the prefix cache (23× flat,
   2026-07-15); long-context TTFT (8k–64k) is pinned and healthy (#1022).
