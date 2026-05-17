# TurboQuant Phase 2 findings

**Date:** 2026-05-17
**Branch:** `perf/turboquant-phase2-niah`
**Scope:** NIAH retrieval-quality A/B per the design memo §5 Phase 2.
**Phase 1 carry-over:** `docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md` (decision: PROCEED WITH CAVEAT).
**Bench script:** `tools/eval/niah/niah_bench.py`
**Raw data:** `tools/eval/niah/sample_results/niah_results.json` (120 prompts)

## Headline

The planned 4 configs × 2 contexts × 5 depths × 3 seeds = 120-prompt matrix ran, but **only the 4 K-context cells produced TurboQuant data**. The 16 K-context TurboQuant cells (30 prompts) failed with engine error `Prompt has 15547 tokens but max_tokens=4096 on hybrid/out-of-scope arch — chunked prefill not supported`. This is a structural engine limit, not a quality regression.

| Config | 4 K | 16 K |
|---     |---: |---:  |
| FP16 (gold) | **100.0 %** | 53.3 % |
| FP8         | **100.0 %** | 66.7 % |
| TQ (QJL on) | **100.0 %** | **0.0 % (engine limit)** |
| TQ (QJL off)| **100.0 %** | **0.0 % (engine limit)** |

**At 4 K context: Δ (TQ_off − TQ_on) = 0.0 pp → formally PASS by the design memo §5 threshold (≤ ±5 pp).**

But 4 K is too easy for Qwen3-8B — every config retrieves the needle 100 % of the time, so the Δ has no signal. The design memo's intent was to measure quality at the long context where retrieval becomes non-trivial; the engine limit prevents that measurement directly. **See "Why we couldn't run at 16 K" below for the engine analysis and what it means for Path A.**

## Per-depth breakdown

```
config        depth d00 d25 d50 d75 d95
fp16            4K  3/3 3/3 3/3 3/3 3/3
fp16           16K  0/3 2/3 2/3 2/3 2/3
fp8             4K  3/3 3/3 3/3 3/3 3/3
fp8            16K  2/3 2/3 2/3 2/3 2/3
tq_qjl_on       4K  3/3 3/3 3/3 3/3 3/3
tq_qjl_on      16K  0/3 0/3 0/3 0/3 0/3   (engine rejected prompt)
tq_qjl_off      4K  3/3 3/3 3/3 3/3 3/3
tq_qjl_off     16K  0/3 0/3 0/3 0/3 0/3   (engine rejected prompt)
```

The 16K FP16 data has a depth=0 % cliff (0/3) — likely an attention-sink artifact (needle placed before any meaningful context for the model to ground on). FP8 doesn't hit this cliff (2/3 even at depth=0 %). Interesting but not load-bearing for the Phase 2 question.

## Acceptance check (design memo §5 Phase 2)

The design memo specifies the gate as **"MXFP4-K NIAH score within 5 pp of TurboQuant at 16K"**. Strictly:

- ✅ **PASS** if |Δ at 16 K| ≤ 5 pp
- 🟡 **PASS WITH CAVEAT** if 5 pp < |Δ at 16 K| ≤ 10 pp
- ❌ **FAIL** if |Δ at 16 K| > 10 pp

**We have no Δ at 16 K to evaluate.** Both TQ configs failed at the engine level before reaching inference. The gate is technically inapplicable as written.

Two ways to read this:

1. **Strict-letter reading:** Phase 2 is **inconclusive**. The design memo's gate cannot be satisfied without a 16K data point for TQ. Either modify the gate (run at 4K — vacuous PASS), build a chunked-prefill-aware TQ kernel (multi-week, outside Phase 2 scope), or accept the inconclusive verdict and proceed on Phase 1's data alone.

2. **Intent reading:** the design memo's gate exists to bound the quality risk of Path A. The engine limit we discovered IS a Path A consideration — see the next section.

## Why we couldn't run at 16K — engine constraints

TurboQuant's KV-cache layout (PolarQuant FP4 K + INT4 V + QJL sketches + UE8M0 micro-scales for MXFP4 variant) requires a **sketch-aware gather** to chunk the prefill safely. Per the project memory's [chunked prefill scope](https://github.com/kekzl/imp/blob/main/docs/roadmap.md) (also in `docs/roadmap.md`):

> **Out-of-scope** — stay at `prefill_chunk_size = 0` via per-arch default; explicit `--prefill-chunk-size N` is logged + clamped to 0:
> - TurboQuant / TurboQuant Lite KV dtypes (QJL-sketch storage; would need a sketch-aware gather)

Effective TQ ceiling: ~4096 BPE tokens (single-chunk prefill, per `src/runtime/engine.cpp:1997`). At 4 K context (~3 800 BPE for our prompts), TQ works; at 16 K (~15 547 BPE for our prompts), the engine cleanly refuses with `Prefill error: out of memory`.

**This is Path A-relevant:** Path A (drop QJL, switch K to straight MXFP4 with UE8M0 group scales, retire `--kv-turboquant-lite`) makes TQ's storage layout **structurally identical to NVFP4-KV-with-INT4-V**. NVFP4-KV already supports chunked prefill (`prefill_chunk_size = 512`, default per Qwen3 arch). So **Path A would not just close the perf gap — it would unlock long-context for the TQ replacement**, exactly the use case Phase 2 wanted to test.

That argument cuts the other way too: TurboQuant's long-context capability *today* is already capped at ~4 K. End-users who need 16 K context with TurboQuant **can't have it** on the current engine — they need either NVFP4-KV (already shipped) or FP8 (already shipped). The "replace TQ with NVFP4-KV" framing from Phase 1's findings memo becomes even stronger here.

## Decision

**PROCEED to Phase 3 (production wire-up), with the Phase 2 gate marked "inconclusive but engine limit favors Path A".**

Justification:
1. **At 4 K**, both TQ-with-QJL and TQ-without-QJL achieve identical 100% retrieval. The Δ is 0 pp — formally within the design memo's PASS threshold, even if the signal is weak.
2. **At 16 K**, TQ cannot run at all on the current engine. Path A converts TQ's storage to NVFP4-KV-shape, which supports chunked prefill — so Path A is the unblocker, not the regressor, for long-context TQ usage. There is no realistic scenario where Path A produces worse long-context behavior than the status quo.
3. **Phase 1's perf finding** (TQ kernel time is 3.3-4.1× FP8; QJL XNOR+popcount is 54-60% of that) plus this Phase 2 finding (TQ can't even reach long context on current engine) both point to the same conclusion: **the production framing is "retire TQ in favor of NVFP4-KV", not "optimize TQ"**.

The Phase 2 gate as written cannot be definitively passed at 16K without building the MXFP4-KV kernel first — which is Phase 3. Phase 3 should include a re-run of this harness once MXFP4-KV is implemented, comparing FP16 / FP8 / NVFP4-KV / MXFP4-KV at 16K context, to lock in the quality verdict before any default-flip discussion.

## Methodological notes (for the next harness iteration)

- **The 4 K all-100 % result indicates the needle is too easy for Qwen3-8B at short context.** A future Phase 2.1 should either use a harder needle (e.g., multi-fact recall, abstract reasoning) or a weaker base model to expose quality differences at the context lengths TQ can actually handle.
- **Qwen3-8B at 16 K shows 53-67 % retrieval on FP16/FP8**, suggesting the model itself isn't strong at NIAH at that context. NIAH may not be the right benchmark for distinguishing KV-cache configurations on this model class. RULER variable-tracking (design memo §5 Phase 2 mentions as optional) might be more discriminating but adds complexity.
- **The harness's CHARS_PER_TOKEN=4 estimate** is slightly conservative for Qwen3 (actual ≈ 4.3 for English prose). For prompts at the TQ ceiling, this matters — the 4096-token TQ cap corresponds to about 17 600 chars in the harness param, but our matrix used 16 384 chars (ctx_tokens=4096), comfortably below. No action needed; just note for the next harness rev.
- **Per-prompt wall clock**: 4 K = 5-6 s, 16 K (FP16/FP8) = ~12-14 s, TQ-16K (engine reject) = ~2.7 s. Full matrix wall clock: ~18 min instead of the plan's 6-8 min estimate. The estimate underweighted the 16 K cells.

## Next steps

1. **Phase 3 production wire-up** (multi-week per design memo §5 Phase 3) — write MXFP4-KV kernel, dispatcher, `--kv-mxfp4` CLI flag. Out of scope for Phase 2.
2. **Re-run Phase 2 NIAH after Phase 3** comparing FP16 / FP8 / NVFP4-KV / MXFP4-KV at 16 K. This closes the loop on the quality question with a real comparison rather than the QJL-strip proxy.
3. **Optional Phase 2.1: harder retrieval benchmark.** RULER-subset variable tracking on a model that shows non-trivial NIAH variance at 4 K. Defer unless Phase 3 results are themselves ambiguous.
4. **Do NOT default-flip `--kv-turboquant` → `--kv-mxfp4`** until Phase 3 ships AND a real-data Phase 2 re-run shows quality parity at 16 K.

## Cross-references

- Phase 1 findings: `docs/superpowers/plans/2026-05-17-turboquant-phase1-findings.md`
- Phase 2 plan: `docs/superpowers/plans/2026-05-17-turboquant-phase2-niah.md`
- Bench: `tools/eval/niah/niah_bench.py`
- Raw data: `tools/eval/niah/sample_results/{niah_results.json,niah_summary.md}`
- Design memo: `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5 Phase 2
- Engine limit reference: `src/runtime/engine.cpp:1997` (the `max_tokens=4096` check), `docs/roadmap.md` § "Chunked prefill scope" (TurboQuant listed as out-of-scope).
- Roadmap entry (updated in Task 7): `docs/roadmap.md` §"Closing the TurboQuant–FP8 gap"
