# TurboQuant Phase 1 findings

**Date:** 2026-05-17
**Branch:** `perf/turboquant-phase1-microbench`
**Scope:** Verify the design memo §1.2 "QJL is the bottleneck" hypothesis on Qwen3-8B Q8_0 before committing to the Path A kernel rewrite.
**Decision gate:** `docs/plans/turboquant_fp8_gap_design_2026_05_17.md` §5
**Bench script:** `tools/analysis/bench_turboquant_components.sh`
**Raw data:** `/tmp/tq_phase1/{tq_full,tq_stripped,fp8,nvfp4}_pp{512,4096}.nsys-rep`

## Measurements

Qwen3-8B Q8_0 (HuggingFace `Qwen/Qwen3-8B-GGUF`), RTX 5090 sm_120a, CUDA 13.2, `imp:test` Docker image, `--no-cuda-graphs`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, 3 nsys reps × 256 generated tokens after prefill, `--temperature 0`. Per-attention-call average kernel time of `paged_attention_splitk_turboquant_kernel<128, USE_MXFP4=true, SKIP_QJL=...>` (TQ) or its FP8/NVFP4 sibling, extracted from `nsys stats --report cuda_gpu_kern_sum`. 36,864 invocations per run.

| Config                          | Avg (ns) pp=512 | Avg (ns) pp=4096 | Δ vs FP8 pp=512 | Δ vs FP8 pp=4096 |
|---                              |              ---:|              ---:|             ---:|             ---:|
| **TQ full**                     |          29 875 |          107 089 |          **+230 %** |          **+314 %** |
| **TQ stripped** (IMP_TQ_SKIP_QJL=1) |     13 517 |           42 567 |              +49 % |              +65 % |
| **FP8** (perf target)           |           9 052 |           25 870 |               0 %  |               0 %  |
| **NVFP4** (Path A ceiling proxy)|          10 908 |           32 796 |              +20.5 %|             +26.8 %|

### Derived acceptance metrics

```
qjl_fraction  =  (TQ_full − TQ_stripped) / TQ_full
ceiling_gap   =  (NVFP4   − FP8)         / FP8
```

| Metric        | pp=512  | pp=4096 | Threshold (design memo §5) | Verdict |
|---            |     ---:|     ---:|---:|---|
| qjl_fraction  | **54.7 %** | **60.3 %** | ≥ 15 % | ✅ **PASS** (>3× the threshold) |
| ceiling_gap   | **20.5 %** | **26.8 %** | ≤ 5 %  | ❌ **FAIL** (~4-5× over) |

## Acceptance criteria evaluation

- ✅ **QJL fraction ≥ 15 % → Path A bottleneck-targeted.**
  The QJL XNOR+popcount correction + Q-side sketch precompute accounts for **over half** of TQ decode kernel time, at both short (pp=512) and long (pp=4096) context. The roadmap's "QJL is algorithm-inherent overhead" diagnosis is **confirmed and stronger than predicted** — the design memo's bracket was 1.20-1.30× FP8 cost (§2.4); the actual TQ-full is **3.3×** FP8 cost at pp=512 and **4.1×** at pp=4096. Removing the QJL path is structurally the right lever.

- ❌ **Ceiling gap ≤ 5 % → Path A perf ceiling NOT confirmed.**
  NVFP4 KV (the proxy for Path A's post-rewrite storage layout) runs **20-27 %** slower than FP8 per attention call. Path A will close most of the 3.3-4.1× TQ-vs-FP8 kernel gap, but will NOT reach FP8 parity; it'll leave a residual ~20-27 % gap from the K-norm extra FP16 load + INT4 V dequant + scale-pool indirection compared to FP8's single-byte E4M3 encoding.

## Decision

**B — PROCEED WITH CAVEAT.**

Path A is bottleneck-targeted and recovers most of the kernel-time gap, but won't deliver FP8-parity TurboQuant. The right framing is **"retire TurboQuant in favour of NVFP4-KV, not optimise it"** (design memo §6 secondary framing) — Path A's storage shape is structurally a thin variant of NVFP4-KV-plus-INT4-V, and imp already has NVFP4-KV in main at ~20-27 % slower than FP8. The big win is the **−2000 LOC code retirement**, not headline perf parity with FP8.

### What Path A would realistically deliver

Closing the kernel-time gap from 3.3×/4.1× → ~1.2-1.3× FP8 cost (matching NVFP4's residual). End-to-end decode tok/s improvement is **bounded by weight-bandwidth-boundedness** per `bitdecoding_long_context_eval_2026_05_14.md` — attention-kernel-only wins typically translate to a fraction of their kernel-time recovery at the tok/s level. The roadmap's 23 % end-to-end gap might close to ~8-12 %, not zero.

### What this means for the design memo's Path A vs Path B

- Path B (optimise QJL per-token, per §3.2) targets a 3-5 % recovery at best — not worth the 3-5 weeks of incremental kernel work given QJL is dominant at 55-60 %, not the marginal 6-10 % the memo §2.3 estimated.
- Path A's full execution (Phases 2-5 of the design memo) remains the right shape, but the framing in Phase 4 ("default-flip if MXFP4-KV is within 3 % of FP8") should be updated — the realistic target is "within 25-30 % of FP8 per attention call." Whether that's worth a default flip depends on whether NVFP4-KV is already a better fit for the workloads TQ was intended to cover (the design memo §1.3 already argues NVFP4 is the right tool for Klasse-A models).

### Other surface findings

1. **Both decode paths dispatched through the splitk variant on Qwen3-8B Q8_0** at all measured contexts (pp=512 generating 256 tokens drives ctx up to 768; pp=4096 → 4352). `paged_attention_decode_turboquant_kernel` (non-splitk) was never invoked. This matches `compute_splitk_splits` choosing num_splits>1 for batch=1, n_heads=32 attention on 32-block contexts. The splitk vs non-splitk performance ratio could shift with single-token decode in real chat scenarios; the kernel-time numbers above are representative of bench-mode decode loops, not interactive single-token decode.
2. **TQ-full's pp=4096 std-dev is 35 % of mean** (37 868 ns / 107 089 ns) — vs FP8 13.2 % and NVFP4 13.8 %. The QJL-on path is meaningfully more variance-prone, possibly from the `atomicOr` in Q-sketch precompute racing across warps. Not a load-bearing finding but worth noting.
3. **TQ-stripped is still 49-65 % slower than FP8** per kernel call (13 517 vs 9 052 ns at pp=512; 42 567 vs 25 870 at pp=4096). This **bounds the residual cost** that NVFP4-KV-with-UE8M0 would NOT recover — about half of the TQ-stripped-vs-FP8 gap is from K-norm + INT4 V + scale pool indirection (i.e., what Path A inherits from the current TQ storage shape), and the other half is the structural difference between PolarQuant (norm + direction split) and NVFP4-KV (per-block E4M3 scale).
4. **ncu data not captured** — the bench script's `ncu` invocations failed silently inside the docker container (host `/usr/local/cuda/bin/nsys` and `ncu` are wrappers that fail with "CUDA Toolkit 13.2 not installed" when run against the imp:test image's CUDA install). nsys reports themselves were generated fine via `nsys profile`; only the post-processing `nsys stats` and `ncu` paths hit the wrapper issue. The host `/usr/local/bin/nsys` works for `stats` extraction (used for this memo). Not load-bearing for the decision; SASS-level instruction breakdown was a nice-to-have, not a gate.

## Next steps

**Phase 2 (NIAH retrieval-quality A/B)** — per design memo §5. Build a minimal needle-in-a-haystack harness in `tests/long_context/` and compare FP16 (gold) / FP8 / TQ (QJL on) / synthetic-MXFP4-K (QJL off, via `IMP_TQ_SKIP_QJL=1`) at 4K and 16K context on Qwen3-8B Q8_0. Acceptance per the design memo: MXFP4-K NIAH score within 5pp of TQ at 16K → green-light Path A; 5-10pp regression → ship with caveat docs; >10pp regression → Path A refuted, shelve.

Phase 2 work item is ~4-6 days. It is the next decision gate; only after Phase 2 lands does Path A become an unambiguous PROCEED.

If Phase 2 regresses, the fallback per the design memo §6 worst-case is to **shelve entirely and keep TurboQuant opt-in with the documented 23 % caveat** — which, given the Phase 1 numbers show TurboQuant is actually 3-4× slower at the kernel level (not 23 % end-to-end), should be revisited as part of a broader TurboQuant retirement scoping doc.

## Cross-references

- Design memo: `docs/plans/turboquant_fp8_gap_design_2026_05_17.md`
- Plan: `docs/superpowers/plans/2026-05-17-turboquant-phase1-microbench.md`
- Bench script: `tools/analysis/bench_turboquant_components.sh`
- Roadmap entry (to be updated in Task 8): `docs/roadmap.md` § "Closing the TurboQuant–FP8 gap"
- Memory pointer (to be added in Task 7 Step 2): `memory/turboquant_phase1_findings_2026_05_17.md`
- Referenced memos: `kv_dtype_tradeoffs_2026_04_24.md`, `bitdecoding_long_context_eval_2026_05_14.md`, `nvfp4_kv_potential_2026_04_25.md`
