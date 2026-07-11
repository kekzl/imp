# Roadmap

Single-author, single-GPU experiment -- "roadmap" means "current focus," not "schedule." Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md); competitive numbers live in [`docs/BENCHMARKS.md`](BENCHMARKS.md).

## Direction: local inference for AI agents

The goal is making imp the fastest local engine for AI agent workloads on consumer Blackwell. Agents generate far more tokens per session (20k-100k+), accumulate context fast, and often run in parallel. This demands long context, concurrent request handling, and high decode throughput.

### Foundations (shipped 2026-05)

- **Long context** (#453) -- chunked-prefill FMHA (`q_offset`), S-matrix 1024→256 MiB, auto `fmha_prefill_threshold`. Context ceiling ~4-6k → 32k+.
- **Concurrent requests** (#454) -- multi-request decode batching (`runtime.max_batch_size`).
- **KV streaming** (#455) -- StreamingLLM auto-enables when the KV cache runs full: sink tokens + sliding window, agent sessions effectively unlimited.

## Current focus: operational robustness for agent workloads

The engine is past the raw-speed land-grab; current work is making it boringly reliable to *operate* under agent load:

- **Fast (re)starts** -- on-disk warm weight cache (cold boots skip weight conversion, #956) and suspend-to-RAM (`/admin/suspend`/`resume`: free the GPU in seconds, resume without re-reading weights, #954).
- **Determinism as a product property** -- greedy request-order independence (decode-graph pool pre-armed in warmup, `runtime.warmup` default-on, #957); see [`determinism.md`](determinism.md).
- **Model-support debt burn-down** -- last hard crash (gemma-3-12b GGUF decode IMA) fixed in #959; remaining blockers under "Known limitations".
- **MLA family expansion** -- DeepSeek-V2-Lite is validated (#802/#803 latent-KV decode, opt-in); DeepSeek-V3 / GLM / Kimi / Ling reuse the same path once weights are staged locally.

## Performance work

The batch=1 *competitive campaigns* are closed as programs -- every lever they left open either shipped or was refuted by measurement -- but targeted wins keep landing where new levers appear:

- **FA2 hd=256 prefill default-on** (#930/#932) -- Qwen3.6/Qwen3.5 hybrids, pp4096 +26% over the WMMA path it replaced.
- **FP8 tile attention** (#899/#900) -- FP8-KV decode tiles + GQA batching, long-context decode +14%.
- **FP8 SSM projection sidecar** (#949) -- per-row-scale FP8 for GDN in/out projections; Qwen3.6-35B NVFP4 decode +19% (tg ~320). Extended to GGUF hybrids' Q8_0-kept GDN projections (dequant→FP8 at init): 35B UD-Q4_K_M decode +21% (tg 272, ahead of llama.cpp) -- closed the last decode combo where llama.cpp led.
- **Speculative decoding economics** (#852/#862-#866) -- hybrid-safe verify + MTP drafts; echo-heavy agent workloads up to +156% on 27B.

Closed competitive records (kept for the record, not active work):

- **NVFP4 prefill vs vLLM -- CLOSED** (re-measured 2026-06-13, commit `290a163a`). FP16-QK FA2 as primary hd=128 prefill lifted pp4096 +21-24%: MoE pp4096 +4% ahead of vLLM, MoE pp2048 +27%, dense pp2048 ~tie. The lone residual gap -- dense pp4096 at ~1.04× -- is structural: every bounded kernel idea (cross-tile pipeline, grouped-GEMM tile axis, chunk-4096, occupancy/2-CTA, fp8-QK, scaled fp8-KV) was measurement-refuted; at pp4096 FA2 sits at ~5% DRAM and the dominant cost is the NVFP4 GEMMs (~59%), a separately-refuted ceiling.
- **kv-fp8 storage default-on -- SHIPPED** for Qwen3 dense/MoE, Llama (Phi-4), Nemotron-H MoE (`kv_cache.dtype=auto` honors the model's FP8 hint where the long-context quality gate passes; ~768 MiB KV saved on dense). Remaining families are blocked, not actionable: Qwen3.6-35B / Qwen3.5 declare no FP8 hint; Gemma-4's baseline PPL on the gate corpus is broken. These stay FP16 (or `--kv-fp8` opt-in).
- **Q4_K_M prefill gap (-38% vs llama.cpp) -- evidence-refuted.** The in-SMEM Q4_K MMQ + HMMA kernel was built (`feat/q4k-mmq-hmma`) and ncu-proved decode-throughput-bound, tying cuBLAS -- closing the gap needs beating cuBLAS or paying 2× weight VRAM (rejected). Practical resolution: use NVFP4 SafeTensors for fast Q4_K-class prefill. Details: [`plans/2026-05-28-q4k-mmq-kernel-design.md`](plans/2026-05-28-q4k-mmq-kernel-design.md).
- **Sawtooth wavefront reordering (#456) -- refuted** (measured 2026-05-29: only lives in the WMMA fallback, unreachable on the hot path; force-routed A/B flat-to-negative). Harness: `tools/analysis/sawtooth_ab.sh`.

## Known limitations

- **Single GPU only.** No tensor parallelism, no multi-GPU.
- **Blackwell only.** No Hopper, Ada, Ampere. No AMD, Intel, Apple, CPU.
- **Qwen3.5-27B MXFP4 fails at load** -- blocked on no public MXFP4 GGUF + NaN bug.
- **Gemma-4 Q4_K_M code-gen drift** -- no longer reproduces (verified 2026-06-13 on the current UD-Q4_K_M; the original file is gone, so it can't be A/B'd). If some other Q4_K_M quant of this model degenerates, fall back to Q5_K_M or Q8_0.

## Investigated and shelved

- **Draft-model speculative decoding** -- separate draft models don't amortize weight reads on a single bandwidth-bound GPU. What *did* ship instead: prompt-lookup n-gram speculation (default-on for batch-1 greedy dense, #668-#670) and MTP self-drafts with hybrid-safe verify (#852) -- the drafts are free, so the economics work.
- **FFN contextual sparsity** -- warp-cooperative layout masks the skip. +0-1% measured.
- **BitDecoding (TC KV decode)** -- decode is weight-bound, not attention-bound. 0% gain.
- **NVFP4 GEMV tuning** -- 6 approaches refuted; decode GEMV runs at 64-73% of HBM peak, structurally bandwidth-bound.
- **FMHA rewrites** -- cluster, TMA bulk, long-context heuristic all A/B tested. cuBLAS wins.
- **MoE offload + CUDA Graphs** -- `expert_overhead_pct=10` default keeps most models on-device. Full kernel-driven slot resolution deferred (multi-week, marginal user impact).
- **CUDA Tile (cuTile C++) -- benchmarked on sm_120, shelved** (2026-05-29). A correct cuTile FA2 autotuned to 26.5 eff-TFLOPS = 3.2% of roofline, order-of-magnitude below imp's hand-written FMHA -- confirms the published 0.53×-FA2 result on this arch (vs 2.5× on B200). Re-evaluate only on a new toolkit showing ≥parity on sm_120. Harness: `tools/analysis/cutile_fa2.py` + `Dockerfile.cutile`.
- **CompileIQ ptxas auto-tuning -- refuted** (2026-05-29). The ptxas search space is flat on imp's hotspots: FA2 is smem-occupancy + barrier-bound, NVFP4 decode is HBM-bound -- codegen touches neither (all sweep points within ±0.4%). Reusable harness: `tools/analysis/Dockerfile.ciq` + `tools/analysis/ptxas_sweep.sh`.
