# imp — Goal Definition

## Mission

**imp shall be the fastest single-GPU LLM inference engine on NVIDIA RTX 5090 (sm_120) for batch size 1, across the model architectures it supports.**

Single-user, single-GPU, latency-first. Not a competitor to vLLM/SGLang on throughput. A weapon for the workstation.

---

## Definition of "best"

Ranked, non-negotiable order:

1. **Decode throughput (tok/s) at batch=1** on RTX 5090 — primary metric. Must lead llama.cpp, vLLM, SGLang, ExLlamaV3, MLC-LLM on every supported architecture and quant combination we ship.
2. **Prefill throughput (tok/s) at batch=1** on RTX 5090 — must match or beat llama.cpp on dense; must close the MoE gap to vLLM (Qwen3-Coder NVFP4 pp512 was 1.14-1.32× behind vLLM 0.20.2 single-seq as of 2026-05-10; closed to **1.056× as of 2026-05-23** post-#374 skip-gather, see *What "best on 5090" requires* below).
3. **Time-to-first-token (TTFT)** at realistic prompt lengths (512, 2048, 8192, 32k) — must be competitive with vLLM despite our batch=1 focus.
4. **VRAM efficiency** — must fit larger models than competitors at equivalent quality (NVFP4, FP8 KV, paged cache). 32 GB on a 5090 should serve everything up to ~70B dense at usable quality.
5. **Quality** — perplexity and downstream eval parity with llama.cpp at the same quant. No silent quality regressions for speed.

Anything below this bar is a bug.

---

## Target hardware (in priority order)

1. **RTX 5090** (sm_120, GB202, 32 GB GDDR7) — the hero target. Every optimization decision is made for this chip first.
2. **RTX PRO 6000 Blackwell** (sm_120, 96 GB) — same arch, more VRAM. Free win if 5090 is fast.
3. **H100 / H200** (sm_90a) — Hopper path stays alive but is not the hero. Maintained, not optimized aggressively.
4. **Consumer Blackwell siblings** (5080, 5070 Ti, etc.) — same sm_120, should work, lower priority for tuning.

Everything else is best-effort via cuBLAS + scalar Flash Attention 2 fallback. Not a goal.

---

## Target models (hero set)

These models must be **best-in-class on 5090**. No exceptions, no excuses:

| Model | Quant | Why |
|---|---|---|
| Qwen3-4B / 8B | Q8_0, NVFP4 | Daily driver dense, fast iteration |
| Qwen3-14B | Q6_K, NVFP4 | Sweet spot for 5090 |
| Qwen3-Coder-30B-A3B | Q6_K, NVFP4 | Hero MoE — the prefill gap closes here |
| DeepSeek-R1-Distill-7B/14B | Q8_0, Q6_K | Reasoning baseline |
| Gemma-3 27B (text + vision) | Q4_K_M, NVFP4 | Multimodal hero |
| Gemma-4 26B-A4B | NVFP4 | MoE follow-up (integration active) |
| Phi-4 14B | Q6_K | Dense alt |
| Mixtral 8x7B | Q4_K_M | MoE classic |
| Nemotron-H | NVFP4 | Hybrid (Mamba2 + Attn + MoE) flagship |
| gpt-oss-20b | MXFP4 | First-class MXFP4 path |

If a hero model regresses against any competitor on the primary metric, that is a release blocker.

---

## What "best on 5090" requires

Concrete technical commitments — these are means, not ends, but progress on the mission is measured against them:

### Compute path
- **NVFP4 must remain the default fast path** on sm_120. FP16/BF16 paths exist for correctness, never as the recommended config. The Blackwell consumer FP16/BF16 throughput nerf is a hardware fact; NVFP4 is the way out and we lean into it harder than anyone.
- **MXFP4 FMHA** (already novel — first such impl) must stay ahead and expand to more shapes. The +6.7–7.9% over FP8 FMHA on Qwen3 is the baseline, not the ceiling.
- **CUTLASS Hopper FMHA path on sm_120** for prefill — keep up with CUTLASS releases.
- **WMMA 8-warp decode kernel** is the decode workhorse on sm_120. Tune for every hero model's head dim.
- **Grouped GEMM dequant for MoE prefill** — gap to vLLM single-seq has substantially closed. Qwen3-Coder-30B-A3B-NVFP4 (cold-median 5×5 trials, 15 s cooldown, 2026-05-23):
  - **pp512 = 17,521 tok/s** (σ wide: 16k-19k spread across trials — cuBLAS-algo-state still drifts at this kernel size despite cold-median methodology)
  - **pp2048 = 18,573 tok/s** (σ tight: 18.3k-18.9k = 3 % spread — exceeds vLLM 0.20.2's pp512 single-seq number)
  - **tg128 @ ctx=512 = 273.24 tok/s** (σ 0.09 — rock solid)
  - **tg128 @ ctx=2048 = 268.46 tok/s** (σ 0.10)

  Compared to vLLM 0.20.2 single-seq pp512 = 18,500 tok/s: **gap = 1.056× (5.6 %)**, was 1.14-1.32× pre-#374. The MoE-prefill skip-gather (#374) + downstream improvements moved us from 14,562 → 17,521 pp512 (+20.3 %). pp2048 actually exceeds vLLM's single-seq pp512 baseline. vLLM's 25.5 k multi-seq is continuous-batching mode, not a fair single-seq comparison — imp is batch=1 by design. **Remaining 5.6 % gap is within cuBLAS-algo-variance band on pp512.** Engineering priority drops here — the MoEProblemShape upstream / custom kernel work path documented in `docs/plans/moe_prefill_cudagraph_via_cutlass_moe_scheduler_2026_05_23.md` is no longer a release blocker.

### Memory
- **Paged KV cache** (block 16) with LRU, prefix caching — keep and extend.
- **FP8 / INT8 / NVFP4 KV cache** — NVFP4 decode cache is already shipping; expand to prefill where quality allows.
- **TurboQuant integration** — KV-cache compression as a first-class feature once stable. The 5090's 32 GB is the bottleneck; squeezing KV is how we serve bigger models than the competition.

### Latency
- **CUDA Graphs everywhere on decode** — already done, never regress.
- **PDL (Programmatic Dependent Launch)** — keep aggressive.
- **TurboDraft (L2-resident speculative decoder)** — ship it. PPM-based, tree drafting. This is the multiplier that puts imp uncatchable on decode for the hero models.
- **No host syncs on the decode hot path.** Ever. CI gates this.

### Surface
- **OpenAI-compatible HTTP server** stays first-class. Tool calling, SSE streaming, logprobs, /tokenize — already done. Maintain compliance with the test suite as OpenAI evolves the spec.
- **C library API** stable enough for embedding.
- **CLI** for benchmarking and interactive use.

---

## What imp is NOT

Defining this is as important as defining the mission. These are explicit non-goals — saying yes to them dilutes the mission:

- **Not a multi-GPU engine.** Tensor/pipeline parallelism is out of scope. Single GPU, period.
- **Not a high-batch serving engine.** vLLM and SGLang own that space. We don't compete on batch=64 throughput.
- **Not a training framework.** Inference only.
- **Not a mobile / embedded engine.** Workstation-class GPUs only.
- **Not a CPU engine.** GPU only. No AVX kernels, no Metal, no Vulkan.
- **Not a model zoo.** Architectures land when they justify the maintenance cost. The hero list above is curated, not exhaustive.
- **Not a research playground.** Every experimental kernel either lands as the default or is removed.

---

## Benchmarking discipline

- **llama-bench methodology** is the canonical comparison. pp512 + tg128 at minimum, plus realistic long-context (pp8192, tg512 @ 16k ctx).
- **Comparisons run on the same machine**, same driver, same CUDA version, same GGUF/SafeTensors weights where possible.
- **Every commit that touches a hot path** must report bench delta vs main on at least one hero model. Regressions need explicit justification.
- **Reproducibility:** bench scripts checked in, weight checksums recorded, results committed to a tracked file.

---

## Release bar

A release is shippable when, on RTX 5090:

1. All hero models pass correctness tests (perplexity within 0.5% of llama.cpp reference at the same quant).
2. Decode tok/s leads llama.cpp by ≥5% on every hero model.
3. Prefill tok/s ≥ llama.cpp on every dense hero model.
4. Prefill tok/s ≥ 50% of vLLM on every MoE hero model (interim — closes to ≥90% by year end).
5. No host syncs on the decode hot path (CI-enforced).
6. OpenAI API compliance suite green.
7. README benchmarks updated, commit hash of competitors recorded.

---

## North-star single number

**Qwen3-14B Q6_K decode tok/s at batch=1, ctx=2048, on RTX 5090.**

This number goes up over time. It never goes down. If a refactor makes it go down, the refactor is wrong, no matter how clean the code looks.

Measurement methodology (always cold-median, never single-shot): 5 independent `imp-cli --bench` invocations × 5 reps × 15 s cooldown between trials, take the median. Resists cuBLAS-algo-state drift over long sessions (`memory/bench_sustained_load_cublas_algo_drift_2026_05_23.md`).

Current: **157.71 tok/s default flags** (May 23 2026, cold-median methodology — 5 trials × 5 reps × 15 s cooldown, σ = 0.16 tok/s across samples; same code as the May 22 measurement, just less cuBLAS-algo-cache-state noise). The 150 milestone is hit by default with margin.
Previous: 150.1 tok/s (May 22 2026, single-shot, post PRs #362 + #364 + #367). 121.4 tok/s (May 2026, +25.5% vs llama.cpp c830f99).
Next milestone: **175 tok/s** — needs architectural work (decode kernel tuning per the 35 % attention / 60 % FFN profile split, FP8 prefill cache coverage for the remaining ~100 layer-tensors that fall back to CUTLASS NVFP4 prefill, or LM_HEAD quantization unlock).
Stretch: 200 tok/s with TurboDraft or draft-model spec-decode on (both blocked: TurboDraft broken on pretrained, no MTP head shipped for Qwen3-14B, draft-model integration is multi-week).
