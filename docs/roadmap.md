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

## Open gaps to the mission (assessed 2026-07-26)

A ranked audit of what still separates imp from "best agentic engine on a 5090". This is a gap list, not a schedule -- nothing here is committed. The raw-speed half of [`GOAL.md`](GOAL.md) is met (batch=1 decode leads llama.cpp by +13-48% on every hero in the 2026-07-12 re-sweep, MoE prefill leads vLLM single-seq, cross-engine PPL parity measured). Every item below sits on the *agentic* half of the mission.

1. **First-party NVFP4 quantizer — mechanism shipped, calibration open.** `imp-quantize` (2026-07-26) converts a dense BF16/FP16 SafeTensors checkpoint to NVFP4 in-tree: the result loads, is detected as a Modelopt-style NVFP4 model and generates coherently, so a model with no published export can reach the NVFP4 path at all. **What it does not yet solve is the quality half.** Scales are uncalibrated round-to-nearest, and that costs PPL +25% (Qwen3-0.6B) / +19% (Qwen3-1.7B) against BF16 on a 13.5k-token corpus, with `degen_suite.py` passing 41/41 on the quantized model. (Measure that on `ppl_corpus_45k.txt`: the 199-token `ppl_corpus.txt` reads +42%/+57% for the same pair and inverts the size trend.) So symptom (a) of the original gap is addressed (no more waiting for someone to publish an export) while (b) is not: checkpoint quality is now *our* problem instead of a third party's. Open work, in order: AWQ/SmoothQuant-class calibration, then MoE expert stacks (3-D, needs the per-expert path), then a head-to-head against a Modelopt export of the same model — which needs a dense model staged in both BF16 and Modelopt-NVFP4 locally, currently not the case. Until calibration lands, a published Modelopt checkpoint still beats what this produces.

2. **Vision stops at Gemma.** `src/vision/` is SigLIP/CLIP for `gemma3` + `gemma4v` and nothing else -- no Qwen3-VL, InternVL or Pixtral. Screenshot / computer-use agents are exactly the workload the mission names, and there is currently no model for them on this engine. Qwen3-VL needs dynamic resolution + M-RoPE in the encoder, so this is encoder work, not weight mapping.

3. ~~**One server, one model.**~~ **Closed** — `server.model_swap` (default on) serves a model other than the loaded one by swapping to it: in-flight generations drain first and are never cancelled (the `/admin/suspend` contract), and a failed load restores the previous model rather than leaving the server empty. Those two were exactly why the first-generation auto-swap had been removed. `/v1/models` now lists the rest of the models directory alongside the loaded one, so a harness can see what it may ask for. Still serial by nature: 32 GB fits one model at a time and the requesting call pays one load (the warm weight cache, #956, makes repeats cheap).

4. **Constrained decoding is JSON-only.** The FSMs cover `json_schema`, `json_object` and the XML tool dialect; there is no GBNF / regex / EBNF grammar surface (vLLM and SGLang ship xgrammar, llama.cpp ships GBNF). Agents that need to pin a diff format, SQL or a tool DSL fall back to prompting. `/v1/rerank` is likewise absent while `/v1/embeddings` ships -- RAG agents use both.

5. **Speculation has no tree and no trained draft head.** No EAGLE / Medusa / multi-candidate path exists in the tree. Prompt-lookup n-gram only drafts spans that already appear in the context, so it contributes nothing on free-form reasoning output, and the verify-in-loop experiment was removed after a nine-class sweep found no prompt class where it won (see `CHANGELOG.md`, Unreleased). Per the 2026-07-22 ceiling review this is the only durable batch=1 performance lever left, and the route to the 175 tok/s north-star milestone.

6. **Context is VRAM-capped with no host spill.** No KV offload to host RAM and no general layer offload (only the MoE expert cache). The auto ceiling is 128K since #1004, but Q6_K on 32 GB tops out near 75K in practice ([`BENCHMARKS.md`](BENCHMARKS.md)); past that StreamingLLM evicts, which is silent context loss rather than a longer window.

7. **Agentic quality is unmeasured against competitors.** `tools/analysis/degen_suite.py`, `tools/agent_bench.py` and the NIAH harness exist, but no cross-engine number is published for tool-call accuracy or format compliance over a long session. Release bar 7 declares the agentic surface green against our own batteries only. "42% faster" is provable today; "breaks tool calls less often" is not.

Explicitly **not** gaps: continuous batching, prefix caching, per-request LoRA, embeddings, the OpenAI / Anthropic / Responses APIs, `/metrics`, suspend/resume, and the sampler surface (DRY, mirostat, typical_p, logit_bias) all ship today. Multi-GPU remains a non-goal.

### Built-in live UI -- shipped

`imp-server` serves a single-page UI at `GET /` (assessed feasible 2026-07-26, shipped the same day). It renders the SSE stream live and draws one bar per token, so inter-token latency is visible while the answer is written. The page is embedded into the binary at build time (`cmake/embed_webui.cmake`), so there is no asset path to locate at runtime. Source: `tools/imp-server/webui/index.html`.

The assessment that preceded it, kept because it is the reason this cost almost nothing -- no engine or protocol work was required:

- Streaming is the real OpenAI wire format -- `text/event-stream` via `set_chunked_content_provider`, `data: {...}\n\n` chunks, terminating `data: [DONE]` (`handlers_chat_stream.cpp`).
- CORS is wide open for any origin, preflight included -- `Access-Control-Allow-Origin: *` plus an `OPTIONS` catch-all (`main.cpp`), so a page served from anywhere can call the API directly, without a proxy.
- Client disconnect is detected on the token loop (`is_writable`), so closing the tab cancels generation instead of leaving the GPU running.
- `reasoning_content` and tool calls already arrive as separate stream channels, so a collapsible thinking pane and streamed tool calls need no server change.

The only client-side constraint: `EventSource` is GET-only, so the page consumes the stream with `fetch()` + `ReadableStream` -- the standard approach.

Note this is **not** on the `GOAL.md` surface commitment (HTTP server, C API, CLI): it is a convenience with a maintenance tail, not a mission item. It stays deliberately small -- one file, no build step, no dependencies -- and it is not a reason to grow a frontend stack. Anything beyond a thin client belongs in Open WebUI or another external front end.

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
