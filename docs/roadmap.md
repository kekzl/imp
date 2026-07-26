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

**Status as of 2026-07-26** (one line each; detail in the entries below):

| # | Gap | State |
|---|---|---|
| 1 | First-party NVFP4 quantizer | **partial** — `imp-quantize` converts and the output runs, but it is **experimental**: uncalibrated, +19-25% PPL. Calibration open. |
| 2 | Vision beyond Gemma | open |
| 3 | One server, one model | **closed** — `server.model_swap` (#1080) |
| 4 | Constrained decoding is JSON-only | **closed for regex** — `response_format: regex` / `guided_regex` ship; GBNF/EBNF and `/v1/rerank` remain separate items |
| 5 | No speculation tree / trained draft head | open — still the only durable batch=1 perf lever |
| 6 | Context VRAM-capped, no host spill | open |
| 7 | Agentic quality unmeasured vs competitors | **closed** — measured across three model families, published in [`BENCHMARKS.md`](BENCHMARKS.md); vLLM/SGLang deliberately out of scope |
| 8 | No GBNF/EBNF grammar surface | open — split out of 4 when regex shipped; regex covers the common agent formats, a full grammar does not exist |
| 9 | `/v1/rerank` absent | open — split out of 4: it needs cross-encoder architecture support (model work), not a grammar surface |
| 10 | Agent-harness batteries are imp-internal | open — `agent_loop_suite.py` probes our own server (#1007 stage 1); stage 2 is running real harness binaries (Claude Code, Aider, OpenHands) against imp |

Shipped alongside, not from this list: the live web UI at `GET /` (#1078) and the streamed non-ASCII corruption fix that building it exposed.

1. **First-party NVFP4 quantizer — EXPERIMENTAL, calibration open.** `imp-quantize` (2026-07-26) converts a dense BF16/FP16 SafeTensors checkpoint to NVFP4 in-tree: the result loads, is detected as a Modelopt-style NVFP4 model and generates coherently, so a model with no published export can reach the NVFP4 path at all. **It is explicitly experimental — not a production quantizer.** Use it to get a model onto the NVFP4 path for evaluation or performance work; do not ship its output as a quality checkpoint. **What it does not yet solve is the quality half.** Scales are uncalibrated round-to-nearest, and that costs PPL +25% (Qwen3-0.6B) / +19% (Qwen3-1.7B) against BF16 on a 13.5k-token corpus, with `degen_suite.py` passing 41/41 on the quantized model. (Measure that on `ppl_corpus_45k.txt`: the 199-token `ppl_corpus.txt` reads +42%/+57% for the same pair and inverts the size trend.) So symptom (a) of the original gap is addressed (no more waiting for someone to publish an export) while (b) is not: checkpoint quality is now *our* problem instead of a third party's. **Refuted on the way (2026-07-26):** picking micro-scales by minimizing reconstruction error instead of `absmax` moved PPL 30.10 -> 29.88 (0.7%) for ~6x the quantization cost — reverted. The micro-block is 16 values, where `absmax` is already near-optimal; the dominant error is the FP4 grid itself, which no scale improves. The lever therefore is not better scales but *moving* the error: AWQ (protect channels with the largest calibration activations) or GPTQ (compensate each column's error in the columns still to be quantized). Open work, in order: AWQ/SmoothQuant-class calibration, then MoE expert stacks (3-D, needs the per-expert path), then a head-to-head against a Modelopt export of the same model — which needs a dense model staged in both BF16 and Modelopt-NVFP4 locally, currently not the case. Until calibration lands, a published Modelopt checkpoint still beats what this produces.

2. **Vision stops at Gemma.** `src/vision/` is SigLIP/CLIP for `gemma3` + `gemma4v` and nothing else -- no Qwen3-VL, InternVL or Pixtral. Screenshot / computer-use agents are exactly the workload the mission names, and there is currently no model for them on this engine. Qwen3-VL needs dynamic resolution + M-RoPE in the encoder, so this is encoder work, not weight mapping.

3. ~~**One server, one model.**~~ **Closed** — `server.model_swap` (default on) serves a model other than the loaded one by swapping to it: in-flight generations drain first and are never cancelled (the `/admin/suspend` contract), and a failed load restores the previous model rather than leaving the server empty. Those two were exactly why the first-generation auto-swap had been removed. `/v1/models` now lists the rest of the models directory alongside the loaded one, so a harness can see what it may ask for. Still serial by nature: 32 GB fits one model at a time and the requesting call pays one load (the warm weight cache, #956, makes repeats cheap).

4. **Regex-constrained decoding — shipped; GBNF and rerank remain.** `response_format: {"type":"regex"}` (and vLLM's `guided_regex`) constrain the whole reply to a pattern, so a diff header, an ID format, an enum or a small DSL is enforceable without prompting and hoping. Built on the `RegexNfa` already in the tree for JSON-Schema `pattern` — a second engine was written and discarded after measuring identical behaviour. What this needed was the decode-time wrapper: `RegexConstrainer` with the JSON constrainers' `apply_mask` contract, a per-state-set mask cache, EOS gated on an accepting state, and — the part that actually took the time — closing every path that bypasses the mask (the spec-ngram and graph-loop routers, two further `apply_mask` call sites, thinking-default suppression, and pooled-manager state that leaked between requests). Still open, tracked as items 8 and 9 above: a full GBNF/EBNF surface, and `/v1/rerank`.

5. **Speculation has no tree and no trained draft head.** No EAGLE / Medusa / multi-candidate path exists in the tree. Prompt-lookup n-gram only drafts spans that already appear in the context, so it contributes nothing on free-form reasoning output, and the verify-in-loop experiment was removed after a nine-class sweep found no prompt class where it won (see `CHANGELOG.md`, Unreleased). Per the 2026-07-22 ceiling review this is the only durable batch=1 performance lever left, and the route to the 175 tok/s north-star milestone.

6. **Context is VRAM-capped with no host spill.** No KV offload to host RAM and no general layer offload (only the MoE expert cache). The auto ceiling is 128K since #1004, but Q6_K on 32 GB tops out near 75K in practice ([`BENCHMARKS.md`](BENCHMARKS.md)); past that StreamingLLM evicts, which is silent context loss rather than a longer window.

7. ~~**Agentic quality is unmeasured against competitors.**~~ **Closed.** `tools/analysis/agentic_compare.py` measures the checks an agent harness depends on against any OpenAI-compatible server, and the results are published in [`BENCHMARKS.md`](BENCHMARKS.md): three model families, four budgets, 8-turn sessions. The headline is a defaults difference, not a capability one — at a 200-token budget imp keeps every contract while llama.cpp needs ~800 because it lets a think-capable model reason first; on a non-thinking model llama.cpp's `json_object` and `tool_choice=auto` have gaps of their own. It also earned its keep immediately by finding a REAL imp bug (Llama-3.2 bare-JSON tool calls were dropped, fixed in #1088) that our own batteries never saw, because they run Qwen. **Not covered: vLLM/SGLang** — different weight format and more VRAM than is free while serving; a deliberate scope cut, not an oversight. Extending to them, or to more families, is now a matter of running the harness, not building one.

8. **No GBNF/EBNF grammar surface.** Split out of item 4 when regex shipped. A regex covers the formats agents actually pin — IDs, enums, dates, version strings, diff headers — but not a recursive grammar (a nested expression language, a full DSL). vLLM/SGLang expose one via xgrammar, llama.cpp via GBNF. The FSM infrastructure and the `apply_mask` contract now exist for three grammars, so the work is a grammar compiler, not plumbing. Note the bypass checklist in the `server-api` skill before starting: a constrainer is only as good as the paths that cannot route around it.

9. **`/v1/rerank` is absent** while `/v1/embeddings` ships — RAG agents use both. Split out of item 4 because it is NOT a grammar surface: a real reranker is a cross-encoder (query+document scored jointly), which needs sequence-classification architecture support in the loader and forward pass. Serving it from bi-encoder embeddings would be an endpoint with the right name and the wrong quality.

10. **The agent-harness batteries only probe our own server.** `tools/analysis/agent_loop_suite.py` (#1007 stage 1) covers the Anthropic tool loop, the OpenAI loop and `/v1/responses` — but by construction it asserts what imp *thinks* correct looks like. Stage 2 is pointing real harness binaries (Claude Code, Aider, OpenHands) at imp and seeing whether a full agent session completes. That catches integration assumptions a self-written probe cannot, in the same way the cross-engine comparison (item 7) caught a tool-call bug our own batteries never saw.

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
