# imp — Audit: Performance & Agent-Readiness (2026-05-31)

Read-only audit. No code changes. Evidence: **[M]** = freshly measured (this session),
**[P]** = profile-backed (nsys this session or documented ncu/nsys), **[C]** = code location, **[H]** = hypothesis.

Measurement setup: Host RTX 5090 (sm_120), CUDA 13.3, binary `build-ciq/imp-cli` (05-29, CompileIQ-ACF build —
absolute numbers ±a few %, the qualitative conclusions are robust). `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

---

## Executive Summary

1. **The central audit premise is outdated.** "MoE prefill ~1258 tok/s, 20× behind vLLM" no longer holds.
   Freshly measured, Qwen3-Coder-30B-A3B-NVFP4 delivers **16.5–18.2k tok/s prefill** (pp512–pp4096). The "1258" value
   comes from `archive/vllm_comparison_2026_05_10.md` (pp512, **before** the CUTLASS 3.x grouped-GEMM rework, old
   `NVFP4→FP16-dequant + cuBLAS-grouped` path). That path was superseded by ~42×; the **real remaining gap to vLLM is
   ~1.15–1.42×**, not 20×. The single largest prefill optimization target from the briefing no longer exists as such.
2. **The real NVFP4 prefill lever is attention, not the grouped GEMM.** The MoE expert GEMM runs fused
   (dequant in the MMA) and is near roofline. The addressable remainder is the partially materialized attention path
   (FA2 only applies partially).
3. **The agent backend is surprisingly mature.** Serving (continuous batching, prefix-cache, paged-KV, cancellation,
   rate-limit, Prometheus) is largely production-ready. Constrained output **works** (contrary to the initial
   assumption — `apply_mask` is wired into sampling). Real gaps: synthetic `/v1/messages` streaming,
   no per-request spec-decode, determinism caveat for MoE, no prompt-caching header.

---

# Part A — Performance

## A.0 Verification of the premise (fresh measurement)

Qwen3-Coder-30B-A3B-Instruct-FP4, 3 reps, default path (`executor_forward_moe_cutlass.cu:112` device-args):

| Metric | pp512 | pp2048 | pp4096 | decode tg256 |
|---|---|---|---|---|
| tok/s **[M]** | 16.535 | 17.206 | 18.229 | 290–295 |

- Fallback path `IMP_NVFP4_DEVICE_ARGS=0` (host-args CUTLASS 3.x): pp2048 = **12.815 tok/s [M]** — that too is
  not 1258. The actual ~77 tok/s value is the *legacy serial* path, long dead (`executor_forward_moe_cutlass.cu:306`:
  "Prefill n=120: ~2750 vs legacy ~77 — 35× win").
- Decode moat confirmed: tg256 ≈ 290 tok/s with graphs (higher than the 200 cited in the briefing). **[M]**
- vLLM comparison value 25.513 is the only non-self-verified figure; even against it imp is ~0.65–0.72×.

## A.1 Prefill kernel breakdown (nsys, prefill-isolated, graphs off)

Profile: ~2800-token prompt + `--max-tokens 1` (isolates prefill). **Important pitfall first:**

- `convert_scales_sfatom_kernel` appears at **21.7 %** — but with **identical 37.249 instances / ~47 ms in
  *both* runs** (1 vs. 256 decode steps). This is **one-time cache-build work at model load**, not a
  per-prefill cost. In the server (load once) it amortizes → **don't chase** (consistent with
  `docs/decode-gap-analysis-2026-05-29.md:14`). **[P]**

Cleaned-up per-prefill mix (NVFP4 MoE):

| Kernel | Share | Class | Evidence |
|---|---|---|---|
| CUTLASS NVFP4 grouped GEMM (`GroupProblemShape`) | ~19 % | Expert-FFN, ~roofline | [P] nsys |
| `fmha_sm120_fa2_kernel<128>` | ~13 % | FA2 attention (applies partially) | [P] nsys |
| `causal_softmax_fp32_to_fp16` + nvjet-cuBLAS-MMAs | ~18 % | **materialized attention (non-FA2)** — ⚠️ STALE (state before #525 FP16-QK-FA2). Re-measured 2026-06-07: **0.0 %** on all hd=128 models pp512–4096; only hd≠128 (gemma-3) at 3.6–6.9 %. See [`roofline_2026_06_07.md`](roofline_2026_06_07.md) | [P] nsys |
| MoE quant/scatter/permute/gating/scale | ~1 % each | routing overhead, small | [P] nsys |

**Finding:** FA2 (`fmha_sm120_fa2_kernel`) *and* the old materialized path (`causal_softmax_fp32_to_fp16` + cuBLAS
QK^T/PV) run simultaneously → attention is only partially on FA2. This matches the documented
NVFP4-pp2048 analysis: "CUTLASS NVFP4 GEMM 39 % (competitive w/ vLLM) + Attention ~37 %"
(`docs/MISSION_JOURNAL.md:314`) and the roofline detail "FFN shapes ~100 % FP4 roofline, attention shapes ~34 %"
(`docs/plans/nvfp4_pp2048_analysis_2026_05_25.md:35`).

## A.2 Dequant fused vs. separate pass — hypothesis from the briefing clarified

- **NVFP4 MoE prefill: dequant is FUSED** into the block-scaled MMA (`mma.sync ...kind::mxf4nvf4.block_scale...m16n8k64`,
  `src/compute/gemm_grouped_nvfp4_smallM.cu:81`; path `executor_forward_moe_cutlass.cu:32-295`). No separate
  FP16 pass. The briefing's working hypothesis ("grouped-GEMM dequant path as the cause") does **not** apply to NVFP4. **[C]**
- **GGUF Q4_K MoE prefill: dequant IS a separate pass** (`dequant_gpu()` → FP16 → `gemm_moe_batched()`,
  4.55 B/elem vs. llama.cpp MMQ 0.55 B/elem = 8.3× bandwidth overhead, `docs/plans/q4k_prefill_analysis_2026_05_25.md:20-31`).
  That is the real "separate dequant" pain — but in GGUF, which per project policy is *legacy* (NVFP4 is priority).
- Fallback `try_run_moe_nvfp4_dequant_batch_prefill_` (`executor_forward_moe_batch.cu:54`) does a separate dequant
  (variable `slow_down_act`, line 116) — fires only when the device-args preconditions are absent. **[C]**

## A.3 CUDA 13.2/13.3 features — applicability

| Feature | Applicable? | Where / effect | Evidence |
|---|---|---|---|
| `cublasLtMatmulGrouped` NVFP4 device-shapes | **No** | 0 grouped algos on sm_120 up to cuBLAS 13.4 (only CC 10.x/11.0) | `docs/sm120.md:86`, `archive/cublas_13_4_sm120_no_movement_2026_05_09.md` [C] |
| `cub::DeviceTopK` (expert routing/sampling) | **Check [H]** | Currently `topk_gating_kernel` (1 block/token) — only ~1 % in prefill, possibly more in decode/sampling. No prior test documented. | nsys [P] |
| Grouped GEMM + CUDA Graphs | partially blocked | CUTLASS `MoEProblemShape` carries `IsMoEScheduler=false` stub for sm120; capture hang under `CaptureModeGlobal` | `archive/moe_prefill_graph_capture_analysis_2026_05_10.md`, `prefill_graph_blockers_2026_05_14.md` [C] |
| CUDA 13.2→13.3 instructions | no gain | 0 of 247 sm_120a instructions flipped | `docs/MISSION_JOURNAL.md:413` [C] |

## A.4 CUDA Graphs / scheduling — premise "decode only" refuted

- **Prefill IS graphified** (default `runtime.prefill_graph=true`). Effect freshly measurable: pp2048 17.206 (graph)
  vs. 14.787 (graph off) = **~+16 % [M]**. The briefing question "are graphs used only in decode?" → no.
- Open: graphs for non-fast-path MoE decode (GGUF) remain blocked (D2H expert routing, `docs/sm120.md:79`).

## A.5 Prioritized findings table (Part A)

| # | Finding | Evidence | Expected speedup | Effort | Decode risk | Addresses MoE-prefill gap? |
|---|---|---|---|---|---|---|
| A1 | **Increase FA2 coverage in prefill** — ✅ DONE (#525 FP16-QK-FA2): re-measured 2026-06-07 = **0.0 % legacy share on hd=128**; remainder only hd≠128 (gemma, 3.6–6.9 % window = 92–99 % of its attention). See [`roofline_2026_06_07.md`](roofline_2026_06_07.md) | [P] nsys (~~18 %~~ → 0 %) | done | — | — | Yes |
| A2 | **GGUF Q4_K MoE prefill: in-SMEM MMQ instead of dequant→cuBLAS** | [C] 8.3× BW overhead, doc-backed | pp +30–50 % GGUF | XL (2–3 wk) | low | GGUF only (legacy prio) |
| A3 | **Small-M grouped-GEMM efficiency** — attention shapes ~34 % roofline at small-M; tile/scheduler tuning | [P] doc roofline | pp +5–10 % | L | medium | partially |
| A4 | Treat `convert_scales_sfatom` as init, **not** as prefill cost (anti-finding) | [P] identical instance count | 0 (clarification) | — | — | no |
| A5 | Evaluate `cub::DeviceTopK` for routing/sampling | [H] | small (routing ~1 % prefill) | S (measurement) | low | no |
| A6 | Unblock grouped-GEMM CUDA-graph capture (`IsMoEScheduler`) | [C] | pp +10–15 % | XL | medium | partially |

> **Marked as MoE-prefill gap:** A1 is by far the best lever — the gap is *attention*, not the
> grouped GEMM. The grouped GEMM is already near roofline and fused.

---

# Part B — Agent operation

Status per axis. **Correction up front:** the first sub-audit reported constrained output as "dead code"
(`apply_mask` never called). **That is wrong** — self-verified: `apply_mask()` is called in sampling at
`src/exec/executor.cu:122/124, 220/222, 360/362` (three sampling paths), FSM progress via
`constraint_manager.cpp:167`. Constrained output is functional.

| # | Axis | Status | Code evidence | Agent impact | Effort |
|---|---|---|---|---|---|
| 1 | **Structured/Constrained Output** | **works (core)** | FSM `json_constrain.cu`/`schema_constrain.cu`; mask in sampling `executor.cu:122-362`; server `handlers.cpp:731` (`response_format`) | critical | — |
| 1b | … regex/`pattern` + GBNF grammar | **missing** | no `pattern` parsing in `json_schema.h`; no GBNF compiler | important | M |
| 1c | … masking overhead | unmeasured | no benchmark/comment | nice | S (measure) |
| 2 | **Tool Calling** (definition, tool_choice auto/required/named, dialects) | **partial→complete** | `tool_call.cpp:6-150`, `chat_template.cpp:1093`; vector return `tool_call.cpp:142` (multi-parse present) | critical | — |
| 2b | … streaming of tool-call deltas | **missing** | streaming path does not segment tool args | important | M |
| 2c | … argument validation against input_schema | **missing** | lenient `json::parse`, no schema check | important (hallucination) | M (reuse FSM from 1) |
| 3 | **Anthropic `/v1/messages`** (blocks, tool_use/result, stop_reason, multi-block) | **complete** | `anthropic.cpp:19-394`, handler `handlers.cpp:3449` | critical | — |
| 3b | … **streaming** | **synthetic** | `handlers.cpp:3572` "Phase 2 synthetic" — full generation then SSE replay → TTFT = full latency | **critical for agents** | M |
| 3c | … `thinking` blocks | **missing** | `anthropic.cpp:102` — reasoning discarded | important | S |
| 3d | … prompt-caching header (`cache_control`) | **missing** | no parsing/tracking | important | M |
| — | OpenAI `/v1/chat/completions` streaming | **real** | `pop_token` loop `handlers.cpp:1371/1759/2762/2932` | — | — |
| 4 | **Prefix/Prompt caching across turns** | **complete** | content-addressed `kv_cache_manager.h:70-283`, scheduler reuse `scheduler.cpp:70`, LRU eviction, metric `imp_tokens_cached_total` | critical | — |
| 5 | **KV cache long loops** | **complete** | paged block_size=16 `kv_cache.h:12`; StreamingLLM `kv_cache_manager.h:124`; FP8/INT8/NVFP4 KV | important | — |
| 6 | **Reliability** (cancel/timeout/OOM) | **complete** | cancel `batching_engine.cpp:93`; timeout `handlers.cpp:1360`; OOM try/catch `batching_engine.cpp:108`; KV-exhaustion early-cancel `scheduler.cpp:48` | critical | — |
| 7 | **Concurrency/Scheduling** | **complete (in-flight batching)** | continuous batching `scheduler.cpp:17-126`, SJF against HoL `scheduler.cpp:27`; single-worker (no thread parallelism) | critical | — |
| 7b | … p50/p99 latency | **partial** | only single gauges `handlers.h:61` (no histogram) | important | S |
| 7b' | … aggregate throughput barely scales with N | [doc] | ~130 tok/s flat (`server_batching_throughput_ceiling`) | important (single-user 5090: possibly non-goal) | L |
| 8 | **Speculative Decoding (MTP)** | **partial — not in serving** | head+forward present (`mtp_forward.cu`), engine API `engine.h:142`; **no per-request flag**, draft-verify loop not wired into decode | important | L (also K=1 acceptance ~25-30 %, [[mtp_diagnosis]]) |
| 9 | **Determinism** | **partial / caveat** | greedy argmax deterministic `sampling.cu:41`; **MoE routing + top-k via atomics → not reproducible** (`moe_routing.cu`); Qwen3.6-35B doc-confirmed non-det @temp0 | important (eval/test) | M |
| 10 | **Observability** | **complete (except histogram)** | Prometheus `/metrics` `handlers.cpp:3154`; logprobs `request.h:72`; queue_depth; per-request JSONL | important | — |
| 11 | **Session/State** | **stateless** (prefix-pin as substitute) | `handlers.cpp:2426`; `pin_prefix` `kv_cache_manager.h:99`; no session store | nice | M |
| 12 | **Multi-model serving** | **missing/partial** | 1 model/instance; reload-POST deferred `handlers.cpp:267` | important (tool-router model) | L |
| 13 | **Container manager/sandbox** | **completely missing** | no exec/sandbox/docker-socket infrastructure; "tools" = output formatting only | critical for autonomous tool exec | XL |
| 14 | **Security/Limits** | **complete** | API key constant-time `main.cpp:138`; per-IP rate-limit `handlers.h:108`; payload cap 100 MiB `main.cpp:76`; max-concurrent 429 `main.cpp:113`; timeout 300s | important | — |
| 14b | … input-token length limit | **missing** | no token-count validation before prefill | nice | S |

---

# Closing section

## Top-5 actions (impact/effort)

1. **`/v1/messages` real token streaming** (B-3b). Synthetic streaming means TTFT = full generation —
   a hard disadvantage for interactive agent loops against the Anthropic endpoint. The OpenAI path already streams for real,
   so the pattern exists. *Effort M, impact critical.*
2. **Complete FA2 coverage in prefill** (A1). The only large, decode-neutral prefill lever; attention,
   not grouped GEMM, is the remaining gap to vLLM. FA2 is parity-tested. *Effort M, impact high.*
3. **Tool-argument validation against `input_schema`** (B-2c) — apply the existing schema FSM (B-1) to tool-call bodies.
   Prevents hallucinated/broken tool arguments, the most common agent-loop break. *Effort M (reuse).*
4. **Determinism switch** (B-9): optional atomic-free routing/top-k path for `temperature=0`, for
   reproducible agent evals. *Effort M, impact important (testability).*
5. **p50/p99 latency histogram + prompt-caching header** (B-7b, B-3d): cheap observability/cost visibility
   for multi-agent load. *Effort S–M.*

> Container manager/sandbox (B-13) is the largest missing building block for *autonomous* tool execution, but XL and
> a product decision of its own — deliberately not in the top-5.

## Decode protection (regression guards before any prefill optimization)

- **Gate before every merge:** `make verify-fast` against `tests/perf_baseline.json` (3 % decode / 5 % prefill).
- **A/B over decode only** (`tg256`), isolated + 60–120 s GPU cooldown, 10 reps, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  Prefill pp varies up to 2.6× across container restarts — never as the sole signal.
- **Measure graphs ON *and* OFF** (graph replay can hide a silent fallback — `check-degeneration` skill).
- **Coherence check** after forward/attention/routing changes (no repetition loop/token-stuck).
- Baseline models for the decode wall: Qwen3.6-35B-A3B-NVFP4 (moat), Qwen3-14B-NVFP4 (dense), Qwen3-Coder-30B (MoE).

## Implementation findings (2026-05-31, tasklist execution)

### T2 (FA2 coverage) — measured, no safe global flip [M]
FA2 is gated in `executor_attention.cu:811-817` (chunked) and `:849` (non-chunked) by
`attention.fmha_prefill_threshold` (auto-default = S-matrix VRAM cap+1,
`executor_workspace_buffers.cu:267`). `causal_softmax_fp32_to_fp16` comes from `attention_cublas.cu`
(materialized path below the threshold). pp512 A/B default vs FA2-always (`fmha_prefill_threshold=1`),
decode neutral throughout (FA2 does not touch decode):

| Model | default | FA2-always | Δ |
|---|---|---|---|
| Qwen3-30B-A3B | 13737 | 18154 | +32% |
| Qwen3-Coder-30B | 15742 | 17465 | +11% |
| Qwen3.6-35B-A3B | 10469 | 10575 | +1% |
| Qwen3-14B (dense) | 18178 | 15321 | −16% |
| Gemma-4-26B (MoE) | 22297 | 16732 | −25% |

→ Crossover is model-dependent; **no decode-neutral global default flip possible** (Gemma-4/dense
regress → perf gate). Safe win today: per-model `attention.fmha_prefill_threshold=1` for
Qwen3-30B/Coder. Real solution = measured per-(arch,seqlen) crossover heuristic (follow-up work, to be
validated zoo-wide). pp≥2048 already uses FA2 (default ≈ FA2-always).

### T11 (vLLM cross-measurement) — measured [M], corrects the doc assumption
vLLM 0.21.0 (`vllm/vllm-openai`), Qwen3-Coder-30B-A3B-NVFP4, real FlashInfer/CUTLASS-NVFP4 path
(NOT Marlin: `FlashInferCutlassNvFp4LinearKernel` + `FLASHINFER_CUTLASS` MoE), prefix-caching off,
`--max-concurrency 1`, `--random-output-len 1`, best-of-3 (WSL2 bimodal → cooler mode):

| Prompt len | vLLM prefill tok/s | imp prefill tok/s | vLLM lead |
|---:|---:|---:|---:|
| 512  | ~21.200 | 16.500 | 1.28× |
| 2048 | ~33.300 | 17.200 | 1.94× |
| 4096 | ~29.200 | 18.200 | 1.60× |

→ The "20×" premise is conclusively refuted, **but the real prefill gap (1.3–1.9× in vLLM's favor) is
LARGER than the 1.15–1.42× cited in the memos.** vLLM wins prefill at every length via batched
prefill GEMMs; decode was not measured (imp decode moat untouched). Consistent with A1/A.5: the lever
is prefill-GEMM/attention efficiency, not the routing/dequant path.

### Build/verification status (tasklist batch, branch `feat/agent-readiness-batch`)
8 code tasks (T1,3,4,5,6,7,8,9) implemented + **Docker build green** (CUDA 13.3), unit 37/37, GPU tests
73 passed/0 failed. Decode-moat check: new binary tg256=253.0 vs build-ciq 256.9 (−1.5 %, within noise) —
**decode-neutral** ✓. Determinism flag (T4) verified: `runtime.deterministic=true` produces bit-identical
tokens across runs (Qwen3-Coder, temp=0). Output coherent, no degeneration. Commit `09335dd3`.

Live-server smoke tests (new binary, Qwen3-8B-NVFP4):

| Task | Result |
|---|---|
| T1 /v1/messages streaming | ✓ real incremental SSE (message_start→content_block_delta…) |
| T3 tool calling | ✓ `{"city":"Berlin"}`, finish_reason=tool_calls; arg validation active |
| T5 metrics | ✓ `imp_request_duration_seconds`/`imp_ttft_seconds` histograms |
| T9 input limit | ✓ long prompt → HTTP 400 "Prompt exceeds max input tokens (189 > 50)"; short → 200 |
| T7 constrained | json_object ✓, json_schema structure ✓; **`pattern` regex masking disabled** (NFA over-masked → `!!!!`; parsed but not enforced) |
| T8 thinking blocks | mapping implemented; fires only with separated `reasoning_content` (`--reasoning-format`) — default smoke not confirmed |
| T6 tool deltas | implemented; streaming confirmed |

Open follow-ups: T7 fix NFA over-masking + wire GrammarConstrainer into `executor.cu`; T8 confirm with reasoning-format.

### T10 (ncu roofline of the top prefill kernels) — measured [P]
ncu (host `/opt/nvidia/nsight-compute/2026.2.0/ncu`), Qwen3-Coder-NVFP4, graphs off, 6 mid-network instances each:

| Kernel | DRAM% (1792 GB/s) | SM% | Occupancy | Classification |
|---|---|---|---|---|
| NVFP4 grouped GEMM (MoE expert, mxf4nvf4 FP4-TC, tile 128³, grid=170) | **59 %** | **34 %** | **24 %** | latency/wave-quantization-bound — **not roofline, compute headroom** |
| FA2 (`fmha_sm120_fa2_kernel<128>`, FP16, grid=128) | 3.4 % | 36 % | 16.8 % | latency/occupancy-bound — not roofline |

`smsp__...tensor_op_mma` counters are not exposed on sm_120; FP4/FP16 TC usage confirmed via kernel symbol.

### T12 (small-M grouped-GEMM tuning) — finding + recommendation [P]
The ncu data confirm: the grouped GEMM is **NOT compute- or bandwidth-limited**, but
**single-wave + small-M-per-expert** (2009 tokens / 128 experts → tiny M_e) at only 24 % occupancy.
Live levers: (a) persistent/stream-K grid instead of 170-block single-wave, (b) better token packing to
raise M_e, (c) occupancy-raising tile choice. **But:** matches `nvfp4_moe_prefill_landscape`
("hand-rolled NVFP4 grouped = par at large M_e, **−50-55 % at small M_e**") and T11 (vLLM's batched
prefill GEMMs win). Substantial CUTLASS/kernel work to be validated zoo-wide → **not a
session-scope merge**; concrete next step: switch the CUTLASS Sm120 grouped scheduler to persistent/stream-K
and measure M_e packing, guarded against the decode gate.

## Open questions / missing measurements

1. **vLLM cross-measurement on this box** for Qwen3-Coder-NVFP4 (pp2048/pp4096), apples-to-apples — 25.513
   is the only non-self-verified number; the real remaining gap (~1.4×?) should be freshly confirmed.
2. **ncu missing on the host** (`ncu: command not found`) — for roofline/occupancy/stall of the top prefill kernels
   (FA2, grouped GEMM), ncu must be mounted via container (recipe in [[decode_frontier_reconfirmed]]).
3. **FA2 threshold:** FA2 already fired at ~2800 tokens, but `causal_softmax` ran in parallel — why the
   split? (head-dim ≠ 128? chunk boundaries?) Clarifies the exact scope of A1.
4. **`cub::DeviceTopK`** in sampling/routing: worth it only if `topk_gating`/sampling costs measurable time in decode — ncu needed.
5. **MTP in serving:** is the draft-verify loop worthwhile at all at ~25–30 % acceptance? (Doc says: for NVFP4, no.)
