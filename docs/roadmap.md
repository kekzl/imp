# Roadmap

Single-author, single-GPU experiment - "roadmap" means "current focus," not
"schedule." Shipped work lives in [`CHANGELOG.md`](../CHANGELOG.md);
competitive numbers live in [`docs/BENCHMARKS.md`](BENCHMARKS.md).

**Format: verdict ledgers.** Every entry is fact + number + decision, one line
or one table row each. The investigation behind a verdict lives in
`docs/plans/`, the PR body, or [`LIMITATIONS.md`](LIMITATIONS.md); this file
records what was measured and what stands. Entries are closed, corrected or
superseded in place, never silently deleted.

**Keeping this file honest.** `scripts/check_doc_citations.py` gates the
`path:line` citations here, and it reaches less far than it looks:

- It checks that the line **exists**, not that it says what the sentence
  claims. `weight_map.cpp:369` pointed at the audio-skip branch until
  2026-08-31; the branch had moved to :380 and the gate was always green,
  because line 369 exists either way. Only reading the file finds that.
- A bare basename resolving to more than one file is reported `AMBIGUOUS` and
  **passes**. Verified by aiming a citation at line 99999 of a 1447-line file:
  path-qualified it fails, bare-basename it does not. Cite the path.
- Duplicate copies of the tree make every basename ambiguous at once. Three
  stale `git worktree` checkouts under `.claude/worktrees/` were doing exactly
  that: removing them took the doc set from 79 unchecked citations to 0.

## Direction: local inference for AI agents

Goal: fastest local engine for AI agent workloads on consumer Blackwell.
Agents generate 20k-100k+ tokens per session, accumulate context fast, run in
parallel: long context, concurrent requests, high decode throughput.

Foundations (shipped 2026-05): chunked-prefill FMHA + 256 MiB S-matrix, ctx
ceiling ~4-6k to 32k+ (#453) · multi-request decode batching (#454) ·
StreamingLLM auto-enable on full KV (#455).

## Current focus: concurrent decode at agentic fan-out

Batch=1 is settled; the working regime is aggregate throughput at tens of
concurrent streams. Ground already shipped: warm weight cache (#956),
suspend-to-RAM (#954), request-order independence (#957), gemma-3 IMA fix
(#959). The serving-lever ledger is in Open item 0 below.

## What a 2026 engine has to do (assessed 2026-08-21)

The bar for one GPU, checked against the tree (not recalled) and against
[vLLM Q3 2026](https://github.com/vllm-project/vllm/issues/48168),
[SGLang Q2 2026](https://github.com/sgl-project/sglang/issues/22949), the
[MLSys 2026 report](https://www.modular.com/blog/three-trends-from-mlsys-2026).
llama.cpp publishes no 2026 roadmap.

### Met

| Expectation | Where |
|---|---|
| Three API dialects natively, not via a shim | OpenAI chat/completions, Anthropic `/v1/messages`, OpenAI Responses; one shared SSE driver |
| Tool calling, gated by real harnesses | aider, Claude Code and the OpenAI Agents SDK drive imp in `make test-agents-external` |
| Constrained decoding past JSON | JSON Schema, regex, GBNF; an uncompilable constraint is a 400, not a free-text answer |
| Prompt caching with explicit breakpoints | prefix cache on by default, `cache_control` per breakpoint, content-salted so a different image is a different key |
| Embeddings and reranking in the same server | `/v1/embeddings`, `/v1/rerank`, validated against llama.cpp on the same GGUF |
| logprobs that agree with what was emitted | at temperature 0 the emitted token IS `top_logprobs[0]` (`tests/test_server_logprobs.py`) |
| Per-request adapter selection | `lora` body field, empty means the base model |
| Latency observability, not just counters | `imp_ttft_seconds`, `imp_inter_token_seconds`, `imp_request_duration_seconds` histograms, plus `imp_queue_depth` and `imp_tokens_cached_total` |
| Auth, rate limiting, backpressure | `--api-key`, per-key rate limit, `max_concurrent`, 429 |
| Every setting reachable from a container, without a name per setting | `IMP_CONFIG` / `IMP_SET` bridge `--config` / `--set`, so a new config key needs no new env name; the 19 hand-written `IMP_*` names are frozen compatibility (#1823) |
| Continuous batching over a paged KV cache | default block n=16, geometry per configuration |
| Chunked prefill and graph-captured decode | CUDA graphs on both paths; gate asserts decode >= 1.3x, measures 2.28x |
| Speculative decoding that pays | n-gram, suffix index and a trained MTP head (+21.3 % at `mtp_k=1`) |
| Quantized KV, including 4-bit | FP8 E4M3, INT8, INT4, NVFP4, and an NVFP4 attention-decode kernel |
| Graceful behaviour when the KV pool fills | StreamingLLM sink plus sliding window; growable pool commits as the card frees up |
| Weight formats a user actually has | GGUF K-quants and IQ, safetensors, NVFP4, MXFP4, native FP8 |
| Model classes, not one family | dense, MoE, MLA, Mamba2/GDN hybrids, vision-language, encoder-only |
| Operating it without a restart | model swap that drains in-flight work, `/admin/suspend` and `/admin/resume` |
| Cold start that is not a full reload | warm on-disk weight cache; vLLM still carries cold start as an open Q3 roadmap issue |
| Reproducibility as a product property | `runtime.deterministic` covers MoE routing atomics, sampling races and GEMM; see [`determinism.md`](determinism.md) |

### Open

Ranked by what an agent workload notices first.

0. **Concurrency scaling on the GDN hybrid vs vLLM: 1.58x gap at 32 streams,
   attributed per component and driven to ~1.08x** (nsys on both engines, same
   checkpoint, same 32-stream wave; full attribution table with PROV in
   [`BENCHMARKS.md`](BENCHMARKS.md)).

   Attribution of the 422 us/token wall delta (2026-08-24):

   | component | us/token | mechanism |
   |---|---:|---|
   | GEMM class | 145 | CUTLASS block-scaled at M=32 vs vLLM's Marlin W4A16 split-K; the "no-split ceiling" survey in [`plans/2026-08-24-qwen38-port.md`](plans/2026-08-24-qwen38-port.md) holds only for no-K-split designs |
   | GPU idle | 143 | imp 15.9% idle vs vLLM 5.2%: 438k vs 200k launches/window (2.2x/token) + 26-42 ms host stalls |
   | small classes | ~135 | mostly launch-coupled |

   Serving-lever ledger (batch=32 aggregate unless stated; alternating pairs,
   fresh server per arm):

   | lever | verdict | measurement | ref |
   |---|---|---|---|
   | small-M mxf4nvf4 GEMM v2 | SHIPPED default-on | +16.0% @32 (992.5 -> 1151.7), +36.0% @8 (363.8 -> 494.6) | #1766; Marlin sidecar #1764 closed unmerged |
   | row-block batched RMSNorm | SHIPPED | +6.8% @32 | #1769 |
   | shared-activation quantize | SHIPPED | +4.6% @32 | #1771 |
   | producer-side quantize fusion | SHIPPED | +2.6% @32 (1160.4 -> 1191.0, 3/3) | #1773 |
   | GDN-out quantize fusion | NEUTRAL, closed | +0.4% over 6 trials | #1774 closed unmerged |
   | gate\|up / in\|z sibling-pair launch | SHIPPED default-on | +1.7% @32 (1713.3 -> 1742.0, 3/3), -112 launches/step | `gemm.nvfp4_smallm_pair`, 2026-08-27 |
   | batched post-step sampling chain | SHIPPED | +2.2% @32 (1740.9 -> 1779.4, 3/3); ~124 launches/step, 6.6% of wall removed | 2026-08-27 |
   | batched residual accumulate (beta=1) | REFUTED | -0.9% median, 3/3 negative (1779.0 -> 1763.6); residual adds already overlap in the graph | `gemm.nvfp4_residual_beta1`, 2026-08-27 |
   | graph prewarm | RETIRED as throughput, SHIPPED as latency | wave-1 aggregate unmoved (629-650 vs 627); wave-1 p50 -3-12% (6.2 s tight vs 6.4-7.1) | #1761, `runtime.graph_prewarm` |
   | batch=1 async-loop recapture fix | ITL fix, not a lever | FRESH captures 128 -> 7 per ~200-tok burst; +0.2% throughput; the 27.8 ms/gap nsys read was CUPTI inflating graph instantiation | 2026-08-27 |
   | host turnaround | ATTRIBUTED, closed as defect class | per step: build 63-82 us / fwd-enqueue 34-47 / distribute 7 / schedule 1-2; outside-step 1.2-1.5 ms = paced serial prefill (`prefill_chunk_decode_cap`, documented ITL trade); turnaround 34 us with no ingest | `diagnostics.step_timing`, 2026-08-27 |
   | prefill concurrent with decode | NEUTRAL both shapes, default-off | short prompts 1771.3 vs 1777.7; 1000-tok ingest 789.7 vs 790.6, TTFT unchanged; no green-context SM partitioning on sm_120, streams displace each other | `runtime.prefill_overlap`; [`plans/2026-08-27-prefill-decode-overlap.md`](plans/2026-08-27-prefill-decode-overlap.md) |
   | ragged cross-sequence prefill | SHIPPED default-on | +6.2% @32 (977.3 -> 1038.2, 12/12 waves), TTFT p50 4.11 -> 2.55 s, p90 4.18 -> 3.57 s | `runtime.prefill_batch`, #1780 |
   | BF16 GDN state | SHIPPED default-on | scan 2.04x isolated (FP32 scan was AT the 1527 GB/s resident ceiling); +12.5% @32 KV-pinned (1210.5 -> 1362.0), +7.7% pure defaults; PPL +0.21% | `gdn.state_bf16`, #1778 |
   | growable KV under aggregate pressure | SHIPPED opt-in | 32x 8k/512: wall median 86.0 -> 65.2 s (-24%), pool 2046 -> 6483 blocks; ceiling was captured post-clamp (ceiling == commit) and growth only fired for one oversized request - both fixed | `kv_cache.growable`, #1794 |
   | NVFP4 GQA-tile decode attention | REFUTED | -9% e2e, 9/9 waves: one layer's KV across 32 seqs (~42 MB) fits the 96 MB L2, per-Q-head re-reads are L2 hits; kernel is L2-latency-bound. bitdecoding_qk -5% on same harness | #1785, branch perf/nvfp4-gqa-decode |
   | auto max_batch_size on hybrids | FIXED | resolver priced hybrid KV 4x too high (224 -> 630 @32); `max_seq_len: auto` was VRAM-blind on packed-4-bit KV | 2026-08-25 |
   | burst serving fixes | SHIPPED | HTTP pool sized to streams, token-charged prefill budget, id-based rotor; 4-wave bench 1047-1073 tok/s every wave | #1762, #1758 (deferred delivery +4-5%) |
   | adaptive MTP chain depth (M=1) | SHIPPED default-on | AIMD 1..mtp_k on acceptance, econ guard prices the depth that ran; mtp_k=2+ngram=false: think chats 111.1-113.3 vs 106.3-108.0 (k=1) and 94.9-110.2 (fixed k=2, bistable); draft-rich 158.1 (parity with fixed, +31% vs k=1); no-think prose 107.8 vs 105.0 fixed / 109.4 k=1. Harness `tools/analysis/mtp_adaptive_ab.sh` | `speculative.mtp_adaptive_k`, #1801 |
   | `mtp_k=auto` as the default (M=1) | SHIPPED default-on | single-stream on a checkpoint with an MTP head drafts with it: 95.8 -> 141.6 tok/s (+48%), Qwen3.8-27B-NVFP4 thinking, 3 alternating rounds, degen 50/0. Declines for concurrent serving | `speculative.mtp_k=-1`, #1809 (+ #1811, it read the raw flag not the resolved batch) |
   | NVFP4 paged-decode load width (M=1) | SHIPPED | one word per lane instead of one `LDG.E.U8` per byte, K and V issued before the warp reduction: 64.0 -> 74.1 tok/s @77k (+15.7%, forced-equal emissions). 4-bit KV went from 13.5% SLOWER than 8-bit to 2.3% faster, which is what makes `dtype=auto` right on both axes here | #1817 |
   | sparse decode at concurrent long context | SHIPPED opt-in | 3 streams x 25k, Qwen3-8B-Q8_0 fp8-KV: 155.6 -> 197.7 tok/s (+27%, 3 alternating trials); metadata now one batched launch per forward. Harness `tools/analysis/serving_sparse_ab.sh` | `attention.sparse_topk_tokens`, #1808 |

   Standing state: measured gap to vLLM **~1.08x pinned** (after #1778 +
   #1765-fix). The auto=28-vs-pinned=32 delta (630 vs 936) is NOT a rotation
   cost - the scheduler admits 28 and the last 4 drain as a near-empty batch;
   auto=28 sustains full rate under continuous arrival. Remaining engine-side
   posts: launch-coupled idle, recurrent-state paging (the lever for 32-way
   concurrency at LONG context, or for freeing the 2544 MiB GDN pool; at 32
   slots it is not the limiter). The Qwen3.8 port roadmap is CLOSED - final
   table in [`plans/2026-08-24-qwen38-port.md`](plans/2026-08-24-qwen38-port.md).

   **Batch=1 roofline, re-derived 2026-08-27** (graphs-ON nsys window, 778
   steps): box reads **1628 GB/s resident** (8 GiB sweep; the 1530 pin was
   stale), spec-off ceiling on Qwen3.8-27B-NVFP4 = **~112 tok/s** (14.5
   GB/token), measured 87.4 = 78%. Decode graph is STRICTLY serial
   (kernel-interval union == sum, factor 1.000, 718k intervals) - the "launch
   classes overlap away" rule does NOT hold on this model at M=1. Step
   anatomy: GEMV classes 9.69 ms at ~1496 GB/s avg (gate_up 1613 / lm_head
   1655 prove the ceiling; kpar 1450 / multirow 1490 / residual 1528 = ~0.4 ms
   class headroom); tail 2.66 ms = attention 0.48 (latency-bound at short ctx,
   both split directions refuted), 96 FP16 alpha/beta GEMVs 0.37, norms 0.30,
   GDN scan+conv 0.32, host/idle 0.44. The way PAST the roofline is the MTP
   verify (weights read once per k+1 rows): 102-110 tok/s at k=1 (#1796),
   k=2 stabilized by the adaptive depth (#1801, ledger row above) and taken by
   default since #1809; k=3
   doomed on economics; verify-chunk GEMMs run at 1300 GB/s = 70% of verify
   kernel time (`speculative.verify_smallm` +3-6% isolated, +1-2% mixed
   pairs, default off); `diagnostics.mtp_prenorm_h=true` lifts accept 70/72
   -> 74/78%, 4/4 pairs, +2-3%.

   ```
   [PROV: commit=a70d7863+wt date=2026-08-27 hw=RTX5090
          model=Qwen3.8-27B-NVFP4 cuda=13.3 path=nsys server window 778 steps
          cmd=`nsys profile ... imp-server` + chat 1024-tok]
   ```

1. ~~**Scheduling has no per-request priority.**~~ **Closed 2026-08-28:**
   `"priority"` body field (vLLM semantics, lower first, default 0, all three
   dialects) is the primary admission sort key in `Scheduler::schedule`;
   shortest-first-with-aging orders within a class; strict dominance across
   classes is the documented contract ([`API.md`](API.md)). Admission order
   only, no preemption. Unit-tested incl. aging-does-not-cross-classes
   (`tests/test_scheduler.cpp`). *History (#1634):* scheduling never was
   arrival order - shortest-first on every arrival starved long prompts;
   `Scheduler::kAgingRounds` bounds that.
2. **Long context is served by a 2023-era answer - half closed 2026-08-28.**
   Shipped: Quest-class top-k page selection for decode
   (`attention.sparse_topk_tokens`, opt-in) - keeps the whole KV, reads the
   top-scoring blocks per step (per-block key min/max bound, device-side,
   graph-safe, dense-identical below `sparse_min_ctx`). Qwen3-8B-Q8_0 fp8-KV
   decode, 3/3 alternating rounds, `make build` image: 32k ctx 160.3 ->
   199.5 tok/s (+24.5%), 16k +4.9%, identity regime -2.6..-2.9%. NIAH 16k:
   dense 15/15, budget-2048 15/15, budget-4096 12/15 where all 3 "fails"
   are think-budget exhaustion at the harness's 384-token cap (needle
   retrieved verbatim at 768). Detail + gates in
   [`plans/2026-08-28-sparse-decode-attention.md`](plans/2026-08-28-sparse-decode-attention.md).
   Spec verify chunks ride the sparse table since 2026-08-29: speculation-on
   32k decode 137.4 -> 176.1 tok/s (+28.2%), ms/verify 233 -> 133, NIAH 32k
   spec-on 15/15 both arms.

   Three things landed after that assessment:

   | | |
   |---|---|
   | concurrent serving, not just batch=1 (#1808) | 3 streams x 25k on Qwen3-8B-Q8_0 fp8-KV: aggregate decode 155.6 -> 197.7 tok/s (+27%, 3 alternating trials). Metadata updates are one batched launch per forward, ragged mapping included |
   | the KV-dtype gate is gone (#1818) | the metadata kernel has an NVFP4 arm and the NVFP4 decode branches consume the compacted block table they used to ignore. Qwen3.8-27B-NVFP4 @77k: 74.3 -> 100.2 tok/s, against 96.8 for FP8 in the same shape |
   | the configured budget was not the effective one (#1819) | every token-to-block conversion used the compile-time block size (16) while this family runs 32-token blocks, so `sparse_topk_tokens=N` bought 2N and `sparse_min_ctx` engaged at twice its stated length. **An existing opt-in setting moved operating point**: configure 2N to keep it |

   The metadata pool must stay VRAM-resident: a `kv_cache.max_blocks` pin
   without its headroom left 689 MiB free and WDDM-spilled every prefill
   kernel by 11% (the pinned path now warns with the exact MiB). Retrieval is
   what pays for a small budget, and it pays steeply: on Qwen3.8-27B-NVFP4
   NIAH reads 10/10 dense, 8/10 at 8192 effective, 5/10 at 4096 - at 8192 the
   budget is worth configuring, below it is not.

   Remaining: gated to non-MLA models of uniform geometry, no prefill
   sparsity, and StreamingLLM eviction (`src/compute/attention_paged_common.cuh:71`)
   is still the only answer under KV-pool pressure.
3. **Speculation does not adapt to the request.** `speculative` is a
   per-request bool and the drafter choice is global. *Half closed
   2026-08-27, and the default moved 2026-08-29:* the MTP chain depth adapts
   per request between 1 and `mtp_k` (`speculative.mtp_adaptive_k`, AIMD on
   acceptance - ledger row in item 0), and `speculative.mtp_k` now defaults to
   `auto`: a single-stream run on a checkpoint that ships an MTP head drafts
   with it instead of leaving the speedup switched off (Qwen3.8-27B-NVFP4
   thinking, 3 alternating rounds, 95.8 -> 141.6 tok/s, +48%; degen 50/0).
   Auto declines for concurrent serving. The resolver read the raw
   `--max-batch` flag rather than the resolved batch size for one release, so
   a server pinned to one stream through `imp.conf` declined the head and left
   +33% unclaimed (#1811). The chain saturates near 2.5 accepted/verify. vLLM
   targets acceptance length >5 (hybrid + linear drafting), SGLang builds
   adaptive per-request spec configs. A speculation tree (gap 5) raises the
   same ceiling from the other side.
4. **No audio; a checkpoint that has it loses it quietly.** Gemma-4 ships
   `model.embed_audio.*`; `src/model/weight_map.cpp:380` folds those tensors
   into the aggregate `skipped` count, so an omni checkpoint loads as
   text+vision and says so nowhere. (The citation read `:369` until
   2026-08-31, eleven lines off and green the whole time - see "Keeping this
   file honest" for why the gate cannot see that class.)
5. **No video.** Gap 2 below; the Qwen3-VL tower does images only.
6. **No KV tier below VRAM.** Gap 6 below - shelved on measurement (6.5x
   bandwidth cliff), not on size.
7. **No multi-GPU / tensor parallelism.** Scope decision:
   [`LIMITATIONS.md`](LIMITATIONS.md) states it first for a reason.
8. **The quantizer refuses 3-D stacked experts.** Gap 1(f) below: needs a
   per-model layout descriptor plus per-expert bias support in loader and MoE
   forward.
9. **No distributed tracing - the id half closed 2026-08-28.** Client-sent
   `X-Request-Id` echoed on every response (refusals included, sanitized,
   128-char cap); generation endpoints answer with the server completion id
   when none sent; `--log-requests` JSONL carries `client_request_id` next
   to `req_id` ([`API.md`](API.md), "Request tracing"). Remaining: no
   OTLP export, no per-request span timing.

## Open gaps to the mission (assessed 2026-07-26)

Ranked audit of what separates imp from "best agentic engine on a 5090". The
raw-speed half of [`GOAL.md`](GOAL.md) is met (batch=1 decode +13-48% vs
llama.cpp on every hero, 2026-07-12 re-sweep; MoE prefill leads vLLM
single-seq; cross-engine PPL parity measured). Everything below is the
*agentic* half.

| # | Gap | State |
|---|---|---|
| 1 | First-party NVFP4 quantizer | **partial** - converts, runs, AWQ calibration ships; findings ledger below |
| 2 | Vision beyond Gemma | **largely closed** - Qwen3-VL end to end (#1163-#1180); no video, no second tower family |
| 3 | One server, one model | **closed** - `server.model_swap` (#1080) |
| 4 | Constrained decoding JSON-only | **closed** - regex (#1091) + GBNF (#1095) |
| 5 | No speculation tree | **half closed** - trained MTP head pays (+21.3% at k=1); no EAGLE/Medusa/multi-candidate tree; `token_recycling` re-measured neutral |
| 6 | Context VRAM-capped, no host spill | **shelved on measurement** - no reproducible trigger on this box, and the spill lands on a 6.5x cliff |
| 7 | Agentic quality unmeasured vs competitors | **closed** - three model families in [`BENCHMARKS.md`](BENCHMARKS.md) |
| 8 | No GBNF/EBNF surface | **closed** - `response_format: grammar` / `grammar` / `guided_grammar` (#1095) |
| 9 | `/v1/rerank` absent | **closed** - Cohere/Jina/vLLM shape, validated vs llama.cpp on the same GGUF |
| 10 | Agent-harness batteries imp-internal | **closed** - real aider / Claude Code / OpenAI Agents SDK in `make test-agents-external` |

Shipped alongside: live web UI at `GET /` (#1078) + the streamed non-ASCII
corruption fix building it exposed.

### 1. First-party NVFP4 quantizer - EXPERIMENTAL, calibration ships

`imp-quantize` converts dense BF16/FP16 SafeTensors to NVFP4 in-tree;
`imp-cli --perplexity <corpus> --calibrate <file>` + `imp-quantize --calib`
does AWQ-class activation calibration. Measured over `ppl_corpus_45k.txt`
(13 537 tokens, calibrated on held-out prose): Qwen3-0.6B BF16 24.06 / RTN
30.10 (+25.1%) / **AWQ 28.48 (+18.3%)**; Qwen3-1.7B (sharded) BF16 17.22 /
RTN 20.43 / **AWQ 19.21 (+11.5%)**. `degen_suite.py` 45/45 on every
checkpoint involved. Still experimental. Detail:
[`quantization.md`](quantization.md).

Findings ledger (each measured, dates inline):

| finding | verdict | numbers |
|---|---|---|
| (a) micro-scale search vs absmax | not worth it | PPL 30.10 -> 29.88 (0.7%) for ~6x cost; 16-value micro-block leaves absmax near-optimal, the FP4 grid dominates - hence AWQ (move the error), not better scales (2026-07-26) |
| (b) o_proj scale folded into v_proj vs FP8 KV | refuted concern | FP8-vs-FP16-KV penalty 0.300 PPL calibrated vs 0.595 RTN - scaled v_proj is FRIENDLIER to FP8 KV (2026-07-31) |
| (c) calibration determinism | forced | without `deterministic_gemm` two runs differ on 94% of floats, PPL moves 1.6%, degen probes flip; `--calibrate` now forces it |
| (d) "MoE not supported" | wrong in the dangerous direction | experts quantized fine (4992 on DeepSeek-V2-Lite); MLA latent projections + router broke and are now refused; 3.28x compression, degen 3 FAIL/32 = strict subset of BF16's 5 (2026-07-31) |
| (e) head-to-head vs Modelopt export | imp ahead on one model | Qwen3-14B, same source weights (bit-identical untouched tensors): Modelopt 10.0301 vs imp-quantize uncalibrated **9.9252** (+1.05%). One model, one corpus; retires "prefer a published export", not more. Export ships input_scale/k_scale/v_scale that imp verifiably does not apply (W4A16 vs W4A4 rounding) |
| (g) calibrate off a quantized twin | works, and exposed a 14B regression | 0.6B: twin-calib 28.8868 vs BF16-calib 28.4782 vs uncalib 30.0979 (3/4 of the gain). 14B: RTN **9.9252** vs twin-calib 12.6016 / Modelopt-twin 12.2853 - two independent quantizers agree, so `--calib` itself HURTS at 14B; ruled out: incomplete plan, degenerate stats, magnitude, FP8 KV. Cause: the search minimises per-group weight-reconstruction error, a local proxy (2026-08-01) |
| (h) why 14B flips: attribution via `--calib-groups` | ANSWERED - the attention pair | vs own RTN baseline (n_rep=5): **BD (FFN groups) -0.1330** (best measured), BCD -0.08, C +0.02, A +0.65, ABCD **+2.68**; interaction C x ABD = **+1.90 = 71% of the damage** (A x C +1.36, BD x C +0.03). On 0.6B (n_rep=2) same interaction +0.05 - 40x smaller, effects add, ABCD wins (-1.21). Mechanism: C's GQA tie is a max, inflating channel weight median 1.346 at n_rep=5 vs 1.000 at n_rep=2, and that statistic IS the search weight. Rule: **`--calib-groups BD` on wide-GQA, ABCD on narrow-GQA**; A hurts both models (+0.28/+0.65). No VRAM ceiling either way (uploads one group at a time: ~0.7 GiB @14B) (2026-08-05) |
| (i) vLLM-loadable output | SHIPPED | `--format vllm` writes compressed-tensors `nvfp4-pack-quantized`; vLLM 0.27.1 loads and generates (0.6B + 27B, 51.8 -> 19.2 GiB). Tensor scale is stored INVERTED between layouts; engines keep ONE scale per fused q/k/v / gate/up group - sharing it is also better quantization (0.6B 30.40 -> 29.42, 27B neutral). Refuted in passing: absmax/(6x448) scaling (0% subnormal micro-scales, as claimed) measures 31.05 - worse than absmax/6. Spin-off fix: compressed-tensors detection keyed on recipe.yaml alone read exports without one as Modelopt and inverted (PPL 1.2e47) (2026-08-16) |

**(f) 3-D stacked experts: refused, and support is not the cheap job it
looked** (2026-08-01). The old refusal never fired (rank check behind a
`.weight` name test no real checkpoint matches) - experts were copied through
as BF16 while `hf_quant_config.json` announced NVFP4 (#1188). De-stacking
evaluated against a real gpt-oss-20b checkpoint and rejected:

- The fused layout is not one layout: `model_config.h:249` documents gpt-oss
  `expert_gate_up_bias_fused` as interleaved, the Gemma-4 split in
  `weight_upload.cu` is concatenated - no shape tells a de-stacker which.
- Expert biases have nowhere to go: gpt-oss applies them as a stacked
  `[ne, ...]` tensor (`moe_add_expert_bias_sorted`); the generic per-expert
  loader branch matches `.weight` only.

Doing it properly = per-model layout descriptor + per-expert bias support in
loader and MoE forward. Deferred on that basis. (Same checkpoint also
exposed: nothing refused a MoE model whose experts all failed to map - imp
generated garbage; fixed separately.)

### 2. Vision beyond Gemma - Qwen3-VL shipped (#1163-#1180)

`Qwen3-VL-4B-Instruct` describes images end to end (`imp-cli --image`,
`/v1/chat/completions`, several images per request). The three pieces, each
one line - two of them were misfiled as encoder work by the original
assessment:

- Encoder dynamic resolution: patch budget (`runtime.vision_max_patches`,
  default 4096), larger images scaled down, 1795x2397 -> 972 tokens.
- M-RoPE lives in the TEXT model: `mrope_section` from `hf_config_loader.cpp`
  drives `rope.cu`; per-token (t,h,w) ids from `mrope_positions.cpp`.
- DeepStack: taps after each of the first `n_deepstack` LM layers at
  image-token positions (`executor_forward.cu`, `deepstack_inject.cu`).

Text-only paths bit-identical to before. Tensor inventory + traps:
[`plans/2026-07-31-qwen3-vl-vision.md`](plans/2026-07-31-qwen3-vl-vision.md).

Measurement debt paid 2026-08-11: `Qwen3VLPipelineTest` was runnable from no
make target (env var only set in `tools/mutation/run.py`, pointing at a GGUF
for a SafeTensors test) - `make test-vision` now runs it; until then "runs end
to end" rested on one manual run.

Remaining:

- ~~One image per request~~ closed: several `image_url` parts encode in
  prompt order into one concatenated embedding; unreadable image = 400 (a
  dropped one would shift later images onto wrong placeholders).
- **No video** - a project, not a task: decoder dependency (only `stb`
  vendored), frame axis, real temporal M-RoPE axis, `<|video_pad|>`,
  non-per-image budget.
- **No VL family with a different tower** - no vision arch registry, only an
  allowlist of `model_type` values naming the same Qwen3-VL layout
  (`vision_tower_supported()`). InternVL/Pixtral = port-sized (config parser,
  name map, loader, encoder forward); Qwen2.5-VL additionally windowed
  encoder attention. InternVL tiling is satisfied by the multi-image path.
- A second model on the EXISTING tower cost two gates, not a port (#1379 +
  #1384): the "fifteen PRs" estimate was wrong for Qwen3.6-35B - blocked by a
  string compare and an unconditional `model.visual.*` skip. Cost by reading
  the checkpoint, not by category.

### 3-10, closed - one entry each

- **(3) Model swap** - `server.model_swap` (default on): in-flight
  generations drain (never cancelled), a failed load restores the previous
  model; `/v1/models` lists the directory. Serial by nature: 32 GB holds one
  model; warm cache (#956) makes repeats cheap.
- **(4) Regex constraint** - `response_format: {"type":"regex"}` +
  `guided_regex`, built on the in-tree `RegexNfa` (a second engine was
  written, measured identical, discarded). The work was the decode-time
  wrapper: mask cache, EOS gating, and closing every mask bypass (spec-ngram
  + graph-loop routers, two `apply_mask` sites, thinking suppression, pooled
  state).
- **(5) Speculation tree** - still no EAGLE/Medusa/multi-candidate verify.
  "No trained draft head" retired 2026-08-19: the MTP head pays +21.3% at
  k=1. `token_recycling` re-measured 2026-08-19 on the fixed build: **-0.27%,
  neutral** (was -7.0% on 2026-07-27; same cause as the MTP flip -
  `ea547a53` made the verify chunk cheaper). Level moved too: 156 tok/s where
  the old measurement read 99.37, because #1102 sits between (NVFP4 decode
  cache was double-charged for KV, leaving 100/280 tensors decoding from
  Q6_K; fixed, sweep flat 162-163 across every capacity).

  | `token_recycling` | tok/s (r1, r2) | mean | drafted | accepted | verifies |
  |---|---:|---:|---:|---:|---:|
  | off | 155.95, 155.07 | 155.51 | 48 | 4 (8.3%) | 3 |
  | on | 154.76, 155.41 | 155.08 | 77 | 9 (11.7%) | 27 |

  ```
  [PROV: commit=d374df1b date=2026-08-19 hw=RTX5090 model=Qwen3-14B-Q6_K
         quant=Q6_K cuda=13.3 path=imp-server n=3 reasoning prompts x 2
         alternating rounds cmd=`imp-server --think-budget 0 --set
         speculative.token_recycling=true|false --set server.prefix_cache=false`,
         tokens from usage.completion_tokens, counters from /metrics]
  ```

- **(6) Context spill below VRAM - do not build it** (scoped 2026-08-01).
  The "silent context loss" half was stale: a prompt past the window is a
  typed refusal; generation hitting the window ends `finish_reason:
  "length"`; StreamingLLM eviction is double-gated (<10% free blocks AND
  FP16 KV) and since 2026-07-31 client-visible
  (`usage.prompt_tokens_details.evicted_tokens`, verified on all three
  dialects). The capacity half is shelved on three measurements: no
  reproducible trigger (AUDIT B84: 4k/32k/128k all granted; holding 128k at
  81 MiB free costs 1% decode), the spill lands on a 6.5x cliff (AUDIT B36:
  1531 vs 237 GB/s resident/spilled - the #1103 mechanism), and each
  transfer blocks the host ~165 us (`executor_elementwise.cu:409`) = 4.7% of
  a 3.5 ms step regardless of bytes. Structural: spill cannot be expressed
  in the block table (negative ids already mean "skipped"; quantized kernels
  don't check the sign), so restore must happen before `batch.cpp` builds
  the table - prefetch has one good hook (`prefill_allocate_kv_blocks_`) and
  decode knows only the next block ~3.5 ms ahead. Revisit on: a pool clamped
  below request, a shallower H2D cliff, or workloads past the 128K ceiling.
- **(7) Agentic quality vs competitors** - `tools/analysis/agentic_compare.py`
  published in [`BENCHMARKS.md`](BENCHMARKS.md) (3 families, 4 budgets,
  8-turn sessions). Headline is a defaults difference: at 200-token budget
  imp keeps every contract, llama.cpp needs ~800. Found a real imp bug on
  first run (Llama-3.2 bare-JSON tool calls dropped, #1088). vLLM/SGLang
  deliberately out of scope (weight format + VRAM).
- **(8) GBNF** - a nondeterministic pushdown simulator
  (`src/compute/gbnf_grammar.cpp`, parser in `gbnf_parser.cpp`) behind the
  same `apply_mask` contract. Two dominating costs: mask build 333 ms -> 12
  ms (interned parse stacks + memoised successor sets), and refusal
  discipline (left recursion incl. nullable-prefix and star-over-nullable,
  undefined rules, missing root, repetition bombs = compile-time 400). UTF-8
  assembled across token boundaries, overlong encodings rejected. Mutation
  run found a real hole: recompile cleared the stack arena but not memoised
  transitions - second grammar on a pooled manager decoded with the first
  one's.
- **(9) /v1/rerank** - causal-LM cross-encoder (Qwen3-Reranker class), query
  + document scored JOINTLY in one prefill-only forward reading two logits;
  the entry's premise (needs BERT-style seq-classification arch) was wrong
  for the current generation. Validated vs llama.cpp on the same GGUF: top-1
  agreement 3/3 queries, median score delta 0.0014. Gate: `make
  test-rerank`. Known: first call after load scores ~1e-3 off later ones
  (cold vs cached-prefix arithmetic); ordering unaffected.
- **(10) External agent harnesses** - stage 2 runs real binaries: aider
  (OpenAI), Claude Code (`ANTHROPIC_BASE_URL`), OpenAI Agents SDK
  (`/v1/responses`), each landing a real edit in a throwaway repo. Paid for
  itself on run 1: with tools present the STREAMING path handed reasoning to
  the user as the answer (non-streaming was correct); pinned at three
  levels. The Responses leg measured a model property that looks like a bug:
  Qwen3-8B at 400-token budget makes the call (232 used), at 1400 it
  reasons past the tool (511) - the leg pins temp 0 / 400 tokens.
  OpenHands stays out (docker-in-docker, disproportionate).

Explicitly NOT gaps: continuous batching, prefix caching, per-request LoRA,
embeddings, the three APIs, `/metrics`, suspend/resume, sampler surface (DRY,
mirostat, typical_p, logit_bias). Multi-GPU remains a non-goal.

### Built-in live UI - shipped

`imp-server` serves a single-page UI at `GET /` (source
`tools/imp-server/webui/index.html`, embedded at build via
`cmake/embed_webui.cmake`): live SSE render, one bar per token. Cost ~nothing
because streaming, CORS, disconnect-cancel and the reasoning channel already
existed; `EventSource` is GET-only so the page uses `fetch()` +
`ReadableStream`. NOT on the `GOAL.md` surface commitment - deliberately one
file, no build step, no dependencies; anything beyond a thin client belongs
in an external front end.

## Performance work

Batch=1 competitive campaigns are closed as programs; targeted wins keep
landing where new levers appear:

- **FA2 hd=256 prefill default-on** (#930/#932) - Qwen3.6/Qwen3.5 hybrids,
  pp4096 +26% over WMMA.
- **FP8 tile attention** (#899/#900) - FP8-KV decode tiles + GQA batching,
  long-context decode +14%.
- **FP8 SSM projection sidecar** (#949) - Qwen3.6-35B NVFP4 decode +19%;
  extended to GGUF hybrids' Q8_0 GDN projections: 35B UD-Q4_K_M decode +21%
  (closed the last decode combo where llama.cpp led).
- **Speculative decoding economics** (#852/#862-#866) - hybrid-safe verify +
  MTP drafts; echo-heavy agent workloads up to +156% on 27B.

### The MTP verify on a GDN hybrid, decomposed (2026-08-17 .. 08-19)

Trigger: an RTX 5060 Ti report of +85% from MTP-2 on Qwen3.8-27B while imp
measured a loss. Not the comparison it looks like: his 47.4 tok/s is WITH MTP
vs baseline 25.7; both engines sit near their rooflines (84% and 93%) and the
4x bandwidth difference means the same absolute draft overhead amortises
against a 39 ms step, not an 11.8 ms one.

Per-verify cost, k=2, guard off, 300 verifies:

| phase | before #1455 | after | note |
|---|---:|---:|---|
| chunk forward (captured, 3 rows) | 17.8 ms | 17.8 ms | vs an 11.8 ms decode step |
| replay of the accepted prefix | 25.1 ms | 17.2 ms | fires in ~37-48% of verifies |
| argmax + D2H + rollback | 16.4 ms | 10.7 ms | includes the replay |
| decode | 64.3 tok/s | 84.4 tok/s | 83.7 without MTP |

Workload sensitivity (same server, defaults, only predictability differs):

| workload | tok/s | draft acceptance | emitted/verify |
|---|---:|---:|---:|
| ordinary prose, no speculation | 89.1 | - | - |
| ordinary prose, n-gram default | 89.4 | 11.2% | - |
| ordinary prose, MTP k=2 | 87.9 | 58% | 2.3 |
| **repeat a text verbatim** | **876.5** | **98.3%** | **36.6** |

**RESOLVED 2026-08-18 by fixing the engine:** two launch defects (a grid over
tokens; a `GemmContext` built without `cur_spec_verify_`) kept every GDN
projection off the small-M batched GEMV. Fixed in `ea547a53`: MTP k=2 104.06
vs 86.21 spec-off (was 75.26 vs 84.47); kernel ms/emitted-token 11.35 ->
8.93. Every "speculation does not pay" statement below described the broken
build. Not a cross-engine claim: vLLM was measured for acceptance only
(59.7% vs imp 58-64% - parity, so the drafter and the published 87% figure
describe a regime nobody pinned).

Retractions, kept so they are not re-derived:

- ~~"Spend on drafters, not the repair path"~~ retracted 2026-08-18: at k=1
  (75.0% accept, 1.748 emitted/verify) MTP still lost on the old build;
  pricing the two levers: acceptance 75 -> 87% buys +3.2%, verify 20.6 ->
  13.6 ms buys +45.7%. Verify price was the lever, backwards from the k=2
  reading.
- ~~"92% more launches per emitted token"~~ withdrawn 2026-08-18: the
  normalisation divided by the REQUESTED 16384 tokens off a `0.00 ms` bench
  line; real emission was ~1e3 tokens. Absolute kernel times stand.
- ~~"0.72 forwards/token so speculation moves fewer bytes"~~ and the
  launch-count framing generally (incl. a 22.3% CUTLASS share from a broken
  CSV parse - nsys kernel names contain commas): launches fell, GPU time did
  not; counting launches never measured cost.
- Accounting error under all of it: **a verify replaces a decode step only
  when accepted; on rejection it is additional** - a full weight sweep per
  verify regardless of emission. That is why chain-length saturation is a
  consequence, not a separate result.

**k-sweep on the fixed build (2026-08-19):**

| k | chunk rows | tok/s (r1, r2) | mean | vs k=0 | accept | emitted/verify | ms/verify |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 1 | 86.04, 86.02 | 86.03 | - | - | - | - |
| **1** | 2 | 104.44, 104.19 | **104.31** | **+21.3%** | 76.0% | 1.76 | 16.89 |
| 2 | 3 | 103.86, 97.78 | 100.82 | +17.2% | 59.9% | 2.20 | 21.82 |
| 3 | 4 | 88.08, 87.34 | 87.71 | +2.0% | 49.9% | 2.50 | 28.52 |

k=0 reproduces to 0.02% across rounds. Blocker at the time was correctness
(2/6 prompts ended after ~40 tokens re-stating the question,
deterministically; detail in [`LIMITATIONS.md`](LIMITATIONS.md)) - and the
control that isolated the head (`ngram=false`) is also what HID that defect;
`deterministic_gemm` pins a degenerate answer stable, not absent.

Slope attribution (nsys k=1 vs k=3, `--cuda-graph-trace=node` mandatory):

| kernel | share of k=3-k=1 growth | per launch k=1 -> k=3 |
|---|---:|---|
| `gemv_nvfp4_kpar_mb_fp16` (batched verify GEMM) | **65.1%** | 26.95 -> 32.87 us, **+22.0%** |
| `gemv_fp16` | 11.5% | 64.02 -> 66.23 us, +3.4% |
| `gemv_nvfp4_multirow_fp32` | 8.4% | 440.10 -> 442.90 us, +0.6% |
| `gdn_scan_fused` (recurrent state) | **2.3%** | 8.72 -> 10.03 us, +15.0% |

The slope lives in the forward (the batched GEMV's imperfect amortisation),
not the recurrent state. Fit: `4.96 ms + 5.82 ms x rows` (was 5.36 + 6.53) -
an extra row still costs 50% of an 11.62 ms decode step, so chain length is
not a lever (k=3 buys 2%). Acceptance did not move (76.0 vs 75.0%). Per
launch reported on purpose: cross-process greedy is not deterministic (700
vs 92 tokens on the same prompt), per-token normalisation across profiled
processes would report that.

```
[PROV: commit=3c3e9ac9 date=2026-08-19 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 2 alternating rounds
       cmd=`imp-server --think-budget 0 --set speculative.ngram=false
       --set speculative.mtp_k=0|1|2|3 --set speculative.mtp_econ_min_emit=0
       --set server.prefix_cache=false`; throughput on the `make build` image,
       tokens from usage.completion_tokens, verifies from /metrics. nsys arms on
       the dev build (nsys ships only in imp:toolchain) — sound because both arms
       are the same binary and the claim is relative, not an absolute rate]
```

**Four-arm table (2026-08-18, tokens counted from API responses):**

| arm | rows | tokens | verifies | kernel ms/token | emitted/verify | cost/verify |
|---|---:|---:|---:|---:|---:|---:|
| k=0 | 1 | 723 | 0 | 11.39 | n/a | 11.39 ms |
| k=1 | 2 | 729 | 424 | 11.33 | 1.715 | 19.43 ms |
| k=2 | 3 | 721 | 334 | 11.35 | 2.153 | 24.44 ms |
| k=3 | 4 | 768 | 291 | 11.98 | 2.629 | 31.50 ms |

Speculation buys no GPU time (kernel ms/token flat k=0..2); cost/verify is
linear in rows (`5.36 + 6.53 x rows`, residuals -0.50/+1.01/-0.52/+0.01) -
an extra row costs 57% of a decode step where bandwidth-bound it should be
near-free. At k=1 the repair path is structurally unreachable, so the second
row alone costs 8.04 ms.

```
[PROV: commit=196a3384 date=2026-08-18 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 256 greedy tokens per
       arm cmd=`--set speculative.ngram=false --set speculative.mtp_k=0|1|2|3
       --set speculative.mtp_econ_min_emit=0 --set server.prefix_cache=false`,
       nsys -t cuda --cuda-graph-trace=node; kernel time = cuda_gpu_kern_sum
       total, tokens from usage.completion_tokens, verifies from /metrics]
```

Buried, so they are not re-run:

- Six drafter-accuracy hypotheses (draft lm_head precision, quantised head,
  gamma=1+W offset, hidden-state convention, RoPE, uninitialised MTP KV):
  all measured dead ([`LIMITATIONS.md`](LIMITATIONS.md)).
- MoE draft head: this checkpoint's head has no experts/router.
- Unfused verify chunk: per-launch flat at 32.5 us across 2-3 rows.
- Repair forward as main cost: unreachable at k=1, ~21% of verifies at k=2.
- Async conditional-graph loop (MTP switches it off): worth 1.0% measured,
  not the assumed 27-45%.
- Recurrent-state mechanism for the greedy divergence: disproven
  (`gdn.chunkwise_scan=false` byte-inert, divergence survives).
- Five decode-side kernel hypotheses (chunkwise scan, fused QK-norm+RoPE,
  multirow split, fused FFN, attention family): all null, each instrument
  proven live first; structurally impossible anyway - every emitted token in
  a spec arm comes from the verify chunk, no decode kernel is on the path.
- Chunk-side kernel choice: `verify_nvfp4_gemm=false` moves the divergence
  offsets (58/130/150 vs 79/332/243), never closes it.
- Correction: ALL three prompts diverge with mtp_k=2 (not two of three);
  control holds across eleven spec-off processes.
- Cross-process spec reproducibility at temp 0: does not hold (8/9 processes
  agree at k=2, the two k=1 processes disagree with each other) - why #1457
  and #1467 both "saw" contradictory things. Documented in
  [`LIMITATIONS.md`](LIMITATIONS.md).
- Economics guard constant: right guard, coarse constant - 4.0 was derived
  on the eager verify; measured break-even 2.42 (now the k-aware default,
  see `speculative.mtp_econ_min_emit`).

### MoE host offload - from "CPU cold experts" to a shipped LRU path

Origin: compute cold experts on the CPU (ktransformers shape) to reach
80B-120B on 32 GB. The campaign measured its way OUT of that design - the
LRU expert cache + streaming won, no AVX kernels, no `GOAL.md` amendment.
All numbers Q4_K_M GGUF unless marked NVFP4.

Budget measurements (2026-08-10 .. 08-11):

| question | answer | numbers |
|---|---|---|
| host streaming bandwidth | 62.5 GB/s | 16 threads, 24 GiB, NT loads; saturates at 4 threads; ~65% of DDR5-6000 dual-channel |
| routing skew | real, grows with expert count | coverage at 40% resident: Qwen3-30B-A3B 84.7% (2.12x flat), gpt-oss-20b 71.6% (1.79x) |
| does a static hot set transfer? | NO - prompt-dependent | cross-validated: -15.2 / -29.5 points vs each prompt's oracle (78.8/58.7 vs 94.0/88.1%) |
| 8 KiB round-trip cost | 86.2 us D2H+H2D, size-independent (latency not bandwidth) | single D2H+sync 45.4, +kernel between 126.5, bare launch 34.6; the old 165 us figure was a host-CALL cost, 3.8x overstated. Trap: pageable H2D returns before arrival - only synchronized numbers compare (pinned times "slower", 45 vs 22 us, for that reason) |
| calibration curve (9 prompts, 300 splits) | first prompt buys almost all | held-out coverage: flat 40.0% (38 tok/s ceiling), 1 prompt 67.6% (63), 3 -> 70.3%, 8 -> 72.6% (71), oracle 92.1% (142) |
| temporal locality | strong - LRU needs no calibration | median reuse distance 2 tokens; 45% repeat next token, 80% within 8; steady-state hit 94.7/90.0/95.3% at 40% residency = 0.38-0.80 new experts/layer/token; warm in ~64 tokens |
| host-compute vs stream-into-VRAM (120B-A5B shape) | streaming wins ~3x | static split + host compute: 620 MB @62.5 GB/s + round trips = 14.0 ms/token; LRU streaming: 120-228 MB @25.6 GB/s = 4.7-8.9 ms; PCIe reaches 50.6 GB/s at 64 MiB batches |
| can a promotion be hidden? | yes IF prefetched a layer ahead - which is unreachable | overlap bench: prefetched +0.3-0.7 ms vs +10.5-10.7 ms in-front; routing for layer N known only at layer N; recency prediction carries 42-47% (4-token window 63-68%) and fails exactly on misses |
| 120B ceiling band | 52 (held-out) .. 103 (matched) tok/s; plan against 63-71 | 0.35-1.37 GB cold/token + 4.1 ms transfers; idle-GPU numbers |

Ledger of the shipped path (Qwen3-30B-A3B, all 48 MoE layers host-resident
unless noted):

| step | verdict | numbers |
|---|---|---|
| `ExpertCache` at full offload | holds its hit rate | 88.7% hit, ~0.9 new experts/layer/token (traces predicted 0.38-0.80); 24.98 tok/s vs 6.63 staging-only vs 311.24 resident |
| offload penalty decomposition | 19x = 2.47x graphs x 7.8x rest | resident graphs-ON 286.94 / graphs-OFF 116.13 / offload ~14.9 tok/s; launches 18 052 (graph-hidden) / 52 024 / 197 809 |
| ~19 ms/token host issue cost | the bound half | 3091 `cudaLaunchKernel`/token (4.58 us median) + 632 memcpy/token; dequant_q4k 47.1% of GPU time |
| fused MMVQ instead of dequant->GEMV | REFUTED cleanly | expert-path kernel time -43% (504.8 -> 288.7 ms), e2e 0% - launch count unchanged; the path is host-bound, only launch count is currency |
| slot-indexed fused MoE kernels (#1370) | SHIPPED, 2.1x | the LRU pool already IS the contiguous tensor the resident kernels index (`expert_cache.h` slot_ptr; `d_lookup_` holds slot indices); decode 22.9 -> 48.3 tok/s, launches 197 809 -> 61 585 (within 18% of resident) |
| correctness evidence for #1370 | perplexity could not see it (E6: teacher-forced = prefill, gated path is n==1) | supported instead by: same kernels/bytes as resident, addressing failure is loud, decode-only A/B token-equal ~25 then tie-diverges, two families coherent multi-turn |
| remaining D2H+sync per layer | why 2.1x not 16x | residency needs host routing; prefetch already refuted above |
| graphs under offload | BLOCKED, flag kept as escape hatch | `moe.allow_graphs_under_offload` capture aborts every attempt (`moe_host_args_capture_guard` throws - host-read routing); was never deliverable, descriptions corrected #1373 |
| NVFP4 "mandatory on-device" | was unenforced - silent wrong answers | force_host_experts=8: fluent WRONG answer at 88.77 tok/s, =48: repeats `ftp`, all exit 0 (three guards each missed it); refusal now in `model/expert_placement.h` |
| NVFP4 offload decode built (2026-08-13) | correct now | resident 361.97 -> 384.03; 8-layer arm 88.77 wrong -> 44.54 correct (it was fast because it skipped its GEMMs); full offload 23.03 correct. Estimate missed the prefill half: per-expert NVFP4 M>1 fallback handed the 593 MiB matrix to `gemm_nvfp4` (IMA) - staged via the same slot pool |
| the promotion's blast radius | 4 more consumers read qtype as location | Phase 0b host pointer into CUTLASS, Phase 3 D2D from host source, micro-scales still uploaded, `can_decode_fast` unstamped. **Rule: `qtype` says HOW to decode bytes, never WHERE they live** |
| prefill thrash gate (NVFP4) | shipped | misses 106 108 -> 30 451 (3.5x), pp512 flat, tg +4.6% (inside spread - the miss count is the result) |
| `moe.pin_host_experts` (default off) | +14.8% pp512 (6/6), 4.4x load time | WSL2 cannot page-lock mmap; pageable staging costs 76.2 us host vs 2.8 pinned (9.6 vs 32.4 GB/s). One slab per (layer, projection) - a per-expert pinned buffer costs 24.7 s of cudaHostAlloc. Method note: 3 paired rounds read decode -33% and that was noise (off arm spans 34.7-66.1); 6 pairs decide |
| per-layer device staging | 2.5x prefill, ONLY with pinning | 317.6 -> 790.8 tok/s (324 MiB, one layer live); pinning off: 252-286 = staging buys nothing (driver stages pageable anyway; pinning is also what makes a projection contiguous). Re-prices pin_host_experts: break-even ~32k -> ~7.5k prompt tokens; gap to resident 61x -> 22x |
| staged CUTLASS grouped prefill | +136% prefill, opt-in on an UNEXPLAINED decode cost | `moe.staged_cutlass_prefill`: pp512 663.2 -> 1563.9 (6/6, <1% spread), tg 59.4 -> 37.7 (-36%, 6/6) after long prefill but FASTER (25.5 -> 30.6) after pp8 - inherited cache-state difference, unexplained, so not default. Campaign total: prefill 285 -> 1564, gap to resident 11x. smallM branch is unreachable (device-args path selected at `executor_forward_moe_cutlass.cu:159` first); remaining kernel lever is `dequantize_nvfp4_kernel` (52% of kernel time) |
| final regime | transfer-bound, as the budget modelled | H2D 150 GB at ~51 GB/s = 41% of step, kernels ~26%, launches ~24%; overlap does not pay (3 GEMVs x 6.4 us << 100 us threshold) |
| cache budget sweep (`moe.expert_cache_budget_pct`, #1374) | 2.47x from a config value | 5% = 24 slots = 36.6% hit = 10.51 tok/s; 15% default = 73/74.1%/20.99; 30% = 146/89.4%/30.51; 50% = 244/96.2%/51.86. Default deliberately NOT raised (on a real non-fitting model that VRAM is KV + weight cache) |
| the floor | exact, not conservative | below `3*top_k` slots/layer the cache retains NOTHING (0.0-0.4% hit at 14/19 slots vs 24-cell working set, slower than bypass). Sizing rule: offload needs `n_layers x 3 x top_k x slot_size` VRAM first - ~3.5 GB (120B-A5B), ~8.6 GB (Qwen3-Next-80B top_k 10) |
| two unpredicted keeps | - | budget runs the "wrong" way and it helps (pool = % of FREE VRAM, so offload grows the cache 32 -> 73 slots, hit 37.3 -> 89.4%); prefill was thrashing (384 active cells vs 73 slots = 19% structural ceiling) - working-set gate worth +5.6% pp512 (5/5) and lifts decode hit 88.7 -> 95.7% |
| capture-able offload, what it would take | routing AND residency device-side | routing exists (`expert_indices`, `d_lookup_`); a miss needs a host-issued H2D with no device-side ask - either zero misses (contradicts premise) or a defined miss result. 30B at 48 tok/s puts the 52-103 band in reach of a 120B |

Measurement traps this path enforces: prefill varies ~15% between runs of
the SAME arm - only paired alternating rounds decide; decode moves with
prefill length (cache warmth: pp8 -> 72.7% -> 13.82 tok/s, pp512 -> 88.7% ->
24.98); cold vs warm differ 2.4x (20.99 vs 49.60) - state which one a number
is. Reproduce: `tools/analysis/expert_cache_offload_sweep.sh` (MODE=ab),
`tools/analysis/moe_routing_skew.sh`, `tools/analysis/host_transfer_latency.cu`,
`tools/analysis/expert_promotion_overlap.cu`; nsys with
`--set runtime.cuda_graphs=never` on any resident comparison arm.

### Closed competitive records

- **NVFP4 prefill vs vLLM - CLOSED** (2026-06-13, `290a163a`): FA2 FP16-QK
  primary hd=128 prefill +21-24% pp4096; MoE pp4096 +4% ahead, MoE pp2048
  +27%, dense pp2048 ~tie. Residual dense pp4096 ~1.04x is structural: every
  bounded kernel idea refuted; FA2 at ~5% DRAM, cost is the NVFP4 GEMMs
  (~59%), a separately-refuted ceiling.
- **kv-fp8 storage default-on - SHIPPED** (Qwen3 dense/MoE, Llama, Nemotron-H
  MoE via `kv_cache.dtype=auto`; ~768 MiB saved on dense). Blocked, not
  actionable: Qwen3.6/3.5 declare no FP8 hint; Gemma-4's gate-corpus baseline
  PPL is broken.
- **Q4_K_M prefill gap (-38% vs llama.cpp) - refuted**: in-SMEM Q4_K MMQ +
  HMMA built and ncu-proved decode-throughput-bound, tying cuBLAS; beating it
  needs 2x weight VRAM (rejected). Use NVFP4 SafeTensors for fast
  Q4_K-class prefill. [`plans/2026-05-28-q4k-mmq-kernel-design.md`](plans/2026-05-28-q4k-mmq-kernel-design.md).
- **Sawtooth wavefront reordering (#456) - refuted** (2026-05-29): only lives
  in the WMMA fallback, unreachable on the hot path; forced A/B
  flat-to-negative.

## Known limitations

- **MTP on Nemotron-3.5: head drafts (41% offline), verify chunk uneconomic.**
  History in three steps, each superseding the last:

  1. 2026-08-12 build: k=1 -41%, k=2 -54%, k=3 -65% (216/166/129 vs 364) -
     draft path uncaptured, three syncs per draft token. Device-side draft
     MoE (`gemv_f16_moe_decode`) bought +14-23% on the spec path and did not
     change the verdict (bound said even a FREE draft caps at +10.7%).
  2. 2026-08-19 re-measure after `ea547a53`: reads -2.0% at k=1 - but only
     because the acceptance-poor floor unbinds after 8 verifies at 0/8
     accepted; serving accept 0-9% vs 43.9% offline did not add up.

     | k | tok/s (r1, r2) | mean | vs k=0 | drafted | accepted | verifies |
     |---|---|---:|---:|---:|---:|---:|
     | 0 | 366.11, 363.45 | 364.78 | - | - | - | - |
     | 1 | 357.45, 357.63 | 357.54 | **-2.0%** | 24 | 2 (8%) | 24 |
     | 2 | 353.59, 349.88 | 351.74 | -3.6% | 48-80 | 1-7 (2-9%) | 24-40 |

     ```
     [PROV: commit=02872bdf date=2026-08-19 hw=RTX5090
            model=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 quant=NVFP4 cuda=13.3
            path=imp-server n=3 prompts x 2 alternating rounds
            cmd=`tools/analysis/mtp_k_sweep.sh` with MTP_MODEL set, counters from
            /metrics, give-up line from the server log]
     ```

  3. **2026-08-20, resolved: the 0-9% was a defect.** A fully rejected verify
     adopts `spec_snap_slab` (`engine_spec_ngram.cpp:944-950`); `run_gdn` has
     written that slab since #847, `run_ssm` NEVER did (`ssm_scan_prefill`
     had no snapshot parameter) - every fully rejected verify on a Mamba2
     hybrid committed uninitialised VRAM as recurrent state. Poison-fill
     proof: 0 of 26 378 240 bytes written without the wiring, 99.71% with it.

     | | offline top-1 accept | serving accept | k=1 decode |
     |---|---:|---:|---:|
     | before | 851/2097 = 40.6% | 0/24 = 0.0% | 354.80, 356.54 tok/s |
     | after | 861/2097 = 41.1% | 590/1507, 587/1510 = 39.2/38.9% | 177.17, 175.12 tok/s |

     The two counters are the same quantity at k=1 and now agree. Economics
     verdict unchanged, reason corrected: guard disabled, k=1 costs -51%
     (176 vs 363); shipped guard lands 258-341 because 1.41 emitted/verify
     sits ON the 1+0.40k break-even and flips between runs. The drafter was
     never the problem; the verify chunk is. Not MTP-specific: the same
     branch serves any drafter (pre-fix n-gram derailed into unrelated prose
     after the first fully-rejected verify). The 270 `mtp.*` unrecognised-
     weight warnings on this checkpoint closed as #1497
     (`divert_to_mtp` gated on `llm_compressor_format`, Modelopt exports
     reach the generic mapper; head uploads anyway).

     ```
     [PROV: commit=8a7f2763 date=2026-08-20 hw=RTX5090
            model=NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 quant=NVFP4 cuda=13.3
            path=spec-verify/mtp-draft
            cmd=`imp-server --think-budget 0 --set speculative.ngram=false --set
                 speculative.mtp_k=1 --set speculative.mtp_econ_min_emit=0 --set
                 server.prefix_cache=false`, 3 prompts x 700 max_tokens; offline arm
                 `imp-cli --mtp-spec-decode 1 --set speculative.hybrid=false` on the
                 same 3 prompts
            n=2 per arm, arms alternated, fresh process per arm]
     ```

  Context: NVIDIA's own DSpark drafter measured -42% in vLLM on this card;
  the model card recommends no speculation at H100-class bandwidth.

- **Single GPU only.** No tensor parallelism, no multi-GPU.
- **Blackwell only.** No Hopper, Ada, Ampere, AMD, Intel, Apple, CPU.
- **Qwen3.5-27B MXFP4 untested** - old wording stale in each part: the
  alpha/beta NaN is moot (GDN alpha/beta pinned FP16_ONLY,
  `tensor_kind_table.cu`); the SafeTensors loader warns rather than refuses;
  the real blocker is no MXFP4 SafeTensors decode path outside gpt-oss.
  Blocked on a decodable checkpoint, not on a bug.
- **Gemma-4 Q4_K_M code-gen drift** - no longer reproduces (2026-06-13 and
  2026-08-11); original file gone, cannot A/B. If another Q4_K_M quant
  degenerates: Q5_K_M or Q8_0.
- ~~Native-FP8 weights decode through their FP16 companion~~ - **closed
  2026-08-12, measured +7.5% median decode** (27 pairs, t=3.33; order
  balanced: ON-first reads +8.2%, OFF-first +3.8%; bytes/bandwidth predicts
  +6.9%; do not quote the +11.1% mean). Mechanism existed
  (`fp8_decode_sidecar` rule) and did not recognise a native-FP8 source;
  `FP8CacheEntry::native_source` now drives it. Two traps: registering
  without the phase-4 rule moved PREFILL to FP8 (status 15); phase 3 then
  freed the FP16 copy prefill needs - visible only under `--bench` decode
  mode 2. One lookup answering two questions, twice.
- ~~No dequant path for native FP8 weights~~ - **closed**: FP16 companion at
  load (Nemotron-3.5 MIXED_PRECISION: 5935 NVFP4 expert tensors + 46 FP8
  Mamba projections; sm_120 has no FP8 prefill GEMM, raw bytes used to reach
  cuBLAS as `CUDA_R_8F_E4M3` = status 15). Costs 1698 MiB FP16 cache; init
  24.4/32.6 GB. See [`MODELS.md`](MODELS.md).

## Investigated and shelved

- **Draft-model speculative decoding** - separate draft models don't amortize
  weight reads on one bandwidth-bound GPU. Shipped instead: prompt-lookup
  n-gram (#668-#670) and MTP self-drafts with hybrid-safe verify (#852).
- **FFN contextual sparsity** - warp-cooperative layout masks the skip,
  +0-1% measured.
- **BitDecoding (TC KV decode) - shelved; the scope now stated** (#1268).
  The original "0% gain, decode is weight-bound" was measured at tg256 = 64
  prefilled tokens where paged attention is 4.3% of the window. Re-measured
  2026-08-21 (Qwen3-8B-Q8_0): **19.9% at 8k, 43.9% at 32k** (the original
  8k figure of 31.1% carried no model/method/PROV; 32k held). Still shelved
  because the levers died, not because attention doesn't matter:
  - Split-count boost (#1270, reverted #1271): +10.0% at 32k on Qwen3-8B
    (`n_kv_heads=8, g=4`), **-7.30%** on Qwen3-30B-A3B (`n_kv_heads=4,
    g=8`); the separating condition was never established.
  - KV block 16 -> 32: neutral everywhere (-0.48 .. +0.07%).
  - "Latency/occupancy-bound at 192 GB/s": retracted - same kernel reaches
    629.6 GB/s at 32k (3.4x) at unchanged 16-17% occupancy (roofline runs
    `dca16b71_20260806_041710`, `120bc0d7_20260807_091356`); low bandwidth
    at 8k is a kernel short of work.
  - Amdahl at 32k: closing 629.6 -> 1127 GB/s is ~20% of the window,
    against a kernel already at 35% of roofline.
  - Re-open on a mechanism, not the share - the share grows with context:

    | model | KV heads / g | layers | ctx | paged attention | ceiling if zero |
    |---|---|---|---|---|---|
    | Qwen3-8B-Q8_0 (dense) | 8 / 4 | 36 | 8k | 19.9% | 1.23x |
    | Qwen3-8B-Q8_0 (dense) | 8 / 4 | 36 | 32k | 43.9% | 1.76x |
    | Qwen3-30B-A3B-NVFP4 (MoE) | 4 / 8 | 48 | 8k | **29.1%** | 1.36x |
    | Qwen3-30B-A3B-NVFP4 (MoE) | 4 / 8 | 48 | 32k | **50.6%** | ~1.96x (1.92-2.01) |

    The share goes UP across the dense/MoE boundary although attention's
    absolute cost falls (-10.5% at halved KV heads): the non-attention half
    falls faster (-32.0%, small active expert set). Method is a differential
    measurement (two runs differing by exactly 256 decode steps, prefill
    cancels; imp emits no NVTX and kernel names cannot split phases). Checks:
    non-attention flat across ctx (836.9 vs 843.7 ms dense, 571.7 vs 574.1
    MoE), kernel sum 91-98% of the wall step, repeat pairs within 0.12 pp,
    instance counts 256 x layers on both models.

    ```
    [PROV: commit=5b884e44 date=2026-08-21 hw=RTX5090 model=Qwen3-8B-Q8_0 quant=Q8_0
           (NVFP4 decode cache, FP8 KV) cuda=13.3 path=imp-cli n=2 runs per context
           (tg=8 and tg=136), 32k repeated once
           cmd=`nsys profile --sample=none --cpuctxsw=none --backtrace=none -t cuda
           --cuda-graph-trace=node -- imp-cli --bench --bench-pp 8192|32768
           --bench-reps 1 --max-tokens 8|136 --max-seq-len 40960
           --set speculative.ngram=false`; shares from
           `nsys stats --report cuda_gpu_kern_sum`, differenced between the two tg
           values; card exclusive, no other compute process, clocks WARM during the
           timed runs (2692 MHz SM at sample, cold 397 MHz before the first run).
           --cuda-graph-trace=node is mandatory: without it nsys does not attribute
           graph-replayed kernels at all.]
    ```
