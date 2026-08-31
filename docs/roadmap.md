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
`path:line` citations here and checks that the line EXISTS, not what it says
(`weight_map.cpp:369` pointed eleven lines off and stayed green until
2026-08-31); a bare basename that resolves to more than one file is reported
`AMBIGUOUS` and passes, so cite the path; stale `git worktree` checkouts make
every basename ambiguous at once.

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
   | batched ban + penalty sweep (serving default `repetition_penalty` 1.05 + 19 banned ids put every row on the inline chain) | SHIPPED | 1766.9 -> 1774.9 tok/s @32 medians; pairs (base -> new) 1747.7 -> 1772.8 (+1.4%), 1772.4 -> 1774.9 (+0.1%), 1766.9 -> 1782.2 (+0.9%), 3/3 positive; 2 launches per row per step -> 1 sweep per step; re-profiled: steady-window idle 14.9% -> 13.6%, sub-100-us gaps 1127 -> 898 ms per 18 s window, the per-row pair gone from the gap table. Harness `tools/analysis/two_image_conc_ab.sh` | 2026-08-31 |
   | serving idle re-attributed on the current build (nsys node-trace, steady window) | MEASURED | idle 14.9% of wall; >1 ms gaps (45% of idle) are CUPTI-inflated graph captures at the wave ramp (waves with and without them run 5.57 vs 5.51 s); real idle ~8%: launch density 6.3% (~1350 gaps/step, 0.4 us avg inside the replay; 16.7k gaps of 10-100 us = per-row sampling chain + 8 pageable H2Ds/step at ~14 us), host turnaround 1.9% (~2 gaps/step of ~200 us). Harness: `tools/analysis/serving_idle_profile.sh` | 2026-08-31 |
   | sparse decode at concurrent long context | SHIPPED opt-in | 3 streams x 25k, Qwen3-8B-Q8_0 fp8-KV: 155.6 -> 197.7 tok/s (+27%, 3 alternating trials); metadata now one batched launch per forward. Harness `tools/analysis/serving_sparse_ab.sh` | `attention.sparse_topk_tokens`, #1808 |

   ```
   [PROV: commit=f0c57e64 date=2026-08-31 hw=RTX5090 model=Qwen3.8-27B-NVFP4
          quant=NVFP4 cuda=13.3 path=imp-server 32 streams x 3 waves x 300 greedy
          tokens (tools/analysis/conc_client.py), flags=max_batch_size=32,
          max_seq_len=4096, kv_cache.max_blocks=2387; idle: nsys
          --cuda-graph-trace=node on the dev build via
          tools/analysis/serving_idle_profile.sh, window 14-32 s;
          throughput: tools/analysis/two_image_conc_ab.sh imp:ab-base vs
          imp:test, 3 alternating trials, median of 3 waves]
   ```

   Standing state: gap to vLLM **~1.08x pinned**. auto=28 vs pinned=32
   (630 vs 936) is admission, not rotation: 28 sustain full rate under
   continuous arrival. Remaining engine-side posts: launch-coupled idle,
   recurrent-state paging (the lever for 32-way concurrency at LONG context;
   not the limiter at 32 slots). Qwen3.8 port roadmap CLOSED:
   [`plans/2026-08-24-qwen38-port.md`](plans/2026-08-24-qwen38-port.md).

   Batch=1 roofline (re-derived 2026-08-27, graphs-ON nsys window, 778
   steps): box reads **1628 GB/s resident** (the 1530 pin was stale);
   Qwen3.8-27B-NVFP4 spec-off ceiling **~112 tok/s** (14.5 GB/token),
   measured 87.4 = 78%; decode graph strictly serial (kernel-interval union
   == sum, 718k intervals).

   | step component | ms | note |
   |---|---:|---|
   | GEMV classes | 9.69 | ~1496 GB/s avg; gate_up 1613 / lm_head 1655 prove the ceiling; ~0.4 ms class headroom |
   | attention | 0.48 | latency-bound at short ctx, both split directions refuted |
   | 96 FP16 alpha/beta GEMVs | 0.37 | |
   | norms | 0.30 | |
   | GDN scan + conv | 0.32 | |
   | host / idle | 0.44 | |

   Past the roofline only through the MTP verify (weights read once per k+1
   rows): 102-110 tok/s at k=1 (#1796), k=2 stable via adaptive depth
   (#1801), default since #1809; k=3 uneconomic; `speculative.verify_smallm`
   +3-6% isolated, +1-2% mixed, default off; `diagnostics.mtp_prenorm_h`
   lifts accept 70/72 -> 74/78%.

   ```
   [PROV: commit=a70d7863+wt date=2026-08-27 hw=RTX5090
          model=Qwen3.8-27B-NVFP4 cuda=13.3 path=nsys server window 778 steps
          cmd=`nsys profile ... imp-server` + chat 1024-tok]
   ```

1. ~~**Scheduling has no per-request priority.**~~ CLOSED 2026-08-28:
   `"priority"` body field (vLLM semantics, lower first, all three dialects)
   is the primary admission sort key, shortest-first-with-aging within a
   class, no preemption ([`API.md`](API.md), `tests/test_scheduler.cpp`).
2. **Long context** - half closed 2026-08-28: Quest-class top-k page
   selection for decode (`attention.sparse_topk_tokens`, opt-in): Qwen3-8B
   32k 160.3 -> 199.5 tok/s (+24.5%), spec verify on the sparse table +28.2%
   at 32k, concurrent 3x25k +27% (#1808), NVFP4 KV arm #1818 (77k: 74.3 ->
   100.2), block-size fix #1819 (configure 2N to keep an old budget).
   Retrieval price on Qwen3.8-27B: NIAH 10/10 dense, 8/10 at 8192, 5/10 at
   4096. Detail:
   [`plans/2026-08-28-sparse-decode-attention.md`](plans/2026-08-28-sparse-decode-attention.md).
   Remaining: MLA models, prefill sparsity, StreamingLLM eviction
   (`src/compute/attention_paged_common.cuh:71`) as the only answer under
   KV-pool pressure.
3. **Speculation does not adapt to the request** - half closed: chain depth
   adapts per request (`speculative.mtp_adaptive_k`, #1801), `mtp_k=auto`
   drafts single-stream with the head (95.8 -> 141.6 tok/s, #1809, #1811).
   Remaining: drafter choice is global; the chain saturates near 2.5
   accepted/verify, and the multi-candidate tree (gap 5) measured no gain
   past it on this card.
4. **No audio.** Gemma-4 ships `model.embed_audio.*`;
   `src/model/weight_map.cpp:380` folds those tensors into the aggregate
   `skipped` count, so an omni checkpoint loads as text+vision and says so
   nowhere.
5. **No video.** Gap 2: the Qwen3-VL tower does images only.
6. **No KV tier below VRAM.** Gap 6: shelved on measurement (6.5x
   bandwidth cliff), not on size.
7. **The quantizer refuses 3-D stacked experts.** Gap 1(f): needs a
   per-model layout descriptor plus per-expert bias support in loader and MoE
   forward.
8. **No distributed tracing** - id half closed 2026-08-28: `X-Request-Id`
   echoed on every response, server completion id when none sent,
   `--log-requests` carries `client_request_id` ([`API.md`](API.md)).
   Remaining: no OTLP export, no per-request span timing.

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
| 5 | No speculation tree | **closed by measurement** - trained MTP head pays (+21.3% at k=1); multi-candidate MTP verify built for dense and GDN hybrids (`speculative.mtp_tree_width`, #1829/#1830): tree ceiling +6..+10 points top-2 over top-1, think traffic -0.8/-5.8% tok/s vs linear adaptive-k, default-off; `token_recycling` neutral |
| 6 | Context VRAM-capped, no host spill | **shelved on measurement** - no reproducible trigger on this box, and the spill lands on a 6.5x cliff |
| 7 | Agentic quality unmeasured vs competitors | **closed** - three model families in [`BENCHMARKS.md`](BENCHMARKS.md) |
| 8 | No GBNF/EBNF surface | **closed** - `response_format: grammar` / `grammar` / `guided_grammar` (#1095) |
| 9 | `/v1/rerank` absent | **closed** - Cohere/Jina/vLLM shape, validated vs llama.cpp on the same GGUF |
| 10 | Agent-harness batteries imp-internal | **closed** - real aider / Claude Code / OpenAI Agents SDK in `make test-agents-external` |

Shipped alongside: live web UI at `GET /` (#1078) + the streamed non-ASCII
corruption fix building it exposed.

### 1. First-party NVFP4 quantizer - EXPERIMENTAL, calibration ships

`imp-quantize` converts dense BF16/FP16 SafeTensors to NVFP4; `--calib` does
AWQ-class activation calibration. `ppl_corpus_45k.txt`: Qwen3-0.6B BF16 24.06
/ RTN 30.10 / **AWQ 28.48**; Qwen3-1.7B 17.22 / 20.43 / **19.21**;
`degen_suite.py` 45/45. Detail: [`quantization.md`](quantization.md).

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

**(f) 3-D stacked experts: refused (2026-08-01).** The old refusal never
fired (#1188: experts copied through as BF16 while `hf_quant_config.json`
announced NVFP4). De-stacking rejected against gpt-oss-20b: the fused layout
is not one layout (gpt-oss interleaved, Gemma-4 concatenated) and expert
biases have no per-expert loader path. Proper support = per-model layout
descriptor + per-expert bias in loader and MoE forward.

### 2. Vision beyond Gemma - Qwen3-VL shipped (#1163-#1180)

`Qwen3-VL-4B-Instruct` end to end (`imp-cli --image`, several images per
request); text paths bit-identical. Pieces: encoder patch budget
(`runtime.vision_max_patches`, default 4096), M-RoPE in the text model
(`mrope_section` -> `rope.cu`), DeepStack taps after the first `n_deepstack`
layers. Gate: `make test-vision` (until 2026-08-11 the pipeline test ran
from no target). Detail:
[`plans/2026-07-31-qwen3-vl-vision.md`](plans/2026-07-31-qwen3-vl-vision.md).

| remaining | state |
|---|---|
| video | a project: decoder dependency (only `stb` vendored), frame axis, temporal M-RoPE, `<|video_pad|>` |
| a VL family with a different tower | port-sized (InternVL/Pixtral); the allowlist `vision_tower_supported()` names one layout. A second model on the SAME tower cost two gates (#1379, #1384) |
| ~~one image per request~~ | closed: several `image_url` parts in prompt order, unreadable image = 400 |

### 3-10, closed - one entry each

Full entries in
[`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md).

| # | gap | verdict | numbers / ref |
|---|---|---|---|
| 3 | model swap | SHIPPED, `server.model_swap` default on; in-flight generations drain, failed load restores the previous model | #1080, warm cache #956 |
| 4 | regex constraint | SHIPPED, `response_format: {"type":"regex"}` + `guided_regex` on the in-tree `RegexNfa`; every mask bypass closed (spec-ngram + graph-loop routers, thinking suppression, pooled state) | #1091 |
| 5 | speculation tree | BUILT AND MEASURED 2026-08-31, not a win: `speculative.mtp_tree_width` W=2 on Qwen3.8-27B-NVFP4, tree ceiling +6..+10 points top-2 over top-1, think traffic -6.4/-6.9% ungated, -0.8/-5.8% with `mtp_tree_margin`; default W=1. `token_recycling` neutral (-0.27%, 2026-08-19) | #1829, #1830, [`plans/2026-08-31-mtp-multicandidate-hybrid.md`](plans/2026-08-31-mtp-multicandidate-hybrid.md) |
| 6 | context spill below VRAM | DO NOT BUILD (2026-08-01): no reproducible trigger (4k/32k/128k all granted), spill lands on a 6.5x cliff (1531 vs 237 GB/s), each transfer blocks the host ~165 us; a prompt past the window is a typed refusal, eviction is client-visible (`usage.prompt_tokens_details.evicted_tokens`) | AUDIT B84, B36 |
| 7 | agentic quality vs competitors | CLOSED: `tools/analysis/agentic_compare.py`, 3 families x 4 budgets x 8-turn sessions; at 200-token budget imp keeps every contract, llama.cpp needs ~800 | [`BENCHMARKS.md`](BENCHMARKS.md), #1088 |
| 8 | GBNF | SHIPPED: nondeterministic pushdown simulator (`src/compute/gbnf_grammar.cpp`), mask build 333 -> 12 ms, uncompilable grammar = 400 | #1095 |
| 9 | `/v1/rerank` | SHIPPED: causal-LM cross-encoder, joint prefill-only scoring; vs llama.cpp top-1 3/3, median score delta 0.0014 | `make test-rerank` |
| 10 | external agent harnesses | SHIPPED: aider, Claude Code, OpenAI Agents SDK land real edits in `make test-agents-external`; pinned the streaming path handing reasoning to the user as the answer | OpenHands out (docker-in-docker) |

Explicitly NOT gaps: continuous batching, prefix caching, per-request LoRA,
embeddings, the three APIs, `/metrics`, suspend/resume, sampler surface (DRY,
mirostat, typical_p, logit_bias).

### Built-in live UI - shipped

`GET /` serves a single-page UI (`tools/imp-server/webui/index.html`,
embedded via `cmake/embed_webui.cmake`): live SSE render, one file, no build
step. Not on the `GOAL.md` surface commitment; anything beyond a thin client
belongs in an external front end.

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

Verdicts (detail: [`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md)):

| finding | verdict | numbers |
|---|---|---|
| "speculation does not pay on the hybrid" | RETRACTED 2026-08-18: two launch defects kept every GDN projection off the small-M batched GEMV (`ea547a53`) | MTP k=2 104.06 vs 86.21 spec-off (was 75.26 vs 84.47); kernel ms/emitted-token 11.35 -> 8.93 |
| k-sweep on the fixed build (2026-08-19) | k=1 wins, chain length is not a lever | k=0 86.03; k=1 **104.31 (+21.3%)**, 76.0% accept, 1.76 emitted/verify, 16.89 ms/verify; k=2 100.82; k=3 87.71 |
| where the per-row cost lives | the forward, not the recurrent state | `4.96 ms + 5.82 ms x rows`; `gemv_nvfp4_kpar_mb_fp16` = 65.1% of the k=3-k=1 growth, `gdn_scan_fused` 2.3% |
| accounting rule | a verify replaces a decode step only when accepted; on rejection it is additional | full weight sweep per verify regardless of emission - why chain-length saturation is a consequence |
| workload sensitivity | predictability, not the drafter, sets the number | prose MTP k=2 87.9 tok/s (58% accept, 2.3 emitted/verify); verbatim repeat 876.5 (98.3%, 36.6) |
| 14 hypotheses (drafter precision, MoE head, unfused chunk, repair forward, async loop, recurrent divergence, five decode kernels, chunk-side kernel choice, cross-process reproducibility, econ constant) | all measured dead or corrected | list in the record; econ break-even measured 2.42 (the k-aware default) |

```
[PROV: commit=3c3e9ac9 date=2026-08-19 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 2 alternating rounds
       cmd=`imp-server --think-budget 0 --set speculative.ngram=false
       --set speculative.mtp_k=0|1|2|3 --set speculative.mtp_econ_min_emit=0
       --set server.prefix_cache=false`; tokens from usage.completion_tokens,
       verifies from /metrics]
```

### MoE host offload - from "CPU cold experts" to a shipped LRU path

Origin: compute cold experts on the CPU (ktransformers shape) to reach
80B-120B on 32 GB. Measured its way OUT of that design: LRU expert cache +
streaming won, no AVX kernels, no `GOAL.md` amendment. Budget and campaign
tables verbatim in
[`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md);
verdicts (Qwen3-30B-A3B Q4_K_M unless marked):

| finding | verdict | numbers |
|---|---|---|
| host compute vs stream-into-VRAM (120B-A5B shape) | streaming wins ~3x | 14.0 ms/token static split + host compute vs 4.7-8.9 LRU streaming; host bandwidth 62.5 GB/s |
| static hot set | does NOT transfer between prompts | -15.2 / -29.5 points vs each prompt's oracle; temporal locality strong (median reuse distance 2 tokens), so LRU needs no calibration |
| `ExpertCache` at full offload | holds its hit rate | 88.7% hit; 24.98 tok/s vs 6.63 staging-only vs 311.24 resident |
| slot-indexed fused MoE kernels (#1370) | SHIPPED, 2.1x | the LRU pool IS the contiguous tensor the resident kernels index |
| fused MMVQ instead of dequant->GEMV | REFUTED | expert kernel time -43%, e2e 0%: the path is host-bound, launch count is the currency |
| graphs under offload | BLOCKED (`moe.allow_graphs_under_offload` capture aborts: host-read routing) | prefetch a layer ahead is unreachable, routing for layer N is known at layer N |
| NVFP4 offload | correct since 2026-08-13; before it "mandatory on-device" was unenforced and answered WRONG at 88.77 tok/s | resident 361.97 -> 384.03; 8-layer arm 44.54 correct; full offload 23.03 |
| `moe.pin_host_experts` (default off) | +14.8% pp512 (6/6), 4.4x load time | WSL2 cannot page-lock mmap; per-layer device staging 317.6 -> 790.8 tok/s only with pinning |
| `moe.staged_cutlass_prefill` | opt-in: +136% prefill, -36% decode after long prompts (unexplained) | pp512 663.2 -> 1563.9, tg 59.4 -> 37.7 |
| cache budget (`moe.expert_cache_budget_pct`, #1374) | 2.47x from a config value; floor is exact | 5% 10.51 tok/s, 15% (default) 20.99, 30% 30.51; below `3*top_k` slots/layer the cache retains nothing |
| final regime | transfer-bound, as modelled | H2D 150 GB at ~51 GB/s = 41% of step, kernels ~26%, launches ~24% |

Measurement rule: prefill varies ~15% between runs of the SAME arm, decode
moves with prefill length (cache warmth) and cold vs warm differ 2.4x - only
paired alternating rounds decide, and every number states which. Reproduce:
`tools/analysis/expert_cache_offload_sweep.sh` (MODE=ab).

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

Owner of every limitation is [`LIMITATIONS.md`](LIMITATIONS.md); this file
keeps the verdicts that came out of roadmap work (full text in
[`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md)).

| item | verdict | numbers |
|---|---|---|
| MTP on Nemotron-3.5 | head drafts (41.1% offline = 39.2/38.9% serving, the 0-9% gap was `run_ssm` never writing the snapshot slab, fixed 2026-08-20), verify chunk uneconomic: k=1 -51% with the guard off (176 vs 363 tok/s); shipped guard lands 258-341 on the 1+0.40k break-even | [`LIMITATIONS.md`](LIMITATIONS.md) "Speculative decoding is not universally profitable" |
| Qwen3.5-27B MXFP4 | blocked on a decodable checkpoint (no MXFP4 SafeTensors decode outside gpt-oss), not a bug | [`LIMITATIONS.md`](LIMITATIONS.md) |
| Gemma-4 Q4_K_M code-gen drift | no longer reproduces (2026-06-13, 2026-08-11); original file gone | fallback Q5_K_M or Q8_0 |
| native-FP8 weights decoding through the FP16 companion | CLOSED 2026-08-12, `FP8CacheEntry::native_source` drives the sidecar | +7.5% median decode, 27 pairs, order balanced |
| no dequant path for native FP8 | CLOSED: FP16 companion at load (sm_120 has no FP8 prefill GEMM) | Nemotron-3.5: 1698 MiB FP16 cache, init 24.4/32.6 GB, [`MODELS.md`](MODELS.md) |

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
  because the levers died: split-count boost +10.0% at 32k on Qwen3-8B,
  **-7.30%** on Qwen3-30B-A3B (#1270, reverted #1271); KV block 16 -> 32
  neutral; "latency-bound at 192 GB/s" retracted (629.6 GB/s at 32k at the
  same 16-17% occupancy); closing 629.6 -> 1127 GB/s is ~20% of the window.
  - Re-open on a mechanism, not the share - the share grows with context:

    | model | KV heads / g | layers | ctx | paged attention | ceiling if zero |
    |---|---|---|---|---|---|
    | Qwen3-8B-Q8_0 (dense) | 8 / 4 | 36 | 8k | 19.9% | 1.23x |
    | Qwen3-8B-Q8_0 (dense) | 8 / 4 | 36 | 32k | 43.9% | 1.76x |
    | Qwen3-30B-A3B-NVFP4 (MoE) | 4 / 8 | 48 | 8k | **29.1%** | 1.36x |
    | Qwen3-30B-A3B-NVFP4 (MoE) | 4 / 8 | 48 | 32k | **50.6%** | ~1.96x (1.92-2.01) |

    The share rises across the dense/MoE boundary because the non-attention
    half falls faster (-32.0%) than attention (-10.5%). Differential
    measurement (two runs differing by exactly 256 decode steps; kernel sum
    91-98% of the wall step, repeat pairs within 0.12 pp).

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
