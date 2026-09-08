# Roadmap

Single-author, single-GPU experiment: "roadmap" means current focus, not
schedule. Shipped work is in [`CHANGELOG.md`](../CHANGELOG.md), competitive
numbers in [`BENCHMARKS.md`](BENCHMARKS.md), limitations in
[`LIMITATIONS.md`](LIMITATIONS.md).

| House rule | |
|---|---|
| Row form | fact + number + decision + ref, one row each |
| Investigation | goes to `docs/plans/`, the PR body or `LIMITATIONS.md`, never into a table cell |
| Lifecycle | entries are closed, corrected or superseded in place, never deleted |
| Citations | `scripts/check_doc_citations.py` checks that a `path:line` EXISTS, not what it says (`weight_map.cpp:369` pointed eleven lines off and stayed green until 2026-08-31); a bare basename matching two files reports `AMBIGUOUS` and passes, so cite the path; a stale `git worktree` checkout makes every basename ambiguous at once |

Detail records: [`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md)
(serving and kernel rows, 08-25 .. 09-04),
[`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md)
(everything moved out on 2026-08-31).

## Direction

| | |
|---|---|
| Goal | fastest local engine for AI agent workloads on consumer Blackwell |
| Workload | 20k-100k+ tokens per session, context accumulates, streams run in parallel |
| Working regime | aggregate throughput at tens of concurrent streams; batch=1 is settled |
| Foundations (2026-05) | chunked-prefill FMHA + 256 MiB S-matrix, ctx ~4-6k to 32k+ (#453), multi-request decode batching (#454), StreamingLLM auto-enable on full KV (#455) |
| Serving ground (2026-08) | warm weight cache (#956), suspend-to-RAM (#954), request-order independence (#957), gemma-3 IMA fix (#959) |

## Standing position (2026-09-04)

| axis | state |
|---|---|
| GDN hybrid @32 vs vLLM | AHEAD. Qwen3.8-27B 1807.9 vs 1447.8 tok/s (+24.9%, vLLM 0.27.1), 1833.8 vs 1410.7 (+30.0%, 0.28.0), @8 573.0 vs 495.8 (+15.6%), @32 x 1082-token prompts 873.4 vs 497.8 (+75.5%), 3/3 each |
| dense NVFP4 @32 vs vLLM | PARITY OR AHEAD (2026-09-08). Qwen3-14B 38-token prompts 3948.9 vs 3817.6 (+3.4%, 2026-09-03); 982-token prompts 2491.2/2515.5/2482.6 vs 2485.9/2490.9/2497.7 (+0.2/+1.0/-0.6%, 2 of 3; was 1845.0 vs 2478.0 = 0.75x before the grouped FP8 decode attention, #1953) |
| batch=1 | 87.4 tok/s spec-off = 78% of the ~112 tok/s roofline (14.5 GB/token at 1628 GB/s resident); past it only through the MTP verify |
| raw-speed half of [`GOAL.md`](GOAL.md) | MET: batch=1 decode +13-48% vs llama.cpp on every hero (2026-07-12 re-sweep), MoE prefill leads vLLM single-seq, cross-engine PPL parity measured. Everything open below is the agentic half |
| admission at fan-out | `auto` resolves 28 vs a pinned 32 (630 vs 936): admission, not rotation, and 28 sustain full rate under continuous arrival |
| next engine-side post | the dense decode step at 32 streams after the grouped FP8 attention and the small-M launch work (ledger 2026-09-08, second row): the small-M GEMM class re-priced at 83% of resident bandwidth by launch durations (5.49 vs 4.56 ms floor per Qwen3-14B step BEFORE the PDL weight prefetch + q\|k\|v launch, which took ~0.3 ms of that), and the prefill forward at 80% of the FP4 peak (Open 2) |

Both engines measured on one client (`tools/analysis/vllm_conc_ab.sh`, 3
alternating trials, same checkpoint); the "1.58x gap" and "~1.08x pinned" of
2026-08-24/26 compared an imp number from this client against a vLLM number
from a 200-token-gen client. Rows with PROV in [`BENCHMARKS.md`](BENCHMARKS.md)
("re-measured on one client", runs 5-9).

## Open

Ranked by what an agent workload notices first.

| # | item | state | ref |
|---|---|---|---|
| 1 | launch-coupled idle @32 | RE-PRICED 2026-09-06, the headroom is ~2%, not ~8%: idle 7.2% of the wall over 873 decode steps, of which the sub-10-us gaps (every launch in the step) are 2.2%, host moments 10 us-1 ms 3.8%. A step carries 1478 device intervals (1296 kernels + 182 memcpy). Launch count and device time rank differently: `nvjet`+`splitKreduce` are 13.2% of the launches for 2.2% of the step, `[memcpy]` 12.3% for 1.8%, `swiglu`+`quantize`+`device_copy` 13.0% for 1.5%, while `gemm_nvfp4_smallm_v2`+`_pair` are 19.6% for 63%. Removing launches that overlap inside the graph buys nothing: #1793 took 65/step out and measured -0.9%. A fusion is worth building here only if it also removes traffic or math | `tools/analysis/nsys_gap_attribution.py --window` (launch census) |
| 2 | paced serving prefill, dense | CLOSED (2026-09-08). Anatomy from a debug-log step timeline (Qwen3-14B-NVFP4, 32 x 982-token prompts, wave 2): each capped step prefilled 1024 rows in 56-59 ms against a 38 ms standalone 1024-row forward (26.6k tok/s; 2048 rows 27.9k, 4096 rows 26.4k), the rest was the decode step and host turnaround the step carried; 29 such steps = 1.68 s ingest. SHIPPED in three parts: `runtime.prefill_cap_fairness` (the cap scales by W x waiting / decoding; W=1 22 steps = 1.52 s, +3.0..+4.2% aggregate; W=4 default a further +1.9..+2.5%), then `runtime.prefill_mixed_decode` (the decoders ride the ragged prefill forward as one-row members with one batched paged-decode attention launch per layer, no separate decode step): the 2048-row step 86 -> 65 ms, ingest 1.455 -> 1.175 s against vLLM's ~1.1 s, aggregate +6.6..+7.8%, TTFT p50 -20..-23%, ITL max 141 -> 80 ms, gaps over 100 ms 22 -> 0. What is left is the prefill forward itself (26-28k rows/s at 2048 rows, 80% of the FP4 peak) | ledger 2026-09-08, #1950, #1951, #1952 |
| 3 | long context | HALF CLOSED. Quest-class top-k page selection opt-in: Qwen3-8B 32k 160.3 -> 199.5 tok/s (+24.5%), NVFP4-KV 77k 74.3 -> 100.2, concurrent 3x25k +27%, spec verify on the sparse table +28.2%. Price on Qwen3.8-27B: NIAH 10/10 dense, 8/10 @8192, 5/10 @4096. Remaining: MLA models, prefill sparsity, StreamingLLM eviction (`src/compute/attention_paged_common.cuh:71`) as the only answer under KV-pool pressure | #1808, #1818, #1819, [plan](plans/2026-08-28-sparse-decode-attention.md) |
| 4 | speculation adapts per request | HALF CLOSED. Chain depth adapts (#1801), `mtp_k=auto` drafts single-stream 95.8 -> 141.6 tok/s (+48%). Remaining: drafter choice is global; the chain saturates near 2.5 accepted/verify and the multi-candidate tree measured no gain past it | #1809, #1811, [plan](plans/2026-08-31-mtp-multicandidate-hybrid.md) |
| 5 | recurrent-state paging | the lever for 32-way concurrency at LONG context, not the limiter at 32 slots. Evicted snapshots reach a pinned host tier since 2026-09-02 (turn-2 TTFT at 8 sessions -50%) | `server.recurrent_snapshot_host_mb` |
| 6 | `--calib` hurts at wide GQA | 14B RTN 9.9252 vs twin-calib 12.6016; the C x ABD interaction is 71% of the damage. Shipped rule: `--calib-groups BD` on wide-GQA, ABCD on narrow-GQA. The search still minimises a local proxy (per-group weight reconstruction) | finding (h) |
| 7 | quantizer refuses 3-D stacked experts | needs a per-model layout descriptor (gpt-oss interleaved vs Gemma-4 concatenated) plus per-expert bias in loader and MoE forward | finding (f) |
| 8 | no audio | BLOCKED ON A CHECKPOINT, not on work. The drop is stated since #1929 (`audio_config` warns, `WeightMap::skip_stats()` counts it), but no local checkpoint carries an encoder: Gemma-4-12B-NVFP4 has exactly one audio tensor, the 640->3840 `embed_audio.embedding_projection`, and its `audio_config` declares `architectures: null` with no layer, head or mel parameters. The same export has 0 `vision_tower.*` against the 26B's 355, so the quantisation dropped both towers. Needs a checkpoint with the tower, as `--mmproj` supplies for vision | [LIMITATIONS](LIMITATIONS.md#model-specific-blockers) |
| 9 | no video | a project: decoder dependency (only `stb` vendored), frame axis, temporal M-RoPE, `<\|video_pad\|>` | |
| 10 | one VL tower family | port-sized (InternVL/Pixtral); `vision_tower_supported()` names one layout, and a second model on the SAME tower cost two gates | #1379, #1384 |
| 11 | no KV tier below VRAM | DO NOT BUILD (2026-08-01): no reproducible trigger (4k/32k/128k all granted), the spill lands on a 6.5x cliff (1531 vs 237 GB/s) and each transfer blocks the host ~165 us; a prompt past the window is a typed refusal, eviction is client-visible | AUDIT B84, B36 |
| 12 | hybrid pp512 `gemm_cublas` hole | PRICED, parked: 24.8% of roofline at 21.5% share = 2-3% of hybrid pp512, and cashing it needs a row stride on every consumer of the packed GDN projection output or a deinterleave pass | ledger 2026-09-02 |

## Closed

| item | closed | verdict |
|---|---|---|
| concurrency scaling vs vLLM on the GDN hybrid | 2026-09-02 | imp leads, see Standing position. The 422 us/token wall delta of 2026-08-24 attributed as GEMM class 145, GPU idle 143 (15.9% vs 5.2%, 438k vs 200k launches/window), small classes ~135 |
| per-request priority | 2026-08-28 | `"priority"` body field (vLLM semantics, lower first, all three dialects) is the primary admission sort key, shortest-first-with-aging within a class, no preemption (`tests/test_scheduler.cpp`) |
| distributed tracing | 2026-09-02 | `X-Request-Id` echoed on every response, `server.otlp_endpoint` exports one OpenTelemetry SERVER span per generation request with queue / prefill / decode children, joined via W3C `traceparent`. Not in it: OTLP/gRPC, TLS, metrics/logs export |
| MTP acceptance gap vs the published 83% | 2026-08-31 | teacher-forced p1 83.5% avg, verify path 84.5% on the same prompts: acceptance is a property of the workload, not an implementation gap; the external 87% p1 belongs to the Qwen3-Next-80B head |
| vision beyond Gemma | #1163-#1180 | Qwen3-VL-4B end to end (`imp-cli --image`, several images per request), text paths bit-identical; patch budget `runtime.vision_max_patches`, M-RoPE, DeepStack taps, gate `make test-vision` ([plan](plans/2026-07-31-qwen3-vl-vision.md)). Video and a second tower stay open |
| Qwen3.8 port roadmap | [plan](plans/2026-08-24-qwen38-port.md) | CLOSED, including the "no-split GEMM ceiling" survey the 1.58x attribution leaned on (it holds for no-K-split designs only) |
| one server, one model | #1080 | `server.model_swap` default on: in-flight generations drain, a failed load restores the previous model |
| constrained decoding past JSON | #1091, #1095 | regex on the in-tree `RegexNfa` (every mask bypass closed) and GBNF via a nondeterministic pushdown simulator (mask build 333 -> 12 ms); an uncompilable grammar is a 400 |
| speculation tree | #1829, #1830 | BUILT AND MEASURED, not a win: `mtp_tree_width` W=2 tree ceiling +6..+10 points top-2, think traffic -0.8/-5.8% vs linear adaptive-k, default W=1; `token_recycling` neutral (-0.27%) |
| agentic quality vs competitors | #1088 | `tools/analysis/agentic_compare.py`, 3 families x 4 budgets x 8-turn sessions; at a 200-token budget imp keeps every contract, llama.cpp needs ~800 |
| `/v1/rerank` | `make test-rerank` | causal-LM cross-encoder, joint prefill-only scoring; vs llama.cpp top-1 3/3, median score delta 0.0014 |
| external agent harnesses | 2026-07 | aider, Claude Code and the OpenAI Agents SDK land real edits in `make test-agents-external`; OpenHands out (docker-in-docker) |
| built-in live UI | #1078 | `GET /` serves one embedded page (`tools/imp-server/webui/index.html`, no build step) showing only what the API returns; developed GPU-less against `webui/dev/mock_server.py` |

Explicitly NOT gaps: continuous batching, prefix caching, per-request LoRA,
embeddings, the three API dialects, `/metrics`, suspend/resume, sampler surface
(DRY, mirostat, typical_p, logit_bias).

## The 2026 bar (assessed 2026-08-21)

Checked against the tree (not recalled) and against
[vLLM Q3 2026](https://github.com/vllm-project/vllm/issues/48168),
[SGLang Q2 2026](https://github.com/sgl-project/sglang/issues/22949), the
[MLSys 2026 report](https://www.modular.com/blog/three-trends-from-mlsys-2026).
llama.cpp publishes no 2026 roadmap. What is not met is in Open above.

| Expectation | Where |
|---|---|
| Three API dialects natively, not via a shim | OpenAI chat/completions, Anthropic `/v1/messages`, OpenAI Responses; one shared SSE driver |
| Tool calling, gated by real harnesses | aider, Claude Code and the OpenAI Agents SDK drive imp in `make test-agents-external` |
| Constrained decoding past JSON | JSON Schema, regex, GBNF; an uncompilable constraint is a 400, not a free-text answer |
| Prompt caching with explicit breakpoints | prefix cache on by default, `cache_control` per breakpoint, content-salted so a different image is a different key |
| Embeddings and reranking in the same server | `/v1/embeddings`, `/v1/rerank`, validated against llama.cpp on the same GGUF |
| logprobs that agree with what was emitted | at temperature 0 the emitted token IS `top_logprobs[0]` (`tests/test_server_logprobs.py`) |
| Per-request adapter selection | `lora` body field, empty means the base model; one adapter active at a time, a request naming another waits for the in-flight ones (serialized, never batched together); the prefix cache is keyed by adapter |
| Latency observability, not just counters | `imp_ttft_seconds`, `imp_inter_token_seconds`, `imp_request_duration_seconds` histograms, plus `imp_queue_depth` and `imp_tokens_cached_total` |
| Auth, rate limiting, backpressure | `--api-key` (one key), per-client-IP rate limit, `max_concurrent`, 429 |
| Every setting reachable from a container, without a name per setting | `IMP_CONFIG` / `IMP_SET` bridge `--config` / `--set`, so a new config key needs no new env name; the 19 hand-written `IMP_*` names are frozen compatibility (#1823) |
| Continuous batching over a paged KV cache | default block n=16, geometry per configuration |
| Chunked prefill and graph-captured decode | CUDA graphs on both paths; gate asserts decode >= 1.3x, measures 2.28x |
| Speculative decoding that pays | n-gram, suffix index and a trained MTP head (+21.3% at `mtp_k=1`) |
| Quantized KV, including 4-bit | FP8 E4M3, INT8, INT4, NVFP4, and an NVFP4 attention-decode kernel |
| Graceful behaviour when the KV pool fills | StreamingLLM sink plus sliding window; growable pool commits as the card frees up |
| Weight formats a user actually has | GGUF K-quants and IQ, safetensors, NVFP4, MXFP4, native FP8 |
| Model classes, not one family | dense, MoE, MLA, Mamba2/GDN hybrids, vision-language, encoder-only |
| Operating it without a restart | model swap that drains in-flight work, `/admin/suspend` and `/admin/resume` |
| Cold start that is not a full reload | warm on-disk weight cache; vLLM still carries cold start as an open Q3 roadmap issue |
| Reproducibility as a product property | `runtime.deterministic` covers MoE routing atomics, sampling races and GEMM; see [`determinism.md`](determinism.md) |

## Lever ledger

One row per lever: verdict, headline number, ref. Measurement narrative in
[`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md).

### Serving throughput (batch=32 aggregate, alternating pairs, fresh server per arm)

| lever | verdict | number | ref |
|---|---|---|---|
| small-M mxf4nvf4 GEMM v2 | SHIPPED default-on | +16.0% @32 (992.5 -> 1151.7), +36.0% @8 | #1766, Marlin sidecar #1764 unmerged |
| row-block batched RMSNorm | SHIPPED | +6.8% | #1769 |
| shared-activation quantize | SHIPPED | +4.6% | #1771 |
| producer-side quantize fusion | SHIPPED | +2.6% (1160.4 -> 1191.0, 3/3) | #1773 |
| GDN-out quantize fusion | NEUTRAL, closed unmerged | +0.4% over 6 trials | #1774 |
| gate\|up and in\|z sibling-pair launch | SHIPPED default-on | +1.7% (1713.3 -> 1742.0, 3/3), -112 launches/step | `gemm.nvfp4_smallm_pair` |
| batched post-step sampling chain | SHIPPED | +2.2% (1740.9 -> 1779.4, 3/3), ~124 launches/step and 6.6% of wall removed | 2026-08-27 |
| small-M GEMM PDL weight prefetch + q\|k\|v sibling launch (dense, Qwen3-14B-NVFP4) | SHIPPED | Anatomy (nsys graph-node trace, 32 x 982-token prompts): the class ran 5.49 ms per decode step against a 4.56 ms resident-bandwidth floor (83%, not the 67% of the 2026-09-03 attribution); per launch q/o 12.9 us vs 9.0 floor (80 CTAs, ramp-bound), k and v 3.9 + 1.5 us reduce each (16 tiles, 10 stripes) vs 1.8, gate\|up pair 66.4 vs 61.6, down 34.2 vs 30.8. Isolated L2-defeating survey: q/o 1021-1089 GB/s, k/v 189, a 7168-row q\|k\|v 1100-1138, pair 1431, down 1401 (`IMP_SMALLM_V2_SHAPES=1 test-quant`). Shipped: every CTA triggers dependents at start and streams its weight ring before `griddepcontrol.wait` (producer W0, consumer warp 0 W1..W5 with a track-only mbarrier arrive, so a late CTA sees stage 0 as early as before), q\|k\|v as one 112-CTA launch (`gemm_nvfp4_smallm_v2_multi_a4`, bit-identical to the singles at stripes=1), the producer chain kept programmatic (row-block norm trigger at start, swiglu/decode quantize registered, residual copy as a kernel node). 32 x 982 tokens, 4 alternating trials, wave medians: 2438.4/2366.8/2451.1/2402.9 -> 2511.2/2483.0/2485.8/2473.0 tok/s (+3.0/+4.9/+1.4/+2.9%), ITL p50 9.0-9.3 -> 8.7-8.9 ms, TTFT equal; q\|k\|v launch 16.8 us in situ vs 12.9 + 2 x 5.4; launches per step 905 -> 756. Hybrid sanity on Qwen3.8-27B-NVFP4-vllm (`two_image_conc_ab.sh`, 32 x 38-token prompts, 2 trials): 1874.3 -> 1917.8, 1843.0 -> 1886.7 (+2.3/+2.4%). Isolated same-kernel loops read the pair +2 us per launch (the next launch's early CTAs share the tail wave's bandwidth); the in-situ chain is the judge. Kernel durations under nsys are no longer additive here (early-launched kernels spin in the wait) | #1954, `tools/analysis/prefill_cap_conc_ab.sh` two-image form |
| batched-decode LM head on the small-M kernel (`gemm.nvfp4_lm_head_smallm`) | SHIPPED default-on | The n <= 32 LM head ran on the CUTLASS 128-row tile: Qwen3-14B-NVFP4 at 32 streams 365 us per decode step (grid 1 x 1187) against a 268 us resident-bandwidth floor for the 437 MB weight, plus a 12.5 us SfAtom activation quantize. The small-M kernel at one stripe (2374 CTAs) reads the same weight at 1393 GB/s isolated (314 us) and writes FP32 logits from its accumulators (`gemm_nvfp4_smallm_v2_a4_f32`, rounds to the FP16 kernel bit for bit); the final norm fuses the activation quantize (`rmsnorm_nvfp4`). One image, knob on vs off, 3 alternating trials x 3 waves, wave medians: Qwen3-14B 32 x 982 tokens 2536.7/2517.6/2536.8 -> 2567.9/2573.7/2541.7 (+1.2/+2.2/+0.2%), ITL p50 8.5-8.7 -> 8.4-8.5 ms; Qwen3.8-27B-NVFP4-vllm 32 x 40 tokens 1881.4/1897.3/1835.5 -> 1892.5/1905.4/1882.6 (+0.6/+0.4/+2.6%). Same W4A4 activation family as the CUTLASS path; spec-verify and n == 1 untouched | #1955, `prefill_cap_conc_ab.sh` |
| eager PDL chain: every tiny decode kernel (residual add, copy, norms, quantize, swiglu, rope) triggers its dependents BEFORE its own `griddepcontrol.wait`, so a GEMM's CTAs land during the previous GEMM instead of during the norm in front of it | REFUTED (2026-09-08) | Motivation from the #1954 trace: q\|k\|v end -> attention start 6.5 us of serialized tiny kernels, o end -> gate\|up start +2.3 us and down end -> q\|k\|v start +2.2 us (the dependents launched only after the predecessor GEMM completed, so the ring prefetch of #1954 had a ~2 us window, not the whole predecessor). Two images (main after #1955 vs the chain), 3 alternating trials: Qwen3-14B-NVFP4 32 x 982 tokens 2576.4/2588.2/2588.2 -> 2547.1/2538.7/2523.4 (-1.1/-1.9/-2.5%, ITL p50 8.4 -> 8.6 ms); Qwen3.8-27B 32 x 38 tokens 1937.7/1960.4 -> 1916.4/1868.1 (-1.1/-4.7%). Three to four grids resident and waiting at once, with the GEMM CTAs (90 KB smem) prefetching while the predecessor GEMM still streams, costs more than the ramp window returns; prefetched bytes are zero-sum once DRAM is saturated. The one-level form (trigger after the wait, #1954) stays | `chain_gaps.py` (scratch), `prefill_cap_conc_ab.sh` two-image form |
| PDL registration of the three plain-launched decode-chain kernels (`rope_forward_kernel`, `elementwise_add_store_fp16_kernel`, `write_kv_cache_fp8_fused_kernel`), wait then trigger like their neighbours | SHIPPED | Two images (main after #1955 vs branch), 3 alternating trials x 3 waves, IGNORE_EOS: Qwen3.8-27B-NVFP4-vllm 32 x 40 tokens 1903.4/1952.8/1929.8 -> 1954.4/1960.5/1962.4 (+2.7/+0.4/+1.7%), ITL p50 15.1-15.5 -> 15.0-15.1 ms; Qwen3-14B-NVFP4 32 x 982 tokens 2576.9/2585.0/2588.8 -> 2579.1/2581.3/2565.4 (+0.1/-0.1/-0.9%, neutral). `DegenerationTest.*` 5/5 on the branch | #1956 |
| batched residual accumulate (beta=1) | REFUTED | -0.9% median, 3/3 negative; residual adds already overlap in the graph | `gemm.nvfp4_residual_beta1` |
| batched ban + penalty sweep | SHIPPED | 1766.9 -> 1774.9 (3/3), 2 launches per row per step -> 1 sweep per step, idle 14.9 -> 13.6% | 2026-08-31 |
| penalties walk the history, not the vocabulary | SHIPPED | per launch at 32 rows: 300-token history 197.1 -> 10.7-23.3 us, 4096-token 2659.3 -> 18.6-34.3 us; @32 3/3 positive, logits bit-identical | `sampling_penalties.cu`, 2026-09-02 |
| conv1d decode, float4 state and one weight load | SHIPPED | 9.61 -> 4.97 us per launch (273.4 -> 144.2 ms of a 10.4 s window), 3/3 positive, bit-exact | `src/compute/ssm.cu`, 2026-09-02 |
| PDL device half (`griddepcontrol`) | SHIPPED default-on | @32 3/3 positive (+1.3% median), M=1 +1.7% median, idle 13.6 -> 10.8% | `runtime.no_pdl` |
| PDL registration for the batched LM-head GEMV (`gemv_nvfp4_kpar_mb_fp32_kernel`, instrumented since #1833, never registered) | SHIPPED | @32 +1.49 / +1.20 / -0.18 % (mean +0.84 %) | AUDIT_arch_2026 A2-2, #1923 |
| `pdl_trigger()` after the KV walk in the 24 remaining paged decode / split-K kernels | SHIPPED, NEUTRAL | @32 +0.84 / -0.40 / -1.23 %, M=1 -0.02 % (Qwen3.8-27B) | AUDIT_arch_2026 A2-5, #1923 |
| NVFP4 K-par GEMV `__launch_bounds__(128)` without the 12-CTA min-blocks term (ptxas 40 -> 42 registers, 12 -> 10 CTAs/SM; the `(128, 16)` the #10 lead used is infeasible on the 1536-thread SM and compiles to the same SASS) | SHIPPED | @32 +1.19 / +2.12 / +0.08 %; M=1 Qwen3-8B-Q8_0 +1.96 / +1.97 / +1.71 %, Qwen3-14B-NVFP4 +0.65 / +0.61 / +0.79 % | #1923 |
| dp4a GEMV MaxL1 carveout restored (lost with the PDL registration in #1833) | SHIPPED | M=1 Qwen3-8B-Q8_0 on the source-precision path (`diagnostics.no_nvfp4_decode_cache`) 145.3 -> 157.3 tok/s, +8.24 / +8.20 / +8.25 % | AUDIT_arch_2026 A1-3, #1923 |
| single-sequence GDN scan on the SPLIT=2 instance (no spill, 255 -> 180 registers) | SHIPPED | M=1 Qwen3.8-27B +0.12 / +0.20 / +0.12 % | AUDIT_arch_2026 A2-3, #1923 |
| one-H2D decode-step staging | NEUTRAL, closed unmerged | 2/3 pairs negative (-0.2 / -1.1 / +0.6%) | branch `perf/decode-step-staging` |
| graph prewarm | RETIRED as throughput, SHIPPED as latency | wave-1 aggregate unmoved (629-650 vs 627), wave-1 p50 -3-12% | #1761, `runtime.graph_prewarm` |
| batch=1 async-loop recapture fix | ITL fix, not a lever | FRESH captures 128 -> 7 per ~200-token burst, +0.2% throughput | 2026-08-27 |
| host turnaround | ATTRIBUTED, closed as a defect class | per step: build 63-82 us, fwd-enqueue 34-47, distribute 7, schedule 1-2; outside-step 1.2-1.5 ms is the paced serial prefill | `diagnostics.step_timing` |
| prefill concurrent with decode | NEUTRAL both shapes, default-off | short prompts 1771.3 vs 1777.7, 1000-token ingest 789.7 vs 790.6; no green-context SM partitioning on sm_120 | `runtime.prefill_overlap`, [plan](plans/2026-08-27-prefill-decode-overlap.md) |
| ragged cross-sequence prefill | SHIPPED default-on | +6.2% (977.3 -> 1038.2, 12/12 waves), TTFT p50 4.11 -> 2.55 s | `runtime.prefill_batch`, #1780 |
| ragged members charged their real rows, per-member chunk-parallel GDN scan | SHIPPED | 1094-token prompts 943.7 -> 1058.0 tok/s (+12.1%), ITL p95 46.2 -> 19.9 ms, gaps > 100 ms 349 -> 224 | 2026-09-03 |
| `prefill_chunk_decode_cap` 2048 | MEASURED, default stays 1024 | +4.4% hybrid / +8.6% dense at ~1k-token prompts, but the decoders' ITL during a foreign ingest +70% / +63% | 2026-09-03 |
| BF16 GDN state | SHIPPED default-on | scan 2.04x isolated, +12.5% KV-pinned (1210.5 -> 1362.0), +7.7% pure defaults, PPL +0.21% | `gdn.state_bf16`, #1778 |
| growable KV under aggregate pressure | SHIPPED, default-on since 2026-09-07 (row above) | 32 x 8k/512: wall 86.0 -> 65.2 s (-24%), pool 2046 -> 6483 blocks | `kv_cache.growable`, #1794 |
| auto `max_batch_size` on hybrids | FIXED | resolver priced hybrid KV 4x too high (224 -> 630 @32); `max_seq_len: auto` was VRAM-blind on packed-4-bit KV | 2026-08-25 |
| burst serving fixes | SHIPPED | HTTP pool sized to streams, token-charged prefill budget, id-based rotor: 1047-1073 tok/s on every one of 4 waves | #1762, #1758 (deferred delivery +4-5%) |
| adaptive MTP chain depth (M=1) | SHIPPED default-on | `mtp_k=2` + `ngram=false`: think chats 111.1-113.3 vs 106.3-108.0 at k=1, draft-rich 158.1 (+31% vs k=1) | `speculative.mtp_adaptive_k`, #1801 |
| `mtp_k=auto` as the default (M=1) | SHIPPED default-on | single stream 95.8 -> 141.6 tok/s (+48%), 3 alternating rounds, degen 50/0; declines for concurrent serving | #1809, #1811 |
| sparse decode at concurrent long context | SHIPPED opt-in | 3 streams x 25k: 155.6 -> 197.7 tok/s (+27%), metadata one batched launch per forward | `attention.sparse_topk_tokens`, #1808 |

### Decode attention kernels

| lever | verdict | number | ref |
|---|---|---|---|
| FP8 paged decode, four tokens per warp iteration | SHIPPED default-on | microbench 32 x 1100 209.1 -> 92.4 us (345 -> 780 GB/s with the paired e4m3 conversion), 32 x 4096 716.9 -> 332.4; serving @32 982-token prompts +25.2%, 38-token +13.9%; vs vLLM 38-token 3948.9 vs 3817.6 (+3.4%, was -7.6%) | `attention.paged_fp8_multitok`, #1872, #1875 |
| FP8 lane-per-token QK variant, and an 8-token instance | REFUTED | -6.7% at 1100 tokens but +32% at 300 (half the lanes idle); 8-token 94.2 vs 91.7 us (registers) | #1875, not in tree |
| FP8 paged decode, 16 lanes per KV row (8 bytes per lane, 4 shuffles per dot) and the Q heads of a KV head grouped per CTA (5/4/3/2 dividing the GQA ratio), q slices re-read from shared memory so the HPC=5 instance holds 128 registers (two CTAs per SM), the next row group's K and V rows in flight during the current reduce | SHIPPED default-on | microbench 32 x 1100: 40/8 heads 95.2 -> 56.4 us (757 -> 1278 GB/s), 32/8 83.5 -> 55.6; 32 x 4096 334.6 -> 186.2 (1442 GB/s); the F16 four-head kernel reads 1368 GB/s on the same 32 x 1100 shape. Qwen3-14B-NVFP4 @32 x 982-token prompts, two images, 3 trials x 3 waves: 2192.7/2193.6/2203.2 -> 2496.6/2480.4/2482.9 tok/s (+13.9/+13.1/+12.7%), ITL p50 10.5 -> 8.7 ms, TTFT p90 unchanged; 38-token prompts 4176.5/4205.6/4212.2 -> 4327.6/4279.4/4322.7 (+3.6/+1.8/+2.6%). Refuted on the way: 8 lanes x 16 bytes per row at HPC 5 needs 180-217 registers (one CTA per SM, 60-66 us), q in registers 66.4 us, V issued after the softmax 63.6, TOK=4 at 16 lanes 63.9; ncu on the register-resident form: 12 active warps per SM, 0.68 eligible per scheduler, DRAM 56% | `attention_paged_fp8_multitok_gqa.cu`, #1953 |
| NVFP4 paged decode, four tokens per warp iteration, split-K target 4 CTAs/SM | SHIPPED default-on | 32 x 1100 123.3 -> 90.0 us, 1 x 77k 293.8 -> 209.6 (-29%); e2e 32k +6.6%, 64k +14.1% | `attention.paged_nvfp4_multitok`, #1876 |
| the same split-K twin for FP8 | REFUTED | its split-K route already runs the cp.async-pipelined scalar kernel at 800-900 GB/s (77k 198.2 vs 200.4 us) | #1876 |
| NVFP4 Q heads grouped per CTA, each K/V row converted once | SHIPPED default-on | 24/4 HD=256 1 x 77k 214.2 -> 177.3 us, 32 x 1100 92.9 -> 68.6; e2e 32k +3.0%, 64k +4.2%, @32 +1.1..1.3% | #1886 |
| NVFP4 group scale once per (token, head), half2 FMA dot over raw E2M1 pairs | SHIPPED default-on | 1 x 77k 177.3 -> 144.5 us (614 GB/s), 1 x 32k 74.4 -> 61.8; e2e 32k +1.6%, 64k +3.2%; cumulative over the day 32k +4.7%, 64k +7.2% | #1887 |
| F16 cluster (DSMEM) GQA route | REMOVED | it was reachable only outside split-K, i.e. exactly in serving: 32 x 1100 2133 -> 317.5 us (6.7x); gemma-3-12b @32 186.4 -> 229.9 tok/s (+23.3%) | #1877, #1878 |
| F16 paged decode, four tokens per warp iteration and up to four Q heads per CTA | SHIPPED default-on | 32/8 HD=128 314.6 -> 98.0 us, 16/8 HD=256 665.0 -> 177.9; @32 Llama-3.2-3B +48.3%, Phi-4 FP16-KV +64.9%, gemma-3-12b +15.8% | `attention.paged_f16_multitok`, #1880 |
| F16 multitok on the split-K route (single stream, long context) | SHIPPED default-on | 32/8 HD=128 32k 197.1 -> 109.3 us (1228 GB/s); Llama-3.2-3B 8k +15.3%, 32k +29.5%, 64k +38.1% | #1882 |
| F16 split-K CTA target 4 per SM for the four-head instance | REFUTED e2e | microbench said yes (32k 98.6 -> 92.4 us), Phi-4 read 8k -4.2% / 32k -1.7%: twice the splits doubles the partials the reduce kernel moves | branch `perf/f16-splitk-target4`, #1885 |
| KV-pressure heuristic counted reclaimable prefix-cache blocks as used, graphs demoted one-way | FIXED | waves 1-3 fell 2387 -> 1443-1485 tok/s (ITL p50 7.8 -> 16.5 ms); after the fix 2392-2450 on every wave with the prefix cache on | #1879 |
| `imp-cli --bench` on F16 KV models at pp >= ~2.3k read 0 tok/s | FIXED | the bench pinned `max_seq_len = pp + tg + 256`, the StreamingLLM valve fired on the bench prompt; headroom is now max(256, 12.5%) | #1883 |

### Prefill kernels

| lever | verdict | number | ref |
|---|---|---|---|
| prefill kernel utilization (the open "%-of-peak" question) | MEASURED | dense NVFP4 GEMM 79.8% of measured FP4 peak @pp4096, 64.9% @pp512, so not the hole; the holes are `gemm_cublas` on the hybrid @pp512 (24.8% of roofline at 21.5% share) and `attn_fa2` @pp4096 dense (22.8% at 21.9%) | roofline run `1d5b9230_20260831_180644` |
| dense FA2 at 2 CTAs/SM | SHIPPED default-on | pp4096 FA2 kernel sum -8.6..-13.0% (3/3), pp +1.8..+6.9%, PPL bit-identical; needs the TWOSLOT tile (69632 -> 34816 B) and a wrapper kernel for `__launch_bounds__(256, 2)` | `attention.fa2_dense_2cta` |
| HD=256 FA2 at 2 CTAs/SM (Bkv=32) | BUILT, opt-in on the PPL trade | FA2 kernel sum -11.2% (3/3), pp512 flat, pp4096 +0.1..0.4%; PPL 4.6283 -> 4.6529 (+0.53%) from the doubled f16 O rescale count | `attention.fa2_hd256_bkv` |
| causal FA2 CTA order, heaviest q-tile first | SHIPPED default-on, small | FA2 sum -1.2% (14B) / -2.2% (27B), 3/3 each, output byte-identical | `attention.fa2_heavy_first` |
| deeper in-CTA FA2 pipelining | PRICED OUT | the shipped instance is tensor-pipe bound (61% of peak sustained, `math_pipe_throttle` top stall, DRAM 60 GB/s) and has no registers left at 128 for a second S tile | 2026-09-02 |
| stream-K scheduler on the CUTLASS NVFP4 prefill GEMM | SHIPPED default-on | pp512 kernel sum -3.9..-6.3% (3/3), pp4096 flat (no shape qualifies), output bit-identical; dispatch only at >= 1 wave with a last wave <= half full | `gemm.nvfp4_cutlass_streamk` |
| `gemm_grouped_nvfp4` (MoE prefill, 53% of the hybrid pp512 window) | REFUTED twice, structural | both designs land at ~60% of the 134 MB weight floor: v2 small-M grouped pp512 +3.5% / pp1024 +15% worse, multi-tile CTA mt64 flat, mt128 +2.4-4.1%, mt32 +4.0-4.5% | 2026-09-01 |
| FP8 prefill for the GDN projections | REFUTED e2e | cuBLASLt 2.0-3.6x isolated but 6/6 e2e pairs negative; the SSM_IN failure was root-caused (FP16 output held `out / row_scale` before the weight scales folded in, inf on small-absmax rows) | branches `perf/fp8-ssm-prefill*`, [plan](plans/2026-08-31-fp8-ssm-prefill.md) |
| MXFP4 prefill attention family (`attention_fmha_mxfp4_sm120.cu` 2021 LOC + `attention_mxfp4_prefill.cu` 532 LOC) | REFUTED, opt-in only (`attention.mxfp4 = "always"`, `attention.mxfp4_paged_kv`) | residual noise compounds with context, +10% NLL at 9k (#868, idea #846); no default configuration reaches it. Re-measure trigger: a change to the sm_120 FP4 MMA path (CUTLASS/PTX); absent that, the ~2550 LOC are the next dead-code sweep's largest item | CHANGELOG "NVFP4-attention research knobs", AUDIT_arch_2026 A1-9 |
| `gemm_cublas` alpha/beta tails on the hybrid @pp512 | PRICED, not built | ~90 us of split-K tails plus a better tile per 5 layers of the 3.3 ms window = 2-3% of hybrid pp512, for a row stride on every consumer of the packed output | 2026-09-02 |

### GDN chunk-parallel prefill scan

Fused scan was 42% of the Qwen3.6-35B pp512 wall (658 us/layer, grid (32,1,1)),
a class the 120-launch ncu roofline window missed. Class kernel sums under
nsys, alternating pairs, vs the fused scan unless stated.

| step | verdict | number | ref |
|---|---|---|---|
| chunk-parallel scan (WY split on state linearity) | SHIPPED | pp512 -32%, pp4096 -47%, e2e 12949 -> 18851 tok/s (+45.6%); PPL +0.03%, costs a 42 MiB engine-lifetime workspace | `gdn.chunkpar_scan`, #1847 |
| state pass (kernel 2) on tensor cores, 3xTF32 on the state-feeding GEMMs | SHIPPED | pp4096 -65%, e2e +69/+81%; plain tf32 everywhere costs PPL +0.13% and is refused | #1848 |
| factor kernel (kernel 1) on tensor cores | SHIPPED | pp4096 -74%, e2e +95% (+16% over #1848); plain tf32 on P@W refuted (cancellation in Qeff) | #1849 |
| blockwise triangular solve | SHIPPED | K1 per CTA 75 -> 49.5 us, pp4096 -79%, e2e +109% vs fused (+8% over #1849); 8 barriers per chunk instead of 128 | #1850 |
| kernel 2 at 8 warps, pipelined staging, strip sized per head count, XOR-swizzled factor tiles | SHIPPED | 27B kernel 2 -31% / kernel 1 -13%, e2e +9.6/+8.0%; 35B -32/-12%, e2e +5.3/+3.6% | #1851, `gdn.chunkpar_strip` |
| state-feeding GEMMs on 3xFP16 m16n8k16 | SHIPPED | 27B K1 -22% / K2 -18%, e2e +4.9/+5.0%; 35B -21/-19%, e2e +5.1/+3.4%; state diff vs fused 9.5e-7 -> 1.3e-6 | #1852 |
| kernel 2 at 2 CTAs/SM, plain tf32 on u_eff, swizzle on K1's T/P tiles, Y_A on fp16 k16 | REFUTED | flat (two CTAs share one tensor pipe); state diff 8.5e-5 fails the 1e-4 bound; K1 +10%; +5% | #1852 |
| ragged chunk-parallel scan for serving prefill | NOT BUILT | ragged fused scans are 2.6% of a 32 x ~1000-token burst window, because `prefill_chunk_decode_cap=1024` already makes most forwards single-sequence | 2026-09-02 |

Quality judge for this campaign: the Qwen3.6-35B deterministic PPL is no judge
below ~0.5% (6.8122..6.8493 across fp32-equivalent variants, MoE routing
flips). Qwen3.8-27B deterministic PPL and the per-block divergence a change
ADDS (`diagnostics.dump_hidden_dir` + `tools/analysis/layer_ab_diff.py`) are.

### Server and latency

| lever | verdict | number | ref |
|---|---|---|---|
| recurrent-snapshot host tier | SHIPPED default-on | 8 sessions x 3 turns: turn-2 TTFT 324/322 -> 163/145 ms (-50%), wall 15.9 -> 13.5 s (-16%), 32 of 40 restores from host; 2 GiB pinned = 25 slabs on the 27B, not VRAM | `server.recurrent_snapshot_host_mb` |
| growable KV pool: default on, grows before the prefix cache is reclaimed, growth capped at the allocator headroom | SHIPPED default-on | reclaim-first emptied the cache at the planned commit while GiBs sat free, and the ceiling (6726 blocks = 3783 MiB on Qwen3.8-27B-NVFP4) was sized before the 3522 MiB library reserve, so an uncapped growth would have spilled. Same 8 x 3 x 3.8k workload as the row below, alternating arms on one binary (`kv_cache.growable=false` vs default), 2 pairs: turn-2 restores at the turn-1 boundary 4/8, 6/8 -> 8/8, 8/8; TTFT p50 7289, 6286 -> 4997, 5274 ms; wall 10.22, 9.78 -> 5.63, 6.07 s; `imp_prefix_cache_evictions_total` 1614, 1139 -> 34, 34; pool 2301 -> 3595 blocks in 2 growths, 1991 MiB free after (headroom 1629), no cap hit | `KVCacheManager::allocate_block_ref_with_eviction`, `KVCache::try_grow_to` |
| recurrent-snapshot save falls back to the host tier | SHIPPED (#1937) | the 3 device slabs are held for whole generations once sessions outnumber them, and `save()` returned false with 25 host slabs free. Qwen3.8-27B-NVFP4, defaults, 8 sessions x 3 turns x 3.8k new tokens: turn-2 restores at the turn-1 boundary 0/8 -> 6/8 (before: 5 at the turn-0 boundary, 3 none), turn-2 TTFT p50/p90 6822/10628 -> 4611/8659 ms, wall 11.13 -> 9.81 s. The 2 remaining misses are KV chain evictions at 73.6k pool tokens vs 8 x 11.4k live | `src/memory/recurrent_snapshot_store.cpp` `save_to_host_` |
| OpenTelemetry span export | SHIPPED, off by default | one SERVER span per generation request from the single accounting point, queue / prefill / decode children, OTLP/HTTP JSON, batches per second or 256 spans | `server.otlp_endpoint`, #1855 |
| reasoning scan hold released on the first word | SHIPPED | chat thinking-off TTFT 97-105 -> 32-62 ms, 1116-token prompt 195-229 -> 130-146, completions 116-147 -> 51-64; the hold of 8 tokens never protected a real chain of thought, the tool path keeps 256 | #1894 |
| `cudaGraphExecUpdate` on the parked exec instead of instantiate per request | SHIPPED | 31 of 34 setups updated in place in 0.1 ms, wall median 545 -> 532 ms, max inter-token gap 25-35 -> 23-26 ms, output hashes identical; the 44 ms instantiate in the #1894 trace was CUPTI-inflated | #1895 |
| short-prompt TTFT, attributed by phase (2026-09-08) | SHIPPED the one software term; the rest is priced | Qwen3.8-27B-NVFP4, one stream, 35-token prompt, 12 waves x 4 tokens, server-side `ttft=` on the new completions log line: admission 0.1 ms; submit -> first prefill chunk launched 10-43; chunk 1 + snapshot save + sync 0.5-12; chunk 2 (3 tokens) + token 1 18-46; async-loop capture + exec update 2; floor 36-38, high mode to 113. The variable terms are the host launch phases of the eager prefill (prefill graph off on this model: `largest NVFP4 weight 2425 MiB > 512 MiB cap`). REFUTED: a write-through host twin for the snapshot store (built, 7/7 tests, measured neutral: medians 51-52 vs 58-62 against main); a keep-alive spin kernel (second context, floor 48 -> 53, high mode stays). SHIPPED: `server.snapshot_min_prompt_tokens` = 256, no snapshot and no 32 + 3 prefill split under it: client TTFT floor 48 -> 35 ms, medians 63/54 -> 36/48 (2 x 12 waves, alternating), server-side floor 36-38 -> 22.5-22.7; a 977-token prompt still snapshots at 976. Next lever: prefill capture under FP8 KV (the resolver turns the prefill graph off because the quantized append syncs its absmax to the host per chunk; `diagnostics.prefill_graph_ignore_dequant_cap` changes nothing here, server floor 22.6 on both arms) plus length-bucketed prefill graphs (the runner is keyed by the exact chunk length, so varying short prompts recapture) | #1948 |
| remaining burst-boundary gap (22-26 ms) | CLOSED BY MEASUREMENT (2026-09-08) | one stream, 300 forced tokens, 3 waves, `diagnostics.spec_trace`: 57 rearm boundaries per 3 requests at max ITL 12-13 ms against a 10.7 ms step (the miss-only wave). What still exceeds 14 ms is the verify step itself (a cached bucket 28-30 ms, the first eager use of a bucket 28-29, the capturing use 61-113 of which capture 4.4-5.7 + instantiate 8.8-9.3 + launch 0.6-0.7 ms, now on the `[spec-capture]` line) and the clock ramp after idle (19.5 ms per token for the first 2-10 tokens, n-gram off too). Shipped alongside: the last burst of every request recaptured because a 32-step cap against 19 tokens left refused the rearm (3 recaptures per 3 requests -> 0, launch seeds identical through the 18 launches before it) | #1895, #1947 |
| prefill cap scaled by waiting / decoding (`runtime.prefill_cap_fairness`, default on) | SHIPPED | Qwen3-14B-NVFP4, 32 x 982-token prompts, 300 forced tokens, 3 trials x 3 waves alternating, wave medians: aggregate +3.7 / +4.2 / +3.0%, TTFT p50 1085/1063/1009 -> 839/827/830 ms, p90 1743/1734/1663 -> 1541/1516/1527, ITL p95 13.2 -> 12.8-13.1, ITL max 106-107 -> 139-141 (the 2048-row step), gaps > 100 ms 3 -> 4-9 of 9568 tokens; protection scenario (31 x 87-token streams + a 4436-token ingest at 2 s): the short streams' ITL during the ingest p95 62.5/62.5/62.6 -> 62.3/62.5/62.7 ms, max 65 -> 65, ingest TTFT 292/295/289 -> 290/290/291; Qwen3.8-27B-NVFP4 burst +0.3 / +1.2 / +1.0%, TTFT p90 2972/2966/3038 -> 2810/2806/2812, ITL max 120-133 -> 189-191 (pool-bound tail of 7.6 s in both arms at the 2387-block pin) | #1950, `tools/analysis/prefill_cap_conc_ab.sh` |
| the fairness weight W (`runtime.prefill_cap_fairness` = waiters per decoder), sweep 1 / 2 / 4 | SHIPPED default 4 | Qwen3-14B-NVFP4, 32 x 982-token burst, three images, 3 trials x 3 waves alternating, wave medians: aggregate W=2 +1.6 / +1.3 / +1.5%, W=4 +2.5 / +2.2 / +1.9% over W=1 (1931 / 1944 / 1939 tok/s, a foreign tenant at 7-9% on the card), TTFT p90 1600/1591/1588 -> 1507/1504/1512 -> 1456/1455/1465 ms, TTFT max 1745-1760 -> 1675-1678 -> 1619-1632, ITL max 144-147 on every arm, gaps > 100 ms 9-10 -> 22 of 9568; protection scenario (31 x 87-token streams + a 4436-token ingest): ITL during the ingest p95 64.7-65.5 / 65.5-66.2 / 65.7-66.0, max 67 / 68-69 / 68, ingest TTFT 300-305 on every arm (identical by construction: 4 x 1 < 31) | #1951 |
| mixed prefill+decode step (`runtime.prefill_mixed_decode`, default on): the decoders ride the ragged prefill forward as one-row members, one batched paged-decode attention launch per layer, sampled and delivered by the decode path; the step runs no separate decode forward | SHIPPED | Qwen3-14B-NVFP4, 3 trials x 3 waves alternating, wave medians. 32 x 982-token burst: aggregate +7.3 / +6.6 / +7.8%, TTFT p50 830/821/842 -> 665/630/660 ms, p90 1426/1412/1431 -> 1116/1116/1126, max 1582-1610 -> 1234-1247, ITL p95 12.8-13.0 -> 12.8-13.2, ITL max 141-142 -> 80-82, gaps > 100 ms 22 -> 0 of 9568; debug-log timeline wave 2: 2048-row step 85.6 -> 65.3 ms, ingest 1.455 -> 1.175 s (26.2k rows/s). 31 x 87-token streams + a 4436-token ingest at 2 s: the short streams' ITL during the ingest p95 62.7/63.4/63.7 -> 52.2/51.3/51.2 ms, max 65-66 -> 53-56, ingest TTFT 293/293/291 -> 242/239/237. Qwen3.8-27B-NVFP4 (recurrent, riders refused by design), 1 trial: +2.2%, TTFT p90 +1.2%, noise. REFUTED on the way: riders through the per-member prefill attention route (32 x 40 one-row FMHA launches per step) read 113.8 ms per 2048-row step, aggregate -10.7%; the batched decode launch is the mechanism. Greedy text of a rider is coherent and diverges from the separate-step arm after ~40 tokens (batch-shape, the 2048-row GEMM against the small-M path), bit-identical on the synthetic dense model under `runtime.deterministic` | #1952, `tools/analysis/prefill_cap_conc_ab.sh` |

### Where the 32-stream window goes (2026-09-02, Qwen3.8-27B-NVFP4-vllm at 1774 tok/s)

| class | share | verdict |
|---|---:|---|
| GEMM small-M pair + v2 | 57% | closed, structural |
| GDN decode scan | 20% | AT THE BANDWIDTH FLOOR: 96 MB per launch = 61 us at 1570 GB/s against 57.4 measured; FP16 state refuted on subnormals, BF16 shipped |
| attention decode | 3.2% | ILP lever priced at ~1% e2e, not built for NVFP4 at short context |
| conv1d decode | 2.7 -> 1.4% | shipped |
| norms | 1.8% | launch-bound, norm+quantize already one kernel, closed |
| alpha/beta cuBLAS | 1.2% | packed alpha+beta GEMM would halve the launches for a stride in the scan's reads, 0.6%, priced |
| penalties | 1.1 -> 0.1% | shipped |
| memcpy | 1.7% of device time | recurrent-snapshot D2H on its own stream, not idle |
| idle | 14.9% measured, ~8% real | >1 ms gaps are CUPTI-inflated captures at the wave ramp (5.57 vs 5.51 s with and without); the remainder is Open item 1 |

```
[PROV: commit=a65200b3+pdl date=2026-08-31 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-cli --bench --bench-pp 512 --bench-reps 3
       --set speculative.ngram=false --set speculative.mtp_k=0 --set
       runtime.no_pdl=true|false, 3 alternating rounds, dev build; @32:
       tools/analysis/two_image_conc_ab.sh imp:ab-base (a65200b3) vs imp:test
       (pdl), 3 alternating trials, median of 3 waves; idle:
       tools/analysis/serving_idle_profile.sh window 14-32 s]
```
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

## Batch=1

Roofline re-derived 2026-08-27 (graphs-ON nsys window, 778 steps): the box
reads **1628 GB/s resident** (the 1530 pin was stale), the Qwen3.8-27B-NVFP4
spec-off ceiling is **~112 tok/s** at 14.5 GB/token, measured 87.4 = 78%, and
the decode graph is strictly serial (kernel-interval union == sum, 718k
intervals).

| step component | ms | note |
|---|---:|---|
| GEMV classes | 9.69 | ~1496 GB/s avg; gate_up 1613 / lm_head 1655 prove the ceiling; ~0.4 ms class headroom |
| attention | 0.48 | latency-bound at short ctx, both split directions refuted |
| 96 FP16 alpha/beta GEMVs | 0.37 | |
| norms | 0.30 | |
| GDN scan + conv | 0.32 | |
| host / idle | 0.44 | |

Past the roofline only through the MTP verify (weights read once per k+1 rows):
102-110 tok/s at k=1 (#1796), k=2 stable via adaptive depth (#1801), default
since #1809; k=3 uneconomic; `speculative.verify_smallm` +3-6% isolated, +1-2%
mixed, default off.

```
[PROV: commit=a70d7863+wt date=2026-08-27 hw=RTX5090
       model=Qwen3.8-27B-NVFP4 cuda=13.3 path=nsys server window 778 steps
       cmd=`nsys profile ... imp-server` + chat 1024-tok]
```

### The MTP verify on a GDN hybrid (2026-08-17 .. 08-19)

Detail: [`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md).

| finding | verdict | numbers |
|---|---|---|
| "speculation does not pay on the hybrid" | RETRACTED 2026-08-18: two launch defects kept every GDN projection off the small-M batched GEMV (`ea547a53`) | MTP k=2 104.06 vs 86.21 spec-off (was 75.26 vs 84.47); kernel ms/emitted-token 11.35 -> 8.93 |
| k-sweep on the fixed build (2026-08-19) | k=1 wins, chain length is not a lever | k=0 86.03; k=1 **104.31 (+21.3%)**, 76.0% accept, 1.76 emitted/verify, 16.89 ms/verify; k=2 100.82; k=3 87.71 |
| where the per-row cost lives | the forward, not the recurrent state | `4.96 ms + 5.82 ms x rows`; `gemv_nvfp4_kpar_mb_fp16` = 65.1% of the k=3-k=1 growth, `gdn_scan_fused` 2.3% |
| accounting rule | a verify replaces a decode step only when accepted; on rejection it is additional | full weight sweep per verify regardless of emission, hence the chain-length saturation |
| workload sensitivity | predictability, not the drafter, sets the number | prose MTP k=2 87.9 tok/s (58% accept, 2.3 emitted/verify); verbatim repeat 876.5 (98.3%, 36.6) |
| 14 hypotheses (drafter precision, MoE head, unfused chunk, repair forward, async loop, recurrent divergence, five decode kernels, chunk-side kernel choice, cross-process reproducibility, econ constant) | all measured dead or corrected | econ break-even measured 2.42, the k-aware default |

```
[PROV: commit=3c3e9ac9 date=2026-08-19 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=imp-server n=3 prompts x 2 alternating rounds
       cmd=`imp-server --think-budget 0 --set speculative.ngram=false
       --set speculative.mtp_k=0|1|2|3 --set speculative.mtp_econ_min_emit=0
       --set server.prefix_cache=false`; tokens from usage.completion_tokens,
       verifies from /metrics]
```

## MoE host offload

Origin: compute cold experts on the CPU (ktransformers shape) to reach 80B-120B
on 32 GB. Measured its way OUT of that design: LRU expert cache plus streaming
won, no AVX kernels, no [`GOAL.md`](GOAL.md) amendment. Budget and campaign
tables verbatim in
[`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md);
Qwen3-30B-A3B Q4_K_M unless marked.

| finding | verdict | numbers |
|---|---|---|
| host compute vs stream-into-VRAM (120B-A5B shape) | streaming wins ~3x | 14.0 ms/token static split + host compute vs 4.7-8.9 LRU streaming; host bandwidth 62.5 GB/s |
| static hot set | does NOT transfer between prompts | -15.2 / -29.5 points vs each prompt's oracle; median reuse distance 2 tokens, so LRU needs no calibration |
| `ExpertCache` at full offload | holds its hit rate | 88.7% hit; 24.98 tok/s vs 6.63 staging-only vs 311.24 resident |
| slot-indexed fused MoE kernels | SHIPPED, 2.1x | the LRU pool IS the contiguous tensor the resident kernels index (#1370) |
| fused MMVQ instead of dequant -> GEMV | REFUTED | expert kernel time -43%, e2e 0%: the path is host-bound, launch count is the currency |
| graphs under offload | BLOCKED | `moe.allow_graphs_under_offload` capture aborts on host-read routing; prefetching a layer ahead is unreachable |
| NVFP4 offload | correct since 2026-08-13 | before it, "mandatory on-device" was unenforced and answered WRONG at 88.77 tok/s; resident 361.97 -> 384.03, full offload 23.03 |
| `moe.pin_host_experts` (default off) | +14.8% pp512 (6/6), 4.4x load time | WSL2 cannot page-lock mmap; per-layer device staging 317.6 -> 790.8 tok/s only with pinning |
| `moe.staged_cutlass_prefill` | opt-in | +136% prefill, -36% decode after long prompts (unexplained): pp512 663.2 -> 1563.9, tg 59.4 -> 37.7 |
| cache budget (`moe.expert_cache_budget_pct`) | 2.47x from a config value, floor is exact | 5% 10.51 tok/s, 15% (default) 20.99, 30% 30.51; below `3*top_k` slots/layer the cache retains nothing (#1374) |
| final regime | transfer-bound, as modelled | H2D 150 GB at ~51 GB/s = 41% of step, kernels ~26%, launches ~24% |

Measurement rule: prefill varies ~15% between runs of the SAME arm, decode
moves with prefill length (cache warmth), cold vs warm differ 2.4x. Only paired
alternating rounds decide, and every number states which. Reproduce:
`tools/analysis/expert_cache_offload_sweep.sh` (MODE=ab).

## First-party NVFP4 quantizer (EXPERIMENTAL, calibration ships)

`imp-quantize` converts dense BF16/FP16 SafeTensors to NVFP4; `--calib` does
AWQ-class activation calibration. `ppl_corpus_45k.txt`: Qwen3-0.6B BF16 24.06 /
RTN 30.10 / **AWQ 28.48**; Qwen3-1.7B 17.22 / 20.43 / **19.21**;
`degen_suite.py` 45/45. Detail: [`quantization.md`](quantization.md).

| finding | verdict | numbers |
|---|---|---|
| (a) micro-scale search vs absmax | not worth it | PPL 30.10 -> 29.88 (0.7%) for ~6x cost; the FP4 grid dominates, hence AWQ (move the error), not better scales (2026-07-26) |
| (b) o_proj scale folded into v_proj vs FP8 KV | refuted concern | FP8-vs-FP16-KV penalty 0.300 PPL calibrated vs 0.595 RTN: scaled v_proj is FRIENDLIER to FP8 KV (2026-07-31) |
| (c) calibration determinism | forced | without `deterministic_gemm` two runs differ on 94% of floats, PPL moves 1.6%, degen probes flip; `--calibrate` now forces it |
| (d) "MoE not supported" | wrong in the dangerous direction | experts quantized fine (4992 on DeepSeek-V2-Lite); MLA latent projections + router broke and are now refused; 3.28x compression, degen 3 FAIL/32 = strict subset of BF16's 5 (2026-07-31) |
| (e) head-to-head vs Modelopt export | imp ahead on one model | Qwen3-14B, same source weights: Modelopt 10.0301 vs imp-quantize uncalibrated **9.9252** (+1.05%). Retires "prefer a published export", not more; the export ships input/k/v scales imp verifiably does not apply |
| (f) 3-D stacked experts | REFUSED (2026-08-01) | the old refusal never fired (#1188: experts copied through as BF16 while `hf_quant_config.json` announced NVFP4). De-stacking rejected against gpt-oss-20b: the fused layout is not one layout and expert biases have no per-expert loader path. Open item 7 |
| (g) calibrate off a quantized twin | works, and exposed a 14B regression | 0.6B twin-calib 28.8868 vs BF16-calib 28.4782 vs uncalib 30.0979; 14B RTN **9.9252** vs twin-calib 12.6016 / Modelopt-twin 12.2853, two independent quantizers agree that `--calib` HURTS at 14B (2026-08-01) |
| (h) why 14B flips, via `--calib-groups` | ANSWERED: the attention pair | vs own RTN (n_rep=5): **BD -0.1330** best, BCD -0.08, C +0.02, A +0.65, ABCD **+2.68**; interaction C x ABD = **+1.90 = 71% of the damage**. On 0.6B (n_rep=2) the same interaction is +0.05, 40x smaller. Rule: `--calib-groups BD` on wide-GQA, ABCD on narrow-GQA (2026-08-05) |
| (i) vLLM-loadable output | SHIPPED | `--format vllm` writes compressed-tensors `nvfp4-pack-quantized`; vLLM 0.27.1 loads and generates (51.8 -> 19.2 GiB). Tensor scale is stored INVERTED between layouts, one scale per fused group is also better quantization (0.6B 30.40 -> 29.42). Refuted: absmax/(6x448) measures 31.05, worse than absmax/6 (2026-08-16) |

## Closed competitive records

| record | verdict |
|---|---|
| NVFP4 prefill vs vLLM | CLOSED 2026-06-13 (`290a163a`): FA2 FP16-QK primary hd=128 prefill +21-24% pp4096, MoE pp4096 +4% ahead, MoE pp2048 +27%, dense pp2048 ~tie. Residual dense pp4096 ~1.04x is structural (FA2 at ~5% DRAM, cost is the NVFP4 GEMMs at ~59%) |
| kv-fp8 storage default-on | SHIPPED for Qwen3 dense/MoE, Llama, Nemotron-H MoE via `kv_cache.dtype=auto`, ~768 MiB saved on dense. Blocked and not actionable: Qwen3.6/3.5 declare no FP8 hint, Gemma-4's gate-corpus baseline PPL is broken |
| Q4_K_M prefill gap (-38% vs llama.cpp) | REFUTED: in-SMEM Q4_K MMQ + HMMA built and ncu-proved decode-throughput-bound, tying cuBLAS; beating it needs 2x weight VRAM (rejected). Use NVFP4 SafeTensors for fast Q4_K-class prefill ([plan](plans/2026-05-28-q4k-mmq-kernel-design.md)) |
| sawtooth wavefront reordering (#456) | REFUTED 2026-05-29: only lives in the WMMA fallback, unreachable on the hot path; forced A/B flat-to-negative |
| batch=1 competitive campaigns | closed as programs; targeted wins still land: FA2 hd=256 prefill +26% over WMMA (#930/#932), FP8 tile attention long-context decode +14% (#899/#900), FP8 SSM projection sidecar 35B decode +19% and GGUF hybrids +21% (#949), speculative decoding economics up to +156% on echo-heavy agent traffic (#852, #862-#866) |

## Known limitations

Owner of every limitation is [`LIMITATIONS.md`](LIMITATIONS.md); this file
keeps the verdicts that came out of roadmap work (full text in
[`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md)).

| item | verdict | numbers |
|---|---|---|
| MTP on Nemotron-3.5 | head drafts, verify chunk uneconomic | 41.1% offline = 39.2/38.9% serving (the 0-9% gap was `run_ssm` never writing the snapshot slab, fixed 2026-08-20); k=1 with the guard off -51% (176 vs 363 tok/s), shipped guard lands 258-341 on the 1+0.40k break-even |
| Qwen3.5-27B MXFP4 | blocked on a decodable checkpoint, not a bug | no MXFP4 SafeTensors decode outside gpt-oss |
| Gemma-4 Q4_K_M code-gen drift | no longer reproduces (2026-06-13, 2026-08-11) | original file gone; fallback Q5_K_M or Q8_0 |
| native-FP8 weights decoding through the FP16 companion | CLOSED 2026-08-12 | `FP8CacheEntry::native_source` drives the sidecar: +7.5% median decode, 27 pairs, order balanced |
| no dequant path for native FP8 | CLOSED | FP16 companion at load (sm_120 has no FP8 prefill GEMM); Nemotron-3.5 1698 MiB FP16 cache, init 24.4/32.6 GB ([`MODELS.md`](MODELS.md)) |

## Investigated and shelved

| item | verdict |
|---|---|
| draft-model speculative decoding | separate draft models don't amortize weight reads on one bandwidth-bound GPU. Shipped instead: prompt-lookup n-gram (#668-#670) and MTP self-drafts with hybrid-safe verify (#852) |
| FFN contextual sparsity | warp-cooperative layout masks the skip, +0-1% measured |
| BitDecoding (TC KV decode) | SHELVED with the scope stated (#1268). The original "0% gain, decode is weight-bound" was measured at tg256 = 64 prefilled tokens, where paged attention is 4.3% of the window; re-measured 2026-08-21 at **19.9% at 8k, 43.9% at 32k**. Still shelved because the levers died: split-count boost +10.0% at 32k on Qwen3-8B but **-7.30%** on Qwen3-30B-A3B (#1270, reverted #1271), KV block 16 -> 32 neutral, "latency-bound at 192 GB/s" retracted (629.6 GB/s at 32k at the same 16-17% occupancy). Re-open on a mechanism, not on the share |

The share grows with context, which is why it is not the trigger:

| model | KV heads / g | layers | ctx | paged attention | ceiling if zero |
|---|---|---|---|---|---|
| Qwen3-8B-Q8_0 (dense) | 8 / 4 | 36 | 8k | 19.9% | 1.23x |
| Qwen3-8B-Q8_0 (dense) | 8 / 4 | 36 | 32k | 43.9% | 1.76x |
| Qwen3-30B-A3B-NVFP4 (MoE) | 4 / 8 | 48 | 8k | **29.1%** | 1.36x |
| Qwen3-30B-A3B-NVFP4 (MoE) | 4 / 8 | 48 | 32k | **50.6%** | ~1.96x (1.92-2.01) |

The share rises across the dense/MoE boundary because the non-attention half
falls faster (-32.0%) than attention (-10.5%). Differential measurement: two
runs differing by exactly 256 decode steps, kernel sum 91-98% of the wall step,
repeat pairs within 0.12 pp.

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
