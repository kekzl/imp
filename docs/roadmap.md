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
| dense NVFP4 @32 vs vLLM | MIXED. Qwen3-14B 38-token prompts 3948.9 vs 3817.6 (+3.4%), 982-token prompts 1845.0 vs 2478.0 (0.75x) |
| batch=1 | 87.4 tok/s spec-off = 78% of the ~112 tok/s roofline (14.5 GB/token at 1628 GB/s resident); past it only through the MTP verify |
| raw-speed half of [`GOAL.md`](GOAL.md) | MET: batch=1 decode +13-48% vs llama.cpp on every hero (2026-07-12 re-sweep), MoE prefill leads vLLM single-seq, cross-engine PPL parity measured. Everything open below is the agentic half |
| admission at fan-out | `auto` resolves 28 vs a pinned 32 (630 vs 936): admission, not rotation, and 28 sustain full rate under continuous arrival |
| next engine-side post | launch-coupled idle (~8%) and the paced serving prefill on dense models |

Both engines measured on one client (`tools/analysis/vllm_conc_ab.sh`, 3
alternating trials, same checkpoint); the "1.58x gap" and "~1.08x pinned" of
2026-08-24/26 compared an imp number from this client against a vLLM number
from a 200-token-gen client. Rows with PROV in [`BENCHMARKS.md`](BENCHMARKS.md)
("re-measured on one client", runs 5-9).

## Open

Ranked by what an agent workload notices first.

| # | item | state | ref |
|---|---|---|---|
| 1 | launch-coupled idle @32 | ~8% real idle: launch density 6.3% (~1350 gaps/step, 0.4 us inside the replay), host turnaround 1.9% (~2 gaps/step of ~200 us). Both direct attacks are closed (one-H2D staging NEUTRAL, host turnaround attributed), so the next cut is fewer launches, not faster ones | `tools/analysis/serving_idle_profile.sh` |
| 2 | paced serving prefill, dense | 31.4k prompt tokens in ~2 s under decode against 26k tok/s standalone; vLLM absorbs the same wave in ~1.1 s. `prefill_chunk_decode_cap` stays 1024: 2048 buys +4.4% hybrid / +8.6% dense and costs the decoders +70% / +63% ITL during someone else's ingest | ledger 2026-09-03 |
| 3 | long context | HALF CLOSED. Quest-class top-k page selection opt-in: Qwen3-8B 32k 160.3 -> 199.5 tok/s (+24.5%), NVFP4-KV 77k 74.3 -> 100.2, concurrent 3x25k +27%, spec verify on the sparse table +28.2%. Price on Qwen3.8-27B: NIAH 10/10 dense, 8/10 @8192, 5/10 @4096. Remaining: MLA models, prefill sparsity, StreamingLLM eviction (`src/compute/attention_paged_common.cuh:71`) as the only answer under KV-pool pressure | #1808, #1818, #1819, [plan](plans/2026-08-28-sparse-decode-attention.md) |
| 4 | speculation adapts per request | HALF CLOSED. Chain depth adapts (#1801), `mtp_k=auto` drafts single-stream 95.8 -> 141.6 tok/s (+48%). Remaining: drafter choice is global; the chain saturates near 2.5 accepted/verify and the multi-candidate tree measured no gain past it | #1809, #1811, [plan](plans/2026-08-31-mtp-multicandidate-hybrid.md) |
| 5 | recurrent-state paging | the lever for 32-way concurrency at LONG context, not the limiter at 32 slots. Evicted snapshots reach a pinned host tier since 2026-09-02 (turn-2 TTFT at 8 sessions -50%) | `server.recurrent_snapshot_host_mb` |
| 6 | `--calib` hurts at wide GQA | 14B RTN 9.9252 vs twin-calib 12.6016; the C x ABD interaction is 71% of the damage. Shipped rule: `--calib-groups BD` on wide-GQA, ABCD on narrow-GQA. The search still minimises a local proxy (per-group weight reconstruction) | finding (h) |
| 7 | quantizer refuses 3-D stacked experts | needs a per-model layout descriptor (gpt-oss interleaved vs Gemma-4 concatenated) plus per-expert bias in loader and MoE forward | finding (f) |
| 8 | no audio | Gemma-4 ships `model.embed_audio.*`; `src/model/weight_map.cpp:380` folds those tensors into the aggregate `skipped` count, so an omni checkpoint loads as text+vision and says so nowhere | |
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
| batched residual accumulate (beta=1) | REFUTED | -0.9% median, 3/3 negative; residual adds already overlap in the graph | `gemm.nvfp4_residual_beta1` |
| batched ban + penalty sweep | SHIPPED | 1766.9 -> 1774.9 (3/3), 2 launches per row per step -> 1 sweep per step, idle 14.9 -> 13.6% | 2026-08-31 |
| penalties walk the history, not the vocabulary | SHIPPED | per launch at 32 rows: 300-token history 197.1 -> 10.7-23.3 us, 4096-token 2659.3 -> 18.6-34.3 us; @32 3/3 positive, logits bit-identical | `sampling_penalties.cu`, 2026-09-02 |
| conv1d decode, float4 state and one weight load | SHIPPED | 9.61 -> 4.97 us per launch (273.4 -> 144.2 ms of a 10.4 s window), 3/3 positive, bit-exact | `src/compute/ssm.cu`, 2026-09-02 |
| PDL device half (`griddepcontrol`) | SHIPPED default-on | @32 3/3 positive (+1.3% median), M=1 +1.7% median, idle 13.6 -> 10.8% | `runtime.no_pdl` |
| one-H2D decode-step staging | NEUTRAL, closed unmerged | 2/3 pairs negative (-0.2 / -1.1 / +0.6%) | branch `perf/decode-step-staging` |
| graph prewarm | RETIRED as throughput, SHIPPED as latency | wave-1 aggregate unmoved (629-650 vs 627), wave-1 p50 -3-12% | #1761, `runtime.graph_prewarm` |
| batch=1 async-loop recapture fix | ITL fix, not a lever | FRESH captures 128 -> 7 per ~200-token burst, +0.2% throughput | 2026-08-27 |
| host turnaround | ATTRIBUTED, closed as a defect class | per step: build 63-82 us, fwd-enqueue 34-47, distribute 7, schedule 1-2; outside-step 1.2-1.5 ms is the paced serial prefill | `diagnostics.step_timing` |
| prefill concurrent with decode | NEUTRAL both shapes, default-off | short prompts 1771.3 vs 1777.7, 1000-token ingest 789.7 vs 790.6; no green-context SM partitioning on sm_120 | `runtime.prefill_overlap`, [plan](plans/2026-08-27-prefill-decode-overlap.md) |
| ragged cross-sequence prefill | SHIPPED default-on | +6.2% (977.3 -> 1038.2, 12/12 waves), TTFT p50 4.11 -> 2.55 s | `runtime.prefill_batch`, #1780 |
| ragged members charged their real rows, per-member chunk-parallel GDN scan | SHIPPED | 1094-token prompts 943.7 -> 1058.0 tok/s (+12.1%), ITL p95 46.2 -> 19.9 ms, gaps > 100 ms 349 -> 224 | 2026-09-03 |
| `prefill_chunk_decode_cap` 2048 | MEASURED, default stays 1024 | +4.4% hybrid / +8.6% dense at ~1k-token prompts, but the decoders' ITL during a foreign ingest +70% / +63% | 2026-09-03 |
| BF16 GDN state | SHIPPED default-on | scan 2.04x isolated, +12.5% KV-pinned (1210.5 -> 1362.0), +7.7% pure defaults, PPL +0.21% | `gdn.state_bf16`, #1778 |
| growable KV under aggregate pressure | SHIPPED opt-in | 32 x 8k/512: wall 86.0 -> 65.2 s (-24%), pool 2046 -> 6483 blocks | `kv_cache.growable`, #1794 |
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
| OpenTelemetry span export | SHIPPED, off by default | one SERVER span per generation request from the single accounting point, queue / prefill / decode children, OTLP/HTTP JSON, batches per second or 256 spans | `server.otlp_endpoint`, #1855 |
| reasoning scan hold released on the first word | SHIPPED | chat thinking-off TTFT 97-105 -> 32-62 ms, 1116-token prompt 195-229 -> 130-146, completions 116-147 -> 51-64; the hold of 8 tokens never protected a real chain of thought, the tool path keeps 256 | #1894 |
| `cudaGraphExecUpdate` on the parked exec instead of instantiate per request | SHIPPED | 31 of 34 setups updated in place in 0.1 ms, wall median 545 -> 532 ms, max inter-token gap 25-35 -> 23-26 ms, output hashes identical; the 44 ms instantiate in the #1894 trace was CUPTI-inflated | #1895 |
| remaining burst-boundary gap (22-26 ms) | OPEN, small | the eager n-gram probe step plus capture and sync, not the instantiate | #1895 |

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
