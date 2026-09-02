. Two-image A/B  (`tools/analysis/two_image_conc_ab.sh`, Qwen3.8-27B-NVFP4-vllm, 3 alternating trials x 3 waves, median tok/s): base 1748.1/1824.6/1794.8 -> 1835.8/1833.4/1850.4, 3/3 pairs positive# Roadmap

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

0. ~~**Concurrency scaling on the GDN hybrid vs vLLM: 1.58x gap at 32 streams,
   attributed per component and driven to ~1.08x**~~ CLOSED 2026-09-02, imp
   leads: one client for both engines (`tools/analysis/vllm_conc_ab.sh`, 3
   alternating trials, same checkpoint), 32 streams imp 1807.9 vs vLLM 0.27.1
   1447.8 tok/s (+24.9%, 3/3) and 1833.8 vs vLLM 0.28.0 1410.7 (+30.0%, 3/3),
   8 streams 573.0 vs 495.8 (+15.6%, 3/3), 32 streams x 1082-token prompts
   873.4 vs 497.8 (+75.5%, 3/3); the "~1.08x" was an imp number from
   this client against a vLLM number from a 200-token-gen client. Table with
   PROV in [`BENCHMARKS.md`](BENCHMARKS.md) "re-measured on one client".
   The attribution below is the record of how the 1.58x was closed (nsys on
   both engines, same checkpoint, same 32-stream wave).

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
   | prefill budget charges ragged members their real rows; ragged GDN scan per member on the chunk-parallel kernel | SHIPPED | Long-prompt burst (32 x 1094-token prompts, 300 forced tokens, `tools/analysis/burst_stream_client.py` under `prefill_cap_conc_ab.sh`): the hybrid's prefix-cache snapshot splits a 1093-token prompt into [0,1024) + [1024,1088) + [1088,1093) (block-aligned snapshot at 1088), and the token budget charged a ragged member its full chunk, so after a 5-row tail the next prompt's chunk did not fit and the step ran the tail alone (a member joining a ragged group is now bounded by the forward's rows; the launch floor is charged once per group; `prefill_batch_decode_cap` counts forwards, a ragged group once). Packing alone read neutral (+2.2% at 1094-token prompts, -2.7% at 982): the packed 3-4-sequence 1024-row forwards ran the fused batched GDN scan, and `runtime.prefill_batch=false` beat them by +5.7%. The ragged forward now runs the chunk-parallel scan per member (h_seq_offsets / h_ssm_slots, members under 128 rows on the fused single-sequence kernel) when the forward holds at least one member of 128+ rows and at most 4 smaller ones; a burst's first forward of 32 short prompts stays on the batched kernel (per-member launches there read -1.5% and +112 ms TTFT at 32 x 38-token prompts). Two-image A/B vs main+ignore_eos, 3 alternating trials x 3 waves, medians: 1094-token prompts 943.7 -> 1058.0 tok/s (+12.1%, pairs +13.7/+6.7/+11.3%), wall 10.17 -> 9.07 s, TTFT p50/p90/max 2994/5027/5275 -> 3329/3854/3909 ms, ITL p95 46.2 -> 19.9 ms, gaps > 100 ms 349 -> 224 per 9568 tokens (the per-member scan without the small-member rule read +15.6% here and -1.5% at 38-token prompts). With the packing in place `prefill_chunk_decode_cap` re-priced on the same shape (config A/B on the branch image, 3 x 3 waves): 2048 reads 1058.3 -> 1105.2 tok/s (+4.4%, pairs +5.0/+2.8/+3.6%), TTFT p90/max 3837/5950 -> 3465/3476 ms, gaps > 100 ms 196 -> 29, but max ITL 139 -> 203 ms; 0 is identical to 2048 on this hybrid (2048 is its chunk ceiling). Default stays 1024 (the max-ITL bound is the knob's contract); agent fan-out with ~1k-token prompts can set 2048; 982-token prompts (under the cap) 1040.5 -> 1128.3 tok/s (+8.4%, pairs +7.2/+8.5/+8.9%), wall 9.23 -> 8.51 s, TTFT p50/p90/max 2596/4104/4339 -> 2674/3353/3409 ms, ITL p95 34.7 -> 19.6 ms, gaps 319 -> 173; 38-token prompts (the headline short-prompt shape) 1846.7 -> 1843.4 tok/s (-0.2%, pairs +0.8/-1.2/-1.1%, neutral), TTFT p50 514 -> 532 ms, ITL p95 17.6 both. Deterministic 8-stream burst on the real checkpoint: all members coherent and on topic on both scan paths (texts differ, greedy diverges on the 1e-6 state difference). `ignore_eos` (vLLM-compatible) added to both dialects for equal token counts per arm; the earlier pangram filler made ~half the 1000-token prompts answer with an immediate EOS on either chunking (not a boundary defect, deterministic PPL 1024 vs 2048 chunks 4.6413 vs 4.6365) | 2026-09-03 |
   | prefill kernel utilization measured (the open "%-peak" gap in this table) | MEASURED | roofline run `1d5b9230_20260831_180644` (ncu clocks locked, 1 restart): dense NVFP4 prefill GEMM (CUTLASS block-scaled) 79.8% of measured FP4 peak @pp4096 / 64.9% @pp512 - NOT the hole; the holes are `gemm_cublas` on the hybrid @pp512 (24.8% roofline at 21.5% share, est. +13.9% window - the FP16 GDN projections) and `attn_fa2` @pp4096 dense (22.8% at 21.9% share, est. +12%). The dense fa2 instance (`fa2<128,128,…,PVF16>`) runs 144 allocated registers x 256 threads = 36864 of the 65536-register file and 70656 B of the 102400 B smem per SM (per-launch ncu, 2026-09-01; the earlier "255 registers" here was `device__attribute_max_registers_per_thread`, not the kernel), so BOTH limits bind at one CTA/SM and 2 CTAs need <=128 regs AND <=50 KB (both taken on 2026-09-01, `attention.fa2_dense_2cta`); ncu on the big pp4096 launches (337 us) shows DRAM at 7.6%, compute 34%, 72% no-eligible cycles - latency-bound inside 8 warps/SM, not bandwidth. The cheap alternative (forcing the existing Bq=64/TWOSLOT band for 2 CTAs/SM at the same 8 warps) measures -2.0%/-1.3% pp4096, 2/2 pairs - CTA overlap loses to the extra softmax rescales. The remaining dense lever is deeper in-CTA pipelining (3-stage cp.async at Bkv=32, or QK/PV stage overlap). The HD=256 instance is a different case, see the `fa2_hd256_bkv` row below. Full table: `tools/roofline/history/runs/1d5b9230_20260831_180644.json` | 2026-08-31 |
   | HD=256 FA2 at 2 CTAs/SM (`attention.fa2_hd256_bkv=32`) | BUILT, opt-in on the PPL trade | per-launch ncu on run 1d5b9230: the hybrid FA2 instance (`fa2<64,256,…,TWOSLOT>`) holds 232 regs x 128 threads = 29696 of 65536 (two CTAs fit) but 68608 B of the 102400 B smem/SM (one fits), 8.3% occupancy; Bkv=32 halves the tile to 34816 B, 226 regs, 0 spills. Isolated 24Q/4KV: Sq=512 +4%, 1024/2048/4096 -12.6/-10.0/-15.1% kernel time. Qwen3.8-27B-NVFP4 pp4096 (1024-token chunks, 384 CTAs), 3 alternating pairs under nsys: FA2 kernel sum 138.17/137.71/137.75 -> 122.53/122.61/122.33 ms (-11.2% median, 3/3), pp 6719.1/6724.9/6731.2 -> 6745.0/6752.2/6737.6 tok/s (3/3 positive, +0.1..+0.4%; the kernel is ~7.6% of the window); pp512 flat (5.244 -> 5.229 ms, 2 pairs). PPL on `ppl_corpus_45k.txt` (13811 tokens, reproducible to 4 digits across 2 runs and deterministic mode): Bkv=64 4.6283 -> Bkv=32 **4.6529 (+0.53%)**; the f32-PV twins read 4.6216 -> 4.6331 and the cuBLAS S-matrix reference 4.6340, so the loss is the doubled f16 O rescale count, and the f32-PV Bkv=32 instance that holds PPL runs at the Bkv=64 time (1.123 vs 1.124 ms isolated). The dense case followed on 2026-09-01 (`attention.fa2_dense_2cta`, next row). Harness `tools/analysis/fa2_hd256_bkv_ab.sh` | 2026-09-01 |
   | stream-K scheduler on the CUTLASS NVFP4 prefill GEMM (`gemm.nvfp4_cutlass_streamk`) | SHIPPED default-on | the pp512 64.9% vs pp4096 79.8% of run 1d5b9230 is wave quantisation: the 128x128 grid at M=512 N=5120 is 160 CTAs on 170 SMs (0.94 waves) and M=512 N=17408 is 544 (3.2 waves). Isolated (weight ring > L2, `test-quant --gtest_filter='CutlassNvfp4StreamKTest.*'`): stream-K pays only where a tail wave exists, 200 CTAs 42.3 -> 35.3 us, 240 42.6 -> 39.9, 544 98.3 -> 85.2; at 160 CTAs it costs (27.0 -> 30.8 forced; the scheduler's heuristic itself picks data-parallel, sk_units=0), at 80 it costs (21.2 -> 22.9), at 320 (tail 0.88) 46.9 -> 49.7. The pingpong 128x64 tile at N=5120 (320 CTAs) reads 36.2 vs 27.0: the 0.94-wave shape has no tiling fix, one CTA/SM is its regime. Isolated readings on the sub-wave shapes move up to 1.6x between runs at healthy clocks (q_proj 27-45 us, the ring was raised to 256 MB); the nsys pairs below are the verdict. Dispatch rule: >= 1 wave and a last wave <= half full, then the heuristic (the stream-K-typed kernel in data-parallel mode reads slower than the plain kernel at 640 CTAs, 109 vs 100 us, so shapes outside the rule keep the plain kernel). Qwen3-14B-NVFP4, `tools/analysis/prefill_kernel_ab.sh`, alternating pairs under nsys: pp512 CUTLASS kernel sum 101.10/103.64/102.11 -> 97.13/97.10/97.25 ms (-3.9/-6.3/-4.8%, 3/3; e2e pp512 spread 19.6-24.2k across arms, the known 37% process-start variance); pp4096 683.7/681.1 -> 683.9/682.0 ms, flat (no shape qualifies). Output bit-identical to data-parallel (the trailing unit continues from the partial in the accumulator). Trap on the way: the engine sizes one workspace at its max shape, where the heuristic picks data-parallel and needs 0 B, so the first cut refused every pp512 gate/up launch into dequant+cuBLAS (22k -> 4.2k tok/s); and the forced decomposition is not monotone either (512x8192 needs 16.8 MB, 4096x8192 11.7 MB); sizing now takes the max over every 128-tile grid up to the max shape (22.2 MB at the contract test's 4096x8192), `CutlassWorkspaceContract` and the new test assert coverage | 2026-09-01 |
   | `gemm_grouped_nvfp4` (MoE prefill, 53% of the hybrid pp512 window at 55% of DRAM bandwidth) | REFUTED twice, structural | Qwen3.6-35B-A3B geometry: 256 experts, gate/up N=512 K=2048, down N=2048 K=512, ~16 rows per expert at pp512; the 128x128x128 tile pads each expert to 128 rows and runs at 57-62% of the 134 MB weight floor (`tests/test_cutlass_grouped_tile_bench.cu`, weight ring > L2). (1) CUTLASS tile sweep: 128x64x128 +18/+32% (gate-up/down), pingpong 128x64 -0.9/+24%, 128x128x256 +17/-0.2% (pp512) and +12/-17% (pp1024), pingpong 128x128 +220-250%; the builder rejects M=64 tiles (SF atom = 128 rows). (2) v2 small-M grouped kernel (32-row tiles, cp.async ring, `nvfp4_gemm_smallm_v2.cu`, branch `perf/moe-smallm-v2-grouped`): isolated gate/up 122 -> 107 us (-12.5%), down +12% (K=512 = two pipeline stages), bit-identical to the dense v2 kernel; e2e on the real routing (nsys, alternating pairs): pp512 kernel sum 78.67/78.67/79.22 -> 81.55/81.86/81.27 ms (+3.5%, 3/3 worse), pp1024 93.10/94.00 -> 107.99/107.95 ms (+15%, 2/2); the first cut that looped an expert's tiles inside one CTA read 79.3 -> 93.7 ms (+18%) and 92.9 -> 134.4 (+45%). Mechanism: every 32-row tile re-streams its expert's weights, so the skewed routing (sum of ceil(M_e/32) tiles, light experts still pay a full pass) multiplies weight traffic where the 128-row tile pays one pass per expert. (3) The multi-tile CTA was then built (template MT 32/64/128 on the v2 body: MT/32 sub-tiles share every weight fragment, branch `perf/moe-grouped-multitile`): isolated balanced mt32 -8.7%, mt64 +1.7%, mt128 +20.8%; on the real routing pp512 class kernel sum mt64 77.90/77.65 -> 77.91/77.81 ms (flat), mt128 +4.1/+2.4%, mt32 +4.5/+4.0%. The skew penalty is gone at mt64 and nothing remains: ~60% of the weight floor is where both designs land. CLOSED, default stays CUTLASS | 2026-09-01 |
   | dense FA2 (hd=128, Bq=128) at 2 CTAs/SM (`attention.fa2_dense_2cta`) | SHIPPED default-on | the two limits the per-launch ncu named: smem via the TWOSLOT tile at Bq=128 (69632 -> 34816 B), registers via `__launch_bounds__(256, 2)` on a separate wrapper kernel (137 -> 128 regs, 24 B spill; the shipped single-CTA instance keeps byte-identical SASS, checked with cuobjdump; a plain min-blocks=1 bound on it moved ptxas to 180 regs, so the wrapper split is load-bearing). Qwen3-14B-NVFP4, `tools/analysis/prefill_kernel_ab.sh`, alternating pairs under nsys: pp4096 FA2 kernel sum 271.73/267.79/289.87 -> 244.20/244.82/252.18 ms (-10.1/-8.6/-13.0%, 3/3), pp 23722/23661/22509 -> 24176/24097/24057 tok/s (+1.9/+1.8/+6.9%); pp1024 27.77/28.17 -> 25.93/25.85 ms (-6.6/-8.2%), pp flat (25951/25627 vs 25567/25990); PPL `ppl_corpus_45k` 10.0277 both arms (bit-identical). Retires the "occupancy is CLOSED" verdict of #1838 for the dense instance: it was closed for the Bq=64 band at the same 8 warps, not for 16 warps/SM | 2026-09-01 |
   | causal FA2 CTA order, heaviest q-tile first (`attention.fa2_heavy_first`) | SHIPPED default-on, small | ncu on the shipped 2-CTA instance (Qwen3-14B-NVFP4 pp4096, 335 us launches, 2026-09-02): tensor pipe 61% of peak sustained on average but 40..94% between SMs, `math_pipe_throttle` the top stall (4.1 per issue), warps active 30% (of a 33% ceiling), DRAM 60 GB/s - the instance is tensor-pipe bound, not latency-bound any more, so the "deeper in-CTA pipelining" lever above is priced out (no registers left at 128 for a second S tile). The visible imbalance is the causal grid: q-tile t attends (t+1)*2 KV tiles and blockIdx.x ran the light tiles first. Reversing the tile index per head (heavy first): FA2 kernel sum pp4096 Qwen3-14B-NVFP4 223.532/224.010/223.556 -> 220.748/221.213/221.309 ms (-1.2%, 3/3), pp 24929.84/24931.24/24960.67 -> 24988.43/24989.40/24981.86 tok/s; Qwen3.8-27B-NVFP4 (hd=256 instance) 120.737/120.594/120.523 -> 117.782/117.953/117.956 ms (-2.2%, 3/3), pp 11813.26/11803.02/11791.90 -> 11821.82/11804.07/11808.37; output byte-identical in both orders (`FmhaFA2HeavyFirstTest`, an off-by-one mutant fails 8/8). The remaining 39% tensor-pipe idle is the softmax phase between QK and PV, which `mma.sync` cannot overlap within a warp; the 2-CTA interleave is what hides it today | 2026-09-02 |
   | FP8 prefill for the GDN projections (`gemm.fp8_ssm_prefill`) | REFUTED e2e | cuBLASLt FP8xFP8 measures 2.0-3.6x at the projection shapes (sm_120/CUDA 13.3), and the SSM_OUT arm holds quality (PPL -0.03% over 13.8k sliced tokens) - but 6/6 e2e pairs are negative (pp512 -7.4/-2.3/-0.7%, pp4096 -1.9/-0.2/-0.2%, tg flat): ssm_out is 16 of 48 FP16 MB per GDN layer and the per-chunk act-quant+rescale overhead eats the ~1% ceiling. The SSM_IN arm (32 MB/layer, the one that would pay) produced uniform logits (PPL 4.09 -> 248320); root cause ISOLATED 2026-09-01 via layer-0 dumps: the FP8 GEMM's FP16 output held `out / row_scale` before the per-row weight scales were folded in, inf on every weight row with a small absmax (`FP8GemmTest.SsmPrefillFp8TinyRowsStayFinite`); fixed with an FP32 512-row chunk + fused rescale (`gemm_fp8_rowscaled`; cuBLASLt's outer-vector scale mode applies `scale[n & ~1]` on sm_120/13.3 and is out). With both projections on the e2e stays negative (pp512 total kernel time -1.9/+13.6/+3.6%, pp4096 flat or a 2x VRAM-spill run). Closed unmerged, branches `perf/fp8-ssm-prefill` and `perf/fp8-ssm-prefill-v2` are the record; full tables [`plans/2026-08-31-fp8-ssm-prefill.md`](plans/2026-08-31-fp8-ssm-prefill.md) | 2026-08-31 |
   | Chunk-parallel GDN prefill scan (`gdn.chunkpar_scan`, default on) | SHIPPED | The nsys steady-state kernel map put `gdn_scan_fused_kernel` at 42% of the Qwen3.6-35B pp512 wall (658 us/layer, grid (32,1,1), sequential over tokens) - a class the roofline's 120-launch ncu window missed entirely; every prior scan route (fused/chunkwise/WY/TC) kept grid=n_heads. New (`src/compute/gdn_scan_chunkpar.cu`): the WY solution split on its linearity in the incoming state (u = u_A - W H_0) makes the per-chunk factors state-independent -> kernel 1 on grid (chunks x heads), kernel 2 a light sequential state pass (three L x 128 x 128 smem matmuls per chunk, column-split 2 CTAs/SM). Class kernel sums (nsys, alternating pairs): pp512 144.7/144.3 -> 98.8/98.2 ms (-32%), e2e +14.9/+21.1%; pp4096 1488/1483 -> 786/787 ms (-47%), e2e 12949/12993 -> 18851/18879 tok/s (+45.6/+45.3%). Deterministic PPL 6.8216 -> 6.8239 (+0.03%); bit-near to the fused kernel (Y 3.1e-5, FP32 state 1.6e-6, GDNScanTest.ChunkparMatchesFused, W-mutant reads 3.9e-1). Costs a 42 MiB engine-lifetime workspace (degrades to the fused route when unavailable). Still ~4x over the compute floor: ncu shows short_scoreboard 2.4 + barrier 2.1 stalls/issue in the intra solve, and accumulator-split/unroll variants of kernel 2 measured WORSE (242 -> 295/309 us; register growth vs the 2-CTA residency) - the remaining lever is an MMA form of the three chunk matmuls | 2026-09-01 |
   | Chunk-parallel GDN scan, state pass (kernel 2) on tensor cores | SHIPPED | The three per-chunk GEMMs (u_eff = U_A - W H, y = Y_A + Qeff H, H' = D0L H + K_d^T u_eff) as `mma.sync` m16n8k8 tf32 with H in shared memory, grid (heads x 4) column split. Plain tf32 on all three: K2 242 -> 65 us per 512-token strip but deterministic PPL 6.8216 -> 6.8304 (+0.13%, the state path compounds the 10-bit operand rounding; FP32 state diff vs fused 3.4e-4). 3xTF32 (error-compensated split) on the two GEMMs that feed the carried state, plain tf32 on y: 90 us, PPL 6.8122, state diff 8.9e-7. Class kernel sums (nsys, alternating pairs): pp512 145.6/144.8 -> 67.4/68.9 ms (-53%; #1847 read 98.5), e2e +10/+29%; pp4096 1530/1574 -> 546/542 ms (-65%; #1847 read 786), e2e 12520/12094 -> 21123/21835 tok/s (+69/+81%; +12..16% over #1847). Registers 110/112, no spills. Kernel 1 (201 us/strip, the WY factor build: Gram, triangular solve, Qeff/Y_A) is now 69% of the scan and the next lever: Gram + Qeff/Y_A are 64x64x128 matmuls, the solve stays scalar or goes blockwise. nvcc 13.3 segfaults on a generic lambda carrying a std::true_type tag inside a kernel (free template function instead) | 2026-09-01 |
   | Chunk-parallel GDN scan, factor kernel (kernel 1) on tensor cores | SHIPPED | Per-CTA phase costs (ncu, test geometry): A 54 us (64 serial per-thread row loads + scalar Gram), B 45 (triangular solve), C 33 (Qeff/Y_A). Now: float4-per-lane row loads, per-token decay/beta in parallel, Gram (K~K~^T, Q~K~^T) as 3xTF32 `mma.sync`, P@W and P@U_A as 3xTF32 -> 75 us per CTA, 201 -> 128 us per 512-token strip in situ. Plain tf32 on P@W refuted: Qeff = D q~ - P W is a difference of O(1) terms (cancellation). Class kernel sums (nsys, alternating pairs, vs the fused scan): pp512 145.2/144.8 -> 52.2/51.9 ms (-64%; #1848 read 67.4/68.9), e2e 10689/10832 -> 15166/12586; pp4096 1498/1526 -> 400.5/399.7 ms (-74%; #1848 read 546/542), e2e 12845/12674 -> 24953/24877 tok/s (+95%; +16% over #1848). Quality: the Qwen3.6-35B deterministic PPL is no judge below ~0.5% - it moved 6.8122..6.8493 across fp32-EQUIVALENT variants (unit-test state diff vs fused 1e-6; MoE routing flips); Qwen3.8-27B-NVFP4 (deterministic hybrid) reads fused 4.6283 -> 4.6148. Registers 94 (K1), no spills. The solve (45 us, 128 barriers, one column per thread) is now 60% of K1 = the next lever (blockwise forward substitution) | 2026-09-01 |
   | Chunk-parallel GDN scan, blockwise triangular solve | SHIPPED | T = beta D KK built once in place of KK (and P = D QK alongside), both RHS staged into the shared-memory histories; per 16-row block the off-diagonal update hist[b] -= T[b, <b] @ hist[<b] as 3xTF32 `mma.sync` (feeds the state), then the 16x16 diagonal block per thread in registers (own column, warp-broadcast T reads): 8 barriers per chunk instead of 128, the up-to-63-term serial chain gone. K1 per CTA 75 -> 49.5 us (ncu), in situ 128 -> 82 us per strip (K2 91); 102 registers, no spills; unit-test state diff vs fused unchanged (9.5e-7). Class kernel sums (nsys, alternating pairs, vs the fused scan): pp512 144.3/144.3 -> 42.5/42.3 ms (-71%; #1849 read 52.2/51.9), e2e 10732/10821 -> 16057/14514; pp4096 1497/1489 -> 307/308 ms (-79%; #1849 read 400), e2e 12864/12920 -> 27027/26792 tok/s (+109% vs fused, +8% over #1849). Qwen3.8-27B deterministic PPL 4.6273 (fused 4.6283, #1849 4.6148). The scan is now 4 x 32 x (82 + 91) us per 512 tokens on the hybrid; what remains in it is staging traffic (five [64 x 128] FP32 factor arrays per chunk through L2) and the 4-way bank conflict on the history B-operand loads (padding the histories to stride 132 does not fit the 99 KB smem cap next to the padded T/P) | 2026-09-01 |
   | Chunk-parallel GDN scan, state pass at 8 warps + pipelined staging, strip per head count, swizzled factor tiles | SHIPPED | Follow-up on the serving-prefill question first: an nsys class map of a 32 x ~1000-token burst (Qwen3.8-27B, 19.9 s window) puts the ragged fused scans at 524 ms = 2.6% of the wall, because `prefill_chunk_decode_cap=1024` turns most prefill forwards into one sequence x ~1000 rows (the chunk-parallel path already) and the fused batched scan parallelises the rare 2-3-sequence forwards across sequences for free; a ragged chunk-parallel scan was therefore not built. The 27B's own scan class is the lever: 48 heads make kernel 2 (192 CTAs, 1 CTA/SM by shared memory, 4 warps) 176 us per strip vs 90 on the 35B, and ncu reads long_scoreboard 4.25 stalls/issue (the global staging loads) at 16.7% warps active. (1) Kernel 2 at 8 warps (row GEMMs split 4 x 4 warp tiles over 8 warps, one state m-tile per warp): -12/-14% (27B/35B). (2) The next factor block prefetched into registers before each GEMM phase, committed after the barrier, U_A/Y_A epilogue reads hoisted: another -19/-21%, 150 regs, no spills. (3) Strip per n_heads: 48 x 8 chunks = 384 kernel-1 CTAs = 2.26 waves; 7 chunks (1.98 waves) reads kernel 1 -12%, 10 chunks on 32 heads -11%; 14-16 chunks read kernel 1 -21% but kernel 2 +11-13%, the strip's factor set (16 x 48 x 160 KB = 126 MB) no longer fits L2 and kernel 2 stages from DRAM, so `gdn.chunkpar_strip=0` picks the largest strip whose factors fit two thirds of L2, then the fullest last wave (workspace sized for 16: 42 -> 84 MiB at 32 heads, 63 -> 126 at 48). (4) XOR swizzle on kernel 1's stride-128 shared tiles: ncu bank conflicts 11.1M -> 0.56M on 17.3M -> 3.7M wavefronts, kernel 1 only -1..-1.5% (4/4 pairs), the remaining top stall is the 3xTF32 math pipe (2.9). pp4096 vs #1850 (nsys kernel sums, alternating pairs main -> final, dev-build binary copies, 2026-09-02): 27B kernel 2 482.8/493.2 -> 336.9/335.7 ms (-31%), kernel 1 370.2/377.0 -> 324.1/323.3 (-13%), e2e 10002/10008 -> 10962/10813 tok/s (+9.6/+8.0%); 35B kernel 2 151.9/151.9 -> 102.8/103.8 (-32%), kernel 1 153.1/150.1 -> 132.9/134.0 (-12%), e2e 27073/26701 -> 28513/27655 (+5.3/+3.6%). Qwen3.8-27B deterministic PPL (`ppl_corpus_45k`, 13811 tokens) 4.6273 in both arms; unit-test diffs vs the fused kernel unchanged (FP32 state 9.5e-7, Y 6.1e-5). Kernel 2 now lives in `src/compute/gdn_scan_chunkpar_pass.cu` (the file-size gate stood at exactly 600 code LOC after #1850) | 2026-09-02 |
   | Chunk-parallel GDN scan: state-feeding GEMMs on 3xFP16 m16n8k16 (both kernels), what the TF32 rate was costing, and a real-data precision judge | SHIPPED | ncu on the #1851 kernels (Qwen3.6-35B pp512): kernel 2 tensor pipe 67% of peak-sustained active, kernel 1 50%, `math_pipe_throttle` the top stall in both (2.8 / 2.9 per issue) - the scan was bound by the TF32 `mma.sync` rate (GeForce Blackwell: FP16 with FP32 accumulate measured 253 TFLOPS, TF32 half of that). Refuted on that basis, pp4096 nsys kernel sums, alternating pairs: kernel 2 at two CTAs per SM (32-row staging passes, 47.6 KB, 122 regs, the 27B's 192-CTA grid in one wave) 332.9/329.7 vs 331.3/334.9 ms, flat (two CTAs share one tensor pipe); the u_eff GEMM on plain tf32: FP32 state diff vs the fused kernel 9.5e-7 -> 8.5e-5 / 1.1e-4 (fails the 1e-4 bound), every link of the state path keeps the compensated form. Shipped: the state-feeding GEMMs (kernel 1: Gram, solve off-diagonal, P@W; kernel 2: u_eff, H update) as 3xFP16 `mma.sync` m16n8k16 (a = a_hi + a_lo in fp16, 22-bit products like 3xTF32, at the FP16/FP32-accumulate rate and k = 16 per instruction), Y_A = P@U_A (output-only, like kernel 2's y GEMM) plain tf32, operand splits hoisted out of the n-tile loops (kernel 1 -1%). Kernel 2 3xTF32 -> 3xFP16: 27B 327.5/328.2 -> 279.9/279.0 ms, 35B 102.6/102.4 -> 87.4/86.9 (-15%); the k16 fragment pattern then collided on the padded strides (ncu: bank conflicts 6% -> 28% of the shared wavefronts, mio_throttle 0.5 -> 1.3), fixed by staging the factor block as a swizzled tile and stride COLS + 4 for the [k][n] tiles: 27B 280.9/279.9 -> 265.5/266.8, 35B 87.2/87.1 -> 83.0/82.9 (-5%), conflicts 13%, 41% tensor pipe, the rest wait / scoreboard stalls at 8 warps and one CTA per SM. The same swizzle on kernel 1's T / P tiles REFUTED: conflicts unchanged (25%, they are not on those loads) and the diagonal-block reads lose their linear pointer, kernel 1 +10%; Y_A on fp16 k16 sharing the Qeff loop: +5% (the plain path pays a split it does not use), so Y_A keeps its own tf32 k8 loop. Kernel 1 (Gram, solve, P@W on 3xFP16, Y_A plain fp16, vs the hoisted 3xTF32 form): 27B 297.1/296.2 -> 244.0/244.6 ms (-18%), 35B 126.1/126.4 -> 104.5/104.5 (-17%). Final vs #1851 (pp4096, nsys kernel sums, alternating pairs main -> this build, dev-build binary copies): 27B kernel 1 312.3/311.4 -> 243.9/247.7 ms (-22%), kernel 2 328.6/327.7 -> 268.0/268.5 (-18%), e2e 11187/11179 -> 11730/11740 tok/s (+4.9/+5.0%); 35B kernel 1 131.7/132.4 -> 104.7/104.4 (-21%), kernel 2 101.7/101.9 -> 82.9/82.5 (-19%), e2e 29487/29394 -> 30990/30403 (+5.1/+3.4%). Precision: unit-test state diff vs the fused kernel 9.5e-7 -> 1.3e-6 (4 heads) / 1.2e-6 -> 1.5e-6 (48 heads), Y 6.1e-5 unchanged. The Qwen3.8-27B deterministic PPL over 13.8k tokens is no judge below ~0.5%: fused 4.6283, #1849 4.6148, #1850 4.6273, Y_A-plain 4.6385, this build 4.6365, all with 1e-6-class state diffs. Mechanism, from `diagnostics.dump_hidden_dir` + `tools/analysis/layer_ab_diff.py` on a 300-token prompt: the layer-0 output already differs 4e-4 relative between the fused and the chunk-parallel scan (the FP16 y output's ulp, 2^-11) and the model amplifies that to 26% relative / cos 0.97 by layer 63; the divergence a block ADDS itself (rel@out - rel@in) is median -0.0003 for the GDN blocks on fused -> #1851 and -0.0000 for #1851 -> this build (attention blocks +0.005 in both), i.e. this change sits in the same class as the shipped fused -> chunk-parallel transition. `GDNScanTest.ChunkparMatchesFused48Heads` now also runs the Qwen3.8-27B head count | 2026-09-02 |
   | Recurrent-snapshot host tier (`server.recurrent_snapshot_host_mb`, default 2048) | SHIPPED default-on | The hybrid prefix cache is the `RecurrentSnapshotStore`: 256 MiB of device slots = 3 slabs on Qwen3.8-27B (79.5 MiB each), so with more concurrent multi-turn sessions than slots every session's snapshot is evicted before its next turn and the whole history is prefilled again. Measured (Qwen3.8-27B-NVFP4, 8 interleaved chat sessions x 3 turns of ~1.1k new tokens each, streaming TTFT, `tools/analysis`-style client, two runs each): TTFT turn 2 median 324/322 ms at 8 sessions vs 153/141 ms at 2 sessions (the full-history re-prefill). Evicted device entries now move to pinned host memory on the save stream (D2H, stream-ordered before the buffer's reuse) and restore from there with one H2D (`cudaMemcpyDefault` on the prefill stream): turn 2 at 8 sessions 163/145 ms (-50%), turn 1 162/142 vs 249/234, the 8x3 set 13.5/13.1 s vs 15.9/15.8 s wall (-16%); 32 of 40 restores came from the host tier. 2 GiB pinned = 25 slabs on the 27B; not VRAM, so the planner is untouched. Byte-exact round trip and the held-entry lifetime are covered by `RecurrentSnapshotStoreTest.EvictedEntriesMoveToHostTierAndRestore` | 2026-09-02 |
   | OpenTelemetry span export (`server.otlp_endpoint`, off by default) | SHIPPED | Roadmap gap 8, the export half. `tools/imp-server/tracing.{h,cpp}`: W3C `traceparent` parsing (version 00, lowercase hex, non-zero ids), one SERVER span per generation request from the single accounting point (`log_request_jsonl`, so the Anthropic and Responses shims emit exactly one span each), `queue` (engine queue_ms) / `prefill` (queue end to first token) / `decode` (first token to last) children for streaming requests, OTLP/HTTP JSON (proto3 mapping: 64-bit fields and timestamps as decimal strings), a background thread batching per second or 256 spans, failures counted and logged once. Unit tests `TracingTest.*` (test-core, no GPU), e2e `tests/test_server_tracing.py` in `make test-server` (a stdlib collector on :4318 reached through `host.docker.internal`): both a streaming and a non-streaming chat request land under the caller's span with ids, token counts and model, children inside the root's interval, no prefill/decode split on the non-streaming one | 2026-09-02 |
   | one-H2D decode-step staging (batch-pool pinned mirror, parity-grouped sampler-arg slab, pinned slot table) | NEUTRAL, closed unmerged | @32 two-image pairs vs the PDL build: 1811.4 -> 1807.5 (-0.2%), 1832.9 -> 1813.3 (-1.1%), 1820.2 -> 1832.0 (+0.6%), 2/3 negative; the 8-14 us H2D gaps overlap host work and the mirror memcpy sits on the critical build path. Branch `perf/decode-step-staging` | 2026-08-31 |
   | PDL device half (`griddepcontrol.wait`/`launch_dependents` in the decode kernels, consumer-keyed graph edges) | SHIPPED default-on | final build: M=1 spec-off pairs 83.88 -> 85.31, 83.78 -> 89.23, 83.91 -> 83.63 tok/s (+1.7% median; host level drifted 84-89 at healthy clocks, pairs only); @32 pairs 1705.4 -> 1709.5, 1688.1 -> 1730.3, 1723.5 -> 1728.0 (3/3 positive, +1.3% median; every @32 series today 3/3); idle 13.6% -> 10.8%, merged device intervals 1.27M -> 0.86M gaps. Kernels without a wait stay unregistered (the blanket list raced greedy determinism). Control `runtime.no_pdl=true` | 2026-08-31 |
   | batched ban + penalty sweep (serving default `repetition_penalty` 1.05 + 19 banned ids put every row on the inline chain) | SHIPPED | 1766.9 -> 1774.9 tok/s @32 medians; pairs (base -> new) 1747.7 -> 1772.8 (+1.4%), 1772.4 -> 1774.9 (+0.1%), 1766.9 -> 1782.2 (+0.9%), 3/3 positive; 2 launches per row per step -> 1 sweep per step; re-profiled: steady-window idle 14.9% -> 13.6%, sub-100-us gaps 1127 -> 898 ms per 18 s window, the per-row pair gone from the gap table. Harness `tools/analysis/two_image_conc_ab.sh` | 2026-08-31 |
   | conv1d decode: float4 state, one weight load (`src/compute/ssm.cu`) | SHIPPED | The serving profile @32 (Qwen3.8-27B-NVFP4-vllm, nsys node-trace, 2026-09-02) lists `ssm_conv1d_decode_f32_silu_kernel` at 46320 launches x 9.4 us = 436 ms of a 16.5 s window, 64% of the bandwidth its 46 B per channel need: the per-channel shift loop issued three loads and four stores on the 4-tap state plus four 2-byte weight loads. kernel_size == 4 (every GDN/Mamba2 model here) now reads the state as one float4, writes it back once, loads the four taps as one 8-byte word; explicit fmaf chain in the contracted loop's order. In situ (`tools/analysis/serving_idle_profile.sh`, 32 streams, steady window): 9.61 -> 4.97 us per launch, 273.4 -> 144.2 ms of a 10.4 s window. Two-image A/B @32 (Qwen3.8-27B-NVFP4-vllm, 3 alternating trials x 3 waves, median tok/s): 1833.9/1766.1/1811.1 -> 1844.2/1853.0/1825.6, 3/3 pairs positive. `SSMConv1dTest.DecodeVectorisedBitExact` (6144 channels, random state, bias) holds output and shifted state bit-exact against a CPU fmaf reference; a shift-by-two mutant fails it | 2026-09-02 |
   | GDN decode scan @32, the 20% class of the serving window | AT THE BANDWIDTH FLOOR, closed | The same profile puts `gdn_scan_fused_kernel` at 47040 launches x 57.4 us = 2700 ms of the 16.5 s window (20%). Bytes per launch at 32 streams on Qwen3.8-27B: 48 value heads x 32 sequences x (128 x 128 BF16 = 32 KB) read and written = 96 MB, 61 us at the 1570 GB/s resident ceiling; the 57.4 us average includes the one-sequence prefill-chunk launches. Nothing below the state's bytes is left to take (FP16 state refuted on subnormals, BF16 shipped #1776) | 2026-09-02 |
   | `gemm_cublas` on the hybrid @pp512 (the 24.8%-of-roofline hole above) | PRICED, not built | The roofline run's own counters (`nvfp4-hybrid_pp512_r0_tc_fp16_dst_fp16.ncu-rep`, exported 2026-09-02): the nvjet kernels behind the FP16 GDN projections already run the f16-accumulate tensor path (`hhh`, 0 ops on the fp16->fp32 counter), 294 TFLOPS on the 128x80 tile and 408 on 112x128 (35-48% of the 838 datasheet rate) for the two large shapes, 28-145 TFLOPS on the 16x64 / 64x64 split-K tails. The tails are the alpha and beta projections (N = n_heads) and the 4-call prefill path (`run_gdn`: ssm_in, gate, alpha, beta as separate GEMMs; only M=1 uses the packed 4-in-1 weight). Fusing them for n > 1 needs a row stride on every consumer of the packed output (conv1d prefill, the scan kernels' alpha/beta reads, RMSNormGated) or a deinterleave pass: about 90 us of split-K tails plus a better tile on the merged N per 5 layers of the 3.3 ms window, 2-3% of hybrid pp512. Parked at that price; FP8 for the same GEMMs stays REFUTED (row above) | 2026-09-02 |
   | penalties walk the history, not the vocabulary (`sampling_penalties.cu`) | SHIPPED | The serving profile @32 (Qwen3.8-27B-NVFP4-vllm, nsys node-trace, 2026-09-02) put `apply_penalties_rows_kernel` at 933 launches x 192 us = 179 ms of a 16.5 s window: the sweep gave every vocab entry a pass over the row's history, O(vocab x n_tokens), and the history grows with the generation (agent sessions run 20k-100k tokens). One block per row now counts the history's tokens (16-bit halves of 32-bit words, word atomics) and the thread that claims a count with a CAS applies the penalty once; counts return to zero for the next step. Kernel time at 32 rows on the 151936 vocab (cudaEvent, 50 launches): 300-token history 197.1 -> 10.7-23.3 us, 4096-token history 2659.3 -> 18.6-34.3 us. Logits bit-identical to the sweep, bans included; two mutants (a claim that never zeroes, bans before penalties) fail the identity test. Scratch 9.3 MiB from the T2 arena (`ExecT2Demand::penalty_counts`). Two-image A/B @32 (`tools/analysis/two_image_conc_ab.sh`, Qwen3.8-27B-NVFP4-vllm, 3 alternating trials x 3 waves, median tok/s): base 1748.1/1824.6/1794.8 -> 1835.8/1833.4/1850.4, 3/3 pairs positive | 2026-09-02 |
   | 32-stream serving classes after the 2026-09-02 profile (Qwen3.8-27B-NVFP4-vllm, 1774 tok/s; `tools/analysis/serving_idle_profile.sh` + per-class sums over the steady window) | CAMPAIGN VERDICT | GEMM small-M pair + v2 57% (closed, structural); GDN decode scan 20% at the state's bandwidth floor (row above); attention decode 3.2%: `paged_attention_decode_nvfp4` grid 32 x 24 at 37.8 us per launch, 8 warps share the context blocks and each warp walks its tokens serially (40-77 steps at 300-600 tokens of context), all 768 CTAs resident (6 per SM) so the split-K rule (`compute_splitk_splits`, < 340 CTAs) would not shorten the critical chain, the lever is two tokens per warp iteration (ILP) at a changed reduction order, about 1% e2e, priced and not built; conv1d decode 2.7% -> 1.4% (row above); penalties 1.1% -> 0.1% (row above); norms 1.8%: `rmsnorm_fp16_rowblock_nvfp4` 128 launches x 2.6 us per step, launch-bound, norm+quantize already one kernel and a row reduction fits no GEMM epilogue, closed; alpha/beta cuBLAS 1.2%: 99 `nvjet 32x16x128 splitK` launches x 2.26 us per step, the n > 1 four-call path, a packed alpha+beta GEMM halves the launches for a stride in the scan's reads, 0.6%, priced; memcpy 1.7% of device time, 79 MB recurrent-snapshot D2H copies on their own stream, not idle. Remaining engine-side post of item 0 is therefore the launch-coupled idle (~8%), whose direct attacks (one-H2D staging, host turnaround) are closed above | 2026-09-02 |
   | serving idle re-attributed on the current build (nsys node-trace, steady window) | MEASURED | idle 14.9% of wall; >1 ms gaps (45% of idle) are CUPTI-inflated graph captures at the wave ramp (waves with and without them run 5.57 vs 5.51 s); real idle ~8%: launch density 6.3% (~1350 gaps/step, 0.4 us avg inside the replay; 16.7k gaps of 10-100 us = per-row sampling chain + 8 pageable H2Ds/step at ~14 us), host turnaround 1.9% (~2 gaps/step of ~200 us). Harness: `tools/analysis/serving_idle_profile.sh` | 2026-08-31 |
   | sparse decode at concurrent long context | SHIPPED opt-in | 3 streams x 25k, Qwen3-8B-Q8_0 fp8-KV: 155.6 -> 197.7 tok/s (+27%, 3 alternating trials); metadata now one batched launch per forward. Harness `tools/analysis/serving_sparse_ab.sh` | `attention.sparse_topk_tokens`, #1808 |

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

   Standing state: **imp ahead of vLLM on the GDN hybrid, behind on dense
   NVFP4** (2026-09-02/03, Qwen3.8-27B +24.9% at 32 / +15.6% at 8 / +75.5%
   at 32 with 1082-token prompts; Qwen3-14B-NVFP4 +7.9% at 8 but -7.6% at 32 and -52.9% at 32 with
   982-token prompts, 3/3 each, `BENCHMARKS.md` runs 5-7); the "~1.08x
   pinned" of 2026-08-26 was a cross-client figure. Next engine-side post:
   the dense serving prefill (31.4k prompt tokens in ~5.4 s under decode,
   ~5.8k tok/s, against vLLM's ~1.1 s), attribution first. auto=28 vs pinned=32
   (630 vs 936) is admission, not rotation: 28 sustain full rate under
   continuous arrival. Remaining engine-side posts: launch-coupled idle,
   recurrent-state paging (the lever for 32-way concurrency at LONG context;
   not the limiter at 32 slots). Qwen3.8 port roadmap CLOSED:
   [`plans/2026-08-24-qwen38-port.md`](plans/2026-08-24-qwen38-port.md).
   2026-09-02 sweep of the open items (#1865, #1866, #1867 and the rows
   dated 2026-09-02 above): the dense FA2 prefill instance is tensor-pipe
   bound and its named pipelining lever is priced out; the small serving
   classes are each shipped or priced in the `32-stream serving classes`
   row; the GDN decode scan sits at its bandwidth floor; the hybrid pp512
   GEMM hole is priced at 2-3% and parked. What remains engine-side is the
   launch-coupled idle.

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
   +3-6% isolated, +1-2% mixed, default off.

   The acceptance gap vs the published 83% reference is CLOSED (2026-08-31,
   Qwen3.8-27B-NVFP4-vllm, k=1 greedy, 4 prompt classes): teacher-forced
   p1 91.2/76.4/78.0/88.2 = 83.5% avg (`scripts/mtp_accuracy_bench.sh`),
   verify-path on the same prompts 92.4/78.9/81.4/85.3 = 84.5% - both
   paths agree AND sit on the reference. The earlier 74-78% figures were
   think-workload accept (76-79% here too): acceptance is a property of
   the workload, not an implementation gap; the external 87% p1 belongs
   to the Qwen3-Next-80B head, a different model.

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
8. ~~**No distributed tracing**~~ CLOSED 2026-09-02: `X-Request-Id` echoed
   on every response (2026-08-28), and `server.otlp_endpoint` exports one
   OpenTelemetry SERVER span per generation request with `queue` / `prefill`
   / `decode` children, joined to the caller's trace via W3C `traceparent`
   (OTLP/HTTP JSON, background batches; [`API.md`](API.md) "Request
   tracing", `tests/test_server_tracing.py` plays the collector). Not in it:
   OTLP/gRPC or TLS, metrics/logs export (Prometheus `/metrics` stays the
   metrics surface).

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
