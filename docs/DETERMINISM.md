<!--
layer: L1
audience: operators
verified: 2026-11-03
commit: 006238cc
-->

# Determinism

What imp guarantees about run-to-run reproducibility, what `[runtime] deterministic` adds, and the
documented limits (tracked in issue #554).

## Guarantees

| mode | guarantee | mechanism | source |
|---|---|---|---|
| `[runtime] deterministic = true` (legacy env `IMP_DETERMINISTIC=1`) | greedy output **bit-identical** across runs in the same context and across fresh processes (incl. GDN/hybrid models such as Qwen3.6-35B); perplexity NLL bit-stable (`imp_perplexity` / `imp-cli --perplexity`) | selects deterministic kernel variants: MoE routing (atomic expert-bucket scatter ordering on the FP32-compute fallback, `moe_routing_permute.cu`; the default F16 fused scatter has no atomics, so this source is absent on the default path); top-k sampling (single-block path, `top_k <= 128`, atomicMax/atomicAdd race removed); GEMM (`deterministic_gemm`, forces the cuBLASLt path with `no_reduce_split` and no timing-based algo selection; does **not** reach the CUTLASS grouped NVFP4 GEMM, known limit 4) | `DetEvalE2ETest`, PR #542; applied engine-side (C API and server, not just CLI); costs a little throughput, strictly OFF by default with zero overhead |
| default (`deterministic = false`), since the request-order-independence fix | greedy **request-order independence within a process**: identical greedy requests produce identical output no matter how many requests preceded them | `runtime.warmup` defaults **true** (pre-arms the decode graph pool so the first real request starts on the same graph state as every later one); `CudaGraphRunner::mark_process_warm()` (warmup teardown no longer resets the per-runner eager pre-capture step, which used to execute only on the first real request, an eager-vs-captured kernel mix that flipped greedy output on near-tie logits); scheduler gates use `graph_path_available()` instead of `is_ready()` (gating loop/pipeline entry on `is_captured()` deferred those paths by one step on the first request only) | measured after the fix: 3 fresh server processes x 12 greedy requests, Qwen3-30B-A3B-NVFP4, 36/36 byte-identical; `runtime.warmup=false` restores the old first-request asymmetry, acceptable for dev/CI, not for evals |
| default, cuBLAS(Lt) algorithm choice | the same algorithm per GEMM shape in every process, so seeded sampled output repeats across server starts | `gemm.algo_timed=false` (default): first heuristic candidate that survives a warmup for `gemm()`'s `cublasLtMatmul`, always the first heuristic candidate in `lt_gemm_ex` (`compute/gemm_lt_ex.h`); `gemm.algo_timed=true` times `gemm()`'s candidates and picks near-ties per process | Qwen3.8-27B-NVFP4, 20 seeds at temperature 1.0: timed picks gave 3 different results in 5 server starts (GEMM K=5120 N=48: tile 312, 8 or 9), heuristic 4/4 identical |
| batch composition (any mode) | **a batch neighbour's content cannot reach another row, bit-exactly**: two batches of identical shape and row lengths, differing only in what neighbouring sequences contain, produce bit-identical logits for the row under test | `ForwardPassTest.DecodeLogitsInvariantToBatchComposition`; a mask fault, a padding leak or a block-table mixup (the #1044/#1045 class) breaks it. FP8 x FP8 activations take the checkpoint's static `input_scale` or a per-token scale (`gemm.fp8_static_act_scale`), never a batch-wide absmax: `BatchInvarianceTest.Fp8DenseDecodeRowIndependentOfNeighbours` (Qwen3-8B-FP8, 32 rows, row 0 bit-equal under two neighbour sets in both modes; the o_proj batch-wide absmax before 2026-11-02 moved its NLL 0.0157, 3/64 top-1 flips), `Fp8ActRowsTest` | hard guarantee; NOT the same as batch invariance, which is out of scope (known limits) |

`deterministic_gemm`'s decode cost sits below this host's noise floor.

- `bench_gate.sh` method: discarded warm-up run, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `--prefill-chunk-size 0`.
- 4 alternating pairs on Qwen3-4B-IQ4_NL, `tg128` tok/s off/on: 295.99/285.80, 268.77/280.03, 278.02/281.24, 266.91/271.08 [PROV: hw=RTX5090 model=Qwen3-4B-IQ4_NL].
- Medians 273.4 off / 280.6 on, separated by less than the off arm's own spread (10.9 %).
- Prefill is deliberately not quoted: [`internals/BENCHMARKING.md`](internals/BENCHMARKING.md) rules it out as an A/B signal, and split-k reduction is where a cost would be most plausible.
- "No measurable cost" is a statement about decode on one model only.

## Known limits

Deliberate boundaries of the guarantee (perf or upstream-API constraints), tracked in issue #554.

| # | limit | detail | source |
|---|---|---|---|
| - | A prefix-cache hit is not bit-equal to a fresh prefill (dense FP16 KV: closed by #2152; dense FP8 KV: closed 2026-11-02) | a restored prompt re-prefills a shorter tail chunk than a fresh prefill. Before #2152 three kernels were picked by row count: RMSNorm row-block at 2..64 rows (`layernorm.cu`), fused QK-norm+RoPE at n <= 64 (`executor_attention.cpp`), Q8_0/Q4_K IMMA split-K at M <= 32 (`mmq_q8_imma.cu`); a 1-row tail took the M=1 decode kernels. Qwen3-8B-Q8_0 first-token logprob moved 0.021 (FP16 KV) and 0.418 (FP8 KV) between chunk 0 and 336. Since #2152 the choice depends on shape only and prompt chunks keep >= 33 rows (`runtime/prompt_tail.h`): FP16 KV bit-identical at chunk 0/176/300/336, `tools/analysis/prefix_resend/prefix_resend_probe.py` 0/2 FAIL (main 1/2). FP8 KV (was 0.105): a chunk read its own K/V at FP16 and the gathered past at FP8, so a row moved with its chunk start; since 2026-11-02 the chunk rounds its own rows to the cached E4M3 value first (`round_prefill_kv_to_cache`, `executor_kv_write.cu`; cache bytes unchanged). Qwen3-1.7B AWQ export (`imp-quantize` declares FP8 KV), `degen_suite.py` resend: hits at 64 and 208 tokens and the cache off give 118/118 tokens, logprobs bit-equal (before: 0.30 nats at token 9, top-2 margin 3.71 -> 0.02 at token 13, flip); `PrefixCacheE2ETest.DenseFp8KvHitsAtTwoOffsetsBitEqual`. The FP8 scale is set by the first prefill after warmup, so an `imp-cli` chunk-size A/B (different first chunk) is not bit-equal. Dense FP8 (Qwen3-8B-FP8, default config, `degen_suite.py` resend): 3/3 PASS since 2026-11-02 (main 0/3); the o_proj activation took a batch-wide absmax over the chunk, so a restored shorter tail quantized differently. Per-token scales (`gemm.fp8_static_act_scale=false`) run the own tile data-parallel at every prompt M (cuBLASLt 13.4 refuses FP8 `OUTER_VEC_32F` on sm_120): resend PASS (stream-K by M: FAIL), pp512 30102 tok/s vs 33657-33710 static, decode and PPL equal (10.7629 vs 10.7647). Mixed NVFP4 + FP8 exports keep a row dependence in the FP8 weight GEMMs (cuBLASLt below `gemm.fp8_tile_min_m`): Qwen3-1.7B 15 % budget, drift 0.23 nats, near-tie flip at token 102 (margins 0.071/0.011), `runtime.deterministic` 0.236 with identical text. NVFP4 MoE (default config): fused router and prompt-row stream-K closed 2026-11-02, [NVFP4 MoE prefix resend](#nvfp4-moe-prefix-resend). Open: INT8/INT4/NVFP4/MXFP4 KV keep the FP16 in-chunk read, MoE router GEMM and hd=512 attention closed by #2167 under `runtime.deterministic` (Gemma-4-26B-A4B NVFP4 chunk 0/288/336 bit-identical), cuBLASLt algorithm per M bucket (`gemm.cpp` `bucket_m`). Hybrid GDN (#2556): `runtime.deterministic` runs GDN alpha/beta on the narrow kernel at split 1, CUTLASS NVFP4 without stream-K, FP32 state and the token-sequential scan; Qwen3.8-27B-NVFP4 chunk 2048/2112/2512/2560 6/6 replies equal (main 1/6 at 2048 vs 2560), prefill pp4096 -42 %. Hybrid hits (#2560): the snapshot split left 1..16 prompt rows after the block floor, which ran the M<=32 kernels; hybrid snapshots now sit >= 33 rows before the prompt end and a restore re-prefills >= 33 rows: `session_interleave_ttft.py` 2 x 3 turns, shared prefix and none, 12/12 replies equal to `server.prefix_cache = false` (was 7/12), hits 4368..4416 tokens. SWA window snapshots (#2562): the prefill split, the ragged split and the restore cap keep the same >= 33 rows (gemma-3-12b Q4_K_M, `kv_cache.swa_snapshot_mb=512`: prompt 3545 saves at 3504, was 3536). The finish save stays at the transcript block floor (the ring may not hold the window 33 rows back). SWA sizing is off under `runtime.deterministic`. Reply KV: a finished request publishes its reply blocks (and on hybrids a transcript snapshot) for the next turn, but that KV came from the decode kernels; under `runtime.deterministic` it publishes the prompt only. gemma-3-12b Q4_K_M, fp16 KV, 2 x 3 turns: 6/6 replies equal to the cache off (was 5/6, the hit at 3584 covered 3 decode rows; now 3568). Qwen3.8-27B-NVFP4: 6/6, hits unchanged | #1314, #2152, #2167, #2556, #2560, #2562 |
| - | Burst-chunked decode as a second order-independence source on Qwen3.8-Flash-Next-NVFP4: **CLOSED** by 478ba983 (#2150) | found 2026-09-21: `degen_suite.py --only kv-growth` FAILed with defaults and PASSed with `runtime.decode_burst=0`, `runtime.cuda_graphs=never` or `runtime.deterministic=true`. Cause: a graph-replayed decode step left the executor-wide PLE "prepared" flag set (per `InferenceState::ple_host_ready` since the per-sequence PLE state), so the next prefill ran on the previous request's PLE n-gram rows, context and conv state. Since 478ba983: kv-growth 0 FAIL in 3/3 runs (v0.45.0: 1 FAIL in 1/1), CHANGELOG [Unreleased] Fixed. Remaining (478ba983): in graph mode the first request after start still differs from later ones (ctx-bucket capture topology), no content leak. `[runtime] deterministic` does **not** hold on the server path with prefix caching on (#1314's title) | #1314, 478ba983 |
| - | Decode graph topology carried across requests: re-capture per request set under `runtime.deterministic` (#2182; the Flash-Next repro is not re-measured) | decode attention bakes split-K `num_splits` from `max_context_len` at capture (`attention_paged.cu`); the pow2 context high-water mark that triggers re-capture only grows and survived `imp_context_reset`, so a second identical request replayed the first request's last capture at every step. `runtime.deterministic` now re-captures when the batch's request ids change (`src/runtime/decode_graph_ctx.h`, `DecodeGraphCtx.*`); the default mode keeps the growth-only mark (#948) | #2182, #948 |
| - | Batch invariance is out of scope (#1314's other half): a request served alongside 45 unrelated ones can answer differently than served alone | **joining a batch changes the answer, not just its last digits**: solo and batched runs genuinely hand the GEMMs different shapes, so they are not bitwise equal; on a real NVFP4 checkpoint, teacher-forced Qwen3-14B-NVFP4, M=1 vs M=32, **7 of 64 greedy tokens pick a different argmax**, mean NLL 3.4403 -> 3.4889; no flag makes batched and solo bit-equal (`gemm.nvfp4_smallm=false` does not shrink the gap: that path quantizes activations too). For output independent of concurrent traffic: pin batch composition or serve at batch 1 | [`PERF.md`](PERF.md) "Batch invariance", `BatchInvarianceTest.NativeNvfp4DecodeSoloVsBatched` (`make test-e2e`) |
| 1 | Dense greedy logit ties | exactly-tied logits can resolve to different argmax tokens across kernel paths and runs; the FP values are bit-identical, tie-breaking is not specified across paths. Greedy-token A/B on tie-heavy prompts (synthetic lists, repetitive corpora) is **invalid as a correctness signal**; use teacher-forced NLL instead (`ChunkedPrefillTest` moved from byte-equality to NLL gates, PR #553) | - |
| 2 | CUB top-k is not tie-stable for `top_k > 128` | `src/compute/sampling.cu` (`DeviceTopK::MaxPairs`): runs with `determinism::not_guaranteed`, descending radix sort not guaranteed stable on the token index for bit-identical probabilities; the single-block path (`top_k <= 128`) tie-breaks by index and is fully deterministic. Fix path if ever needed: fold the vocab index into the sort key `(prob, -index)`, or request `determinism::guaranteed` | - |
| 3 | `typical_p` shared-memory FP atomicAdd | `src/compute/sampling_filters.cu` (bucket histogram): per-bucket probability mass accumulated via `atomicAdd`, scheduling-dependent order. **CLOSED**: under `runtime.deterministic` each warp owns a histogram row, lanes hitting one bucket per iteration summed in lane order (`__match_any_sync`), then a fixed-order cross-warp sum; the atomic path stays the default | `SamplingTest.TypicalPDeterministicPathIsBitStableAndMatchesAtomicPath` |
| 4 | The dense NVFP4 GEMM is gated by its callers, not by the flag | `runtime.deterministic` reaches 4 files, 5 reads through `process_diag_deterministic_gemm()`: `gemm.cpp` (2), `sampling_topk_topp.cu` (1), `sampling_filters.cu` (1), `moe_routing_permute.cu` (1); the token permute is stable on both paths since #2465. Two #2167 gates read `process_diag_deterministic()` (the flag as set; Gemma-4 and FP8 KV promote only `deterministic_gemm`): `executor_attention_internal.h` (`attention.hd512_prefill=auto`, hd=512 prefill on the fixed-order FMHA) and `executor_forward_moe_batch.cu` (row-invariant FP32 router GEMM). `src/quant/nvfp4_gemm_tile.cu`, the dense NVFP4 GEMM (own tile since 2026-10-31, CUTLASS removed), reads none of them: its callers pass no stream-K workspace under `runtime.deterministic`, so only the data-parallel 128x64 / 128x128 / 128x256 tiles run, bit-equal to each other and invariant to M. `tools/check_determinism_sites.py` pins the per-file read count (scans all of `src/` with comments stripped; `--selftest` plants 8 drifts, part of the `docs` gate group). Measured 2026-08-23, Qwen3.8-27B-NVFP4, teacher-forced NLL over `tools/analysis/ppl_corpus.txt`, 3 fresh processes: `deterministic=true` gives 1.3113/1.3113/1.3113; `deterministic=false` gives 1.3113/1.2889/1.2889. The mode **does** make an NVFP4 checkpoint reproducible through the sites it covers; a tile change that re-enables stream-K without a workspace check is caught by neither the flag nor the gate. Greedy bytes cannot see this (the same 6 runs produced 1 identical output); compare NLL, not bytes | #1574 |
| 5 | Cross-context-in-process | `tests/test_determinism_e2e.cpp`: `DISABLED_GreedyReproducibleAcrossFreshContexts` / `DISABLED_PerplexityBitIdenticalAcrossFreshContexts`; creating a new context inside the same process may not reproduce bit-identically. Same-context and fresh-process reproducibility ARE guaranteed; for reproducible eval sweeps over multiple contexts, one process per context. Measured 2026-08-10 on `main`, `deterministic=1`: `gpt-oss-20b-mxfp4` (MoE) same-context pass 3/3 (graphs on/off), fresh contexts **fail 2/2**; `Qwen3-4B-Instruct-2507-Q8_0` (dense) passes both. The MoE row carries this limit today, not GDN (the same-context guarantee holds on both); #1337 is the identified fix for the dense half, the MoE half was last seen red before #1341 (names #1299, decode-loop burst boundaries, a code attribution, not an A/B) | #1299, #1337, #1341 |
| 6 | The build is part of the envelope | every CUDA translation unit is compiled with `--use_fast_math` (`cmake/CompilerFlags.cmake`), in both shipped configurations, a deliberate perf choice, stable for a given binary; the guarantees above are about **one binary**, not the source tree. Pin the image, not just the commit, when a result has to be reproducible later | #1576 |

Prefix-cache hit probes (#1314): short probe is the whole `tests/api/test_chat.py` (a 16-token
answer to "What is 2+2?"), `Llama-3.2-3B-Instruct-IQ4_XS`, five fresh servers per arm; long probe
is the three `DetEvalE2ETest` prompts at 96 tokens, three fresh servers per arm.

| arm | short probe | long probe (Qwen3-4B-Q8_0) | long probe (Llama-3.2-3B) |
|---|---|---|---|
| default | **5/5 diverge** | **1 of 3 prompts, 3/3 reps** | **1 of 3 prompts, 3/3 reps** |
| `runtime.deterministic_gemm = true` | 0/5 | **1 of 3 prompts, 3/3 reps** | **1 of 3 prompts, 3/3 reps** |
| `runtime.deterministic = true` | 0/5 | **1 of 3 prompts, 3/3 reps** | - |
| `server.prefix_cache = false` | 0/5 | **0 of 3, 3/3 reps** | - |

The GEMM knobs move the particular near-tie the short probe lands on; over 96 tokens the divergence returns with `deterministic = true` still set.

- `PrefixCacheE2ETest.FreshVsPrefixHitTokenEqual` asserts the strong version of the guarantee and passes: its long multi-block prompt has no margin this narrow.
- The gate is right about what it measures; the promise above is wider than what the gate can see.

## NVFP4 MoE prefix resend

Measured 2026-11-02 on `fix/moe-router-exact-sum`: `degen_suite.py` resend shape (turn 1 generated, turn 2 sent 3 times), default config, max |dlogp| of the chosen token over the 120-token answer before the first flip. The first send re-prefills 218-225 rows, the resend 33-42.

| model, arm | max dlogp | first flip (top-2 margin) |
|---|---:|---|
| Qwen3-30B-A3B, main | 0.0856 | token 23 (0.545) |
| same, without NVFP4 tile or LM-head stream-K, attention or other deterministic sites; `runtime.deterministic_gemm` | 0.0856 each | token 23 |
| same, `moe.router_fused=false` (cuBLAS router) | 0.1377 | token 44 (0.463) |
| same, FP32 row-invariant router, `runtime.deterministic=true`, fix | 0 | none |
| Gemma-4-26B-A4B, main | 0.0607 | token 33 (0.036) |
| same, exact router only | 0.1795 | none |
| same, exact router + NVFP4 tile without stream-K | 0 | none |
| same, exact router + hd512 FMHA, APA/RA2, LM head, `deterministic_gemm` | 0.1795 each | none |
| Gemma-4-26B-A4B, fix | 0 | none |
| Nemotron-3-Nano-30B-A3B, main and fix | 0 | none (hybrid snapshot split: both sends prefill 33 rows) |

- Router: `moe_router_fused` summed fp16 k-tile partials in fp32 with a split-K count picked by M (225 rows split 8, 33 rows split 16). It now sums them exactly in int64 units of 2^-24 (`src/compute/moe_router_fused.cu`), `MoeRouterFusedTest.RowsIndependentOfMAndSplit`.
- NVFP4 tile: 42 rows of Gemma-4's global q_proj (N 8192) ran split-2 stream-K, 218 rows data-parallel. `gemm.nvfp4_prompt_streamk=auto` keeps prompt rows (M >= 33) data-parallel on MoE, `Nvfp4PromptStreamK.PromptRowsInvariantToM`. Real activations often sum exactly in fp32, so the class shows on some prompts only (`MoePrefixHitE2ETest` text: router only).
- Open: the cuBLAS router from 1024 rows under `gemm.cublas_fp16_acc` (Nemotron-H) and dense NVFP4 prompt stream-K.
- Open, MLA (DeepSeek-V2-Lite `imp-quantize` export, 2026-11-03, hits 64 then 240, 32-token greedy text equal, logprobs not; `runtime.deterministic` and graphs off: still red): per-stage dumps put the first divergence at layer 0's FP16 `kv_a_proj_with_mqa` (N 576, uncached tier, cuBLASLt split-K 16/9/8/4 by M bucket); hd=192 prefill attention runs materialized cuBLAS, FA2 and FMHA decline hd=192. `MoePrefixHitE2ETest` skips MLA with this reason.

Cost, `imp-cli --bench` (pp512 + tg128, pp4096, `--prefill-chunk-size 0`) and `two_image_conc_ab.sh` conc 16, 3 alternating pairs, medians main -> fix [PROV: commit=2c10bbe2 date=2026-11-02 hw=RTX5090 model=see-table quant=NVFP4 cuda=13.4 path=own-tile cmd=imp-cli-bench n=3]:

| model | pp512 tok/s | pp4096 tok/s | tg128 tok/s | conc 16 tok/s |
|---|---|---|---|---|
| Qwen3-30B-A3B | 34919 -> 34688 (-0.66 %) | 54945 -> 55058 (+0.21 %) | 380.96 -> 381.94 (+0.26 %) | 1771.4 -> 1779.0 (+0.43 %) |
| Gemma-4-26B-A4B | 37301 -> 37021 (-0.75 %) | 55279 -> 55319 (+0.07 %) | 270.26 -> 269.51 (-0.28 %) | 1850.2 -> 1858.8 (+0.46 %) |
| Qwen3-8B-NVFP4, prompt stream-K off | 48615 -> 47605 (-2.08 %) | 47293 -> 46980 (-0.66 %) | 283.11 -> 282.91 (-0.07 %) | - |

- Dense keeps prompt stream-K (`auto` = on): every Qwen3-8B pp512 run off sat below every run on.
- Router kernel alone, K 2048, us main -> fix: M 2 8.59 -> 7.75, M 16 8.50 -> 7.76, M 512 10.29 -> 10.67, M 2048 19.86 -> 20.66, M 4096 26.20 -> 29.92 (`MoeRouterFusedBench`).

## FP8 KV PPL below FP16 KV

Measured 2026-11-02 at e66bf1d9: `imp-cli --perplexity`, `runtime.deterministic_gemm=true`, `attention.apa_eps=0`, chunk 2048 unless noted. "before" = `round_prefill_kv_to_cache` removed (#3002).

| Model, corpus | FP16 KV | FP8 before | FP8 after |
|---|---:|---:|---:|
| Qwen3-1.7B BF16, `ppl_corpus_45k` | 16.8827 | 16.7016 | 16.7006 |
| same, chunk 1 | 16.8715 | 16.6698 | 16.7103 |
| Qwen3-1.7B BF16, `prose_30k` | 21.8016 | 21.7212 | 21.8740 |
| Qwen3-0.6B BF16, `ppl_corpus_45k` / `prose_30k` | 22.9272 / 29.5530 | 23.3635 / 29.7411 | 23.5063 / 30.1791 |
| Qwen3-8B-NVFP4, `ppl_corpus_45k` chunk 2048 / 512 | 11.3005 / 11.2751 | 11.2968 / 11.2653 | 11.2394 / 11.2766 |
| Qwen3-8B-NVFP4, `prose_30k` | 15.2656 | 15.0877 | 15.2175 |

- E4M3 rounding of K, not a defect: an FP16 cache with K/V rounded in place (scale 1) gives 16.7006 / 21.8740, bit-equal to FP8 KV. K only 16.7003, V only 16.8841; K in layers 3-27 16.6914, layers 0-2 16.9201.
- The scale (`fp8_kv_inv_scale_`) is a no-op here: 91 of 92 layers (1.7B, 0.6B, 8B) calibrate to the 1.0 floor, absmax over the first 2048 tokens < 448 (1.7B K 22-394, V 1.2-398); 0.6B layer 0 1.095. K at scale 2 (no saturation below 896) 16.7086; clipping K at 112 (scale 1/4) 18.4028.
- Broad, grows with context: 1.7B `ppl_corpus_45k` FP8 minus FP16 NLL -0.0108 nats/token, 10806/13536 positions move by > 0.001, -0.005 (tokens 0-2047) to -0.022 (10240-12287). Before vs after #3002: -0.00006.
- The sign follows corpus and model (1.7B prose +0.33 %, 0.6B +2.5 %): judge FP8 KV by per-position NLL distance to FP16, not by PPL against FP16 on one corpus.
- Since #3002 prefill agrees with the decode-shaped path: 1.7B chunk 1 / 512 / 2048 16.7103 / 16.7205 / 16.7006.
- Qwen3-8B-NVFP4 moves more with the harness than with the KV dtype: FP16 KV 11.3005 / 11.2751 at chunk 2048 / 512, 11.3145 with default APA; chunk 1 drops `prose_30k` by 5 % (FP16 14.4738, FP8 14.4631). The #3002 commit's 11.3201 -> 11.2202 (APA on) sits in that band.

### Static scales (`kv_cache.fp8_scale`)

Measured 2026-11-02 at 8ad05028 (main 65f984d0 + this change), same harness.

- `legacy` (default): the 1.0 floor above.
- `checkpoint`: `k_scale` / `v_scale` tensors; every local ModelOpt FP8-KV export ships 1.0, so bit-equal to legacy.
- `calibrate hN`: fixed-text prefill at warmup, `pow2ceil(amax * N / 448)`.
- Mean |dNLL|: per-position distance to FP16 KV (`diagnostics.ppl_dump=full`).

| Model, corpus | FP16 | legacy | cal h1 | cal h2 | cal h4 | mean abs dNLL legacy / h1 / h2 / h4 |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B, `ppl_45k` | 16.8827 | 16.7006 | 16.6987 | 16.7335 | 16.7197 | 0.1301 / 0.1298 / 0.1303 / 0.1299 |
| Qwen3-1.7B, `prose_30k` | 21.8016 | 21.8740 | 21.9419 | 21.9069 | 21.8289 | 0.1261 / 0.1254 / 0.1244 / 0.1249 |
| Qwen3-8B-NVFP4, `ppl_45k` | 11.2587 | 11.2335 | 11.2953 | 11.2769 | 11.2725 | 0.2575 / 0.2565 / 0.2549 / 0.2547 |
| Qwen3-8B-NVFP4, `prose_30k` | 14.9318 | 15.0141 | 14.9080 | 14.9959 | 14.9093 | 0.2857 / 0.2895 / 0.2746 / 0.2790 |
| Qwen3-8B-FP8, `ppl_45k` | 10.7754 | 10.7375 | 10.7699 | 10.7598 | 10.7613 | 0.1247 / 0.1249 / 0.1245 / 0.1251 |
| Qwen3-8B-FP8, `prose_30k` | 14.2089 | 14.2606 | 14.2272 | 14.1509 | 14.2715 | 0.1300 / 0.1302 / 0.1252 / 0.1327 |
| Gemma-4-26B-A4B NVFP4, `ppl_corpus_45k_gemma4_turn` | 17.8489 | 18.4248 | 17.6053 | 18.0085 | 17.7327 | 0.4952 / 0.4694 / 0.4808 / 0.4843 |

- Qwen3: no arm wins. Every arm moves PPL and distance both ways within the spread of the arms themselves (8B-FP8 prose h2 0.1252, h4 0.1327); default stays `legacy`.
- Gemma-4: every static arm beats legacy (h8 17.4282, 0.4787 too): legacy +3.23 % PPL over FP16, calibrate -2.4 % to +0.89 %. Its K is tiny (calibration K amax 0.94-1.78, V 15.2-22.0 = the unscaled `v_norm` bound sqrt(256)), so at scale 1 18.0 % of K entries sit below the E4M3 normal floor 2^-6 (edge counts below). At 87a61491 shared `calibrate` trails legacy on this corpus: [separate K/V scales](#separate-kv-scales-kv_cachefp8_scale_split).
- Power-of-two scales keep the E4M3 normal-range grid of scale 1 (`Fp8KvStaticScale.PowerOfTwoKeepsTheUnitGridOnNormals`): the arms differ only in the subnormal and saturation edges, so the K rounding effect above is the same in all of them.
- Unchanged elsewhere (Qwen3-8B-NVFP4, legacy vs calibrate h2): NIAH 8k/16k/32k x 3 depths 9/9 both; `degen_suite.py --quick` 37 checks, 0 FAIL calibrate (legacy 1 FAIL: the known dense-NVFP4 prefix-resend flip); tg128 452.9 vs 453.7 tok/s (3 runs each); conc 16 aggregate 3472 vs 3605 tok/s (6 pairs, both orders, calibrate ahead in all 6, cause not isolated).

[PROV: commit=8ad05028 date=2026-11-02 hw=RTX5090 model=Qwen3-8B quant=NVFP4 cuda=13.4.3 path=imp-cli+server-api cmd=`imp-cli --bench --bench-pp 512 --bench-reps 5`, `tools/analysis/conc_arms_ab.sh`
       n=3 runs (tg128), 6 trials x 3 waves (conc 16, CONC=16 MBS=16 SEQ=4096) note=kv_cache.dtype=fp8, speculative.mtp_k=0]
- `calibrate` needs one prefill chunk for its 370-390 tokens: at `--prefill-chunk-size 1` it warns and keeps legacy.
- `PrefixCacheE2ETest.DenseFp8Kv*HitsAtTwoOffsetsBitEqual` is red on Qwen3-1.7B BF16 in both modes and on main (identical greedy tokens), green on Qwen3-8B-NVFP4.

### Separate K/V scales (`kv_cache.fp8_scale_split`)

Measured 2026-11-03 on main 87a61491 + this change, same harness (`--prefill-chunk-size 2048`), all arms in one image; main's own image gives the same legacy and `calibrate` PPL on Gemma-4 (17.9006, 18.0870).

- `split` (`checkpoint` / `calibrate` only, default off): K and V get their own power-of-two scale per layer instead of `max(K, V)`. Gemma-4 calibration: K scale 2^-7 (2^-6 on 2 layers), V 2^-3.
- Decode kernels keep one `kv_scale`: `fp8_kv_decode_scales` passes `{softmax_scale * k / v, v}` (exact for powers of two, `KvFp8Scale.DecodeFoldIsExact`); write, prefill round trip and gather take both scales.
- Edge counts: calibration prefill (390 tokens), E4M3 subnormal + flushed share of non-zero entries at each arm's scale (INFO line `fp8_scale=calibrate edges`).

| Model, corpus | FP16 | legacy | cal h2 | split h2 | mean abs dNLL legacy / cal / split | mean dNLL legacy / cal / split |
|---|---:|---:|---:|---:|---|---|
| Gemma-4-26B-A4B NVFP4, `ppl_corpus_45k_gemma4_turn` | 17.5635 | 17.9006 | 18.0870 | 17.6854 | 0.4692 / 0.4739 / 0.4659 | +0.0190 / +0.0294 / +0.0069 |
| Gemma-4-26B-A4B NVFP4, `calib_corpus_gemma4_turn` (37330 tok) | 30.7326 | 30.9120 | 30.8182 | 30.3282 | 0.6051 / 0.6043 / 0.5965 | +0.0058 / +0.0028 / -0.0132 |
| Qwen3-30B-A3B NVFP4, `ppl_45k` | 10.9299 | 10.8848 | 10.9474 | 10.9211 | 0.2586 / 0.2518 / 0.2536 | -0.0041 / +0.0016 / -0.0008 |
| Qwen3-8B-FP8, `ppl_45k` | 10.7786 | 10.7793 | 10.7376 | 10.7130 | 0.1250 / 0.1239 / 0.1226 | +0.0001 / -0.0038 / -0.0061 |

| Model | K edges legacy / shared / split | V edges legacy / shared / split |
|---|---|---|
| Gemma-4-26B-A4B | 17.98 % / 2.35 % / 0.16 % | 1.89 % / 0.24 % / 0.24 % |
| Qwen3-30B-A3B | (K amax > V: shared = split) | 22.14 % / 6.92 % / 0.21 % |
| Qwen3-8B-FP8 | 1.77 % / 0.44 % / 0.35 % | 7.27 % / 3.01 % / 0.20 % |

- Gemma-4 turn corpus, PPL over FP16 by headroom: shared h1 / h2 / h4 +2.35 / +2.98 / +2.76 %, split +0.30 / +0.69 / +2.58 %; legacy +1.92 %. Mean abs dNLL split h1 / h4 0.4757 / 0.4789.
- Split removes the edge values (Gemma-4 K 17.98 % -> 0.16 %) and shifts Gemma-4 NLL down on both corpora (-0.0225 and -0.0160 nats/token vs shared), but the per-position distance to FP16 moves by less than its spread across headrooms (0.4659..0.4789): the rest of the FP8 KV gap is the 3-bit mantissa on normal values, which no power-of-two scale moves.
- Qwen3 neutral: split distance sits between or below the other arms (Qwen3-30B 0.2536 vs 0.2518 / 0.2586, Qwen3-8B-FP8 0.1226 vs 0.1239 / 0.1250), PPL within -0.61..-0.08 % of FP16. `degen_suite.py --quick` on Gemma-4 split: 37 checks, 0 FAIL.
- Decode cost, Gemma-4 FP8 KV, medians of 3 rotated trials, main legacy / own legacy / own split: tg128 299.15 / 299.13 / 299.25 tok/s, conc 16 aggregate 2281.3 / 2306.1 / 2403.0 tok/s (split ahead in all 3 trials, as `calibrate` in #3016, cause not isolated).

[PROV: commit=87a61491+kvsplit date=2026-11-03 hw=RTX5090 model=Gemma-4-26B-A4B-it quant=NVFP4 cuda=13.4.3 path=imp-cli+server-api cmd=`imp-cli --bench --bench-pp 512 --bench-reps 5 --max-tokens 128`, `conc_client.py 16 3`
       n=3 trials per arm note=kv_cache.dtype=fp8, runtime.max_batch_size=16, IGNORE_EOS=1]
- Default stays `legacy`: the Gemma-4 gain does not hold at h4, and `calibrate` skips the persisted prefix-cache restore. For FP8 KV on Gemma-4: `kv_cache.fp8_scale=calibrate` with `kv_cache.fp8_scale_split=true`.

## Recipe: reproducible evals

```ini
# imp.conf
[runtime]
deterministic = true
```

- Pin the binary too: the same image tag, not just the same commit (known limit 6).
- Compare **teacher-forced NLL** (`imp-cli --perplexity`), not greedy bytes.
- `temperature=0` / greedy only on prompts without logit ties, `top_k <= 128`.
- GDN models: fresh process per context (known limit 5).
- `imp-cli` logs to stdout; strip log lines before hashing output.
