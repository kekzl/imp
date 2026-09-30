<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# Determinism

What imp guarantees about run-to-run reproducibility, what `[runtime] deterministic` adds, and the
documented limits (tracked in issue #554).

## Guarantees

| mode | guarantee | mechanism | source |
|---|---|---|---|
| `[runtime] deterministic = true` (legacy env `IMP_DETERMINISTIC=1`) | greedy output **bit-identical** across runs in the same context and across fresh processes (incl. GDN/hybrid models such as Qwen3.6-35B); perplexity NLL bit-stable (`imp_perplexity` / `imp-cli --perplexity`) | selects deterministic kernel variants: MoE routing (atomic expert-bucket scatter ordering on the FP32-compute fallback, `moe_routing_permute.cu`; the default F16 fused scatter has no atomics, so this source is absent on the default path); top-k sampling (single-block path, `top_k <= 128`, atomicMax/atomicAdd race removed); GEMM (`deterministic_gemm`, forces the cuBLASLt path with `no_reduce_split` and no timing-based algo selection; does **not** reach the CUTLASS grouped NVFP4 GEMM, known limit 4) | `DetEvalE2ETest`, PR #542; applied engine-side (C API and server, not just CLI); costs a little throughput, strictly OFF by default with zero overhead |
| default (`deterministic = false`), since the request-order-independence fix | greedy **request-order independence within a process**: identical greedy requests produce identical output no matter how many requests preceded them | `runtime.warmup` defaults **true** (pre-arms the decode graph pool so the first real request starts on the same graph state as every later one); `CudaGraphRunner::mark_process_warm()` (warmup teardown no longer resets the per-runner eager pre-capture step, which used to execute only on the first real request, an eager-vs-captured kernel mix that flipped greedy output on near-tie logits); scheduler gates use `graph_path_available()` instead of `is_ready()` (gating loop/pipeline entry on `is_captured()` deferred those paths by one step on the first request only) | measured after the fix: 3 fresh server processes x 12 greedy requests, Qwen3-30B-A3B-NVFP4, 36/36 byte-identical; `runtime.warmup=false` restores the old first-request asymmetry, acceptable for dev/CI, not for evals |
| batch composition (any mode) | **a batch neighbour's content cannot reach another row, bit-exactly**: two batches of identical shape and row lengths, differing only in what neighbouring sequences contain, produce bit-identical logits for the row under test | `ForwardPassTest.DecodeLogitsInvariantToBatchComposition`; a mask fault, a padding leak or a block-table mixup (the #1044/#1045 class) breaks it | hard guarantee; NOT the same as batch invariance, which is out of scope (known limits) |

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
| - | A prefix-cache hit is not bit-equal to a fresh prefill (dense FP16 KV: closed by #2152) | a restored prompt re-prefills a shorter tail chunk than a fresh prefill. Before #2152 three kernels were picked by row count: RMSNorm row-block at 2..64 rows (`layernorm.cu`), fused QK-norm+RoPE at n <= 64 (`executor_attention.cpp`), Q8_0/Q4_K IMMA split-K at M <= 32 (`mmq_q8_imma.cu`); a 1-row tail took the M=1 decode kernels. Qwen3-8B-Q8_0 first-token logprob moved 0.021 (FP16 KV) and 0.418 (FP8 KV) between chunk 0 and 336. Since #2152 the choice depends on shape only and prompt chunks keep >= 33 rows (`runtime/prompt_tail.h`): FP16 KV bit-identical at chunk 0/176/300/336, `tools/analysis/prefix_resend/prefix_resend_probe.py` 0/2 FAIL (main 1/2). Open: FP8 KV 0.105 (E4M3 mantissa, separate K/V scales measured equal: V round-trip error 0.0265 both), MoE router GEMM and hd=512 attention closed by #2167 under `runtime.deterministic` (Gemma-4-26B-A4B NVFP4 chunk 0/288/336 bit-identical), cuBLASLt algorithm per M bucket (`gemm.cpp` `bucket_m`) | #1314, #2152, #2167 |
| - | Burst-chunked decode as a second order-independence source on Qwen3.8-Flash-Next-NVFP4: **CLOSED** by 478ba983 (#2150) | found 2026-09-21: `degen_suite.py --only kv-growth` FAILed with defaults and PASSed with `runtime.decode_burst=0`, `runtime.cuda_graphs=never` or `runtime.deterministic=true`. Cause: a graph-replayed decode step left the executor-wide PLE "prepared" flag set (per `InferenceState::ple_host_ready` since the per-sequence PLE state), so the next prefill ran on the previous request's PLE n-gram rows, context and conv state. Since 478ba983: kv-growth 0 FAIL in 3/3 runs (v0.45.0: 1 FAIL in 1/1), CHANGELOG [Unreleased] Fixed. Remaining (478ba983): in graph mode the first request after start still differs from later ones (ctx-bucket capture topology), no content leak. `[runtime] deterministic` does **not** hold on the server path with prefix caching on (#1314's title) | #1314, 478ba983 |
| - | Batch invariance is out of scope (#1314's other half): a request served alongside 45 unrelated ones can answer differently than served alone | **joining a batch changes the answer, not just its last digits**: solo and batched runs genuinely hand the GEMMs different shapes, so they are not bitwise equal; on a real NVFP4 checkpoint, teacher-forced Qwen3-14B-NVFP4, M=1 vs M=32, **7 of 64 greedy tokens pick a different argmax**, mean NLL 3.4403 -> 3.4889; no flag makes batched and solo bit-equal (`gemm.nvfp4_smallm=false` does not shrink the gap). For output independent of concurrent traffic: pin batch composition or serve at batch 1 | [`PERF.md`](PERF.md) "Batch invariance", `BatchInvarianceTest.NativeNvfp4DecodeSoloVsBatched` (`make test-e2e`) |
| 1 | Dense greedy logit ties | exactly-tied logits can resolve to different argmax tokens across kernel paths and runs; the FP values are bit-identical, tie-breaking is not specified across paths. Greedy-token A/B on tie-heavy prompts (synthetic lists, repetitive corpora) is **invalid as a correctness signal**; use teacher-forced NLL instead (`ChunkedPrefillTest` moved from byte-equality to NLL gates, PR #553) | - |
| 2 | CUB top-k is not tie-stable for `top_k > 128` | `src/compute/sampling.cu` (`DeviceTopK::MaxPairs`): runs with `determinism::not_guaranteed`, descending radix sort not guaranteed stable on the token index for bit-identical probabilities; the single-block path (`top_k <= 128`) tie-breaks by index and is fully deterministic. Fix path if ever needed: fold the vocab index into the sort key `(prob, -index)`, or request `determinism::guaranteed` | - |
| 3 | `typical_p` shared-memory FP atomicAdd | `src/compute/sampling_filters.cu` (bucket histogram): per-bucket probability mass accumulated via `atomicAdd`, scheduling-dependent order. **CLOSED**: under `runtime.deterministic` each warp owns a histogram row, lanes hitting one bucket per iteration summed in lane order (`__match_any_sync`), then a fixed-order cross-warp sum; the atomic path stays the default | `SamplingTest.TypicalPDeterministicPathIsBitStableAndMatchesAtomicPath` |
| 4 | The CUTLASS NVFP4 GEMM is not gated | `runtime.deterministic` reaches 6 files, 8 reads through `process_diag_deterministic_gemm()`: `gemm.cpp` (2), `cublas_gemm_algo.h` (1, fixed algorithm instead of `CUBLAS_GEMM_AUTOTUNE` for every `cublasGemm*Ex`, #2168), `sampling_topk_topp.cu` (1), `sampling_filters.cu` (1), `moe_routing.cu` (2), `moe_routing_permute.cu` (1). Two #2167 gates read `process_diag_deterministic()` (the flag as set; Gemma-4 and FP8 KV promote only `deterministic_gemm`): `executor_attention_internal.h` (`attention.hd512_prefill=auto`, hd=512 prefill on the fixed-order FMHA) and `executor_forward_moe_batch.cu` (row-invariant FP32 router GEMM). `gemm_cutlass_grouped_3x.cu`, the primary GEMM for NVFP4 weights and every GGUF quant, reads none of them. `tools/check_determinism_sites.py` pins the per-file read count (scans all of `src/` with comments stripped; `--selftest` plants 8 drifts, part of the `docs` gate group). Measured 2026-08-23, Qwen3.8-27B-NVFP4, teacher-forced NLL over `tools/analysis/ppl_corpus.txt`, 3 fresh processes: `deterministic=true` gives 1.3113/1.3113/1.3113; `deterministic=false` gives 1.3113/1.2889/1.2889. The mode **does** make an NVFP4 checkpoint reproducible through the sites it covers; nothing in the CUTLASS path is pinned, so a future change there is caught by neither the flag nor the gate. Greedy bytes cannot see this (the same 6 runs produced 1 identical output); compare NLL, not bytes | #1574 |
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
