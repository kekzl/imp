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
| - | A prefix-cache hit is not bit-equal to a fresh prefill | the same prompt reaches cuBLASLt with a different M dimension than a fresh prefill (different algo pick, possibly a different split-k reduction); a greedy decision with a margin smaller than that drift lands either way, and the first request of a process runs fresh so it is the one that differs; scale: paths agree to <= 5e-3 logprob at every position with identical top-5 sets, the flip happened on a 0.018-nat margin. **Only `server.prefix_cache = false` removes it**; `deterministic_gemm` and `deterministic = true` move which near-tie the short probe lands on but do not remove the phenomenon over longer generations (see probe table below) | #1314 |
| - | Burst-chunked decode is a second order-independence source, found 2026-09-21 on Qwen3.8-Flash-Next-NVFP4 | `degen_suite.py --only kv-growth` asks the same greedy question before and after the KV pool grows; defaults FAIL (`2,3,5,7,...` becomes `2, 3, 5, 7, ...`), `server.prefix_cache=false` FAILs, `deterministic_gemm=true` FAILs, `CUBLAS_WORKSPACE_CONFIG=:4096:8` FAILs; `runtime.decode_burst=0` PASSes, `runtime.cuda_graphs=never` PASSes, `runtime.deterministic=true` PASSes (it sets `decode_burst` unbounded); three repeats on a pool that does not grow are byte-identical, so the trigger is growth, not run-to-run noise. **For order-independent greedy output: `server.prefix_cache = false` AND `runtime.decode_burst = 0`.** `[runtime] deterministic` does **not** hold on the server path with prefix caching on (#1314's title) | #1314 |
| - | Batch invariance is out of scope (#1314's other half): a request served alongside 45 unrelated ones can answer differently than served alone | **joining a batch changes the answer, not just its last digits**: solo and batched runs genuinely hand the GEMMs different shapes, so they are not bitwise equal; on a real NVFP4 checkpoint, teacher-forced Qwen3-14B-NVFP4, M=1 vs M=32, **7 of 64 greedy tokens pick a different argmax**, mean NLL 3.4403 -> 3.4889; no flag makes batched and solo bit-equal (`gemm.nvfp4_smallm=false` does not shrink the gap). For output independent of concurrent traffic: pin batch composition or serve at batch 1 | [`PERF.md`](PERF.md) "Batch invariance", `BatchInvarianceTest.NativeNvfp4DecodeSoloVsBatched` (`make test-e2e`) |
| 1 | Dense greedy logit ties | exactly-tied logits can resolve to different argmax tokens across kernel paths and runs; the FP values are bit-identical, tie-breaking is not specified across paths. Greedy-token A/B on tie-heavy prompts (synthetic lists, repetitive corpora) is **invalid as a correctness signal**; use teacher-forced NLL instead (`ChunkedPrefillTest` moved from byte-equality to NLL gates, PR #553) | - |
| 2 | CUB top-k is not tie-stable for `top_k > 128` | `src/compute/sampling.cu` (`DeviceTopK::MaxPairs`): runs with `determinism::not_guaranteed`, descending radix sort not guaranteed stable on the token index for bit-identical probabilities; the single-block path (`top_k <= 128`) tie-breaks by index and is fully deterministic. Fix path if ever needed: fold the vocab index into the sort key `(prob, -index)`, or request `determinism::guaranteed` | - |
| 3 | `typical_p` shared-memory FP atomicAdd | `src/compute/sampling_filters.cu` (bucket histogram): per-bucket probability mass accumulated via `atomicAdd`, scheduling-dependent order. **CLOSED**: under `runtime.deterministic` each warp owns a histogram row, lanes hitting one bucket per iteration summed in lane order (`__match_any_sync`), then a fixed-order cross-warp sum; the atomic path stays the default | `SamplingTest.TypicalPDeterministicPathIsBitStableAndMatchesAtomicPath` |
| 4 | The CUTLASS NVFP4 GEMM is not gated | `runtime.deterministic` reaches 5 files, 7 reads, all through `process_diag_deterministic_gemm()`: `gemm.cu` (2), `sampling_topk_topp.cu` (1), `sampling_filters.cu` (1), `moe_routing.cu` (2), `moe_routing_permute.cu` (1). `gemm_cutlass_grouped_3x.cu`, the primary GEMM for NVFP4 weights and every GGUF quant, reads none of them. `tools/check_determinism_sites.py` pins the per-file read count (scans all of `src/` with comments stripped; `--selftest` plants 8 drifts, part of the `docs` gate group). Measured 2026-08-23, Qwen3.8-27B-NVFP4, teacher-forced NLL over `tools/analysis/ppl_corpus.txt`, 3 fresh processes: `deterministic=true` gives 1.3113/1.3113/1.3113; `deterministic=false` gives 1.3113/1.2889/1.2889. The mode **does** make an NVFP4 checkpoint reproducible through the sites it covers; nothing in the CUTLASS path is pinned, so a future change there is caught by neither the flag nor the gate. Greedy bytes cannot see this (the same 6 runs produced 1 identical output); compare NLL, not bytes | #1574 |
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
