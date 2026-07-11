# Determinism

What imp guarantees about run-to-run reproducibility, what `[runtime]
deterministic` adds, and the documented limits (tracked in issue #554).

## Shipped guarantees

`[runtime] deterministic = true` (legacy env `IMP_DETERMINISTIC=1`) is the
opt-in full-reproducibility mode for temperature=0 evals. It eliminates the
known run-to-run non-determinism sources by selecting deterministic kernel
variants:

- **MoE token routing** — atomic expert-bucket scatter ordering.
- **Top-k sampling** — atomicMax/atomicAdd softmax-stat races (single-block
  path, `top_k <= 128`).
- **GEMM** — implies `deterministic_gemm` (cuBLASLt `no_reduce_split`;
  timing-based algo selection is itself a non-determinism source).

With it ON, the gated guarantees (`DetEvalE2ETest`, PR #542) are:

- **Greedy output bit-identical** across runs in the *same context* and
  across *fresh processes* (incl. GDN/hybrid models such as Qwen3.6-35B).
- **Perplexity NLL bit-stable** (`imp_perplexity` / `imp-cli --perplexity`)
  — the teacher-forced harness is the determinism-proof A/B instrument.

The mode is applied engine-side (effective through the C API and server, not
just the CLI tools). Costs a little throughput; strictly OFF by default —
the default path runs the exact same kernels as before with zero overhead.

## Default-mode guarantee (since the request-order-independence fix)

Without `[runtime] deterministic`, the DEFAULT path now guarantees greedy
**request-order independence within a process**: identical greedy requests
produce identical output no matter how many requests preceded them. Three
pieces make this hold (all landed together):

- `runtime.warmup` defaults to **true**: engine warmup pre-arms the decode
  graph pool, so the first real request starts with the same graph state as
  every later one.
- `CudaGraphRunner::mark_process_warm()` — warmup's teardown used to reset
  the per-runner eager pre-capture step, which then executed only in the
  FIRST real request: one step on a numerically different kernel mix
  (eager vs captured graph), flipping greedy output on near-tie logits.
- Scheduler gates use `graph_path_available()` instead of `is_ready()` —
  gating loop/pipeline entry on `is_captured()` deferred those paths by one
  step on the first request only.

This was the documented "30B-NVFP4-MoE greedy nondeterministic at temp=0"
flipper: repro was 1 divergent + N identical runs, always the first request
of a process. Measured after the fix: 3 fresh server processes x 12 greedy
requests on Qwen3-30B-A3B-NVFP4 — 36/36 byte-identical (even across
processes, though cross-process stability additionally depends on cuBLAS
algo selection; see `deterministic_gemm` for the hard guarantee).

`runtime.warmup=false` restores the old init time and with it the
first-request asymmetry — acceptable for dev/CI, not for evals.

## Known limits

These are the documented boundaries of the guarantee. They are deliberate
(perf or upstream-API constraints), tracked in issue #554, and live here so
they are not only discoverable as code comments.

### 1. Dense greedy logit ties

Exactly-tied logits can resolve to different argmax tokens across kernel
paths and runs — the FP values are bit-identical, but tie-breaking is not
specified across paths. Greedy-token A/B comparisons on tie-heavy prompts
(synthetic lists, repetitive corpora) are therefore **invalid as a
correctness signal**; use teacher-forced NLL instead (this is why
`ChunkedPrefillTest` moved from byte-equality to NLL gates in PR #553).

### 2. CUB top-k is not tie-stable for `top_k > 128`

`src/compute/sampling.cu` (`DeviceTopK::MaxPairs`): the CUB path runs with
`determinism::not_guaranteed`, and the descending radix sort is not
guaranteed stable on the token index for bit-identical probabilities — two
tied tokens can swap order between runs. The single-block path
(`top_k <= 128`) tie-breaks by index and is fully deterministic. Fix path if
ever needed: fold the vocab index into the sort key (`(prob, -index)`) or
request `determinism::guaranteed`.

### 3. `typical_p` shared-memory FP atomicAdd

`src/compute/sampling.cu` (bucket histogram): per-bucket probability mass is
accumulated via shared-memory FP `atomicAdd`, whose ordering is
scheduling-dependent. Under deterministic mode this remains a documented
exception — `typical_p` is not part of the temp=0 eval surface.

### 4. GDN cross-context-in-process

`tests/test_determinism_e2e.cpp`:
`DISABLED_GreedyReproducibleAcrossFreshContexts` /
`DISABLED_PerplexityBitIdenticalAcrossFreshContexts` — on GDN/hybrid models,
creating a *new context inside the same process* may not reproduce
bit-identically (VRAM-layout-sensitive recurrent-state slots). Same-context
and fresh-process reproducibility ARE guaranteed (see above). For
reproducible eval sweeps over multiple contexts: one process per context.

## Recipe: reproducible evals

```ini
# imp.conf
[runtime]
deterministic = true
```

- Compare **teacher-forced NLL** (`imp-cli --perplexity`), not greedy bytes.
- temperature=0 / greedy only on prompts without logit ties, `top_k <= 128`.
- GDN models: fresh process per context.
- `imp-cli` logs to stdout — strip log lines before hashing output.
