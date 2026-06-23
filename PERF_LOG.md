# PERF_LOG — Agentic Server Hardening

Append-only. Each entry: date, build, protocol, before/after. Newest first.

---

## 2026-06-24 · Concurrency cliff — profiled root cause (corrects earlier note)

**Method:** env-gated per-step instrumentation (`IMP_PROFILE_STEPS`, reverted —
not committed) logging prefill/decode batch size + wall (with a stream sync, so
absolute ms are inflated; ratios/cadence are the signal). Qwen3-4B-2507-Q8_0,
16 concurrent requests sharing a warm ~3367-tok cached prefix, max_tokens=40.

**Measured cadence (one c=16 burst):**
- Prefill: a few requests prefill individually (~70–167 ms each, first is cold)
  as they arrive staggered, then a batch of 14 prefills in ONE step at
  **28.9 ms/req** — i.e. **batched prefill is ~3× cheaper per request than
  single prefills** (same-cycle arrivals amortize the per-step setup).
- Decode steady state: **batch-15 decode ≈ 200 ms/step** vs batch-1 ≈ 130 ms
  (sync-inflated; harness-real ≈ 188 ms vs 36 ms). So 15× the sequences for
  ~1.5× (sync) / ~5× (real) the step time → **decode batches POSITIVELY**
  (≈3× aggregate token throughput at c=15). The high *per-sequence* ITL under
  load is the normal latency↔throughput trade of batching, not a bug.

**Corrected diagnosis:** the earlier entry's "decode is the cliff" intuition was
wrong, and "all prefills serialize" was only half right. Decode scales fine. The
real reducible cost is the **per-request prefill fixed overhead**: a cache-hit
prefill of ~12 uncached tokens still costs ~30–70 ms of GPU time (should be a few
ms), which only amortizes when many requests land in the same scheduler cycle.
Staggered real-agent arrivals each pay it → TTFT grows under concurrency.

**Next lever (focused follow-up):** nsys a single cache-hit prefill (no profiling
sync) to attribute the 30–70 ms — suspects: prefill runs eager (no CUDA-graph
capture, `executor_workspace_buffers.cu:923` notes graph capture disabled when
the largest NVFP4 weight exceeds the 512 MiB workspace cap), per-prefill
workspace `ensure_*`, green-context reconfig, metadata upload. Reducing it (or a
graph-captured cache-hit prefill fast path) directly lowers TTFT under load
without touching decode or the −2% single-stream gate. Deep but well-scoped.

---

## 2026-06-24 · Phase 5b — deterministic mode validation (existing feature)

**Feature:** opt-in ordered MoE reduction via `--set runtime.deterministic=true`
(or `IMP_DETERMINISTIC=1`) — already implemented (`moe_routing.cu` deterministic
kernels, wired through `deterministic_gemm`). No code change; this is a validation.

**Protocol:** Qwen3-30B-A3B-NVFP4-Modelopt (MoE), greedy (temp=0, seed=1),
max_tokens=220, same prompt, server single-stream (batch-1), md5 of response.

| mode | result |
|---|---|
| OFF | warmup run + steady runs diverge in length/hash (run1 965 vs runs2-5 1000) |
| **ON** | warmup differs (cold), **runs 1-5 bit-identical** (md5 d25564…, len 980) |

**Reading:** deterministic mode delivers ≥5-run bit-identical steady-state output
(Phase-5b acceptance met). Caveat: the FIRST request after model load is not
reproducible even with the flag ON (cold cuBLAS-algo / graph-capture / workspace
warmup) — discard one warmup turn for bit-exact reproduction. Per-request
determinism is a **non-goal**: MoE kernel selection is global per launch, so
det + non-det requests can't co-batch under continuous batching. Throughput cost
is in the single-block deterministic permute (cheap at decode/batch-1, severe at
large-batch prefill per the code comments) — keep it opt-in, server-flag only.

---

## 2026-06-24 · Phase 7 — agent benchmark harness baseline

**Build:** `feat/agentic-server-hardening` @ Phase 6 (commit 27b08582), CUDA 13.3, `imp:test`.
**Tool:** `tools/agent_bench.py` (stdlib; streaming SSE, threaded concurrency).
**Server:** `imp-server --model Qwen3-4B-Instruct-2507-Q8_0.gguf` (defaults: prefix_cache ON,
max_batch_size auto→29, KV 8803 blocks / 140848 tokens, max-concurrent 64). Single RTX 5090.
**Protocol:** static prefix ~3367 tok (cache_prompt pinned) + short dynamic suffix, max_tokens=64,
2 warmup turns discarded, n = max(4×concurrency, 8) streamed requests per level.

### Prompt-cache TTFT (single stream)
| | TTFT |
|---|---|
| cold (fresh prefix, cached=0) | 370.8 ms |
| warm (cached=4528/4541) | 221.8 ms |
| **speedup** | **1.67×** (streaming, 3367-tok prefix) |

Non-stream control probe at 4541-tok prefix: cold 0.406 s (cached=0) → warm 0.068 s
(cached=4528) = **6.0×**. Cache sharing verified under 8-way concurrency: every concurrent
request reports `cached=4512/4524` (shared, not recomputed).

### TTFT / ITL under concurrency (ms)
| concurrency | TTFT p50 | p90 | p99 | ITL p50 | ITL p99 |
|---|---|---|---|---|---|
| 1  |   225.3 |   230.5 |   234.2 |  36.3 |  36.8 |
| 4  |   261.3 |   270.1 |   270.6 |  24.1 |  39.9 |
| 16 |  1443.1 |  1455.1 |  1458.2 | 188.2 | 190.6 |
| 64 | 17375.9 | 32301.4 | 32726.6 | 289.1 | 293.7 |

### Reading
- **Real streaming confirmed:** TTFT (225 ms) ≪ E2E — first token bounded by prefill, not full
  generation. (Phase-1 streaming was already CLOSED per the scout audit.)
- **Prompt caching is the headline agentic win:** 1.67–6× lower TTFT for a shared static prefix,
  shared correctly across concurrent requests.
- **Concurrency frontier (open optimization target):** TTFT is excellent at c≤4 (225→261 ms) but
  degrades sharply at c≥16 (1.4 s) and c=64 (17 s p50), even though the shared prefix is cached
  and KV/batch capacity are not exhausted (16 < batch 29, KV fits). The cost is in prefill
  admission/scheduling under concurrency, not KV or cache. This is the #1 lever for "fastest
  agentic server under load" — flagged for a dedicated continuous-batching/prefill-pipelining
  investigation. Not a regression (no prior agent baseline existed); this entry IS the baseline.

  **Root cause (diagnosed):** `Engine::step()` runs *all* prefills in the batch sequentially
  (`step_prefill` loops `step_prefill_one` per request, `engine_scheduler.cpp:367-368`) and only
  *then* runs one decode step (`engine_scheduler.cpp:75-85`). Decode produces the first token, so
  every concurrent request's first token waits for the **entire** prefill batch to finish. With
  16 cache-hit requests the prefill work per request is tiny (~12 uncached tokens) but each is a
  separate non-graph forward with ~90 ms fixed overhead → 16×90 ms ≈ the observed 1443 ms (note
  p50≈p90≈p99: they all unblock together). Two candidate fixes, both non-trivial: (a) ragged
  *batched* prefill — one forward over the whole prefill batch (proper fix, deep: ragged attention
  + chunked-prefill + graph interplay); (b) interleave decode between prefills / cap prefills per
  step (cheaper, latency-fairness trade). A per-step prefill-*count* cap would not touch single-
  stream pp/tg (batch size 1), so the −2% throughput gate is not at risk — but the win is bounded
  by the per-prefill fixed overhead, so (a) is the real lever. Deferred to a profiling-led effort.

### Gate status
- Decode/prefill throughput: untouched by Phases N1/5a/6 (host-side + KV-persist only) — no
  hot-path kernel change, `−2%` throughput gate not at risk.
- TTFT p50 @ c≤4 is bounded by prefill, not E2E (Phase-1 acceptance).
- Warm-cache TTFT < cold-cache TTFT by a clear margin (Phase-2 acceptance).
