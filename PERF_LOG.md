# PERF_LOG — Agentic Server Hardening

Append-only. Each entry: date, build, protocol, before/after. Newest first.

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

### Gate status
- Decode/prefill throughput: untouched by Phases N1/5a/6 (host-side + KV-persist only) — no
  hot-path kernel change, `−2%` throughput gate not at risk.
- TTFT p50 @ c≤4 is bounded by prefill, not E2E (Phase-1 acceptance).
- Warm-cache TTFT < cold-cache TTFT by a clear margin (Phase-2 acceptance).
