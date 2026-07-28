# PERF_LOG — Agentic Server Hardening

Append-only. Each entry: date, build, protocol, before/after. Newest first.

---

## 2026-07-28 · Memory architecture A7 steps 0-2a — hot path untouched, decode/prefill neutral

Three additive commits (Backend + phase guard + I1 gate; arena/block pool/
scratch stack/stable spans; `plan_memory()`), plus the deletion of the dead
`src/core/allocator.{h,cpp}`. **Nothing in the engine calls any of it yet** —
verified: no file outside `src/memory/` includes `backend.h`, `arena.h`,
`block_pool.h`, `scratch_stack.h`, `span.h` or `plan.h`. The only linked-in
change to an existing path is that `imp` no longer compiles two unreferenced
TUs.

Gate-matched harness (Qwen3-8B-Q8_0, `--bench-pp 512 --bench-reps 5
--max-tokens 128 --prefill-chunk-size 0 --set speculative.ngram=false`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, 3 trials, GPU idle before the run, healthy
under load at 2917 MHz SM / 14001 MHz mem):

| metric | baseline (`tests/perf_baseline.json`, pinned 2026-07-26) | now (median of 3) | Δ |
|---|---:|---:|---:|
| pp512 | 12406.87 | 12354.01 | −0.43% |
| tg128 | 287.19 | 284.29 | −1.01% |

Trials — pp512: 12311.00 / 12354.01 / 12405.93 · tg128: 284.29 / 284.13 /
285.07. Both inside the CI gate (5% prefill / 3% decode). The decode −1.01%
is at this box's noise floor rather than a regression: the commits add no
call sites to the forward pass, and this host has documented intra-minute
decode volatility (#999) and whole-day 8-15% depression (#526). Baseline not
re-pinned — nothing intentionally moved perf.

CPU lane: 608 tests green (`test-core`, no GPU). Gates: `File size` OK,
new `Alloc sites` gate OK at 78 files / 713 sites (down from 79 / 717).

---

## 2026-07-17 · CUDA-graph RAII owners (WI-3) — decode neutral

`CudaGraph`/`CudaGraphExec` move-only owners in core/cuda_raii.h; adopted by
CudaGraphCapture, CudaGraphConditionalRunner, SpecVerifyGraph and the
spec-capture locals — every manual `cudaGraphDestroy`/`cudaGraphExecDestroy`
+ null-out pair replaced by reset()/move, throw paths structurally safe.
Semantics preserved 1:1 (exec-update fast path, drop_graph_keep_exec,
mem-trim-on-reset). Validation: GPU suite 0 fail, capture/green-ctx tests
11/11, DegenerationTest 5/5, verify-fast OK. Decode (same harness as
baseline, 5 trials, spec-OFF Coder-30B): tg256 402.59 -> 402.92 (+0.08%,
noise). Note: the "Engine teardown: cleared a leaked CUDA error (graph
update constraint)" WARN in the Degeneration battery is PRE-EXISTING
(reproduced on unmodified main, same day) — the ExecUpdate->reinstantiate
fallback has always left the sticky error for the teardown net; candidate
one-line follow-up: clear it at the fallback site.

## 2026-07-17 · Post-launch error checks (399 sites) + KV prefix-hash double-free fix — decode neutral

Hardening WI-1/WI-2 (branch hardening/launch-checks-and-kv-churn, baseline
`docs/audit/DISPATCH_BASELINE_2026_07_17.md`):
`IMP_CUDA_CHECK_LAUNCH()` (cudaPeekAtLastError — logs at the launch site, does
NOT clear, downstream propagation unchanged) after 399 previously-unchecked
kernel launches in 82 .cu files, plus the KVCacheManager stale-prefix-hash
double-free fix (below). Perf gate, same harness as the Phase-0 baseline
(median of 5 isolated trials, spec-OFF, healthy clocks sampled 2902/13801):

```
docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v $HOME/models:/models imp:test \
  imp-cli --model /models/Qwen3-Coder-30B-A3B-Instruct-FP4 --bench --bench-pp 512 --bench-reps 10 \
  --set speculative.ngram=false
```

| Metric | Baseline (same day) | After | Δ |
|---|---|---|---|
| tg256 @pp512 (graphs ON) | 402.59 | 402.90 | **+0.08% (noise)** |
| pp512 | 19,739 | 20,304 | within restart variance |

No-graphs decode (informational, no before-arm — debug path only): 118.8/126.7/122.1
tok/s over 3 trials; theoretical check overhead <0.1 ms on an 8.4 ms wall step.

**KV double-free root cause** (found by the new `LeakUnderSustainedChurn` test —
free-count exceeded pool size, 34>32): `rollback()` and
`rollback_partial_allocation()` freed hash-REGISTERED blocks to ref 0 without
erasing `block_hash_to_id_`/`block_id_to_hash_`; a later same-prefix
`allocate_blocks_with_prefix` hit the stale entry, took the "actively
referenced — share it" branch and `inc_ref`'d a block sitting in the free
list → the next free pushed it into the free list a second time. Production
trigger: KV-pool pressure during prefix-cache allocation (scheduler
`prefill_allocate_kv_blocks_` → partial-rollback → client retries same
prefix) = silent cross-request KV corruption. Fix:
`free_block_dropping_stale_hash()` in both rollback paths (erase hash entries
when the free drops ref to 0; shared blocks keep their entry) + a loud WARN
guard in the reuse path that treats a ref==0 non-cached hash hit as a miss.
Validation: 45/45 KVCacheManager tests (incl. 200-cycle churn), PrefixCacheE2E
4/4 with Qwen3-4B Q8_0, full GPU suite 0 failures.

## 2026-07-07 · Token-tiled FP8 split-K decode attention (hd=128) — long-ctx decode +51%

`paged_attention_splitk_fp8_pipeline_kernel` was the top GPU-time consumer at
long ctx (70% of decode wall @16k, PERF_LOG 06-18) and latency-bound at ~10%
DRAM: the inner loop commits V[t]+K[t+1] in one cp.async group, so every token
serially waits a global->smem round trip. Replaced (hd=128, block_size % 16 == 0,
knob `attention.fp8_tile`) by a token-tiled kernel: bulk cp.async of whole
16-token KV chunks (double-buffered per warp), 16 parallel QK dots, one
tile-wise online-softmax step, plus a wave-quantization-aware split count (the
tile kernel is smem-capped at 1 block/SM; the shared heuristic left a ~2.07-wave
grid with a near-idle third wave).

ncu @16k, Qwen3-Coder-30B-A3B-FP4 (FP8 KV), deterministic kernel time x30:
99.98 us -> 69.3 us (tiling) -> **51.5 us (-48.5%)** with wave-aware splits;
DRAM 9.6 -> 18.9%, long-scoreboard stall 4.7 -> 1.9.

e2e decode (graphs ON, 3 trials x 3 reps, clocks healthy 2932/13801):

| ctx | pipeline | tile | delta |
|---|---|---|---|
| 512 | 342.2 | 395.3 | +15.5% |
| 4096 | 267.2 | 339.6 | +27.1% |
| 16384 | 157.1 | 237.4 | **+51.1%** |

(The 06-18 "split-K knob = 0.00 e2e" refutation measured split-count changes on
the old latency-bound kernel; the kernel itself was on the critical path after
all — halving its time moves short-ctx decode too.)

Verified: test-attention + test-kv green (SplitKConsistency covers the new
kernel), degen_suite 22/0 vs the live server, needle recall @14.3k tokens PASS,
stderr clean. Remaining headroom: kernel is L2-bound at ~19% DRAM; GQA batching
(one block computes all 8 q-heads per kv-head, /8 L2 traffic) is the next lever.

---

## 2026-06-24 · CORRECTION — the "concurrency cliff" was a harness artifact

The Phase-7 baseline below (c=16 TTFT 1443 ms, c=64 17376 ms) and the
"concurrency-prefill cliff" it spawned were **wrong** — they measured the Python
*client*, not the server. The threaded harness parsed N concurrent SSE streams in
Python threads; at 16+ streams the GIL serializes the per-token JSON parsing, so
the reported TTFT/ITL was the client's parsing throughput. Re-measured by driving
each request with a separate **`curl` OS process** (true parallelism, no shared
interpreter; TTFT = curl `time_total` for a `max_tokens=1` request = prefill + one
decode = time to first content token):

| concurrency | TTFT p50 (TRUE, curl) | old threaded harness | inflation |
|---|---|---|---|
| 1  |   37 ms |   225 ms | 6× |
| 4  |   98 ms |   261 ms | 2.7× |
| 16 |  237 ms |  1443 ms | 6× |
| 64 |  905 ms | 17376 ms | **19×** |

Prompt cache cold→warm (max_tokens=1): **207.7 → 28.9 ms = 7.2×** (the threaded
harness's 1.67× was also GIL-depressed). Single-stream ITL is **~4.6 ms**, not the
36 ms the threaded harness reported — even c=1 was client-bound.

**Conclusion:** imp's server concurrency is GOOD — c=16 TTFT ~237 ms, c=64 ~905 ms
on one RTX 5090 with a shared cached prefix. There is NO cliff. The earlier
"prefill serialization / eager-dispatch" root-cause stands as a *modest, optional*
lever (c=64's ~900 ms is ~64 cache-hit prefills at ~8 ms each + a decode step —
batching the prefills could shave it) but it is NOT urgent and NOT a defect. The
flagship optimization is **deprioritized**. The harness (`tools/agent_bench.py`)
was rewritten to use curl-process concurrency; the numbers above are the real
baseline. ITL still grows with concurrency (c=1 4.6 ms → c=64 ~594 ms) — that is
the normal batched-decode latency↔throughput trade, working as intended.

Lesson: never measure server concurrency with a GIL-bound single-process client.

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

---

## 2026-06-24 · Soundness & hardening audit (audit/soundness-hardening)

Model: Qwen3-Coder-30B-A3B NVFP4, CUDA 13.3, `imp-cli --bench`, 7 reps/run, 2 cold
restarts. Decode tg256 is the gate signal (prefill pp512 carries the ±2.6× cuBLAS
restart variance). GPU verified free + warm-clocked. Full ledger: AUDIT.md (pass 2),
docs/audit/AUDIT_REPORT.md.

### Phase-0 baseline → post-fix (all soundness fixes)
| shape | baseline | post-fix | gate (3% band) |
|---|---|---|---|
| tg256 @ pp512  | 341.6 | 342.2 | ≥ 331.4 ✓ |
| tg256 @ pp2048 | 322.0 | 321.7 | ≥ 312.4 ✓ |
| tg256 @ pp4096 | 266.4 | 266.2 | ≥ 258.4 ✓ |

Decode gate **HELD** at every step. Build GREEN · GPU suite 0 failures (1266 tests).

### F-A2 bounded decode-burst (tg256 @ pp512, non-deterministic mode)
| runtime.decode_burst | tok/s |
|---|---|
| 128 (new default) | 341.1 |
| 256 | 342.5 |
| 0 (unbounded, legacy) | 342.1 |

Bounding the non-streaming decode loop for cancel-responsiveness costs ~0
(`burst_rearm` makes relaunch nearly free). Determinism gate: in `deterministic`
mode the loop stays unbounded → greedy byte-identical across fresh processes (the
unbounded fully-on-device loop is the only greedy-reproducible decode path).

---

## 2026-06-25 — NVFP4-MoE decode occupancy/BW campaign → PHASE-1 STOP (ground truth stale, re-anchored)

**Commit:** `aa05c518` (main, post #784 gemv split + #785 CI). **Model:** `/models/Qwen3-30B-A3B-NVFP4-Modelopt`, cell `nvfp4-moe | tg256`. **Decision: do NOT edit — the §0 dispatch ground truth is stale; the named in-scope targets are not on this model's decode hot path.** Per dispatch §1.2 (>20% deviation ⇒ STOP & re-anchor) and §4 anti-cheat (no forced micro-opts, no out-of-scope kernel touches).

### Method
Clean Release build (sm_120a). GPU free, warm (mem 13801 MHz, SM ~2775). Re-roofline of the one cell via `tools/roofline/roofline measure --models nvfp4-moe --shapes tg256` (run `aa05c518_20260625_020221`, ncu full kernel-replay). Warm tg256 baseline: **328.7 tok/s** (7 reps), pp512 26.1k tok/s.

### Fresh ncu (refutes §0)
| kernel (decode hot) | time% | GB/s (% of 1792) | occ% |
|---|--:|--:|--:|
| `paged_attention_splitk_fp8_pipeline_kernel<128>` | 17.9 | 16 (1%) | **29** |
| `paged_attention_reduce_kernel` | 4.1 | 36 | 8 |
| `gemv_nvfp4_moe_gate_up_mr<8>` | 13.2 | 893 (50%) | 82 |
| `gemv_nvfp4_multirow_fp32<8>` | 12.3 | 1357 (76%) | 71 |
| `gemv_nvfp4_moe_swiglu_mr<8>` | 12.3 | 480 (27%) | 85 |
| `gemv_nvfp4_residual` | 6.9 | 578 | 42 |
| `gemv_nvfp4_qkv_fused` | 6.0 | 732 | 47 |

### Why the §0 limiters are refuted
- **P1 (attn_decode 12% occ, `num_splits` too conservative) — REFUTED.** No nvfp4 attention kernel is launched at all for this model. Since the KV-fp8 auto-default for Qwen3 MoE, decode attention runs through `paged_attention_splitk_fp8_pipeline_kernel` (attention_paged_**fp8**.cu, OUT of the dispatch's editable scope) at **29% occ** with split-K already active (the reduce kernel is present). Editing `attention_paged_nvfp4.cu` / `compute_splitk_splits` cannot move this cell — that path is dead code here. (Grid math for the nvfp4 path would have given `num_splits=11` ⇒ 352 CTAs anyway; the 12% figure was a 1-block/SM register-bound reading of a kernel that no longer runs.)
- **P2 (gemv_nvfp4 37% peak, not saturating, occupancy headroom) — REFUTED.** The gemv kernels run at **71–85% occ** and **480–1357 GB/s (27–76% peak)**; `multirow_fp32` already hits 76% of peak BW. There is no occupancy headroom and no single "663 GB/s @ 65% occ" kernel. The #784 split + prior multirow work moved this well past the §0 snapshot.

### Remaining real lever (out of named scope — needs a re-scope decision)
The least-saturated hot kernel is `gemv_nvfp4_moe_swiglu_mr` (12.3% time, **85% occ but only 27% peak BW** → compute/latency-bound, not occupancy- or BW-starved) in `src/quant/nvfp4_gemv_moe.cu`. The attention path, if pursued, is `attention_paged_fp8.cu`. Both are outside the dispatch's stated editable scope (`attention_paged_nvfp4.cu`, `gemv_nvfp4_kpar`). No edit made; awaiting a re-scoped dispatch.

Roofline history: `tools/roofline/history/runs/aa05c518_20260625_020221.json`.

### Re-scope → H1 (nvfp4_gemv_moe.cu swiglu_mr): REFUTED, reverted

Owner re-scoped to the least-saturated hot kernel, `gemv_nvfp4_moe_swiglu_mr_kernel`
(12.3% of decode kernel-time, 85% occ, only 27% peak BW — the down-projection GEMV
with SwiGLU fused on the input).

**ncu limiter (confirmed before edit):** latency-bound, neither saturated (Compute
47% / DRAM 30%); highest-utilized pipe by executed instructions = **XU 43%** (the
`expf`/division in `silu(gate)*up`), top stall = short-scoreboard (~37%, SFU/smem
latency). Root cause: `silu(gate[k])*up[k]` is recomputed once **per output row**
(NR=8 warps/block each recompute the same activation).

**H1 (bit-exact):** compute `act[k]=silu(gate[k])*up[k]` once per block into shared
memory, all NR warps read it. Same float expression + FMA order ⇒ byte-identical.

**Gates:** determinism byte-identical (post greedy token-seq == pre, self-deterministic);
test-quant 187/187, test-moe-gdn 106/106. Correctness perfect.

**Perf — REFUTED.** ncu kernel duration pre **12.8–13.1 µs → post 14.1–14.4 µs (≈ +10%
SLOWER)**: Compute dropped (47%→36%, redundant exp removed) but Memory rose (34%→44%,
smem round-trip) and the `__syncthreads` barrier **serializes the fill phase, exposing
latency that 85% occupancy already hid** across warps. e2e tg256 (imp:preedit vs post,
3 cold restarts × 7 reps, warm 2917 MHz / 14001 MHz): pre median **328.9** / post
**327.8** = **−0.34% (noise)**.

**Why e2e didn't move (structural):** a +10% kernel-time change on a "12.3%" kernel
produced only −0.34% e2e — i.e. swiglu_mr largely **overlaps under the async CUDA-graph
decode loop (off critical path)**; the ncu time-share overstates its wall-clock weight
(same lesson as the spec-decode split-K finding). Combined with P2 (gemv already
71–85% occ, multirow at 76% peak BW), **no single in-scope MoE-FFN kernel lever moves
nvfp4-moe|tg256 ≥6%** — the cell is at/near its structural decode ceiling at batch 1.
Change reverted; the high-occupancy per-row recompute is the better design here.

**Decision:** STOP with a documented negative result (dispatch §5). Theoretical
remaining lever (low prior given the overlap evidence): `gemv_nvfp4_moe_gate_up_mr`
at 50% peak BW — but the swiglu A/B shows kernel-time changes in this FFN don't
propagate to tg256, so it is unlikely to clear the gate without a structural (cross-
kernel / graph-level) change, which is out of this dispatch's scope.

### nsys re-measure + H2 (unfuse swiglu decode): SHIPPED +2.6%

The H1 e2e A/B (kernel +10% slower → only −0.34% e2e) had suggested swiglu_mr was
off the critical path. **nsys (`cuda_gpu_kern_sum`, --no-cuda-graphs) refutes that** —
true decode wall-clock shares (no ncu-replay distortion):

| Time% | kernel |
|------:|--------|
| 18.1 | gemv_nvfp4_moe_gate_up_mr |
| 17.0 | paged_attention_splitk_fp8_pipeline |
| **15.5** | gemv_nvfp4_moe_swiglu_mr |
| 8.5 | gemv_nvfp4_qkv_fused · 8.0 residual · 5.1 topk_gating · … |

swiglu_mr is genuinely **15.5% of decode wall-clock and on the critical path** — H1
failed specifically because the `__syncthreads` barrier cost more than the dedup
saved, NOT because the kernel is cheap. Also confirmed `--use_fast_math` is ON, so
`expf` already lowers to `__expf` (single MUFU) — the per-exp cost is already minimal;
only the **rows×-redundant** silu is removable.

**H2 (shipped):** remove the redundancy without an in-kernel barrier by computing the
activation once globally. The gpt-oss and GGUF/mmvq MoE paths already do exactly this
(`apply_expert_activation(gate,up→act_buf)` once + a plain `gemv_nvfp4_moe_decode`).
Switched the Qwen3 NVFP4 MoE decode path (`run_moe_decode_fast`) to the same shape,
turning the XU-bound fused swiglu-down kernel into a bandwidth-bound plain MoE GEMV.
Reuses existing tested kernels; only the dispatch wiring changed.

**Result (Qwen3-30B-A3B-NVFP4, warm 2902/13801, 3 cold restarts × 7 reps):**
tg256 **330.45 → 338.99 median = +2.58%**; tg128 +2.0% (generalizes, not shape-special).
Gates: greedy decode **byte-identical** to baseline (fp16 activation rounding absorbed
by the 4-bit NVFP4 down weight) + self-deterministic; test-quant 187/187, test-moe-gdn
106/106; prefill pp2048 within the 5% gate (separate path); nvfp4-dense tg256 neutral.

Below the dispatch's ≥6% primary target, but a real low-variance decode win on a hero
model — shipped per the "small percents matter" directive. The fused
`gemv_nvfp4_moe_swiglu_*` kernels are now dead on the NVFP4 MoE path (left in place;
removable in a hygiene follow-up). Remaining bigger levers (gate_up_mr 18.1%,
attention-fp8 17.0%) need their own campaigns.

### Follow-up: gate_up_mr lever assessed = structural ceiling; dead swiglu kernels removed

After the swiglu-unfuse shipped, profiled the next-largest decode kernel
`gemv_nvfp4_moe_gate_up_mr` (18.1% wall-clock). ncu: DRAM 60% (1.05 TB/s),
Compute 28%, L1-hit 87%, **82% short-scoreboard stall on global loads** →
load-latency-bound. Cross-kernel comparison of the same `warp_k_loop` family:

| kernel | DRAM% | sectors/req | bytes/sector |
|--------|------:|------------:|-------------:|
| gemv_nvfp4_moe_decode | 40 | 8.1 | 54.7 |
| gemv_nvfp4_moe_gate_up_mr | 60 | 16.3 | 55.9 |
| gemv_nvfp4_multirow (big-K) | 89 | 18.3 | 56.2 |

sectors/req + bytes/sector are **family-inherent** (~55% is the NVFP4 packing, not a
gate_up coalescing bug). The DRAM% spread is **shape-driven**: large-K GEMVs saturate
BW (89%), the small per-expert-K (768) MoE GEMVs are latency-bound at batch 1 — the
known **batch-1 MoE wall** (re-confirmed; the 2-mb/lane MLP variant was already
measured neutral). No viable ≥2% in-scope lever; not pursued (anti-cheat: no forced
rewrite against a structural wall). Remaining big lever (attention-fp8, 17%) is a
separate kernel/campaign.

Hygiene: removed the now-dead `gemv_nvfp4_moe_swiglu_{decode,mr,decode_kernel}` +
launcher + PDL registration (the decode path uses apply_expert_activation + plain
gemv_nvfp4_moe_decode since the unfuse). Shared header helpers
(`dot_micro_block_swiglu`, `gemv_nvfp4_row_swiglu`) kept — still used by the dense /
mxfp4 SwiGLU paths.
