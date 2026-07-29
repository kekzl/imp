# Phase-0 baseline snapshot — dispatch-hardening campaign (2026-07-17)

Anti-cheat reference for the "DISPATCH_HARDENING" brief. Every later claim in the
campaign is diffed against this table. Captured on `main` @ `f7119ad1` (v0.19.1),
RTX 5090 / driver 610.62 / CUDA 13.3, WSL2, healthy host (2902 MHz SM / 13801 MHz
mem sampled DURING the bench). No source edits in this phase.

Note on brief staleness: the brief describes the repo as "C++20, ~161k LOC" and
lists gaps that were closed between May and July 2026. Measured reality below;
claim-by-claim verification in the companion matrix at the end.

## Baseline table

| Dimension | Metric | Baseline (2026-07-17) |
|---|---|---|
| Build | compiler warnings, full rebuild, `-Wall -Wextra -Wpedantic` (CXX) | **2** (1 in src: `-Wnarrowing` `src/runtime/engine_kv_cache_init.cpp:43`; 1 in tests: `-Wunused-result` `tests/test_tool_stream_filter.cpp:238`) + 5 informational ptxas C7504 (`setmaxnreg` ignored) |
| Build | clang-tidy | advisory CI job (differential); GitHub API 503 during snapshot — not re-run locally to keep the bench host quiet. Local target: `make tidy` |
| Build | file-size gate `tools/check_filesize.py` | **0 violations** (31 warn, 27 allowlisted; 582 files scanned) |
| Build | LOC (src/ + include/, .cpp/.cu/.h/.cuh/.hpp) | **119,813** (brief's "161k" includes tests/tools) |
| Safety | compute-sanitizer (memcheck/racecheck/synccheck) | **N/A on this host** — WSL2/WDDM exposes no debugger interface (documented, `make sanitize` is native-Linux-only) |
| Safety | ASan+UBSan host code (`IMP_SANITIZERS=ON`, RelWithDebInfo, test-core + test-text) | **0 imp-code findings.** test-core 274/274 pass, test-text 192/192 pass (1 skipped). 1 UBSan hit: misaligned int store in vendored `third_party/stb/stb_image_resize2.h:8659` (third-party, known stb pattern). LeakSanitizer: 50,960 B / 5 allocations, all traces without imp frames (CUDA-driver/dlopen one-time allocs) |
| Tests | CPU unit suite (`make test-unit`) | **37/37 pass** |
| Tests | GPU suite (`make test-gpu`, all split binaries) | **exit 0, 0 failures** (1,617 test cases ran; 143 SKIPPED = model-gated without `IMP_TEST_MODEL`; 2 DISABLED = known determinism boundaries, intentionally kept) |
| Perf | Qwen3-Coder-30B-A3B-Instruct-FP4, spec-OFF, median of 5 isolated trials | see table below |
| Roofline | pinned BASELINE run | `cf1b382a_20260711_193211` (ncu clock-locked, 2 restarts/cell) — see `roofline_2026_07_11.md` |

## Perf baseline (median of N=5, isolated processes, same harness)

Model: `Qwen3-Coder-30B-A3B-Instruct-FP4` (NVFP4 MoE, the brief's designated model).
Command (one line, per cell):

```
docker run --rm --gpus all -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v $HOME/models:/models imp:test \
  imp-cli --model /models/Qwen3-Coder-30B-A3B-Instruct-FP4 --bench --bench-pp <PP> --bench-reps 10 \
  --set speculative.ngram=false
```

| Cell | Median tok/s | Trial spread | Notes |
|---|---|---|---|
| pp512 | **19,739** | 17,177 – 22,695 (±14%) | spread = documented cuBLAS-autotune restart variance |
| pp2048 | **46,843** | 45,401 – 47,169 | |
| pp4096 | **41,030** | 40,932 – 41,138 | |
| tg256 @pp512 | **402.6** | 402.2 – 403.8 (<0.5%) | decode = the reliable A/B signal |
| tg256 @pp2048 | **367.7** | 367.4 – 367.8 | |
| tg256 @pp4096 | **346.4** | 346.3 – 346.8 | |

Clocks sampled during the run: 2902 MHz SM / 13801 MHz mem / ~394 W — healthy-host
day, numbers trustworthy. CI decode gate (different model/metric, for reference):
Qwen3-8B Q8_0 spec-OFF tg128 = 288.02 (band 275–290), `tests/perf_baseline.json`.

## Roofline utilization per hot kernel (BASELINE `cf1b382a`, nvfp4-moe cells)

| Kernel class | Cell | Time share | %-roofline (med) | bound-by |
|---|---|---|---|---|
| gemm_grouped_nvfp4 | pp512 | 49.7% | 51.4 | memory |
| gemm_cutlass_nvfp4 | pp2048 | 9.9% | 47.2 | compute |
| attn_fa2 | pp4096 | 37.3% | 22.0 | compute |
| gemv_nvfp4 | tg256 | 60.6% | 37.6 | memory |
| attn_decode_paged | tg256 | 11.8% | 2.3 | memory |
| moe_routing | tg256 | 13.5% | 2.2 | memory (launch-latency class — no-graphs artifact, see benchmark-cuda STOP #7) |

Legacy-attention coverage (Module 2 of the roofline report): **0.0% of window in
all 21 cells** — the materialized `causal_softmax`+cuBLAS path holds no measurable
prefill time on any standard cell.

## Claim-verification matrix (brief vs. repo @ v0.19.1)

Four independent read-only scouts verified each brief claim against code/docs.

| Brief claim | Verdict | Evidence |
|---|---|---|
| "C++20, ~161k LOC" | STALE | C++23 since PR #916 (`CMakeLists.txt:4-7`); 119,813 LOC src+include |
| G: README carries stale "1258 tok/s / 20× behind" | ALREADY FIXED | zero live-doc hits; survives only in archive/audit docs marked *refuted* (`docs/GOAL.md:99`, `docs/audit/housekeeping_2026_06_13.md`) |
| G: stale multi-arch language in CLAUDE.md/AGENTS.md | NOT FOUND | all sm_90/sm_100/Hopper/WGMMA mentions are explicit *exclusion* statements (`AGENTS.md:11`, `CLAUDE.md:84`, `README.md:27,29,83,147`, `docs/sm120.md:23`) |
| G: README should say "~200 tok/s decode, pp ≈ 16.5k/17.2k/18.2k" | STALE — WOULD REGRESS DOCS | current README numbers (07-12 sweep) are higher and SHA-anchored: decode 271–390 tok/s NVFP4 heroes; measured today: pp2048 = 46.8k tok/s. Applying the brief's numbers would be a factual downgrade → refused per invariant §0 |
| B: "primary lever = FA2 coverage on MoE prefill; legacy path ~18% overhead" | STALE — NO HOT TARGET | legacy path = 0.0% of prefill window on all cells (roofline Module 2); hd=128 (incl. Coder-30B) is 100% FA2 since #478/#525/#932. Remaining cuBLAS tail is deliberate: Gemma-4 hd=512 globals (cuBLAS 2.8–4.6× FASTER than fused, PR #1042), gpt-oss sinks < threshold (accuracy reference), q_offset>0 short chunks (<1% upside), vision (non-causal), debug/parity paths. "Retire legacy" would regress Gemma-4 |
| B: "investigate cuBLAS autotune pinning" | OPEN (known) | variance documented (2.6× across restarts, README:87, BENCHMARKS.md:14); decode-first methodology is the shipped mitigation; pinning attempt = deterministic_gemm exists (`gemm.cu:374`, pins cuBLASLt algo, deterministic mode only) |
| A: MoE-atomics determinism flag | ALREADY IMPLEMENTED | `runtime.deterministic` → ordered single-thread scatter (`moe_routing.cu:444,605,720`), implies `deterministic_gemm`; asserting tests: `test_determinism_e2e.cpp` (greedy + bit-identical PPL), `test_moe_executor.cu:163,220` (token-equality + bounded logit drift). Known boundary: cross-fresh-context on GDN-hybrids = 2 DISABLED tests (intentional) |
| D: `/v1/messages` streaming synthetic | ALREADY IMPLEMENTED | true per-token SSE since #754 (`handlers_messages.cpp:353-427`, `anthropic.cpp:58-263`); TTFT measured at first token (`stream_driver.cpp:233-235`) |
| D: per-request speculative toggle absent | ALREADY IMPLEMENTED | `"speculative"` bool on both APIs (`handlers_chat_params.cpp:208-211`, `anthropic.cpp:327-330`); note: MTP-head spec remains load-time-only |
| D: cache_control not implemented | PARTIAL | parsed + honored as all-or-nothing prompt-prefix pin (`anthropic.cpp:291-310,410-413`, reports `cache_read_input_tokens`); per-breakpoint granularity + ephemeral TTL tiers not modeled |
| D: no p50/p99 metrics endpoint | ALREADY IMPLEMENTED | Prometheus histograms on `/metrics`: `imp_request_duration_seconds`, `imp_ttft_seconds`, `imp_inter_token_seconds` (`handlers_misc.cpp:185-224`); p50/p99 via `histogram_quantile()` |
| C: RAII coverage gaps | **CONFIRMED — REAL FINDING** (partly addressed 2026-07-29) | wrappers exist (`CudaStream`/`CudaEvent`/`Buffer`/`PoolAllocator`/`VRAMAllocator`) but **no owner for `cudaGraph_t`/`cudaGraphExec_t`/`cudaMemPool_t`** — still true. The allocation half moved: `src/memory/backend.h`'s move-only `Region` owns the migrated tiers and the site census is gated by `tools/check_alloc_sites.py` (the `~801` figure here is a one-off count; read the gate instead). Graph/mempool handles remain unowned |
| A: missing post-launch error checks | **CONFIRMED — REAL FINDING** | 415 kernel-launch sites in src/, only ~5 have `cudaGetLastError` within 3 lines (~1%); 50+ launch-containing files have zero checks. `IMP_CUDA_CHECK_LOG` (601 uses) covers API returns, not launches |
| C: KV leak-under-churn regression test | **CONFIRMED — REAL GAP** → **CLOSED 2026-07-29** (#1106; see Bottom line item 3) | LRU eviction + prefix churn well covered (54 cases in `test_kv_cache.cpp`, 9 eviction + 10 prefix-integrity tests), but no sustained N-iteration churn test asserting free-block count returns to baseline — now `tests/test_memory_allocators.cpp`, 5000 randomised steps against the `BlockPool` backing `KVCache` |
| E: golden/numerical/determinism test matrix | PARTIAL (pre-existing) | ~574 GTest incl. per-quant + golden coverage; per-arch golden matrix not 1:1 with the brief's list; determinism tests exist (above) |

## Bottom line for the campaign

Work items with real substance, in brief-priority order (A → C):

1. **A: post-launch `cudaGetLastError` coverage** (~410 unchecked launch sites) —
   mechanical, zero-perf-risk (debug-gated or log-once), high diagnostic value.
2. **C: RAII owners for CUDA graphs** (`cudaGraph_t`/`cudaGraphExec_t`; 26+38 raw
   sites in `cuda_graph.cu`/`engine_graph_decode.cpp`) and audit of the 801 raw
   alloc sites for throw-path leaks (most sit behind init-once paths — triage, not
   blanket rewrite).
   **Superseded 2026-07-29 for the allocation half** (#1106/#1107): `Backend`/`Region`
   is the move-only owner for the migrated tiers, and the census is now a gate rather
   than a doc figure — `tools/check_alloc_sites.py` with a monotonically shrinking
   allowlist (blocking CI job `Alloc sites`). Read the current count from the gate.
   Note that site count and runtime traffic are different quantities: #1107 removed
   96% of the runtime allocations while leaving the site count flat (`AUDIT.md` B34).
   Graph handles still have no RAII owner.
3. **C: KV leak-under-churn stress test** (bounded: N alloc/evict/prefix cycles,
   assert pool returns to baseline). **Delivered 2026-07-29** in the CPU lane
   (`tests/test_memory_allocators.cpp`: 5000-step randomised block-pool churn with
   conservation, plus refcount balance across exception paths) against the `BlockPool`
   that now backs `KVCache`. Scope the assertion to *pool* accounting: at device level
   WSL2/WDDM never returns a process's peak commitment, so "back to baseline" is not
   observable through `cudaMemGetInfo` (`AUDIT.md` B36).
4. **A: fix the single src warning** (`-Wnarrowing`, `engine_kv_cache_init.cpp:43`)
   + the test `-Wunused-result`.
5. **D (optional, additive): cache_control per-breakpoint granularity / TTL tiers**
   — only if API parity is wanted; current pin semantics are functional.
6. **F (third-party, low): UBSan misaligned-store in vendored stb** — upstream
   pattern; suppress or bump vendored copy, not a hand-patch candidate.

Everything else in the brief (B attention lever, D streaming/toggle/metrics, G doc
purges, A determinism caveat) is already shipped, refuted by measurement, or would
actively regress the repo (README numbers). Per invariant §0 those lines of work
are STOPPED with this report as evidence.
