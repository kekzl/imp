# Benchmarks

Reproducibly anchored measurements. Every row states **when**, **on what
commit**, **with which CUDA version and quant**, and **the exact command** —
re-run the command on the stated commit to reproduce. The commit SHA is the
authoritative version; tagged releases snapshot a SHA.

**Hardware (constant across all runs):** single RTX 5090 (GB202, 32 GB GDDR7,
water-cooled — never thermally throttled), Ryzen host, WSL2, Docker.
**Method:** greedy (temp = 0), CUDA Graphs on, 10 repetitions, isolated run
(one model per process, no concurrent GPU work), clocks warmed before timing
(the GPU idles at low clocks and takes ~1 s to ramp — cold first runs read
artificially LOW). `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Decode (tg) is the
reliable A/B signal; prefill (pp) varies up to 2.6× across container restarts
(cuBLAS autotuning) and is therefore not tabulated for comparisons.

The CI-gated canonical baseline lives in
[`tests/perf_baseline.json`](../tests/perf_baseline.json) (3% decode / 5%
prefill regression gate); refresh it via `scripts/gen_perf_baseline.sh`.

**Toolchain (current: `v0.18.1`):** C++23, Ubuntu 26.04 / GCC 15.2, CUDA 13.3.
The C++20→C++23 move in v0.17.0 is perf-neutral — Qwen3-8B-Q8_0 decode re-measured
`tg128 = 287` (baseline 269.5, within good-host-day range), so the tabulated
numbers below carry over unchanged.

## GGUF decode

| Date | Commit | CUDA | Model | Quant | Metric | tok/s | Command |
|---|---|---|---|---|---|---:|---|
| 2026-06-12 | `perf_baseline.json` (CI gate) | 13.3 | Qwen3-8B | Q8_0 | tg128 | 269.5 | `imp-cli --model Qwen3-8B-Q8_0.gguf --bench --bench-pp 16 --bench-reps 10 --max-tokens 128` |
| 2026-06-12 | `perf_baseline_north_star.json` | 13.3 | Qwen3-14B | Q6_K | tg128 @ctx2048 | 156.0 | `… --max-seq-len 2048` |
| 2026-06-09 | `ec9145b3` | 13.3 | Qwen3-14B | Q6_K | tg128 | 164 | `imp-cli --model Qwen3-14B-Q6_K.gguf --bench --bench-pp 16 --bench-reps 10 --max-tokens 128` |
| 2026-06-09 | `ec9145b3` | 13.3 | Gemma-4-26B-A4B | Q4_K_M | tg128 | 273 | `imp-cli --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --bench --bench-pp 16 --bench-reps 10 --max-tokens 128` |
| 2026-07-11 | `6946a6cd` | 13.3 | Qwen3.6-35B-A3B (hybrid) | Q4_K_M | tg256 | 213 | `imp-cli --model Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --bench --bench-pp 16 --bench-reps 10 --max-tokens 256` — legacy path; superseded by the row below |
| 2026-07-11 | fp8-ssm-gguf | 13.3 | Qwen3.6-35B-A3B (hybrid) | Q4_K_M | tg256 | **272** | same command — `gemm.fp8_ssm_proj` now also covers the Q8_0-kept GDN projections of UD quants (dequant→FP8 sidecar at init): 224.4 → 272.0 defaults (+21%), 219.2 → 265.9 spec-off, same session. PPL 4.215 → 4.289 (+1.8%, 201-token corpus) — documented trade like `nvfp4_lm_head_gdn`; degen_suite 33/33 PASS. Now ahead of llama.cpp (~229) |

> Canonical gated decode number = `perf_baseline.json` Qwen3-8B-Q8_0 tg128 =
> 269.5 (cold-median, 5 trials × 5 reps, 2026-06-12, clocks verified healthy
> during the bench: 2880 MHz SM / 13801 MHz mem / ~487 W). The previous
> baseline (286.4, 2026-06-07) was sampled on a documented peak day — its 3 %
> gate threshold (277.8) sat INSIDE the normal healthy range (266–278), so
> ordinary days could fail spuriously. Single-session warm reads carry the
> documented ±5–10 % host/driver day-to-day variance (issue #526) — sample
> `clocks.mem` during the bench (healthy = 13801 MHz / ~500 W under prefill).

Against llama.cpp (b8445+, full offload, flash attention on): imp wins dense
GGUF decode by **+37–72%**. The MoE/hybrid GDN-projection tax that used to put
Qwen3.6-35B behind was closed on the **NVFP4** path by the #949 FP8
SSM-projection sidecar (35B NVFP4 decode 257 → ~320 tok/s) and on the **GGUF
Q4_K** hybrid path by extending that sidecar to the Q8_0-kept GDN projections
(35B Q4_K decode 224 → 272 tok/s, 2026-07-11) — both now ahead of llama.cpp's
~229. GGUF remains the legacy path — NVFP4 SafeTensors is the priority.

## GGUF prefill (pp512, INT8-IMMA family — default on since #617)

All rows 2026-06-07, CUDA 13.3, 10 reps, fresh container per run, healthy-host
clocks logged. llama.cpp = build 19e92c3, same GGUF files, same day, `-fa 1
-ngl 999 -r 5`. Command pattern: `imp-cli --model <gguf> --bench --bench-pp
512 --bench-reps 10`.

| Commit | Model | Quant | imp tok/s | llama.cpp | verdict |
|---|---|---|---:|---:|---|
| `62d96a0e` | Qwen3-30B-A3B (MoE) | Q4_K_M | **9 970** | 9 288 | **imp +7%** |
| `3dd945d5` | Qwen3-14B | Q6_K | **6 617** | 6 522 | **imp +1.5%** |
| `84790dac` | Gemma-4-26B-A4B (MoE) | Q4_K_M | **8 946** | 10 749 | 1.20× behind |
| `#617` | Qwen3-8B | Q8_0 | **12 131** | 13 724 | 1.13× behind |
| `62d96a0e` | Qwen3.6-35B-A3B (hybrid) | Q4_K_M | **5 165** | 8 027 | 1.55× behind (GDN share is quality-locked FP16) |

Morning-of-2026-06-07 baselines for the same five rows were 3 968 / 5 262 /
4 231 / 8 401 / 3 675 — the day's IMMA + fp16-acc work (#608–#616) moved GGUF
prefill +36 % to +151 % per model. PPL teacher-forced gates: neutral or
better on all except Qwen3.6-35B (+0.55 %, documented trade).

## NVFP4 SafeTensors decode (tg256)

All rows: 2026-06-09, commit `ec9145b3` (v0.10.0 base), CUDA 13.3, NVFP4
prequant (Model Optimizer / llm-compressor exports), warm isolated runs (one
model per process, per-model warmup, 10 reps), command pattern `imp-cli
--model <model-dir>/ --bench --bench-pp 16 --bench-reps 10 --max-tokens 256`.
Decode carries ±5–10 % day-to-day variance (issue #526); clocks logged healthy
(mem 13801 MHz).

| Model | Params (active) | tok/s |
|---|---|---:|
| Qwen3-8B-cortecs | 8.2B | 270 |
| Qwen3-14B | 14B | 159 |
| Qwen3-30B-A3B-Modelopt | 30B (3B) | 305 |
| Qwen3-Coder-30B-A3B | 30B (3B) | 338 |
| Qwen3.6-35B-A3B | 35B (3B) | 257 → **320**¹ᵇ |
| Gemma-4-26B-A4B | 26B (4B) | 266 |
| Nemotron-3-Nano-30B | 30B (3B) | 128 → **148**¹ᶜ |
| gpt-oss-20b¹ | 21B (3.6B) | 325 |

¹ᵇ 2026-07-10, commit `80864b06` + `gemm.fp8_ssm_proj` (default on since that
PR): FP8 E4M3 per-row-scale decode sidecar for the BF16 GDN in/out projections
— tg256 268.6 → **320.3** spec-off (+19.2%), 261 → 308 with default
speculation; PPL flat (8.021 → 8.012 same-session teacher-forced). Same
command pattern as the table; healthy-host clocks sampled during the run.

¹ᶜ 2026-07-11, commit `6946a6cd` + `gemm.fp8_ssm_proj`: the same FP8
SSM-projection sidecar (221 MiB on Nemotron's 12 GDN projections) lifts
Nemotron-3-Nano-30B decode 128 → **148** tok/s (+16%), PPL flat (4.184 →
4.117). Still the slowest 30B — the Mamba2 scan + attention-projection share is
arch-limited — but the GDN-projection part of the FP16 tax is gone.

¹ gpt-oss (PRs #572/#574): SafeTensors MXFP4 source, experts converted to
NVFP4 at load (bit-exact nibbles, power-of-two scales) and registered for the
CUTLASS grouped-GEMM prefill — pp512 ≈ 16-19k tok/s. Attention stays on the
cuBLAS path (attention sinks). Decode 310-345 depending on host state.

On `sm_120`, native-NVFP4 decode is effectively uncontested (vLLM gates its
NVFP4 path on `tcgen05`/falls back to Marlin on the 5090; llama.cpp has no
native NVFP4 path).

## NVFP4 prefill (post FP16-QK FA2 primary hd=128 prefill, #687)

imp rows: 2026-06-13, commit `290a163a`, CUDA 13.3, median of 3 isolated
trials × 40 reps, fresh container per run, clocks verified healthy (warm
40-rep windows; cross-trial spread <1%). Command:
`imp-cli --model <dir> --bench --bench-pp <n> --bench-reps 40 --max-tokens 256`.
vLLM reference: 0.22.1 FlashInfer-NVFP4 (fp8 KV), same host, measured
2026-06-11 — older than the imp rows, so the ratios carry that cross-day caveat.

| Model | pp | imp tok/s | vLLM tok/s (06-11) | verdict |
|---|---|---:|---:|---|
| Qwen3-30B-A3B (MoE) | 2048 | **43 646** | 34 500 | **imp +27%** |
| Qwen3-30B-A3B (MoE) | 4096 | **37 639** | 36 200 | **imp +4%** (was 1.19× behind) |
| Qwen3-14B (dense) | 2048 | **26 918** | 26 600 | ~tie (imp +1.2%) |
| Qwen3-14B (dense) | 4096 | **24 232** | 25 300 | 1.04× behind (was 1.27×) |

**#687 (FP16-QK FA2 as the primary hd=128 prefill) closed most of the pp4096
gap:** MoE pp4096 went 30 300 → 37 639 (**+24%**, now *ahead* of vLLM) and dense
pp4096 19 938 → 24 232 (**+21%**, gap 1.27× → 1.04×). pp2048 moved +3.8–4.7%.
imp also wins TTFT/pp512 outright (2.1–3.4×, vLLM has a flat-cost small-M regime).
The lone surviving NVFP4-prefill gap is dense pp4096 at ~1.04×. Decode (tg256
@ctx2048): 14B 159, 30B-A3B ~317. Nemotron-3-Nano is arch-limited (hybrid
Mamba2 + attention FP16-projection mix).

## Long context (pp8192 / tg512 @ 16k ctx)

First tracked long-context table (the GOAL benchmarking discipline asks for
pp8192 + tg at 16k; nothing was tabulated before 2026-07-11). All rows
2026-07-11, commit `e66f24b5`, CUDA 13.3, isolated runs, healthy-host clocks;
command pattern `imp-cli --model <m> --bench --bench-pp {8192|16384}
--bench-reps {5|3} --max-tokens {64|512} --max-seq-len {9216|17408}`. pp
carries the usual restart variance; tg is the signal.

All rows re-measured 2026-07-11 on `905630e2` after the three fixes the
first sweep triggered (#967 streaming-eviction OOB, #968 dense spec ctx cap,
#969 cheap-KV floor); the discovery-day numbers are kept for the record.

| Model | Quant | pp8192 tok/s | tg512 @16k (defaults) | discovery-day @16k |
|---|---|---:|---:|---:|
| Qwen3-8B | Q8_0 | 13 268 | **151.9** (214.9 with `--kv-fp8`) | 58.7 (spec ungated, #968) |
| Qwen3-Coder-30B-A3B | NVFP4 | 35 516 | 269.5 | 269.5 |
| Qwen3.6-35B-A3B | NVFP4 | 14 887 | **264.3** | IMA crash (#963/#967), then 72.7 streaming (#969) |
| Qwen3.6-35B-A3B | Q4_K_M (GGUF) | 9 436 (pp16384) | **234.2** | 69.6 under streaming + silent OOB reads (#967/#969) |

What the first sweep found and what fixed it:

- **#963/#967**: StreamingLLM's middle-block eviction retained the window
  ceil-aligned while the decode kernels read floor-aligned — one evicted
  block read per step at non-aligned ctx: an IMA on the VRAM-full NVFP4
  35B, silent garbage attention on the GGUF variant.
- **#964/#968**: the graph-captured dense chunk verify is sized to the pow2
  ctx tier (floor 4096), so default-on n-gram speculation cost −62% @16k
  despite 100% accept. `speculative.draft_ctx_cap` (default 2048, dense
  only) gates it; the structural fix (verify cost following live ctx) is
  still open in #964.
- **#969**: the auto KV floor stopped at 16384 tokens, so the 35B's 16.4k
  pool hit the >90% streaming valve on a 16k prompt that fits outright.
  The floor now covers max_seq_len + 12.5% headroom when that costs ≤1 GiB
  (hybrids: ~377 MiB) — both 35B variants now run 16k fully streaming-free
  (graphs on, full attention).
- FP8-KV is worth **+39%** at 16k on Qwen3-8B but does not auto-engage on
  GGUF sources (the `kv_cache.dtype=auto` FP8 upgrade is keyed on the
  checkpoint's `kv_cache_quant_algo` hint, which GGUF files don't carry) —
  `--kv-fp8` is a manual win for long-context GGUF sessions.

## Concurrent serving throughput (batched decode)

The VRAM-aware auto `max_batch_size` (#736) made concurrent decode the common
server path; two batched-decode kernel fixes target it. Aggregate throughput =
Σ completion tokens / wall-clock across N concurrent `POST /v1/chat/completions`
against a live `imp-server`. This is a server-level number (not the greedy
single-stream `--bench`), so it carries the same ±5–10 % host day-to-day decode
variance (issue #526) — clocks logged healthy here (SM ~2880 MHz, mem 13801 MHz,
up to 439 W).

| Date | Commit | Model | Concurrency | Aggregate tok/s | Note |
|---|---|---|---:|---:|---|
| 2026-07-12 | batched-sampling PR | Qwen3-Coder-30B-A3B-FP4 | 16 | **1 173** sustained closed-loop median (1 138/1 173/1 213 over 45-60 s windows; ~73 tok/s = 13.6 ms TPOT per stream) | with `gemm.nvfp4_lm_head_cutlass=true` (opt-in). Levers over the 861 baseline: batched sampling readback (one pinned D2H + one sync per step instead of 16 pageable-D2H round-trips), row-parallel top-k/top-p kernels (one partial+finalize launch pair for the whole batch), kernel-based residual copies (WDDM D2D submission cost), static ban-list device cache, CUTLASS lm_head. Now ABOVE the published vLLM reference (1 157 aggregate, 13.6 ms TPOT — cloudrift.ai) while keeping 5.4× its single-stream decode (396 tok/s, unchanged). Closed-loop methodology matches vLLM bench's max-concurrency mode; the old fixed-batch harness (waits for all 16, includes ramp-down tails) reads ~1 150 on the same build |
| 2026-07-12 | `98cfafe3` | Qwen3-Coder-30B-A3B-FP4 | 16 | 861 (716/874/861, median; ~54 tok/s per stream) | healthy clocks sampled during run (2880/13801, 308 W); single-stream same day: 384 tok/s. Reference point: vLLM serving the same model class (AWQ 4-bit) on an RTX 5090 reports 1 157 aggregate @16 with ~73 tok/s per stream (cloudrift.ai, 2026-07) — imp trades ~26% aggregate for 5.3× single-stream (by design: latency-first, see GOAL) |
| 2026-07-11 | `e66f24b5` | Qwen3-14B-NVFP4 | 16 | **864** (822/904/864, median) | re-measure post #941-#943/#951/#957; `gemm.nvfp4_lm_head_cutlass=true` adds ~+8% (932/946) and stays coherent |
| 2026-06-23 | `b56e9ae5` | Qwen3-14B-NVFP4 | 16 | 767 | #745 + #746 |
| 2026-06-23 | pre-`#745` | Qwen3-14B-NVFP4 | 16 | 472 | single-block sampler + per-row LM head |

**+62 %** from the two fixes. Drivers (nsys, graphs-off, share of decode GPU
time): top-k/top-p sampler **36 % → 6 %** (single-block `<<<1>>>` → multi-block,
737 → 83 µs/call, #745); NVFP4 LM head **18 % → 7 %** (per-sequence M=1 GEMV loop
→ batched-M, 4.2 → 1.5 ms/step, #746). **Single-stream decode is unchanged by
design** — both fixes touch only the n>1 / eager-sampling paths; the greedy
argmax + M=1 GEMV paths (and the `perf_baseline.json` gate) are untouched.

Command: `imp-server --model Qwen3-14B-NVFP4 --max-batch 16`, then 16 concurrent
`POST /v1/chat/completions` (`max_tokens` 200, `temperature` 0.7); aggregate =
Σ `completion_tokens` / wall-clock. At sustained n>1 the dominant remaining cost
is the CUTLASS M=16 NVFP4 GEMMs (~57 %, the real per-layer projection compute,
already batched and largely launch-hidden under CUDA Graphs).

## Multi-turn TTFT (hybrid prefix caching, #831 / v0.15.0)

Agentic chat re-sends the full conversation every turn, so on a recurrent
(SSM/GDN) model — where prefix caching was disabled before #831 — per-turn TTFT
grew linearly with history (full re-prefill each turn). Recurrent-state
snapshots make it prefill only the delta. Server-level numbers (SSE, so they
carry the ±5–10 % host day-to-day decode variance, issue #526); TTFT = wall time
to the first streamed token. Setup: `imp-server --model Qwen3.6-35B-A3B-NVFP4
--set runtime.max_seq_len=12288`, 6-turn growing-history replay (~2 k tokens
added per turn), `max_tokens` 60, `temperature` 0, streaming. Both columns same
host/day (2026-07-02, CUDA 13.3); `v0.14.0` = `2316f2fd`, `v0.15.0` = `e80a26a4`.

| Turn | History | v0.14.0 TTFT | v0.15.0 TTFT | `cached_tokens` |
|---|---|---:|---:|---:|
| 1 | fresh | 1.62 s | 1.89 s | — |
| 2 | ~2 k | 2.85 s | 1.41 s | 2 016 |
| 3 | ~4 k | 3.08 s | 1.55 s | 4 064 |
| 4 | ~6 k | 4.31 s | 1.40 s | 6 080 |
| 5 | ~8 k | 5.68 s | 1.51 s | 8 128 |
| 6 | ~10 k | 6.70 s | **1.94 s** | 10 176 |

v0.14.0 grows linearly with history; v0.15.0 stays flat (**3.5× at ~10 k tokens**,
gap widening with context). `usage.prompt_tokens_details.cached_tokens` (and
Anthropic `cache_read_input_tokens`) report the hybrid hits. Nemotron-3-Nano-30B
(pure SSM, CUDA graphs off) holds a flat ~0.22 s TTFT across the same replay
(turn 1: 0.34 s). Snapshot store on the 35B: 4 × 63.8 MiB slots (default
`server.recurrent_snapshot_mb` = 256).

## Output-quality gate

Throughput numbers say nothing about correctness — that lesson is paid for
(see git history around 2026-06-04). Every perf-relevant change must also pass
`python3 tools/analysis/degen_suite.py` against a running server.

*(This file is updated in the same commit as the measurement-relevant change;
check `git log BENCHMARKS.md` for the measurement provenance trail.)*
