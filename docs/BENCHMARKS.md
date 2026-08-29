<!--
layer: L1
audience: operators
verified: 2026-08-28
commit: be825e4a
-->

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

> **Correction, 2026-08-14** (this file is a record, so the method note above is
> left as written): the 2.6× was a carried-forward citation. cuBLAS algo
> re-timing measures 3.50 % over nine process starts, and the spread is a
> property of the *model*, not of cuBLAS: 0.6-1.2 % on Qwen3-8B Q8_0 against
> 37.6 % on a fully resident NVFP4 MoE model. Current figures and provenance:
> [`PERF.md`](PERF.md).

The CI-gated canonical baseline lives in
[`tests/perf_baseline.json`](../tests/perf_baseline.json) (8% decode / 8%
prefill regression gate, plus a 10% peak-VRAM ceiling over the pinned
`metrics.memory_mb.own_peak_mb` — see [`BENCHMARKING.md`](internals/BENCHMARKING.md));
refresh it via `scripts/gen_perf_baseline.sh`.

**Toolchain (current: `v0.31.0`):** C++23, Ubuntu 26.04 / GCC 15.2, CUDA 13.3
(13.3.1 since v0.20.1; the rows below were taken on 13.3.0 and carry over —
re-measured perf-neutral, decode 287.95 vs 288.38 tok/s, median of three).
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
| 2026-08-28 | `899301c6` | 13.3 | Qwen3-8B | Q8_0 (fp8 KV) | tg136 @32k ctx | **199.5** | `imp-cli --model Qwen3-8B-Q8_0.gguf --kv-fp8 --bench --bench-pp 32768 --bench-reps 1 --max-tokens 136 --max-seq-len 40960 --set speculative.ngram=false --set attention.sparse_topk_tokens=4096` — sparse decode attention; dense same command without the last flag: 160.3 (+24.5%, 3/3 alternating rounds). 16k: 212.0 vs 202.1 (+4.9%) |
| 2026-08-29 | sparse-serving | 13.3 | Qwen3-8B | Q8_0 (fp8 KV) | serving decode, 3 streams x 25k ctx | **197.7** | `tools/analysis/serving_sparse_ab.sh` (KVBLOCKS=5000 CONC=3 TARGET_CHARS=100000 SEQLEN=26624), tg8/tg520 differential, fresh server per arm, 3 alternating trials — dense median 155.6 (spread 150.3-173.8), sparse 197.7 (194.4-198.2), +27%. Both arms VRAM-resident; at 689 MiB free the ON arm WDDM-spills and every prefill kernel runs +11% (invalid numbers, #1103 class) |
| 2026-08-29 | sparse-verify-chunks | 13.3 | Qwen3-8B | Q8_0 (fp8 KV) | tg @32k ctx, spec on | **176.1** | `imp-cli --model Qwen3-8B-Q8_0.gguf --kv-fp8 --prompt-file <32k NIAH prompt> --max-tokens 384 --temperature 0 --seed 42 --max-seq-len 33800 --set attention.sparse_topk_tokens=4096` — verify chunks on the sparse table; same command on `cefd5e81` (chunks dense): 137.4; all-dense: 124.5 (3/3 alternating rounds, n-gram spec default-on, 5.25-5.67 tok/verify) |

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

### Competitive re-sweep 2026-08-21 (llama.cpp build 10524 `9ee9fc04c`)

Reproduce with `make bench-competitive`. The competitor image is pinned **by
digest** in [`scripts/bench_competitive.sh`](../scripts/bench_competitive.sh),
not by tag: `:full-cuda` moves, and the two sweeps below were each compared
against a build that nothing in the repo recorded.

Two imp columns, because one number cannot mean both things. Where the n-gram
drafter engages, a dense `--bench` run accepts essentially every draft (measured
here: 504 of 504 on Qwen3-8B) and `llama-bench` has no equivalent. The spec-off
column is the same command with `--set speculative.ngram=false`. The 2026-07-12
sweep tabulated the defaults column only.

**Where that acceptance comes from, since this file said the wrong thing about
it until 2026-08-21.** Not from the prompt, and this follows from the source
rather than from a measurement:

- `imp-cli --bench` builds the prompt as `tokens[i] = i % vocab_size`
  (`tools/imp-cli/main.cpp:401`). At `--bench-pp 512` against a ~151k vocab the
  counter never wraps, so the prompt is 0..511 strictly increasing.
- Every 6-gram in a strictly increasing sequence of distinct ids is unique.
- The drafter is prompt-lookup over `input + prediction + output` with
  `speculative.min_match = 6` (`src/core/config/speculative.h:169`).

So the prompt contributes **zero** matches by construction, and any draft that
exists came from the **generation**. Under `ignore_eos` a synthetic counting
prompt sends some checkpoints into a periodic continuation that prompt-lookup
then predicts exactly, and which checkpoints those are is not a property of the
model. Same prompt, same build, same day:

| checkpoint | drafted | accepted |
|---|---:|---:|
| Qwen3-8B Q8_0 | 504 | 504 (100 %) |
| Qwen3-14B **NVFP4** | 504 | 504 (100 %) |
| Qwen3-14B **Q6_K** | 96 | 6 (6.2 %) |

The two 14B rows are the same model at two quantisations. The quantisation
changes the greedy continuation, one of them loops and the other does not, and
the drafter follows. The Q6_K arm is also bistable across fresh processes: three
runs gave `drafted=0`, then 96, then 96. So a decode A/B that leaves speculation
on is not merely measuring the verify path, it is measuring whether this
checkpoint happened to loop.

It is also why the spec-off column is not optional. At batch 1 decode is
dominated by weight reads, so a degenerate continuation does not move tok/s on
its own. It moves it only through speculation, and that is the 504-of-504
effect.

| Model (shared quant) | imp default | imp spec-off | llama.cpp | imp lead | lead 07-12 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Q8_0 | 396.76 | 284.60 | 159.74 | **+148 %** | +48 % |
| Qwen3-14B Q6_K | 162.06¹ | 163.38 | 112.83 | **+44 %** | +42 % |
| Qwen3.6-35B-A3B UD-Q4_K_M (hybrid) | 283.99 | 284.48 | 220.53 | **+29 %** | +18 % |
| Gemma-4-26B-A4B UD-Q4_K_M (MoE) | 244.16 | 244.24 | 210.34 | **+16 %** | +21 % |
| Qwen3-30B-A3B Q4_K_M (MoE, non-hero) | 314.62 | 314.64 | 303.28 | +3.7 % | +1.7 % |
| gpt-oss-20b MXFP4² | 412.41 | 412.09 | 330.09 | **+25 %** | +13-19 % |

**Release bar 2 holds on all four heroes, in both columns.** The narrowest
margin is Gemma-4 at +16 %.

**llama.cpp did not erode the lead.** b10524 measures flat to slightly slower
than b9976 on every shared model here (8B 160.5 → 159.74, 14B 115.3 → 112.83,
35B 226.5 → 220.53, Gemma-4 216.0 → 210.34, 30B 319.2 → 303.28). The six weeks
of batch-1 decode work between the two builds does not show up at these shapes.

**One imp-side regression, and it is Gemma-4:** 261.3 → 244.16, -6.6 %, against
llama.cpp's -2.6 % on the same model. That is what took the lead from +21 % to
+16 %. Not investigated here; this sweep measures, it does not fix.

**The n-gram drafter never engages on Qwen3-14B Q6_K:** `verify_steps=0
miss_steps=72 drafted=0` over a full bench run. Both imp columns therefore
measure the same path on that model, which is what makes the pair a
repeatability control: where speculation is inert the two columns must agree,
and on 35B, Gemma-4, 30B and gpt-oss they agree to **0.2 %**.

¹ The sweep produced 154.77 for this row, 5.3 % below its own spec-off column
where the two must agree. Two isolated re-measurements gave 162.08 and 162.04,
and the tabulated value is their median.

**The first explanation for that gap was wrong and is corrected here.** It was
attributed to the card not being settled after a 16 GiB competitor model
unloads, and the settle between arms was raised from 5 s to 20 s on that basis.
A re-run at 20 s produced 155.23. Not the cause.

**The arm is bimodal, not noisy.** Four isolated runs, same command, same build:

| run | `drafted` | tok/s |
|---|---:|---:|
| 1 | 176 | 153.56 |
| 2 | 176 | 153.88 |
| 3 | **0** | **162.13** |
| 4 | 176 | 154.65 |

The n-gram drafter engages on some processes and not others, and where it
engages on this checkpoint it accepts 6.2 % at ~50 ms per verify against a
~6.2 ms decode step: roughly eight decode steps spent to return two tokens. So
this checkpoint's default throughput has two values, 162 quiet and ~154 firing,
and the tabulated 162.06 is the quiet mode.

It is a cold-start effect rather than a property of the model. On a single
1024-token request the same checkpoint accepts **104 of 288, 36.1 %, at 6.78
tokens per verify**: prompt-lookup has nothing to match against until the
generation is long enough, and a 128-token bench rep is entirely cold start. The
economics guard meant to catch this (`engine_spec_ngram.cpp:175`, the long-context economics guard) cannot: it
arms on `spec_verifies >= 8` **per request**, and a 128-token request produces
about one verify.

² Basis changed since 07-12. That row compared imp on SafeTensors against
llama.cpp on GGUF; the SafeTensors checkpoint is no longer on this host, so both
engines here read the same `gpt-oss-20b-mxfp4.gguf`. The row is more
apples-to-apples than the one it replaces, not less, but it is not the same
measurement.

[PROV: commit=fa21f28e date=2026-08-21 hw=RTX5090 tree=imp-campaign (fresh `make build`)
 image=ghcr.io/ggml-org/llama.cpp@sha256:c49f4d485fb08d3002fcbd6b43be8b18758b4a2f021243b42968f64a37b57e1d
 cmd=`bash scripts/bench_competitive.sh` -> imp `imp-cli --model <m> --bench --bench-pp 512 --bench-reps 10 --max-tokens 128 --temperature 0`,
     llama `llama-bench -m <m> -p 512 -n 128 -r 5 -ngl 99`
 note=one process per (engine, model); GPU verified idle (1502 MiB WSLg baseline, no containers) before the run]

### Competitive re-sweep 2026-07-12 (llama.cpp build 9976 `e3546c794`)

Same-day, same host state, both engines pp512/tg128: imp
`imp-cli --model <m> --bench --bench-pp 512 --bench-reps 10 --max-tokens 128
--temperature 0` (commit `7811658a`, defaults) vs llama.cpp
`llama-bench -m <m> -p 512 -n 128 -r 5 -ngl 99` (image
`ghcr.io/ggml-org/llama.cpp:full-cuda`, pulled 2026-07-12). Full imp hero
matrix appended to [`scoreboard.tsv`](scoreboard.tsv).

| Model (shared quant) | imp tg128 | llama.cpp tg128 | imp lead |
|---|---:|---:|---:|
| Qwen3-8B Q8_0 | 237.3 | 160.5 ± 1.3 | **+48%** |
| Qwen3-14B Q6_K | 163.2 | 115.3 ± 0.2 | **+42%** |
| Qwen3.6-35B-A3B UD-Q4_K_M (hybrid) | 266.8 | 226.5 ± 5.9 | **+18%** |
| Gemma-4-26B-A4B UD-Q4_K_M (MoE) | 261.3 | 216.0 ± 3.5 | **+21%** |
| Qwen3-30B-A3B Q4_K_M (MoE, non-hero) | 324.7 | 319.2 ± 7.0 | +1.7% |
| gpt-oss-20b MXFP4 (imp: SafeTensors, llama: GGUF) | 389.7–391.2¹ᵍ | 328.6–345.4 (run-to-run ±5%) | **+13–19%** |

¹ᵍ imp re-measured 2026-07-13 at commit `63df2d30` (PR #990,
`gemm.fp8_attn_proj` FP8 decode sidecar for the BF16 q/k/v/o projections,
default auto): tg128 391.2 median of 3 isolated trials, same command as the
sweep, healthy clocks sampled. The llama.cpp column is the unchanged
2026-07-12 b9976 measurement. Pre-sidecar imp measured 343.6–350.3 — a
statistical tie, formerly tracked in #984 (resolved by PR #990).

llama.cpp's MoE/MXFP4 decode improved substantially at b9976 (the 07-12 sweep
measured a gpt-oss tie); the FP8 attention-projection sidecar restored the
lead. The non-hero Qwen3-30B GGUF margin remains noise-level. Dense GGUF, the
hybrid hero and Gemma-4 remain clear wins; NVFP4 SafeTensors stays uncontested
(no llama.cpp counterpart). The pp512 column is not tabulated per the prefill
variance policy above; on this sweep imp led NVFP4 prefill (e.g. 8B NVFP4
36.4k vs llama Q8 14.1k) while llama led dense GGUF Q8 prefill (14.1k vs
12.7k, known best-effort surface, release bar 3).

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

> **Prefill numbers pinned between 2026-06 and #1061 read high.** Prefix
> caching went default-on for the server/CLI in #758, and `imp-cli --bench`
> repeats the same prompt — so the repeated reps partly measured cache hits
> instead of prefill. #1061 disables prefix caching for one-shot CLI runs (a
> single-generation process never re-sees a prefix), which restored honest
> numbers. Qwen3-8B Q8_0 pp512 is the clean illustration: **12 131** here
> (2026-06-07, before the default flip) → 14 515 pinned 2026-07-15 (with cache
> hits) → **12 407** re-pinned 2026-07-26 (cache off again). Decode was never
> affected. Re-pin evidence: bisect to `d8bc45a8`, plus `--set
> server.prefix_cache=true` on current main reproducing the old band
> (14 123 / 14 661 / 14 780). Do not read the 07-15 → 07-26 prefill drop as a
> regression.

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
| Nemotron-3-Nano-30B | 30B (3B) | 148 → **386**¹ᶜ |
| gpt-oss-20b¹ | 21B (3.6B) | 325 → **391**¹ᵈ |

¹ᵇ 2026-07-10, commit `80864b06` + `gemm.fp8_ssm_proj` (default on since that
PR): FP8 E4M3 per-row-scale decode sidecar for the BF16 GDN in/out projections
— tg256 268.6 → **320.3** spec-off (+19.2%), 261 → 308 with default
speculation; PPL flat (8.021 → 8.012 same-session teacher-forced). Same
command pattern as the table; healthy-host clocks sampled during the run.

¹ᶜ 2026-08-12, PR #1389: **148 → 386 tok/s**. The FP8 SSM-projection sidecar
had taken it 128 → 148 on 2026-07-11 (`6946a6cd`, PPL flat 4.184 → 4.117), and
this table then called the remainder "arch-limited". That was wrong: CUDA graphs
were being demoted for pure-SSM layers on an unverified assumption, so the model
decoded eagerly. Removing the demotion is the 2.6x. Nothing about the Mamba2
scan was the limit.

¹ᵈ 2026-07-13, commit `63df2d30` (PR #990) + `gemm.fp8_attn_proj` (default
auto since that PR): FP8 E4M3 per-row-scale decode sidecar for the BF16
q/k/v/o attention projections — the roofline cell
(`docs/archive/roofline_gptoss_2026_07_13.md`) showed them at 33.5% of the
decode window as 2 B/elem FP16 GEMVs. tg128 ~350 → **391.2** (+12%).
Teacher-forced PPL unaffected by construction (decode-only sidecar;
nsys-verified zero FP8 kernels in a `--perplexity` run); degen_suite 33/33.

¹ gpt-oss (PRs #572/#574): SafeTensors MXFP4 source, experts converted to
NVFP4 at load (bit-exact nibbles, power-of-two scales) and registered for the
CUTLASS grouped-GEMM prefill — pp512 ≈ 16-19k tok/s. Attention stays on the
cuBLAS path (attention sinks). Decode 310-345 depending on host state.

On `sm_120`, native-NVFP4 decode is effectively uncontested (vLLM gates its
NVFP4 path on an opcode family consumer Blackwell does not have (see
[`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md)) and falls back to
Marlin on the 5090; llama.cpp has no
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
@ctx2048): 14B 159, 30B-A3B ~317. (The claim that once stood here — that
Nemotron-3-Nano is arch-limited — was disproved on 2026-08-12; see note ¹ᶜ.)

## Qwen3.8-27B, the quickstart model (2026-08-16)

The model the [README](../README.md) and [`QUICKSTART.md`](QUICKSTART.md)
walk a first-time reader through.

2026-08-16, commit `52efa361`, CUDA 13.3, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
median of 3 isolated processes × 10 reps, card verified free before each run.
The checkpoint is `imp-quantize` output from the FP8 release
(`Qwen/Qwen3.8-27B-FP8`, 28.75 GiB in, 18.80 GiB out), which is what
`scripts/stage-model.sh` produces. Command:
`imp-cli --model <dir> --bench --bench-pp <16|512> --bench-reps 10 --max-tokens 128`.

| metric | tok/s | spread across processes |
|---|---:|---|
| decode tg128 | **87.36** | 87.34 / 87.36 / 87.39, 0.06 % |
| prefill pp512 | **7 565.65** | 7 503 / 7 566 / 7 577, 0.98 % |

Weights land at 17.9 GiB resident, leaving ~7.7 GiB for the KV cache on a 32 GB
card. Decode here is far steadier than the 8 % gate threshold suggests: three
processes agreed to 0.06 %, which is the quiet-host case the gate cannot assume.

A dense-GDN 27B at 87 tok/s is bounded by weight bandwidth, not by the LM head:
that head is 2.4 GiB of the 17.9 and is served from the NVFP4 decode cache, a
trade measured at +10.4 % decode for +0.99 % perplexity
([`quantization.md`](quantization.md)).

## imp vs vLLM on one checkpoint (2026-08-16)

The first cross-engine comparison on Qwen3.8-27B, and the first where **both
engines read the same file**: `imp-quantize --format vllm` writes
compressed-tensors, so the same 19.2 GiB NVFP4 checkpoint is served by each.
Earlier comparisons had to use different exports.

Same host, same checkpoint, same client (`curl` against `/v1/chat/completions`),
same request: one prompt, `temperature=0`, `max_tokens=128`, non-streaming.
End-to-end tok/s = `completion_tokens / wall time`, first request discarded as
warmup, three measured. CUDA graphs on in both.

| engine | tok/s | runs |
|---|---:|---|
| imp | **81.58** | 80.53 / 81.58 / 82.11 |
| vLLM 0.27.1 | 69.71 | 69.89 / 68.95 / 69.71 |

imp is **17 % ahead** on single-stream decode. Three things that number needs:

- **vLLM has no native FP4 kernel on `sm_120`** and falls back to Marlin, which
  it says itself, and it additionally warns that this model's shapes need
  thread-tile padding, "padded/sliced on every forward; performance may be
  degraded". This is a consumer-Blackwell result, not a statement about vLLM on
  the hardware it targets.
- **Batch 1 is imp's design point and not vLLM's.** vLLM owns continuous
  batching at high concurrency; that is not what this measures.
- Each arm is one process with three requests, not three alternating processes.
  The spread within each arm (2.0 % and 1.4 %) is well inside the 17 % gap, but
  it is a weaker design than the perf gate's.

**vLLM needs `--max-num-seqs` lowered on this model or it will not start with
graphs.** Qwen3.8-27B is a GDN hybrid, so vLLM allocates one Mamba cache block
per decode sequence; with 19 GiB of weights on a 32 GB card only 169 blocks fit
against a default `max_num_seqs` of 256, and startup aborts with

```
ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (169).
Each decode sequence requires one Mamba cache block, so CUDA graph capture
cannot proceed.
```

`--max-num-seqs 64` was used here. Running with `--enforce-eager` instead hides
the error and costs vLLM most of its decode speed (12.2 tok/s in an earlier,
discarded run), so an eager-mode comparison is not a comparison.

## imp vs vLLM at concurrency, same checkpoint (2026-08-25)

The missing half of the 2026-08-16 comparison: what happens past batch 1.
Same 19.6 GiB compressed-tensors NVFP4 checkpoint served by both engines
(`imp-quantize --format vllm` from `Qwen/Qwen3.8-27B-FP8`), same host, same
client and prompt (200-token technical answer, `temperature=0`,
`/v1/completions`, `cache_prompt=false`), aggregate tok/s = sum of completion
tokens / wall, median of 3 waves per point.

[PROV: commit=de24ee09 date=2026-08-25 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4-CT cuda=13.3 path=server-api cmd=bench_conc.py n=3
       vllm=v0.27.1 flags=--gpu-memory-utilization 0.90 --max-model-len 16384
       --max-num-seqs 32, VLLM_WSL2_ENABLE_PIN_MEMORY=1;
       imp=--max-concurrent 32 --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096]

| concurrency | imp (pinned 32/4096) | imp (defaults) | vLLM 0.27.1 |
|---:|---:|---:|---:|
| 1 | **84.65** | 86.44 | 69.13 |
| 8 | 358.39 | 192.88* | **503.55** |
| 32 | 935.70 | 630.19* | **1475.19** |

\* defaults measured before the auto-batch fix landed in the same branch
resolved `max_batch_size: auto` 5 → 28; the 630.19 row is auto=28. The
pre-fix default (auto=5) read 224.68.

Readings:

- **Single stream imp leads by 22%** (native FP4 decode against Marlin
  dequant-to-BF16 — vLLM's own startup log says sm_120 has no native FP4
  path for it), consistent with the 2026-08-16 result.
- **From concurrency 8 up vLLM leads: +41% at 8, +58% at 32.** Its
  per-stream decode falls 69 -> 46 tok/s across 1 -> 32 streams where imp
  falls 85 -> 29. The batched-GDN work in this branch closed the gap from
  ~6.5x (81.5 tok/s aggregate before it) to 1.58x; the rest is the open
  concurrency-scaling gap and lives in `docs/roadmap.md`.
- vLLM serves 32 streams over 16k model len inside 0.90 utilization on this
  card — the Mamba-cache-block startup constraint from 2026-08-16 is handled
  by `--max-num-seqs 32`.

### The 1.58x concurrency gap, attributed (2026-08-25)

Both engines profiled under nsys (`--cuda-graph-trace=node`, host Nsight
2026.1.3 mounted into each container) serving the identical 32-stream wave
(6400 tokens, 200 per stream, same checkpoint, same client). Under the
profiler imp reads 908 tok/s and vLLM 1477 — both within noise of their
unprofiled numbers, so the windows are representative. Per emitted token,
steady-state wave only:

[PROV: commit=9ff730db date=2026-08-25 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4-CT cuda=13.3 path=server-api cmd=nsys+load32.py n=1
       window=wave2 tokens=6400]

| us/token | imp | vLLM 0.27.1 | delta | share of gap |
|---|---:|---:|---:|---:|
| GEMM class | 613 | 468 (Marlin 413 + bf16 rest 55) | 145 | 34% |
| **GPU idle** | 179 | 36 | 143 | 34% |
| norms | 35 | 10 | 26 | 6% |
| GDN scan | 181 | 158 | 23 | 6% |
| M=1 GEMV (wave tails) | 22 | — | 22 | 5% |
| activation quantize | 15 | — | 15 | 4% |
| conv1d | 17 | 5 | 11 | 3% |
| attention decode | 22 | 11 | 10 | 2% |
| cuBLAS fp16 + other | 38 | 18 | 25 | 6% |
| **wall** | **1125** | **703** | **422** | 100% |

Three readings, one of them a correction:

- **The "structural FP4 trade" framing was wrong.** imp's GEMM class costs
  MORE than vLLM's despite native FP4: Marlin (W4A16, split-K with atomic
  reduction, built for small M) beats the CUTLASS block-scaled cooperative
  tile by ~24% on the same shapes and the same card. The ~41 us "ceiling"
  the five-approach survey established holds only for no-K-split designs —
  Marlin is the running existence proof for the split-K route.
- **Idle is as large as the GEMM deficit.** imp leaves the GPU empty 15.9%
  of the steady-state window (vLLM: 5.2%): 415k inter-kernel gaps, 687 ms
  of them under 100 us (launch/replay density — imp issues 438k kernels in
  the window against vLLM's 200k, 2.2x per token) and 484 ms above 100 us
  (host moments; the largest single gaps are 42/39/26 ms).
- The remaining ~135 us/token is a sum of small classes, mostly coupled to
  the launch count.

Ceiling if both engine-side posts closed: ~1125 - 288 = ~837 us/token,
i.e. ~1195 tok/s aggregate — within 19% of vLLM without touching the scan.

**Update after the idle-post fixes (2026-08-25 night):** deferred token
delivery (#1758), the graph prewarm (#1761) and the burst-serving fixes
(#1762: HTTP pool sized to streams, token-charged prefill budget,
id-based prefill rotor) moved the same 32-stream wave from 936-990 to
**1028-1073 tok/s on every wave** — the wave-1 ramp (629) is gone
entirely. n=1 reads 84.0, n=8 374.5, n=32 aggregate 1050.1 (median of
3). Against vLLM's 1477 profiled the gap stands at ~1.4x; what remains
of the engine-side posts is the GEMM class (Marlin port, a standalone
project) and cross-sequence prefill batching (`docs/roadmap.md` item
0(a)/(d)).

[PROV: commit=957653ea date=2026-08-25 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4-CT cuda=13.3 path=server-api cmd=bench_conc.py+waves3.py n=3
       flags=max_batch_size=32,max_seq_len=4096]

**Update after the 2026-08-26 pair (producer-quantize fusion #1773, BF16
GDN state #1776 + plan fix #1777):** the same pinned 32-stream wave reads
**1362.0 tok/s** median (alternating flag A/B, 3/3 pairs: FP32 state
1210.5), and pure defaults read 906.9 (from 842.2). Against vLLM's 1477
profiled the pinned gap stands at **~1.08x**. The scan itself is at the
bandwidth ceiling in both dtypes (1527/1570 GB/s isolated); what remains
is the launch-coupled idle, cross-sequence prefill batching and the
elementwise fusion tail (`docs/roadmap.md` item 0).

[PROV: commit=b516d9a7 date=2026-08-26 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4-CT cuda=13.3 path=server-api cmd=gdn_bf16_ab.sh+gdn_default_ab.sh
       n=3/arm flags=max_batch_size=32,max_seq_len=4096,kv_cache.max_blocks=2387
       (pinned arm) / defaults (default arm)]


### The GEMM-class lever, shipped (2026-08-25, night)

The native mxf4nvf4 small-M GEMM (v2, #1766, `gemm.nvfp4_smallm` default ON)
closes the table's largest engine-side post. Same checkpoint, same 32-stream
wave shape (32x unique short prompts, 300-tok greedy gens), server under nsys
(`--cuda-graph-trace=node`, Nsight 2026.1.3), kernel-time sums over the
3-wave window divided by the 28665 emitted tokens:

[PROV: commit=41b9d7e1 date=2026-08-25 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=server-api cmd=nsys+conc_client.py n=1
       window=3-waves tokens=28665 flags=--max-concurrent 32
       --set runtime.max_batch_size=32 --set runtime.max_seq_len=4096]

| us/token | imp before (steady wave) | imp with v2 (3-wave window) | vLLM 0.27.1 |
|---|---:|---:|---:|
| GEMM class | 613 | **388.9** | 468 (Marlin 413 + bf16 rest 55) |
| activation quantize | 15 | 19.4 | — |
| GDN scan | 181 | 188.8 | 158 |
| norms | 35 | 36.2 | 10 |

The v2 kernel is the top GPU kernel of the serving run (51.5% of kernel
time, med 29.2 us/launch in-situ vs the CUTLASS tile's 41.4) and the CUTLASS
block-scaled tile disappears from the decode class (remaining instances are
prefill). Window caveat: this window includes wave boundaries and tails, so
its wall (953 us/token) and idle are NOT comparable to the steady-state
wave-2 method above; the like-for-like end-to-end number is the alternating
A/B: 992.5 -> 1151.7 tok/s aggregate at 32 streams (+16.0%), 363.8 -> 494.6
at 8 (+36.0%), 3 trials/arm, `tools/analysis/smallm_v2_conc_ab.sh`.

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
The Qwen3-8B defaults cell was re-measured 2026-07-12 on `f3c228a0` after
#977 made FP8 KV the default on hint-less Qwen3 GGUFs.

| Model | Quant | pp8192 tok/s | tg512 @16k (defaults) | discovery-day @16k |
|---|---|---:|---:|---:|
| Qwen3-8B | Q8_0 | 13 268 | **208.9** (FP8 KV auto since #977; 151.9 on FP16 KV) | 58.7 (spec ungated, #968) |
| Qwen3-Coder-30B-A3B | NVFP4 | 35 516 | 269.5 | 269.5 |
| Qwen3.6-35B-A3B | NVFP4 | 14 887 | **264.3** | IMA crash (#963/#967), then 72.7 streaming (#969) |
| Qwen3.6-35B-A3B | Q4_K_M (GGUF) | 9 436 (pp16384) | **234.2** | 69.6 under streaming + silent OOB reads (#967/#969) |

**128K single-chunk prefill reference** (2026-07-24, commit `d8bc45a8`, CUDA
13.3, healthy-host clocks sampled during the run — 13 801 MHz mem / ~575 W):
Qwen3-14B **NVFP4** `pp131072 = 3 792 tok/s` (34.6 s TTFT), command
`imp-cli --model Qwen3-14B-NVFP4 --bench --bench-pp 131072 --bench-reps 5
--prefill-chunk-size 0 --max-tokens 1 --temperature 0 --max-seq-len 140000`.
The auto `max_seq_len` ceiling is 128K since `d8bc45a8`. The Q6_K GGUF
north-star model cannot host this measurement on 32 GB (its KV pool tops out
near 75K tokens beside the dual GGUF+NVFP4 weight residency), which is why
the CI TTFT gate band in `tests/perf_baseline_north_star.json` ends at
pp65536 — the 64K row is that model's VRAM-feasible ceiling, not a coverage
gap.

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
- FP8-KV is worth **+39..41%** at 16k on Qwen3-8B. It originally did not
  auto-engage on GGUF sources (the `kv_cache.dtype=auto` upgrade was keyed
  on the checkpoint's `kv_cache_quant_algo` hint, which GGUF files don't
  carry) — **fixed in #977**: hint-less Qwen3 dense/MoE checkpoints now
  default to FP8 KV via the PPL-gated no-hint arch allowlist
  (`kv_fp8_no_hint_default_safe`). Other GGUF families still need
  `--kv-fp8` until measured.

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
| 2026-07-12 | decode-pipeline PR | Qwen3-Coder-30B-A3B-FP4 | 16 | **970** STREAMING closed-loop median (915 with `runtime.decode_pipeline=false`, **+6.0%**; 6 trials each; TPOT 17.0 → 16.1 ms) | pipelined batched decode: step N+1 is enqueued device-side (token chain + graph replay + sampler enqueue) before step N's tokens are read back, so per-token SSE delivery and host bookkeeping overlap GPU compute. Confirmed with a GIL-free client (4 processes × 4 streams): ON 1 065 vs OFF 1 013 median (+5.2%, 3 trials each) — a single-process Python SSE client depresses the absolute numbers ~100 tok/s, the pipeline delta survives either way. Streaming rows are NOT comparable to the non-streaming rows below; non-streaming @16 reads ~1 070 on this build with the pipeline on or off — the non-streaming harness attributes whole 200-token completions at once, so its window quantization (~±70 tok/s) swamps a single-digit delta. Same server config as the 1 173 row; healthy clocks sampled during runs |
| 2026-07-12 | batched-sampling PR | Qwen3-Coder-30B-A3B-FP4 | 16 | **1 173** sustained closed-loop median (1 138/1 173/1 213 over 45-60 s windows; ~73 tok/s = 13.6 ms TPOT per stream) | with `gemm.nvfp4_lm_head_cutlass=true` (opt-in at the time; default ON since 2026-07-12 after the per-family PPL sweep — MoE/hybrid +1.9–2.1%, dense +0.2–0.5%/noise-level, batch=1 bit-identical). Levers over the 861 baseline: batched sampling readback (one pinned D2H + one sync per step instead of 16 pageable-D2H round-trips), row-parallel top-k/top-p kernels (one partial+finalize launch pair for the whole batch), kernel-based residual copies (WDDM D2D submission cost), static ban-list device cache, CUTLASS lm_head. Now ABOVE the published vLLM reference (1 157 aggregate, 13.6 ms TPOT — cloudrift.ai) while keeping 5.4× its single-stream decode (396 tok/s, unchanged). Closed-loop methodology matches vLLM bench's max-concurrency mode; the old fixed-batch harness (waits for all 16, includes ramp-down tails) reads ~1 150 on the same build |
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

### Attention-model replay (prefix cache, `tools/agent_replay_bench.py`)

The same growing-transcript shape on the standard attention path (paged prefix
cache, default-on). Harness replays a scripted coding-agent session (system +
tool spec, then per turn: user ask → assistant tool call → tool result), timing
per-turn TTFT (`max_tokens=1` `time_total`) as the transcript grows; the
cache-OFF arm defeats the cache with a per-turn-novel prefix (note:
`cache_prompt=false` does **not** disable the server-global prefix cache — only
prefix novelty does). `make bench-agentic MODEL=<name>` boots the server and
runs this plus the concurrency harness. Numbers 2026-07-15, CUDA 13.3, healthy
clocks; TTFT to first content token.

| Model | Turn 0 (≈360 tok) | Turn 15 (≈5.2 k tok) cache-ON | Turn 15 cache-OFF | Deepest-turn speedup |
|---|---:|---:|---:|---:|
| Qwen3-8B Q8_0 (dense) | 32 ms | **19–35 ms flat** | 439 ms | **23×** |
| Qwen3-Coder-30B-A3B-FP4 (MoE) | 20 ms | **33 ms** | 168 ms | **5.2×** |

cache-ON TTFT stays flat as the prompt grows to ~5 k tokens (only the ~320-token
new suffix is prefilled each turn; `cached_tokens` tracks `prompt_tokens` within
~30); cache-OFF grows ~linearly with depth (8.6× / 7.1× turn0→turn15). The MoE's
smaller cache-OFF slope reflects its 3 B active params (cheaper full re-prefill).
Concurrency TTFT/ITL on the same models via `tools/agent_bench.py`: Coder-30B
single-stream ITL 3.3 ms (269 tok/s, matches the hero decode baseline), aggregate
363 tok/s at 16 concurrent; Qwen3-8B 143 → 293 tok/s (1 → 16).

## Output-quality gate

Throughput numbers say nothing about correctness — that lesson is paid for
(see git history around 2026-06-04). Every perf-relevant change must also pass
`python3 tools/analysis/degen_suite.py` against a running server.

*(This file is updated in the same commit as the measurement-relevant change;
check `git log BENCHMARKS.md` for the measurement provenance trail.)*

## Agentic reliability vs llama.cpp (2026-07-26)

Speed was published per hero model; whether the *JSON contract* or a *tool call*
survives was not — against another engine, same model, same requests. This is
the first cross-engine measurement of that (roadmap gap 7).

**Setup**: Qwen3-8B-Q8_0 GGUF on both engines, same prompts, `temperature=0`,
5 repetitions per case, `max_tokens=200` (a budget an agent would plausibly
set). imp commit at `docs/skills-generalize`; llama.cpp `ff067f76d` (build
10133) served with `--jinja -fa 1 -ngl 99`. Harness:
`tools/analysis/agentic_compare.py` (re-runnable, engine-agnostic).

| Case | imp (default) | llama.cpp (default) | llama.cpp (`enable_thinking:false`) |
|---|:--:|:--:|:--:|
| `json_schema` schema-valid | **5/5** (10 tok) | 0/5 | 5/5 (23 tok) |
| `json_object` parses | **5/5** (27 tok) | 0/5 | 5/5 (18 tok) |
| `tool_choice=required` emits a call | 5/5 (21 tok) | 5/5 (197 tok) | 5/5 (26 tok) |
| tool arguments parse + required field | 5/5 | 5/5 | 5/5 |
| `tool_choice=auto` does not force a call | 5/5 | 5/5 | 5/5 |

**Read this as a defaults difference, not a capability difference.** llama.cpp's
constrained decoding is correct — given `enable_thinking:false`, or simply a
larger budget (the same schema request completes at 447 tokens, ~420 of them
reasoning), it passes everything. What differs is what happens *out of the box*:
a think-capable model spends the whole 200-token budget reasoning and returns an
empty `content`, so the agent gets nothing. imp suppresses thinking for
json/tool requests automatically, which is why it answers in 10 tokens without
the client knowing to configure anything.

Tool calling is equally reliable on both — llama.cpp emits the call even while
thinking, it just pays ~9× the tokens for it at this budget.

### Budget sweep (Qwen3-8B, default settings, 3 reps)

Where each engine starts keeping the contract, as the agent's `max_tokens` grows:

| budget | imp | llama.cpp |
|---|---|---|
| 100 | all 6 categories pass | only `tool_choice=auto` passes |
| 200 | all 6 pass | tools flaky (1/3), JSON 0/3 |
| 400 | all 6 pass | tools pass, JSON still 0-1/3 |
| 800 | all 6 pass | all 6 pass (`json_schema` costs 447 tok) |

imp is budget-independent because it does not spend the budget thinking on
json/tool requests; llama.cpp needs roughly 800 tokens before a think-capable
model gets to the answer.

### Control: a model that does not think (Llama-3.2-3B-Q8_0, budget 200)

If the difference above is really *thinking*, it should vanish on a model that
has none. It does — and the control found a genuine bug in **imp**, not in
llama.cpp:

| Case | imp (before) | imp (after fix) | llama.cpp |
|---|:--:|:--:|:--:|
| `json_schema` / `json_object` / multi-turn | 3/3 | 3/3 | 3/3 · **0/3** · 3/3 |
| `tool_choice=required` + args | **0/3** | **3/3** | 3/3 |
| `tool_choice=auto` stays optional | 3/3 | 3/3 | **0/3** |

imp was dropping Llama-3.2 tool calls: the model emits a bare JSON object where
Llama 3.1 used the `<function=F>` envelope, so a correct call was handed back as
`content` and an agent saw none. Fixed (parser accepts the bare form when the
name matches a tool the request offered). The two llama.cpp cells are its own
gaps at this budget: `json_object` returned non-JSON, and `tool_choice=auto`
forced a call on a plain chat turn.

### Three families, 8-turn sessions (budget 200, 3 reps)

| Model | imp | llama.cpp | note |
|---|:--:|:--:|---|
| Qwen3-8B-Q8_0 (thinks) | **6/6** | 3/6 | llama.cpp reaches 6/6 at budget 800 |
| Llama-3.2-3B-Q8_0 | **6/6** | 4/6 | imp was 4/6 until the tool-call fix this found |
| gemma-3-12b-Q4_K_M | 5/6 | 4/6 | `tool_forced` fails on BOTH — Gemma-3 has no native function calling, a model limit, not an engine gap |

The 8-turn `json_multiturn` check passes everywhere on both engines: neither
loses the JSON contract as history grows, which is the failure mode template
drift and KV reuse would produce.

Where they differ consistently: **`json_object` holds on imp across all three
models and fails on llama.cpp for two of them** (returns prose, not JSON), and
llama.cpp's `tool_choice=auto` forced a call on a plain chat turn with
Llama-3.2. imp's own gap was Llama-3.2 tool calls (fixed; see the CHANGELOG).

Scope: three model families, four budgets, 8-turn sessions, 3-5 repetitions,
one llama.cpp build, GGUF only. **vLLM/SGLang are not covered** — they would
need a different weight format and more VRAM than this box has free while
serving; that is the honest remaining gap, not an oversight.
