<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# Benchmarks

Reproducibly anchored measurements: every row states **when**, **on what commit**, **with which
CUDA version and quant**, and **the exact command**; re-run the command on the stated commit to
reproduce. The commit SHA is authoritative, tagged releases snapshot a SHA; every sweep this file
superseded (older competitive re-sweeps, the imp-vs-vLLM concurrency progression, the
batched-decode optimization history) moved verbatim to
[`archive/benchmarks_pre_v0.44.md`](archive/benchmarks_pre_v0.44.md).

**Hardware (constant across all runs):** single RTX 5090 (GB202, 32 GB GDDR7, water-cooled,
never thermally throttled), Ryzen host, WSL2, Docker; **method:** greedy (temp = 0), CUDA Graphs
on, 10 repetitions, isolated run (one model per process, no concurrent GPU work), clocks warmed
before timing, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Decode (tg) is the reliable A/B signal; prefill
(pp) varies up to 2.6x across container restarts (cuBLAS autotuning) and is not tabulated for
comparisons; **toolchain (current):** C++23, Ubuntu 26.04 / GCC 15.2, CUDA 13.4.1.

The CI-gated canonical baseline lives in [`tests/perf_baseline.json`](../tests/perf_baseline.json)
(8% decode / 8% prefill regression gate, 10% peak-VRAM ceiling); pinned values, thresholds and
methodology: [`PERF.md`](PERF.md). Refresh via `scripts/gen_perf_baseline.sh`.

## Decode vs llama.cpp (dense GGUF)

Same card, same GGUF, same flags, decode tok/s; imp defaults (n-gram speculation on) against
llama.cpp defaults, full offload, flash attention on.

| Model (shared quant) | imp default | imp spec-off | llama.cpp | imp lead |
|---|---:|---:|---:|---:|
| Qwen3-8B Q8_0 | **385.4** | 284.6 | 160.1 | +141% (+78% spec-off) |
| Qwen3-14B Q6_K | **162.5** | | 114.8 | +42% |
| Qwen3.6-35B-A3B UD-Q4_K_M | **287.9** | | 235.8 | +22% |
| gpt-oss-20b MXFP4 | **382.7** | | 335.9 | +14% |
| Gemma-4-26B-A4B UD-Q4_K_M | **245.0** | | 214.4 | +14% |
| Qwen3-30B-A3B Q4_K_M (non-hero) | 305.5 | | 295.7 | +3% |

[PROV: commit=83cb5178 date=2026-08-30 hw=RTX5090 model=six-model-sweep quant=per-row cuda=13.3
       path=gguf cmd=`make bench-competitive` n=6x2 note=imp defaults vs llama.cpp defaults, full
       offload, flash attention on; spec-off measured for Qwen3-8B only
       (`--set speculative.ngram=false`)]

The spec-off column exists because `imp-cli --bench` builds a strictly-increasing synthetic
prompt (`tools/imp-cli/mode_bench.cpp:19`) that the n-gram drafter's own generation can loop
into: a decode A/B that leaves speculation on can measure whether a checkpoint happens to loop,
not only the verify path. Reproduce with `make bench-competitive` (competitor image pinned by
digest in [`scripts/bench_competitive.sh`](../scripts/bench_competitive.sh), not by tag); two
earlier sweeps (2026-07-12, 2026-08-21) with the full acceptance-mechanism analysis and a
bimodal-drafter case study: [archive](archive/benchmarks_pre_v0.44.md).

## Feature validations (decode)

One-off measurements for shipped decode features, each still the only sweep on record for that
feature.

| Date | Commit | Model (quant) | Feature | Result |
|---|---|---|---|---|
| 2026-07-11 | `fp8-ssm-gguf` | Qwen3.6-35B-A3B hybrid (Q4_K_M) | `gemm.fp8_ssm_proj` on Q8_0-kept GDN projections | tg256 224.4 -> **272.0** (+21%), PPL +1.8% (201-tok corpus), now ahead of llama.cpp (~229) |
| 2026-08-28 | `899301c6` | Qwen3-8B (Q8_0, fp8 KV) | sparse decode attention, 32k ctx | tg136 160.3 -> **199.5** (+24.5%, 3/3 alternating) |
| 2026-08-29 | `sparse-serving` | Qwen3-8B (Q8_0, fp8 KV) | sparse attn, 3 streams x 25k ctx | 155.6 -> **197.7** (+27%, `tools/analysis/serving_sparse_ab.sh`) |
| 2026-08-30 | `nvfp4-loadwidth` | Qwen3.8-27B (NVFP4, nvfp4 KV) | server decode @77k ctx, batch 8 | 64.1 -> **74.2** (+15.7%, 18/18 runs at 125 chunks) |
| 2026-08-30 | `nvfp4-sparse` | Qwen3.8-27B (NVFP4, nvfp4 KV) | + sparse attn @77k ctx, batch 8 | 74.3 -> **100.2** (+35%, NIAH 8-9/10 vs dense 10/10) |

Commands: `imp-cli --bench` for the first three, `imp-server --max-batch 8
[--set attention.sparse_topk_tokens=N]` plus a forced-length client for the last two, all CUDA 13.3.

## GGUF prefill (pp512, INT8-IMMA family, default on since #617)

All rows 2026-06-07, CUDA 13.3, 10 reps, fresh container per run; llama.cpp = build `19e92c3`,
same GGUF files, same day, `-fa 1 -ngl 999 -r 5`. Command:
`imp-cli --model <gguf> --bench --bench-pp 512 --bench-reps 10`.

| Commit | Model (quant) | imp tok/s | llama.cpp | verdict |
|---|---|---:|---:|---|
| `62d96a0e` | Qwen3-30B-A3B MoE (Q4_K_M) | **9970** | 9288 | imp +7% |
| `3dd945d5` | Qwen3-14B (Q6_K) | **6617** | 6522 | imp +1.5% |
| `84790dac` | Gemma-4-26B-A4B MoE (Q4_K_M) | **8946** | 10749 | 1.20x behind |
| `#617` | Qwen3-8B (Q8_0) | **12131** | 13724 | 1.13x behind |
| `62d96a0e` | Qwen3.6-35B-A3B hybrid (Q4_K_M) | **5165** | 8027 | 1.55x behind (GDN share quality-locked FP16) |

**Pre-2026-07-26 prefill figures read high**: prefix caching went default-on in #758 and
`imp-cli --bench` repeats the same prompt, so reps partly measured cache hits; #1061 disabled it
for one-shot CLI runs; Qwen3-8B Q8_0 pp512 went 12131 (above, pre-flip) -> 14515 pinned
2026-07-15 (cache hits) -> 12407 re-pinned 2026-07-26 (cache off again, current band, see
[`PERF.md`](PERF.md)). Decode unaffected; bisect evidence: [archive](archive/benchmarks_pre_v0.44.md).

## NVFP4 SafeTensors decode (tg256)

Baseline rows 2026-06-09, commit `ec9145b3`, CUDA 13.3, warm isolated runs (one model per
process, 10 reps), command `imp-cli --model <dir>/ --bench --bench-pp 16 --bench-reps 10
--max-tokens 256`. Superseding measurements noted per row.

| Model | Params (active) | tok/s |
|---|---|---:|
| Qwen3-8B-cortecs | 8.2B | 270 |
| Qwen3-14B | 14B | 159 |
| Qwen3-30B-A3B-Modelopt | 30B (3B) | 305 |
| Qwen3-Coder-30B-A3B | 30B (3B) | 338 |
| Qwen3.6-35B-A3B | 35B (3B) | 257 -> **320** (2026-07-10, `80864b06`, `gemm.fp8_ssm_proj` sidecar, +19.2% spec-off) |
| Gemma-4-26B-A4B | 26B (4B) | 266 |
| Nemotron-3-Nano-30B | 30B (3B) | 148 -> **386** (2026-08-12, PR #1389: CUDA graphs were wrongly demoted for pure-SSM layers, not an arch limit) |
| gpt-oss-20b | 21B (3.6B) | 325 -> **391** (2026-07-13, `63df2d30`, `gemm.fp8_attn_proj` sidecar, +12%, teacher-forced PPL unaffected by construction) |

gpt-oss experts convert SafeTensors MXFP4 to NVFP4 at load (bit-exact nibbles, power-of-two
scales) and register for the CUTLASS grouped-GEMM prefill (pp512 ~16-19k tok/s); attention stays
on cuBLAS. On `sm_120`, native NVFP4 decode is effectively uncontested: vLLM falls back to
Marlin (its NVFP4 path gates on an opcode family consumer Blackwell lacks, see
[`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md)); llama.cpp has no native NVFP4 path;
per-row superseding evidence: [archive](archive/benchmarks_pre_v0.44.md).

## NVFP4 prefill (FP16-QK FA2 primary hd=128 prefill, #687)

imp rows 2026-06-13, commit `290a163a`, CUDA 13.3, median of 3 isolated trials x 40 reps;
command: `imp-cli --model <dir> --bench --bench-pp <n> --bench-reps 40 --max-tokens 256`. vLLM
reference: 0.22.1 FlashInfer-NVFP4 (fp8 KV), same host, measured 2026-06-11 (older than the imp
rows, ratios carry that cross-day caveat).

| Model | pp | imp tok/s | vLLM tok/s (06-11) | verdict |
|---|---|---:|---:|---|
| Qwen3-30B-A3B MoE | 2048 | **43646** | 34500 | imp +27% |
| Qwen3-30B-A3B MoE | 4096 | **37639** | 36200 | imp +4% |
| Qwen3-14B dense | 2048 | **26918** | 26600 | ~tie (+1.2%) |
| Qwen3-14B dense | 4096 | **24232** | 25300 | 1.04x behind |

imp also wins TTFT/pp512 outright (2.1-3.4x, vLLM has a flat-cost small-M regime). Decode
(tg256 @ctx2048): 14B 159, 30B-A3B ~317.

## Qwen3.8-27B, the quickstart model

The model the [README](../README.md) and [`QUICKSTART.md`](QUICKSTART.md) walk a first-time
reader through: `imp-quantize` output from the FP8 release (`Qwen/Qwen3.8-27B-FP8`, 28.75 GiB
in, 18.80 GiB out), which `scripts/stage-model.sh` produces.

[PROV: commit=52efa361 date=2026-08-16 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4-FP8-export
       cuda=13.3 path=nvfp4-decode cmd=`imp-cli --model <dir> --bench --bench-pp <16|512>
       --bench-reps 10 --max-tokens 128` n=3-processes]

| metric | tok/s | spread across processes |
|---|---:|---|
| decode tg128 | **87.36** | 87.34 / 87.36 / 87.39, 0.06% |
| prefill pp512 | **7565.65** | 7503 / 7566 / 7577, 0.98% |

Weights land at 17.9 GiB resident, ~7.7 GiB left for KV on a 32 GB card. Bounded by weight
bandwidth, not the LM head (2.4 GiB of the 17.9, served from the NVFP4 decode cache, a trade
measured at +10.4% decode for +0.99% perplexity, see [`quantization.md`](quantization.md)).

## imp vs vLLM at concurrency

Same 19.6 GiB compressed-tensors NVFP4 checkpoint on both engines (`imp-quantize --format vllm`
from `Qwen/Qwen3.8-27B-FP8`), same client (`tools/analysis/conc_client.py`, 300-token greedy
gens, `/v1/completions`, aggregate = completion tokens / wall, median of 3 waves), fresh server
per arm, 3 alternating trials, imp pinned (`--max-concurrent CONC --set
runtime.max_batch_size=32 --set runtime.max_seq_len=4096 --set kv_cache.max_blocks=2387`).
Harness `tools/analysis/vllm_conc_ab.sh`.

[PROV: commit=a44298cb date=2026-09-02 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4-CT
       cuda=13.3 path=server-api cmd=tools/analysis/vllm_conc_ab.sh n=3-trials-x-3-waves-per-arm
       vllm=v0.27.1 / v0.28.0]

| run | shape | imp tok/s (median) | vLLM tok/s (median) | imp vs vLLM |
|---|---|---:|---:|---:|
| 1 | 32 streams, 38-tok prompts, vLLM 0.27.1 | **1807.9** | 1447.8 | **+24.9%** |
| 2 | 32 streams, 38-tok prompts, vLLM 0.28.0 | **1833.8** | 1410.7 | **+30.0%** |
| 3 | 8 streams, 36-tok prompts, vLLM 0.27.1 | **573.0** | 495.8 | **+15.6%** |
| 4 | 32 streams, 1082-tok prompts, vLLM 0.27.1 | **873.4** | 497.8 | **+75.5%** |
| 5 | dense Qwen3-14B-NVFP4, 32 streams, 38-tok, vLLM 0.27.1 | 3480.4 | **3767.9** | -7.6% |
| 6 | dense Qwen3-14B-NVFP4, 8 streams, 36-tok, vLLM 0.27.1 | **1085.4** | 1005.8 | **+7.9%** |
| 7 | dense Qwen3-14B-NVFP4, 32 streams, 982-tok, vLLM 0.27.1, imp KV pool 8192 blocks | 1470.6 | **2492.6** | -41.0% |
| 10 | run 9 with `attention.paged_fp8_multitok=4`, paced serving prefill (#1950-#1952), grouped FP8 decode attention (#1953) | 2491.2 | 2490.9 | +0.0% (2/3 pairs) |

The GDN-hybrid checkpoint (runs 1-4) leads vLLM at every shape tried; the dense counter-probe
(runs 5-10) crosses over between 8 and 32 streams, where vLLM's Marlin W4A16 path and larger
prompts favour vLLM; single-stream imp leads 17-22% on both checkpoints (native FP4 decode vs
vLLM's Marlin dequant-to-BF16 fallback, no native FP4 path on `sm_120`). The 32-stream GDN-hybrid
aggregate moved 81.5 -> 1807.9 tok/s across this branch's levers; runs 8-9 and the profiling
breakdown that attributed the gap (GEMM class, GPU idle, launch density):
[archive](archive/benchmarks_pre_v0.44.md).

## Long context (pp8192 / tg512 @ 16k ctx)

All rows 2026-07-11, commit `905630e2` (re-measured after #967/#968/#969), CUDA 13.3, isolated
runs; command: `imp-cli --model <m> --bench --bench-pp {8192|16384} --bench-reps {5|3}
--max-tokens {64|512} --max-seq-len {9216|17408}`. pp carries the usual restart variance; tg is
the signal.

| Model | Quant | pp8192 tok/s | tg512 @16k (defaults) |
|---|---|---:|---:|
| Qwen3-8B | Q8_0 | 13268 | **208.9** (FP8 KV auto since #977; 151.9 on FP16 KV) |
| Qwen3-Coder-30B-A3B | NVFP4 | 35516 | 269.5 |
| Qwen3.6-35B-A3B | NVFP4 | 14887 | **264.3** |
| Qwen3.6-35B-A3B | Q4_K_M (GGUF) | 9436 (pp16384) | **234.2** |

FP8-KV is worth **+39..41%** at 16k on Qwen3-8B; auto-engages on hint-less GGUF since #977
(`kv_fp8_no_hint_default_safe` allowlist). **128K single-chunk prefill** (2026-07-24, `d8bc45a8`):
Qwen3-14B NVFP4 `pp131072 = 3792 tok/s` (34.6s TTFT, `--prefill-chunk-size 0
--max-seq-len 140000`); auto `max_seq_len` ceiling is 128K since that commit; discovery-day
(pre-fix) numbers and the #963/#964/#967/#968/#969 root causes: [archive](archive/benchmarks_pre_v0.44.md).

## Concurrent serving throughput (batched decode)

Aggregate throughput = sum of completion tokens / wall-clock across N concurrent
`POST /v1/chat/completions` on `imp-server --model Qwen3-14B-NVFP4 --max-batch 16`
(`max_tokens=200`, `temperature=0.7`).

| Date | Commit | Model | Concurrency | Aggregate tok/s |
|---|---|---|---:|---:|
| 2026-07-12 | batched-sampling PR | Qwen3-Coder-30B-A3B-FP4 | 16 | **1173** sustained closed-loop median, above vLLM's published 1157 reference (cloudrift.ai) while keeping 5.4x its own single-stream decode (396 tok/s) |
| 2026-06-23 | pre-`#745` | Qwen3-14B-NVFP4 | 16 | 472 (single-block sampler + per-row LM head, the pre-fix floor) |

**+62% from two fixes** (#745 sampler, #746 LM head): top-k/top-p sampler 36% -> 6% of decode GPU
time; NVFP4 LM head 18% -> 7% (per-sequence M=1 GEMV loop -> batched-M). Single-stream decode is
unchanged by design, both fixes touch only the n>1 path; full optimization sequence (pipelined
batched decode, row-parallel sampling, CUTLASS LM head) and the vLLM reference details:
[archive](archive/benchmarks_pre_v0.44.md).

### F16 KV decode attention at 32 streams (2026-09-03)

Models whose KV stays FP16 under `kv_cache.dtype=auto` (GGUF without an FP8 hint: Llama, Mistral,
Gemma, Phi) moved from a cooperative GQA kernel (22% of DRAM bandwidth) to a four-tokens-per-warp
kernel (#1880/#1882). Aggregate tok/s at 32 concurrent streams, 1000-token prompts, 300-token
completions, `ignore_eos`, harness `tools/analysis/prefill_cap_conc_ab.sh`.

[PROV: commit=f70e072a date=2026-09-03 hw=RTX5090 model=three-model-sweep quant=per-row cuda=13.3
       path=paged-attention-f16 cmd=tools/analysis/prefill_cap_conc_ab.sh
       n=2-trials-x-3-waves flags=CONC=32,PLEN=1000,KV_BLOCKS=3000,IGNORE_EOS=1]

| Model | KV | cooperative | multitok (new default) | delta |
|---|---|---:|---:|---:|
| Llama-3.2-3B-Instruct-Q8_0 | F16 (auto) | 1623.9 | **2407.6** | +48.3% |
| Phi-4-reasoning-plus-NVFP4 | F16 (forced) | 1099.1 | **1812.0** | +64.9% |
| gemma-3-12b-it-Q4_K_M | F16 (auto) | 218.4 | **252.8** | +15.8% |

Single-stream long-context (`scripts/bench_longctx_ab.sh`, tg128, #1882 split-K vs the #1880
tree), Llama-3.2-3B-Instruct-Q8_0: 8k 366.2 -> **422.3** (+15.3%), 32k 183.1 -> **237.2**
(+29.5%), 64k 110.8 -> **153.1** (+38.1%).

## Multi-turn TTFT (hybrid prefix caching, #831 / v0.15.0)

Setup: `imp-server --model Qwen3.6-35B-A3B-NVFP4 --set runtime.max_seq_len=12288`, 6-turn
growing-history replay (~2k tokens per turn), `max_tokens=60`, `temperature=0`, streaming.
2026-07-02, CUDA 13.3: `v0.14.0` = `2316f2fd`, `v0.15.0` = `e80a26a4`.

| Turn | History | v0.14.0 TTFT | v0.15.0 TTFT | `cached_tokens` |
|---|---|---:|---:|---:|
| 1 | fresh | 1.62s | 1.89s | - |
| 2 | ~2k | 2.85s | 1.41s | 2016 |
| 4 | ~6k | 4.31s | 1.40s | 6080 |
| 6 | ~10k | 6.70s | **1.94s** | 10176 |

v0.14.0 grows linearly with history; v0.15.0 stays flat (**3.5x at ~10k tokens**, gap widening
with context). Recurrent-state snapshots (`server.recurrent_snapshot_mb`, default 256) enable
prefill-only deltas on SSM/GDN models; `usage.prompt_tokens_details.cached_tokens` reports the
hybrid hits.

### Attention-model replay (prefix cache)

Same growing-transcript shape on the standard attention path (paged prefix cache, default-on).
2026-07-15, CUDA 13.3, harness `tools/agent_replay_bench.py`, TTFT to first content token.

| Model | Turn 0 (~360 tok) | Turn 15 (~5.2k tok) cache-ON | Turn 15 cache-OFF | speedup |
|---|---:|---:|---:|---:|
| Qwen3-8B Q8_0 (dense) | 32ms | **19-35ms flat** | 439ms | **23x** |
| Qwen3-Coder-30B-A3B-FP4 (MoE) | 20ms | **33ms** | 168ms | **5.2x** |

`make bench-agentic MODEL=<name>` boots the server and runs this plus the concurrency harness.

## Output-quality gate

Every perf-relevant change must pass `python3 tools/analysis/degen_suite.py` against a running server.

## Agentic reliability vs llama.cpp (2026-07-26)

Qwen3-8B-Q8_0 GGUF on both engines, same prompts, `temperature=0`, 5 reps per case,
`max_tokens=200`; llama.cpp `ff067f76d` (build 10133) with `--jinja -fa 1 -ngl 99`. Harness:
`tools/analysis/agentic_compare.py`.

| Case | imp (default) | llama.cpp (default) | llama.cpp (`enable_thinking:false`) |
|---|:--:|:--:|:--:|
| `json_schema` schema-valid | **5/5** (10 tok) | 0/5 | 5/5 (23 tok) |
| `json_object` parses | **5/5** (27 tok) | 0/5 | 5/5 (18 tok) |
| `tool_choice=required` emits a call | 5/5 | 5/5 (197 tok) | 5/5 |
| tool args parse + required field | 5/5 | 5/5 | 5/5 |
| `tool_choice=auto` does not force a call | 5/5 | 5/5 | 5/5 |

- Defaults difference, not capability difference: llama.cpp's think-capable model spends its whole budget on reasoning and returns empty `content` unless told not to think; imp suppresses thinking for json/tool requests automatically.
- Budget sweep (Qwen3-8B, 3 reps): imp passes all 6 categories from budget 100; llama.cpp needs budget 800 (`json_schema` then costs 447 tok) to match.
- Three families, 8-turn sessions (budget 200, 3 reps): imp 6/6, 6/6, 5/6 (Gemma-3 has no native function calling, a model limit) against llama.cpp 3/6, 4/6, 4/6 (reaches 6/6 at budget 800).
- `json_object` holds on imp across all three models, fails on llama.cpp for two.

**Scope**: three model families, four budgets, 8-turn sessions, 3-5 repetitions, one llama.cpp
build, GGUF only; a non-thinking-model control (Llama-3.2-3B) found and fixed an imp tool-call
regression along the way; vLLM/SGLang not covered. Full per-budget tables and the control-run
detail: [archive](archive/benchmarks_pre_v0.44.md).
