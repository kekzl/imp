# Benchmarks

Reproducibly anchored measurements. Every row states **when**, **on what
commit**, **with which CUDA version and quant**, and **the exact command** —
re-run the command on the stated commit to reproduce. The commit SHA is the
authoritative version; tagged releases (current: **v0.10.0**) snapshot a SHA.

**Hardware (constant across all runs):** single RTX 5090 (GB202, 32 GB GDDR7,
water-cooled — never thermally throttled), Ryzen host, WSL2, Docker.
**Method:** greedy (temp = 0), CUDA Graphs on, 10 repetitions, isolated run
(one model per process, no concurrent GPU work), clocks warmed before timing
(the GPU idles at low clocks and takes ~1 s to ramp — cold first runs read
artificially LOW). `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Decode (tg) is the
reliable A/B signal; prefill (pp) varies up to 2.6× across container restarts
(cuBLAS autotuning) and is therefore not tabulated for comparisons.

The CI-gated canonical baseline lives in
[`tests/perf_baseline.json`](tests/perf_baseline.json) (3% decode / 5%
prefill regression gate); refresh it via `scripts/gen_perf_baseline.sh`.

## GGUF decode

| Date | Commit | CUDA | Model | Quant | Metric | tok/s | Command |
|---|---|---|---|---|---|---:|---|
| 2026-06-07 | `perf_baseline.json` (CI gate) | 13.3 | Qwen3-8B | Q8_0 | tg128 | 286.4 | `imp-cli --model Qwen3-8B-Q8_0.gguf --bench --bench-pp 16 --bench-reps 10 --max-tokens 128` |
| 2026-05-29 | `perf_baseline_north_star.json` | 13.3 | Qwen3-14B | Q6_K | tg128 @ctx2048 | 157.7 | `… --max-seq-len 2048` |
| 2026-06-09 | `ec9145b3` | 13.3 | Qwen3-14B | Q6_K | tg128 | 164 | `imp-cli --model Qwen3-14B-Q6_K.gguf --bench --bench-pp 16 --bench-reps 10 --max-tokens 128` |
| 2026-06-09 | `ec9145b3` | 13.3 | Gemma-4-26B-A4B | Q4_K_M | tg128 | 273 | `imp-cli --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --bench --bench-pp 16 --bench-reps 10 --max-tokens 128` |

> Canonical gated decode number = `perf_baseline.json` Qwen3-8B-Q8_0 tg128 =
> 286.4 (cold-median, 5 trials × 5 reps). Single-session warm reads carry the
> documented ±5–10 % host/driver day-to-day variance (issue #526) — sample
> `clocks.mem` during the bench (healthy = 13801 MHz / ~500 W under prefill).

Against llama.cpp (b8445+, full offload, flash attention on): imp wins dense
GGUF decode by **+37–72%** and loses MoE/hybrid GGUF decode on
Qwen3.6-35B-A3B (~−31%, structural FP16-projection tax on the GDN path).

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
| Qwen3.6-35B-A3B | 35B (3B) | 257 |
| Gemma-4-26B-A4B | 26B (4B) | 266 |
| Nemotron-3-Nano-30B | 30B (3B) | 128 |
| gpt-oss-20b¹ | 21B (3.6B) | 325 |

¹ gpt-oss (PRs #572/#574): SafeTensors MXFP4 source, experts converted to
NVFP4 at load (bit-exact nibbles, power-of-two scales) and registered for the
CUTLASS grouped-GEMM prefill — pp512 ≈ 16-19k tok/s. Attention stays on the
cuBLAS path (attention sinks). Decode 310-345 depending on host state.

On `sm_120`, native-NVFP4 decode is effectively uncontested (vLLM gates its
NVFP4 path on `tcgen05`/falls back to Marlin on the 5090; llama.cpp has no
native NVFP4 path). NVFP4 **prefill** trails vLLM by ~1.4× — the gap is in
attention, not the grouped GEMM (near roofline). Nemotron-3-Nano is
arch-limited (hybrid Mamba2 + attention FP16-projection mix).

## Output-quality gate

Throughput numbers say nothing about correctness — that lesson is paid for
(see git history around 2026-06-04). Every perf-relevant change must also pass
`python3 tools/analysis/degen_suite.py` against a running server.

*(This file is updated in the same commit as the measurement-relevant change;
check `git log BENCHMARKS.md` for the measurement provenance trail.)*
