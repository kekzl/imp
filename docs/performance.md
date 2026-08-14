<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# Performance

## Methodology

| | |
|---|---|
| Hardware | Single NVIDIA RTX 5090, 32 GB GDDR7, `sm_120a`, custom water loop |
| Toolchain | CUDA 13.3 (rows dated before 2026-05-30: 13.2.1), CUTLASS 4.7.0 (rows before 2026-08-13: v4.5.1–v4.6.2), GCC 13.3, Release Docker build |
| imp config | NVFP4 decode cache + FP16 prefill (FP8 auto-disabled on sm_120), CUDA Graphs on |
| llama.cpp | `b8445+`, flash attention on, full offload (`-ngl 99`) |
| Sampling | Greedy (temp = 0) |
| Repetitions | 3 (decode); prefill spread depends on the model, see [`PERF.md`](PERF.md) |
| Reported | Mean; decode (`tg256`) is the reliable A/B signal |

Refresh the baseline with `scripts/gen_perf_baseline.sh` after any intentional perf
change — or any intentional memory change: the same file pins peak VRAM
(`metrics.memory_mb.own_peak_mb`, gated at `thresholds.vram_increase_pct` by
`scripts/verify.sh`), and a refresh re-pins it silently.

**Last refreshed**: decode numbers are owned by the SHA-anchored
[`BENCHMARKS.md`](BENCHMARKS.md) (see Decode Throughput below — no table is
duplicated here). Prefill + KV-cache tables below are
**historical** (2026-05-27 era, CUDA 13.2.1) — prefill moves enough across
process starts that it is not maintained as a comparison table.
llama.cpp / vLLM comparison from cross-engine bench 2026-05-24.

**Bench-mode caveat**: `--bench --max-tokens 128` sizes the engine to the bench
workload, so it does not measure the served regime (`imp-server` defaults to the
model's full context). Use `imp-cli --prompt` or a real server request for production
numbers. The old mechanism behind this caveat — bench-mode KV sizing changing what
was left for the NVFP4 cache — no longer applies: since #1106 the weight caches are
built *before* the KV pool and the pool takes the measured residual, so the cache
budget no longer depends on how much KV was allocated first. Two related traps are
fixed rather than caveated: the double-charged cache reservation (#1100/#1102) and
the cache-starvation collapse at 0 MiB free (#1103).

## Decode Throughput

The per-model decode table is **not duplicated here** — it drifts. The canonical,
SHA-anchored decode numbers (model · quant · metric · tok/s · commit · CUDA · exact
command) live in [`BENCHMARKS.md`](BENCHMARKS.md), and the CI gate is
`tests/perf_baseline.json` (refresh via `scripts/gen_perf_baseline.sh`). Heroes for
orientation: Q8 tg128 ≈ 268 · 14B-Q6_K north-star ≈ 158 @ctx2048 · NVFP4 MoE
tg256 in the 250–340 range (e.g. Qwen3-Coder-30B 338, Qwen3.6-35B 320 since the
#949 FP8 SSM-projection sidecar, was 257).

Decode is measured with CUDA Graphs ON, 10 reps, isolated + clock-warmed. Healthy-host
sanity check: ~2850 MHz SM / 13801 MHz mem / ~500 W during the bench — decode can read
8–15% low on depressed-host days (issue #526). **Rule out held VRAM before reaching for that explanation**: `nvidia-smi --query-gpu=memory.used` against the ~1.3–1.6 GiB WSLg baseline, because `--query-compute-apps` is blank on WSL2 even while memory is committed. A 16.4 GiB leftover from a killed container read −5.5% at healthy clocks on 2026-08-11 and passed at −1.24% once it cleared.

## Prefill Throughput (pp512) — historical (2026-05-27)

| Model | Params | Quant | imp | llama.cpp | Notes |
|-------|-------:|-------|----:|----------:|------|
| Qwen3-4B | 4.0B | Q8_0 | **23189** | 21337 | |
| Qwen3-8B | 8.2B | Q8_0 | **14453** | 14172 | |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **14091** | 11149 | |
| Qwen3.5-9B (GDN) | 9.2B | Q8_0 | **10162** | — | |
| Llama-3.2-3B | 3.2B | Q8_0 | **27041** | — | |
| Qwen3.6-35B-A3B | 35B (3B active) | Q4_K_M | **3076** | — | |
| Qwen3.6-35B-A3B | 35B (3B active) | NVFP4 | **1092** | — | |
| Nemotron-3-Nano-30B-A3B | 30B (3B active) | NVFP4 | **11,532** | — | hybrid Mamba2+MoE+attention |
| Gemma-4-26B-A4B-it | 26B (4B active) | Q4_K_M | **1840** | 196 | |
| Gemma-4-26B-A4B-it | 26B (4B active) | NVFP4 | **1472** | — | |

## KV Cache Quantization (Llama-3.2-3B Q8_0) — historical (2026-05-27)

| KV Cache | Decode tg256 | Prefill pp512 | tg @ 5K ctx | tg @ 20K ctx | VRAM |
|----------|------:|--------:|------:|------:|------:|
| **FP16 (default)** | **319** | **25808** | 213 | 156 | 100% |
| FP8 E4M3 (`--kv-fp8`) | 319 | 25808 | 213 | 156 | 50% |
| INT4 (`--kv-int4`) | 305 | 16272 | 190 | 122 | 25% |
| NVFP4 (`--kv-nvfp4`) | parity-FP16 | — | — | — | 25% |
| MXFP4-KV (`--kv-mxfp4`) | parity-NVFP4 | — | — | — | 25% |

Default is **FP16**. FP8 has perf parity with FP16 on Qwen3 and Qwen3.5/3.6 GDN but breaks Llama, Mistral-Small-3.1, DeepSeek-R1-Distill out of the box. Use `--kv-fp8` per-model after testing.

## Notes

- **Prefill variance**: prefill moves far more than decode across process starts, and it is the MoE path that moves, not cuBLAS autotuning as long assumed. Figures and provenance: [`PERF.md`](PERF.md). Compare decode for A/B testing, or pair and alternate the arms.
- **Gemma-4**: CUDA Graphs enabled. Decode is 1.21x llama.cpp on Q4_K_M. Q4_K_M can degenerate on complex code-gen prompts — use Q5_K_M or Q8_0 when output quality matters.
- **GDN models**: Use FP16 prefill weights for numerical stability, reducing prefill throughput ~8% vs FP8.
- **MXFP4 Prefill**: `--mxfp4-prefill` uses CUTLASS block-scaled GEMM; currently ~10% slower than FP8 cuBLASLt due to activation quantization overhead.
- **NVFP4 / MXFP4 KV**: Both store FP4 nibbles at 25% of FP16, differ in scale encoding. NVFP4-KV is decode-parity with FP16 via PTX `cvt.rn.f16x2.e2m1x2` gather. MXFP4-KV is approximately NVFP4 quality at 16K context.
