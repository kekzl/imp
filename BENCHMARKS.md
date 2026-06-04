# Benchmarks

Reproducibly anchored measurements. Every row states **when**, **on what
commit**, **with which CUDA version and quant**, and **the exact command** —
re-run the command on the stated commit to reproduce. There are no versioned
releases of this PoC; the commit SHA is the version.

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
| 2026-05-29 | `perf_baseline.json` | 13.3 | Qwen3-8B | Q8_0 | tg128 | 258.9 | `imp-cli --model Qwen3-8B-Q8_0.gguf --bench --bench-tg 128` |
| 2026-05-29 | `perf_baseline.json` | 13.3 | Qwen3-14B | Q6_K | tg128 @ctx2048 | 157.7 | `imp-cli --model Qwen3-14B-Q6_K.gguf --bench --bench-tg 128 --ctx 2048` |
| 2026-05-30 | `bebafd5` | 13.3 | Gemma-4-26B-A4B | Q4_K_M | tg128 | 259 | `imp-cli --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --bench --bench-tg 128` |

Against llama.cpp (b8445+, full offload, flash attention on): imp wins dense
GGUF decode by **+37–72%**, loses GGUF prefill by 1.3–2.4× (no custom IMMA
prefill kernel), and loses MoE/hybrid GGUF decode on Qwen3.6-35B-A3B (~−31%,
structural FP16-projection tax on the GDN path).

## NVFP4 SafeTensors decode (tg256)

All rows: 2026-05-30, commit `bebafd5` (clean main), CUDA 13.3, NVFP4
prequant (Model Optimizer / llm-compressor exports), command pattern
`imp-cli --model <model-dir>/ --bench --bench-tg 256`.

| Model | Params (active) | tok/s |
|---|---|---:|
| Qwen3-8B-cortecs | 8.2B | 277 |
| Qwen3-14B | 14B | 168 |
| Qwen3-30B-A3B-Modelopt | 30B (3B) | 307 |
| Qwen3-Coder-30B-A3B | 30B (3B) | 307 |
| Qwen3.6-35B-A3B | 35B (3B) | 245 |
| Gemma-4-26B-A4B | 26B (4B) | 259 |
| Nemotron-3-Nano-30B | 30B (3B) | 126 |

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
