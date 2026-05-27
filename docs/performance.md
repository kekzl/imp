# Performance

## Methodology

| | |
|---|---|
| Hardware | Single NVIDIA RTX 5090, 32 GB GDDR7, `sm_120a`, custom water loop |
| Toolchain | CUDA 13.2.1, CUTLASS v4.5.1, GCC 13.3, Release Docker build |
| imp config | NVFP4 decode cache + FP16 prefill (FP8 auto-disabled on sm_120), CUDA Graphs on |
| llama.cpp | `b8445+`, flash attention on, full offload (`-ngl 99`) |
| Sampling | Greedy (temp = 0) |
| Repetitions | 3 (decode); prefill varies up to ±2.6× across container restarts (cuBLAS algo selection) |
| Reported | Mean; decode (`tg256`) is the reliable A/B signal |

Refresh the CI baseline with `scripts/gen_perf_baseline.sh` after any intentional perf change.

**Last refreshed**: 2026-05-27 (post PRs #405–#448).
llama.cpp / vLLM comparison from cross-engine bench 2026-05-24.

**Bench-mode caveat**: `--bench --max-tokens 128` allocates less KV VRAM than production, which changes the NVFP4 cache budget. MoE models are most affected — use `imp-cli --prompt` for production numbers.

## Decode Throughput (tg256)

| Model | Params | Quant | imp tg128 | imp pp512 | Notes |
|-------|-------:|-------|----------:|----------:|------|
| Qwen3-8B | 8.2B | Q8_0 | **274** | 26,540 | |
| Qwen3-14B | 12B | Q6_K | **165** | 16,369 | north-star model |
| Qwen3-8B | 8.2B | NVFP4 | **226** | 25,539 | SafeTensors cortecs |
| Qwen3-Coder-30B-A3B | 30B (3B active) | NVFP4 | **270** | 16,800 | SafeTensors Modelopt |
| Qwen3.6-35B-A3B | 35B (3B active) | NVFP4 | **227** | 10,906 | SafeTensors Modelopt |
| Gemma-4-26B-A4B-it | 26B (4B active) | Q4_K_M | **256** | 4,645 | GGUF |
| Nemotron-3-Nano-30B-A3B | 30B (3B active) | NVFP4 | **97** | 11,532 | hybrid Mamba2+MoE+attention, arch-limited |

Decode (tg128) measured in production mode (full context, graphs ON). Prefill (pp512) from `--bench` mode.

## Prefill Throughput (pp512)

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

## KV Cache Quantization (Llama-3.2-3B Q8_0)

| KV Cache | Decode tg256 | Prefill pp512 | tg @ 5K ctx | tg @ 20K ctx | VRAM |
|----------|------:|--------:|------:|------:|------:|
| **FP16 (default)** | **319** | **25808** | 213 | 156 | 100% |
| FP8 E4M3 (`--kv-fp8`) | 319 | 25808 | 213 | 156 | 50% |
| INT4 (`--kv-int4`) | 305 | 16272 | 190 | 122 | 25% |
| NVFP4 (`--kv-nvfp4`) | parity-FP16 | — | — | — | 25% |
| MXFP4-KV (`--kv-mxfp4`) | parity-NVFP4 | — | — | — | 25% |

Default is **FP16**. FP8 has perf parity with FP16 on Qwen3 and Qwen3.5/3.6 GDN but breaks Llama, Mistral-Small-3.1, DeepSeek-R1-Distill out of the box. Use `--kv-fp8` per-model after testing.

## Notes

- **Prefill variance**: cuBLAS autotuning can cause up to 2.6x variance in prefill numbers between container restarts. Decode is stable — compare decode only for reliable A/B testing.
- **Gemma-4**: CUDA Graphs enabled. Decode is 1.21x llama.cpp on Q4_K_M. Q4_K_M can degenerate on complex code-gen prompts — use Q5_K_M or Q8_0 when output quality matters.
- **GDN models**: Use FP16 prefill weights for numerical stability, reducing prefill throughput ~8% vs FP8.
- **MXFP4 Prefill**: `--mxfp4-prefill` uses CUTLASS block-scaled GEMM; currently ~10% slower than FP8 cuBLASLt due to activation quantization overhead.
- **NVFP4 / MXFP4 KV**: Both store FP4 nibbles at 25% of FP16, differ in scale encoding. NVFP4-KV is decode-parity with FP16 via PTX `cvt.rn.f16x2.e2m1x2` gather. MXFP4-KV is approximately NVFP4 quality at 16K context.
