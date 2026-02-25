# imp vs llama.cpp Benchmark Report

**Date:** 2026-02-25 10:57 UTC
**GPU:** NVIDIA GeForce RTX 5090 (32607 MiB)
**Driver:** 591.86 | **CUDA:** 13.1
**Build:** sm_120 native, -O3 -Xptxas -O3 --use_fast_math, CUDA graphs ON, PDL ON
**OS:** Linux 6.6.87.2-microsoft-standard-WSL2

## Methodology

- **Prompt processing (pp):** synthetic 512-token prompt (both imp bench and llama-bench); imp (real) uses a ~500-token text prompt
- **Text generation (tg):** 128 tokens, temperature=0 (greedy)
- **Repetitions:** 3 reps averaged (imp bench + llama-bench); imp (real) is a single run
- **llama-bench:** flash attention enabled, all layers on GPU
- **imp:** CUDA graphs + PDL enabled (default), non-blocking stream, all layers on GPU
- Both engines: batch size 1, single sequence

## Results

| Model | Quant | Engine | pp tok/s | tg tok/s |
|-------|-------|--------|----------|----------|
| Qwen3-4B-Instruct | Q8_0 | llama.cpp | 19507.54 | 210.29 |
| Qwen3-4B-Instruct | Q8_0 | imp (bench) | 3066.83 | 209.91 |
| Qwen3-4B-Instruct | Q8_0 | imp (real) | 1118.12 | 206.16 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | llama.cpp | 6276.84 | 197.74 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (bench) | 849.96 | 214.29 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (real) | 449.56 | 205.39 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | 241.98 | 27.12 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (bench) | 250.94 | 24.23 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (real) | 81.04 | 7.53 |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- **imp (bench)** uses synthetic tokens with warmup + averaged reps (apples-to-apples with llama-bench)
- **imp (real)** tokenizes a real text prompt (single run, no warmup)
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp features: CUDA graphs (decode), PDL (kernel overlap), MoE decode fast path (device-side expert dispatch), non-blocking stream, 64 MiB cuBLAS workspace
- Build: sm_120 native (RTX 5090), `-O3 -Xptxas -O3 --use_fast_math -march=native`
