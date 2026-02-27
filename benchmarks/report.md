# imp vs llama.cpp Benchmark Report

**Date:** 2026-02-27 16:53 UTC
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
| Qwen3-4B-Instruct | Q8_0 | llama.cpp | N/A | N/A |
| Qwen3-4B-Instruct | Q8_0 | imp (bench) | 19851.67 | 209.41 |
| Qwen3-4B-Instruct | Q8_0 | imp (real) | 1178.97 | 202.70 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | llama.cpp | N/A | N/A |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (bench) | 952.99 | 216.69 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (real) | 319.66 | 209.97 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | N/A | N/A |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (bench) | 1401.78 | 68.83 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (real) | 454.10 | 62.32 |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- **imp (bench)** uses synthetic tokens with warmup + averaged reps (apples-to-apples with llama-bench)
- **imp (real)** tokenizes a real text prompt (single run, no warmup)
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp features: CUDA graphs (decode), PDL (kernel overlap), MoE decode fast path (device-side expert dispatch), non-blocking stream, 64 MiB cuBLAS workspace
- Build: sm_120 native (RTX 5090), `-O3 -Xptxas -O3 --use_fast_math -march=native`
