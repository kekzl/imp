# imp vs llama.cpp Benchmark Report

**Date:** 2026-02-27 21:05 UTC
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
| Phi-4-Mini-Instruct | Q8_0 | llama.cpp | 24164.17 | 243.22 |
| Phi-4-Mini-Instruct | Q8_0 | imp (bench) | 19764.21 | 214.44 |
| Phi-4-Mini-Instruct | Q8_0 | imp (real) | 1279.08 | 210.27 |
| Qwen3-4B-Instruct | Q8_0 | llama.cpp | 18893.43 | 203.20 |
| Qwen3-4B-Instruct | Q8_0 | imp (bench) | 19113.63 | 205.32 |
| Qwen3-4B-Instruct | Q8_0 | imp (real) | 1155.17 | 199.02 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | llama.cpp | 14371.39 | 156.57 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | imp (bench) | 13092.77 | 143.68 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | imp (real) | 1259.47 | 140.86 |
| Gemma-3-12B-IT | Q8_0 | llama.cpp | 8431.06 | 88.90 |
| Gemma-3-12B-IT | Q8_0 | imp (bench) | 144.96 | 0.00 |
| Gemma-3-12B-IT | Q8_0 | imp (real) | 91.21 | 21.70 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | llama.cpp | 5385.68 | 100.02 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | imp (bench) | 103.00 | 0.00 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | imp (real) | 88.83 | 29.92 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | llama.cpp | 6126.56 | 202.45 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (bench) | 956.69 | 212.29 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (real) | 349.11 | 205.40 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | 115.85 | 22.72 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (bench) | 1381.31 | 70.22 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (real) | 544.49 | 0.00 |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- **imp (bench)** uses synthetic tokens with warmup + averaged reps (apples-to-apples with llama-bench)
- **imp (real)** tokenizes a real text prompt (single run, no warmup)
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp features: CUDA graphs (decode), PDL (kernel overlap), MoE decode fast path (device-side expert dispatch), non-blocking stream, 64 MiB cuBLAS workspace
- Build: sm_120 native (RTX 5090), `-O3 -Xptxas -O3 --use_fast_math -march=native`
