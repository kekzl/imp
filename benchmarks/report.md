# imp vs llama.cpp Benchmark Report

**Date:** 2026-02-28 19:12 UTC
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
| Phi-4-Mini-Instruct | Q8_0 | llama.cpp | 26629.93 | 250.96 |
| Phi-4-Mini-Instruct | Q8_0 | imp (bench) | 20610.21 | 231.26 |
| Phi-4-Mini-Instruct | Q8_0 | imp (real) | 1225.22 | 234.83 |
| Qwen3-4B-Instruct | Q8_0 | llama.cpp | 20667.27 | 217.54 |
| Qwen3-4B-Instruct | Q8_0 | imp (bench) | 19048.58 | 221.89 |
| Qwen3-4B-Instruct | Q8_0 | imp (real) | 1325.45 | 227.18 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | llama.cpp | 15368.91 | 164.30 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | imp (bench) | 13379.97 | 147.59 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | imp (real) | 973.70 | 149.65 |
| Gemma-3-12B-IT | Q8_0 | llama.cpp | 9033.08 | 91.41 |
| Gemma-3-12B-IT | Q8_0 | imp (bench) | 6485.03 | 83.01 |
| Gemma-3-12B-IT | Q8_0 | imp (real) | 606.88 | 82.44 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | llama.cpp | 6242.44 | 102.22 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | imp (bench) | 5738.41 | 86.81 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | imp (real) | 1209.82 | 86.15 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | llama.cpp | 6113.22 | 206.61 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (bench) | 1387.13 | 221.74 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (real) | 856.02 | 217.76 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | 226.67 | 25.77 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (bench) | 1250.85 | 58.74 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (real) | 629.77 | 57.13 |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- **imp (bench)** uses synthetic tokens with warmup + averaged reps (apples-to-apples with llama-bench)
- **imp (real)** tokenizes a real text prompt (single run, no warmup)
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp features: CUDA graphs (decode), PDL (kernel overlap), MoE decode fast path (device-side expert dispatch), non-blocking stream, 64 MiB cuBLAS workspace
- Build: sm_120 native (RTX 5090), `-O3 -Xptxas -O3 --use_fast_math -march=native`
