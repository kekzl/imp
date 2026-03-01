# imp vs llama.cpp Benchmark Report

**Date:** 2026-03-01 20:23 UTC
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
| Phi-4-Mini-Instruct | Q8_0 | llama.cpp | 24051.46 | 245.77 |
| Phi-4-Mini-Instruct | Q8_0 | imp (bench) | 20556.42 | 240.92 |
| Phi-4-Mini-Instruct | Q8_0 | imp (real) | 1444.49 | 241.11 |
| Qwen3-4B-Instruct | Q8_0 | llama.cpp | 19071.95 | 213.75 |
| Qwen3-4B-Instruct | Q8_0 | imp (bench) | 19329.58 | 229.49 |
| Qwen3-4B-Instruct | Q8_0 | imp (real) | 1911.35 | 230.96 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | llama.cpp | 13349.77 | 159.74 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | imp (bench) | 13269.87 | 158.52 |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | imp (real) | 1719.99 | 93.41 |
| Gemma-3-12B-IT | Q8_0 | llama.cpp | 8844.30 | 89.33 |
| Gemma-3-12B-IT | Q8_0 | imp (bench) | 6335.06 | 84.70 |
| Gemma-3-12B-IT | Q8_0 | imp (real) | 864.90 | 83.92 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | llama.cpp | 6054.22 | 100.96 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | imp (bench) | 5675.26 | 88.02 |
| DeepSeek-R1-Distill-Qwen-14B | Q6_K | imp (real) | 1914.72 | 87.19 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | llama.cpp | 5882.70 | 203.41 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (bench) | 4449.81 | 234.62 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (real) | 1754.45 | 229.22 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | 2748.83 | 159.30 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (bench) | 1251.27 | 62.87 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp (real) | 744.94 | 61.24 |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- **imp (bench)** uses synthetic tokens with warmup + averaged reps (apples-to-apples with llama-bench)
- **imp (real)** tokenizes a real text prompt (single run, no warmup)
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp features: CUDA graphs (decode), PDL (kernel overlap), MoE decode fast path (device-side expert dispatch), non-blocking stream, 64 MiB cuBLAS workspace
- Build: sm_120 native (RTX 5090), `-O3 -Xptxas -O3 --use_fast_math -march=native`
