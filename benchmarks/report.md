# imp vs llama.cpp Benchmark Report

**Date:** 2026-02-24 13:55 UTC
**GPU:** NVIDIA GeForce RTX 5090 (32607 MiB)
**Driver:** 591.86 | **CUDA:** 13.1
**OS:** Linux 6.6.87.2-microsoft-standard-WSL2

## Methodology

- **Prompt processing (pp):** llama-bench uses synthetic 512-token prompt; imp uses a real ~500-token text prompt
- **Text generation (tg):** 128 tokens, temperature=0 (greedy)
- **llama-bench:** 3 repetitions, flash attention enabled, all layers on GPU
- **imp:** single run, all layers on GPU, chat template disabled for fair comparison
- Both engines: batch size 1, single sequence

## Results

| Model | Quant | Engine | pp tok/s | tg tok/s |
|-------|-------|--------|----------|----------|
| Qwen3-4B-Instruct | Q8_0 | llama.cpp | 19521.44 | 218.24 |
| Qwen3-4B-Instruct | Q8_0 | imp | 1138.28 | 25.44 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | llama.cpp | 6177.76 | 201.17 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp | 116.40 | 5.10 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | 2011.13 | 179.47 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp | 79.74 | 7.68 |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- llama.cpp uses synthetic prompt tokens; imp tokenizes a real text prompt, so pp token counts may differ slightly
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp uses Blackwell-optimized TCGEN05 attention on sm_120; llama.cpp uses its own Flash Attention
