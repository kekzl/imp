# imp vs llama.cpp Benchmark Report

**Date:** 2026-02-24 14:43 UTC
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
| Qwen3-4B-Instruct | Q8_0 | llama.cpp | 19692.67 | 213.23 |
| Qwen3-4B-Instruct | Q8_0 | imp | 1290.73 | 42.24 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | llama.cpp | 6061.17 | 200.49 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp | 105.28 | 4.80 |
| Qwen3-Coder-30B-A3B-Instruct | Q6_K | imp (GPU experts) | 9.35 | 23.71 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | 129.91 | 18.34 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp | 66.64 | 7.36 |

## GPU-Resident MoE Expert Weights (2026-02-24)

Expert weights uploaded to GPU as raw quantized bytes when VRAM permits (2 GiB reserve).
Eliminates per-expert H2D copy during inference — dequant runs GPU-to-GPU.

| Model | Experts on GPU | tg before | tg after | Speedup |
|-------|---------------|-----------|----------|---------|
| Qwen3-Coder-30B-A3B Q6_K | Yes (22.15 GiB) | 4.80 tok/s | 23.71 tok/s | 4.9x |
| Nemotron-3-Nano-30B-A3B Q6_K | No (29.07 GiB > 28.42 free) | 7.36 tok/s | — | host fallback |
| Qwen3-4B Q8_0 (dense, no MoE) | n/a | 42.24 tok/s | 53.40 tok/s | no regression |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- llama.cpp uses synthetic prompt tokens; imp tokenizes a real text prompt, so pp token counts may differ slightly
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp uses Blackwell-optimized TCGEN05 attention on sm_120; llama.cpp uses its own Flash Attention
