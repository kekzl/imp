# imp vs llama.cpp Benchmark Report

**Date:** 2026-02-26 23:03 UTC
**GPU:** NVIDIA GeForce RTX 5090 (32607 MiB)
**Driver:** 591.86 | **CUDA:** 13.1
**Build:** sm_120 native, -O3 -Xptxas -O3 --use_fast_math, CUDA graphs ON, PDL ON
**OS:** Linux 6.6.87.2-microsoft-standard-WSL2
**imp:** commit 33cd6a1 (fused KV batched GEMM, cuBLAS attention, Q6_K GEMV)
**llama.cpp:** build c830f99

## Methodology

- **Prompt processing (pp):** synthetic 512-token prompt (both imp bench and llama-bench)
- **Text generation (tg):** 128 tokens, temperature=0 (greedy)
- **Repetitions:** 3 reps averaged
- **llama-bench:** flash attention enabled (`-fa 1`), all layers on GPU (`-ngl 99`)
- **imp:** CUDA graphs + PDL enabled (default), non-blocking stream, all layers on GPU
- Both engines: batch size 1, single sequence

## Results

| Model | Quant | Engine | pp tok/s | tg tok/s |
|-------|-------|--------|----------|----------|
| Qwen3-4B | Q8_0 | llama.cpp | 20,323.87 | 212.09 |
| Qwen3-4B | Q8_0 | imp | 18,487.03 | 203.41 |
| Qwen3-30B-A3B (MoE) | Q6_K | llama.cpp | 6,070.10 | 197.13 |
| Qwen3-30B-A3B (MoE) | Q6_K | imp | 5,806.77 | 208.64 |

### Analysis

**Qwen3-4B Q8_0 (dense):**
- Prefill: imp is **9.0% behind** llama.cpp (18,487 vs 20,324 tok/s)
- Decode: imp is **4.1% behind** (203 vs 212 tok/s)

**Qwen3-30B-A3B Q6_K (MoE, 128 experts, top-8):**
- Prefill: imp is **4.3% behind** llama.cpp (5,807 vs 6,070 tok/s)
- Decode: imp is **5.8% ahead** (209 vs 197 tok/s)

### Historical (previous run, pre-fused-KV)

| Model | Quant | Engine | pp tok/s | tg tok/s |
|-------|-------|--------|----------|----------|
| Qwen3-4B | Q8_0 | llama.cpp | 19,280 | 214 |
| Qwen3-4B | Q8_0 | imp | 18,687 | 205 |
| Qwen3-30B-A3B (MoE) | Q6_K | llama.cpp | 6,091 | 200 |
| Qwen3-30B-A3B (MoE) | Q6_K | imp | 6,167 | 210 |
| Nemotron-3-Nano-30B-A3B | Q6_K | llama.cpp | 123 | 18 |
| Nemotron-3-Nano-30B-A3B | Q6_K | imp | 227 | 22 |

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp features: CUDA graphs (decode), PDL (kernel overlap), fused KV batched GEMM (prefill), cuBLAS batched attention (prefill), dp4a GEMV (decode), non-blocking stream, 64 MiB cuBLAS workspace
- Build: sm_120 native (RTX 5090), `-O3 -Xptxas -O3 --use_fast_math -march=native`
