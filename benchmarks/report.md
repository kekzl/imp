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

## P2-P9 Optimization Impact (e2cf896 vs a9ae9b4)

**Date:** 2026-02-28
**Prompt:** "Explain the theory of relativity in simple terms." (10-19 tokens)
**Generation:** 128 tokens, temperature=0 (greedy), CUDA graphs ON, single run (warm)

| Model | Quant | Phase | Before | After | Delta |
|-------|-------|-------|--------|-------|-------|
| Phi-4-Mini-Instruct | Q8_0 | pp tok/s | 57.69 | 105.95 | +83.7% |
| Phi-4-Mini-Instruct | Q8_0 | tg tok/s | 189.77 | 189.84 | +0.0% |
| Qwen3-4B-Instruct | Q8_0 | pp tok/s | 130.85 | 165.65 | +26.6% |
| Qwen3-4B-Instruct | Q8_0 | tg tok/s | 186.06 | 183.75 | -1.2% |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | pp tok/s | 107.22 | 110.60 | +3.2% |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | tg tok/s | 133.52 | 133.15 | -0.3% |

**Optimizations applied:**
- P2: Q4_0 dp4a GEMV kernels (no Q4_0 models available to measure)
- P3: Down-projection residual fusion (dequant + beta=1.0 GEMM)
- P4: Gate+up batched GEMM (cublasGemmStridedBatchedEx)
- P5: PDL registration for RMSNorm, RoPE, SwiGLU, GeGLU
- P6: Double-buffered FP16 shared memory in GQA paged attention
- P7: Shortest-first request reordering in scheduler
- P8: Async weight upload on separate stream
- P9: Event-based prefill sync (overlaps host bookkeeping with GPU)

**Analysis:** Prefill shows significant gains from P4 (gate+up batched GEMM) and P3 (fused residual), especially on smaller models where kernel launch overhead is a larger fraction. Decode is flat because CUDA graphs already eliminate launch overhead, and these Q8_0 models already had dp4a GEMV. P7 (scheduler) and P8 (async upload) benefit multi-request and init-time scenarios not captured here.

## Notes

- **pp tok/s** = prompt processing throughput (prefill phase)
- **tg tok/s** = text generation throughput (autoregressive decode phase)
- **imp (bench)** uses synthetic tokens with warmup + averaged reps (apples-to-apples with llama-bench)
- **imp (real)** tokenizes a real text prompt (single run, no warmup)
- Both engines offload all layers to GPU (`-ngl 99` / default)
- imp features: CUDA graphs (decode), PDL (kernel overlap), MoE decode fast path (device-side expert dispatch), non-blocking stream, 64 MiB cuBLAS workspace
- Build: sm_120 native (RTX 5090), `-O3 -Xptxas -O3 --use_fast_math -march=native`
