# Benchmarks

All benchmarks on a single **NVIDIA RTX 5090** (32 GB GDDR7, Blackwell sm_120).
Models loaded from GGUF. Each test runs 5 repetitions; averages reported.

- **imp v0.2** — NVFP4 decode cache + FP8 prefill cache, CUDA graphs, PDL
- **llama.cpp** b5285 — flash attention enabled (`-fa 1`), full GPU offload (`-ngl 99`)

## Decode Throughput (tg128)

Tokens generated per second — the metric that determines how fast a model responds.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-1.7B | 1.7B | Q8_0 | **477** | 446 | **+7%** |
| Qwen3-4B | 4.0B | Q8_0 | **393** | 244 | **+61%** |
| Phi-4-mini | 3.8B | Q8_0 | 264 | **277** | -5% |
| DeepSeek-R1-Distill-Qwen-7B | 7.6B | Q8_0 | **283** | 176 | **+61%** |
| Mistral-7B | 7.2B | Q8_0 | **270** | 173 | **+56%** |
| Llama-3.1-8B | 8.0B | Q8_0 | **262** | 165 | **+59%** |
| Qwen3-8B | 8.2B | Q8_0 | **262** | 157 | **+67%** |
| Gemma-3-12B | 11.8B | Q8_0 | **146** | 98 | **+49%** |
| DeepSeek-R1-Distill-Qwen-14B | 14.8B | Q6_K | **126** | 110 | **+15%** |
| Qwen3-Coder-30B MoE | 30.5B | Q6_K | **293** | 251 | **+17%** |
| Mixtral-8x7B MoE | 46.7B | Q4_K_M | 64 | — | *host offload* |
| Nemotron-30B-A3B | 30.5B | Q6_K | 86 | — | *Mamba2 hybrid* |

## Prefill Throughput (pp512)

Tokens processed per second during the prompt ingestion phase.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-1.7B | 1.7B | Q8_0 | **39506** | 38464 | +3% |
| Qwen3-4B | 4.0B | Q8_0 | **27240** | 21337 | **+28%** |
| Phi-4-mini | 3.8B | Q8_0 | 20949 | **27259** | -23% |
| DeepSeek-R1-Distill-Qwen-7B | 7.6B | Q8_0 | **21386** | 15867 | **+35%** |
| Mistral-7B | 7.2B | Q8_0 | **19661** | 15097 | **+30%** |
| Llama-3.1-8B | 8.0B | Q8_0 | **18611** | 15152 | **+23%** |
| Qwen3-8B | 8.2B | Q8_0 | **17486** | 14172 | **+23%** |
| Gemma-3-12B | 11.8B | Q8_0 | **11262** | 9269 | **+22%** |
| DeepSeek-R1-Distill-Qwen-14B | 14.8B | Q6_K | **10264** | 6367 | **+61%** |
| Qwen3-Coder-30B MoE | 30.5B | Q6_K | 5722 | **6090** | -6% |
| Mixtral-8x7B MoE | 46.7B | Q4_K_M | 7390 | — | *host offload* |
| Nemotron-30B-A3B | 30.5B | Q6_K | — | — | *bench mode broken* |

## Notes

- **Phi-4-mini**: imp disables NVFP4 for dense models with d_model < 4096 (not enough weight data to amortize NVFP4 dequant overhead). Falls back to dp4a Q8_0 path. llama.cpp's Q8_0 CUDA kernels are well-tuned for this size.
- **Mixtral-8x7B**: Expert weights are partially offloaded to host memory in imp. llama.cpp cannot load this model on a single 32 GB GPU.
- **Nemotron-30B-A3B**: Mamba2 + Attention + MoE hybrid architecture. Not supported by llama.cpp. imp's benchmark mode is broken for this model due to layer offloading incompatibility with CUDA graphs; real prompt inference works (86 tok/s decode).
- **Prefill variance**: cuBLAS autotuning can cause up to 2.6x variance in prefill numbers between container restarts. Decode numbers are stable and reproducible.
- **llama.cpp version**: Docker image `ghcr.io/ggml-org/llama.cpp:full-cuda`, build 35bee03.

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5090 |
| Architecture | Blackwell (GB202), sm_120 |
| VRAM | 32 GB GDDR7, 512-bit, 1792 GB/s |
| SMs | 170 |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 (5th gen) |
| FP4 Tensor | 3,354 TOPS |
| FP8 Tensor | 1,677 TFLOPS |
| L2 Cache | 96 MB |
| TDP | 575 W |
| Cooling | Custom water loop (no thermal throttling) |
