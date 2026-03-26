# Benchmarks

All benchmarks on a single **NVIDIA RTX 5090** (32 GB GDDR7, Blackwell sm_120).
Models loaded from GGUF. Each test runs 5 repetitions; averages reported.

- **imp v0.4** — NVFP4 decode cache + FP8 prefill cache, CUDA graphs, PDL
- **llama.cpp** b8445 — flash attention enabled, full GPU offload (`-ngl 99`)

## Decode Throughput (tg128)

Tokens generated per second — the metric that determines how fast a model responds.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **390** | 244 | **+60%** |
| Qwen3-8B | 8.2B | Q8_0 | **264** | 157 | **+68%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **327** | 180 | **+82%** |
| Gemma-3-12B | 11.8B | Q8_0 | **139** | 98 | **+42%** |
| Qwen3-Coder-30B MoE | 30.5B | Q6_K | **265** | 251 | **+6%** |

## Prefill Throughput (pp512)

Tokens processed per second during the prompt ingestion phase.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **25801** | 21337 | **+21%** |
| Qwen3-8B | 8.2B | Q8_0 | **15819** | 14172 | **+12%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **16017** | 11149 | **+44%** |
| Gemma-3-12B | 11.8B | Q8_0 | **8479** | 9269 | -9% |
| Qwen3-Coder-30B MoE | 30.5B | Q6_K | 5645 | **6090** | -7% |

## KV Cache Quantization (Qwen3-8B Q8_0)

| KV Cache | Decode (tok/s) | Prefill (tok/s) | Quality |
|----------|------:|--------:|---------|
| FP8 E4M3 (default) | **248** | **17950** | Baseline |
| INT4 | 233 | 17693 | Good |
| TurboQuant (PolarQuant + QJL) | 191 | 16006 | = FP8 |
| TurboQuant Lite (QJL only) | 190 | 15417 | Degraded |

## Notes

- **Prefill variance**: cuBLAS autotuning can cause up to 2.6x variance in prefill numbers between container restarts (GPU temperature unrelated — 25°C at idle). Decode numbers are stable. Compare decode only for reliable A/B testing.
- **Qwen3.5 GDN**: Gated DeltaNet hybrid architecture (24 GDN + 8 attention layers). Benchmark throughput is correct; output quality has a known divergence from llama.cpp under investigation.
- **TurboQuant**: PolarQuant INT4 K directions + QJL sketch correction + INT4 V. MXFP4 variant (FP4 E2M1 + UE8M0 micro-scales) available for K directions on sm_120+.
- **MXFP4 Prefill**: CUTLASS block-scaled GEMM for prefill (`--mxfp4-prefill`). Currently ~10% slower than FP8 cuBLASLt for Q8_0 models due to activation quantization overhead. Native MXFP4 GGUF format planned to eliminate this.

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
