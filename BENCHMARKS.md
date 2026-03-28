# Benchmarks

All benchmarks on a single **NVIDIA RTX 5090** (32 GB GDDR7, Blackwell sm_120).
Models loaded from GGUF. Each test runs 3 repetitions; averages reported.

- **imp v0.5.1** — NVFP4 decode cache + FP16 prefill (GDN) / FP8 prefill (non-GDN), CUDA graphs, PDL, multi-turn GDN fix
- **llama.cpp** b8445 — flash attention enabled, full GPU offload (`-ngl 99`)

## Decode Throughput (tg256)

Tokens generated per second — the metric that determines how fast a model responds.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **375** | 244 | **+54%** |
| Qwen3-8B | 8.2B | Q8_0 | **255** | 157 | **+62%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **308** | 180 | **+71%** |
| Qwen3.5-9B (GDN) | 9.2B | Q8_0 | **134** | — | — |
| Gemma-3-12B | 11.8B | Q8_0 | **129** | 98 | **+32%** |

## Prefill Throughput (pp512)

Tokens processed per second during the prompt ingestion phase.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **24055** | 21337 | **+13%** |
| Qwen3-8B | 8.2B | Q8_0 | **17746** | 14172 | **+25%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **14687** | 11149 | **+32%** |
| Qwen3.5-9B (GDN) | 9.2B | Q8_0 | **8418** | — | — |
| Gemma-3-12B | 11.8B | Q8_0 | **6998** | 9269 | -25% |

**Note**: GDN models now use FP16 prefill weights (v0.5.1) instead of FP8 for numerical stability. This reduces prefill throughput by ~8% vs v0.5 FP8 numbers but fixes multi-turn chat degeneration.

## Multi-Turn Chat Quality (GDN — fixed in v0.5.1)

| Scenario | imp v0.5 | imp v0.5.1 | llama.cpp |
|----------|----------|------------|-----------|
| Single-turn | ✅ correct | ✅ correct | ✅ correct |
| 2-turn chat | ❌ degenerate | ✅ correct | ✅ correct |
| 5-turn chat (4B) | ❌ garbage | ✅ correct | ✅ correct |
| 7-turn chat (9B) | ❌ garbage | ✅ correct | ✅ correct |

**Root cause**: FP8 E4M3 weights (3-bit mantissa) introduce precision errors that accumulate through the GDN delta rule scan. Chat template special tokens (`<|im_start|>`, `<|im_end|>`) amplify the divergence because their embedding projections are more sensitive to quantization noise. After ~20-50 special tokens (3-7 turns), the recurrent state becomes numerically unstable.

**Fix**: GDN models now automatically use FP16 weight cache for prefill. Additionally, chunked prefill state management was fixed to preserve recurrent state across chunk boundaries, and prefix caching is disabled for recurrent models.

## KV Cache Quantization (Qwen3-8B Q8_0)

| KV Cache | Decode (tok/s) | Prefill (tok/s) | Quality |
|----------|------:|--------:|---------|
| FP8 E4M3 (default) | **248** | **17950** | Baseline |
| INT4 | 233 | 17693 | Good |
| TurboQuant (PolarQuant + QJL) | 191 | 16006 | = FP8 |
| TurboQuant Lite (QJL only) | 190 | 15417 | Degraded |

## Notes

- **Qwen3.5 GDN**: Gated DeltaNet hybrid architecture (24 GDN + 8 attention + 32 FFN layers). Output quality matches llama.cpp for both single-turn and multi-turn.
- **TurboQuant**: PolarQuant INT4 K directions + QJL sketch correction + INT4 V. MXFP4 variant available on sm_120+.
- **Prefill variance**: cuBLAS autotuning can cause up to 2.6x variance in prefill numbers between container restarts. Decode numbers are stable. Compare decode only for reliable A/B testing.
- **MXFP4 Prefill**: CUTLASS block-scaled GEMM for prefill (`--mxfp4-prefill`). Currently ~10% slower than FP8 cuBLASLt for Q8_0 models due to activation quantization overhead.

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
