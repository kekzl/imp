# Benchmarks

All benchmarks on a single **NVIDIA RTX 5090** (32 GB GDDR7, Blackwell sm_120).
Models loaded from GGUF. Each test runs 3 repetitions; averages reported.

- **imp v0.5** — NVFP4 decode cache + FP8 prefill cache, CUDA graphs, PDL, GDN fix
- **llama.cpp** b8445 — flash attention enabled, full GPU offload (`-ngl 99`)

## Decode Throughput (tg256)

Tokens generated per second — the metric that determines how fast a model responds.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **374** | 244 | **+53%** |
| Qwen3-8B | 8.2B | Q8_0 | **251** | 157 | **+60%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **308** | 180 | **+71%** |
| Qwen3.5-9B (GDN) | 9.2B | Q8_0 | **132** | — | — |
| Gemma-3-12B | 11.8B | Q8_0 | **125** | 98 | **+28%** |

## Prefill Throughput (pp512)

Tokens processed per second during the prompt ingestion phase.

| Model | Params | Quant | imp | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **20376** | 21337 | -4% |
| Qwen3-8B | 8.2B | Q8_0 | **17633** | 14172 | **+24%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **15971** | 11149 | **+43%** |
| Qwen3.5-9B (GDN) | 9.2B | Q8_0 | **8386** | — | — |
| Gemma-3-12B | 11.8B | Q8_0 | **7088** | 9269 | -24% |

## Output Quality (Qwen3.5 GDN — fixed in v0.5)

| Prompt | imp v0.4 | imp v0.5 | llama.cpp |
|--------|----------|----------|-----------|
| "The capital of France is" | ❌ "a cultural" | ✅ "Paris" | ✅ "Paris" |
| "What is 2+2?" (chat) | ❌ degenerate | ✅ "4" | ✅ "4" |
| "Write a haiku" (chat) | ❌ garbage | ✅ coherent | ✅ coherent |
| Quantum entanglement (9B) | ❌ newlines | ✅ correct | ✅ correct |

**Root cause**: `post_attn_norm` was applied twice on Qwen3.5 attention layers (8 of 32). Fixed by detecting true sandwich norm (Gemma-3) vs FFN input norm (Qwen3.5).

## KV Cache Quantization (Qwen3-8B Q8_0)

| KV Cache | Decode (tok/s) | Prefill (tok/s) | Quality |
|----------|------:|--------:|---------|
| FP8 E4M3 (default) | **248** | **17950** | Baseline |
| INT4 | 233 | 17693 | Good |
| TurboQuant (PolarQuant + QJL) | 191 | 16006 | = FP8 |
| TurboQuant Lite (QJL only) | 190 | 15417 | Degraded |

## Known Issues

- **Qwen3.5 multi-turn chat**: GDN models may degenerate on multi-turn conversations with chat template (single-turn works correctly). Under investigation — likely related to SSM state management with repeated special tokens.
- **Prefill variance**: cuBLAS autotuning can cause up to 2.6x variance in prefill numbers between container restarts. Decode numbers are stable. Compare decode only for reliable A/B testing.

## Notes

- **Qwen3.5 GDN**: Gated DeltaNet hybrid architecture (24 GDN + 8 attention + 32 FFN layers). Output quality now matches llama.cpp for single-turn prompts.
- **TurboQuant**: PolarQuant INT4 K directions + QJL sketch correction + INT4 V. MXFP4 variant available on sm_120+.
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
