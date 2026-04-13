# Benchmarks

All benchmarks on a single **NVIDIA RTX 5090** (32 GB GDDR7, Blackwell sm_120).
Models loaded from GGUF or SafeTensors (NVFP4 prequant). Each test runs 3 repetitions; averages reported.

- **imp v0.6** — NVFP4 decode + FP8 prefill, CUDA 13.2, CUTLASS v4.4.2, SafeTensors + NVFP4 prequant support
- **llama.cpp** b8445 — flash attention enabled, full GPU offload (`-ngl 99`)

## Decode Throughput (tg256)

Tokens generated per second — the metric that determines how fast a model responds.

| Model | Params | Quant | imp v0.6 | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **377** | 244 | **+55%** |
| Qwen3-8B | 8.2B | Q8_0 | **255** | 157 | **+62%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **306** | 180 | **+70%** |
| Qwen3.5-9B (GDN) | 9.2B | Q8_0 | **134** | — | — |
| Llama-3.2-3B | 3.2B | Q8_0 | **208** | — | — |
| Qwen3-Coder-30B-A3B | 30B (3B active) | NVFP4 | **38** | — | — |

## Prefill Throughput (pp512)

Tokens processed per second during the prompt ingestion phase.

| Model | Params | Quant | imp v0.6 | llama.cpp | Delta |
|-------|-------:|-------|----:|----------:|------:|
| Qwen3-4B | 4.0B | Q8_0 | **27201** | 21337 | **+27%** |
| Qwen3-8B | 8.2B | Q8_0 | **17636** | 14172 | **+24%** |
| Qwen3.5-4B (GDN) | 4.0B | Q8_0 | **14823** | 11149 | **+33%** |
| Qwen3.5-9B (GDN) | 9.2B | Q8_0 | **8520** | — | — |
| Llama-3.2-3B | 3.2B | Q8_0 | **22544** | — | — |
| Qwen3-Coder-30B-A3B | 30B (3B active) | NVFP4 | **90** | — | — |

**Note**: GDN models now use FP16 prefill weights (v0.5.1) instead of FP8 for numerical stability. This reduces prefill throughput by ~8% vs v0.5 FP8 numbers but fixes multi-turn chat degeneration.

## Multi-Turn Chat Quality (GDN — fixed in v0.5.1)

| Scenario | imp v0.5 | imp v0.5.1 | llama.cpp |
|----------|----------|------------|-----------|
| Single-turn | ✅ correct | ✅ correct | ✅ correct |
| 2-turn chat | ❌ degenerate | ✅ correct | ✅ correct |
| 5-turn chat (4B) | ❌ garbage | ✅ correct | ✅ correct |
| 7-turn chat (9B) | ❌ garbage | ✅ correct | ✅ correct |

**v0.5.1 root cause**: FP8 weight precision + chunked prefill state management.

**v0.6 root cause (Qwen3.5 "broken" output)**: The Jinja2 engine lacked `{% macro %}` support. Qwen3.5's chat template uses a `render_content` macro for multimodal content handling — without macro support, user content rendered as `"None"`, causing the model to ignore prompts. Fixed in v0.6 with full Jinja2 macro support.

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
- **Qwen3-Coder-30B-A3B**: NVIDIA Model Optimizer NVFP4 prequant (128 experts, 8 active). Loaded from SafeTensors. Decode uses per-expert NVFP4 GEMV (serial dispatch); prefill uses CUTLASS NVFP4 GEMM for dense + per-expert NVFP4 GEMV for MoE. Multi-turn chat verified working.

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
