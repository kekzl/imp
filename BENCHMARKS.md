# Benchmarks

All benchmarks on a single **NVIDIA RTX 5090** (32 GB GDDR7, Blackwell sm_120).
Models loaded from GGUF or SafeTensors (NVFP4 prequant). Each test runs 3 repetitions; averages reported.

- **imp v0.7** — NVFP4 decode + FP8 prefill, CUDA 13.2.1, CUTLASS v4.4.2, SafeTensors + NVFP4 prequant, FP8 FMHA long-context path
- **llama.cpp** b8445+ — flash attention enabled, full GPU offload (`-ngl 99`)

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
| Gemma-4-26B-A4B-it | 26B (4B active) | Q4_K_M | **183** | 151 | **+21%** |
| Gemma-4-26B-A4B-it | 26B (4B active) | Q5_K_M | **55** | — | — |

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
| Gemma-4-26B-A4B-it | 26B (4B active) | Q4_K_M | **1650** | 196 | **+742%** |

**Gemma-4 notes**: CUDA Graphs are now enabled (PRs #11–#14 unified `forward_decode_async`, PR #20 rope_freqs fix, 2026-04-20 SWA long-context fix). Decode is now **1.21× llama.cpp** on Q4_K_M. The previous gap was two separate bugs: pipeline kernel split-K only issued one 16-byte `cp.async` per load (missing half the data at head_dim=512 on global layers) and cuBLAS dispatch gate forced global layers through a broken FMHA fallback above n=1024. Prefill remains dominated by CUTLASS grouped-GEMM advantage vs llama.cpp's serial expert processing. Q5_K_M recommended when output quality matters on complex prompts — Q4_K_M can degenerate on code-gen (see `docs/BENCHMARKS.md` footnote).

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

## Long-Context Prefill (v0.7)

Before v0.7 the FP8 FMHA prefill path at `n > 1024` emitted NaN on every
attention layer due to a shared-memory S_tile overlap (fixed in PR #33). The
cliff was invisible to all `pp512` / `pp1024` benches because those lengths
dispatch to cuBLAS attention. Post-fix numbers below verify the path is not
only correct but also competitive with llama.cpp across the 2K–8K range.

All measurements: RTX 5090, greedy, 2-rep average, tokens/sec.

| Model | pp512 | pp1024 | pp2048 | pp4096 | pp8192 |
|---|---:|---:|---:|---:|---:|
| **Qwen3-4B Q8_0** — imp v0.7 | 22 984 | 27 115 | 18 880 | 13 568 | 13 566 |
| llama.cpp | 15 786 | 12 437 | 13 083 | 11 009 | 7 978 |
| _speedup_ | ×1.46 | ×2.18 | ×1.44 | ×1.23 | **×1.70** |
| **Qwen3-8B Q8_0** — imp v0.7 | 13 849 | 17 428 | 13 999 | 11 105 | 11 050 |
| llama.cpp | 11 349 | 11 172 | 10 079 | 8 755 | 6 749 |
| _speedup_ | ×1.22 | ×1.56 | ×1.39 | ×1.27 | **×1.64** |
| **Qwen3-32B Q4_K_M** — imp v0.7 | 1 932 | 2 316 | 2 301 | 2 040 | 2 040 |
| llama.cpp | 3 094 | 2 929 | 2 684 | 2 302 | 1 802 |
| _speedup_ | ×0.62 | ×0.79 | ×0.86 | ×0.89 | **×1.13** |
| **Mistral-24B Q6_K** — imp v0.7 | 2 092 | 2 906 | 3 312 | 3 591 | 3 595 |
| llama.cpp | 3 914 | 3 855 | 3 683 | 3 469 | 3 058 |
| _speedup_ | ×0.53 | ×0.75 | ×0.90 | ×1.04 | **×1.18** |
| **Qwen3.5-4B GDN Q8_0** — imp v0.7 | 13 494 | 14 778 | 13 487 | 13 016 | 13 090 |

**Observations:**

- **pp=8192 is imp's strongest point** — ×1.13 to ×1.70 faster than llama.cpp on
  every model tested. Pre-v0.7 this range was garbage.
- **Qwen3-4B/8B show a 1024→2048 throughput dip** (27 k → 19 k tok/s on 4B) because
  the dispatcher switches from cuBLAS attention to FP8 FMHA at n=1024. Output
  remains correct; smoothing the cliff is future work (raise the cuBLAS cap or
  tune the FP8-FMHA kernel).
- **Qwen3-32B Q4_K_M is weight-bound** — throughput is flat across lengths
  because the dense GEMMs dominate over attention cost.
- **GDN (Qwen3.5-4B) is flat by design** — O(n) prefill, not O(n²); only 8 of 32
  layers are attention, so the FMHA fix barely shows up in these numbers but is
  still required for correct output.
- **pp=512 on large dense models** (Qwen3-32B, Mistral-24B) is ~0.5–0.6× llama.cpp —
  a known cuBLAS autotuning / launch-overhead issue unrelated to this release.

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
- **Gemma-4 output quality**: Q4_K_M can degenerate on complex code-gen prompts (Fibonacci → backtick loop). Root cause is accumulated FP16 drift over 30 layers, not a single-layer bug. Q5_K_M and Q8_0 produce clean output — use those when quality matters. Long context up to ~11800 tokens supported with `--min-kv-tokens 14000` (from 2026-04-20 KV-budget fix).

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
