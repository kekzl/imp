# Recommended Models for imp

Models tested and verified on **RTX 5090** (32 GB GDDR7). The same binary
runs on **RTX PRO 5000 Blackwell** (48 GB) and **RTX PRO 6000 Blackwell**
(96 GB) — workstation cards just fit larger MoE models without expert
offload, no tuning required.

Sorted by quality-per-VRAM. NVFP4 prequant SafeTensors is the primary
path (highest decode tok/s on Blackwell); GGUF supported as legacy.

## Dense Transformers (Attention-only)

### Qwen3 — Best general-purpose

| Model | Quant | VRAM | Decode tok/s | Prefill tok/s | Notes |
|-------|-------|------|-------------|---------------|-------|
| **Qwen3-4B** | Q8_0 | 4.0 GB | 401 | 27,201 | Fast, great for coding + tool use |
| **Qwen3-4B** | MXFP4 | 2.8 GB | 124 | — | Smallest footprint, Blackwell-native |
| **Qwen3-8B** | Q8_0 | 8.2 GB | 255 | 17,636 | Sweet spot quality/speed |
| **Qwen3-8B** | Q4_K_M | 5.0 GB | ~320 | ~12,000 | Good balance |
| **Qwen3-32B** | Q4_K_M | 19 GB | ~85 | ~4,500 | Frontier quality, fits 32 GB |

### Llama 3.2

| Model | Quant | VRAM | Decode tok/s | Notes |
|-------|-------|------|-------------|-------|
| **Llama-3.2-3B** | Q8_0 | 3.2 GB | ~400 | Lightweight, good instruction following |

### Mistral / Devstral

| Model | Quant | VRAM | Notes |
|-------|-------|------|-------|
| **Mistral-Small-3.1-24B** | Q6_K | 19 GB | Strong reasoning, fits 32 GB |
| **Devstral-Small-2507** | Q4_K_M | 14 GB | Code-specialized Mistral |

### Gemma 3

| Model | Quant | VRAM | Decode tok/s | Notes |
|-------|-------|------|-------------|-------|
| **Gemma-3-12B** | Q8_0 | 12 GB | 129 | Multimodal (text + vision via mmproj) |
| **Gemma-3-27B** | Q4_K_M | 16 GB | ~60 | Largest Gemma, quality leader |

## Hybrid Architectures (GDN + Attention)

### Qwen3.5 / 3.6 — Gated DeltaNet (fastest for long context)

| Model | Quant | VRAM | Decode tok/s | Prefill tok/s | Notes |
|-------|-------|------|-------------|---------------|-------|
| **Qwen3.5-4B** | Q8_0 | 4.2 GB | 220 | 13,676 | GDN hybrid — linear-time long context |
| **Qwen3.5-9B** | Q8_0 | 8.9 GB | 140 | 9,483 | Best GDN quality/speed |
| **Qwen3.5-27B** | Q4_K_M | 16 GB | ~45 | ~3,000 | Frontier GDN, fits 32 GB |
| **Qwen3.5-27B** | Q8_0 | 27 GB | ~35 | ~2,200 | Highest quality, tight fit |
| **Qwen3.5-27B** | MXFP4 | — | — | — | ⚠ Loads OOM on 32 GB; see TODO.md |

## Mixture of Experts (MoE)

NVFP4 prequant is the primary path on Blackwell — CUDA Graphs capture
end-to-end via `cache_moe_native_nvfp4` (no per-layer D2H expert-routing
sync), and decode runs at 200+ tok/s on most MoE models post PR #88.

| Model | Quant | VRAM | Decode tok/s | Notes |
|-------|-------|------|-------------|-------|
| **Qwen3-Coder-30B-A3B** | NVFP4 | 16 GB | **272** | NVIDIA Modelopt SafeTensors; CUDA Graphs ON post #88 (was 51 with `--no-cuda-graphs`) |
| **Qwen3-Coder-30B-A3B** | Q6_K | 24 GB | 234 | Code MoE post `moe.expert_overhead_pct=10` auto-pick |
| **Qwen3.6-35B-A3B** | NVFP4 | 22 GB | **217** | GDN+MoE, llm-compressor; native tool calling verified |
| **Qwen3.6-35B-A3B** | Q4_K_M | 22 GB | 143 | GDN + MoE GGUF |
| **Gemma-4-26B-A4B-it** | NVFP4 | 16 GB | **213** | llm-compressor; native tool calling verified on Q8_0/Q4_K_M, NVFP4 falls back to prompt-based |
| **Gemma-4-26B-A4B-it** | Q4_K_M | 14 GB | 183 | 1.21× llama.cpp; CUDA Graphs ON |
| **Gemma-4-26B-A4B-it** | Q5_K_M | 17 GB | 65 | Best quality/speed for complex code-gen |
| **Gemma-4-26B-A4B-it** | Q8_0 | 26 GB | — | Native tool calling, cleanest output |
| **Mistral-Small-3.2** | NVFP4 | — | 101 | llm-compressor; long-prose quality caveat above ~250 tokens (TODO.md) |
| **Qwen3-30B-A3B** | NVFP4-Modelopt | — | — | Mistral-3.2 replacement when long-context coherence matters |
| **DeepSeek-R1-Distill-Qwen-14B** | Q6_K | 12 GB | — | Reasoning-optimised (R1 distillation) |
| **Nemotron-3-Nano-30B-A3B** | Q6_K | 32 GB | — | Mamba2+Attention+MoE hybrid, tight fit on 32 GB |

## Quick Recommendations

| Use Case | Model | Why |
|----------|-------|-----|
| **Fastest possible** | Qwen3-4B Q8_0 | 401 tok/s decode |
| **Best quality ≤8 GB** | Qwen3-8B Q8_0 | Strong all-round |
| **Best quality ≤16 GB** | Qwen3.5-27B Q4_K_M | GDN + large model |
| **Best quality ≤32 GB** | Qwen3-32B Q4_K_M | Dense frontier |
| **Long context** | Qwen3.6-35B-A3B Q4_K_M | GDN+MoE = O(1) per token + sparse compute |
| **Coding (MoE NVFP4)** | Qwen3-Coder-30B-A3B NVFP4 | 272 tok/s post #88, CUDA Graphs |
| **Coding (MoE GGUF)** | Qwen3-Coder-30B-A3B Q6_K | 234 tok/s post `moe.expert_overhead_pct=10` |
| **Coding (dense)** | Devstral-Small Q4_K_M | Code-specialised |
| **Big-MoE NVFP4** | Gemma-4-26B-A4B NVFP4 | 213 tok/s, decode fast-path, native tools |
| **Vision** | Gemma-3-12B Q8_0 | Text + image (mmproj) |
| **Reasoning** | DeepSeek-R1-Distill-14B Q6_K | Chain-of-thought |
| **Smallest footprint** | Qwen3-4B MXFP4 | 2.8 GB, Blackwell FP4 |

## Quantization Format Guide

| Format | Bits/weight | Quality | Decode Speed | Prefill Speed | Notes |
|--------|-------------|---------|-------------|---------------|-------|
| Q8_0 | 8.0 | ★★★★★ | ★★★★ | ★★★★★ | Best quality, baseline |
| Q6_K | 6.5 | ★★★★½ | ★★★★½ | ★★★★ | Great balance |
| Q4_K_M | 4.5 | ★★★★ | ★★★★★ | ★★★½ | Most VRAM-efficient |
| MXFP4 | 4.5 | ★★★½ | ★★★ | ★★★★★ | Blackwell Tensor Cores |
| NVFP4 | 4.0 | ★★★½ | ★★★ | ★★★★★ | Via Model Optimizer |

## Download

All models available from HuggingFace. imp supports both GGUF files and SafeTensors directories:

```bash
# GGUF (direct file or HuggingFace download)
./imp-cli --model Qwen/Qwen3-8B-GGUF --prompt "Hello"
./imp-cli --model models/Qwen3-8B-Q8_0.gguf --prompt "Hello"

# SafeTensors (NVFP4 prequant from Model Optimizer)
./imp-cli --model models/Qwen3-Coder-30B-A3B-FP4/ --prompt "Hello"

# Or download manually
huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q8_0.gguf --local-dir ./models/
```
