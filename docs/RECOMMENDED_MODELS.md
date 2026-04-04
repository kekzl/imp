# Recommended Models for imp

Models tested and verified on RTX 5090 (32 GB GDDR7). Sorted by quality-per-VRAM.

## Dense Transformers (Attention-only)

### Qwen3 — Best general-purpose

| Model | Quant | VRAM | Decode tok/s | Prefill tok/s | Notes |
|-------|-------|------|-------------|---------------|-------|
| **Qwen3-4B** | Q8_0 | 4.0 GB | 375 | 24,055 | Fast, great for coding + tool use |
| **Qwen3-4B** | MXFP4 | 2.8 GB | 243 | 6,424 | Smallest footprint, Blackwell-native |
| **Qwen3-8B** | Q8_0 | 8.2 GB | 255 | 17,746 | Sweet spot quality/speed |
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

### Qwen3.5 — Gated DeltaNet (fastest for long context)

| Model | Quant | VRAM | Decode tok/s | Prefill tok/s | Notes |
|-------|-------|------|-------------|---------------|-------|
| **Qwen3.5-4B** | Q8_0 | 4.2 GB | 308 | 14,687 | GDN hybrid — linear-time long context |
| **Qwen3.5-9B** | Q8_0 | 8.9 GB | 134 | 8,418 | Best GDN quality/speed |
| **Qwen3.5-27B** | Q4_K_M | 16 GB | ~45 | ~3,000 | Frontier GDN, fits 32 GB |
| **Qwen3.5-27B** | Q8_0 | 27 GB | ~35 | ~2,200 | Highest quality, tight fit |

## Mixture of Experts (MoE)

| Model | Quant | VRAM | Notes |
|-------|-------|------|-------|
| **Qwen3.5-35B-A3B** (MoE) | Q6_K | 27 GB | 35B total, 3B active — fast decode |
| **Qwen3-Coder-30B-A3B** | Q6_K | 24 GB | Code-specialized MoE |
| **DeepSeek-R1-Distill-Qwen-14B** | Q6_K | 12 GB | Reasoning-optimized (R1 distillation) |
| **Nemotron-3-Nano-30B-A3B** | Q6_K | 32 GB | Mamba2+Attention+MoE hybrid, tight fit |

## Quick Recommendations

| Use Case | Model | Why |
|----------|-------|-----|
| **Fastest possible** | Qwen3-4B Q8_0 | 375 tok/s decode |
| **Best quality ≤8 GB** | Qwen3-8B Q8_0 | Strong all-round |
| **Best quality ≤16 GB** | Qwen3.5-27B Q4_K_M | GDN + large model |
| **Best quality ≤32 GB** | Qwen3-32B Q4_K_M | Dense frontier |
| **Long context** | Qwen3.5-9B Q8_0 | GDN = O(1) per token |
| **Coding** | Devstral-Small Q4_K_M | Code-specialized |
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

All models available from HuggingFace. imp-cli auto-downloads when given a repo ID:

```bash
# Direct HuggingFace download
./imp-cli --model Qwen/Qwen3-8B-GGUF --prompt "Hello"

# Or download manually
huggingface-cli download Qwen/Qwen3-8B-GGUF qwen3-8b-q8_0.gguf --local-dir ./models/
```
