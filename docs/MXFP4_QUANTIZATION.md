# MXFP4 Quantization for imp

## Overview

MXFP4 (Microscaling FP4 E2M1) is the optimal quantization format for Blackwell GPUs (RTX 5090).
It uses the FP4 Tensor Cores at 3354 TOPS — 2× FP8 and 4× FP16 throughput.

## Recommended Quantization Tool

**[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)** is the recommended tool
for producing MXFP4/NVFP4 quantized models.

### Why Model Optimizer over manual conversion

| Aspect | Model Optimizer | Naive absmax (deprecated) |
|--------|----------------|--------------------------|
| Calibration | AWQ / SmoothQuant on calibration dataset | Per-group absmax only |
| Quality | <1% perplexity increase | 2-3% perplexity increase |
| Scale optimization | Per-layer adaptive | Fixed per-block |
| MoE support | Expert-aware quantization | No MoE awareness |
| Output format | HuggingFace SafeTensors | Custom GGUF (removed) |

### Quantization workflow

```bash
# 1. Install Model Optimizer
pip install nvidia-modelopt

# 2. Quantize a HuggingFace model to NVFP4
python -m modelopt.llm.ptq \
  --model Qwen/Qwen3-8B \
  --quant nvfp4 \
  --output ./Qwen3-8B-nvfp4/

# 3. Load in imp (SafeTensors)
./imp-cli --model ./Qwen3-8B-nvfp4/ --prompt "Hello"
```

### Supported quantization modes

| Mode | What's quantized | Use case |
|------|-----------------|----------|
| `nvfp4` | All linear layers | Maximum compression |
| `nvfp4_mlp_only` | MLP/FFN layers only | Balanced quality/speed |
| `nvfp4_experts_only` | MoE expert layers only | MoE models |
| `nvfp4_omlp_only` | MLP + output projection | Quality-sensitive |

### Data format

**NVFP4 (from Model Optimizer):**
- Weight data: packed FP4 E2M1 nibbles (2 per byte)
- Micro-scales: FP8 E4M3 per 16 elements
- Tensor scale: single FP32 per tensor

**MXFP4 (CUTLASS native):**
- Weight data: packed FP4 E2M1 nibbles (2 per byte)
- Micro-scales: UE8M0 per 32 elements
- Conversion from NVFP4 → MXFP4 happens at model load time

## imp inference pipeline

```
Model Optimizer SafeTensors
  → SafeTensors loader (FP4 packed + scales)
  → Prefill: CUTLASS MXFP4 GEMM (sm_120 Tensor Cores)
  → Decode:  MXFP4 GEMV (register-cached warp-parallel)
```

## Legacy note

The custom `convert_mxfp4.py` converter and MXFP4 GGUF format have been removed
in favor of Model Optimizer's calibrated quantization. Existing MXFP4 GGUF files
can still be loaded (GGUF type 31), but new models should use Model Optimizer.
