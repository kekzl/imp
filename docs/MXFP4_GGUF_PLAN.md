# MXFP4 GGUF Weight Format — Implementation Plan

## Motivation

MXFP4 weights are **Tensor-Core-native** on Blackwell (sm_120). Unlike Q4_K_M which requires runtime dequantization to FP16 before cuBLAS GEMM, MXFP4 data feeds directly into `wgmma.mma_async` via CUTLASS block-scaled ops — zero dequant overhead.

Current state: imp already has a complete MXFP4 CUTLASS GEMM path (`--mxfp4-prefill`), but it only works for Q8_0 models where NVFP4 cache exists as source. For Q4_K_M (most common), there's no conversion path. Native MXFP4 GGUF eliminates this gap.

## Format Specification

### Block Layout
```
MXFP4 Block (17 bytes):
├── 32 × FP4 E2M1 elements (16 bytes, 2 nibbles per byte)
└── 1 × UE8M0 scale (1 byte, pure exponent: value = 2^(bits-127))

Effective: 4.25 bits/weight
```

### GGUF Type
```c
GGML_TYPE_MXFP4 = <new_id>   // 32 elements per block, 17 bytes
// Block size 32 matches Blackwell SFVecSize=32 for hardware MMA
```

### Memory Layout (GPU-Ready)
Packed data must be in CUTLASS SfAtom layout for direct consumption:
- Data: `[N, K/2]` packed FP4 nibbles (row-major, 2 elements per byte)
- Scales: SfAtom format `[ceil(N/128), ceil(K/128), 512]` — matches hardware tile

### VRAM Comparison (Qwen3-32B, 32.5B params)
| Format | Size | bits/weight | Tensor Core Native |
|--------|------|-------------|-------------------|
| FP16 | 65 GB | 16 | Yes (but too large) |
| Q8_0 | 34 GB | 8.5 | No (dequant to FP16) |
| Q4_K_M | 18.5 GB | ~4.5 | No (dequant to FP16) |
| **MXFP4** | **17.3 GB** | **4.25** | **Yes** |

## Converter Pipeline

### Offline: `tools/convert_mxfp4.py`

```
Input:  HuggingFace BF16/FP16 SafeTensors
Output: MXFP4 GGUF file

Steps per weight matrix [N, K]:
1. Load FP32 from SafeTensors
2. Block-Hadamard rotation (K-dim, block_size=32)
   - Walsh-Hadamard transform per 32-element block
   - Redistributes outliers evenly within each block
   - Rotation is BAKED INTO weights (zero runtime cost)
3. Per-32-element group:
   a. absmax = max(|w[0..31]|)
   b. scale = ceil_pow2(absmax / 6.0)  → UE8M0 encoding
   c. quantize: q[i] = round_nearest(w[i] / scale) → FP4 E2M1
4. Pack: 2 nibbles per byte (lo|hi), scale as raw UE8M0 byte
5. Write GGUF tensor with type=GGML_TYPE_MXFP4
```

### Optional: MR-GPTQ Calibration
For higher quality (closes the last ~1% accuracy gap):
- Use 128 calibration samples (e.g., wikitext-2)
- GPTQ-style column-by-column error minimization
- Adjusts quantized values to minimize layer-wise reconstruction error
- Adds ~10 min to conversion but improves perplexity significantly

### Dependencies
- PyTorch (weight loading + Hadamard transforms)
- `gguf` Python package (GGUF writer)
- No GPU needed for basic conversion (CPU-only)
- GPU optional for MR-GPTQ calibration

## Runtime Integration

### Weight Loading (`src/model/gguf_loader.cpp`)
- Add `GGML_TYPE_MXFP4` to type enum
- `dtype_size()` returns 17 bytes per 32 elements
- mmap data directly — already in GPU-ready packed format

### Weight Upload (`src/model/weight_upload.cu`)
- MXFP4 weights: upload packed data to GPU as-is (no dequant)
- Convert scale bytes from linear layout to SfAtom layout on upload
- Register in `wcache_.cutlass_mxfp4` directly (skip NVFP4 intermediate)

### Forward Pass
- **Prefill (M>1)**: `gemm_dispatch` → `gemm_mxfp4_cutlass_sm120` (already implemented)
  - Only activation quantization needed (FP16 → MXFP4, ~2 μs per GEMM)
- **Decode (M=1)**: New `gemv_mxfp4` kernel needed
  - Each thread dequants FP4 E2M1 × UE8M0 scale → FP32, dot with activation
  - Similar to existing `gemv_nvfp4_kpar` but with UE8M0 instead of UE4M3 scales
  - Alternative: dequant MXFP4 → FP16 at startup, use FP16 GEMV (simpler, more VRAM)

### Activation Quantization
Online FP16 → MXFP4 quantization for activations (already implemented):
- `quantize_fp16_to_mxfp4_cutlass()` in `gemm_cutlass_mxfp4_sm120.cu`
- Per-32 group: absmax → UE8M0 scale → FP4 E2M1 quantize → pack
- Cost: ~2 μs for [512, 4096] on RTX 5090

## Expected Performance

### Prefill (Compute-Bound)
- Q4_K_M path: dequant Q4_K_M → FP16 → cuBLAS FP16 GEMM
- MXFP4 path: quantize activation FP16 → MXFP4 → CUTLASS MXFP4 GEMM
- Expected: **2-4× prefill speedup** (same Tensor Core throughput as FP4, no weight dequant)
- Bottleneck shifts from weight dequant to activation quantization

### Decode (Memory-Bound)
- Both paths are memory-bound (weight loading dominates)
- MXFP4 weights are ~5% smaller than Q4_K_M → ~5% faster decode
- With custom GEMV: potentially faster (UE8M0 dequant simpler than Q4_K_M)

### Quality
- Naive MXFP4: ~2-3% perplexity increase vs FP16
- With block-Hadamard: ~1-2% perplexity increase
- With MR-GPTQ calibration: <1% perplexity increase (matches Q4_K_M quality)

## Implementation Order

1. **GGUF type + loader** (~50 lines) — define format, parse blocks
2. **Basic converter** (~200 lines Python) — Safetensors → block-Hadamard → MXFP4 → GGUF
3. **Weight upload** (~30 lines) — upload + register in cutlass_mxfp4 cache
4. **Test with existing MXFP4 GEMM** — prefill should work immediately
5. **MXFP4 GEMV** (~100 lines CUDA) — decode path
6. **MR-GPTQ calibration** (~150 lines Python) — quality improvement
7. **Benchmark** — A/B test vs Q4_K_M on Qwen3-32B

## References

- MR-GPTQ (ICLR 2026): arxiv:2509.23202
- OAS + MBS: arxiv:2603.08713
- Block Rotation: arxiv:2511.04214
- CUTLASS v4.4.1 block-scaled GEMM examples
- OCP MXFP4 specification (Open Compute Project)
