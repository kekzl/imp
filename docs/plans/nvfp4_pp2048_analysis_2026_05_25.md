# NVFP4 Dense pp2048 Performance Analysis

**Date:** 2026-05-25
**Model:** Qwen3-8B-NVFP4-cortecs (SafeTensors, 36 layers, d_model=4096)
**Gap:** imp 21.6k tok/s vs vLLM 42.4k tok/s at pp2048 (-49%)

## Profile (nsys, pp2048 × 3 reps, no CUDA graphs)

| Kernel | % GPU Time | Total (ms) | Instances | Avg (µs) |
|--------|-----------|-----------|-----------|----------|
| CUTLASS NVFP4 GEMM | 45.7% | 256 | 8,064 | 31.7 |
| causal_softmax_fp32_to_fp16 | 13.6% | 76 | 1,008 | 75.4 |
| cuBLAS FP16 GEMM (Q*K^T, attn*V) | 9.9% | 55.5 | 1,028 | 54.0 |
| rmsnorm | 7.6% | 42 | 4,352 | 9.7 |
| cuBLAS FP16 GEMM (FFN shapes) | 5.3% | 30 | 771 | 38.9 |
| cuBLAS GEMV (LM head) | 4.5% | 25 | 32 | 794 |
| NVFP4 activation quantize | 3.8% | 21 | 8,064 | 2.6 |

## Root Cause

**28.8% of GPU time is unfused attention** (softmax 13.6% + cuBLAS attention GEMMs 15.2%). imp uses:
1. cuBLAS Q*K^T (FP16)
2. Separate causal_softmax kernel (FP32→FP16)
3. cuBLAS attn*V (FP16)

vLLM uses FlashAttention-2 which fuses all three into one kernel with:
- No materialization of the full N×N attention matrix
- O(N) memory instead of O(N²)
- Single kernel launch

At pp2048, the O(N²) attention matrix is 2048×2048 × 32 heads × 4 bytes = 512 MiB per layer. Reading+writing this through DRAM twice (softmax read + attn*V read) is ~1 GiB/layer bandwidth cost that FlashAttention eliminates.

## CUTLASS NVFP4 GEMM Efficiency

- FFN shapes (N=12288, K=4096): median 15.4µs ≈ **100% FP4 roofline**
- Attention shapes (N=4096, K=4096): ~15µs for 17 GFLOP ≈ **34% efficiency**

Small-N GEMMs underutilize the GPU. Fusing QKV (3 separate → 1 fused) would improve attention GEMM efficiency: N=5120 instead of N=4096+512+512.

## Levers

### A: Re-evaluate FMHA for NVFP4 at long context (1-2 days)
Previous FMHA refutation (2026-05-20) was on FP16/FP8 at moderate lengths. At pp2048 with NVFP4, the unfused attention is a larger fraction (28.8% vs ~15% for FP16). The FMHA kernel might break even at shorter lengths for NVFP4 because the GEMM is faster (more time ratio for attention).

### B: Fuse QKV projection (1 day)
The FP8 path already has fused QKV. For CUTLASS NVFP4: quantize activation once, run one GEMM with N=Q_dim+K_dim+V_dim instead of three. Saves 2 activation quantization passes.

### C: cuBLAS attention algo tuning (hours)
The cuBLAS `cutlass_80_tensorop_s16816gemm_f16_64x64_32x6` tile might not be optimal. Force algo search with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
