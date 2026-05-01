# SM120 Optimization Status

## Hardware: RTX 5090 (GB202, Blackwell Consumer)

SM120 uses `mma.sync` (register-based), NOT tcgen05/TMEM/WGMMA.
Native MMA shapes: `m16n8k16` (FP16), `m16n8k32` (FP8), `m16n8k64` (FP4 block-scaled).

## Completed Optimizations

| Item | Status | Measured Impact |
|------|--------|-----------------|
| FP8 Attention Prefill (m16n8k32 QK^T) | ✅ Done | +3.3% pp4096 |
| L2 Cache Persistence for KV Cache | ✅ Done | Measurable at >4K ctx |
| Split-K Flash Decode (aggressive) | ✅ Done | No regression at tg256 |
| 16-byte cp.async for HD=256 | ✅ Done | Halves instruction count |
| Fused 3-way bias + sandwich-norm kernels | ✅ Done | -3 launches/layer |
| HD=256 FMHA (Bq=32) | ✅ Done | Enables sm120 FMHA for Qwen3.5 |

## Why Some Items Don't Help at Batch=1 Decode

**Decode at batch=1 is 100% memory-bound** on RTX 5090:
- KV cache load: 2048 tokens × 8 heads × 128 dim × 2 bytes = 4 MB per layer
- At 1.8 TB/s bandwidth: 2.2 µs per layer for KV load
- Score compute (32 FLOPs/token × 2048): 65K FLOPs → 0.08 µs at 838 TFLOPS
- **Memory:Compute ratio = 28:1** → compute is free

This means:
- **FP8 decode scores**: No impact (compute is already idle)
- **Warp specialization**: Marginal (all warps already stream KV at full bandwidth)
- **TMA for paged KV**: Minimal gain (cp.async already achieves near-peak bandwidth)
- **Triple buffering**: Already double-buffered, marginal improvement

## What Would Actually Help Decode

1. **Smaller KV cache** (INT4, TurboQuant) — already implemented, ~2x bandwidth reduction
2. **Speculative decoding** (TurboDraft) — amortizes weight loads over N tokens
3. **GPU clock boost** — RTX 5090 boosts correctly to ~2445 MHz under load (max 3090 MHz)
4. **Batch>1 decode** — multiple sequences share weight loads, increases arithmetic intensity

## NVFP4 Prequant (Model Optimizer) — NEW

| Item | Status | Notes |
|------|--------|-------|
| SafeTensors NVFP4 loading | ✅ Done | Phase 0 direct registration, no re-quantization |
| BF16→FP16 weight conversion | ✅ Done | Norms, router, embeddings, LM head |
| CUTLASS NVFP4 prefill GEMM | ✅ Done | Dense layers via gemm_dispatch() |
| Per-expert NVFP4 GEMV (decode) | ✅ Done | Serial dispatch, legacy MoE path |
| CUDA graphs for non-fast-path MoE | ⛔ Disabled | D2H routing memcpy incompatible |
| CUDA graphs for NVFP4-prequant MoE fast-path | ✅ Done (PR #85) | `cache_moe_native_nvfp4` builds contiguous expert buffer device-side |
| Packed MoE NVFP4 dispatch | ✅ Done (PR #85) | Contiguous `[ne, N, K_packed]` buffer per layer per projection |
| Tested: Qwen3-Coder-30B-A3B (NVFP4) | ✅ 51 tok/s | `--no-cuda-graphs` for coherence |
| Tested: Qwen3-Coder-30B-A3B (Q6_K) | ✅ 234 tok/s | post moe_expert_offload_fix (PR #54) |
| Tested: Qwen3.6-35B-A3B (NVFP4) | ✅ 117–142 tok/s | post #85 fast-path (was 8.34) |
| Tested: Gemma-4-26B-A4B (NVFP4) | ✅ 157–180 tok/s | post #85 fast-path (was ~42) |

## Open Items

| Item | Impact | Feasibility |
|------|--------|-------------|
| Generalise NVFP4-prequant fast-path to GGUF MoE decode | High — removes D2H sync per layer per token | Medium-High — needs device-side expert routing for GGUF MoE |
| Project B Stage 5 (`mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`) | 2-4× MXFP4 prefill attention | Medium — layouts decoded byte-exact (PR #55), integration is the open item |
| FP8 TC-GEMV for batch decode (M=2-16) | Medium (batch>1 only) | Medium |
| TMA for contiguous KV (non-paged) | Small (~5%) | Medium |
| `cublasLtMatmulGrouped` with NVFP4 + device-side shapes (CUDA 13.2 U1) | High for general MoE — host-sync-free expert dispatch | Medium |
| `cub::DeviceTopK` (AIR algorithm) | Medium — 5× faster top_k>128 | Low |
| Per-arch FP8 KV stride fix | High — remove `engine.cpp:547` Gemma-4 carve-out + unblock Llama / Mistral / DeepSeek | High — needs per-layer head_dim awareness in KV write/read kernels |
