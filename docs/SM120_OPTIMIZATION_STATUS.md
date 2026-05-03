# SM120 Optimization Status

## Hardware: Blackwell GB202 family — RTX 5090 + RTX PRO 5000 / 6000

`sm_120` (consumer + workstation Blackwell) uses register-based `mma.sync`,
NOT data-center `tcgen05` / `TMEM` / `WGMMA`. Native MMA shapes:
`m16n8k16` (FP16), `m16n8k32` (FP8), `m16n8k64` (FP4 block-scaled).

Same `sm_120f` binary on RTX 5090 (32 GB), RTX PRO 5000 Blackwell (48 GB),
RTX PRO 6000 Blackwell (96 GB) — kernels and dispatch identical, only VRAM
ceiling and clock differ.

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

1. **Smaller weights** (NVFP4 prequant) — primary path on Blackwell. ×2 bandwidth reduction vs FP8 plus a contiguous-expert MoE fast-path that keeps CUDA Graphs intact (272 tok/s on Qwen3-Coder-30B-A3B-NVFP4 post PR #88).
2. **Smaller KV cache** (INT4, TurboQuant) — implemented; INT4 currently shows decode regression at long ctx (TODO.md), TurboQuant `~3 bits/elem` saves 60% K-traffic.
3. **GPU clock boost** — RTX 5090 boosts to ~2445 MHz under load (max 3090 MHz). PRO 6000 Blackwell similar; PRO 5000 is power-limited at 300 W.
4. **Batch>1 decode** — multiple sequences share weight loads. Continuous batching is partially implemented; decode kernels target batch=1 today.

(N-gram speculative decoding is implemented as opt-in; EAGLE / self-speculative / DFlash / TurboDraft were all evaluated and dropped — single 5090 decode is bandwidth-bound and none of those variants amortise weight reads at batch=1. See TODO.md "Speculative decoding — abandoned options".)

## NVFP4 Prequant — Primary Path

| Item | Status | Notes |
|------|--------|-------|
| SafeTensors NVFP4 loading (Modelopt + llm-compressor) | ✅ Done | Phase 0 direct registration, no re-quantization |
| BF16→FP16 weight conversion | ✅ Done | Norms, router, embeddings, LM head |
| CUTLASS NVFP4 prefill GEMM | ✅ Done | Dense layers via gemm_dispatch() |
| CUTLASS NVFP4×NVFP4 prefill cache for prequant | ✅ Done (PR #88) | Replaced dequant→cuBLAS fallback; CUDA Graphs now safe by default |
| NVFP4 MoE decode fast-path | ✅ Done (PR #85) | `cache_moe_native_nvfp4` builds contiguous `[ne, N, K_packed]` buffer per layer per projection |
| Packed MoE NVFP4 dispatch | ✅ Done | Active for all NVFP4 prequant MoE models |
| CUDA Graphs (NVFP4 prequant MoE) | ✅ Captures end-to-end | `cache_moe_native_nvfp4` runs entirely device-side (no D2H expert-offsets sync) |
| CUDA Graphs (legacy GGUF MoE) | ⛔ Still incompatible | D2H routing memcpy per layer per token (open work item) |
| Tested: Qwen3-Coder-30B-A3B (NVFP4) | ✅ **272 tok/s** (post #88, was 51 with `--no-graphs`) | Modelopt SafeTensors |
| Tested: Qwen3.6-35B-A3B (NVFP4) | ✅ **217 tok/s** (post #88) | llm-compressor; native tool calls verified |
| Tested: Gemma-4-26B-A4B (NVFP4) | ✅ **213 tok/s** (post #88) | llm-compressor |
| Tested: Mistral-Small-3.2 (NVFP4) | ✅ 101 tok/s (post #88) | Long-prose quality caveat (TODO.md) |
| Tested: Qwen3-Coder-30B-A3B (Q6_K, GGUF) | ✅ 234 tok/s | Post `moe.expert_overhead_pct=10` auto-pick |

## Open Items

| Item | Impact | Feasibility |
|------|--------|-------------|
| Generalise NVFP4-prequant fast-path to GGUF MoE decode | High — removes D2H sync per layer per token | Medium-High — needs device-side expert routing for GGUF MoE |
| `mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` MMA integration | 2-4× MXFP4 prefill attention | Medium — layouts decoded byte-exact (PR #55), integration is the open item |
| FP8 TC-GEMV for batch decode (M=2-16) | Medium (batch>1 only) | Medium |
| TMA for contiguous KV (non-paged) | Small (~5%) | Medium |
| `cublasLtMatmulGrouped` with NVFP4 + device-side shapes (CUDA 13.2 U1) | High for general MoE — host-sync-free expert dispatch | Medium |
| `cub::DeviceTopK` (AIR algorithm) | Medium — 5× faster top_k>128 | Low |
| Per-layer head_dim FP8 KV write/read | High — removes Gemma-4 force-FP16 carve-out, blocks dual-head_dim FP8 KV on hybrid SWA + global layers | High — kernels need per-layer head_dim awareness, allocator side already done (PR #89) |
