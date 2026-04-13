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
| CUDA graphs for MoE | ⛔ Disabled | D2H routing memcpy incompatible |
| Packed MoE NVFP4 dispatch | 🔲 TODO | Would enable fused gate+up MoE GEMV |
| Tested: Qwen3-Coder-30B-A3B | ✅ 38 tok/s | 128 experts, single+multi-turn verified |

## Open Items

| Item | Impact | Feasibility |
|------|--------|-------------|
| Packed MoE NVFP4 (fused dispatch) | High for MoE prequant | Medium — pack per-expert into NvFP4MoEQuantResult |
| FP8 TC-GEMV for batch decode (M=2-16) | Medium (batch>1 only) | Medium |
| TMA for contiguous KV (non-paged) | Small (~5%) | Medium |
| Example 93 cluster decode pattern | High for long ctx | Very High effort, sm100a only |
