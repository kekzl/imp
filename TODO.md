# TODO

## Performance

- [x] **Gemma-3-12B Decode Bench** — ~~Bench mode "internal error"~~.
  Fix: `ignore_eos=true` during prefill (synthetic tokens → immediate EOS → FINISHED).
  `imp_decode_step` sets ignore_eos correctly for actual decode.

- [x] **Gemma-3-12B NVFP4 Decode** — head_dim=256 KV cache consumed VRAM, limiting
  NVFP4 budget. GeGLU+GEMV fusion boosted decode: 59→100 tok/s (+70%).
  KV budget trade-off (10% cap for mode 2) + incremental NVFP4 processing
  (net VRAM-negative per FP16→NVFP4 conversion) achieved 334/337 tensor
  coverage. Gemma-3-12B decode: 86→119 tok/s (bench), 96 tok/s (real prompt).

- [x] **CUTLASS FMHA head_dim=96** — Added template `Shape<_128, _96, _96>`.
  Phi4-mini actually has head_dim=128 (rope_dim=96 is partial RoPE, not head_dim).
  CUTLASS FMHA already worked correctly. Template added for future models with head_dim=96.

- [x] **Init Async Quantization** — Eliminated per-tensor `cudaStreamSynchronize` in
  weight cache construction. FP8 overflow, FP16→FP8 migration, and NVFP4 cache now
  use async calibrate+quantize with single sync at end. Bulk cudaMalloc for FP8 data
  and scale buffers. DeepSeek-14B init: 55.4s → ~35s (37% faster).

- [x] **Decode Kernel Fusion** — Fused activation+GEMV+residual kernels:
  - SwiGLU+NVFP4 GEMV+residual: Qwen3-4B 273→298 (+9%), Qwen3-8B 138→184 (+33%)
  - GeGLU+NVFP4 GEMV+residual: Gemma-3-12B 59→100 (+70%)
  - Multi-row dispatch (8 rows/block) for all fused NVFP4 GEMV variants
  - MoE SwiGLU+NVFP4 GEMV fusion for down projection
  - NVFP4 GEMV kernels registered with PDL, launches via pdl::launch()
  Tested: RMSNorm+GEMV fusion regresses (extra norm_w loads + muls in inner
  loop outweigh kernel launch savings). Multi-row threshold >512 also
  regresses (lower occupancy for large-K projections).

- [x] **CUTLASS FMHA Softcap** — Added `SoftcapCausalFusion` for models with
  logit soft-capping (Gemma-2). Uses `__constant__` memory for softcap params,
  applies scale→tanh→undo-scale in `before_softmax` hook. Dispatch relaxed to
  attempt CUTLASS for softcap models (graceful fallback). HD=256 tiles require
  >200 KB smem — exceeds RTX 5090 limit (99 KB), so Gemma-3 still uses WMMA.
  Gemma-3 doesn't actually use softcap (only Gemma-2 does).

- [x] **dp4a Fused Act+GEMV+Residual** — Attempted fusing SwiGLU/GeGLU + Q8_1
  quantize + dp4a GEMV + residual into a single kernel. Two-pass approach
  (pass 1: compute amax, pass 2: recompute + quantize + dp4a) to avoid float[32]
  register pressure. **Result: 22% regression** (Qwen3-4B 150→117 tok/s).
  Root cause: 2x gate/up L2 reads + 2x SwiGLU ALU per inner loop iteration.
  The kpar GEMV is memory-bound on weight reads — doubling input bandwidth
  and ALU per block saturates L2. Same pattern as attention O-proj inline quant
  (deliberately kept as separate quant + kpar for higher occupancy).
  Reverted — separate `swiglu_quantize_q8_1 + gemv_q*_q8_1_residual` is faster.

- [x] **NVFP4 LM Head** — FP32 output NVFP4 GEMV for output projection (LM head).
  Saves ~47% weight reads vs Q8_0 dp4a for vocab_size×d_model decode GEMV.
  Output projection collected first in NVFP4 budget to ensure inclusion.
  Large weights that exceed dequant scratch use temporary buffer during init.
  Multi-row (NR=8) dispatch for K ≤ 8192, kpar (128 threads) for larger K.
  Correctness verified on Qwen3-4B, Qwen3-8B, DeepSeek-7B.

- [x] **KV Cache Budget Trade-off** — In NVFP4 mode 2, KV cache allocation
  capped at 10% of available VRAM (was 80%). Excess KV blocks sit unused while
  weight caching directly improves decode throughput. Also skip 2x KV headroom
  and don't enforce needed_blocks floor. Qwen3-8B real prompt: 110→174 tok/s
  decode by trading 100K→30K KV tokens for full NVFP4 coverage.

- [x] **MoE NVFP4 Auto-Detection** — Removed d_model < 4096 threshold for MoE
  models. Expert weights dominate VRAM and sparse activation limits quantization
  error accumulation. Qwen3-Coder-30B: 247→265 tok/s (+7%), Nemotron-30B: 68→75
  tok/s (+10%).

- [x] **cuBLASLt NVFP4 Probe** — status=7 (INVALID_VALUE). Investigated: cuBLASLt
  handle works (FP16 GEMM passes), CUDA_R_4F_E2M1 data type recognized (enum=33),
  but no FP4 GEMM kernels exist for sm_120 in cuBLAS 13.2. Tested all combinations
  (OP_T/OP_N, BF16/FP32/FP16 output, with/without D_SCALE_MODE) — all fail.
  Root cause: cuBLASLt FP4 kernels only compiled for sm_100 (data center Blackwell).
  Fixed probe config (BF16 output, OP_T, D_SCALE_MODE) to auto-activate when NVIDIA
  adds sm_120 support. CUTLASS NVFP4 path is primary — no performance impact.
