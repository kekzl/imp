# TODO

## Performance

- [x] **Gemma-3-12B Decode Bench** — ~~Bench mode "internal error"~~.
  Fix: `ignore_eos=true` during prefill (synthetic tokens → immediate EOS → FINISHED).
  `imp_decode_step` sets ignore_eos correctly for actual decode.

- [~] **Gemma-3-12B NVFP4 Decode** — head_dim=256 KV cache consumes VRAM, limiting
  NVFP4 budget. GeGLU+GEMV fusion boosted decode: 59→100 tok/s (+70%).
  Remaining: VRAM budget still limited by head_dim=256 KV cache.

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
  Remaining: RMSNorm+GEMV fusion for further kernel count reduction.

- [ ] **cuBLASLt NVFP4 Probe** — status=7 (INTERNAL_ERROR), likely
  driver/CUDA version issue. CUTLASS NVFP4 fallback works fine.
  Low priority — no performance impact.
