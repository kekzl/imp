# TODO

## Performance

- [x] **Gemma-3-12B Decode Bench** — ~~Bench mode "internal error"~~.
  Fix: `ignore_eos=true` during prefill (synthetic tokens → immediate EOS → FINISHED).
  `imp_decode_step` sets ignore_eos correctly for actual decode.

- [ ] **Gemma-3-12B NVFP4 Decode** — head_dim=256 KV cache consumes too much VRAM.
  NVFP4 budget = 0 tensors (no room after KV+weights). dp4a fallback works,
  but without NVFP4 bandwidth boost: ~89 tok/s (bench) vs ~44 tok/s (full KV).
  Fix: reduce KV cache preset, or implement NVFP4 for post-norm architecture.

- [x] **CUTLASS FMHA head_dim=96** — Added template `Shape<_128, _96, _96>`.
  Phi4-mini actually has head_dim=128 (rope_dim=96 is partial RoPE, not head_dim).
  CUTLASS FMHA already worked correctly. Template added for future models with head_dim=96.

- [x] **Init Async Quantization** — Eliminated per-tensor `cudaStreamSynchronize` in
  weight cache construction. FP8 overflow, FP16→FP8 migration, and NVFP4 cache now
  use async calibrate+quantize with single sync at end. Bulk cudaMalloc for FP8 data
  and scale buffers. DeepSeek-14B init: 55.4s → ~35s (37% faster).

- [ ] **Small Model Decode MBU** — ~450 kernels/decode step, launch overhead dominates.
  Qwen3-4B/Phi4-mini reach only ~57% MBU (theoretical max at 5us/kernel).
  CUDA Graphs + PDL already active — remaining gap is per-kernel granularity.
  Fix: kernel fusion (RMSNorm+Quantize+GEMV), GEMV block size 256→512.

- [ ] **cuBLASLt NVFP4 Probe** — status=7 (INTERNAL_ERROR), likely
  driver/CUDA version issue. CUTLASS NVFP4 fallback works fine.
  Low priority — no performance impact.
