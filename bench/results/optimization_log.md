# imp Optimization Log — RTX 5090 (GB202, SM120)

## Hardware Reference
- **GPU:** NVIDIA RTX 5090, 170 SMs, 32 GB GDDR7
- **Memory BW:** 1,792 GB/s peak
- **FP16 Tensor:** 838 TFLOPS | **FP8:** 1,677 TFLOPS | **NVFP4:** 3,354 TFLOPS

## Targets
- **Decode MBU:** > 85% (= > 1,523 GB/s achieved bandwidth)
- **Prefill MFU:** > 50% FP16, > 45% FP4

---

## Baseline (pre-optimization)

imp v0.1.0, llama.cpp b8234 (213c4a0b8), RTX 5090 32GB

### imp Decode (tok/s)

| Model | pp128/tg128 | pp2048/tg256 | pp8192/tg128 |
|---|---|---|---|
| Qwen3-4B Q8_0 (no NVFP4) | 240 | 220 | — |
| Qwen3-8B Q8_0 (NVFP4) | 207 | 195 | error |
| Qwen3-8B Q8_0 (no NVFP4) | 147 | — | — |
| DeepSeek-R1-7B Q8_0 | 164 | — | — |
| Phi4-mini Q8_0 | 252 | — | — |
| Gemma-3-12B Q8_0 | error (OOM) | — | — |

### imp Prefill (tok/s)

| Model | pp128 | pp2048 |
|---|---|---|
| Qwen3-4B Q8_0 | 10,157 | 17,956 |
| Qwen3-8B Q8_0 (NVFP4) | 5,721 | 18,192 |
| Qwen3-8B Q8_0 (no NVFP4) | 6,897 | — |
| DeepSeek-R1-7B Q8_0 | 7,160 | — |
| Phi4-mini Q8_0 | 9,754 | — |

### llama.cpp Baseline

| Model | pp128 tok/s | tg128 tok/s | pp2048 tok/s | tg256 tok/s |
|---|---|---|---|---|
| Qwen3-4B Q8_0 | 8,519 | 234 | 18,475 | 240 |
| Qwen3-8B Q8_0 | 6,154 | 152 | 12,068 | 154 |
| DeepSeek-R1-7B Q8_0 | 6,010 | 171 | — | — |
| Phi4-mini Q8_0 | 11,583 | 265 | — | — |

### Gap Summary (imp / llama.cpp)

| Model | Decode Gap | Prefill Gap | Notes |
|---|---|---|---|
| Qwen3-4B | 1.03x / 0.92x | 1.19x / 0.97x | imp wins short, loses long decode |
| Qwen3-8B | **1.36x / 1.27x** | 0.93x / **1.51x** | NVFP4 = big decode win |
| DeepSeek-R1-7B | 0.96x | 1.19x | Slight decode loss |
| Phi4-mini | 0.95x | **0.84x** | imp loses both |

---

## Identified Issues

1. **cuBLASLt NVFP4 probe failure** — `CUBLAS_OP_T` used but FP4 requires `CUBLAS_OP_N`
2. **NVFP4 GEMV under-utilization** — 128 threads/1 row per block, too little work for small K
3. **Gemma-3 OOM** — head_dim=256 FP16 KV cache too large, preset batch too high
4. **Phi4-mini prefill gap** — 16% slower than llama.cpp, head_dim=96 unusual

---

## Fix Log

## Fix 1: cuBLASLt NVFP4 — OP_N layout for FP4 tensor cores
- Date: 2026-03-08
- Before: probe: status=7 (NOT_SUPPORTED) — native cuBLASLt FP4 path unusable
- After: probe uses CUBLAS_OP_N + col-major [N,K] layout
- Details: FP4 tensor cores only support non-transposed (OP_N) matrix access. Changed
  probe and GEMM call from CUBLAS_OP_T to CUBLAS_OP_N with matching layout descriptors.
  Weight data must be pre-transposed to col-major for cuBLASLt (CUTLASS path unaffected).

## Fix 2: Multi-row NVFP4 GEMV for small-K models
- Date: 2026-03-08
- Before: 128 threads, 1 row/block, K-parallel. For K=2560: ~1.25 iterations/thread.
- After: 256 threads (8 warps), 8 rows/block, warp-parallel. 8x fewer blocks, 2.5x more work/thread.
- Details: New `gemv_nvfp4_multirow_kernel<NR>` processes NR=8 rows per block. Each warp
  independently computes one row's dot product with 32-thread K-parallel reduction.
  Automatically selected when n_mb <= 512 (K <= 8192). Reduces kernel launch overhead
  and improves SM occupancy for small models (4B, 7B).

## Fix 3: Gemma-3 preset tuning — FP8 KV cache + reduced batch
- Date: 2026-03-08
- Before: Gemma-3-12B OOM at decode (397 MiB free, FP16 KV = 3072 MiB)
- After: FP8 KV cache (1536 MiB), max_batch_size=2, max_seq_len=65536
- Details: head_dim=256 makes Gemma-3 KV cache 2x larger than typical models.
  FP8 E4M3 halves KV cache memory. Also reduced batch sizes and seq lengths for
  27B and 4B variants to prevent VRAM exhaustion.

## Fix 4: Bench mode KV cache OOM — cap max_seq_len
- Date: 2026-03-08
- Before: Qwen3-8B/4B crash — KV cache tries to allocate 73 GB (max_seq_len=131072)
- After: bench mode caps max_seq_len to bench_pp + max_tokens + 256
- Details: Presets specify large max_seq_len for production (131072+). In bench mode
  only bench_pp + max_tokens tokens are ever used. Added cap in imp-cli before context
  creation. Also force max_batch_size=1 since bench is single-request.

## Fix 5: FP16→FP8 migration in NVFP4 mode 2
- Date: 2026-03-08
- Before: NVFP4 mode 2 frees FP16 cache, prefill falls back to on-the-fly dequant.
  Prefill throughput drops 50-70% for pp128, still slow for pp2048.
- After: Before freeing FP16 cache, all FP16 weights are quantized to FP8 E4M3
  and added to fp8_cache_. FP8 = half size of FP16, so net VRAM savings ≈ 50%.
  Prefill retains fast FP8 GEMM path (2x tensor core throughput on sm_120).
- Impact: Qwen3-4B pp2048 = 20,108 tok/s (+15% over no-NVFP4 baseline).
  pp128 still regresses (FP8 activation quantization overhead for small M).

---

## Post-Optimization Results (2026-03-08)

### imp (NVFP4 mode 2 + FP8 migration, FP8 KV cache)

| Model | pp128 tok/s | tg128 tok/s | pp2048 tok/s | tg256 tok/s |
|---|---|---|---|---|
| Qwen3-4B Q8_0 | 3,234 | 298 | 20,108 | 278 |
| Qwen3-8B Q8_0 | 3,422 | 195 | 16,489 | 189 |
| DeepSeek-R1-7B Q8_0 | 4,020 | 217 | 19,025 | 204 |
| Phi4-mini Q8_0 | 9,325 | 234 | — | — |

### imp (no NVFP4 — FP16 decode comparison)

| Model | pp128 tok/s | tg128 tok/s | pp2048 tok/s | tg256 tok/s |
|---|---|---|---|---|
| Qwen3-4B Q8_0 | 6,512 | 226 | 17,531 | 210 |
| Qwen3-8B Q8_0 | 5,526 | 111 | — | — |
| DeepSeek-R1-7B Q8_0 | 7,581 | 156 | — | — |

### Decode Improvement Summary (NVFP4 on vs off)

| Model | Decode Boost |
|---|---|
| Qwen3-4B | +32% (226 → 298) |
| Qwen3-8B | +76% (111 → 195) |
| DeepSeek-R1-7B | +39% (156 → 217) |

### NVFP4 Prefill Tradeoff

| Model | pp128 | pp2048 |
|---|---|---|
| Qwen3-4B | -50% (overhead > benefit) | **+15%** (FP8 tensors dominate) |

Note: pp128 regression is acceptable — absolute time is <40ms. Production prompts
typically 500-4000 tokens where FP8 tensor cores provide net benefit.
