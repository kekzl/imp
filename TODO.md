# TODO

## Open Work

### MXFP4 Native GGUF Weight Format
Plan documented in `docs/MXFP4_GGUF_PLAN.md`. Native MXFP4 weights would feed directly into Blackwell tensor cores via CUTLASS — zero dequant overhead, expected 2-4x prefill speedup vs Q4_K_M.

**Status:** CUTLASS MXFP4 GEMM path is fully implemented (`--mxfp4-prefill`), but only works when NVFP4 cache exists as source data. Native MXFP4 GGUF eliminates this dependency.

**Quality caveat (measured 2026-04-23):** Qwen3-4B wikitext-2 PPL shows Q8_0 = 8.48, Q4_K_M = 8.67 (+2.2 %). MXFP4 round-to-nearest literature sits at +5–15 % — **worse than Q4_K_M** unless MR-GPTQ calibrated. The "Optional" calibration step (item 5 below) is effectively **required** for this project to be worth shipping.

**Remaining:**
1. GGUF type extension + loader (~50 lines)
2. Python converter: SafeTensors → block-Hadamard → MXFP4 → GGUF (~200 lines)
3. Weight upload: mmap → GPU-ready format (~30 lines)
4. MXFP4 GEMV for decode (~100 lines CUDA)
5. **Required** for competitive quality: MR-GPTQ calibration (~150 lines Python)

### TurboQuant Optimization
Current gap vs FP8 baseline: -23% decode (191 vs 248 tok/s). This is algorithm-inherent — QJL sketch computation adds per-token overhead.

**Optimized so far:**
- Warp-cooperative Q sketch (eliminated per-thread atomicOr contention)
- Warp-parallel QJL XNOR+popcount (32 lanes instead of lane-0 serial)
- INT4 dequant: div→mul, L1 prefetch, dead code removal
- PolarQuant INT4 symmetric clamp [-7,7] (was [-8,7])

**Remaining:** QJL overhead is inherent (~8% of decode). Only way to close the gap: remove QJL entirely (use MXFP4 K directions with group micro-scales instead).

### Speculative Decoding
- **EAGLE-3**: Dead end on single GPU (56 tok/s vs 306 baseline). Draft model shares same weights = 78% cost per layer.
- **Self-speculative**: Dead end (50% of baseline). Memory-bound decode can't amortize.
- **DFlash**: Not feasible — no draft model for Qwen3-32B, training requires datacenter GPUs.
- **N-gram speculation**: Implemented (`src/runtime/ngram_spec.cpp`), uses multi-sequence decode verify. +10% on repetitive content, ~0% overhead on non-repetitive. CLI: `--ngram-spec`.
- **TurboDraft (PPM + Classifier)**: Dead end. PPM 0% acceptance on real text; SVD classifier too lossy.
- **Pseudo-prefill verify bug**: Fixed in NgramSpec — now uses multi-sequence decode verify.

---

## Completed (v0.7) — 2026-04-23

### Long-context correctness
- [x] FP8 FMHA S_tile smem overlap fix (PR #33) — pp>1024 now coherent across all tested models; up to ×1.70 vs llama.cpp at pp=8192
- [x] Regression test `FmhaFP8Test.Qwen35LikeHD256_GQA41_SeqMultiTile` catches the bug class
- [x] Audited MXFP4 / FP16 / WMMA FMHA kernels for the same pointer-vs-slot mismatch — only FP8 was affected
- [x] Verified all FMHA variants on Qwen3-4B/8B, Qwen3.5-4B/9B GDN, Llama-3.2-3B, Qwen3-32B, Mistral-24B at pp=512/1024/2048/4096/8192

### Qwen 3.5 / Qwen 3.6 GDN
- [x] `gdn_scan_fused_kernel` launch_bounds fix (HD=128 miscompile) — PR #30
- [x] Partial-RoPE pair-offset fix — PR #30
- [x] `ssm_state_dtype` never auto-downgraded for GDN architectures — PR #28
- [x] GDN L2-norm PyTorch-style `rsqrtf(fmaxf(sum_sq, 1e-12))`
- [x] L2-window CUDA errors (clamped to `cudaDevAttrMaxAccessPolicyWindowSize`)
- [x] GDN reference infrastructure + Qwen 3.6 cache preservation (PR #25)
- [x] Qwen 3.6 `ModelArch::QWEN36_MOE` scaffold (PR #23)

### Gemma-4
- [x] SWA long-context degeneration (>1024 prompt tokens) — PR #21
- [x] rope_freqs on global layers — PR #20
- [x] Host-resident MoE fused gate_up split (e879bcd)
- [x] CUDA graphs for decode fast-path (PRs #11–#14)
- [x] Q4_K_M split-K pipeline cp.async loop (head_dim=512) — 55 → 183 tok/s
- [x] 3120-token KV ceiling fix — now 11 242 tok with `--min-kv-tokens 14000`
- [x] FP32 router + half rope_dim on global layers (5a1e844)

### Platform
- [x] CUDA 13.2.1 base images (PR #16)
- [x] Stream priorities, mem-sync domains, cluster spread (PR #17)
- [x] CUTLASS 3.x NVFP4 Grouped GEMM scaffold (PR #22)
- [x] StreamingLLM smart KV cache — attention sinks + sliding window (PR #26)
- [x] Weight-storage refactor: `TensorKind` + `StoragePlanner` + `gemm_dispatch` (PR #27)
- [x] `IMP_DEBUG_RAW` meta-flag (PR #29), `IMP_EXPERT_OVERHEAD_PCT` hint (PR #32)
- [x] `tools/analysis/layer_diff.py` for per-layer tensor diff vs llama.cpp
- [x] `Gemma4GraphsTest` e2e regression

### Known deferred
- 1024→2048 throughput cliff on small dense models (Qwen3-4B: 27k → 19k tok/s at dispatch switch). Correct but unoptimized. Options: raise cuBLAS cap past 1024, or tune FP8 FMHA occupancy / Bq.
- pp=512 on large dense models (Qwen3-32B, Mistral-24B): ~0.5–0.6× llama.cpp. Suspected cuBLAS autotuning variance + launch-overhead-bound regime.

---

## Completed (v0.6)

### NVFP4 Prequant SafeTensors Support
- [x] NVIDIA Model Optimizer NVFP4 models load from SafeTensors (tested: Qwen3-Coder-30B-A3B-FP4)
- [x] Phase 0 direct weight registration in `wcache_.nvfp4` (no re-quantization)
- [x] CUTLASS NVFP4 conversion for prefill GEMM (Phase 3b)
- [x] Per-expert NVFP4 GEMV in MoE legacy dispatch path
- [x] LM head prequant scale support (weight_scale, weight_scale_2, input_scale)
- [x] Shape bug fix: use `NvFP4QuantResult.N/.K` instead of packed tensor shape in GEMV dispatch
- [x] BF16→FP16 host-side conversion for non-quantized weights (norms, router, embeddings, lm_head)
- [x] CUDA graphs disabled for MoE models (D2H routing memcpy incompatible with graph capture)

### Server & Format Support
- [x] SafeTensors model loading in imp-server (was GGUF-only)
- [x] `resolve_model_auto()` with format auto-detection (SafeTensors directory vs GGUF file)
- [x] Server model list includes both GGUF files and SafeTensors directories
- [x] ~~Server hot-swap between GGUF and SafeTensors models~~ (reverted post-v0.6: `--model` is now required at startup, POST/DELETE `/v1/models` removed)
- [x] Chat template array-format support in `tokenizer_config.json` (HuggingFace convention)

### Verified
- [x] Qwen3-Coder-30B-A3B-FP4: single-turn, multi-turn, code gen, math — all correct
- [x] Benchmark: 38 tok/s decode (tg256), 90 tok/s prefill (pp512) on RTX 5090
- [x] 536/536 unit tests pass

## Completed (v0.4)

### Performance
- [x] NVFP4 decode cache: 50% VRAM savings, all dense weights → FP4 E2M1
- [x] FP8 prefill cache: FP8×FP8 cuBLASLt, 2x tensor core throughput
- [x] SwiGLU+GEMV fusion: +33% Qwen3-8B, +5.5% Qwen3-4B
- [x] GeGLU+GEMV fusion: +70% Gemma-3-12B
- [x] NVFP4 prmt register LUT: +4.7% Qwen3-4B, +7.7% Qwen3-8B, +16% Gemma-3-12B
- [x] RMSNorm vectorization: float4 loads, 100% cache line utilization
- [x] rmsnorm_quantize_q8_1: 256→1024 threads, 2x speedup
- [x] NVFP4 multi-row occupancy: __launch_bounds__ 6→8, +5% gate_up
- [x] NVFP4 LM head in async graph loop: -47% LM head latency
- [x] MoE fused TC kernel: persistent work-queue dispatch, -38% kernel time
- [x] Fused token-centric MoE scatter: no atomics, +3.1% MoE prefill
- [x] L2 cache tuning: streaming KV loads + persisting reservation, +2-4% decode

### Architecture Support
- [x] Qwen3.5 GDN (Gated DeltaNet): fused scan kernel, partial RoPE, output gate
- [x] Nemotron-H (Mamba2 + Attention + MoE hybrid)
- [x] Gemma-3 vision (SigLIP encoder, mmproj.gguf)

### Infrastructure
- [x] TurboQuant KV cache (PolarQuant + QJL + INT4 V)
- [x] TurboQuant MXFP4 variant (FP4 E2M1 + UE8M0 micro-scales for K)
- [x] CUTLASS MXFP4 prefill GEMM (sm_120 block-scaled tensor ops)
- [x] CUTLASS MXFP4 prefill attention (Q·K^T only)
- [x] Hadamard transform kernel (block-diagonal WHT, 16/32/64/128)
- [x] NVFP4→MXFP4 scale conversion
- [x] Code quality: 5 rounds refactoring, -1224 lines across 42 files

### Dead Ends (confirmed no benefit)
- RMSNorm+GEMV fusion, multi-row threshold >512, MoE SwiGLU+GEMV fusion
- PDL registration, dp4a fused act+GEMV, CUTLASS FMHA HD=256
- NVFP4 half2 FMA, split accumulators, __ldcs streaming weights
- Self-speculative decoding, EAGLE-3 (single GPU)
- NVFP4 prmt for SwiGLU/GeGLU, CUTLASS NVFP4 TC GEMM for M=1
- Paged attention __launch_bounds__, RMSNorm+NVFP4 LM head fusion
- cudaAccessPolicyWindow for decode activations

---

## CUTLASS 4.4.2 (DONE) + PTX 9.2

### CUTLASS 4.4.2 — Upgraded
Already on v4.4.2. SM120 fixes (SMEM alignment, memory fence, PDL, Hopper FMHA perf) are automatic via headers.

**Completed 2026-04-20:** CUTLASS 2.x `GemmGrouped` path removed. NVFP4 MoE now dispatches via CUTLASS 3.x `GroupProblemShape` (`gemm_cutlass_grouped_3x.cu`) with fused per-expert quantize — 30×+ prefill speedup on Qwen3-Coder-30B-A3B-FP4. FP16 MoE grouped path (Gemma-4 etc.) now goes directly to cuBLAS grouped, which is +24% faster than the retired 2.x wrapper on Gemma-4 Q5_K_M.

SM120 GEMM architecture notes:
- Pingpong (2×4 MMA warps) and Cooperative (1×8 MMA warps) schedules
- Default: KernelTmaWarpSpecializedCooperative
- GeForce: Cluster 1×1×1 only (no multicast), TN layout only
- SM120 FMHA: exists but blocked by wiring issues in fmha_v2 (WIP upstream)

### PTX ISA 9.2 Opportunities
Available in CUDA 13.2 but not yet used in imp:

**High value for attention/decode:**
- **`cp.async.bulk` with `.ignore_oob`**: OOB reads return zero instead of crashing.
  Eliminates bounds-checking in TMA descriptors for variable sequence lengths.
  Major simplification for paged attention with partial last blocks.
- **`st.async` with `.b128`**: 16-byte async stores. Perfect for KV cache writeback
  (one instruction per FP16 KV vector slot at head_dim=64).
- **`cvt .bf16x2` ↔ narrow types** (`.e2m1x2`, `.e4m3x2`): packed FP4/FP8 pair
  conversion. 2x throughput for KV cache quantize/dequantize pipeline.

**Medium value:**
- **`u8x4`/`s8x4` SIMD** for add/sub/min/max: packed 4-byte integer ops.
  Useful for index operations in KV cache management.
- **`add.sat`** for u16x2/s16x2/u32: overflow-safe index arithmetic.
- **`.scale_vec::4X` with `.ue8m0`** for MXFP4 MMA: finer scale granularity
  (1 scale per 4 elements vs per block). Better quantization accuracy.

**From PTX 9.1 (already available):**
- **`cvt .f16x2`/`.bf16x2` → `.e2m1x2`**: online FP16→FP4 quantization.
  Quantize K/V on-the-fly before KV cache write → 50% VRAM savings.

### tcgen05 on SM120
Tensor core instruction set for Blackwell. Available but constrained:
- No multicast (cluster 1×1×1), TN layout only
- Mixing tcgen05 with CUDA-core ops (softmax between MMA steps) requires
  expensive sync between Generic and Async proxies
- 256-bit loads available on SM120f (family feature)
- Flash Attention 4 pattern (fused softmax + MMA) still research-grade

---

## Research / Future

### BitDecoding (arxiv:2503.18773)
Tensor-core-based decoding with low-bit KV cache. 8.6x vs FP16 FlashDecoding on Blackwell. Requires MXFP4 KV cache format — builds on TurboQuant MXFP4 infrastructure.

### DeltaKV (arxiv:2602.08005)
Residual-based KV compression. 187 tok/s at 128k context on Blackwell PRO 6000. Orthogonal to weight quantization.

### CUDA 13.2 / CCCL 3.2 Features
Available in our CUDA 13.2.1 toolkit but not yet used:

**High Priority:**
- **Grouped GEMM with CUDA Graphs + device-side shapes** (cuBLASLt):
  Host-sync-free MoE expert dispatch — expert routing results stay on GPU,
  no D2H copy needed. Up to 4x speedup over multi-stream GEMM for MoE.
  Directly relevant for Qwen3-Coder-30B MoE and DeepSeek models.
- **`cub::DeviceTopK`** (`device_topk.cuh`): O(n) top-k via AIR algorithm.
  5x faster than radix sort for the top_k > 128 fallback path.
  Warp/Block-scope variants on CCCL roadmap for fused sampling.
- **`cub::DeviceSegmentedReduce` (fixed-size)**: uniform segment_size variant,
  up to 66x speedup for small segments. Perfect for per-head reductions in MHA
  where all heads have the same dimension.

**Medium Priority:**
- **`cudaMemcpyWithAttributesAsync`**: L2 persistence hints on individual transfers.
  Could pin frequently-accessed prefix cache segments in L2 without batched API.
- **Host Task Spin-Wait Dispatch** (`cudaLaunchHostFunc`): lower CPU-side callback
  latency. Relevant for dynamic token routing in speculative decoding.
- **`add.f32x2` native PTX** (Blackwell): native float2 ops for softmax reductions
  and attention score accumulation — reduces instruction count.
- **PTX ISA 9.2**: Extended FP4 cvt variants (.f16x2/.bf16x2 → FP4/FP8), `.scale_vec::4X`
  for MXFP4 MMA. Could improve NVFP4 quantization pipeline throughput.

**Investigate / Re-benchmark:**
- **NVFP4/MXFP8 small M,N perf** (cuBLAS 13.2): cuBLAS improved block-scaled
  kernels for M,N ≤ 32 on Blackwell. Previously, TC GEMM for M=1 (decode) was
  a dead end due to setup overhead. Re-benchmark: could NVFP4 cuBLASLt M=1 now
  beat the scalar prmt GEMV? If so, decode gets tensor core acceleration for free.
- **Per-batch device-side alpha/beta** (`CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE`):
  enables per-head scaling in GQA attention without extra kernel launches.
  Relevant for mixed-precision KV cache heads.

**Low Priority / Already Fixed / Verified:**
- **LMEM reduction on WDDM** (R595 driver): less local memory overhead on WSL2.
- **cuBLASLt Algo 66 bugfix**: concurrent TMA matmul corruption on sm_120 — fixed in 13.2.
- **NVCC GB202 CuTe GEMM corruption**: fixed in 13.1, we're on 13.2.
- **FP8 illegal memory access on GeForce**: fixed in 13.2. We use FP8 prefill cache.
- **Grouped GEMM k=0 gotcha**: our MoE code already filters empty experts (count==0 → skip)
  before calling cublasGemmGroupedBatchedEx. No action needed.

### sm_120f (Blackwell Family Feature Set)
Switched from sm_120a to sm_120f in commit fa3ced6:
- Enables TMA warp-specialized grouped GEMM tactics in CUTLASS/cuBLAS
- Forward-compatible within Blackwell family (sm_120, sm_121)
- Resolved ptxas C7600 register allocation bug for TurboQuant
- +10% TurboQuant prefill, +7.6% MXFP4 prefill
