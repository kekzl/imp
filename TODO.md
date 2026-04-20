# TODO

## Open Work

### GDN / Qwen3.5 Output Quality — FIXED
Root cause: Jinja2 engine lacked macro support. Qwen3.5's chat template uses
`{% macro render_content() %}` — without macro support, user content rendered
as "None", causing the model to ignore prompts entirely.

Fix: Jinja2 macro support (MacroNode, parse_macro, call_macro). GDN kernels
were correct all along. Both Qwen3.5-4B and 9B now produce coherent output.

### MXFP4 Native GGUF Weight Format
Plan documented in `docs/MXFP4_GGUF_PLAN.md`. Native MXFP4 weights would feed directly into Blackwell tensor cores via CUTLASS — zero dequant overhead, expected 2-4x prefill speedup vs Q4_K_M.

**Status:** CUTLASS MXFP4 GEMM path is fully implemented (`--mxfp4-prefill`), but only works when NVFP4 cache exists as source data. Native MXFP4 GGUF eliminates this dependency.

**Remaining:**
1. GGUF type extension + loader (~50 lines)
2. Python converter: SafeTensors → block-Hadamard → MXFP4 → GGUF (~200 lines)
3. Weight upload: mmap → GPU-ready format (~30 lines)
4. MXFP4 GEMV for decode (~100 lines CUDA)
5. Optional: MR-GPTQ calibration for better quality (~150 lines Python)

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

**Remaining opportunity:** MoE Grouped GEMM uses CUTLASS 2.x API with D2H sync workaround (`gemm_cutlass_grouped_sm120.cu:115-131`). Migration to CUTLASS 3.x device-side problem shapes would eliminate 2 cudaStreamSynchronize() per MoE forward. Low priority — only matters at MoE-heavy workloads.

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
