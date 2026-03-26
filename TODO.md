# TODO

## Open Work

### GDN / Qwen3.5 Output Quality
Qwen3.5 Gated DeltaNet models produce degenerate output after ~15-30 tokens (repetition loops), while llama.cpp generates correct output with the same GGUF files. Benchmark throughput is correct (327 tok/s for 4B).

**Fixed so far:**
- SSM/GDN state reset between requests (prefix caching leak)
- Shared memory race condition in fused multi-token scan kernel
- L2 normalization epsilon (additive → max-based, matching llama.cpp)

**Still broken:** Prefill logits diverge slightly from llama.cpp even for 2-token prompts. The divergence accumulates through 24 GDN layers. Code review found no math errors — needs layer-by-layer tensor dump comparison against llama.cpp.

**Files:** `src/compute/gdn.cu`, `src/graph/executor_forward.cu` (run_gdn)

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
- **N-gram speculation**: Implemented (`src/runtime/ngram_spec.cpp`), viable for repetitive prompts.

---

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

## Upgrade Path: CUTLASS 4.4.2 + PTX 9.2

### CUTLASS 4.4.1 → 4.4.2
We're on CUTLASS v4.4.1 (FetchContent). Upgrade to 4.4.2 brings:
- **SM120f compilation** for examples and NVFP4/MX Grouped GEMM profiler
- **Hopper FMHA causal perf fix**: unnecessary convergence barriers in mbarrier sync
- **SM120 memory fence fix** for CLC scheduler Pingpong kernel
- **SM120 SMEM alignment fix** for scale factors (was causing garbage output)
- **SM120 PDL fix** for Grouped GEMM
- **Example 87**: SM120 Blockwise GEMM (reference for GGUF-dequant GEMM)
- **Example 92**: MoE low-latency kernels (TMA weights, CPASYNC tokens)
- **Block-scaled sparse kernels** for SM100/SM120
- **Heuristics-based autotuning** via nvidia-matmul-heuristics

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
Available in our CUDA 13.2.0 toolkit but not yet used:

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
