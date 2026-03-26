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

## Research / Future

### BitDecoding (arxiv:2503.18773)
Tensor-core-based decoding with low-bit KV cache. 8.6x vs FP16 FlashDecoding on Blackwell. Requires MXFP4 KV cache format — builds on TurboQuant MXFP4 infrastructure.

### DeltaKV (arxiv:2602.08005)
Residual-based KV compression. 187 tok/s at 128k context on Blackwell PRO 6000. Orthogonal to weight quantization.

### CUDA 13.2 Features
- `IMP_CUDA_13_2=1` flag exists in CMakeLists.txt but no code uses it yet
- cuBLASLt MXFP4/NVFP4 GEMM: blocked on sm_120 kernel availability
- Sampling already uses optimized fused top-k kernel (cub::DeviceTopK not needed)
