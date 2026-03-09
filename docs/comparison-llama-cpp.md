# imp vs llama.cpp — Technical Comparison

**Date:** 2026-02-26
**Scope:** Architecture, features, and performance of imp (commit 33cd6a1) vs llama.cpp (build c830f99, Feb 2026)

---

## Executive Summary

imp and llama.cpp occupy different positions in the LLM inference landscape. llama.cpp prioritizes **hardware breadth** — running on CPUs, NVIDIA/AMD/Intel GPUs, Apple Silicon, and browsers — with support for 161 model architectures and 30+ quantization formats. imp prioritizes **depth on NVIDIA Hopper/Blackwell**, exploiting architecture-specific features (WMMA, TCGEN05, Green Contexts, PDL, CUDA Graphs) to minimize latency on a narrow hardware target.

On an RTX 5090 (Blackwell, sm_120) with Qwen3-4B Q8_0, imp trails llama.cpp by ~5% on prefill and matches on decode. On the MoE model (Qwen3-Coder-30B-A3B Q6_K), imp leads by ~2% on prefill and ~10% on decode.

---

## Benchmark Results

**Hardware:** NVIDIA GeForce RTX 5090 (32 GB), Driver 591.86, CUDA 13.1, WSL2
**Methodology:** 512-token prompt, 128 generated tokens, temperature=0, batch size 1, 3 reps averaged

| Model | Quant | Engine | pp (tok/s) | tg (tok/s) |
|-------|-------|--------|------------|------------|
| Qwen3-4B | Q8_0 | llama.cpp | 20,324 | 212 |
| Qwen3-4B | Q8_0 | imp | 19,251 | 212 |
| Qwen3-30B-A3B (MoE) | Q6_K | llama.cpp | 6,070 | 197 |
| Qwen3-30B-A3B (MoE) | Q6_K | imp | 6,189 | 218 |

Both engines: flash attention enabled, all layers on GPU, single sequence.
imp: CUDA graphs + PDL enabled. llama.cpp: default CUDA settings.

---

## Architecture & Design

| | imp | llama.cpp |
|---|---|---|
| Language | C++20 / CUDA C++20 | C11 (GGML) / C++17 (llama) |
| API | C-compatible public headers | C-compatible public headers |
| GPU backends | CUDA only | CUDA, Metal, Vulkan, ROCm, SYCL, CANN, OpenCL, WebGPU |
| CPU backend | None | SSE4.2, AVX/AVX2/AVX-512, AMX, ARM NEON/SVE/SME |
| Target architectures | sm_90a, sm_100, sm_120 | sm_61 through sm_100+ |
| Multi-GPU | Single GPU | Layer split, row split, graph split |
| Build system | CMake 3.25+ | CMake 3.14+ |
| Dependencies | CUDA 13.1, cuBLAS, cuBLASLt, GoogleTest | None required (backends optional) |
| Tensor library | Custom `Tensor` class | GGML (standalone C tensor library) |
| Backend design | Monolithic CUDA | Pluggable `ggml_backend_i` interface |

imp is a single-backend engine that assumes NVIDIA hardware. llama.cpp's pluggable backend system trades some per-platform optimization depth for broad hardware support.

---

## Model Support

| | imp | llama.cpp |
|---|---|---|
| Model format (runtime) | GGUF + SafeTensors | GGUF only |
| Model architectures | ~10 (LLaMA, Mistral, Mixtral, DeepSeek, Qwen3, Qwen3-MoE, Phi-4, Gemma-3, Nemotron-H, generic) | 161 (dense, MoE, SSM, recurrent, encoder, vision, embedding) |
| Hybrid SSM+Attention | Nemotron-H (Mamba2 + Attention + MoE) | Jamba, Falcon-H1, Mamba, Mamba2 |
| Vision models | Gemma-3 (SigLIP ViT, 896x896, 256 tokens) | CLIP, Gemma3N, Qwen2VL |
| Embedding models | No | BERT, Nomic, Jina, Llama Embed |

imp's native SafeTensors support avoids the GGUF conversion step. llama.cpp requires `convert_hf_to_gguf.py` for SafeTensors models but covers far more architectures.

---

## Quantization

| Format | imp | llama.cpp |
|--------|-----|-----------|
| FP32 / FP16 / BF16 | FP16 primary, FP32 logits | All three |
| Q4_0 | Yes | Yes |
| Q4_K_M | Yes | Yes |
| Q6_K | Yes (dedicated GEMV kernel) | Yes |
| Q8_0 | Yes (dp4a GEMV) | Yes |
| Q2_K / Q3_K / Q5_K | Yes (dp4a GEMV) | Yes |
| IQ (importance-weighted) | No | Yes (IQ1_S through IQ4_NL) |
| TQ (ternary) | No | Yes (TQ1_0, TQ2_0) |
| FP8 E4M3 | Yes (cuBLAS GEMM, per-tensor scale) | Not a native GGUF type |
| NVFP4 (FP4 E2M1) | Yes (Blackwell-native, two-level scaling) | Experimental MXFP4 (compute-only) |
| INT8 | Yes (per-channel dequant) | Via Q8_0 block format |

llama.cpp has the broadest quantization coverage in the ecosystem (30+ types). imp covers fewer formats but includes FP8 and NVFP4 paths that exploit Hopper/Blackwell tensor core hardware directly.

**GEMM strategy for quantized weights:**
- **imp:** Pre-dequantizes weights to FP16 at init, then uses cuBLAS FP16 GEMM for prefill. Decode uses fused dp4a GEMV (quantize input to Q8_1, then `int8x4` dot product against raw quantized weights). Fused QKV GEMV for decode, strided batched K/V GEMM for prefill.
- **llama.cpp:** MMQ kernels fuse dequant + matmul in custom CUDA kernels. MMVQ for decode. cuBLAS used for large FP16 GEMM. No pre-dequant FP16 cache.

---

## Attention & KV Cache

| | imp | llama.cpp |
|---|---|---|
| Prefill attention | cuBLAS batched GEMM (S=QK^T, softmax, O=PV), Flash Attention 2 fallback | Custom Flash Attention kernel |
| Decode attention | Paged Attention with split-K | Standard attention (no PagedAttention) |
| Attention dispatch | Architecture-specific: scalar (any) -> WMMA (sm_90+) -> TCGEN05 (sm_120+) | Generic Flash Attention kernel |
| KV cache layout | Paged blocks, `[block, slot, kv_head, head_dim]`, block size 16 | Ring buffer, cell-based slots |
| KV cache eviction | LRU eviction + prefix caching | Slot reuse, defragmentation |
| KV cache quantization | FP16 and FP8 E4M3 | FP16, Q8_0, Q4_0 |
| Sliding window | Supported | Supported (ISWA for interleaved) |

imp's paged KV cache follows the vLLM convention, enabling efficient memory utilization with variable-length sequences. llama.cpp's ring-buffer approach is simpler but less memory-efficient under high fragmentation.

imp dispatches architecture-specific attention kernels per compute capability level — TCGEN05 on Blackwell leverages the systolic array for attention, while WMMA on Hopper uses Warp Matrix Multiply-Accumulate. llama.cpp uses a single Flash Attention implementation across all CUDA architectures.

---

## CUDA Optimization Features

| Feature | imp | llama.cpp |
|---------|-----|-----------|
| CUDA Graphs | Yes (decode iterations captured) | Yes (~1.2x speedup reported) |
| Programmatic Dependent Launch (PDL) | Yes (overlaps kernel tails/heads) | Open issue #15479, not merged |
| Green Contexts | Yes (SM partitioning for prefill/decode) | Not implemented |
| TCGEN05 (Blackwell systolic) | Yes (attention kernel) | Not implemented |
| WMMA (Hopper tensor core) | Yes (attention kernel) | Used in FA kernel |
| Fused RMSNorm + Q8_1 quantize | Yes (decode: skip FP16 intermediate) | No |
| Fused QKV GEMV | Yes (decode: one kernel for Q+K+V) | No |
| Batched K/V GEMM | Yes (prefill: `cublasGemmStridedBatchedEx`) | No |
| Fused O-projection + residual | Yes (cuBLAS beta=1 for prefill, dp4a+residual for decode) | No |
| dp4a (INT8 SIMD) GEMV | Yes (Q6_K and Q8_0 via Q8_1 input quantization) | Yes (MMQ kernels) |

imp's kernel fusion strategy aggressively reduces launch overhead and eliminates intermediate buffers. The PDL + CUDA Graphs combination is particularly effective for decode, where many small kernels dominate runtime.

---

## Speculative Decoding

| | imp | llama.cpp |
|---|---|---|
| Draft model | Yes | Yes |
| Stochastic acceptance | Yes (non-greedy sampling) | Not documented |
| N-gram methods | No | 4 variants (simple, map-k, map-k4v, mod) |
| KV cache rollback | Yes | Yes |
| Max draft tokens | Configurable | Default 16 |

llama.cpp offers more speculative decoding strategies. imp's implementation focuses on draft model + target verification with proper stochastic acceptance for non-greedy sampling modes.

---

## Batching & Serving

| | imp | llama.cpp |
|---|---|---|
| Continuous batching | Yes (Scheduler with prefill/decode separation) | Yes (llama-server) |
| Prefix caching | Yes (KV cache LRU with prefix sharing) | Yes (sequence ID sharing) |
| HTTP server | Yes (imp-server, OpenAI-compatible API, SSE streaming) | Yes (llama-server, OpenAI-compatible API) |
| Concurrent sequences | Yes | Yes |
| Prompt caching across requests | Via KV cache prefix matching | Via shared sequence IDs |

Both engines ship OpenAI-compatible HTTP servers with SSE streaming. imp-server supports `/v1/chat/completions`, `/v1/completions`, tool/function calling, logprobs, vision (base64 images), and API key auth.

---

## Memory Efficiency

For Qwen3-4B Q8_0 on RTX 5090 (32 GB):

| Component | imp | llama.cpp |
|-----------|-----|-----------|
| Model weights (Q8_0) | ~4.0 GiB | ~4.0 GiB |
| FP16 weight cache (prefill) | ~1.6 GiB (incl. fused KV) | None (MMQ fuses dequant) |
| KV cache | Paged, on-demand | Ring buffer, pre-allocated |
| cuBLAS workspace | Pre-allocated at init | Per-call or pre-allocated |
| Peak VRAM | ~7-8 GiB (model + FP16 cache + KV) | ~5-6 GiB (model + KV) |

imp trades ~1.6 GiB additional VRAM for its FP16 weight cache to enable fast cuBLAS GEMM during prefill. This is worthwhile on cards with sufficient VRAM (RTX 5090: 32 GB) but could be limiting on smaller cards. The cache has a VRAM budget system that gracefully falls back to on-the-fly dequant when memory is tight.

---

## Build & Portability

| | imp | llama.cpp |
|---|---|---|
| Min CUDA version | 13.1 (hard requirement) | 11.7+ |
| Target GPUs | sm_90a, sm_100, sm_120 | sm_61+ |
| C++ standard | C++20 | C++17 (C11 for GGML) |
| External dependencies | cuBLAS, cuBLASLt, GoogleTest | None required |
| Platforms | Linux (WSL2 tested) | Linux, macOS, Windows, Android, iOS, WebAssembly |
| Binary size | Static library + tools | Static/shared library + tools + server |

llama.cpp runs on essentially any platform with a C compiler. imp requires a modern NVIDIA GPU with CUDA 13.1, limiting it to datacenter and high-end consumer Hopper/Blackwell hardware.

---

## Strengths & Trade-offs

### imp strengths
- **Architecture-specific optimizations:** TCGEN05, WMMA, PDL, Green Contexts — features llama.cpp doesn't use
- **Aggressive kernel fusion:** Fused RMSNorm+quantize, fused QKV GEMV, batched K/V GEMM, fused O+residual
- **Native SafeTensors support:** No conversion step needed
- **Paged KV cache:** vLLM-style memory efficiency with LRU eviction and prefix caching
- **FP8/NVFP4:** First-class support for Hopper/Blackwell quantization formats

### llama.cpp strengths
- **Hardware breadth:** Runs on CPU, NVIDIA, AMD, Intel, Apple, browser
- **Model coverage:** 161 architectures vs ~8
- **Quantization variety:** 30+ formats including sub-2-bit (IQ1_S, TQ1_0)
- **Production-ready serving:** Mature HTTP server with extensive OpenAI API compatibility
- **Multi-GPU:** Layer/row/graph split across multiple GPUs
- **Community & ecosystem:** Largest open-source LLM inference community, extensive tooling

### When to choose imp
- Single high-end NVIDIA GPU (Hopper/Blackwell)
- Latency-sensitive decode with small batch sizes
- Models available in GGUF or SafeTensors with supported architectures
- Need for FP8/NVFP4 quantization paths
- Embedding into a custom C/C++ application

### When to choose llama.cpp
- Non-NVIDIA or mixed hardware environments
- Broad model architecture support needed
- Sub-4-bit quantization for memory-constrained deployment
- Multi-GPU inference required
- Production HTTP serving with OpenAI API compatibility
- Platforms beyond Linux (macOS, Windows, mobile, web)

---

## Acknowledgment

imp would not exist without llama.cpp. Georgi Gerganov and the llama.cpp community have done extraordinary work in making LLM inference accessible to everyone — on every platform, every hardware vendor, every budget. The GGUF format that imp loads natively is a llama.cpp creation. The quantization schemes (Q4_0, Q4_K_M, Q6_K, Q8_0) that imp implements originate from the GGML ecosystem. The benchmark methodology, the model support matrix, the entire concept of "local LLM inference" as a practical reality — all of this was pioneered and popularized by llama.cpp.

imp takes a different path: trading hardware breadth for depth on a single vendor. But every optimization in imp is measured against llama.cpp as the baseline. When imp is faster, it's often by single-digit percentages — a testament to how well-optimized llama.cpp already is despite supporting orders of magnitude more hardware. When llama.cpp is faster (as on dense prefill), it serves as motivation to find the next bottleneck.

We have deep respect for the engineering quality, the relentless pace of development, and the open-source ethos of the llama.cpp project. It has raised the bar for the entire local inference ecosystem.
