# CLAUDE.md

## Project Overview

**imp** is a high-performance LLM inference engine written in C++20 and CUDA. It targets NVIDIA Hopper (sm_90a) and Blackwell (sm_100, sm_120) GPUs, leveraging CUDA 13.1+ features such as Green Contexts, Programmatic Dependent Launch (PDL), and CUDA Graphs. The engine supports GGUF and SafeTensors model formats, multiple quantization schemes (FP8, INT8, INT4, NVFP4), and architectures including LLaMA, Mistral, Mixtral, DeepSeek, Qwen3, Gemma-3 (text + vision), and Nemotron-H. Vision support uses a SigLIP encoder for Gemma-3 multimodal via separate mmproj.gguf files.

## Repository Structure

```
imp/
├── include/imp/          # Public C API headers
│   ├── imp.h             # Main API: model load, context, generate, tokenize
│   ├── config.h          # ImpConfig struct and defaults
│   ├── types.h           # Enums: DType, ModelArch, QuantType, ModelFormat
│   └── error.h           # Error codes and imp_error_string()
├── src/
│   ├── core/             # Tensor, Buffer, Allocator, Logging, Threading
│   ├── compute/          # CUDA kernels: GEMM, attention, RoPE, LayerNorm,
│   │                     #   activation, embedding, sampling, softmax, MoE routing
│   ├── memory/           # GPU/pinned allocators, KV cache (block-based), KV cache manager
│   ├── model/            # Model loading (GGUF/SafeTensors), weight upload, tokenizer
│   ├── quant/            # Quantization: FP8, INT8, FP16 dequant, NVFP4, quant GEMM
│   ├── graph/            # Compute graph DAG (Op, Graph, GraphExecutor)
│   ├── runtime/          # Engine, Scheduler, Request, Batch, Green Contexts,
│   │                     #   CUDA Graphs, PDL, Speculative Decoding
│   ├── vision/           # SigLIP vision encoder, mmproj GGUF loader, image preprocessing
│   └── api/              # C API implementation (imp_api.cpp)
├── tools/
│   ├── imp-cli/          # CLI tool: interactive and single-prompt inference
│   ├── imp-server/       # OpenAI-compatible HTTP server (SSE streaming)
│   └── imp-bench/        # Benchmark tool: GEMM, attention, end-to-end
├── third_party/stb/      # stb_image headers (image loading for vision)
├── tests/                # Google Test suite (22 test files)
├── cmake/                # Custom CMake modules (CompilerFlags, FindCUDAToolkit131)
├── CMakeLists.txt        # Build configuration
└── .gitignore
```

## Build System

CMake 3.25+ with C++20 and CUDA 20 standards. The project builds as a static library (`imp`) plus optional tools and tests.

### Build Commands

```bash
# Configure (out-of-source build)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build build -j$(nproc)

# Build with specific options
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DIMP_BUILD_TESTS=ON \
  -DIMP_BUILD_TOOLS=ON \
  -DIMP_BUILD_BENCH=ON

# Release with debug info
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Build Options

| Option | Default | Description |
|---|---|---|
| `IMP_BUILD_TESTS` | ON | Build the Google Test suite |
| `IMP_BUILD_TOOLS` | ON | Build imp-cli and imp-bench |
| `IMP_BUILD_BENCH` | ON | Build benchmark tool |
| `IMP_BUILD_SERVER` | ON | Build imp-server (OpenAI-compatible HTTP server) |
| `CMAKE_CUDA_ARCHITECTURES` | `90a;100;120` | Target GPU architectures |

### Dependencies

- **CUDA Toolkit 13.1+** (required) — cudart, cuda_driver, cublas, cublasLt
- **CUTLASS v4.4.1** (fetched via FetchContent) — Hopper FMHA (Example 88), MoE Grouped GEMM
- **Google Test v1.14.0** (fetched via FetchContent when tests enabled)
- **stb_image / stb_image_resize2** (vendored in `third_party/stb/`) — image loading for vision
- **pthread** (linked privately)

### Target GPU: NVIDIA RTX 5090 (GB202, Blackwell)

| Spec | Value |
|---|---|
| Compute Capability | sm_120a |
| SMs | 170 |
| CUDA Cores | 21,760 (128/SM) |
| Tensor Cores | 680 (5th gen, 4/SM) |
| Boost Clock | 2,407 MHz |
| VRAM | 32 GB GDDR7, 512-bit bus |
| Memory Bandwidth | 1,792 GB/s (28 Gbps/pin) |
| TDP | 575 W |

**Cache Hierarchy:**

| Level | Size | Notes |
|---|---|---|
| L0 Instruction Cache | 32 KB/SM | |
| L1 Data Cache / Shared Memory | 128 KB/SM | Configurable split (e.g. 64/64, 100/28, 28/100) |
| L2 Cache | 96 MB | Unified, shared across all SMs |
| L3 Cache | n/a | L3 only on data center Blackwell (B200/B300) |

**Tensor Core Throughput (at boost clock):**

| Precision | Dense | 2:4 Sparse |
|---|---|---|
| FP4 (NVFP4 E2M1) | 3,354 TOPS | 6,708 TOPS |
| FP8 (E4M3/E5M2) | 1,677 TFLOPS | 3,354 TFLOPS |
| FP16 / BF16 | 838 TFLOPS | 1,677 TFLOPS |
| INT8 (dp4a) | 1,677 TOPS | 3,354 TOPS |
| FP32 (CUDA Cores) | 105 TFLOPS | — |

**Key for imp kernel tuning:**
- L2 is large enough to cache full KV blocks for moderate context lengths
- 128 KB configurable L1/SMEM per SM — attention kernels use high SMEM configs
- NVFP4 tensor cores give 2x FP8 throughput — decode GEMV is still memory-bound
- 170 SMs → split-K paged attention targets ~340 blocks (2 blocks/SM occupancy)

### Hardware Constraints

Only one GPU is available. **Always test models sequentially** — never run multiple model instances in parallel.

## Running Tests

```bash
# Build tests
cmake -B build -DIMP_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Run all tests via CTest
cd build && ctest --output-on-failure

# Or run the test binary directly
./build/imp-tests

# Run specific test
./build/imp-tests --gtest_filter="TensorTest.*"
```

Tests require an NVIDIA GPU with the appropriate compute capability. Test files are in `tests/` and use Google Test (GTest).

### Test Files

| File | Covers |
|---|---|
| `test_tensor.cpp` | Tensor construction, strides, reshape, slicing |
| `test_gguf_loader.cpp` | GGUF model file parsing |
| `test_tokenizer.cpp` | Tokenizer encode/decode |
| `test_kv_cache.cpp` | KV cache block allocation, ref counting, LRU |
| `test_attention.cu` | Flash attention prefill kernels |
| `test_attention_tc.cu` | Tensor-core attention (Hopper WMMA) |
| `test_paged_attention.cu` | Paged attention decode (split-K, FP8, INT8) |
| `test_rope.cu` | Rotary positional embeddings |
| `test_layernorm.cu` | RMSNorm kernels |
| `test_activation.cu` | SwiGLU, GeGLU activation kernels |
| `test_embedding.cu` | Token embedding lookup |
| `test_gemm.cu` | GEMM/GEMV correctness |
| `test_moe.cu` | Mixture-of-Experts routing |
| `test_moe_executor.cu` | MoE end-to-end execution |
| `test_quant.cu` | Quantization kernels |
| `test_quant_integration.cu` | Quantized inference pipeline |
| `test_fp8_gemm.cu` | FP8 GEMM correctness |
| `test_fp8_kv_cache.cu` | FP8 E4M3 KV cache read/write |
| `test_nvfp4_quant.cu` | NVFP4 quantization |
| `test_sampling.cu` | Sampling kernels (argmax, top-k/p) |
| `test_reduce.cu` | Reduction kernels |
| `test_green_ctx.cu` | CUDA Green Context SM partitioning |
| `test_chat_template.cpp` | Chat template rendering |
| `test_e2e.cpp` | End-to-end generation pipeline |
| `test_continuous_batching.cpp` | Continuous batching scheduler |
| `test_speculative.cpp` | Speculative decoding (draft + verify) |

## Tools

### imp-cli

Interactive and single-shot LLM inference.

```bash
./build/imp-cli --model path/to/model.gguf --prompt "Hello world"
./build/imp-cli --model path/to/model.gguf --interactive
```

Options: `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--interactive`, `--device`, `--mmproj`, `--image`.

### imp-bench

Benchmarks for GEMM, attention, and end-to-end inference.

```bash
./build/imp-bench
```

## Benchmark Results (v0.2, RTX 5090, 2026-03-14)

All benchmarks on a single NVIDIA RTX 5090 (32 GB GDDR7, Blackwell sm_120). Models loaded from GGUF. imp uses NVFP4 decode cache + FP8 prefill cache. llama.cpp b5285 with flash attention enabled (`-fa 1`).

### Decode Throughput (tg128, tok/s)

| Model | Quant | imp v0.2 | llama.cpp | Speedup |
|-------|-------|----------|-----------|---------|
| Qwen3-4B | Q8_0 | **393** | 244 | **+61%** |
| Qwen3-8B | Q8_0 | **262** | 157 | **+67%** |
| Gemma-3-12B | Q8_0 | **146** | 98 | **+49%** |
| Phi-4-mini | Q8_0 | 264 | **277** | -5% |
| Qwen3-Coder-30B MoE | Q6_K | **293** | 251 | **+17%** |

### Prefill Throughput (pp512, tok/s)

| Model | Quant | imp v0.2 | llama.cpp | Speedup |
|-------|-------|----------|-----------|---------|
| Qwen3-4B | Q8_0 | **27240** | 21337 | **+28%** |
| Qwen3-8B | Q8_0 | **17486** | 14172 | **+23%** |
| Gemma-3-12B | Q8_0 | **11262** | 9269 | **+22%** |
| Phi-4-mini | Q8_0 | 20949 | **27259** | -23% |
| Qwen3-Coder-30B MoE | Q6_K | 5722 | **6090** | -6% |

**Notes:**
- Phi-4-mini: imp disables NVFP4 for d_model < 4096 dense models, using dp4a Q8_0 path instead. llama.cpp's Q8_0 CUDA kernels are well-tuned for this model size.
- Prefill numbers have high variance due to cuBLAS autotuning algorithm selection between container restarts (up to 2.6x range on Gemma-3). Decode numbers are stable.

## Code Conventions

### Language and Standards
- **C++20** for host code, **CUDA C++20** for device code
- Public API is C-compatible (`extern "C"` in `include/imp/`)
- Internal code uses the `imp` namespace

### Naming
- Classes and structs: `PascalCase` (`GraphExecutor`, `KVCacheManager`)
- Functions and methods: `snake_case` (`forward_batch`, `allocate_blocks`)
- Member variables: `trailing_underscore_` (`model_`, `config_`)
- Constants: `kPascalCase` (`kMaxDims`, `kKVBlockSize`)
- Enums: `PascalCase` values (`DType::FP16`, `OpType::ATTENTION_PREFILL`)
- C API: `imp_` prefix with `snake_case` (`imp_model_load`, `imp_context_create`)
- Macros: `IMP_UPPER_CASE` (`IMP_LOG_ERROR`, `IMP_CUDA_13_1`)

### File Organization
- Headers (`.h`) and implementations (`.cpp` / `.cu`) are co-located in `src/` subdirectories
- Public headers live in `include/imp/` and use `#pragma once`
- CUDA files use `.cu` extension; pure C++ uses `.cpp`
- Each `src/` subdirectory corresponds to a logical module (core, compute, memory, model, quant, graph, runtime, vision, api)

### Error Handling
- C API returns `ImpError` codes (negative values indicate errors, 0 = success)
- Internal C++ code uses `bool` return values (true = success) with logging
- Logging uses `IMP_LOG_DEBUG/INFO/WARN/ERROR/FATAL` macros (defined in `src/core/logging.h`)
- CUDA errors are checked and logged (not thrown as exceptions)

### Memory Management
- GPU memory: `device_allocator.cu` with `cudaMalloc`/`cudaFree`
- Pinned host memory: `pinned_allocator.cpp`
- KV cache: block-based allocation with configurable block size (`kKVBlockSize = 16` tokens)
- Model weights: mmap'd from disk, then uploaded/dequantized to GPU
- `std::unique_ptr` and `std::shared_ptr` for ownership; raw pointers for non-owning references

### Compiler Flags
- C++: `-Wall -Wextra -Wpedantic`
- CUDA: `--expt-relaxed-constexpr --extended-lambda`
- Debug builds define `IMP_DEBUG=1`
- Release builds use `-O3 --use_fast_math`

## Architecture Notes

### Inference Pipeline
1. **Model Loading** — GGUF or SafeTensors parsed and weights mmap'd (`src/model/`)
2. **Weight Upload** — Weights dequantized and uploaded to GPU (`weight_upload.cu`)
3. **Graph Construction** — Transformer DAG built for visualization/debug (`src/graph/`)
4. **Execution** — `GraphExecutor` runs a hardcoded forward pass (no graph walking at runtime)
5. **Scheduling** — `Scheduler` manages continuous batching with prefill/decode separation
6. **KV Cache** — Paged block allocation with LRU eviction and prefix caching
7. **Sampling** — Temperature, top-p, top-k sampling from logits

### Attention Dispatch
Runtime dispatch based on GPU compute capability:
- **Prefill (sm_90+)**: CUTLASS Hopper FMHA (`attention_cutlass_fmha.cu`) — WGMMA + TMA, falls back to WMMA if unsupported config (softcap, sliding window) or disabled via `IMP_NO_CUTLASS_FMHA=1`
- **Decode / Fallback sm_120+ (Blackwell)**: WMMA 8-warp attention (`attention_blackwell.cu`)
- **Decode / Fallback sm_90+ (Hopper)**: WMMA tensor-core attention (`attention_tc.cu`)
- **< sm_90**: Scalar Flash Attention 2 (`attention.cu`)

### Quantization Support
- **FP8 E4M3**: Per-tensor scale, FP8 GEMM via cuBLAS
- **INT8**: Per-channel dequantization
- **INT4 (Q4_0, Q4_K_M)**: GGML-compatible block formats
- **NVFP4 (FP4_E2M1)**: Blackwell-native, two-level micro-scale + tensor-scale

### CUDA 13.1 Features
- **Green Contexts**: SM partitioning for concurrent prefill/decode (`green_ctx.cu`)
- **PDL (Programmatic Dependent Launch)**: Overlaps kernel tails with next kernel heads (`pdl.cu`)
- **CUDA Graphs**: Captured decode iterations for reduced launch overhead (`cuda_graph.cu`)

### Supported Model Architectures
- LLaMA (dense transformer)
- Mistral (GQA variant)
- Mixtral (Mixture-of-Experts)
- DeepSeek (MoE)
- Qwen3 / Qwen3-MoE
- Gemma-3 (text + vision via SigLIP encoder)
- Nemotron-H (Mamba2 + Attention + MoE hybrid)
- Generic fallback

### Vision (Multimodal)
Gemma-3 vision uses a frozen 400M-parameter SigLIP ViT that produces 256 image tokens per image, projected into the LLM's embedding space. The vision encoder weights ship as a separate `mmproj.gguf` file. The pipeline: load image → resize 896x896 → normalize → extract 14x14 patches → 27 SigLIP transformer layers → 4x4 avg pool → RMSNorm + linear projection → replace `<image_soft_token>` embeddings before LLM prefill.

### Speculative Decoding
Draft model generates K candidate tokens, target model verifies in a single pass. Uses stochastic acceptance for non-greedy sampling. KV cache manager supports rollback for rejected tokens.

## Verification Before Commit

**Every change MUST be verified in this order before `git add`, `git commit`, and `git push`:**

1. **Tests** — Build and run the test suite (`ctest --output-on-failure` or `./imp-tests`). All tests must pass.
2. **Performance** — Run benchmarks (`--bench`) on affected models. Verify no regressions in tok/s (prefill and decode).
3. **Real prompts** — Test with actual prompts (`--prompt "..."`) on at least 2-3 models to confirm correct, coherent output.

Only after all three checks pass may the changes be committed and pushed.
