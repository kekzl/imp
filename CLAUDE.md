# CLAUDE.md

## Project Overview

**imp** is a high-performance LLM inference engine written in C++20 and CUDA. It targets NVIDIA Hopper (sm_90a) and Blackwell (sm_100, sm_120) GPUs, leveraging CUDA 13.1+ features such as Green Contexts, Programmatic Dependent Launch (PDL), and CUDA Graphs. The engine supports GGUF and SafeTensors model formats, multiple quantization schemes (FP8, INT8, INT4, NVFP4), and architectures including LLaMA, Mistral, Mixtral, and DeepSeek.

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
│   └── api/              # C API implementation (imp_api.cpp)
├── tools/
│   ├── imp-cli/          # CLI tool: interactive and single-prompt inference
│   └── imp-bench/        # Benchmark tool: GEMM, attention, end-to-end
├── tests/                # Google Test suite (17 test files)
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
| `CMAKE_CUDA_ARCHITECTURES` | `90a;100;120` | Target GPU architectures |

### Dependencies

- **CUDA Toolkit 13.1+** (required) — cudart, cuda_driver, cublas, cublasLt
- **Google Test v1.14.0** (fetched via FetchContent when tests enabled)
- **pthread** (linked privately)

No other external dependencies. The project is self-contained.

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
| `test_kv_cache.cpp` | KV cache block allocation, ref counting, LRU |
| `test_attention.cu` | Flash attention prefill kernels |
| `test_attention_tc.cu` | Tensor-core attention (Hopper WMMA) |
| `test_rope.cu` | Rotary positional embeddings |
| `test_layernorm.cu` | RMSNorm kernels |
| `test_moe.cu` | Mixture-of-Experts routing |
| `test_moe_executor.cu` | MoE end-to-end execution |
| `test_quant.cu` | Quantization kernels |
| `test_quant_integration.cu` | Quantized inference pipeline |
| `test_fp8_gemm.cu` | FP8 GEMM correctness |
| `test_nvfp4_quant.cu` | NVFP4 quantization |
| `test_green_ctx.cu` | CUDA Green Context SM partitioning |
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

Options: `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--interactive`, `--device`.

### imp-bench

Benchmarks for GEMM, attention, and end-to-end inference.

```bash
./build/imp-bench
```

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
- Each `src/` subdirectory corresponds to a logical module (core, compute, memory, model, quant, graph, runtime, api)

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
- **sm_120+ (Blackwell)**: TCGEN05 systolic attention (`attention_blackwell.cu`)
- **sm_90+ (Hopper)**: WMMA tensor-core attention (`attention_tc.cu`)
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
- Generic fallback

### Speculative Decoding
Draft model generates K candidate tokens, target model verifies in a single pass. Uses stochastic acceptance for non-greedy sampling. KV cache manager supports rollback for rejected tokens.
