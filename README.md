<p align="center">
  <img src="logo.svg" alt="imp" width="500">
</p>

<p align="center">
  High-performance LLM inference engine for NVIDIA Hopper and Blackwell GPUs.
</p>

---

imp is a C++20/CUDA inference engine that targets a narrow set of hardware (sm_90a, sm_100, sm_120) and exploits architecture-specific features — WMMA tensor-core attention, TCGEN05 systolic attention, Programmatic Dependent Launch (PDL), Green Contexts, and CUDA Graphs — to minimize latency for single-GPU deployment.

## Performance

Measured on RTX 5090 (Blackwell, 32 GB), CUDA 13.1, batch size 1, 512-token prompt, 128 generated tokens:

| Model | Quant | pp (tok/s) | tg (tok/s) |
|-------|-------|------------|------------|
| Qwen3-4B | Q8_0 | 18,487 | 203 |
| Qwen3-30B-A3B (MoE) | Q6_K | 5,807 | 209 |

See [`benchmarks/report.md`](benchmarks/report.md) for full results including llama.cpp comparison, and [`docs/comparison-llama-cpp.md`](docs/comparison-llama-cpp.md) for a detailed technical comparison.

## Features

- **Model formats:** GGUF and SafeTensors (loaded natively, no conversion step)
- **Architectures:** LLaMA, Mistral, Mixtral, DeepSeek, Qwen3, Qwen3-MoE, Nemotron-H (Mamba2 + Attention + MoE)
- **Quantization:** Q4_0, Q4_K_M, Q6_K, Q8_0, FP8 E4M3, NVFP4 (FP4 E2M1), INT8
- **Attention dispatch:** scalar Flash Attention 2 (any GPU) &rarr; WMMA (sm_90+) &rarr; TCGEN05 (sm_120+)
- **KV cache:** paged block allocation (block size 16), LRU eviction, prefix caching, FP16/FP8 storage
- **Decode optimizations:** CUDA Graphs, PDL, fused RMSNorm+Q8_1 quantize, fused QKV GEMV, dp4a-accelerated GEMV
- **Prefill optimizations:** FP16 weight cache, batched K/V GEMM, cuBLAS batched attention, fused O-projection+residual
- **Continuous batching:** scheduler with prefill/decode separation
- **Speculative decoding:** draft model + target verification with stochastic acceptance
- **Green Contexts:** SM partitioning for concurrent prefill/decode workloads

## Requirements

- NVIDIA GPU: Hopper (sm_90a) or Blackwell (sm_100, sm_120)
- CUDA Toolkit 13.1+
- CMake 3.25+
- C++20-capable compiler (GCC 11+, Clang 14+)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Build options:

| Option | Default | Description |
|--------|---------|-------------|
| `IMP_BUILD_TESTS` | ON | Build Google Test suite |
| `IMP_BUILD_TOOLS` | ON | Build imp-cli and imp-bench |
| `IMP_BUILD_BENCH` | ON | Build benchmark tool |
| `CMAKE_CUDA_ARCHITECTURES` | `90a;100;120` | Target GPU architectures |

## Usage

### CLI

```bash
# Single prompt
./build/imp-cli --model path/to/model.gguf --prompt "The capital of France is"

# Interactive chat
./build/imp-cli --model path/to/model.gguf --interactive

# Benchmark
./build/imp-cli --model path/to/model.gguf --bench --bench-pp 512 --bench-reps 3
```

Full CLI options:

```
--model <path>          Path to model file (GGUF or SafeTensors)
--prompt <text>         Input prompt
--max-tokens <n>        Maximum tokens to generate (default: 256)
--temperature <f>       Sampling temperature (default: 0.7)
--top-p <f>             Top-p nucleus sampling (default: 0.9)
--top-k <n>             Top-k sampling (default: 40)
--seed <n>              Random seed, -1 for random (default: -1)
--interactive           Interactive chat mode
--device <n>            CUDA device ID (default: 0)
--gpu-layers <n>        Layers on GPU, -1 = all (default: -1)
--ssm-fp16              FP16 SSM state (saves ~50% SSM VRAM)
--no-cuda-graphs        Disable CUDA Graph capture for decode
--chat-template <t>     auto, none, chatml, llama2, llama3, nemotron
--bench                 Synthetic benchmark mode
--bench-pp <n>          Prompt token count for benchmark (default: 512)
--bench-reps <n>        Benchmark repetitions (default: 3)
```

### C API

```c
#include <imp/imp.h>

ImpModel model;
imp_model_load("model.gguf", IMP_FORMAT_GGUF, &model);

ImpConfig cfg = imp_config_default();
cfg.enable_cuda_graphs = 1;

ImpContext ctx;
imp_context_create(model, &cfg, &ctx);

ImpGenerateParams params = imp_generate_params_default();
params.max_tokens = 128;
params.temperature = 0.7f;

char output[4096];
size_t output_len;
imp_generate(ctx, "The capital of France is", &params, output, sizeof(output), &output_len);

printf("%.*s\n", (int)output_len, output);

imp_context_free(ctx);
imp_model_free(model);
```

The C API also supports token-level control via `imp_prefill` / `imp_decode_step` for custom generation loops, and `imp_set_draft_model` for speculative decoding.

## Project Structure

```
imp/
├── include/imp/          Public C API (imp.h, config.h, types.h, error.h)
├── src/
│   ├── core/             Tensor, Buffer, Allocator, Logging, Threading
│   ├── compute/          CUDA kernels: GEMM, attention, RoPE, LayerNorm,
│   │                       activation, embedding, sampling, MoE routing
│   ├── memory/           KV cache (paged), SSM state, device/pinned allocators
│   ├── model/            Model loading (GGUF/SafeTensors), tokenizer, weight upload
│   ├── quant/            FP8, NVFP4, INT4/INT8 dequant, quantized GEMM
│   ├── graph/            GraphExecutor (hardcoded transformer forward pass)
│   ├── runtime/          Engine, Scheduler, Batch, CUDA Graphs, PDL,
│   │                       Green Contexts, Speculative Decoding
│   └── api/              C API implementation
├── tools/
│   ├── imp-cli/          CLI tool (interactive + single-prompt + benchmark)
│   └── imp-bench/        Standalone benchmarks (GEMM, attention, end-to-end)
├── tests/                Google Test suite (17 test files)
├── benchmarks/           Performance reports
└── docs/                 Technical documentation
```

## Tests

```bash
cmake -B build -DIMP_BUILD_TESTS=ON
cmake --build build -j$(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Run specific tests
./build/imp-tests --gtest_filter="TensorTest.*"
```

Tests require a compatible NVIDIA GPU. Coverage includes tensor operations, GGUF parsing, KV cache management, attention kernels (scalar + tensor core), RoPE, LayerNorm, MoE routing, quantization, FP8/NVFP4, Green Contexts, continuous batching, speculative decoding, and end-to-end generation.

## Architecture

### Inference Pipeline

1. **Model loading** — GGUF or SafeTensors parsed, weights mmap'd from disk
2. **Weight upload** — Weights dequantized (Q8_0/Q6_K &rarr; FP16) and uploaded to GPU; fused KV weight tensors created for batched prefill GEMM
3. **Forward pass** — `GraphExecutor` runs a hardcoded transformer forward pass (no graph walking at runtime)
4. **Scheduling** — `Scheduler` manages continuous batching with prefill/decode separation
5. **KV cache** — Paged block allocation with LRU eviction and prefix caching
6. **Sampling** — Temperature, top-p, top-k from FP32 logits

### Attention Dispatch

Runtime dispatch based on GPU compute capability:

| GPU | Kernel | Path |
|-----|--------|------|
| sm_120+ (Blackwell) | TCGEN05 systolic attention | `attention_blackwell.cu` |
| sm_90+ (Hopper) | WMMA tensor-core attention | `attention_tc.cu` |
| Any | Scalar Flash Attention 2 | `attention.cu` |

Prefill also supports a cuBLAS batched-GEMM attention path (`attention_cublas.cu`) for small-to-medium sequences.

### Decode Kernel Fusion

For single-token decode (n=1), the forward pass fuses multiple operations to minimize kernel launches:

- **RMSNorm + Q8_1 quantize:** single kernel, skips FP16 intermediate buffer
- **Fused QKV GEMV:** one kernel computes Q, K, V projections (dp4a-accelerated)
- **O-projection + residual:** dp4a GEMV writes `W_o @ attn_out + residual` directly

### Prefill Optimization

For multi-token prefill (n>1), weights are pre-dequantized to FP16 at init time:

- **Batched K/V GEMM:** K and V projections computed in a single `cublasGemmStridedBatchedEx` call (K/V workspace is contiguous)
- **Fused O-projection + residual:** cuBLAS `beta=1` fuses `hidden = attn_out @ W_o^T + hidden` without separate residual copy
- **FP16 cache with VRAM budget:** gracefully falls back to on-the-fly dequant when VRAM is tight

## Documentation

- [`docs/comparison-llama-cpp.md`](docs/comparison-llama-cpp.md) — Detailed technical comparison with llama.cpp
- [`docs/memory-management-comparison.md`](docs/memory-management-comparison.md) — KV cache and memory management design
- [`benchmarks/report.md`](benchmarks/report.md) — Performance benchmarks

## Acknowledgments

imp was built by [@kekzl](https://github.com/kekzl) with [Claude Code](https://claude.ai/claude-code).

imp stands on the shoulders of [llama.cpp](https://github.com/ggerganov/llama.cpp). The GGUF format, the quantization schemes, and the entire concept of practical local LLM inference were pioneered by Georgi Gerganov and the llama.cpp community. See [`docs/comparison-llama-cpp.md`](docs/comparison-llama-cpp.md) for a detailed technical comparison and full acknowledgment.

## License

MIT License. See [LICENSE](LICENSE) for details.
