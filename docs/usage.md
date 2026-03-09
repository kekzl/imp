# imp — Usage & Reference

Detailed build instructions, CLI/server usage, C API, project structure, and architecture reference.

---

## Requirements

- NVIDIA GPU: Blackwell (sm_120, sm_100) or Hopper (sm_90a)
- CUDA Toolkit 13.1+
- CMake 3.25+
- C++20 compiler (GCC 11+, Clang 14+)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

| Option | Default | Description |
|---|---|---|
| `IMP_BUILD_TESTS` | ON | Google Test suite |
| `IMP_BUILD_TOOLS` | ON | imp-cli |
| `IMP_BUILD_BENCH` | ON | imp-bench |
| `IMP_BUILD_SERVER` | ON | imp-server |
| `CMAKE_CUDA_ARCHITECTURES` | `90a;100;120` | Target GPU architectures |

## CLI

```bash
# Single prompt
./build/imp-cli --model model.gguf --prompt "Hello, world!"

# Interactive chat
./build/imp-cli --model model.gguf --interactive

# Vision (Gemma-3)
./build/imp-cli --model gemma-3-12b-it.gguf --mmproj mmproj.gguf \
  --image photo.jpg --prompt "Describe this image"

# FP8 KV cache (halves KV memory)
./build/imp-cli --model model.gguf --kv-fp8 --interactive

# NVFP4 decode cache
./build/imp-cli --model model.gguf --decode-nvfp4 --interactive

# Benchmark (matches llama-bench methodology)
./build/imp-cli --model model.gguf --bench --bench-pp 512 --max-tokens 128 --bench-reps 5
```

<details>
<summary>Full CLI options</summary>

```
Model:
  --model <path>            Path to GGUF or SafeTensors model
  --mmproj <path>           Vision encoder GGUF for multimodal
  --image <path>            Input image (requires --mmproj)
  --device <n>              CUDA device ID (default: 0)
  --gpu-layers <n>          Layers on GPU, -1 = all (default: -1)

Generation:
  --prompt <text>           Input prompt
  --max-tokens <n>          Max tokens to generate (default: 256)
  --interactive             Interactive chat mode
  --stop <str>              Stop sequence (repeatable, up to 4)
  --chat-template <t>       auto|none|chatml|llama2|llama3|nemotron|gemma|deepseek_r1|phi

Sampling:
  --temperature <f>         (default: 0.7)
  --top-p <f>               (default: 0.9)
  --top-k <n>               (default: 40)
  --min-p <f>               (default: 0.0, disabled)
  --typical-p <f>           (default: 1.0, disabled)
  --repeat-penalty <f>      (default: 1.0, disabled)
  --repeat-last-n <n>       Penalty window (default: 0, all tokens)
  --frequency-penalty <f>   (default: 0.0)
  --presence-penalty <f>    (default: 0.0)
  --seed <n>                -1 for random (default: -1)
  --dry-multiplier <f>      DRY penalty scale (default: 0.0, disabled)
  --dry-base <f>            DRY exponential base (default: 1.75)
  --dry-allowed-length <n>  (default: 2)
  --dry-penalty-last-n <n>  (default: 0, all)
  --mirostat <n>            0=off, 2=v2 (default: 0)
  --mirostat-tau <f>        (default: 5.0)
  --mirostat-eta <f>        (default: 0.1)

Performance:
  --kv-fp8                  FP8 E4M3 KV cache
  --kv-int8                 INT8 KV cache
  --prefill-fp8             FP8 weight cache for prefill
  --prefill-chunk-size <n>  Max tokens per prefill chunk (default: 0)
  --decode-nvfp4            NVFP4 decode cache (FP16 prefill + NVFP4 decode)
  --decode-nvfp4-only       NVFP4 decode-only (saves VRAM, slower prefill)
  --no-nvfp4                Disable NVFP4 auto-detection
  --ssm-fp16                FP16 SSM state
  --no-cuda-graphs          Disable CUDA Graphs

Benchmark:
  --bench                   Synthetic benchmark mode (warmup + timed reps)
  --bench-pp <n>            Prompt tokens (default: 512)
  --bench-reps <n>          Repetitions (default: 3)
```

</details>

## Server (OpenAI-compatible)

```bash
# Start
./build/imp-server --model model.gguf --port 8080

# With vision
./build/imp-server --model gemma-3-12b-it.gguf --mmproj mmproj.gguf

# Chat completion
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

Works with the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
for chunk in client.chat.completions.create(
    model="imp", messages=[{"role": "user", "content": "Hi"}],
    stream=True, max_tokens=64
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Supports `/v1/chat/completions`, `/v1/completions`, `/tokenize`, `/detokenize`, `/health`, tool/function calling, streaming usage stats, logprobs, and API key auth (`--api-key`).

## C API

```c
#include <imp/imp.h>

ImpModel model;
imp_model_load("model.gguf", IMP_FORMAT_GGUF, &model);

ImpConfig cfg = imp_config_default();
ImpContext ctx;
imp_context_create(model, &cfg, &ctx);

ImpGenerateParams params = imp_generate_params_default();
params.max_tokens = 128;

char output[4096];
size_t output_len;
imp_generate(ctx, "The capital of France is", &params, output, sizeof(output), &output_len);
printf("%.*s\n", (int)output_len, output);

imp_context_free(ctx);
imp_model_free(model);
```

Token-level control via `imp_prefill`/`imp_decode_step`, speculative decoding via `imp_set_draft_model`, vision via `imp_set_image`.

## Project Structure

```
imp/
├── include/imp/          Public C API (imp.h, config.h, types.h, error.h)
├── src/
│   ├── core/             Tensor, Buffer, Allocator, Logging, Threading
│   ├── compute/          CUDA kernels (GEMM, attention, RoPE, LayerNorm, sampling, MoE)
│   ├── memory/           KV cache (paged), SSM state, device/pinned allocators
│   ├── model/            Model loading (GGUF/SafeTensors), tokenizer, weight upload
│   ├── quant/            FP8, NVFP4, INT4/INT8 dequant, quantized GEMM
│   ├── graph/            GraphExecutor (hardcoded transformer forward pass)
│   ├── runtime/          Engine, Scheduler, CUDA Graphs, PDL, Green Contexts,
│   │                       Speculative Decoding
│   ├── vision/           SigLIP encoder, image preprocessing, mmproj loader
│   └── api/              C API implementation
├── tools/
│   ├── imp-cli/          CLI (interactive + single-prompt + benchmark)
│   ├── imp-server/       OpenAI-compatible HTTP server
│   └── imp-bench/        Standalone benchmarks
├── tests/                Google Test suite (26 files, 289 tests)
└── third_party/stb/      stb_image (image loading for vision)
```

## Tests

```bash
cmake -B build -DIMP_BUILD_TESTS=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

Covers: tensor ops, GGUF parsing, KV cache, attention (scalar + WMMA), RoPE, LayerNorm, MoE, quantization, FP8/NVFP4, Green Contexts, continuous batching, speculative decoding, end-to-end generation.

## Architecture

### Inference Pipeline

1. **Load** — GGUF/SafeTensors parsed, weights mmap'd
2. **Upload** — dequantize to FP16, upload to GPU, fuse KV weights for batched GEMM
3. **Forward** — `GraphExecutor` runs hardcoded transformer forward (no graph walking)
4. **Schedule** — continuous batching with prefill/decode separation
5. **KV cache** — paged blocks, LRU eviction, prefix caching
6. **Sample** — temperature, top-p/k, min-p, DRY, Mirostat from FP32 logits

### Attention Dispatch

| GPU | Prefill | Decode |
|---|---|---|
| sm_120+ (Blackwell) | CUTLASS Hopper FMHA | WMMA 8-warp |
| sm_90+ (Hopper) | CUTLASS Hopper FMHA | WMMA 4-warp |
| Any | cuBLAS | Scalar Flash Attention 2 |
