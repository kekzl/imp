# imp — Usage & Reference

Build instructions, CLI/server usage, configuration, C API, project structure.

---

## Requirements

- **NVIDIA Blackwell GB202** (sm_120f) — RTX 5090, RTX PRO 5000 Blackwell, or RTX PRO 6000 Blackwell. Same binary, same kernels; the workstation cards just have more VRAM (48 / 96 GB) for bigger MoE models without expert offload.
- **CUDA Toolkit 13.2+** — `cudart`, `cuda_driver`, `cublas`, `cublasLt`
- **CMake 3.25+**
- **C++20 compiler** (GCC 11+, Clang 14+)

CUTLASS v4.4.2 and Google Test v1.14.0 are fetched automatically via
`FetchContent`. `stb_image` and `stb_image_resize2` are vendored in
`third_party/stb/`.

## Build

The canonical workflow is Docker via the Makefile (`make build` →
`imp:test`). Host builds also work when CUDA 13.2+ is installed natively.

```bash
# Host build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Docker build (canonical)
make build           # → imp:test image with full GPU passthrough
make verify-fast     # build + filtered tests + perf gate + smoke prompt (~90 s)
make verify          # full pre-merge gate (~5 min)
```

| CMake option | Default | Description |
|---|---|---|
| `IMP_BUILD_TESTS` | ON | GTest suite (~700 tests across 8 binaries) |
| `IMP_BUILD_TOOLS` | ON | imp-cli |
| `IMP_BUILD_BENCH` | ON | imp-bench |
| `IMP_BUILD_SERVER` | ON | imp-server |
| `IMP_SANITIZERS` | OFF | ASAN + UBSAN (host C++ code only) |
| `CMAKE_CUDA_ARCHITECTURES` | hard-pinned `sm_120f` | RTX 5090 only |

`sm_120f` is set via raw `--generate-code=arch=compute_120f,code=sm_120` in
`CMakeLists.txt` (CMake < 3.31 workaround for the family-feature target).
Don't override `CMAKE_CUDA_ARCHITECTURES`.

## Configuration — `imp.conf`

`imp.conf` is the runtime configuration interface (PR #72). It replaces
~50 former `IMP_*` environment variables with a sectioned TOML-subset file.
See `imp.conf.example` in the repo root for the full schema with defaults
and inline comments.

**Loading precedence** (first non-empty wins):

1. `--config <path>` CLI flag
2. `$IMP_CONFIG` environment variable
3. `./imp.conf` (working directory)
4. `~/.config/imp/imp.conf`
5. embedded defaults (no file)

**Per-run overrides** on top of the loaded config:

```bash
imp-cli --set kv_cache.dtype=fp8 --set runtime.cuda_graphs=never \
        --model X.gguf --prompt "..."
```

The most common keys are also exposed as named CLI flags (`--kv-fp8`,
`--no-cuda-graphs`, …) for convenience.

## CLI — imp-cli

```bash
# Single prompt (GGUF)
./build/imp-cli --model model.gguf --prompt "Hello, world!"

# SafeTensors directory (NVFP4 prequant from Model Optimizer or llm-compressor)
./build/imp-cli --model ./Qwen3-Coder-30B-A3B-FP4/ --prompt "Hello"

# Interactive chat
./build/imp-cli --model model.gguf --interactive

# Vision (Gemma-3)
./build/imp-cli --model gemma-3-12b-it.gguf --mmproj mmproj.gguf \
                --image photo.jpg --prompt "Describe this image"

# FP8 KV cache (halves KV memory; opt-in per model — default is FP16 since PR #51)
./build/imp-cli --model model.gguf --kv-fp8 --interactive

# NVFP4 decode cache
./build/imp-cli --model model.gguf --decode-nvfp4 --interactive

# Long-context prompt (trade weight-cache VRAM for KV headroom)
./build/imp-cli --model gemma-4-26B-A4B-it-Q4_K_M.gguf \
                --min-kv-tokens 14000 --prompt "$(cat long.txt)"

# Benchmark (matches llama-bench methodology)
./build/imp-cli --model model.gguf --bench --bench-pp 512 \
                --max-tokens 128 --bench-reps 5
```

Format auto-detection: directories containing `model.safetensors` or
`model.safetensors.index.json` load as SafeTensors. Everything else loads
as GGUF.

`--max-seq-len` and `--min-kv-tokens` control KV-cache VRAM reservation.
Auto defaults target ~60% of free VRAM for KV, sized for the actual KV
dtype after model-specific overrides (e.g. Gemma-4 → FP16 KV via the
`engine.cpp:547` carve-out). `--min-kv-tokens` overrides the defensive
80% cap and trades FP16 weight-cache capacity for more context.

<details>
<summary>Full CLI options</summary>

```
Model:
  --model <path>            Path to GGUF or SafeTensors model
  --mmproj <path>           Vision encoder GGUF for multimodal
  --image <path>            Input image (requires --mmproj)
  --device <n>              CUDA device ID (default: 0)
  --gpu-layers <n>          Layers on GPU, -1 = all (default: -1)
  --config <path>           Path to imp.conf (overrides search-path)
  --set section.key=value   Per-run override (repeatable)

Generation:
  --prompt <text>           Input prompt
  --max-tokens <n>          Max tokens to generate (default: 256)
  --max-seq-len <n>         KV context ceiling in tokens (default: auto)
  --min-kv-tokens <n>       Minimum KV capacity in tokens (default: auto)
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

Performance:
  --kv-fp8                  FP8 E4M3 KV cache (opt-in; default FP16 since PR #51)
  --kv-int8                 INT8 KV cache
  --kv-int4                 INT4 KV cache (quality cost; long-ctx only)
  --kv-turboquant           PolarQuant + QJL (long-ctx only)
  --kv-fp16                 Force FP16 KV cache (the current default)
  --prefill-fp8             FP8 weight cache for prefill
  --prefill-chunk-size <n>  Max tokens per prefill chunk (default: 0)
  --decode-nvfp4            NVFP4 decode cache (FP16 prefill + NVFP4 decode)
  --decode-nvfp4-only       NVFP4 decode-only (saves VRAM, slower prefill)
  --no-nvfp4                Disable NVFP4 auto-detection
  --ssm-fp16                FP16 SSM state
  --no-cuda-graphs          Disable CUDA Graphs
  --ngram-spec              N-gram speculative decoding (draft from history)
  --ngram-spec-k <n>        Max draft tokens per step (default: 5)
  --mxfp4-prefill           CUTLASS MXFP4 GEMM for prefill

Benchmark:
  --bench                   Synthetic benchmark mode (warmup + timed reps)
  --bench-pp <n>            Prompt tokens (default: 512)
  --bench-reps <n>          Repetitions (default: 3)
```

</details>

## Server — imp-server (OpenAI + Anthropic compatible)

`--model` is required at startup. Both GGUF and SafeTensors are accepted.

```bash
# Start with GGUF
./build/imp-server --model model.gguf --port 8080

# Start with SafeTensors (NVFP4 prequant)
./build/imp-server --model ./Qwen3-Coder-30B-A3B-FP4/ --port 8080

# With vision
./build/imp-server --model gemma-3-12b-it.gguf --mmproj mmproj.gguf
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/v1/messages` (Anthropic-compatible, streaming + non-streaming),
`/tokenize`, `/detokenize`, `/health`. Tool/function calling, streaming
usage stats, logprobs, and API-key auth (`--api-key`) supported.
`/v1/models` lists available GGUF and SafeTensors models in the models
directory.

```bash
# OpenAI chat completion
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
imp_generate(ctx, "The capital of France is", &params,
             output, sizeof(output), &output_len);
printf("%.*s\n", (int)output_len, output);

imp_context_free(ctx);
imp_model_free(model);
```

Token-level control via `imp_prefill` / `imp_decode_step`, speculative
decoding via `imp_set_draft_model`, vision via `imp_set_image`.

## Project Structure

```
imp/
├── include/imp/          Public C API (imp.h, config.h, types.h, error.h)
├── src/
│   ├── core/             Tensor, Buffer, Allocator, Logging, Threading
│   ├── compute/          CUDA kernels (GEMM, attention, RoPE, LayerNorm, sampling, MoE)
│   ├── memory/           KV cache (paged), SSM state, device/pinned allocators
│   ├── model/            Model loading (GGUF + SafeTensors), tokenizer, weight upload
│   ├── quant/            FP8, NVFP4, INT4/INT8 dequant, quantised GEMM
│   ├── graph/            GraphExecutor (hardcoded transformer forward pass)
│   ├── runtime/          Engine, Scheduler, CUDA Graphs, PDL, Green Contexts,
│   │                     Speculative Decoding, RuntimeConfig (imp.conf parser)
│   ├── vision/           SigLIP encoder, image preprocessing, mmproj loader
│   └── api/              C API implementation
├── tools/
│   ├── imp-cli/          CLI (interactive + single-prompt + benchmark)
│   ├── imp-server/       OpenAI + Anthropic-compatible HTTP server
│   └── imp-bench/        Standalone benchmarks
├── tests/                Google Test suite (~700 tests across 8 binaries)
└── third_party/stb/      stb_image (image loading for vision)
```

## Tests

```bash
make test-unit             # CPU-only filter (~5 s)
make test-gpu              # full CUDA suite (~30 s)
make test-e2e              # real-model E2E (Qwen3-4B, Qwen3.5-4B GDN, Gemma-4)
make bench                 # full benchmark suite across baseline models
```

Covers: tensor ops, GGUF + SafeTensors parsing, KV cache, attention
(paged FP16/FP8/INT4 + FMHA FP16/FP8/MXFP4), RoPE, LayerNorm, MoE
(legacy + CUTLASS 3.x grouped), quantisation, FP8/NVFP4, Green Contexts,
continuous batching, n-gram speculative decoding, end-to-end generation
including NVFP4 prequant from both Model Optimizer and llm-compressor.

## Architecture

### Inference pipeline

1. **Load** — GGUF or SafeTensors parsed, weights mmap'd. SafeTensors
   BF16 → FP16; NVFP4 prequant scales (`weight_scale`, `weight_scale_2`)
   uploaded as separate sidecars.
2. **Upload** — weights dequantised / converted and uploaded to GPU.
3. **Forward** — `GraphExecutor` runs a hardcoded transformer forward
   (no graph walking at runtime).
4. **Schedule** — continuous batching with prefill / decode separation.
5. **KV cache** — paged blocks (`block_size = 16` tokens), LRU eviction,
   prefix caching. Default FP16 since PR #51; FP8/INT8/INT4/NVFP4/TurboQuant
   opt-in.
6. **Sample** — temperature, top-p/k, min-p, typical-p, repetition / DRY /
   Mirostat from FP32 logits.

### Attention dispatch (sm_120f only)

Runtime dispatch with no architecture checks (the build is sm_120f-only).

| Phase | Path |
|---|---|
| Prefill | MXFP4 FMHA (if enabled) → FP8 FMHA → FP16 FMHA → Blackwell WMMA 128×64 |
| Decode | Paged attention with split-K (FP16 / FP8 / INT4) |

Overrides via `imp.conf`:

- `attention.mxfp4 = "always"` — force MXFP4 prefill FMHA
- `attention.fp8_fmha = "never"` — force FP16 instead of FP8 FMHA
- `attention.fmha_sm120 = "never"` — force WMMA fallback

Per-run via `--set`. Legacy `IMP_MXFP4_ATTENTION` / `IMP_NO_FP8_FMHA` /
`IMP_NO_FMHA_SM120` env vars still exist as dev escape hatches in
`attention_dispatch.cu`, but `imp.conf` is the supported interface.
