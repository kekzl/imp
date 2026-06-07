# imp — Usage & Reference

Build instructions, CLI/server usage, configuration, C API, project structure.

---

## Requirements

- **NVIDIA Blackwell GB202** (sm_120a) — RTX 5090, RTX PRO 5000 Blackwell, or RTX PRO 6000 Blackwell. Same binary, same kernels; the workstation cards just have more VRAM (48 / 96 GB) for bigger MoE models without expert offload.
- **CUDA Toolkit 13.3** (13.2 minimum enforced by CMake; 13.3 is the canonical toolchain Docker and CI build with) — `cudart`, `cuda_driver`, `cublas`, `cublasLt`
- **CMake 3.25+**
- **C++20 compiler** (GCC 11+, Clang 14+)

CUTLASS v4.5.1 and Google Test v1.17.0 are fetched automatically via
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
| `CMAKE_CUDA_ARCHITECTURES` | hard-pinned `sm_120a` | RTX 5090 / RTX PRO 6000 |

`sm_120a` SASS + `compute_120f` PTX fallback are set via raw `--generate-code`
in `CMakeLists.txt` (CMake < 3.31 workaround). Don't override
`CMAKE_CUDA_ARCHITECTURES`.

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
  --revision <rev>          HuggingFace revision when --model is a hub repo id
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
  --kv-nvfp4                NVFP4 KV cache (FP4 + E4M3 scales, 25% of FP16)
  --kv-mxfp4                MXFP4-KV cache (FP4 + UE8M0 scales, 25% of FP16)
  --kv-fp16                 Force FP16 KV cache (the current default)
  (--kv-turboquant{,-lite}  DEPRECATED post-PR #251; aliased to --kv-mxfp4)
  --prefill-fp8             FP8 weight cache for prefill
  --prefill-chunk-size <n>  Max tokens per prefill chunk (default: 0)
  --decode-nvfp4            NVFP4 decode cache (FP16 prefill + NVFP4 decode)
  --decode-nvfp4-only       NVFP4 decode-only (saves VRAM, slower prefill)
  --no-nvfp4                Disable NVFP4 auto-detection
  --ssm-fp16                FP16 SSM state
  --no-cuda-graphs          Disable CUDA Graphs
  --mxfp4-prefill           CUTLASS MXFP4 GEMM for prefill
  --prefix-caching          Enable prefix caching in the CLI engine
  --mtp-spec-decode <k>     MTP speculative decoding with K draft tokens
                            (models with a native MTP head only)
  --streaming-kv            Streaming-KV attention (sinks + sliding window)
  --no-streaming-kv-auto    Disable streaming-KV auto-enable heuristic
  --stream-sinks <n>        Streaming-KV: number of attention-sink tokens
  --stream-window <n>       Streaming-KV: sliding-window size in tokens

Benchmark / eval:
  --bench                   Synthetic benchmark mode (warmup + timed reps)
  --bench-pp <n>            Prompt tokens (default: 512)
  --bench-reps <n>          Repetitions (default: 3)
  --perplexity <file>       Teacher-forced perplexity over a text file
                            (deterministic eval harness, PR #481)
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

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/models`, `/v1/messages` (Anthropic-compatible, streaming +
non-streaming), `/tokenize`, `/detokenize`, `/health`. Tool/function
calling, streaming usage stats, logprobs, and API-key auth
(`--api-key`) supported.
`/v1/models` lists the model the server is serving (OpenAI semantics: the
server exposes exactly what it can serve). Requests must name that model —
any other `model` value gets `404 model_not_found`; inference requests never
trigger a model load/swap. To switch models, restart the server with a
different `--model`.

Server-only flags (not on `imp-cli`):

| Flag | Effect |
|---|---|
| `--api-key <key>` | Require `Authorization: Bearer <key>` on requests |
| `--max-concurrent <n>` | Max simultaneous requests (default 64, 0 = unlimited) |
| `--rate-limit <n>` | Max requests/min per IP (default 0 = unlimited) |
| `--log-requests <path>` | Append per-request JSONL with prompt + response content + timing to `<path>` (opt-in; off by default) |
| `--reasoning-format <f>` | `deepseek` (default) or `none` — controls `<think>` channel handling |
| `--think-budget <f>` | Fraction of `max_tokens` reserved for reasoning (default 0.5, 0 = disabled) |

```bash
# The model id is the served model's name (= /v1/models data[0].id)
MODEL=$(curl -s http://localhost:8080/v1/models | jq -r '.data[0].id')

# OpenAI chat completion
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":64}"

# Streaming
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":true}"
```

Works with the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
model = client.models.list().data[0].id
for chunk in client.chat.completions.create(
    model=model, messages=[{"role": "user", "content": "Hi"}],
    stream=True, max_tokens=64
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### LoRA adapters (hot-swap)

PEFT adapters are applied as runtime low-rank deltas on the activation path —
no weight patching, so they compose with every quant tier (FP16 cache, NVFP4
decode cache, raw-GGUF dp4a). Load at startup, select per request:

```bash
imp-server --model base.gguf --lora style=/adapters/style --lora med=/adapters/med
```

```jsonc
// any /v1/chat/completions or /v1/completions body:
{ "model": "base.gguf", "lora": "style", "messages": [...] }
// "lora" absent or "" = base model; unknown names → 400.
```

Swapping re-captures decode CUDA graphs on the next request (~100 ms) —
adapters are engine-global between requests (single-user semantics, imp's
batch=1 mission). v1 scope: per-layer `q/k/v/o/gate/up/down_proj` adapters on
standard pre-norm archs; sandwich-norm o/down (Gemma) and MoE-expert targets
are declined with a log. C API: `imp_lora_load()` / `imp_lora_set()`.

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

Token-level control via `imp_prefill` / `imp_decode_step`, vision
via `imp_set_image`.

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
│   ├── exec/             GraphExecutor (hardcoded transformer forward pass)
│   ├── runtime/          Engine, Scheduler, CUDA Graphs, PDL, Green Contexts,
│   │                     RuntimeConfig (imp.conf parser)
│   ├── vision/           SigLIP encoder, image preprocessing, mmproj loader
│   └── api/              C API implementation
├── tools/
│   ├── imp-cli/          CLI (interactive + single-prompt + benchmark)
│   ├── imp-server/       OpenAI + Anthropic-compatible HTTP server
│   └── imp-bench/        Standalone benchmarks
├── tests/                Google Test suite (~700 tests across 8 binaries)
└── third_party/stb/      stb_image (image loading for vision)
```

