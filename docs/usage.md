<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# imp — Usage & Reference

Build instructions, CLI/server usage, configuration, C API, project structure.

---

## Requirements

- **NVIDIA Blackwell GB202** (sm_120a) — RTX 5090, RTX PRO 5000 Blackwell, or RTX PRO 6000 Blackwell. Same binary, same kernels; the workstation cards just have more VRAM (48 / 96 GB) for bigger MoE models without expert offload.
- **CUDA Toolkit 13.3** (13.2 minimum enforced by CMake; 13.3 is the canonical toolchain Docker and CI build with) — `cudart`, `cuda_driver`, `cublas`, `cublasLt`
- **CMake 3.25+**
- **C++23 compiler** (GCC 13+, Clang 16+) — `CMAKE_CXX_STANDARD 23` is required, not a preference

CUTLASS v4.6.2 and Google Test v1.17.0 are fetched automatically via
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
make verify-fast     # build + filtered tests + perf gate + peak-VRAM gate + smoke prompt
make verify          # full pre-merge gate (~5 min)
```

| CMake option | Default | Description |
|---|---|---|
| `IMP_BUILD_TESTS` | ON | GTest suite (2125 cases across 8 binaries) |
| `IMP_BUILD_TOOLS` | ON | imp-cli |
| `IMP_BUILD_BENCH` | ON | imp-bench |
| `IMP_BUILD_SERVER` | ON | imp-server |
| `IMP_SANITIZERS` | OFF | ASAN + UBSAN (host C++ code only) |
| `IMP_ALLOC_INTERPOSE` | OFF | Wrap `cudaMalloc`/`cudaMallocAsync` to attribute steady-state allocations (diagnostic; costs ~3% decode, so never benchmark with it on) |
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
./build/imp-cli --model ./Qwen3-Coder-30B-A3B-FP4 --prompt "Hello"

# Interactive chat
./build/imp-cli --model model.gguf --interactive

# Vision — Qwen3-VL carries its tower in the checkpoint
./build/imp-cli --model ./Qwen3-VL-4B-Instruct --image photo.jpg \
                --prompt "Describe this image"

# Vision — Gemma-3 needs its encoder as a separate mmproj
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
80% cap and trades FP16 weight-cache capacity for more context. The
budget planner's envelope itself is tunable via imp.conf `[vram]`:
`kv_fraction` (default 0.8 — the KV share of post-reserve VRAM) and
`reserve_floor_pct` (default 10 — the free-VRAM headroom floor as % of
total). See `imp.conf.example`.

To serve a context longer than the model's native window, inject RoPE
scaling at load via imp.conf `[rope]` (or `--set`), e.g. a native-32k
model at 128k:

```bash
imp-server --model model.gguf --set rope.scaling=yarn --set rope.factor=4
```

The override mirrors model-declared `rope_scaling` metadata (YaRN or
linear), raises the detected context window to `factor × orig_ctx`, and
is refused for LongRoPE/llama3 per-dimension tables, MLA, and NoPE
models. Quality past the native window is the checkpoint's YaRN
extrapolation quality — validate on your workload.

`--vram-budget <mb>` (also `[runtime] vram_budget_mb` in imp.conf) hard-caps
this process's VRAM: every sizing decision — weight caches, KV clamp, expert
offload, workspaces, upload gates — sees a virtual GPU of that size, so
multiple imp-server processes can share one card:

```bash
imp-server --model Qwen3-4B-Instruct-2507-Q8_0.gguf --port 8080 --vram-budget 9000 &
imp-server --model Llama-3.2-3B-Instruct-Q8_0.gguf  --port 8081 --vram-budget 8000 &
```

The cap binds, but it is not exact — and the overshoot is measured rather than
guessed. Two charges sit outside the sizing
gates: the CUDA primary context (~1.7 GiB on this host, allocated before imp
takes its baseline snapshot, so no budget can cover it) and ~1.8 GiB of
dequant scratch, CUTLASS scale-factor buffers, pinned staging and workspaces
whose allocation sites don't consult the budget. Measured on Qwen3-8B-Q8_0,
`--vram-budget 16000` peaks at 19468 MiB — so leave ~3.5 GiB of real headroom
between the sum of budgets and the card. `--mem-report` prints the peak
against the cap and marks it `[OVER BUDGET]` when it exceeds it, so the gap is
visible rather than inferred.

Before #1109 the flag did nothing measurable: every term of the planner's
reserve is a percentage of the (virtual) card, while the cuBLAS/CUTLASS
reserve claimed on the first forward pass is a ~3.9 GiB constant, so shrinking
the budget shrank the reserve and left the constant uncovered. The reserve is
now floored at that measured charge (tune with `[vram] library_reserve_mb`).
A model whose weights don't fit the budget fails cleanly at load instead of
starving the neighbour, and a budget that cannot hold a single `max_seq_len`
sequence is refused at init — naming the blocks available, the blocks needed
and the MiB to add — rather than loading and then failing every request.

<details>
<summary>Full CLI options</summary>

```
Model:
  --model <path>            Path to GGUF or SafeTensors model
  --revision <rev>          HuggingFace revision when --model is a hub repo id
  --mmproj <path>           Vision encoder GGUF (Gemma-3/Gemma-4; Qwen3-VL
                            carries its tower in the checkpoint)
  --image <path>            Input image (needs a model with a vision tower).
                            Repeat for several images (Qwen3-VL only)
  --device <n>              CUDA device ID (default: 0)
  --gpu-layers <n>          Layers on GPU, -1 = all (default: -1)
  --config <path>           Path to imp.conf (overrides search-path)
  --set section.key=value   Per-run override (repeatable)
  --json                    One JSON document on stdout, every human line on
                            stderr (see "Machine-readable output" below)

Generation:
  --prompt <text>           Input prompt
  --max-tokens <n>          Max tokens to generate (default: 256)
  --max-seq-len <n>         KV context ceiling in tokens (default: auto)
  --min-kv-tokens <n>       Minimum KV capacity in tokens (default: auto)
  --vram-budget <mb>        Hard per-process VRAM cap in MiB (default: 0 = uncapped)
  --mem-report              Print the full VRAM attribution table at init
                            (lifecycle checkpoints, per-pool notes, named
                            charges, own_peak vs the cap, residual)
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
  --kv-fp16                 Force FP16 KV cache (opts out of the auto FP8
                            upgrade for models with a kv-FP8 author hint)
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

### Machine-readable output — `--json`

`--json` puts **exactly one JSON document on stdout** and every human line on
stderr, so a caller pipes it into `jq` instead of regexing a column layout that
is not a contract (#1583). It works on `imp-cli --bench`, `--perplexity` and
`--prompt`, and on `imp-bench`; `--interactive` refuses it, because a token
stream is not one document.

```bash
$ imp-cli --model "$MODEL" --bench --bench-pp 128 --bench-reps 1 --max-tokens 16 --json 2>/dev/null
{"mode":"bench","model":"...","prefill_tps":5502.57,"decode_tps":438.59,"pp_tokens":128,
 "pp_ms":23.26,"tg_tokens":16,"tg_ms":36.48,"reps":1,"peak_vram_mib":11188}

$ imp-bench gemm --json 2>/dev/null
{"mode":"bench-suite","requested":1,"run":1,"wall_s":3.40,
 "benchmarks":[{"name":"gemm","measured":true,"seconds":3.399}]}
```

| mode | keys |
|---|---|
| `bench` | `prefill_tps`, `decode_tps`, `pp_tokens`, `pp_ms`, `tg_tokens`, `tg_ms`, `reps`, `peak_vram_mib` |
| `perplexity` | `perplexity`, `tokens`, `corpus`, `calibration` (when `--calibrate-out`) |
| `generate` | `text`, `prompt_tokens`, `completion_tokens`, `prefill_tps`, `decode_tps`, `prefill_ms`, `decode_ms`, `total_ms` |
| `bench-suite` | `requested`, `run`, `wall_s`, `benchmarks[].{name,measured,seconds}` |

`text` is what stdout would have shown, not `decode(output_ids)`: the hidden
stop and think markers stay hidden, so the document and the terminal agree.

`imp-bench` reports per-benchmark *timings*, not the tables. The five bench
entry points return `bool`, and their numbers have no shared shape to
serialise; the consumer that needed machine-readable throughput is
`scripts/gen_perf_baseline.sh`, and it reads `imp-cli --bench --json`.

## Server — imp-server (OpenAI + Anthropic compatible)

Both GGUF and SafeTensors are accepted. `--model` is optional: without it the
server starts model-less and the first request naming a model under
`--models-dir` loads it (requests that resolve to nothing get a 503).

```bash
# Start with GGUF
./build/imp-server --model model.gguf --port 8080

# No model at startup — pick one per request from a directory
./build/imp-server --models-dir ~/models --port 8080

# Start with SafeTensors (NVFP4 prequant)
./build/imp-server --model ./Qwen3-Coder-30B-A3B-FP4 --port 8080

# With vision — Qwen3-VL needs no second file
./build/imp-server --model ./Qwen3-VL-4B-Instruct
./build/imp-server --model gemma-3-12b-it.gguf --mmproj mmproj.gguf
```

Endpoints: `/v1/chat/completions`, `/v1/responses` (OpenAI Responses API —
the Agents SDK / Codex dialect; stateless, so use `store: false` and resend
the transcript in `input`), `/v1/completions`, `/v1/embeddings`,
`/v1/models`, `/v1/messages` (Anthropic-compatible, streaming +
non-streaming), `/tokenize`, `/detokenize`, `/health`, `/props`, `/info`,
`/admin/suspend`, `/admin/resume`.
Tool/function calling, streaming usage stats, logprobs, and API-key auth
(`--api-key`) supported.

**Warm weight cache.** The first cold load of a model writes a cache file
(`<model-name>-<hash>.impwcache`) into `~/.cache/imp/warm` (override with
`[warm_cache] dir`) holding the converted weight buffers; subsequent starts
mmap it and skip the conversion work. Raw quant payloads are never
duplicated, so the cache is small for raw-served GGUF quants and
NVFP4-prequant models and ~model-size for BF16-dense checkpoints. On by
default (`[warm_cache] enabled = false` to opt out); stale caches (changed
model file) are detected and ignored — delete the directory at any time. In
containers, mount a persistent volume at the cache dir; if it is not
writable, loads simply stay cold (INFO log). Delete the cache file at any time — the next
load is simply cold and rewrites it.

**Suspend to RAM.** `POST /admin/suspend` drains in-flight requests, parks
the model weights in host RAM, and frees the GPU completely (with
`[suspend] device_reset` — the default — the CUDA context is reset too, so
`nvidia-smi` shows ~0 MiB for the process). `POST /admin/resume` restores
the weights from RAM (no mmap re-read, no requantization) and serves again
in seconds. Sessions/KV do not survive — only the weights stay warm. While
suspended, inference endpoints answer 503 and `/health` reports
`"suspended": true` (HTTP 200 — it is a deliberate operator state, not a
fault). Capture fails cleanly (507) when host `MemAvailable` is below the
snapshot size + `[suspend] host_ram_headroom_mb`; models whose device
weight buffers are transformed in place after upload (native MXFP4 GGUF,
gpt-oss, Gemma-4 fused-expert split) are refused with 501.
`/v1/models` lists the model the server is serving (OpenAI semantics: the
server exposes exactly what it can serve). Requests must name that model —
any other `model` value gets `404 model_not_found`; inference requests never
trigger a model load/swap. To switch models, restart the server with a
different `--model`.

**Context-window auto-detection.** The served context length is exposed in
the three conventions OpenAI-compatible clients already probe, so no
hard-coded table is needed: `/v1/models` carries vLLM's `max_model_len` and
llama.cpp's `meta.n_ctx_train` on the model object, `GET /props` returns the
llama.cpp `n_ctx` (top-level and under `default_generation_settings`), and
`GET /info` returns TGI's `max_total_tokens` / `max_input_tokens`. All three
report the same window, and it is what the KV pool can actually hold: the
resolver's `max_seq_len` is a plan, the pool is clamped after it, and on a tight
card the two differ (97204 against 52256 on Qwen3.8-27B-NVFP4). `/health`'s
`kv_capacity_tokens` has always been the real number; the probes report the
smaller of the two now (#1542).

Server-only flags (not on `imp-cli`):

| Flag | Effect |
|---|---|
| `--host <addr>` | Listen address (default `127.0.0.1`) |
| `--port <n>` | Listen port (default `8080`) |
| `--max-batch <n>` | Decode batch / KV+workspace sizing (default 0 = auto) |
| `--models-dir <path>` | Directory to scan for `.gguf` models (auto-load on select) |
| `--lora NAME=PATH` | Load a PEFT LoRA adapter (repeatable) — see the LoRA section below |
| `--api-key <key>` | Require `Authorization: Bearer <key>` on requests |
| `--max-concurrent <n>` | Max simultaneous requests (default 64, 0 = unlimited) |
| `--rate-limit <n>` | Max requests/min per IP (default 0 = unlimited) |
| `--log-requests <path>` | Append per-request JSONL with prompt + response content + timing to `<path>` (opt-in; off by default) |
| `--reasoning-format <f>` | `deepseek` (default) or `none` — controls `<think>` channel handling |
| `--think-budget <f>` | Fraction of `max_tokens` reserved for reasoning (default 0.5, 0 = disabled) |
| `--request-timeout <s>` | Per-request timeout in seconds (default 300, 0 = unlimited) |
| `--max-input-tokens <n>` | Reject prompts longer than n tokens with HTTP 400 (default 0 = unlimited) |
| `--prefix-cache <path>` | Persist the prefix cache to `<path>` across restarts |

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

> The C API is consumable only from a **source build**: `cmake --install` stages
> `libimp.a` and the `include/imp/` headers, which you link against. The prebuilt
> `ghcr.io/kekzl/imp` runtime image ships only the `imp-server` / `imp-cli` /
> `imp-bench` binaries — not the static library or headers — so embedding the C
> API means building from source (or copying the lib/headers out of the Docker
> `builder` stage).

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
via `imp_set_image` (pass NULL to clear) and `imp_add_image` for a second and
further picture on the Qwen3-VL tower.

## Project Structure

```
imp/
├── include/imp/          Public C API (imp.h, config.h, types.h, error.h)
├── src/
│   ├── core/             Tensor, Buffer, Allocator, Logging, Threading
│   ├── compute/          CUDA kernels (GEMM, attention, RoPE, LayerNorm, sampling, MoE)
│   ├── memory/           Driver backend, tier allocators (arena, block pool,
│   │                     scratch stack, graph slots), capacity planner,
│   │                     KV cache (paged), SSM state
│   ├── model/            Model loading (GGUF + SafeTensors), tokenizer, weight upload
│   ├── quant/            FP8, NVFP4, INT4/INT8 dequant, quantised GEMM
│   ├── exec/             GraphExecutor (hardcoded transformer forward pass)
│   ├── runtime/          Engine, Scheduler, CUDA Graphs, PDL, Green Contexts,
│   │                     RuntimeConfig (imp.conf parser)
│   ├── vision/           SigLIP + Qwen3-VL encoders, image preprocessing,
│   │                     mmproj loader, DeepStack injection
│   └── api/              C API implementation
├── tools/
│   ├── imp-cli/          CLI (interactive + single-prompt + benchmark)
│   ├── imp-server/       OpenAI + Anthropic-compatible HTTP server
│   └── imp-bench/        Standalone benchmarks
├── tests/                Google Test suite (2125 cases across 8 binaries)
└── third_party/stb/      stb_image (image loading for vision)
```

