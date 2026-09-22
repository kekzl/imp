<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# Configuration reference

`imp.conf`, CLI flags (`imp-cli` and `imp-server`), `--json`, LoRA and the C API.

- Endpoints and request fields: [`API.md`](API.md).
- Deployment/env vars: [`DEPLOYMENT.md`](DEPLOYMENT.md).

## `imp.conf`

Sectioned TOML-subset config file (PR #72), replacing ~50 former `IMP_*` env vars. Full schema with defaults: `imp.conf.example` in the repo root; source of truth: `src/runtime/config.h`.

**Loading precedence** (first non-empty wins): `--config <path>` CLI flag -> `$IMP_CONFIG` env var -> `./imp.conf` -> `~/.config/imp/imp.conf` -> embedded defaults.

**Per-run overrides** on top of the loaded config:

```bash
imp-cli --set kv_cache.dtype=fp8 --set runtime.cuda_graphs=never \
        --model X.gguf --prompt "..."
```

An unknown key in `imp.conf` is a warning (a config file may outlive the build that understood every key). **An unknown key in `--set` / `IMP_SET` is an error**, not a warning - a typo used to measure the default silently.

Keys most often touched, with defaults (`src/runtime/config.h`, `src/core/config/*.h`):

| key | default | effect |
|---|---|---|
| `runtime.max_seq_len` | `0` (auto) | KV context ceiling in tokens |
| `runtime.max_batch_size` | `0` (auto) | decode batch / KV+workspace sizing |
| `runtime.think_answer_reserve` | `256` | tokens reserved for the answer after forced reasoning close |
| `kv_cache.dtype` | `auto` | widest saving measured safe for the family; `auto`/`fp16`/`fp8`/`int8`/`int4`/`nvfp4`/`mxfp4` |
| `kv_cache.growable` | `true` | pool grows at admission instead of committing the full plan at start |
| `kv_cache.growable_initial_pct` | `25` | % of the planned pool committed at startup when growable |
| `vram.kv_fraction` (imp.conf `[vram]` section) | `0.8` | KV share of post-reserve VRAM |
| `vram.reserve_floor_pct` | `10` | free-VRAM headroom floor as % of total |
| `vram.library_reserve_mb` | `-1` (auto) | floor for the measured cuBLAS/CUTLASS first-forward charge (~3.9 GiB observed); since #1109 the planner's reserve is floored at it - before that, shrinking the budget shrank the reserve and left the constant uncovered |
| `rope.scaling` (imp.conf `[rope]` section) | `""` (off) | `yarn` \| `linear`, runtime RoPE-scaling override, mirrors the model's declared `rope_scaling` metadata |
| `rope.factor` | `1.0` | context-extension factor; raises the detected window to `factor x orig_ctx`, e.g. `4.0` stretches 32k -> 128k |
| `rope.orig_ctx` | `0` (auto) | native training context the factor applies to |
| `moe.expert_cache_budget_pct` | `0` (auto) | host-expert cache budget; only relevant when MoE experts don't fit, see [`PERF.md`](PERF.md) |
| `attention.sparse_topk_tokens` | `0` (off) | sparse attention token budget |
| `server.model_swap` | `true` | a request naming another model in the directory swaps to it |
| `server.model_swap_drain_ms` | `60000` | drain budget for in-flight generations during a swap/shutdown |
| `server.agent_scan_limit` | `256` | tokens held back to scan for `<think>` when tools are present (8 without tools) |
| `server.otlp_endpoint` | `""` (off) | OTLP/HTTP traces URL |
| `server.otlp_service_name` | `imp-server` | OpenTelemetry service name |
| `warm_cache.enabled` | `true` | persist transformed weight uploads across restarts |
| `warm_cache.dir` | `""` (`$XDG_CACHE_HOME/imp/warm` or `~/.cache/imp/warm`) | warm weight cache directory |
| `suspend.device_reset` | `true` | `cudaDeviceReset()` on `/admin/suspend` so the process reads ~0 MiB |
| `suspend.host_ram_headroom_mb` | `2048` | host RAM the suspend snapshot must leave free |

RoPE example, serving past the model's native window:

```bash
imp-server --model model.gguf --set rope.scaling=yarn --set rope.factor=4
```

Refused (logged) for LongRoPE/llama3 per-dimension frequency tables, MLA (mscale entanglement) and NoPE models. Quality past the native window is the checkpoint's own YaRN extrapolation quality.

## CLI - `imp-cli`

Format auto-detection: a directory with `model.safetensors`/`model.safetensors.index.json` loads as SafeTensors; everything else loads as GGUF.

| flag | default | notes |
|---|---|---|
| `--model <path>` | required | GGUF file, SafeTensors directory, or HF repo id |
| `--revision <rev>` | - | HF revision when `--model` is a hub repo id |
| `--mmproj <path>` | - | vision encoder GGUF (Gemma-3/4; Qwen3-VL carries its tower in the checkpoint) |
| `--image <path>` | - | repeatable (Qwen3-VL: several images) |
| `--device <n>` | `0` | CUDA device id |
| `--gpu-layers <n>` | `-1` (all) | layers on GPU |
| `--config <path>`, `--set sec.key=val` | - | see `imp.conf` above |
| `--json` | off | one JSON document on stdout, human lines on stderr - see `--json` below |
| `--prompt <text>` / `--prompt-file <path>` | - | mutually exclusive; `--prompt-file` reads the whole file verbatim (argv is capped at ~128 KiB, ~32k tokens) |
| `--max-tokens <n>` | `8192` (`16384` with `--interactive`) | max tokens to generate; same default as `imp-server` |
| `--max-seq-len <n>` / `--min-kv-tokens <n>` | auto | KV context ceiling / floor in tokens |
| `--vram-budget <mb>` | `0` (uncapped) | hard per-process VRAM cap, see below |
| `--mem-report` | off | full VRAM attribution table at init |
| `--interactive` | off | interactive chat; refuses `--json` (a token stream is not one document) |
| `--stop <str>` | - | repeatable, up to 4 |
| `--chat-template <t>` | `auto` | `auto\|none\|chatml\|llama2\|llama3\|nemotron\|gemma\|deepseek_r1\|phi` |
| `--temperature <f>` / `--top-p <f>` / `--top-k <n>` | `0.7` / `0.9` / `40` | |
| `--min-p <f>` / `--typical-p <f>` | `0.0` (off) / `1.0` (off) | |
| `--repeat-penalty <f>` / `--repeat-last-n <n>` | `1.0` (off) / `0` (all) | |
| `--frequency-penalty <f>` / `--presence-penalty <f>` | `0.0` / `0.0` | |
| `--seed <n>` | `-1` (random) | |
| `--dry-multiplier <f>` / `--dry-base <f>` / `--dry-allowed-length <n>` / `--dry-penalty-last-n <n>` | `0.0` (off) / `1.75` / `2` / `0` (all) | DRY n-gram penalty |
| `--mirostat <n>` / `--mirostat-tau <f>` / `--mirostat-eta <f>` | `0` (off, `2`=v2) / `5.0` / `0.1` | |
| `--kv-fp8` / `--kv-fp16` / `--kv-int8` / `--kv-int4` / `--kv-nvfp4` / `--kv-mxfp4` | FP16 since PR #51 | opt-in KV cache dtype; FP8 is not safe on every model (Mistral-Small-3.1 Q6_K, DeepSeek-R1, Qwen3.5-GDN Q8_0, Gemma-4 all hit a stride bug) |
| `--prefill-fp8` / `--no-fp8-prefill` | auto | FP8 E4M3 weight cache for ~2x prefill throughput / disable it (pair with `--kv-fp16 --no-nvfp4` for a full-FP16 path) |
| `--prefill-chunk-size <n>` | `0` (per-arch) | max tokens per prefill chunk |
| `--decode-nvfp4` / `--decode-nvfp4-only` / `--no-nvfp4` | auto | mode 1 additive (FP8 prefill + NVFP4 decode) / mode 2 replacement (NVFP4-only) / disable NVFP4 auto-detection |
| `--dual-path-quant` | off | FP8 attention + NVFP4 FFN |
| `--mxfp4-prefill` | off | CUTLASS MXFP4 GEMM for prefill |
| `--ssm-fp16` | off | FP16 SSM `h_state` (saves ~50% SSM VRAM) |
| `--no-cuda-graphs` | off | disable CUDA Graph capture for decode |
| `--prefix-caching` | off | reuse KV blocks for shared token prefixes (CLI engine) |
| `--mtp-spec-decode <k>` | off | MTP drafting, chain length k (checkpoint needs a native MTP head) |
| `--streaming-kv` / `--no-streaming-kv-auto` | off / off | enable StreamingLLM (sinks + window) / disable its auto-enable at >90% KV full |
| `--stream-sinks <n>` / `--stream-window <n>` | `4` / model's `sliding_window` | |
| `--token-trace` | off | `[tok=ID 'piece']` marker on stderr for every generated token, not just the first ten |
| `--bench` / `--bench-pp <n>` / `--bench-reps <n>` | off / `512` / `3` | synthetic benchmark mode (matches llama-bench methodology) |
| `--perplexity <file>` / `--calibrate <out>` | - | teacher-forced perplexity over a text file (deterministic eval harness, PR #481); `--calibrate` also writes activation-calibration stats (input for `imp-quantize --calib`) |

KV-cache VRAM reservation: `--max-seq-len`/`--min-kv-tokens` control it; auto targets ~60% of free VRAM for KV, sized for the actual KV dtype after model-specific overrides (Gemma-4 -> FP16 KV via the `engine.cpp:500` carve-out). `--min-kv-tokens` overrides the defensive 80% cap, trading FP16 weight-cache capacity for more context - e.g. for a long-context prompt: `--min-kv-tokens 14000 --prompt "$(cat long.txt)"`.

`--vram-budget <mb>` (also `[runtime] vram_budget_mb`) hard-caps this process's VRAM: every sizing decision (weight caches, KV clamp, expert offload, workspaces, upload gates) sees a virtual GPU of that size, so multiple `imp-server` processes can share one card.

- The cap binds but is not exact: the CUDA primary context (~1.7 GiB on the reference host) and ~1.8 GiB of dequant scratch/CUTLASS scale buffers/pinned staging sit outside the sizing gates.
- Measured on Qwen3-8B-Q8_0: `--vram-budget 16000` peaks at 19468 MiB, leave ~3.5 GiB of headroom between the sum of budgets and the card.
- `--mem-report` prints the peak against the cap and marks `[OVER BUDGET]`.
- A budget too small for one `max_seq_len` sequence is refused at init, naming the blocks available, needed and the MiB to add.

## `--json` - machine-readable output

One JSON document on stdout, every human line on stderr (#1583). Works on `imp-cli --bench`, `--perplexity`, `--prompt`, and `imp-bench`; `--interactive` refuses it.

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

`text` is what stdout would have shown, not `decode(output_ids)`: hidden stop/think markers stay hidden. `imp-bench` reports per-benchmark timings, not tables; machine-readable throughput comes from `imp-cli --bench --json` (what `scripts/gen_perf_baseline.sh` reads).

## Server flags (`imp-server` only, not on `imp-cli`)

Both GGUF and SafeTensors accepted; `--model` is optional (model-less start, first request naming a model under `--models-dir` loads it; unresolved names get 503). Per-request caps (`--max-n`, `--max-batch-items`, `--max-logit-bias`, `--max-images-per-request`, HTTP timeouts) and auth flags: [`DEPLOYMENT.md`](DEPLOYMENT.md#auth-and-exposure).

| flag | default | effect |
|---|---|---|
| `--host <addr>` / `--port <n>` | `127.0.0.1` / `8080` | listen address/port |
| `--max-batch <n>` | `0` (auto) | decode batch / KV+workspace sizing |
| `--models-dir <path>` | - | directory to scan for `.gguf` models (auto-load on select) |
| `--lora NAME=PATH` | - | load a PEFT LoRA adapter, repeatable - see LoRA below |
| `--api-key <key>` | - | require `Authorization: Bearer <key>` |
| `--max-concurrent <n>` | `64` (`0`=unlimited) | max simultaneous requests |
| `--rate-limit <n>` | `0` (unlimited) | max requests/min per IP |
| `--log-requests <path>` | off | append per-request JSONL with prompt + response content + timing |
| `--reasoning-format <f>` | `deepseek` | `deepseek` or `none` - controls `<think>` channel handling |
| `--think-budget <f>` | `0.5` (`0`=off) | fraction of `max_tokens` a reasoning model may spend thinking; force-closed at `max_tokens - max(reserve, max_tokens/4)`, paired with `runtime.think_answer_reserve` |
| `--request-timeout <s>` | `300` (`0`=unlimited) | per-request timeout |
| `--max-input-tokens <n>` | `0` (unlimited) | reject longer prompts with 400; also gates `/tokenize`, `/detokenize`, `/v1/messages/count_tokens` |
| `--prefix-cache <path>` | off | persist the prefix cache across restarts |

`--vram-budget` (above) lets several `imp-server` processes share one card:

```bash
imp-server --model Qwen3-4B-Instruct-2507-Q8_0.gguf --port 8080 --vram-budget 9000 &
imp-server --model Llama-3.2-3B-Instruct-Q8_0.gguf  --port 8081 --vram-budget 8000 &
```

```bash
MODEL=$(curl -s http://localhost:8080/v1/models | jq -r '.data[0].id')
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":64}"
```

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

## LoRA adapters (hot-swap)

PEFT adapters apply as runtime low-rank deltas on the activation path (no weight patching), so they compose with every quant tier. Load at startup, select per request:

```bash
imp-server --model base.gguf --lora style=/adapters/style --lora med=/adapters/med
```

```jsonc
{ "model": "base.gguf", "lora": "style", "messages": [...] }
// "lora" absent or "" = base model; unknown names -> 400.
```

One adapter is active at a time: a request naming a different one waits for in-flight requests, then the worker switches (decode graphs re-capture, ~100 ms); it is never batched with them.

- The prefix cache is keyed by adapter.
- v1 scope: per-layer `q/k/v/o/gate/up/down_proj` adapters on standard pre-norm archs.
- Sandwich-norm o/down (Gemma) and MoE-expert targets are declined with a log.
- C API: `imp_lora_load()` / `imp_lora_set()`.

## C API

Source build only: `cmake --install` stages `libimp.a` and `include/imp/` headers. The prebuilt `ghcr.io/kekzl/imp` runtime image ships only the `imp-server`/`imp-cli`/`imp-bench` binaries (copy the static lib and headers out of the Docker `builder` stage if needed).

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

Token-level control: `imp_prefill` / `imp_decode_step`. Vision: `imp_set_image` (NULL clears it), `imp_add_image` for further pictures on the Qwen3-VL tower.

## Project structure

```
imp/
├── include/imp/       Public C API (imp.h, config.h, types.h, error.h)
├── src/
│   ├── core/           Tensor, Buffer, Allocator, Logging, Threading
│   ├── compute/        CUDA kernels (GEMM, attention, RoPE, LayerNorm, sampling, MoE)
│   ├── memory/         Driver backend, tier allocators, capacity planner, KV cache, SSM state
│   ├── model/          Model loading (GGUF + SafeTensors), tokenizer, weight upload
│   ├── quant/          FP8, NVFP4, INT4/INT8 dequant, quantised GEMM
│   ├── exec/           GraphExecutor (hardcoded transformer forward pass)
│   ├── runtime/        Engine, Scheduler, CUDA Graphs, PDL, Green Contexts, RuntimeConfig
│   ├── vision/         SigLIP + Qwen3-VL encoders, preprocessing, mmproj loader
│   └── api/            C API implementation
├── tools/
│   ├── imp-cli/        CLI (interactive + single-prompt + benchmark)
│   ├── imp-server/     OpenAI + Anthropic-compatible HTTP server
│   └── imp-bench/      Standalone benchmarks
├── tests/              Google Test suite, 8 module binaries
└── third_party/stb/    stb_image (image loading for vision)
```
