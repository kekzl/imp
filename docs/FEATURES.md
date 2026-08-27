<!--
layer: L1
audience: operators
verified: 2026-08-28
commit: be825e4a
-->

# Feature matrix

**Single source of truth.** The README shows a generated extract; nothing else
states what imp supports.

Legend:

- ✅ **verified** — code path plus a test that runs in a gate
- 🟡 **implemented** — code path, no test coverage. Every 🟡 also appears in [`LIMITATIONS.md`](LIMITATIONS.md)
- ⚪ **not implemented** — deliberately absent, with the reasoning in [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md)

**What "verified" means here.** CI has no GPU runner: the `test` job is gated
behind `vars.HAS_GPU_RUNNER`, and the lane that runs executes in under a
second without launching a CUDA kernel. Current case count:
`python3 tools/check_test_lanes.py --report` (a literal here was 248 low
within nine days, #1673). For anything GPU-shaped the gate is
`make verify-fast`, locally, before push. No document in this repo should
claim CI tests the kernels.

## Model architectures

Source: `src/model/model_arch.h`, `src/model/model.cpp`.

| architecture | status | note |
|---|---|---|
| LLaMA, Mistral, Mixtral | ✅ | |
| Phi-4 | 🟡 | **an alias onto the LLaMA path** (`model.cpp:316`), not a separate loader. No checkpoint of its own in any gate (#1680) |
| DeepSeek, incl. V2 multi-head latent attention | ✅ | validated on DeepSeek-V2-Lite; latent-KV decode is opt-in |
| Qwen3, Qwen3-MoE | ✅ | the pinned gate model is Qwen3-8B-Q8_0 |
| Qwen3.5, Qwen3.5-MoE | ✅ | Gated DeltaNet family |
| Qwen3.6-MoE | ✅ | |
| gpt-oss | ✅ | MXFP4 experts, learned attention sinks |
| Gemma-3 (text + SigLIP vision) | ✅ | |
| Gemma-4 | ✅ | |
| Nemotron-H MoE | ✅ | |
| nomic-bert (encoder / embeddings) | ✅ | bidirectional, no KV, mean-pooled |
| Qwen3-VL | ✅ | `make test-vision` (gemma-3-4b-vl, Qwen3-VL-4B-Instruct) |
| Qwen3.6-35B-A3B on the same tower | 🟡 | shares the tower; `make test-vision` runs neither checkpoint against it (#1680) |
| Llama-4 | 🟡 | arch exists, no dedicated gate |

Per-checkpoint detail, including what each one needs at load, is in
[`MODELS.md`](MODELS.md).

## Quantisation

Source: `src/core/qtype.h`.

| format | status |
|---|---|
| Q4_0, Q8_0 | ✅ |
| Q4_1, Q5_0, Q5_1 | 🟡 | dequant path, no gate reads a checkpoint in these (#1680) |
| Q4_K, Q5_K, Q6_K | ✅ |
| Q2_K, Q3_K, Q8_K | 🟡 | dequant path, no gate reads a checkpoint in these (#1680) |
| IQ4_NL, IQ4_XS | ✅ |
| F32, F16, BF16 | ✅ |
| NVFP4 (two-level block scaling) | ✅ the primary weight path |
| MXFP4 | ✅ gpt-oss experts, converted to NVFP4 at load |
| FP8 E4M3 | ✅ KV dtype, and native FP8 weights since v0.25.0 |
| FP8 E5M2 | 🟡 type exists, no gate |
| INT8 / INT4 KV | ✅ / see limitations |

## Serving

| feature | status | note |
|---|---|---|
| OpenAI `/v1/chat/completions`, `/v1/completions` | ✅ | |
| Anthropic `/v1/messages`, `/v1/messages/count_tokens` | ✅ | native, no shim |
| OpenAI Responses `/v1/responses` | ✅ | the dialect Codex and the Agents SDK speak |
| SSE streaming, per token, all three dialects | ✅ | one shared driver since v0.18.1 |
| `/v1/embeddings` | ✅ | |
| `/v1/rerank` (Cohere/Jina/vLLM shape) | 🟡 | the llama.cpp cross-check is opt-in behind `COMPARE_URL=` (`Makefile:290-291`), so the default gate does not run it (#1680) |
| `/tokenize`, `/detokenize`, `/v1/models`, `/health`, `/metrics`, `/props`, `/info` | ✅ | |
| `/admin/suspend`, `/admin/resume` | 🟡 | frees the GPU in seconds, resumes without re-reading weights. No gate exercises it (#1680) |
| model swap on request (`server.model_swap`) | 🟡 | in-flight generations drain, never cancelled. No gate exercises it (#1680) |
| prefix caching, `cache_control` per breakpoint | ✅ | on by default for the server |
| tool calling | ✅ | gated by real aider, Claude Code and OpenAI Agents SDK runs |
| constrained decoding: JSON Schema, regex, GBNF | ✅ | an uncompilable constraint is a 400, not a free-text answer |
| per-request speculative toggle | ✅ | |
| `reasoning_effort` body field | ✅ | passed to the chat template verbatim; legal values are the template's, not ours. On Qwen3.8-27B the three it accepts render three different prompts (11 / 41 / 53 prompt tokens for medium / low / xhigh on the same message) |
| auth (`--api-key`), `--metrics-require-auth` | ✅ | |
| embedded web UI at `GET /` | ✅ | |
| logprobs | ✅ non-streaming | `tests/test_server_logprobs.py` (in `make test-server`, not in CI): at temperature 0 the emitted token IS `top_logprobs[0]` and shares its logprob. The streaming path emits none, #1588 |
| C library API, CLI | ✅ | |

## Engine

| feature | status | note |
|---|---|---|
| NVFP4 block-scaled `mma.sync` GEMM/GEMV | ✅ | |
| FP8 `f8f6f4` attention scores | ✅ | |
| TMA bulk-tensor loads | ✅ | `gemm_grouped_nvfp4_smallM.cu:65`, emits UTMALDG |
| FlashAttention-2 style prefill, hd=128 and hd=256 | ✅ | default since #930 |
| Paged KV cache | ✅ | default block `n=16`; the geometry is per-configuration |
| Growable KV pool | ✅ opt-in | `kv_cache.growable`: reserves address space for the planned pool, commits what the card can spare, grows on demand. Needs CUDA virtual memory management |
| CUDA graphs, decode | ✅ | gate asserts ≥1.3x, measured 2.28x |
| CUDA graphs, prefill | ✅ | default on; disabled per-model when one NVFP4 weight exceeds the dequant-workspace cap |
| continuous batching, concurrent decode | ✅ | |
| speculative decoding: n-gram, suffix, MTP | ✅ | economics differ per model, see limitations |
| Gated DeltaNet / Mamba2 hybrids | ✅ | |
| MoE host offload for GGUF experts | ✅ | see the figure in [`PERF.md`](PERF.md#moe-host-offload) |
| MoE host offload for NVFP4 experts | 🟡 | served from the expert cache; no automatic gate, see [`LIMITATIONS.md`](LIMITATIONS.md) |
| multi-GPU, tensor parallelism | ⚪ | |
| non-Blackwell GPUs, CPU inference | ⚪ | |
