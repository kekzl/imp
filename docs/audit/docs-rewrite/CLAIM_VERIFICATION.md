<!--
layer: L3
audience: agents
verified: 2026-08-13
commit: f7831dba
-->

# Claim verification (phase 2, validator)

Every claim the rewritten docs may make, checked against the code. Status legend
is the dispatch's §2.2:

- `✅ verified` — code path **and** a test that runs in a gate
- `🟡 implemented` — code path, no test coverage → must appear in `LIMITATIONS.md`
- `⚪ not implemented` — deliberately absent → `DESIGN_DECISIONS.md`

A claim with neither code path nor test does not survive into the docs.

**Test-lane caveat that changes what `✅` means here.** CI has no GPU runner
(`vars.HAS_GPU_RUNNER` gates the `test` job off). The CPU lane runs 1130 cases in
0.39 s without executing a single CUDA kernel. So `✅` means "a test exists and
runs in a gate", where the gate for anything GPU-shaped is `make verify-fast`
**locally, before push**, not GitHub Actions. Any doc that says "tested in CI"
about a kernel would be false.

## Model architectures

Source of truth: `src/model/model_arch.h:7` (the enum) and `src/model/model.cpp:294`
(the string map). The dispatch's §4 list is a **subset** of what ships and misses
seven families.

| architecture | enum | status | evidence |
|---|---|---|---|
| LLaMA | `LLAMA` | ✅ | `model.cpp:171`, `model.cpp:294` |
| Mistral | `MISTRAL` | ✅ | `model_arch.h:9` |
| Mixtral | `MIXTRAL` | ✅ | `model_arch.h:10` |
| DeepSeek (incl. V2 MLA) | `DEEPSEEK` | ✅ | `model.cpp:177`; MLA validated on DeepSeek-V2-Lite (#802/#803) |
| Qwen3 / Qwen3-MoE | `QWEN3`, `QWEN3_MOE` | ✅ | `model.cpp:180`; the pinned gate model is Qwen3-8B-Q8_0 |
| Qwen3.5 / Qwen3.5-MoE | `QWEN35`, `QWEN35_MOE` | ✅ | `model_arch.h:13-14`; GDN family |
| Qwen3.6-MoE | `QWEN36_MOE` | ✅ | `model_arch.h:15` |
| gpt-oss | `GPT_OSS` | ✅ | `model_arch.h:16`; MXFP4 experts, attention sinks |
| Gemma-3 (text + SigLIP vision) | `GEMMA3` | ✅ | `model.cpp:310`, `gguf_loader.cpp:406` |
| Gemma-4 | `GEMMA4` | ✅ | `model_arch.h:18` |
| Llama-4 | `LLAMA4` | 🟡 | `model_arch.h:19`; no dedicated gate found |
| Nemotron-H MoE | `NEMOTRON_H_MOE` | ✅ | `model_arch.h:11` |
| nomic-bert (encoder/embedder) | `NOMIC_BERT` | ✅ | `model_arch.h:20`, #836 |
| Phi-4 | **maps to `LLAMA`** | ✅ | `model.cpp:316` `{"phi3", ModelArch::LLAMA}` — it is not its own arch, and the docs must say so |
| Qwen3-VL | on the Qwen3 towers | ✅ | `make test-vision`; `Qwen3VLPipelineTest` |

**Correction the docs must carry:** the dispatch lists "Phi-4" as an architecture.
It is an alias onto the LLaMA path, not a separate loader. Documenting it as its
own architecture would send a reader looking for a file that does not exist.

## Quantisation formats

Source of truth: `src/core/qtype.h:14`.

| format | status | note |
|---|---|---|
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | ✅ | wire-stable GGUF block quants |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | ✅ | K-quants; Q4_K_M and Q6_K are hero quants |
| IQ4_NL, IQ4_XS | ✅ | `qtype.h:30-31` |
| BF16, F16, F32 | ✅ | |
| MXFP4 (+ `MXFP4_KV`) | ✅ | gpt-oss experts; converted to NVFP4 at load |
| FP8_E4M3 | ✅ | KV dtype and, since v0.25.0, native FP8 weights |
| FP8_E5M2 | 🟡 | `qtype.h:38`, no gate found |
| INT8, INT4 | ✅ | KV dtypes; INT4 KV applies sinks correctly and is still unusable (v0.24.0) |
| NVFP4 | ✅ | the primary weight path |

The dispatch's quant list omits IQ4_NL/IQ4_XS, MXFP4, Q2_K/Q3_K and the
`_KV`-only types. Its list is not wrong, it is incomplete.

## HTTP surface

Source of truth: route registrations in `tools/imp-server/`.

| endpoint | status |
|---|---|
| `POST /v1/chat/completions` | ✅ |
| `POST /v1/completions` | ✅ |
| `POST /v1/messages`, `/v1/messages/count_tokens` | ✅ |
| `POST /v1/responses` | ✅ |
| `POST /v1/embeddings` | ✅ |
| `POST /v1/rerank`, `POST /rerank` | ✅ `make test-rerank`, validated against llama.cpp on the same GGUF |
| `POST /tokenize`, `POST /detokenize` | ✅ |
| `GET /v1/models`, `/health`, `/metrics`, `/props`, `/info` | ✅ |
| `POST /admin/suspend`, `/admin/resume` | ✅ |
| `GET /` (web UI) | ✅ embedded at build time |

Feature-level, where the dispatch's §6.3 was stale:

| claim | status | evidence |
|---|---|---|
| SSE streaming, per token, all three dialects | ✅ | one shared driver since v0.18.1 (`handlers_chat_stream.cpp:33`); `/v1/messages` emits real `content_block_delta` (`handlers_messages.cpp:146`) |
| `/v1/messages` streaming is "synthetic" | **refuted** | see above. Remove from any limitations list |
| per-request speculative toggle | ✅ | `handlers_chat_params.cpp:246`; also bridged from the Anthropic shape (`anthropic.cpp:333`) |
| `cache_control` | ✅ | 4 sites; per-breakpoint pin boundary (#1046) |
| prefix caching | ✅ | on by default for the server since v0.12.1 |
| JSON-Schema constraining, regex, GBNF | ✅ | `make test-*`; a constraint that cannot compile is a 400 (v0.23.0) |
| tool calling | ✅ | plus external gates: aider, Claude Code, OpenAI Agents SDK |
| auth (`--api-key`), `--metrics-require-auth` | ✅ | v0.21.0 |
| C library API | ✅ | `src/api/imp_api.cpp`; errors translated to `ImpError` at that boundary |
| CLI (`imp-cli`) | ✅ | with the known limit that `--prompt` prints ~10 tokens, so byte-level A/Bs go through the server |
| logprobs | 🟡 | present in the parameter surface; no dedicated gate found |

## Kernels and runtime

| claim | status | evidence |
|---|---|---|
| NVFP4 block-scaled MMA (`mma.sync … mxf4nvf4`) | ✅ | `src/quant/nvfp4_gemm.cu`, `nvfp4_gemv_moe.cu`; 15 nvfp4 test files |
| FP8 MMA `kind::f8f6f4` for attention scores | ✅ | `attention_fmha_sm120.cu` |
| TMA bulk-tensor loads | ✅ | `gemm_grouped_nvfp4_smallM.cu:65`, "Emits UTMALDG on SM120" |
| TMA **warp-specialized grouped GEMM** | **unresolved** | `CLAUDE.md:87` and `docs/sm120.md:31` contradict each other → `OPEN_QUESTIONS.md` Q1. Docs must not claim it either way |
| Paged KV | ✅ | default block `n=16` (`engine.cpp`, `engine_kv_cache_init.cpp`); an NVFP4 run logged `block_size=32`, so the geometry is per-configuration, not a constant |
| LRU expert cache + prefix cache | ✅ | `expert_cache.h`; measured 64.3 % hit rate cold on a fully offloaded 30B |
| CUDA graphs, **decode** | ✅ | `engine_graph_decode.cpp`; the gate asserts a ≥1.3x speedup and measured 2.28x |
| CUDA graphs, **prefill** | ✅ default-on | `config.h:103` `prefill_graph = true`; Blocker B retired once `graph_capture_mode="relaxed"` became default. Disabled per-model when one NVFP4 weight exceeds the dequant-workspace cap |
| Gated DeltaNet | ✅ | `gdn_scan.cu`, `gdn_scan_tc.cu` |
| continuous batching | ✅ | `scheduler.cpp`, `engine_scheduler.cpp` |
| speculative decoding | ✅ | n-gram, suffix, MTP drafts; `test_ngram_draft.cpp`, `test_suffix_draft.cpp`, `test_mtp_forward.cpp` |
| multi-GPU / tensor parallelism | ⚪ | absent by design → `DESIGN_DECISIONS.md` |

**Stale code comment found on the way:** `src/runtime/engine_init_resolver.cpp:565`
says "prefill is never graph-captured". `runtime.prefill_graph` defaults to
`true`. The comment contradicts the default it sits next to. Logged in
`PURGE_LOG.md`; fixing it is a code change, not a docs change, so it is listed
rather than done here.

## Language violations (dispatch §2.4)

| file | line | verdict |
|---|---|---|
| `src/core/qtype.h` | 15 | **violation** — "Wire-stable 0..31 (kompatibel mit GGUF block-quant Wire-Format)" |
| `tools/analysis/degen_corpus.jsonl` | 23, 159, 230 | **not a violation** — German *prompts*, deliberate multilingual degeneration fixtures. Translating them would delete the test |

## Numbers the rewritten docs may use

Only these, because only these have a provenance chain in-tree:

| figure | source | referent |
|---|---|---|
| decode 287.19 tok/s | `tests/perf_baseline.json` | Qwen3-8B-Q8_0, tg128, spec off, RTX 5090 |
| prefill 12407 / pp4096 15325 tok/s | same | same model |
| peak VRAM 20716 MiB | same | same |
| gate thresholds 8 % decode / 8 % prefill / 10 % VRAM | `tests/perf_baseline.json`, #1400 | |
| hero sweep 2026-07-12 | `docs/BENCHMARKS.md` | per model, each named |

The dispatch's §2.7 figures are **not** in this list. They name
Qwen3-Coder-30B-A3B at 290-295 tok/s decode and "~1.4x behind vLLM"; the
in-tree 2026-07-12 sweep reads that model at 389 tok/s and `docs/roadmap.md`
records imp ahead of vLLM on MoE single-sequence prefill. Reproducing §2.7 would
publish a regression that did not happen.
