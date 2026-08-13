---
layer: L0
audience: newcomers
verified: 2026-08-13
commit: 81ffa573
---

<p align="center">
  <img src="docs/logo.svg" alt="imp" width="500">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/kekzl/imp?style=flat&color=blue" alt="License"></a>
  <img src="https://img.shields.io/badge/CUDA-13.3-76b900?style=flat&logo=nvidia" alt="CUDA 13.3">
  <img src="https://img.shields.io/badge/C++-23-00599C?style=flat&logo=cplusplus" alt="C++23">
</p>

---

**imp is an LLM inference engine that targets exactly one chip: the NVIDIA RTX 5090.**

**What it is**

- A from-scratch C++23/CUDA engine for consumer Blackwell (`sm_120a`), with its own GGUF and SafeTensors loaders, tokenizer, paged KV cache and kernels.
- A server that speaks **both** the OpenAI and the Anthropic APIs natively, so an agent stack written against either runs without a shim.
- Also a C library and a CLI, not only a server.

**What it is not**

- Portable. There is no CPU path, no other GPU, no fallback.
- A multi-GPU or datacenter-batching engine.
- A supported product. One author, no SLO, no support rotation.

## Is imp for you?

| Yes, if | No, if |
|---|---|
| you run an RTX 5090 / 5080 / 5070 Ti / PRO 6000 | you have anything else, including datacenter Blackwell ([why](docs/internals/ARCHITECTURE.md)) |
| you have one GPU and one user, or an agent loop | you serve high-concurrency batched traffic |
| you want NVFP4 weights served natively, without dequant | you need a portable engine across a fleet |
| you want an Anthropic-compatible endpoint without a proxy | you need SLOs, a support contract or a security response process |

If more than one cell on the right applies to you, use
[llama.cpp](https://github.com/ggerganov/llama.cpp) for breadth or
[vLLM](https://github.com/vllm-project/vllm) for scale. Both are better at those
jobs than imp is, and imp does not try to be.

## 60-second quickstart

Everything runs in Docker. You do not need a CUDA toolkit on the host.

```bash
mkdir -p models   # drop a GGUF or a SafeTensors directory in here

docker run --gpus all -v ./models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 8080:8080 ghcr.io/kekzl/imp:latest --model /models/your-model.gguf

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model.gguf","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

Expected: a JSON completion object whose `choices[0].message.content` holds the
reply. The model id is the file or directory basename, `GET /v1/models` lists it,
and the field is required.

The cache volume is optional and pays for itself on the second start: it holds
the transformed weights (Qwen3-14B-NVFP4 init 7.9 s → 2.1 s).

Or open <http://localhost:8080> for a small built-in chat UI that streams the
answer and plots inter-token latency as it is written.

## Numbers

<!-- PERF:BEGIN -->
| metric | value | threshold |
|---|---|---|
| decode tg128 | **287.19 tok/s** | 8 % |
| prefill pp128 | 4885.13 tok/s | 8 % |
| prefill pp512 | **12406.87 tok/s** | 8 % |
| prefill pp4096 | 15324.7 tok/s | 8 % |
| peak VRAM (own) | 20716 MiB | 10 % |

[PROV: commit=1e4fad60 date=2026-07-26 hw=RTX5090 model=Qwen3-8B-Q8_0 quant=Q8_0
       cuda=13.3 path=gguf-dp4a cmd=`make verify-fast` n=5x5]
<!-- PERF:END -->

**Read the caveat before quoting these.** Decode on this host moves several
percent between sessions with nothing changed: the same tree read 287.63 one day
and 276.92 the next at healthy clocks. Prefill moves more, because cuBLAS
re-times its algo selection per process. That is why the regression thresholds
are 8 % and not 3 %.

Competitive figures per model, each with its date, commit and command, are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Methodology and what counts as a
number at all: [`docs/PERF.md`](docs/PERF.md).

## What works today

Full matrix with per-item status: [`docs/FEATURES.md`](docs/FEATURES.md).
✅ = code path plus a gated test, 🟡 = code path, no test.

| area | |
|---|---|
| **Models** | ✅ Qwen3 / 3.5 / 3.6 (dense + MoE), LLaMA, Mistral, Mixtral, DeepSeek incl. V2 latent attention, Gemma-3 and Gemma-4, gpt-oss, Nemotron-H, nomic-bert embeddings. 🟡 Llama-4 |
| **Vision** | ✅ Qwen3-VL and Qwen3.6-35B-A3B (tower in the checkpoint), Gemma-3/4 via `--mmproj`. No video |
| **Quantisation** | ✅ NVFP4 (native, the primary path), MXFP4, GGUF Q2_K–Q8_0, IQ4_NL/XS, FP8 E4M3 weights and KV, INT8/INT4 KV |
| **APIs** | ✅ OpenAI chat/completions/responses/embeddings, Anthropic `/v1/messages`, `/v1/rerank`, per-token SSE on all three dialects, tool calling, JSON-Schema / regex / GBNF constrained decoding, prefix caching with `cache_control` |
| **Serving** | ✅ continuous batching, paged KV, model swap on request, suspend/resume to free the GPU, Prometheus `/metrics`, API-key auth |
| **Engine** | ✅ NVFP4 block-scaled `mma.sync` GEMM/GEMV, FP8 `f8f6f4` attention scores, FlashAttention-2 prefill (hd 128 and 256), CUDA graphs on prefill and decode, Gated DeltaNet and Mamba2, speculative decoding |

## Where imp loses

The five that should decide against imp, in plain terms. The rest are in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

1. **One GPU, one chip.** No multi-GPU, no tensor parallelism, and no plan for
   either: consumer Blackwell has no NVLink, so TP is net-negative on the
   workload imp targets ([why](docs/DESIGN_DECISIONS.md)).
2. **No GPU in continuous integration.** Every kernel correctness and performance
   check runs on one person's desktop before a push. If that fails your risk
   model, it should.
3. **One model resident at a time.** 32 GB fits one. Serving a second is a swap,
   and the requesting call pays the load.
4. **Measurements move.** Decode varies several percent between sessions on this
   host; prefill varies more. Any single number, including ours, is one sample.
5. **Single-author project.** No support rotation, no SLO, no security response
   process.

## How it works in five minutes

A request arrives at the HTTP layer and becomes a `Request` on the **scheduler**,
which runs continuous batching: it admits as many requests as the KV budget
allows and steps them together rather than serving them one at a time.

Before a new request generates anything, its prompt has to be read. That is
**prefill**, and it is compute-bound: thousands of tokens go through the model at
once, so the GEMMs are large and the GPU is busy. If an earlier request shares a
prefix with this one, the **prefix cache** hands over the already-computed
key/value pairs and prefill only runs on the tail, which is why a multi-turn
agent loop gets cheaper as it grows.

Those key/value pairs live in the **paged KV cache**: fixed-size blocks with an
index, so a sequence does not need one contiguous allocation and blocks can be
shared between requests that share a prefix. Same idea as virtual memory.

Then **decode** starts, one token at a time, and the picture inverts. There is
only one token to process, so the arithmetic is trivial and the cost is entirely
reading the weights out of memory. Decode is bandwidth-bound, which is the single
most important fact about this engine: **making decode faster means moving fewer
bytes**, not doing less maths. That is why the weights are stored in 4 bits
(NVFP4) and read by kernels that multiply them without unpacking them to 16 bits
first. It is also why CUDA graphs matter, replaying a recorded sequence of kernel
launches instead of re-issuing hundreds of them per token.

The generated token is streamed back as it appears, not after the reply is done.

That is the whole loop: **request → scheduler → prefill (compute-bound) → paged
KV → decode (bandwidth-bound) → stream**.

*Everything above this line is the five-minute version. The real thing is in*
[`docs/internals/`](docs/internals/).

## Go deeper

| I want to … | Read |
|---|---|
| just run it | [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| put it behind a proxy, with auth and metrics | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| know which API fields actually work | [`docs/API.md`](docs/API.md) |
| know which models and quants load | [`docs/MODELS.md`](docs/MODELS.md) |
| know why it is this fast, or this slow | [`docs/PERF.md`](docs/PERF.md), [`docs/internals/BENCHMARKING.md`](docs/internals/BENCHMARKING.md) |
| fix something that went wrong | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| understand or replace a kernel | [`docs/internals/KERNELS.md`](docs/internals/KERNELS.md) |
| know why there is no multi-GPU | [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) |
| work on it as an AI agent | [`CLAUDE.md`](CLAUDE.md) |

## Build from source

Tracks `main` rather than the latest release.

```bash
git clone https://github.com/kekzl/imp.git && cd imp
docker compose build imp-server
docker run --gpus all -v ./models:/models -p 8080:8080 \
  imp:latest --model /models/your-model.gguf
```

Contributor workflow, build targets and the test lanes:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## How this was built

Every line of imp was written by an AI coding agent (Claude Code), across ~138k
lines of engine C++/CUDA plus tooling and tests. The repository keeps its own
audit trail of that: what was measured, what was refuted, and what was shelved,
in [`docs/audit/`](docs/audit/) and [`docs/MISSION_JOURNAL.md`](docs/MISSION_JOURNAL.md).

## License

MIT. See [`LICENSE`](LICENSE).
