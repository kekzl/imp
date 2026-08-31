<!--
layer: L0
audience: newcomers
verified: 2026-08-30
commit: 83cb5178
-->

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

Everything runs in Docker. You do not need a CUDA toolkit on the host. The
worked example is **Qwen3.8-27B**: a 27B multimodal model, quantized to NVFP4 so
it fits a single 5090 with room for a real context.

**Get the weights** (19.2 GiB, download only, nothing to convert). The
checkpoint is [kekzle/Qwen3.8-27B-NVFP4-vllm](https://huggingface.co/kekzle/Qwen3.8-27B-NVFP4-vllm),
an `imp-quantize --format vllm` export: the same directory also loads in
vLLM (verified on 0.27.1). `make build`
first: the script needs the `imp:test` image and exits 1 without it, even on
this download-only path (#1682). `docker compose build imp-server` produces
`imp:latest`, which is a different tag.

```bash
make build                                     # ~3.5 min, produces imp:test
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4-vllm ~/models/Qwen3.8-27B-NVFP4-vllm
```

**Serve it and ask.** This is the 60 seconds:

```bash
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4-vllm

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-27B-NVFP4-vllm","messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":64}'
```

Expected: a JSON completion object whose `choices[0].message.content` explains
Rayleigh scattering. The model id is the file or directory basename,
`GET /v1/models` lists it, and the field is required. A GGUF file works the same
way: swap the path.

The weights take 18.3 GiB on the card and answer at **~102 tok/s**, leaving
~7 GiB for the KV cache on a 32 GB 5090.

[PROV: commit=f243179c date=2026-08-31 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4
       cuda=13.3 path=nvfp4-safetensors n=2 image=ghcr.io/kekzl/imp:latest
       cmd=`imp-cli --model … --prompt … --max-tokens 128 --temperature 0` (102.8/101.9 tok/s)]

### Bringing your own model

That checkpoint was made with the in-tree quantizer, and the same command builds
one from any BF16 or FP8 release. Qwen3.8-27B needs it: no published NVFP4 export
of it runs here, because they keep attention in FP8 and this card has no kernel
for that.

```bash
scripts/stage-model.sh Qwen/Qwen3.8-27B-FP8 ~/models/my-Qwen3.8-NVFP4
```

28.8 GiB down, ~25 minutes of conversion, 18.8 GiB out. The FP8 release is a
third of the BF16 download and costs 0.24 % perplexity against converting the
BF16 one. The script forecasts the output size and whether it fits the card
*before* writing anything, so a model that would not fit costs seconds rather
than half an hour. Details and the quality numbers:
[`docs/quantization.md`](docs/quantization.md).

The cache volume is optional and pays for itself on the second start: it holds
the transformed weights (Qwen3-14B-NVFP4 init 7.9 s → 2.1 s).

Or open <http://localhost:8080> for a small built-in chat UI that streams the
answer and plots inter-token latency as it is written.

## How fast is it, really

Same card, same GGUF, same flags, decode tok/s. imp v0.33.0 against llama.cpp
(image pinned by digest, `ghcr.io/ggml-org/llama.cpp@sha256:c49f4d48…`),
measured 2026-08-30 on one RTX 5090:

| model | imp | llama.cpp | |
|---|---:|---:|---:|
| Qwen3-8B Q8_0 | **385.4** | 160.1 | **+141 %** |
| Qwen3-14B Q6_K | **162.5** | 114.8 | **+42 %** |
| Qwen3.6-35B-A3B UD-Q4_K_M | **287.9** | 235.8 | **+22 %** |
| gpt-oss-20b MXFP4 | **382.7** | 335.9 | **+14 %** |
| Qwen3-30B-A3B Q4_K_M | 305.5 | 295.7 | +3 % |

[PROV: commit=83cb5178 date=2026-08-30 hw=RTX5090 model=six-model-sweep
       quant=per-row cuda=13.3 path=gguf cmd=`make bench-competitive` n=6x2
       note2=sixth row Gemma-4-26B-A4B UD-Q4_K_M 245.0 vs 214.4, +14 %, omitted here for length
       note=imp defaults vs llama.cpp defaults, full offload, flash attention on]

The last row is the honest bottom of the range, not an outlier we forgot to
delete: on a MoE that is already bandwidth-bound at batch 1 there is little left
to win. imp's default enables n-gram speculation, which is most of the Qwen3-8B
figure; with it off that row reads 284.6, still +78 %.

**The 32 GB is the other half of the story.** NVFP4 KV means a 27B model holds
**126 432 tokens of context** on this card, where 8-bit KV stops at 86 848. At a
77k prompt it decodes 74.3 tok/s, or 100.2 with sparse decode switched on.

[PROV: commit=3921547d date=2026-08-30 hw=RTX5090 model=Qwen3.8-27B-NVFP4
       quant=NVFP4 cuda=13.3 path=nvfp4-kv cmd=`imp-server --max-batch 8` + a
       client that caps max_tokens so both arms emit the same count n=3x3
       note=sparse arm adds `--set attention.sparse_topk_tokens=8192`; it trades
       retrieval accuracy for the speed, see docs/MODELS.md]

Per-model history with dates, commits and exact commands:
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Methodology and what counts as a
number at all: [`docs/PERF.md`](docs/PERF.md).

### What CI defends

The table above is a sweep. The number the regression gate pins on every push is
a different one, on one model:

<!-- PERF:BEGIN -->
| metric | value | threshold |
|---|---|---|
| decode tg128 | **287.19 tok/s** | 8 % |
| prefill pp128 | 4885.13 tok/s | 8 % |
| prefill pp512 | **12406.87 tok/s** | 8 % |
| prefill pp4096 | 15324.7 tok/s | 8 % |
| peak VRAM (own) | 20716 MiB | 10 % |

[PROV: commit=unknown date=2026-07-26 hw=RTX5090 model=Qwen3-8B-Q8_0 quant=Q8_0
       cuda=unknown path=gguf-dp4a cmd=`make verify-fast` n=5x5]
<!-- PERF:END -->

Decode on this host moves several percent between sessions with nothing changed
(the same tree read 287.63 one day and 276.92 the next at healthy clocks), and
prefill moves more because cuBLAS re-times its algo selection per process. That
is why the thresholds are 8 % and not 3 %.

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
docker run --gpus all -v ./models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 \
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
