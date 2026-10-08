<!--
layer: L0
audience: newcomers
verified: 2026-09-30
commit: 0ef52509
-->

<p align="center">
  <img src="docs/logo.svg" alt="imp" width="500">
</p>

<p align="center"><b>Run open LLMs on an NVIDIA RTX 5090 at the speed the card can do</b><br>
one <code>docker run</code> · OpenAI and Anthropic API · built-in chat · Linux or WSL2</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/kekzl/imp?style=flat&color=blue" alt="License"></a>
  <img src="https://img.shields.io/badge/CUDA-13.4-76b900?style=flat&logo=nvidia" alt="CUDA 13.4">
  <img src="https://img.shields.io/badge/C++-23-00599C?style=flat&logo=cplusplus" alt="C++23">
</p>

imp is an inference engine written for exactly one chip, the RTX 5090 generation (`sm_120a`). It does not trade speed for portability: it computes on the card's native 4-bit (FP4) number format, where general engines fall back to slower paths.

- **Free and open source** (MIT).
- **One card only:** RTX 5090 (32 GB, the tested one), 5080, 5070 Ti, RTX PRO 6000. No other GPU, no CPU mode, no multi-GPU. For breadth use [llama.cpp](https://github.com/ggerganov/llama.cpp), for many GPUs [vLLM](https://github.com/vllm-project/vllm): [who fits where](docs/PERF.md#competitive-standing).
- **One author, no support rotation.** Issues are welcome, answers are not guaranteed.

> **Jump to:** [How fast?](#how-fast-is-it) · [Which model?](#which-model-should-i-pick) · [Install](#install) · [Flash-Next](#quickstart-qwen38-flash-next) · [Using it](#using-it) · [Problems?](#something-went-wrong) · [How it works](#how-does-it-work) · [All the docs](#go-deeper)

## How fast is it?

Same RTX 5090, same model file, both engines on their defaults. **tokens/s** = how fast the answer appears (a token is about ¾ of a word).

| Model | imp | llama.cpp | imp ahead by |
| --- | ---: | ---: | ---: |
| **Qwen3-8B** (Q8_0) | 385.4 tokens/s | 160.1 tokens/s | +141% |
| **Qwen3.6-35B-A3B** (Q4_K_M) | 287.9 tokens/s | 235.8 tokens/s | +22% |
| **gpt-oss-20b** (MXFP4) | 382.7 tokens/s | 335.9 tokens/s | +14% |
| **Gemma-4-26B-A4B** (Q4_K_M) | 245.0 tokens/s | 214.4 tokens/s | +14% |

[PROV: commit=83cb5178 date=2026-08-30 hw=RTX5090 model=six-model-sweep quant=per-row cuda=13.3
       path=gguf cmd=`make bench-competitive` n=6x2 note=imp defaults vs llama.cpp defaults, full
       offload, flash attention on; spec-off measured for Qwen3-8B only
       (`--set speculative.ngram=false`)]

The smallest lead in that sweep was +3% (Qwen3-30B-A3B). All rows, and imp vs vLLM with many users at once: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

**Bigger than the card: Qwen3.8-Flash-Next** (NVFP4). Its 56.25 GiB of experts stay in host RAM, the card caches the busiest ones: **65.31 tokens/s**. Full speed needs the experts plus 6 GiB free in host RAM (~62 GiB); with less, imp warns and serves from a slower path.

[PROV: commit=80c3a110 date=2026-09-30 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4
       path=host-resident-experts cmd=`scripts/accept_2272.sh` of #2326 n=4 prompts x 3 runs note=128 tokens greedy, mtp_k=0;
       56.25 GiB = load log `NVFP4 experts (56.25 GiB) do not fit`, main 62431977; 6 GiB = kHeadroom in src/model/weight_upload.cpp]

**What CI defends:** every push measures one pinned model (Qwen3-8B Q8_0, speculation off, hence below the row above) and fails on an 8% move (decode on the test host moves several percent between sessions with nothing changed: [`docs/PERF.md`](docs/PERF.md)). **decode** = writing the answer, **prefill** = reading your prompt.

<!-- PERF:BEGIN -->
| metric | value | threshold |
|---|---|---|
| decode tg128 | **282.97 tok/s** | 8 % |
| prefill pp128 | 6132.05 tok/s | 8 % |
| prefill pp512 | **14053.01 tok/s** | 8 % |
| prefill pp4096 | 16197.82 tok/s | 8 % |
| peak VRAM (own) | 20264 MiB | 10 % |

[PROV: commit=702a9cd3 date=2026-09-28 hw=RTX5090 model=Qwen3-8B-Q8_0 quant=Q8_0
       cuda=13.4 path=gguf-dp4a cmd=`make verify-fast` n=5x5]
<!-- PERF:END -->

## Which model should I pick?

| You want | Take | Why |
| --- | --- | --- |
| **Not sure** | **Qwen3.8-27B** NVFP4 (the [install](#install) example) | reads text and images, fits one 5090 with room for a long context |
| the fastest replies | **Qwen3-8B** Q8_0 | the fastest row above |
| more knowledge, still fast | **Qwen3.6-35B-A3B** | mixture of experts: large, but only a small part works per token |
| another model family | **gpt-oss-20b** or **Gemma-4-26B-A4B** | both in the table above |
| the biggest model, and ~62 GiB of free RAM | **Qwen3.8-Flash-Next** NVFP4 | too big for 32 GB: the experts live in host RAM ([above](#how-fast-is-it)) |

**Will it fit?** Weights plus the KV cache (the model's memory of the conversation) must fit in the card's 32 GB. imp sizes the context to what is left after the weights, and refuses at load a model whose experts do not fit instead of running it wrong. Every family that loads, with its size: [`docs/MODELS.md`](docs/MODELS.md).

**Two file formats, no conversion step:** a **GGUF file** (as used by llama.cpp) or a **SafeTensors directory** in NVFP4 (NVIDIA Model Optimizer, llm-compressor, or imp's own `imp-quantize`). Your own BF16/FP8 checkpoint: [`docs/quantization.md`](docs/quantization.md).

## Install

**You need:** one of the cards above, Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Linux or Windows with WSL2. No CUDA toolkit on the host. Checklist: [`docs/QUICKSTART.md`](docs/QUICKSTART.md#requirements).

```bash
make build                                                     # 1. the image: minutes the first time, seconds after
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4-vllm ~/models/Qwen3.8-27B-NVFP4-vllm   # 2. weights, 19.2 GiB download
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4-vllm   # 3. serve
```

The weights take 18.3 GiB on the card and answer at **~102 tokens/s**, leaving ~7 GiB for the conversation memory. Next time, the same `docker run`: the `imp-cache` volume keeps the converted weights, so a restart skips the conversion.

[PROV: commit=f243179c date=2026-08-31 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4
       cuda=13.3 path=nvfp4-safetensors n=2 image=ghcr.io/kekzl/imp:latest
       cmd=`imp-cli --model … --prompt … --max-tokens 128 --temperature 0` (102.8/101.9 tok/s)]

**Shortcut:** `--model hf://<org>/<repo>[:<file>.gguf]` downloads a repo inside the container into `/models` (resumable, checksummed, `HF_TOKEN` for gated repos): [`docs/CONFIG.md`](docs/CONFIG.md#fetching-from-hugging-face). Full walkthrough: [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

## Quickstart: Qwen3.8-Flash-Next
**You need:** RTX 5090, ~62 GiB free host RAM ([why](#how-fast-is-it)), 123.6 GiB disk for [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
```bash
docker pull ghcr.io/kekzl/imp:latest
IMP_IMAGE=ghcr.io/kekzl/imp:latest STAGE_DIR=~/models/Qwen3.8-Flash-Next-NVFP4 scripts/stage-model.sh nvidia/Qwen3.8-Flash-Next-NVFP4 ~/models/Qwen3.8-Flash-Next-NVFP4
docker run -d --name imp-server --gpus all -v ~/models:/models -p 127.0.0.1:8090:8090 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-Flash-Next-NVFP4 --host 0.0.0.0 --port 8090
curl -s http://localhost:8090/health | jq -c '{status,kv_blocks_total,kv_capacity_tokens}'
```
**Ready** after 97.7 s: `Server listening on http://0.0.0.0:8090` in `docker logs imp-server`, `/health` prints `{"status":"ok","kv_blocks_total":1900,"kv_capacity_tokens":30400}`. Image v0.46.0 or later; RAM, measurements: [`docs/QUICKSTART.md`](docs/QUICKSTART.md#qwen38-flash-next).
## Using it

- **In the browser:** <http://localhost:8080>, the built-in chat.
- **Your apps and coding agents:** an "OpenAI-compatible" provider with base URL **`http://localhost:8080/v1`**; Anthropic apps: `http://localhost:8080/v1/messages`. The model id is the file or directory name (`GET /v1/models`).
- **Tool calling, JSON output, streaming** on every API dialect; which fields work: [`docs/API.md`](docs/API.md).
- **Thinking models:** the reasoning comes back in `reasoning_content`, the answer in `content`.
- **Proxy, API key, metrics:** [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). **Without the server:** `imp-cli` in the image, or the C library (`include/imp/imp.h`).

## Something went wrong?

| You see | Do |
| --- | --- |
| replies much slower than the table above | something else is on the card: on WSL2 `nvidia-smi` does not show containers, check `docker ps`; then look for `graphs=1` in the log |
| after a restart every prompt is cancelled, `/health` says `kv_pool_floored` | the old process still held the card at start: free the card, start again |
| `503` | no model loaded or suspended: `GET /v1/models`, `POST /admin/resume` |
| empty `content`, full `reasoning_content` | the model used its budget thinking: raise `max_tokens` |
| nothing arrives until the answer is done | a reverse proxy buffers the stream: `proxy_buffering off` (nginx) |
| `400 vision_unavailable` on an image | the checkpoint loaded text-only; which models see: [`docs/MODELS.md`](docs/MODELS.md) |

**Still stuck?** The [full table](docs/TROUBLESHOOTING.md), or an issue with the server log.

## How does it work?

- **One chip, no compromises for others.** The RTX 5090 multiplies 4-bit numbers (NVFP4) directly; imp keeps weights in that format and computes on it, where portable engines unpack to 16 bits or skip the format.
- **Everything is its own:** file loaders, tokenizer, the conversation memory (a paged KV cache shared between requests), the GPU kernels. No framework underneath.
- **Replay, not re-launch.** Each answer step is recorded once as a CUDA graph and replayed, so the CPU does not hand the GPU thousands of small jobs per token.
- **Guess, then check.** A cheap drafter proposes the next tokens, the model checks them all in one pass and keeps what it agrees with: same answer, fewer passes.
- **Many users at once:** requests share the card (continuous batching), a repeated prompt prefix is read once (prefix caching).

What exists and what is tested: [`docs/FEATURES.md`](docs/FEATURES.md). The whole design: [`docs/internals/ARCHITECTURE.md`](docs/internals/ARCHITECTURE.md). Every doc by layer and reader: [`docs/README.md`](docs/README.md).

## Go deeper

| I want to … | Read |
|---|---|
| run it, step by step | [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| know which API fields work, which models load | [`docs/API.md`](docs/API.md), [`docs/MODELS.md`](docs/MODELS.md) |
| know why it is this fast, or this slow | [`docs/PERF.md`](docs/PERF.md), [`docs/internals/BENCHMARKING.md`](docs/internals/BENCHMARKING.md) |
| understand or replace a kernel | [`docs/internals/KERNELS.md`](docs/internals/KERNELS.md) |
| know where imp loses, and why no multi-GPU | [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) |
| build from source (`docker compose build imp-server`, tracks `main`), or contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| work on it as an AI agent | [`CLAUDE.md`](CLAUDE.md) |

## How this was built

Every line of imp was written by an AI coding agent (Claude Code). The repository keeps its own audit trail: [`docs/audit/`](docs/audit/), [`docs/MISSION_JOURNAL.md`](docs/MISSION_JOURNAL.md).

## License

MIT. See [`LICENSE`](LICENSE). One kernel file is adapted from Apache-2.0 code (SageAttention); the APA prefill (`src/compute/apa/`) is [kekzl/apa](https://github.com/kekzl/apa) under MPL-2.0. Licence texts and notices are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), shipped in the image at `/usr/share/doc/imp/`.
