<!--
layer: L0
audience: newcomers
verified: 2026-10-12
commit: 95a448b7
-->

<p align="center">
  <img src="docs/logo.svg" alt="imp" width="500">
</p>

<p align="center"><b>Your RTX 5090 is faster than your inference engine. imp fixes that.</b><br>
one <code>docker run</code> · OpenAI and Anthropic API · built-in chat · Linux or WSL2</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/kekzl/imp?style=flat&color=blue" alt="License"></a>
  <img src="https://img.shields.io/badge/CUDA-13.4-76b900?style=flat&logo=nvidia" alt="CUDA 13.4">
  <img src="https://img.shields.io/badge/C++-23-00599C?style=flat&logo=cplusplus" alt="C++23">
</p>

General-purpose engines run on every GPU, so none of them is built around yours. imp is written for exactly one chip, the RTX 5090 generation (`sm_120a`), and uses what that chip has: native 4-bit (FP4) math, its tensor cores, its memory bandwidth.

### Qwen3.8-27B: text and images, on one card

| Same RTX 5090, each engine on its defaults | **imp** | vLLM 0.31.0 | llama.cpp |
| --- | ---: | ---: | ---: |
| one user | **91.0-93.7 tokens/s** | 68.9-70.0 tokens/s | 61.8-63.0 tokens/s |
| 32 users at once (total) | **1963.5-1977.0 tokens/s** | 1354.8-1449.4 tokens/s | 162.0-166.8 tokens/s |

[PROV: commit=54921da6 date=2026-10-11 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 (imp, vLLM: Qwen3.8-27B-NVFP4-vllm) / Q4_K_M (llama.cpp: ggml-org Qwen3.8-27B-Q4_K_M.gguf)
       cuda=13.4.2 path=server-api cmd=`tools/analysis/engine3_conc.sh` n=2 trials x 3 waves, range of the trial medians
       note=300-token greedy answers, ignore_eos; llama.cpp image server-cuda sha256 9d835c2f]

imp and vLLM read the same NVFP4 checkpoint, llama.cpp the Q4_K_M GGUF. On that same GGUF imp leads too: **72.7-73.4** vs 60.5-61.0 tokens/s for one user, **240.0-244.4** vs 154.5-155.1 for 32 users at once.
[PROV: commit=95a448b7 date=2026-10-12 hw=RTX5090 model=Qwen3.8-27B quant=Q4_K_M (ggml-org GGUF, both engines) cuda=13.4.2 path=server-api cmd=`ARMS="imp-gguf llama-gguf" tools/analysis/engine3_conc.sh` n=2 trials x 3 waves note=same harness as above; #2688]

### Qwen3.8-Flash-Next: four times bigger than the card

The 123.6 GiB checkpoint does not fit in 32 GB. imp keeps its 56.25 GiB of experts in host RAM and caches the busiest ones on the card: **87.89-98.57 tokens/s** on one RTX 5090 with ~62 GiB of free RAM, reading a 512-token prompt at 1094.74-1096.10 tokens/s ([quickstart](#quickstart-qwen38-flash-next)).

[PROV: commit=54921da6 date=2026-10-11 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.2
       path=host-resident-experts cmd=`imp-cli --bench --bench-pp 512 --bench-reps 3 --max-tokens 128
       --bench-prompt-file novel_prose_2026.txt` n=2 processes note=real-text prompt, imp defaults; 56.25 GiB =
       load log `NVFP4 experts (56.25 GiB) do not fit`, main 62431977]

**tokens/s** = how fast the answer appears (a token is about ¾ of a word). Six more models against llama.cpp on the same GGUF file, and every other comparison: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Why imp

- **Fast where you feel it.** Answers stream faster, many users share the card (table above), long prompts are read with FP4 attention, and a repeated prompt prefix is read once.
- **Models bigger than the card.** Mixture-of-experts models keep their experts in host RAM and cache the busiest on the card (Flash-Next above).
- **Drop-in for your tools.** OpenAI and Anthropic APIs, tool calling, JSON output, streaming, thinking models, a built-in chat in the browser. Point your coding agent at it and go.
- **No conversion step.** Load the GGUF you already have, or an NVFP4 SafeTensors checkpoint straight from Hugging Face.
- **One container.** No CUDA toolkit, no Python environment on your machine: Docker plus the NVIDIA driver.
- **Free and open source** (MIT).

## Is it for you?

| You have | imp |
| --- | --- |
| RTX 5090 (32 GB, the tested one), 5080, 5070 Ti, RTX PRO 6000 | yes |
| any other GPU, CPU only, several GPUs | no: use [llama.cpp](https://github.com/ggerganov/llama.cpp) for breadth, [vLLM](https://github.com/vllm-project/vllm) for many GPUs ([who fits where](docs/PERF.md#competitive-standing)) |

imp has one author: issues are welcome, answers are not guaranteed.

> **Jump to:** [Which model?](#which-model-should-i-pick) · [Install](#install) · [Flash-Next](#quickstart-qwen38-flash-next) · [Using it](#using-it) · [Problems?](#something-went-wrong) · [How it works](#how-does-it-work) · [All the docs](#go-deeper)

## Which model should I pick?

| You want | Take | Why |
| --- | --- | --- |
| **Not sure** | **Qwen3.8-27B** NVFP4 (the [install](#install) example) | reads text and images, fits one 5090 with room for a long context |
| the fastest replies | **Qwen3-8B** Q8_0 | the fastest of the [GGUF sweep](docs/BENCHMARKS.md) |
| more knowledge, still fast | **Qwen3.6-35B-A3B** | mixture of experts: large, but only a small part works per token |
| another model family | **gpt-oss-20b** or **Gemma-4-26B-A4B** | both in the [GGUF sweep](docs/BENCHMARKS.md) |
| the biggest model, and ~62 GiB of free RAM | **Qwen3.8-Flash-Next** NVFP4 | too big for 32 GB: the experts live in host RAM ([above](#qwen38-flash-next-four-times-bigger-than-the-card)) |

**Will it fit?** Weights plus the KV cache (the model's memory of the conversation) must fit in the card's 32 GB. imp sizes the context to what is left after the weights, and refuses at load a model whose experts do not fit instead of running it wrong. Every family that loads, with its size: [`docs/MODELS.md`](docs/MODELS.md).

**Two file formats, no conversion step:** a **GGUF file** (as used by llama.cpp) or a **SafeTensors directory** in NVFP4 (NVIDIA Model Optimizer, llm-compressor, or imp's own `imp-quantize`). Your own BF16/FP8 checkpoint: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

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

**No Docker:** every release attaches `imp-<version>-linux-x86_64.tar.gz` (CUDA libraries bundled, needs only the NVIDIA driver): [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#without-docker). **Shortcut:** `--model hf://<org>/<repo>[:<file>.gguf]` downloads a repo inside the container into `/models` (resumable, checksummed, `HF_TOKEN` for gated repos): [`docs/CONFIG.md`](docs/CONFIG.md#fetching-from-hugging-face). Full walkthrough: [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

## Quickstart: Qwen3.8-Flash-Next
**You need:** RTX 5090, ~62 GiB free host RAM (the 56.25 GiB of experts plus free headroom; less RAM serves from a slower path), 123.6 GiB disk for [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
```bash
docker pull ghcr.io/kekzl/imp:latest
IMP_IMAGE=ghcr.io/kekzl/imp:latest STAGE_DIR=~/models/Qwen3.8-Flash-Next-NVFP4 scripts/stage-model.sh nvidia/Qwen3.8-Flash-Next-NVFP4 ~/models/Qwen3.8-Flash-Next-NVFP4
docker run -d --name imp-server --gpus all -v ~/models:/models -p 127.0.0.1:8090:8090 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-Flash-Next-NVFP4 --host 0.0.0.0 --port 8090
curl -s http://localhost:8090/health | jq -c '{status,kv_blocks_total,kv_capacity_tokens}'
```
**Ready** after about 64 s: `Server listening on http://0.0.0.0:8090` in `docker logs imp-server`, `/health` prints `{"status":"ok","kv_blocks_total":2025,"kv_capacity_tokens":32400}`. Image v0.46.0 or later; RAM, measurements: [`docs/QUICKSTART.md`](docs/QUICKSTART.md#qwen38-flash-next).
## Using it

- **In the browser:** <http://localhost:8080>, the built-in chat: saved conversations, sampling controls, live server metrics, GPU energy per answer ([tour](docs/QUICKSTART.md#4-or-use-the-other-surfaces)).
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
