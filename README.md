<!--
layer: L0
audience: newcomers
verified: 2026-10-27
commit: 48f0058b
-->

<p align="center">
  <img src="docs/logo.svg" alt="imp" width="500">
</p>

<p align="center"><b>Built for Blackwell. Nothing else.</b><br>
RTX 5050 to 5090 · RTX PRO 2000 to 6000 · one <code>docker run</code> · OpenAI and Anthropic API · built-in chat · Linux or WSL2</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/kekzl/imp?style=flat&color=blue" alt="License"></a>
  <img src="https://img.shields.io/badge/CUDA-13.4-76b900?style=flat&logo=nvidia" alt="CUDA 13.4">
  <img src="https://img.shields.io/badge/C++-23-00599C?style=flat&logo=cplusplus" alt="C++23">
</p>

**imp is an LLM server for one GPU generation only: NVIDIA Blackwell `sm_120a`, every GeForce RTX 50 and RTX PRO Blackwell card.** It is for people who run models on their own single card and point chat apps and coding agents at it (OpenAI and Anthropic API: Claude Code, Codex, any OpenAI client). Built and tuned on one RTX 5090, it uses what that chip has: native 4-bit (FP4) math, its tensor cores, its memory bandwidth. One container, no CUDA toolkit or Python on your machine, MIT.

<p align="center"><img src="docs/assets/demo.gif" width="760" alt="terminal: the 123.6 GiB Qwen3.8-Flash-Next streams an answer on one 32 GB RTX 5090, with tokens per second"></p>

### Measured on one RTX 5090

| Qwen3.8-27B, tokens/s | **imp** | vLLM 0.31.0 | llama.cpp |
| --- | ---: | ---: | ---: |
| one user | **96.3** | 69.9 | |
| 32 users at once (total) | **1979.1** | 1414.2 | |
| one user, same Q4_K_M GGUF file | **77.1** | | 62.9 |
| 32 users at once, same GGUF file | **1365.2** | | 178.7 |

<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 (imp, vLLM: Qwen3.8-27B-NVFP4-vllm) / Q4_K_M (imp, llama.cpp: ggml-org GGUF) cuda=13.4.3 path=server-api
       cmd=`tools/analysis/engine3_conc.sh` n=3 trials x 3 waves note=imp ghcr v0.59.0; one-user imp row speculative.mtp_k=0 ngram=false; MTP k=2 on both: `tools/analysis/mtp_vs_vllm.sh` 3 starts x 12 chat requests; docs/audit/launch_2026_10_27/] -->

Server API, 300-token greedy answers to short prompts, median of 3 fresh-server trials, each engine on its defaults; imp and vLLM read the same NVFP4 checkpoint, imp and llama.cpp the same GGUF file. No speculative decoding in the one-user rows (imp NVFP4: switched off; imp GGUF: its default drafts nothing, 77.7 with all of it forced off). With the model's own MTP draft head on in both engines (imp: `--max-batch 1`), 6 real chat prompts: imp **126.49**, vLLM 116.84; on short template completions the two are within noise. Ranges, commands, raw logs: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#qwen38-27b-three-engines). **tokens/s** = how fast the answer appears (a token is about ¾ of a word).

**Bigger than the card:** the 123.6 GiB **Qwen3.8-Flash-Next** answers at **92.1 tokens/s** for one user on a 32 GB RTX 5090, its experts in host RAM ([how](#models-larger-than-vram)).
<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.3 path=host-resident-experts cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` n=3 server starts x 6 prompts] -->

### Where imp loses, and what it is not

- **One chip.** Tested on the RTX 5090; every other `sm_120a` card (RTX 5050 to 5090 D V2, RTX PRO 2000 to 6000 Blackwell) runs the same kernels, VRAM decides the model ([which card](docs/QUICKSTART.md#which-card)). No multi-GPU, no CPU, no other GPU: use [llama.cpp](https://github.com/ggerganov/llama.cpp) for breadth, [vLLM](https://github.com/vllm-project/vllm) for many GPUs.
- **Dense models under heavy load:** vLLM led Qwen3-14B at 32 users (-7.6 %, 2026-09-02); prompt reading on dense GGUF is level with llama.cpp, not ahead ([`BENCHMARKS.md`](docs/BENCHMARKS.md)).
- **Models bigger than the card** depend on host RAM, PCIe and how often the topic changes the set of busy experts.
- **One maintainer, no GPU in CI:** every GPU test and benchmark runs on the maintainer's RTX 5090; issues are welcome, answers are not guaranteed. Every gap with its number: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

> **Jump to:** [Which model?](#which-model-should-i-pick) · [Install](#install) · [Bigger than VRAM](#models-larger-than-vram) · [Using it](#using-it) · [How it works](#how-does-it-work) · [All the docs](#go-deeper)

## Which model should I pick?

| You want | Take | Why |
| --- | --- | --- |
| **Not sure** | **Qwen3.8-27B** NVFP4 (the [install](#install) example) | reads text and images, fits a 32 GB card with room for a long context (24 GB: short context) |
| the fastest replies, or a 12-16 GB card | **Qwen3-8B** Q8_0 (8 GB card: **Qwen3-4B** Q8_0) | the fastest of the [GGUF sweep](docs/BENCHMARKS.md); picks per VRAM: [which card](docs/QUICKSTART.md#which-card) |
| more knowledge, still fast | **Qwen3.6-35B-A3B** | mixture of experts: large, but only a small part works per token |
| another model family | **gpt-oss-20b** or **Gemma-4-26B-A4B** | both in the [GGUF sweep](docs/BENCHMARKS.md) |
| the biggest model, and ~68 GiB of free RAM | **Qwen3.8-Flash-Next** NVFP4 | bigger than any card's VRAM: the experts live in host RAM ([below](#models-larger-than-vram)) |

**Will it fit?** Weights plus the KV cache (the model's memory of the conversation) must fit in the card's VRAM (8 GB on an RTX 5060 up to 96 GB on an RTX PRO 6000, [per card](docs/QUICKSTART.md#which-card)). imp sizes the context to what is left after the weights, and refuses at load a model whose experts do not fit instead of running it wrong. Every family that loads, with its size: [`docs/MODELS.md`](docs/MODELS.md).

**Two file formats, no conversion step:** a **GGUF file** (as used by llama.cpp) or a **SafeTensors directory** in NVFP4 (NVIDIA Model Optimizer, llm-compressor, or imp's own `imp-quantize`). Your own BF16/FP8 checkpoint: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

## Install

**You need:** one of the cards above, Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Linux or Windows with WSL2. No build: the image comes from `ghcr.io`. Checklist: [`docs/QUICKSTART.md`](docs/QUICKSTART.md#requirements).

```bash
git clone https://github.com/kekzl/imp && cd imp
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4-vllm ~/models/Qwen3.8-27B-NVFP4-vllm   # 1. weights, 19.2 GiB download
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4-vllm   # 2. serve
```

Ready when the log says `Server listening`: open <http://localhost:8080>. Measured with exactly this command: **96.5 tokens/s** for one user (6 real prompts, 400 tokens each), first token after 27-29 ms, 19.7 GiB of the card in use. The `imp-cache` volume keeps converted weights and the measured memory plan for the next start.
<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4 cuda=13.4.3 path=server-api cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` n=5 server starts x 6 prompts note=max_batch_size auto 32, no speculation; VRAM = card peak minus idle] -->

**No Docker:** every release attaches `imp-<version>-linux-x86_64.tar.gz` (CUDA libraries bundled, needs only the NVIDIA driver): [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#without-docker). **Shortcut:** `--model hf://<org>/<repo>[:<file>.gguf]` downloads inside the container (needs `--user "$(id -u):$(id -g)"` on a bind-mounted `~/models`): [`docs/CONFIG.md`](docs/CONFIG.md#fetching-from-hugging-face). Full walkthrough: [`docs/QUICKSTART.md`](docs/QUICKSTART.md). **Your card's numbers:** `scripts/bench-community.sh` prints a block to post in [Discussions](https://github.com/kekzl/imp/discussions).

## Models larger than VRAM

Mixture-of-experts models use a few experts per token. When the experts do not fit, imp keeps them in pinned host RAM, caches the busiest on the card (least recently used out) and copies a missing expert over PCIe when a token needs it. Nothing to configure: imp decides at load.

**Qwen3.8-Flash-Next** ([`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), 123.6 GiB, 56.25 GiB of experts) on one RTX 5090:

| | measured |
| --- | --- |
| one user, 6 real prompts x 400 tokens | **92.1 tokens/s** (89.2-96.4 across the prompts, 91.8-92.2 across 3 starts), first token after 392 ms |
| reading a 512-token prompt | 1338.18 tokens/s (1331.51-1341.98) |
| host RAM taken at load | ~68 GiB (`MemAvailable` 76.6-77.0 -> 8.9-9.2 GiB on a 78 GiB host) |
| card memory in use, start to ready | 25.7 GiB of 32; 38-43 s (50 s on the first start, which records the memory plan) |

<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.3 path=host-resident-experts cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` +
       `imp-cli --bench --bench-pp 512 --bench-reps 5 --max-tokens 128 --bench-prompt-file novel_prose_2026.txt` n=3 server starts, 3 bench processes note=40 GiB row: CHANGELOG [0.50.0] #2621] -->

**You need:** a 32 GB card or bigger, the host RAM above free, 123.6 GiB of disk. With less RAM imp reads missing experts from disk instead (at a 40 GiB container limit: 46.9..60.0 tokens/s, v0.50.0, [`CHANGELOG.md`](CHANGELOG.md)). Speed depends on PCIe (test card: PCIe 5.0 x16) and on how often a conversation changes topic.

```bash
STAGE_DIR=~/models/Qwen3.8-Flash-Next-NVFP4 scripts/stage-model.sh nvidia/Qwen3.8-Flash-Next-NVFP4 ~/models/Qwen3.8-Flash-Next-NVFP4
docker run -d --name imp-server --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8090:8090 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-Flash-Next-NVFP4 --host 0.0.0.0 --port 8090
```

**Ready** when `docker logs imp-server` prints `Server listening on http://0.0.0.0:8090`. More: [`docs/QUICKSTART.md`](docs/QUICKSTART.md#qwen38-flash-next).

## Using it

- **In the browser:** <http://localhost:8080>, the built-in chat: saved conversations, sampling controls, live tok/s, server metrics, GPU energy per answer ([tour](docs/QUICKSTART.md#4-or-use-the-other-surfaces)).<br><img src="docs/webui.png" width="720" alt="the built-in web UI: Qwen3.8-Flash-Next answering with a table and code">
- **Your apps and coding agents:** an "OpenAI-compatible" provider with base URL **`http://localhost:8080/v1`**; Anthropic apps: `http://localhost:8080/v1/messages`. The model id is the file or directory name (`GET /v1/models`).
- **Tool calling, JSON output, streaming** on every API dialect; which fields work: [`docs/API.md`](docs/API.md).
- **Thinking models:** the reasoning comes back in `reasoning_content`, the answer in `content`.
- **Proxy, API key, metrics:** [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). **Without the server:** `imp-cli` in the image, or the C library (`include/imp/imp.h`).
- **Something went wrong?** Slow replies, `503`, empty `content`, a stalled stream: [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md), or an issue with the server log.

## How does it work?

- **One chip generation, no compromises for others.** Every Blackwell `sm_120a` card multiplies 4-bit numbers (NVFP4) directly; imp keeps weights in that format and computes on it, where portable engines unpack to 16 bits or skip the format.
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
