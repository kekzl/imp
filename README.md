<!--
layer: L0
audience: newcomers
verified: 2026-10-31
commit: ed4422b7
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

**imp runs AI chat models on your own NVIDIA graphics card** (any GeForce RTX 50 or RTX PRO Blackwell). Chat apps and coding agents talk to it the way they talk to ChatGPT or Claude, but nothing leaves your machine. It is built for this one card generation only, and that is why it is fast. One container, nothing else to install, MIT license.

<p align="center"><img src="docs/assets/demo.gif" width="760" alt="terminal: the 123.6 GiB Qwen3.8-Flash-Next streams an answer on one 32 GB RTX 5090, with tokens per second"></p>

## How fast

| Qwen3.8-27B on one RTX 5090, tokens/s | **imp** | vLLM 0.31.0 |
| --- | ---: | ---: |
| one user | **96.3** | 69.9 |
| 32 users at once (total) | **1979.1** | 1414.2 |

<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 (Qwen3.8-27B-NVFP4-vllm on both) cuda=13.4.3 path=server-api
       cmd=`tools/analysis/engine3_conc.sh` n=3 trials x 3 waves note=imp ghcr v0.59.0; one-user imp row speculative.mtp_k=0 ngram=false; docs/audit/launch_2026_10_27/] -->

Both engines read the same model file. A token is about ¾ of a word, so 96 tokens/s is roughly 70 words per second. Every model, method and raw log: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

**Bigger than the card:** the 123.6 GiB Qwen3.8-Flash-Next answers at **92.1 tokens/s** on a 32 GB card, its experts in host RAM ([how](docs/BIG_MODELS.md)).
<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.3 path=host-resident-experts cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` n=3 server starts x 6 prompts] -->

## Quickstart (Docker)

**You need:** an RTX 50 or RTX PRO Blackwell card, Linux or Windows with WSL2, Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). No CUDA toolkit, no Python.

**Qwen3.8-27B**, the default: text and images, fits a 24-32 GB card.

```bash
git clone https://github.com/kekzl/imp && cd imp
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4-vllm ~/models/Qwen3.8-27B-NVFP4-vllm   # 1. download the model, 19.2 GiB
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4-vllm   # 2. start
```

When the log says `Server listening`, open <http://localhost:8080> and chat. On an RTX 5090 this setup answers at **96.5 tokens/s** and uses 19.7 GiB of the card.
<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4 cuda=13.4.3 path=server-api cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` n=5 server starts x 6 prompts note=max_batch_size auto 32, no speculation; VRAM = card peak minus idle] -->

**Qwen3.8-Flash-Next**, the biggest: 123.6 GiB on a 32 GB card, needs ~68 GiB free RAM and 124 GiB of disk.

```bash
STAGE_DIR=~/models/Qwen3.8-Flash-Next-NVFP4 scripts/stage-model.sh nvidia/Qwen3.8-Flash-Next-NVFP4 ~/models/Qwen3.8-Flash-Next-NVFP4
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-Flash-Next-NVFP4
```

Ready after about 40 s. How a model bigger than the card runs: [`docs/BIG_MODELS.md`](docs/BIG_MODELS.md).
<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.3 path=host-resident-experts cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` n=3 server starts x 6 prompts note=load 38-43 s with the imp-cache volume] -->

Step by step, what the log should say, other models: [`docs/QUICKSTART.md`](docs/QUICKSTART.md). Build the image yourself or run without Docker: [`docs/BUILD.md`](docs/BUILD.md).

## Which model?

| You want | Take |
| --- | --- |
| **not sure** | **Qwen3.8-27B** (the first example above): reads text and images, fits a 24-32 GB card |
| fast answers on a 12-16 GB card | **Qwen3-8B** |
| code and agents, fast | **Qwen3-Coder-30B-A3B** |
| the biggest, with ~68 GiB free RAM | **Qwen3.8-Flash-Next** ([how](docs/BIG_MODELS.md)) |

imp is built for models in **SafeTensors** format (NVFP4 or FP8). Which models work and how big they are: [`docs/MODELS.md`](docs/MODELS.md). Which card holds what: [`docs/QUICKSTART.md`](docs/QUICKSTART.md#which-card).

## Using it

- **In the browser:** <http://localhost:8080>, a built-in chat with saved conversations and live speed.<br><img src="docs/webui.png" width="720" alt="the built-in web UI: Qwen3.8-Flash-Next answering with a table and code">
- **Your apps and coding agents** (Claude Code, Codex, any OpenAI app): set the base URL to **`http://localhost:8080/v1`**.
- **Tool calls, JSON output, thinking models, streaming:** [`docs/API.md`](docs/API.md).
- **Something went wrong?** [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md), or open an issue with the server log.

## What imp is not

- **Not for other hardware:** one GPU, NVIDIA Blackwell only. No multi-GPU, no CPU, no AMD: use [vLLM](https://github.com/vllm-project/vllm) for those.
- **Not always ahead:** vLLM was faster on Qwen3-14B with 32 users at once (-7.6 %, 2026-09-02, [`BENCHMARKS.md`](docs/BENCHMARKS.md)). Every known gap: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).
- **Not a company product:** one maintainer, tested on one RTX 5090; issues are welcome, answers are not guaranteed.

## More

| I want to … | Read |
|---|---|
| know how it works | [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) |
| run it behind a proxy, with an API key | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| see every setting | [`docs/CONFIG.md`](docs/CONFIG.md) |
| build from source | [`docs/BUILD.md`](docs/BUILD.md) |
| contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| browse all docs | [`docs/README.md`](docs/README.md) |

Every line of imp was written by an AI coding agent (Claude Code); the audit trail is in [`docs/audit/`](docs/audit/).

## License

MIT, see [`LICENSE`](LICENSE). One kernel file is adapted from Apache-2.0 code (SageAttention); the APA prefill (`src/compute/apa/`) is [kekzl/apa](https://github.com/kekzl/apa) under MPL-2.0. Notices: [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), shipped in the image at `/usr/share/doc/imp/`.
