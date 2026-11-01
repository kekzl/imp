<!--
layer: L1
audience: operators
verified: 2026-11-01
commit: 216a779c
-->

# Quickstart

Three commands, all in Docker. The host needs an NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); no CUDA toolkit, no Python.

## 1. Download a model

```bash
git clone https://github.com/kekzl/imp && cd imp
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4-vllm ~/models/Qwen3.8-27B-NVFP4-vllm
```

19.2 GiB, for a 24 or 32 GB card. Smaller card: [which card](#which-card).

## 2. Start the server

```bash
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4-vllm
```

Ready when the log prints `Server listening`. Keep the `imp-cache` volume: the next start skips the weight conversion ([`DEPLOYMENT.md`](DEPLOYMENT.md#compose)).

## 3. Ask it something

Open <http://localhost:8080> for the built-in chat ([`WEB_UI.md`](WEB_UI.md)), or:

```bash
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.8-27B-NVFP4-vllm", "messages": [{"role": "user", "content": "Why is the sky blue?"}]}' \
  | jq -r '.choices[0].message.content'
```

`model` is the directory name; `GET /v1/models` lists it. Done: the rest of this page is for a different card, a different model, or a problem.

## Which card

Every compute capability 12.0 card runs the same `sm_120a` kernels ([NVIDIA list](https://developer.nvidia.com/cuda-gpus)); tested is the RTX 5090, the other rows are untested. Weight sizes: [`MODELS.md`](MODELS.md); the KV cache takes what is left.

| VRAM | Cards | Picks (weights) |
| --- | --- | --- |
| 96 GB | RTX PRO 6000 Blackwell (Workstation, Max-Q, Server) | every 32 GB pick with a far longer context |
| 48 GB | RTX PRO 5000 Blackwell | every 32 GB pick with a longer context |
| 32 GB | **RTX 5090** (tested), RTX 5090 D, RTX PRO 4500 Blackwell | Qwen3.8-27B NVFP4 (17 GB), Qwen3-Coder-30B-A3B NVFP4 (17 GB), Nemotron-3-Nano-30B-A3B NVFP4 (18 GB); Qwen3.8-Flash-Next with ~67 GiB host RAM |
| 24 GB | RTX 5090 D V2, RTX PRO 4000 Blackwell | Qwen3.8-27B NVFP4 (17 GB), Qwen3-30B-A3B NVFP4 (16 GB), short context |
| 16 GB | RTX 5080, RTX 5070 Ti, RTX 5060 Ti 16 GB, RTX PRO 2000 Blackwell | Qwen3-14B NVFP4 (10 GB), Qwen3-8B Q8_0 (8.2 GB) |
| 12 GB | RTX 5070 | Qwen3-8B Q8_0 (8.2 GB), Qwen3-8B NVFP4 (5.0 GB) |
| 8 GB | RTX 5060, RTX 5060 Ti 8 GB, RTX 5050 | Qwen3-4B Q8_0 (4.0 GB), Llama-3.2-3B Q8_0 (3.2 GB) |

- RTX 5090 D and 5090 D V2 (China market): the RTX 5090 die (GB202).
- Any other card is refused at start, no fallback.
- DGX Spark (GB10) is compute capability 12.1: no image in the default build ([`LIMITATIONS.md`](LIMITATIONS.md)).

## Other models

- `stage-model.sh <org>/<repo> <dir>` downloads any checkpoint from [`MODELS.md`](MODELS.md); one that loads as is also runs as `--model hf://<org>/<repo>` ([`CONFIG.md`](CONFIG.md#fetching-from-hugging-face)).
- imp is built for **SafeTensors** directories with NVFP4 or FP8 weights (NVIDIA Model Optimizer, llm-compressor, `imp-quantize`). A **GGUF** file still loads as legacy, without new speedups.
- Bigger than the card, e.g. Qwen3.8-Flash-Next: [`BIG_MODELS.md`](BIG_MODELS.md).

### Converting a source that is not NVFP4 yet

```bash
scripts/stage-model.sh Qwen/Qwen3.8-27B-FP8 ~/models/my-Qwen3.8-NVFP4
```

- Qwen3.8-27B: 28.8 GiB down, ~25 minutes on the card, 18.8 GiB out. It forecasts the output size first and stops in seconds if the result would not fit the card.
- Prefer an FP8 source: 28.8 GiB instead of 51.8 GiB for BF16, for 0.24 % perplexity (4.6262 against 4.6151 on `ppl_corpus_45k.txt`).
- A BF16 or FP8-only model needs this to run at all: FP8 has no GEMM on `sm_120`.
- Needs only `curl`, `jq` and Docker. With your own image: [`BUILD.md`](BUILD.md).

## Something looks wrong

| Symptom | Check |
| --- | --- |
| start fails, out of memory | `nvidia-smi`; on WSL2 also `docker ps`: a container can hold the card without showing in `nvidia-smi` |
| slow answers | the startup line `Resolved dispatch: ...` must end in `graphs=1`; `graphs=0` decodes several times slower |
| slow MoE model | its decode-cache line should read `FULL` |
| Qwen3.8-27B warns that prefill graph capture is disabled | expected: its LM head exceeds the capture workspace cap; decode capture stays on |
| anything else | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |

## Faster single-user decode

Qwen3.8-27B ships a trained MTP head that the default load leaves off. Append to the `docker run` line:

```bash
  --set speculative.mtp_k=1 --set speculative.ngram=false
```

+17-21 % single-stream decode for 0.79 GiB of VRAM; numbers and the trade: [`LIMITATIONS.md`](LIMITATIONS.md). `GET /health` names it as `mtp_head_hint` while the head is unloaded.

## Other ways in

- **Anthropic clients**: the server answers `/v1/messages` natively. `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=dummy your-agent`, no proxy.
- **CLI, no server**: `imp-cli` runs one prompt or an interactive chat ([`CONFIG.md`](CONFIG.md)).
- **C library**: `imp.h` ([`CONFIG.md`](CONFIG.md#c-api)).

## Next

| | |
|---|---|
| the chat UI, feature by feature | [`WEB_UI.md`](WEB_UI.md) |
| build it yourself, or run without Docker | [`BUILD.md`](BUILD.md) |
| put it behind a proxy, with auth | [`DEPLOYMENT.md`](DEPLOYMENT.md) |
| post your card's numbers | `scripts/bench-community.sh` prints a block for [Discussions](https://github.com/kekzl/imp/discussions) |
| which API fields actually work | [`API.md`](API.md) |
| it did something odd | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |
| all CLI flags and `imp.conf` keys | [`CONFIG.md`](CONFIG.md) |
