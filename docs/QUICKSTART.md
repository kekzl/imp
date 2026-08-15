<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# Quickstart

From nothing to an answered completion. Everything runs in Docker; the host
needs a driver, not a CUDA toolkit.

## Requirements

- An `sm_120a` GPU: RTX 5090 (32 GB, the tested one), 5080, 5070 Ti, or RTX PRO
  6000. Nothing else works, and imp will say so rather than fall back.
- A recent NVIDIA driver and the NVIDIA Container Toolkit, so `--gpus all` works.
- Disk for the model. A 30B-class NVFP4 checkpoint is ~16 GB.

Check the GPU is visible and free before you start. On WSL2 `nvidia-smi` alone is
not enough: a container can hold the card without appearing in it, so check
`docker ps` too.

```bash
nvidia-smi
docker ps
```

## 1. Get a model

imp reads two formats and needs no conversion step for either:

- a **GGUF file** (`Qwen3-8B-Q8_0.gguf`)
- a **SafeTensors directory** with NVFP4 weights, as exported by NVIDIA Model
  Optimizer or llm-compressor

```bash
mkdir -p models
# then download into ./models/
```

Which checkpoints load, and what each needs: [`MODELS.md`](MODELS.md).

### Staging a model from HuggingFace

`scripts/stage-model.sh` fetches a repo into a local directory, and converts it
if it is not NVFP4 yet.

```bash
# ready-made NVFP4: download only
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4 ~/models/Qwen3.8-27B-NVFP4

# BF16 or FP8 source: download, then convert (~25 min for a 27B)
scripts/stage-model.sh Qwen/Qwen3.8-27B-FP8 ~/models/Qwen3.8-27B-NVFP4
```

It exists because the obvious ways to fetch a repo are unavailable here by
design: `git clone` needs git-lfs, and `huggingface-cli` needs Python, neither of
which the clean-host policy installs. This needs only `curl`, `jq` and Docker.
imp itself never fetches (`src/model/hf_hub.h`), so something has to.

For a source that does need converting it forecasts the output size and whether
it fits the card **before** spending the 25 minutes. Prefer an FP8 source where
one exists: for Qwen3.8-27B it is 28.8 GiB against 51.8 GiB for BF16, and costs
0.24 % perplexity (4.6262 against 4.6151 on `ppl_corpus_45k.txt`).

Some models ship only as BF16 or FP8, which on this card is a wall rather than
an inconvenience: FP8 has no GEMM on `sm_120`, and a 27B in either format does
not fit. Converting is what makes them run at all.

## 2. Start the server

```bash
docker run --gpus all \
  -v ./models:/models \
  -v imp-cache:/home/imp/.cache/imp \
  -p 8080:8080 \
  ghcr.io/kekzl/imp:latest --model /models/your-model.gguf
```

The cache volume is optional and pays for itself immediately: it holds the
transformed weights, so a second start skips the conversion. On
Qwen3-14B-NVFP4 that is 7.9 s cold against 2.1 s warm.

Watch for these lines in the log; they tell you the engine picked the fast paths:

```
Resolved dispatch: attn_prefill=fa2_fp16qk attn_decode=paged_fp8 ... graphs=1
NVFP4 decode caches: FULL (48/48 MoE layers) — decode graph capture eligible
```

If `graphs=0`, or the cache line says anything other than `FULL`, see
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) before you benchmark anything.

## 3. Ask it something

The `model` field is required, and its value is the file or directory basename.
`GET /v1/models` lists what is loaded.

```bash
curl -s http://localhost:8080/v1/models | jq -r '.data[].id'

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "your-model.gguf",
        "messages": [{"role": "user", "content": "Name the capital of France."}],
        "max_tokens": 64
      }' | jq -r '.choices[0].message.content'
```

Expected: `Paris`, possibly with a sentence around it.

## 4. Or use the other surfaces

**Browser.** Open <http://localhost:8080>. The server ships a single-page chat UI
that streams the answer and draws one bar per token, so inter-token latency is
visible while the reply is written.

**Anthropic clients.** The same server answers `/v1/messages` natively. Point
`ANTHROPIC_BASE_URL` at it and an Anthropic-shaped client works unchanged, with
no proxy in between.

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=dummy your-agent
```

**CLI, no server.** `imp-cli` runs a single prompt or an interactive chat.
Command reference: [`usage.md`](usage.md).

**C library.** `imp.h` exposes the engine directly. Reference: [`usage.md`](usage.md#c-api).

## Next

| | |
|---|---|
| put it behind a proxy, with auth | [`DEPLOYMENT.md`](DEPLOYMENT.md) |
| which API fields actually work | [`API.md`](API.md) |
| it did something odd | [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |
| all CLI flags and `imp.conf` keys | [`usage.md`](usage.md) |
