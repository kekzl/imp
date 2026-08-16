<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# Quickstart

From nothing to an answered completion. Everything runs in Docker; the host
needs a driver, not a CUDA toolkit.

The worked example is **Qwen3.8-27B**, the same one the
[README](../README.md#60-second-quickstart) uses. This page is the long version:
what to check first, what the startup log should say, and the other ways in.

## Requirements

- An `sm_120a` GPU: RTX 5090 (32 GB, the tested one), 5080, 5070 Ti, or RTX PRO
  6000. Nothing else works, and imp will say so rather than fall back.
- A recent NVIDIA driver and the NVIDIA Container Toolkit, so `--gpus all` works.
- Disk for the model. Qwen3.8-27B in NVFP4 is 19.2 GiB; a 30B-class checkpoint
  is ~16 GB. Converting one yourself needs room for the source as well.

Check the GPU is visible and free before you start. On WSL2 `nvidia-smi` alone is
not enough: a container can hold the card without appearing in it, so check
`docker ps` too.

```bash
nvidia-smi
docker ps
```

## 1. Get a model

```bash
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4 ~/models/Qwen3.8-27B-NVFP4
```

19.2 GiB, download only: that repo is already NVFP4, so the script fetches it and
stops. imp reads two formats and needs no conversion step for either:

- a **GGUF file** (`Qwen3-8B-Q8_0.gguf`)
- a **SafeTensors directory** with NVFP4 weights, as exported by NVIDIA Model
  Optimizer, llm-compressor, or `imp-quantize`

Which checkpoints load, and what each needs: [`MODELS.md`](MODELS.md).

### Converting a source that is not NVFP4 yet

The same script takes a BF16 or FP8 repo and converts it. Qwen3.8-27B needs
this if you build it yourself, which is where the checkpoint above came from:

```bash
scripts/stage-model.sh Qwen/Qwen3.8-27B-FP8 ~/models/my-Qwen3.8-NVFP4
```

28.8 GiB down, ~25 minutes on the card, 18.8 GiB out.

The script exists because the obvious ways to fetch a repo are unavailable here
by design: `git clone` needs git-lfs, and `huggingface-cli` needs Python, neither
of which the clean-host policy installs. This needs only `curl`, `jq` and Docker.
imp itself never fetches (`src/model/hf_hub.h`), so something has to.

Before converting anything it forecasts the output size and whether it fits the
card, so a checkpoint that would miss the card costs seconds instead of 25
minutes. Prefer an FP8 source where one exists: for Qwen3.8-27B it is 28.8 GiB
against 51.8 GiB for BF16, and costs 0.24 % perplexity (4.6262 against 4.6151 on
`ppl_corpus_45k.txt`).

Some models ship only as BF16 or FP8, which on this card is a wall rather than
an inconvenience: FP8 has no GEMM on `sm_120`, and a 27B in either format does
not fit. Converting is what makes them run at all.

## 2. Start the server

```bash
docker run --gpus all \
  -v ~/models:/models \
  -v imp-cache:/home/imp/.cache/imp \
  -p 8080:8080 \
  ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4
```

The cache volume is optional and pays for itself immediately: it holds the
transformed weights, so a second start skips the conversion. On
Qwen3-14B-NVFP4 that is 7.9 s cold against 2.1 s warm.

Watch for these lines; they tell you the engine picked the fast paths. On
Qwen3.8-27B:

```
CUTLASS sm_120 NVFP4 weight cache: 401 tensors, 1525.78 MiB
NVFP4 LM head: quantized FP16 [248320 x 5120] -> NVFP4 (682.0 MiB), decode GEMV fast path
Resolved dispatch: attn_prefill=fa2_fp16qk attn_decode=paged_fp16 moe_prefill=n/a (dense) graphs=1
```

`graphs=1` is the one that matters; at `graphs=0` decode runs several times
slower, so see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) before benchmarking
anything. A MoE checkpoint prints a decode-cache line instead of `n/a (dense)`,
and it should read `FULL`.

One warning is expected on this model and is not a fault: the LM head is larger
than the graph-capture workspace cap, so **prefill** graph capture is disabled
while decode capture stays on. Weights land at 17.9 GiB, leaving room for the KV
cache on a 32 GB card.

## 3. Ask it something

The `model` field is required, and its value is the file or directory basename.
`GET /v1/models` lists what is loaded.

```bash
curl -s http://localhost:8080/v1/models | jq -r '.data[].id'
# Qwen3.8-27B-NVFP4

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3.8-27B-NVFP4",
        "messages": [{"role": "user", "content": "Why is the sky blue?"}],
        "max_tokens": 64
      }' | jq -r '.choices[0].message.content'
```

Expected: an explanation of Rayleigh scattering, written at about 85 tok/s.

[PROV: commit=8118d14d date=2026-08-16 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4
       cuda=13.3 path=nvfp4-safetensors n=2 image=ghcr.io/kekzl/imp:latest
       cmd=`imp-cli --model … --prompt … --max-tokens 128 --temperature 0`]

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
