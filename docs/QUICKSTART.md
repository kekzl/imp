<!--
layer: L1
audience: operators
verified: 2026-08-31
commit: f243179c
-->

# Quickstart

From nothing to an answered completion. Everything runs in Docker; the host
needs a driver, not a CUDA toolkit.

Worked example: **Qwen3.8-27B**, same as the
[README](../README.md#60-second-quickstart). This page is the long version:
what to check first, what the startup log should say, the other ways in.

## Requirements

- An `sm_120a` GPU: RTX 5090 (32 GB, the tested one), 5080, 5070 Ti, or RTX
  PRO 6000. Nothing else works; imp says so rather than fall back.
- Recent NVIDIA driver plus NVIDIA Container Toolkit (`--gpus all`).
- Disk for the model: Qwen3.8-27B in NVFP4 is 19.2 GiB, a 30B-class
  checkpoint ~16 GB. Converting yourself needs room for the source too.

Check the GPU is visible and free first. On WSL2 `nvidia-smi` alone is not
enough: a container can hold the card without appearing in it, check
`docker ps` too.

```bash
nvidia-smi
docker ps
```

## 1. Get a model

`stage-model.sh` needs the `imp:test` image and exits 1 without it, including on
the download-only path (#1682). `make build` is what produces that tag;
`docker compose build imp-server` produces `imp:latest`.

```bash
make build                                     # ~3.5 min, produces imp:test
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4-vllm ~/models/Qwen3.8-27B-NVFP4-vllm
```

19.2 GiB, download only: that repo is already NVFP4 (an `imp-quantize
--format vllm` export - the same directory also loads in vLLM, verified on
0.27.1), so the script fetches it and stops. imp reads two formats, no conversion step for either:

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

The script needs only `curl`, `jq` and Docker: `git clone` needs git-lfs and
`huggingface-cli` needs Python, neither installed by the clean-host policy,
and imp itself never fetches (`src/model/hf_hub.h`).

Before converting it forecasts the output size and whether it fits the card:
a checkpoint that would miss the card costs seconds instead of 25 minutes.
Prefer an FP8 source where one exists: for Qwen3.8-27B 28.8 GiB against
51.8 GiB for BF16, costing 0.24 % perplexity (4.6262 against 4.6151 on
`ppl_corpus_45k.txt`).

Models shipping only as BF16 or FP8 need the conversion to run at all: FP8
has no GEMM on `sm_120`, and a 27B in either format does not fit.

## 2. Start the server

```bash
docker run --gpus all \
  -v ~/models:/models \
  -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 \
  ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4-vllm
```

Mount the cache volume. It holds two things; the second costs VRAM rather
than time:

- the transformed weights, so a second start skips the conversion: on
  Qwen3-14B-NVFP4, 7.9 s cold against 2.1 s warm;
- the measured library reserve. Without it the memory plan charges a 3900 MiB
  constant on every start, out of the KV pool. Measured on Qwen3-14B-Q6_K:
  0 MiB planned with the path mounted against 3900 MiB without, handing
  639 MiB back to the pools per restart. On a model whose first forward
  claims almost nothing the gap is the whole constant; the server says so
  after its first forward ("library reserve MISMATCH").

A `docker run --rm` without this volume re-measures every start and never
uses the answer.

Watch for these lines (the engine picked the fast paths). On Qwen3.8-27B:

```
CUTLASS sm_120 NVFP4 weight cache: 401 tensors, 1525.78 MiB
NVFP4 LM head: quantized FP16 [248320 x 5120] -> NVFP4 (682.0 MiB), decode GEMV fast path
Resolved dispatch: attn_prefill=fa2_fp16qk attn_decode=paged_nvfp4 moe_prefill=n/a (dense) graphs=1
```

`graphs=1` is the one that matters: at `graphs=0` decode runs several times
slower, see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) before benchmarking.
A MoE checkpoint prints a decode-cache line instead of `n/a (dense)`; it
should read `FULL`.

One warning is expected on this model, not a fault: the LM head exceeds the
graph-capture workspace cap, so **prefill** graph capture is disabled while
decode capture stays on. Weights land at 18.3 GiB, leaving room for the KV
cache on a 32 GB card.

This checkpoint ships a trained MTP head the default load leaves off. Adding

```bash
  --set speculative.mtp_k=1 --set speculative.ngram=false
```

buys +17-21 % single-stream decode for 0.79 GiB of VRAM. `GET /health`
repeats the hint as `mtp_head_hint` whenever the head is present but
unloaded; numbers, provenance and the trade:
[`LIMITATIONS.md`](LIMITATIONS.md).

## 3. Ask it something

The `model` field is required, and its value is the file or directory basename.
`GET /v1/models` lists what is loaded.

```bash
curl -s http://localhost:8080/v1/models | jq -r '.data[].id'
# Qwen3.8-27B-NVFP4-vllm

curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3.8-27B-NVFP4-vllm",
        "messages": [{"role": "user", "content": "Why is the sky blue?"}],
        "max_tokens": 64
      }' | jq -r '.choices[0].message.content'
```

Expected: an explanation of Rayleigh scattering, written at about 102 tok/s
(the embedded MTP head is taken automatically on a single-stream run).

[PROV: commit=f243179c date=2026-08-31 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4
       cuda=13.3 path=nvfp4-safetensors n=2 image=ghcr.io/kekzl/imp:latest
       cmd=`imp-cli --model … --prompt … --max-tokens 128 --temperature 0` (102.8/101.9 tok/s)]

## 4. Or use the other surfaces

**Browser.** <http://localhost:8080>: single-page chat UI, streams the answer,
draws one bar per token (inter-token latency visible while the reply is
written).

**Anthropic clients.** The same server answers `/v1/messages` natively. Point
`ANTHROPIC_BASE_URL` at it; an Anthropic-shaped client works unchanged, no
proxy.

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
