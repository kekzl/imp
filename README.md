<!--
layer: L0
audience: newcomers
verified: 2026-09-22
commit: 9cbb8004
-->

<p align="center">
  <img src="docs/logo.svg" alt="imp" width="500">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/kekzl/imp?style=flat&color=blue" alt="License"></a>
  <img src="https://img.shields.io/badge/CUDA-13.4-76b900?style=flat&logo=nvidia" alt="CUDA 13.4">
  <img src="https://img.shields.io/badge/C++-23-00599C?style=flat&logo=cplusplus" alt="C++23">
</p>

---

**imp is an LLM inference engine that targets exactly one chip: the NVIDIA RTX 5090.**

- From-scratch C++23/CUDA engine for consumer Blackwell (`sm_120a`): own GGUF and SafeTensors loaders, tokenizer, paged KV cache, kernels.
- A server that speaks **both** the OpenAI and the Anthropic APIs natively - no shim for either.
- Also a C library and a CLI, not only a server.
- Not portable (no CPU path, no other GPU), not multi-GPU, not a supported product (one author, no SLO, no support rotation).

Who this fits, who should use [llama.cpp](https://github.com/ggerganov/llama.cpp) (breadth) or [vLLM](https://github.com/vllm-project/vllm) (scale-out) instead, and the numbers behind that call: [`docs/PERF.md`](docs/PERF.md#competitive-standing).

## Requirements

- `sm_120a` GPU only: RTX 5090 (32 GB, the tested one), 5080, 5070 Ti, or RTX PRO 6000. Nothing else works.
- Docker with the NVIDIA Container Toolkit (`--gpus all`); the host needs no CUDA toolkit.

Full checklist, including the WSL2 `docker ps` caveat: [`docs/QUICKSTART.md`](docs/QUICKSTART.md#requirements).

## 3-command quickstart

Worked example: **Qwen3.8-27B**, a 27B multimodal model quantized to NVFP4 so it fits one 5090 with room for a real context.

**1. Build the image** (~6 min the first time, seconds after via ccache):

```bash
make build
```

**2. Get the weights** (19.2 GiB, download only): [kekzle/Qwen3.8-27B-NVFP4-vllm](https://huggingface.co/kekzle/Qwen3.8-27B-NVFP4-vllm), an `imp-quantize --format vllm` export that also loads in vLLM.

```bash
scripts/stage-model.sh kekzle/Qwen3.8-27B-NVFP4-vllm ~/models/Qwen3.8-27B-NVFP4-vllm
```

**3. Serve:**

```bash
docker run --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-27B-NVFP4-vllm
```

Open <http://localhost:8080> for the built-in chat UI, or `curl` `/v1/chat/completions` (model id is the file/directory basename, `GET /v1/models` lists it). The weights take 18.3 GiB on the card and answer at **~102 tok/s**, leaving ~7 GiB for the KV cache on a 32 GB 5090.

[PROV: commit=f243179c date=2026-08-31 hw=RTX5090 model=Qwen3.8-27B-NVFP4-vllm quant=NVFP4
       cuda=13.3 path=nvfp4-safetensors n=2 image=ghcr.io/kekzl/imp:latest
       cmd=`imp-cli --model … --prompt … --max-tokens 128 --temperature 0` (102.8/101.9 tok/s)]

Full walkthrough (screenshot, other model formats, bringing your own BF16/FP8 checkpoint, the MTP head): [`docs/QUICKSTART.md`](docs/QUICKSTART.md). Quantizing it yourself, quality numbers: [`docs/quantization.md`](docs/quantization.md).

## What works today

Full matrix with per-item status: [`docs/FEATURES.md`](docs/FEATURES.md). ✅ = code path plus a gated test, 🟡 = code path, no test.

| area | |
|---|---|
| **Models** | ✅ Qwen3 / 3.5 / 3.6 / 3.8 (dense, MoE, Gated-DeltaNet hybrids), LLaMA, Mistral, Mixtral, DeepSeek incl. V2 latent attention, Gemma-3 and Gemma-4, gpt-oss, Nemotron-H, nomic-bert embeddings. 🟡 Llama-4 |
| **Vision** | ✅ Qwen3-VL and Qwen3.6-35B-A3B (tower in the checkpoint), Gemma-3/4 via `--mmproj`. No video |
| **Quantisation** | ✅ NVFP4 (native, the primary path), MXFP4, GGUF Q2_K-Q8_0, IQ4_NL/XS, FP8 E4M3 weights and KV, INT8/INT4 KV |
| **APIs** | ✅ OpenAI chat/completions/responses/embeddings, Anthropic `/v1/messages`, `/v1/rerank`, per-token SSE on all three dialects, tool calling, JSON-Schema / regex / GBNF constrained decoding, prefix caching with `cache_control` |
| **Serving** | ✅ continuous batching, paged KV, model swap on request, suspend/resume to free the GPU, Prometheus `/metrics`, API-key auth |
| **Engine** | ✅ NVFP4 block-scaled `mma.sync` GEMM/GEMV, FP8 `f8f6f4` attention scores, FlashAttention-2 prefill (hd 128 and 256), CUDA graphs on prefill and decode, Gated DeltaNet and Mamba2, speculative decoding |

## What CI defends

Every push measures this on one pinned model (Qwen3-8B Q8_0); the regression gate is 8 % either way. Full sweep across models, competitive numbers vs llama.cpp/vLLM (imp led llama.cpp +141 % down to +3 % across six models on 2026-08-30, imp led vLLM +25 % at 32 streams / -7.6 % on a dense checkpoint at the same width on 2026-09-02), and why the threshold is 8 % and not 3 %: [`docs/PERF.md`](docs/PERF.md). Per-model history with exact commands: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

<!-- PERF:BEGIN -->
| metric | value | threshold |
|---|---|---|
| decode tg128 | **299.61 tok/s** | 8 % |
| prefill pp128 | 5031.75 tok/s | 8 % |
| prefill pp512 | **12707.44 tok/s** | 8 % |
| prefill pp4096 | 15776.32 tok/s | 8 % |
| peak VRAM (own) | 20642 MiB | 10 % |

[PROV: commit=6fe00f81 date=2026-09-17 hw=RTX5090 model=Qwen3-8B-Q8_0 quant=Q8_0
       cuda=13.4 path=gguf-dp4a cmd=`make verify-fast` n=5x5]
<!-- PERF:END -->

Decode on this host moves several percent between sessions with nothing changed, and prefill moves more (the MoE path, not cuBLAS); that is why the thresholds are 8 % and not 3 %. Detail: [`docs/PERF.md`](docs/PERF.md).

Provenance for the headline numbers above: llama.cpp pinned by digest `ghcr.io/ggml-org/llama.cpp@sha256:c49f4d48…` (imp v0.33.0, 2026-08-30, `make bench-competitive`); vLLM run with `--gpu-memory-utilization 0.90 --max-model-len 16384 --max-num-seqs N` (`tools/analysis/vllm_conc_ab.sh`) - dense Qwen3-14B-NVFP4, 32 streams, after `attention.paged_fp8_multitok=4`: imp **3948.9** vs vLLM 3817.6 tok/s (+3.4 %).

## Go deeper

| I want to … | Read |
|---|---|
| just run it | [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| put it behind a proxy, with auth and metrics | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| know which API fields actually work | [`docs/API.md`](docs/API.md) |
| know which models and quants load | [`docs/MODELS.md`](docs/MODELS.md) |
| know how it works | [`docs/internals/ARCHITECTURE.md`](docs/internals/ARCHITECTURE.md) |
| know why it is this fast, or this slow | [`docs/PERF.md`](docs/PERF.md), [`docs/internals/BENCHMARKING.md`](docs/internals/BENCHMARKING.md) |
| fix something that went wrong | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| understand or replace a kernel | [`docs/internals/KERNELS.md`](docs/internals/KERNELS.md) |
| know why there is no multi-GPU, and the rest of where imp loses | [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md), [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) |
| build from source, or contribute | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| work on it as an AI agent | [`CLAUDE.md`](CLAUDE.md) |

## Build from source

Tracks `main` rather than the latest release.

```bash
git clone https://github.com/kekzl/imp.git && cd imp
docker compose build imp-server
docker run --gpus all -v ./models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8080:8080 imp:latest --model /models/your-model.gguf
```

Contributor workflow, build targets, test lanes: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## How this was built

Every line of imp was written by an AI coding agent (Claude Code), across ~138k lines of engine C++/CUDA plus tooling and tests. The repository keeps its own audit trail of that: [`docs/audit/`](docs/audit/), [`docs/MISSION_JOURNAL.md`](docs/MISSION_JOURNAL.md).

## License

MIT. See [`LICENSE`](LICENSE). One kernel file is adapted from Apache-2.0 code (SageAttention); its licence text and notice are in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), shipped in the image at `/usr/share/doc/imp/`.
