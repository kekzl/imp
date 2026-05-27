# Supported models

Model families with a known-working code path on `main`. Throughput numbers come from [`performance.md`](performance.md) — see that doc for methodology and the cuBLAS prefill-variance caveat. VRAM figures are model weights only (default KV cache adds 1–4 GiB depending on context).

Anything not on this list may still load (the GGUF and SafeTensors paths cover most LLaMA-derived architectures), but it has not been verified end-to-end.

## Dense transformers

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF) | Q8_0 | 4.0 GB | 236 | GGUF |
| Qwen3-4B | MXFP4 | 2.8 GB | 124 | GGUF (imp-converted) |
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | Q8_0 | 8.2 GB | **260** | GGUF |
| [Qwen3-8B](https://huggingface.co/cortecs/Qwen3-8B-NVFP4) | NVFP4 | 5.0 GB | **238** | SafeTensors (cortecs) |
| [Qwen3-14B](https://huggingface.co/unsloth/Qwen3-14B-GGUF) | Q6_K | 12 GB | **158** | GGUF |
| [Qwen3-14B](https://huggingface.co/nvidia/Qwen3-14B-NVFP4) | NVFP4 | 10 GB | 105 | SafeTensors (nvidia) |
| [Qwen3-32B](https://huggingface.co/unsloth/Qwen3-32B-GGUF) | Q4_K_M | 19 GB | — | GGUF |
| [Phi-4-reasoning-plus](https://huggingface.co/nvidia/Phi-4-reasoning-plus-NVFP4) | NVFP4 | 9.0 GB | 115 | SafeTensors (nvidia), fused projections |
| [Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF) | Q8_0 | 3.2 GB | 306 | GGUF |
| [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | Q6_K | 19 GB | — | GGUF |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) | Q8_0 | 7.6 GB | — | GGUF |
| [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF) | Q6_K | 12 GB | — | GGUF |

## Hybrid (Gated DeltaNet + attention)

GDN models use FP16 prefill instead of FP8 (~8% slower than FP8 dense, but eliminates multi-turn state collapse). Linear-time scan vs O(n²) attention.

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Q8_0 | 4.2 GB | 222 | GGUF |
| [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | Q8_0 | 8.9 GB | 142 | GGUF |
| [Qwen3.5-27B](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | Q4_K_M | 16 GB | — | GGUF |
| Qwen3.5-27B | MXFP4 | — | — | Loads OOM on 32 GB — see [roadmap](roadmap.md) |

## Mixture-of-Experts

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [Qwen3-Coder-30B-A3B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | Q6_K | 24 GB | 236 | GGUF |
| [Qwen3-Coder-30B-A3B](https://huggingface.co/NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4) | NVFP4 | 16 GB | 270 | SafeTensors (Modelopt) |
| [Qwen3-30B-A3B](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4) | NVFP4 | 16 GB | 158 | SafeTensors (Modelopt) |
| [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Q4_K_M | 22 GB | 243 | GGUF |
| [Qwen3.6-35B-A3B](https://huggingface.co/mmangkad/Qwen3.6-35B-A3B-NVFP4) | NVFP4 | 18 GB | 154 | SafeTensors (Modelopt) |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q4_K_M | 14 GB | 187 | GGUF |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q5_K_M | 17 GB | 65 | GGUF, recommended for code-gen |
| [Gemma-4-26B-A4B-it](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | NVFP4 | 14 GB | 163 | SafeTensors (Modelopt) |
| [Nemotron-3-Nano-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) | NVFP4 | 18 GB | 47 | SafeTensors (Modelopt), Mamba2+attn+MoE, arch-limited |
| [Nemotron-Labs-3-Elastic-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4) | NVFP4 | 18 GB | 70 | SafeTensors (QAD), same arch as Nano |

## Vision

Gemma-3 is the only multimodal family currently supported. The vision encoder weights ship as a separate `mmproj.gguf` file:

| Model | Quant | VRAM | Decode `tg256` | Notes |
|---|---|---:|---:|---|
| [Gemma-3-12B-it](https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF) | Q8_0 | 12 GB | 129 | text + vision (includes mmproj) |
| [Gemma-3-27B-it](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF) | Q4_K_M | 16 GB | — | largest Gemma-3 |

Run with both flags:

```bash
imp-cli --model gemma-3-12b-it-Q8_0.gguf --mmproj mmproj-google_gemma-3-12b-it-f16.gguf \
        --image photo.jpg --prompt "Describe this image"
```

## Format notes

- **GGUF** — standard llama.cpp format. `Q*_K`, `Q8_0`, `Q*_0`, MXFP4 (imp-proprietary tensor type 31). Loaded directly from a single file. Most quants come from [unsloth](https://huggingface.co/unsloth) or [bartowski](https://huggingface.co/bartowski).
- **SafeTensors NVFP4 prequant** — produced by [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (Modelopt) or [llm-compressor](https://github.com/vllm-project/llm-compressor). Loaded from a directory with `config.json` + sharded `*.safetensors`. The Modelopt path is more thoroughly tested; llm-compressor degenerates past ~30 tokens on several models (see [roadmap](roadmap.md)).

For the underlying quantization formats and when each one is used internally, see [`quantization.md`](quantization.md).

## Loading

```bash
# GGUF — file path
imp-cli --model models/Qwen3-8B-Q8_0.gguf --prompt "Hello"

# SafeTensors — directory path
imp-cli --model models/Qwen3-Coder-30B-A3B-FP4/ --prompt "Hello"

# Vision — text model + mmproj
imp-cli --model gemma-3-12b-it-Q8_0.gguf --mmproj mmproj-google_gemma-3-12b-it-f16.gguf \
        --image photo.jpg --prompt "Describe"
```

imp does not download weights — stage them yourself via `huggingface-cli download` or `git clone`:

```bash
# Example: download a GGUF model
huggingface-cli download unsloth/Qwen3-8B-GGUF Qwen3-8B-Q8_0.gguf --local-dir models/

# Example: download an NVFP4 SafeTensors model
git clone https://huggingface.co/nvidia/Qwen3-14B-NVFP4 models/Qwen3-14B-NVFP4/
```
