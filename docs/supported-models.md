# Supported models

Model families with a known-working code path on `main`. Throughput numbers come from [`performance.md`](performance.md) — see that doc for methodology and the cuBLAS prefill-variance caveat. VRAM figures are model weights only (default KV cache adds 1–4 GiB depending on context).

Anything not on this list may still load (the GGUF and SafeTensors paths cover most LLaMA-derived architectures), but it has not been verified end-to-end.

## Dense transformers

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF) | Q8_0 | 4.0 GB | 236 | GGUF |
| Qwen3-4B | MXFP4 | 2.8 GB | 124 | GGUF (imp-converted) |
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | Q8_0 | 8.2 GB | **268** (tg128, CI baseline #540) | GGUF |
| [Qwen3-8B](https://huggingface.co/cortecs/Qwen3-8B-NVFP4) | NVFP4 | 5.0 GB | **277** | SafeTensors (cortecs) |
| [Qwen3-14B](https://huggingface.co/unsloth/Qwen3-14B-GGUF) | Q6_K | 12 GB | **158** | GGUF |
| [Qwen3-14B](https://huggingface.co/nvidia/Qwen3-14B-NVFP4) | NVFP4 | 10 GB | 168 | SafeTensors (nvidia) |
| [Qwen3-32B](https://huggingface.co/unsloth/Qwen3-32B-GGUF) | Q4_K_M | 19 GB | — | GGUF |
| [Phi-4-reasoning-plus](https://huggingface.co/nvidia/Phi-4-reasoning-plus-NVFP4) | NVFP4 | 9.0 GB | 157 | SafeTensors (nvidia), fused projections |
| [Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF) | Q8_0 | 3.2 GB | 306 | GGUF |
| [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | Q6_K | 19 GB | — | GGUF |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) | Q8_0 | 7.6 GB | — | GGUF |
| [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF) | Q6_K | 12 GB | — | GGUF |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | bf16 | 28 GB | ~30 (eager) | SafeTensors — **MLA** (first Multi-head Latent Attention arch); experts host-offloaded on 32 GB → graphs disabled. PPL 6.06 vs HF 5.07 (+19.6%). |
| [DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) | bf16 | 30 GB | — (eager) | SafeTensors — **MLA** (same `deepseek_v2` arch as V2-Lite: kv_lora_rank=512, q_lora_rank=0); experts host-offloaded on 32 GB → graphs disabled. Second MLA checkpoint, code-specialized — coherent codegen verified. |

## Hybrid (Gated DeltaNet + attention)

GDN models use FP16 prefill instead of FP8 (~8% slower than FP8 dense, but eliminates multi-turn state collapse). Linear-time scan vs O(n²) attention.

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Q8_0 | 4.2 GB | 222 | GGUF |
| [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | Q8_0 | 8.9 GB | 142 | GGUF |
| [Qwen3.5-27B](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | Q4_K_M | 16 GB | — | GGUF |
| [Qwable-3.6-27B](https://huggingface.co/Mia-AiLab/Qwable-3.6-27b) | Q4_K_M | 16 GB | ~18 | GGUF, validated dense-GDN 27B (Qwen3.6-27B fine-tune, 64 layers: 16 attn + 48 GDN). ~29 GB resident (Q4_K + NVFP4 decode cache + GDN state) → relies on the auto KV clamp to serve. Heavy trace-reasoner: give it generous `max_tokens`. |
| [Qwen3.6-27B-Text-NVFP4-MTP](https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP) | NVFP4 | 17 GB | — | SafeTensors (Modelopt), dense-GDN 27B. **Checkpoint quantizes the GDN `linear_attn` projections to NVFP4** — `ssm_in`/`ssm_out`/`gdn_gate` run native NVFP4, `gdn_alpha`/`gdn_beta` (FP16_ONLY) are dequanted to FP16 at load (#812). |
| Qwen3.5-27B | MXFP4 | — | — | Loads OOM on 32 GB — see [roadmap](roadmap.md) |

## Mixture-of-Experts

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [Qwen3-Coder-30B-A3B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | Q6_K | 24 GB | 236 | GGUF |
| [Qwen3-Coder-30B-A3B](https://huggingface.co/NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4) | NVFP4 | 16 GB | 338 | SafeTensors (Modelopt) |
| [Qwen3-30B-A3B](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4) | NVFP4 | 16 GB | 307 | SafeTensors (Modelopt) |
| [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Q4_K_M | 22 GB | 243 | GGUF |
| [Qwen3.6-35B-A3B](https://huggingface.co/mmangkad/Qwen3.6-35B-A3B-NVFP4) | NVFP4 | 18 GB | 257 | SafeTensors (Modelopt) |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q4_K_M | 14 GB | 273 (tg128) | GGUF |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q5_K_M | 17 GB | 65 | GGUF, recommended for code-gen |
| [Gemma-4-26B-A4B-it](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | NVFP4 | 14 GB | 266 | SafeTensors (Modelopt) |
| [Nemotron-3-Nano-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) | NVFP4 | 18 GB | 126 | SafeTensors (Modelopt), Mamba2+attn+MoE, arch-limited (FP16 GDN/attn-projection tax) |
| [Nemotron-Labs-3-Elastic-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4) | NVFP4 | 18 GB | 70 | SafeTensors (QAD), same arch as Nano |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | MXFP4 (native) | 15 GB | **345** | SafeTensors; experts converted to NVFP4 at load. Also loads the official GGUF (Q8_0- or bf16-dense + MXFP4 experts, e.g. `gpt-oss-20b-mxfp4.gguf` — the Q8_0 residual rescale was fixed in #808). Harmony chat format (analysis/final channels split into `reasoning_content`/`content`). Use temperature 1.0 — greedy loops in the analysis channel (model-intrinsic). Prefill ≈ 16-19k tok/s (CUTLASS grouped GEMM). |

## Vision

Gemma-3 and Gemma-4 are the multimodal families currently supported. The vision encoder weights ship as a separate `mmproj.gguf` file:

| Model | Quant | VRAM | Decode `tg256` | Notes |
|---|---|---:|---:|---|
| [Gemma-3-12B-it](https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF) | Q8_0 | 12 GB | 129 | text + vision (includes mmproj) |
| [Gemma-3-27B-it](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF) | Q4_K_M | 16 GB | — | largest Gemma-3 |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q4_K_M | 14 GB | 273 (tg128) | text + vision via the gemma4v encoder (separate BF16 mmproj) — see [`vision_gemma4v_spec.md`](vision_gemma4v_spec.md) |

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
