<!--
layer: L1
audience: operators
verified: 2026-08-28
commit: be825e4a
-->

# Supported models

The decode column is a per-model figure, not a gate; each comes from the
sweep named below. The gated number CI defends is a different model, in
[`PERF.md`](PERF.md).

[PROV: commit=2230e1c2 date=2026-07-12 hw=RTX5090 model=per-row quant=per-row
       cuda=13.3 path=per-row cmd=`imp-cli --bench --bench-pp 512 --bench-reps 5`
       n=5x5 note=greedy, spec off, CUDA graphs on; full detail per row in BENCHMARKS.md]

Model families with a known-working code path on `main`. Throughput numbers:
[`performance.md`](performance.md) (methodology, cuBLAS prefill-variance
caveat).

- VRAM figures are model weights only. The KV cache is sized *on top*, from
  what is left after the weight caches are built: it scales with free VRAM
  and the configured context (a dense server default lands around 4.6 GiB; a
  small model on an idle card takes far more). Bound it with
  `--set runtime.max_seq_len=N` / `kv_cache` settings; read the actual split
  with `--mem-report`.
- **`--max-seq-len` is imp-cli only**: `imp-server --max-seq-len N` hits the
  unknown-argument branch and exits 1 (#1681).
- Anything not on this list may still load (the GGUF and SafeTensors paths
  cover most LLaMA-derived architectures) but is not verified end-to-end.

## Dense transformers

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) | Q8_0 | 0.2 GB | embeddings (`/v1/embeddings`, HF-oracle cos ≥ 0.999) | GGUF |
| [Qwen3-Reranker-0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | Q8_0 | 0.7 GB | reranking (`/v1/rerank`; top-1 agrees with llama.cpp on the same GGUF, median score delta 0.0014) | GGUF |
| [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF) | Q8_0 | 4.0 GB | 236 | GGUF |
| Qwen3-4B | MXFP4 | 2.8 GB | 124 | GGUF (imp-converted) |
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | Q8_0 | 8.2 GB | **268** (tg128, CI baseline #540) | GGUF |
| [Qwen3-8B](https://huggingface.co/cortecs/Qwen3-8B-NVFP4) | NVFP4 | 5.0 GB | **277** | SafeTensors (cortecs) |
| [Qwen3-14B](https://huggingface.co/unsloth/Qwen3-14B-GGUF) | Q6_K | 12 GB | **158** | GGUF |
| [Qwen3-14B](https://huggingface.co/nvidia/Qwen3-14B-NVFP4) | NVFP4 | 10 GB | 168 | SafeTensors (nvidia) |
| [Qwen3-32B](https://huggingface.co/unsloth/Qwen3-32B-GGUF) | Q4_K_M | 19 GB | — | GGUF |
| [Phi-4-reasoning-plus](https://huggingface.co/nvidia/Phi-4-reasoning-plus-NVFP4) | NVFP4 | 9.0 GB | 157 | SafeTensors (nvidia), fused projections |
| [Gemma-4-12B](https://huggingface.co/AxionML/Gemma-4-12B-NVFP4) | NVFP4 | 11 GB | — | SafeTensors (Modelopt) — **dense** Gemma-4, `gemma4_unified` multimodal wrapper (nested `text_config`, `model.language_model.*` prefix, vision/audio embedders skipped). FFN in NVFP4, attention BF16. |
| [Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF) | Q8_0 | 3.2 GB | 306 | GGUF |
| [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | Q6_K | 19 GB | — | GGUF |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) | Q8_0 | 7.6 GB | — | GGUF |
| [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF) | Q6_K | 12 GB | — | GGUF |
| [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | bf16 | 28 GB | ~30 (eager) | SafeTensors — **MLA** (first Multi-head Latent Attention arch); experts host-offloaded on 32 GB → graphs disabled. **Prefill runs on cuBLAS**: MLA's `head_dim` is 192 (qk_nope 128 + qk_rope 64), which neither FA2 (128/256) nor the tiled FMHA (64/96/128/256/512) serves, so chunks stay bounded by the S-matrix rather than running O(n). Quantizes to NVFP4 in-tree (29.3 → 8.9 GiB, 3.28x; PPL 13.28 on `ppl_corpus_45k`, and note this is a **base** model: it continues text rather than answering instructions). Same-corpus teacher-forced PPL within ~3% of HF bf16 (imp 6.43 vs HF 6.25 on a 534-tok corpus). The earlier "+19.6%" figure was a cross-corpus artifact compounded by a YaRN rope-mscale bug (imp inflated the RoPE cos/sin by 1.261× where HF applies 1.0; fixed 2026-07-07). |
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
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | NVFP4 | 17 GB | 90 | SafeTensors, dense-GDN 27B (64 layers: 16 attn + 48 GDN) and multimodal. The text config is field-identical to Qwen3.6-27B above. **No published NVFP4 export runs on this card**: they are mixed-precision with FP8 attention and `sm_120` has no FP8 GEMM. Quantize the BF16 release yourself (`imp-quantize`, 51.8 → 19.2 GiB; `--format vllm` makes the same checkpoint loadable by vLLM, verified on 0.27.1). Teacher-forced PPL 4.62 on `ppl_corpus_45k.txt` against 7.55 for Qwen3.6-27B on the same corpus. The FP8 release works as a source too and is a third of the download (28.8 GiB), for 0.24% more perplexity; it does **not** run unquantized, its weights need 26 952 MiB and the upload aborts. **Give it room to think**: in a long conversation it spends its token budget on reasoning first, so `max_tokens` below ~400 can return an empty `content` (74-turn session: several empty replies at 260, 74/74 clean at 600; the same session degenerates on vLLM too). **Its embedded `mtp.*` head is taken automatically on a single-stream run** (`speculative.mtp_k=-1` auto): 95.8 -> 141.6 tok/s on a thinking prompt, 2026-08-29. A server taking concurrent requests declines it and says so in `/health`; `--set speculative.mtp_k=2 --set speculative.ngram=false` forces it. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Qwen3.5-27B | MXFP4 | — | — | Loads OOM on 32 GB — see [roadmap](roadmap.md) |

## Mixture-of-Experts

| Model | Quant | VRAM | Decode `tg256` | Format |
|---|---|---:|---:|---|
| [Qwen3-Coder-30B-A3B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | Q6_K | 24 GB | 236 | GGUF |
| [Qwen3-Coder-30B-A3B](https://huggingface.co/NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4) | NVFP4 | 16 GB | 338 | SafeTensors (Modelopt) |
| [Qwen3-30B-A3B](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4) | NVFP4 | 16 GB | 307 | SafeTensors (Modelopt) |
| [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Q4_K_M | 22 GB | 243 | GGUF |
| [Qwen3.6-35B-A3B](https://huggingface.co/mmangkad/Qwen3.6-35B-A3B-NVFP4) | NVFP4 | 18 GB | 320 | SafeTensors (Modelopt); 257 → 320 since the #949 FP8 SSM-projection sidecar closed the FP16 GDN tax |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q4_K_M | 14 GB | 273 (tg128) | GGUF |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q5_K_M | 17 GB | 65 | GGUF, recommended for code-gen |
| [Gemma-4-26B-A4B-it](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) | NVFP4 | 14 GB | 266 | SafeTensors (Modelopt). Quality note: the NVFP4 expert quant reads ~+48% prose PPL vs UD-Q4_K_M (checkpoint-intrinsic, both compute paths — audit 2026-07-13); at near-equal decode speed, prefer UD-Q4_K_M for quality. |
| [Nemotron-3-Nano-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) | NVFP4 | 18 GB | **386** | SafeTensors (Modelopt), Mamba2+attn+MoE. Was 148 and described as "arch-limited" until 2026-08-12: the limit was a CUDA-graph demotion for pure-SSM layers, not the architecture. Lifting it moved decode 127 → 386. |
| [Nemotron-Labs-3-Elastic-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4) | NVFP4 | 18 GB | **381** | SafeTensors (QAD), same arch as Nano; 70 → 381 with the same graph fix |
| [Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | NVFP4+FP8 | 21 GB | **362** | SafeTensors (Modelopt **MIXED_PRECISION**), same arch as Nano at 1M context. First checkpoint here with native FP8 *weights*: 46 Mamba in/out projections FP8, 5935 expert tensors NVFP4. The FP8 half gets an FP16 companion at load (sm_120 has no FP8 prefill GEMM) — 1698 MiB of cache, init at 24.4 of 32.6 GB. Decode serves those weights from the checkpoint's own FP8 bytes (the FP16 copy is prefill-only, since sm_120 has no FP8 prefill GEMM) — measured median +7.5% decode over 27 order-balanced pairs vs. routing decode through FP16. Its MTP head loads via `--mtp-spec-decode` (Nemotron layout) and drafts at 41% top-1 offline, matched by 39% on the serving path since the Mamba2 snapshot fix of 2026-08-20 — but a verify chunk emits only ~1.41 tokens and costs more than that, so k=1 still loses decode rate. Leave `speculative.mtp_k` at 0 here. Measured table in `docs/roadmap.md`. |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | MXFP4 (native) | 15 GB | **391** | SafeTensors; experts converted to NVFP4 at load. Also loads the official GGUF (Q8_0- or bf16-dense + MXFP4 experts, e.g. `gpt-oss-20b-mxfp4.gguf` — the Q8_0 residual rescale was fixed in #808). Harmony chat format (analysis/final channels split into `reasoning_content`/`content`). Use temperature 1.0 — greedy loops in the analysis channel (model-intrinsic). Prefill ≈ 16-19k tok/s (CUTLASS grouped GEMM). |

## Vision

Three multimodal families are supported, in two shapes. Gemma-3 and Gemma-4 keep
their vision encoder in a separate `mmproj.gguf`; Qwen3-VL carries its tower in the
checkpoint, so there is no second file and no second flag.

| Model | Quant | Format | Notes |
|---|---|---|---|
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | BF16 | SafeTensors | Tower ships with the checkpoint — no `--mmproj`. Dynamic resolution (a 1795x2397 photo becomes 972 image tokens), DeepStack taps into the LM's first layers, three-axis M-RoPE |
| [Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4) | NVFP4 | SafeTensors (llm-compressor) | Same tower under a different `vision_config.model_type` (`qwen3_5_moe`), 27 blocks, no DeepStack. Text weights NVFP4, tower BF16 (851.8 MiB) — the mixed precision is the checkpoint's, not a conversion. Tight: init lands at ~31.1 of 32.6 GB. The `mmangkad` Modelopt export in the text table above is a different upload and was not tested for vision |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | NVFP4 | SafeTensors | The same tower a third time, under `vision_config.model_type` `qwen3_5`: 27 blocks, no DeepStack, the same 333 `model.visual.*` tensors and the same `image_token_id`. Only `out_hidden_size` differs (5120), which is the LM width. Tower stays BF16 (878.8 MiB): `imp-quantize` keeps `model.visual.*` at source precision, because the upload path takes F16/BF16/F32 only |
| [Gemma-3-12B-it](https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF) | Q8_0 | GGUF | text + vision, `tg256` 129 |
| [Gemma-3-27B-it](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF) | Q4_K_M | GGUF | largest Gemma-3 |
| [Gemma-4-26B-A4B-it](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | Q4_K_M | GGUF | text + vision via the gemma4v encoder (separate BF16 mmproj), `tg128` 273 — see [`vision_gemma4v_spec.md`](vision_gemma4v_spec.md) |

Several images per request are supported on this tower (repeat `--image`, or
several `image_url` parts), read in prompt order.

`Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`,
`Qwen3_5MoeForConditionalGeneration` and `Qwen3_5ForConditionalGeneration`
are registered; the dense 4B, the Qwen3.6-35B MoE and the dense Qwen3.8-27B
are validated end to end. A VL checkpoint also loads text-only: the tower is
never run without an image.

`vision_config.model_type` decides, matched against an allowlist
(`vision_tower_supported()`): `qwen3_vl` and `qwen3_5_moe` name the same
tower layout. Allowlist rather than shape fingerprint on purpose: a
checkpoint that merely *resembles* the layout must keep hitting the loud
text-only path.

### What cannot see, and how you find out

The list above is exhaustive: **a multimodal SafeTensors checkpoint from any other
family loads text-only.** `Gemma-4-26B-A4B-it-NVFP4` and the Qwen3.5 MoE
checkpoints carry a `vision_config`, but imp's SafeTensors loader only understands
the Qwen3-VL tower layout — for everything else it logs

```
WARN Multimodal model detected (vision_config present, model_type='…').
     imp's SafeTensors loader handles only the language head; the vision tower
     will be skipped.
```

and serves the language model alone. Gemma-4 *does* see images, but only
through the GGUF + `--mmproj` pair in the table, not from its NVFP4
SafeTensors export.

Since #1198 this is visible from the client, not only the server log: a
request carrying `image_url` parts the loaded model cannot use is refused
with **400 `vision_unavailable`** instead of being answered from the text
alone (a fluent description of a picture the model never received is
indistinguishable from a real one).

```bash
# Qwen3-VL — one flag, the tower comes with the model
imp-cli --model models/Qwen3-VL-4B-Instruct/ --image photo.jpg \
        --prompt "Describe this image"

# Gemma — text model plus its mmproj
imp-cli --model gemma-3-12b-it-Q8_0.gguf --mmproj mmproj-google_gemma-3-12b-it-f16.gguf \
        --image photo.jpg --prompt "Describe this image"
```

Image tokens cost VRAM in the encoder workspace, sized by
`runtime.vision_max_patches` (default 4096 ≈ 1024x1024). Larger images are scaled
down to fit that budget rather than refused.

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

# Vision — Qwen3-VL carries its tower; Gemma needs an mmproj
imp-cli --model models/Qwen3-VL-4B-Instruct/ --image photo.jpg --prompt "Describe"
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
