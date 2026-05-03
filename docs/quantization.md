# Quantization Formats in imp

imp targets Blackwell `sm_120f` and uses two complementary quantization
families: **NVFP4 / MXFP4** (FP4 + microscale, primary path on Blackwell)
and **GGUF K-quants** (Q4_K_M, Q5_K, Q6_K, Q8_0, supported as legacy).
This page covers where to get models for each path and what imp does
with them at runtime; for the kernel-level dispatch see CLAUDE.md and
`docs/SM120_OPTIMIZATION_STATUS.md`.

## Primary path: NVFP4 prequant SafeTensors

NVFP4 (FP4 E2M1 + per-block UE4M3 scales + per-tensor FP32 scale) feeds
directly into Blackwell's 5th-gen Tensor Cores at 3354 TOPS — 2× FP8
and 4× FP16 throughput. imp consumes pre-calibrated NVFP4 SafeTensors
from two upstream toolchains:

| Format | Producer | Layout |
|---|---|---|
| **Modelopt NVFP4** | [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | per-block scale interleaved with weight bytes, per-tensor `weight_scale_2` |
| **llm-compressor NVFP4** | [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) | per-channel `weight_scale` + per-tensor `weight_scale_2`, optional `input_scale` |

imp's `safetensors_loader.cpp` autodetects which layout a checkpoint
uses and dispatches accordingly. Both produce coherent output post
PR #88 (CUTLASS NVFP4×NVFP4 prefill cache). On Blackwell:

- **Decode**: 200+ tok/s on most NVFP4 prequant MoE models
  (Qwen3-Coder-30B 272, Qwen3.6-35B 217, Gemma-4-26B 213,
  Mistral-3.2-24B 101) via the `cache_moe_native_nvfp4`
  contiguous-expert fast-path. CUDA Graphs capture end-to-end; no
  per-layer D2H expert-routing sync.
- **Prefill**: CUTLASS SM120 NVFP4 GEMM directly on packed weights,
  no dequantization round-trip.

### Where to get NVFP4 models

```bash
# NVIDIA — broad coverage of Qwen3 / Llama / Gemma family
huggingface-cli download nvidia/Qwen3-Coder-30B-A3B-Instruct-FP4 \
  --local-dir ./models/Qwen3-Coder-30B-NVFP4

# RedHatAI — llm-compressor NVFP4 catalog (Mistral, Gemma, Qwen)
huggingface-cli download RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --local-dir ./models/Qwen3.6-35B-A3B-NVFP4
```

Or quantize one yourself:

```bash
pip install nvidia-modelopt
python -m modelopt.llm.ptq \
  --model Qwen/Qwen3-8B \
  --quant nvfp4 \
  --output ./Qwen3-8B-NVFP4/
```

### Quality

NVFP4 with proper calibration (AWQ / SmoothQuant per Modelopt, or
llm-compressor's GPTQ) lands within 1% of FP8 perplexity on the
models we've tested. Naive absmax FP4 (no calibration) is 2-3% worse
than Q4_K_M and is not supported — imp expects the calibrated
SafeTensors shipped by the toolchains above.

Known caveats:

- **Mistral-Small-3.2 long-prose** above ~250 tokens: model-level
  SmoothQuant calibration issue, partial workaround in PR #88, see
  TODO.md.
- **Gemma-4 NVFP4 native tool calling**: FP4 quantization depresses
  the `<|tool_call>` (token id 48) emit logit. Tool calling works,
  but for Gemma-4 NVFP4 use Open WebUI's prompt-based "Default"
  function-calling mode rather than "Native". Q4_K_M / Q8_0 GGUF
  Gemma-4 emits the native format reliably.

## MXFP4 (CUTLASS native, attention only)

MXFP4 (FP4 E2M1 + UE8M0 32-element microscales) is the CUTLASS-native
block-scale format used in `attention_fmha_mxfp4_sm120.cu` for prefill
attention. imp converts NVFP4 weights to MXFP4 layout at load time
when the FMHA path requests it. There is no SafeTensors MXFP4 source
format — MXFP4 is an internal kernel format, not a user-facing
checkpoint format.

The `mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` MMA integration is
an open work item (TODO.md) for 2-4× MXFP4 prefill attention.

## Legacy path: GGUF K-quants

GGUF (Q2_K / Q3_K / Q4_0 / Q4_K_M / Q5_K / Q6_K / Q8_0) is fully
supported. imp parses GGUF natively, matches llama.cpp's tensor name
mapping, and runs decode via `dp4a` GEMV kernels (K-parallel for small
matrices, row-parallel with shared-memory cached activations for
large matrices). Use cases:

- Models where no NVFP4 checkpoint exists yet
- Compatibility comparisons with llama.cpp
- Cases where Q5_K_M / Q8_0 quality > NVFP4 quality (Gemma-4 complex
  code-gen is the documented example — Q4_K_M can degenerate on
  Fibonacci-style prompts, Q8_0 stays clean)

There is also an experimental imp-proprietary MXFP4 GGUF tensor-type
31 (legacy from earlier MXFP4 calibration work). New work should
target NVFP4 SafeTensors instead.

## Other quantizations

| Format | Use | Status |
|---|---|---|
| FP8 E4M3 | KV cache (`kv_cache.dtype="fp8"`), prefill weight cache | Default-FP16 KV since PR #51; FP8 opt-in per model after testing — TODO.md has the per-arch table |
| INT8 | KV cache (`kv_cache.dtype="int8"`) | Opt-in |
| INT4 | KV cache (`kv_cache.dtype="int4"`) | Coherent at all ctx lengths but -22% decode at 20K ctx — VRAM-pressure use only |
| TurboQuant | KV cache (`kv_cache.dtype="turboquant"`) | PolarQuant INT4 K + QJL sketch + INT4 V; ~3 bits/elem average |
