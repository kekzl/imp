# Quantization

imp supports both GGUF quantization (loaded directly from llama.cpp-compatible files) and SafeTensors NVFP4 prequant (produced by external calibration tools). This page explains what each format is, where it is used inside the engine, and what the trade-offs are.

For per-model picks see [`supported-models.md`](supported-models.md). For benchmark numbers see [`performance.md`](performance.md).

## Formats and where they show up

| Format | Bits / weight | Source | Used for |
|---|---:|---|---|
| Q8_0 | 8.0 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q6_K | 6.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q5_K_M | 5.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q4_K_M | 4.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| Q4_0 | 4.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| IQ4_NL / IQ4_XS | 4.5 / 4.25 | GGUF | dequant→FP16 cache decode + dequant→cuBLAS prefill (no dp4a/MMVQ kernels) |
| FP8 E4M3 | 8.0 | runtime | KV cache (opt-in), prefill weight cache |
| INT8 | 8.0 | runtime | KV cache (opt-in) |
| INT4 | 4.0 | runtime | KV cache (long-ctx, opt-in) |
| NVFP4 | 4.0 | SafeTensors | weights (decode + prefill), KV cache |
| MXFP4 | 4.5 | GGUF | weights (decode + prefill attention) |

GGUF formats are mmap'd from disk and uploaded as-is to GPU; the `*.K` quants store block scales in the format the dp4a kernels expect. NVFP4 prequant arrives in two-byte-per-element packed form with FP8 E4M3 micro-scales (per-16) and an FP32 tensor scale; imp registers these directly into the NVFP4 decode cache and CUTLASS NVFP4 GEMM path with no re-quantization.

## NVFP4 prequant (SafeTensors)

Calibrated per-tensor scales using AWQ or SmoothQuant. Two upstream tools produce compatible files:

| Tool | Status |
|---|---|
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (Modelopt) | Primary path. Coherent on Qwen3-Coder-30B, Mistral-3.2, Qwen3.6, Gemma-4 (after PR #88 lit up the CUTLASS NVFP4×NVFP4 prefill cache). |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | Loads, but several models degenerate past ~30 tokens. See [roadmap](roadmap.md). Prefer Modelopt where available. |
| `imp-quantize` (in-tree) | Uncalibrated round-to-nearest — works, but costs measurably more quality than the two above. See below. |

### imp-quantize — converting a checkpoint yourself

`imp-quantize` turns a dense BF16/FP16 SafeTensors checkpoint into an NVFP4 one
without leaving the repo, for models nobody has published an export for:

```bash
imp-quantize --model ./Qwen3-1.7B --out ./Qwen3-1.7B-nvfp4   # --dry-run to preview
imp-cli --model ./Qwen3-1.7B-nvfp4 --prompt "Hello"
```

It writes the layout the loader already recognises — `<prefix>.weight` (U8,
packed nibbles), `.weight_scale` (F8_E4M3 micro-scales), `.weight_scale_2`
(F32 tensor scale), plus `hf_quant_config.json` — copies the tokenizer and
config files, and rebuilds the shard index when the source is sharded.
Embeddings, norms and (unless `--lm-head`) the LM head stay full precision.

**Quality: worse than a calibrated export, measurably.** The scales are plain
round-to-nearest over the weights; there is no activation calibration, so
nothing protects the channels that matter most. Measured on 2026-07-26 with
`imp-cli --perplexity` over the same 199-token corpus, `deterministic_gemm`:

| Model | BF16 | imp-quantize NVFP4 | cost |
|---|---:|---:|---:|
| Qwen3-0.6B | 13.51 | 19.19 | +42% |
| Qwen3-1.7B | 8.93 | 14.01 | +57% |

The loss does not shrink with size across that pair, which points at the
missing calibration rather than a small-model artifact. Output is coherent and
factually correct on short prompts, and the pipeline is verified end to end
(the result loads, is detected as `NVFP4 model (Model Optimizer)`, and
generates) — but **prefer a Modelopt export when one exists**. Adding
AWQ/SmoothQuant-class calibration is the open work; see [roadmap](roadmap.md).

MoE checkpoints are not supported yet: expert stacks are 3-D and need the
per-expert path. They are reported and left unquantized rather than mangled.

Workflow with Modelopt:

```bash
pip install nvidia-modelopt

python -m modelopt.llm.ptq \
  --model Qwen/Qwen3-8B \
  --quant nvfp4 \
  --output ./Qwen3-8B-nvfp4/

imp-cli --model ./Qwen3-8B-nvfp4/ --prompt "Hello"
```

Modelopt quantization modes:

| Mode | What's quantized |
|---|---|
| `nvfp4` | all linear layers |
| `nvfp4_mlp_only` | MLP / FFN layers only |
| `nvfp4_experts_only` | MoE expert layers only |
| `nvfp4_omlp_only` | MLP + output projection |

### NVFP4 internal pipeline

Dense layers:

```
SafeTensors NVFP4 packed weights + scales
  → loader (BF16 norms / router → FP16, packed FP4 stays packed)
  → Phase 0: register in NVFP4 decode cache (no re-quant)
  → Phase 3b: CUTLASS scale-factor layout (SfAtom) for prefill
  → prefill: CUTLASS NVFP4 GEMM via gemm_dispatch() (sm_120 tensor cores)
  → decode:  NVFP4 GEMV (prmt register LUT, K-parallel)
```

MoE layers (Modelopt SafeTensors, per-expert):

```
SafeTensors per-expert weights
  → cache_moe_native_nvfp4 builds one contiguous [ne, N, K_packed]
    buffer per layer per projection (D2D-memcpy from per-expert tensors)
  → per-expert tensors freed inline (32 GiB VRAM ceiling on 35B-A3B)
  → CUDA Graphs capture cleanly via the decode fast-path
```

Without `cache_moe_native_nvfp4` the legacy FP16 dequant + cuBLAS sm_80 WMMA fallback fires per layer per token, killing CUDA Graphs and dropping decode 5–17×.

NVFP4 KV cache (`--kv-nvfp4`) supports chunked prefill since PR #149 — past chunks' K/V are gathered from the paged cache via `paged_kv_gather_nvfp4_to_fp16` (PTX `cvt.rn.f16x2.e2m1x2` inner loop + UE4M3 scale fold) and concatenated with the current chunk before rectangular cuBLAS attention. Hybrid GDN+MoE / Mamba2+MoE archs (Qwen3.5/3.6, Nemotron-H) are in scope since PR #156.

## MXFP4 (GGUF)

MXFP4 uses the same FP4 E2M1 nibble layout as NVFP4 but with UE8M0 micro-scales (per 32 elements) and no separate tensor scale. This matches the format the Blackwell tensor cores expect natively, so MXFP4 prefill goes through CUTLASS at full FP4 throughput.

imp ships MXFP4 inside GGUF using a proprietary tensor-type code (31). llama.cpp reads this as the removed `Q4_0_4_4` format, so cross-tool perplexity comparison is not possible without a standard MXFP4 export.

Round-to-nearest MXFP4 is +5–15% perplexity vs Q8_0, worse than Q4_K_M (+2.2% on Qwen3-4B wikitext-2). MR-GPTQ calibration would close this gap; it is on the [roadmap](roadmap.md).

## KV cache element type

Set via `--kv-fp8` / `--kv-int8` / `--kv-int4` / `--kv-nvfp4` / `--kv-mxfp4`, or in `imp.conf`:

```toml
[kv_cache]
dtype = "auto"  # auto (default) | fp16 | fp8 | int8 | int4 | nvfp4 | mxfp4
```

`dtype = "auto"` (the default) keeps FP16 but upgrades to FP8 E4M3 for models whose author declares `kv_cache_quant_algo=FP8` (Modelopt NVFP4 checkpoints) **and** whose arch family has passed the long-context FP8-KV quality gate (`kv_fp8_hint_default_safe` in `src/model/model.cpp`). Currently allowlisted: **Qwen3 dense + Qwen3 MoE** — measured on a 3.9k-token context, FP8 vs FP16 KV: Qwen3-14B PPL 13.95→14.10 (+1.07%), Qwen3-30B-A3B ~16.20→~15.99 (neutral), both coherent, ~768 MiB KV VRAM saved. Other hint-declaring families (Phi-4, Nemotron-H, Qwen3.5/3.6, Gemma-4) stay FP16 until measured; pass `--kv-fp8` to force, or `dtype = "fp16"` to opt out.

The default flipped to FP16 in PR #51 — FP8 had been silently breaking Llama, Mistral, and DeepSeek at first decode. Beyond the `auto` allowlist, FP8 stays opt-in; it is verified coherent on Qwen3 dense, Qwen3.5 / 3.6 GDN, Llama-3.2, and Gemma-4 (FP8 KV warmup-calibration bug fixed in PR #89; Gemma-4 dual-head_dim carve-out removed in PR #91).

INT4 KV is for VRAM-pressure cases only — coherent but ~22% decode regression at 20K context. NVFP4-KV (`--kv-nvfp4`) and MXFP4-KV (`--kv-mxfp4`, PR #249) both store FP4 at 25% of FP16 — NVFP4 uses E4M3 micro-scales, MXFP4 uses UE8M0; both ship chunked prefill via `paged_kv_gather_*_to_fp16`. (TurboQuant was retired in PR #251; its deprecated `--kv-turboquant{,-lite}` alias flags were removed 2026-07-07.)

## Choosing a quant

Quick guidance, not a benchmark:

- **Q8_0** is the cleanest baseline. Use it when output quality matters and VRAM allows.
- **Q4_K_M** is the most VRAM-efficient GGUF. Sufficient for most chat; can degenerate on long code-gen on Gemma-4 — use Q5_K_M or Q8_0 there.
- **Q6_K** sits in between. Good MoE pick on Qwen3-Coder-30B.
- **IQ4_NL / IQ4_XS** (i-quants) load and run since #556 via the dequant path (FP16-cache decode like Q4_K, dequant→cuBLAS prefill). Supported for community-quant compatibility — at equal VRAM prefer Q4_K_M, which has dedicated dp4a/MMVQ kernels. The IQ1/IQ2/IQ3 families remain unsupported.
- **NVFP4** (SafeTensors prequant) gives the highest decode throughput on prequant-aware models (current per-model numbers in [`BENCHMARKS.md`](BENCHMARKS.md)). Requires AWQ/SmoothQuant calibration; only Modelopt is fully tested.
- **MXFP4** is GGUF-native FP4. Smallest footprint (Qwen3-4B at 2.8 GB), but quality lags Q4_K_M without MR-GPTQ calibration.
