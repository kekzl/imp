<!--
layer: L1
audience: operators
verified: 2026-09-22
commit: 9cbb8004
-->

# Quantization

imp reads GGUF quantization (llama.cpp-compatible files, loaded directly) and SafeTensors NVFP4
prequant (external calibration tools); per-model picks [`MODELS.md`](MODELS.md), benchmark
numbers [`PERF.md`](PERF.md) and [`BENCHMARKS.md`](BENCHMARKS.md), loader/GEMM internals
[`internals/QUANT_PIPELINE.md`](internals/QUANT_PIPELINE.md).

## Formats and where they show up

| Format | Bits / weight | Source | Used for |
|---|---:|---|---|
| Q8_0 / Q6_K / Q5_K_M / Q4_K_M / Q4_0 | 8.0 / 6.5 / 5.5 / 4.5 / 4.5 | GGUF | dp4a GEMV decode + cuBLAS prefill |
| IQ4_NL / IQ4_XS | 4.5 / 4.25 | GGUF | dequant->FP16 cache decode + dequant->cuBLAS prefill (no dp4a/MMVQ kernels) |
| FP8 E4M3 | 8.0 | runtime | KV cache (opt-in), prefill weight cache |
| INT8 | 8.0 | runtime | KV cache (opt-in) |
| INT4 | 4.0 | runtime | KV cache (long-ctx, opt-in) |
| NVFP4 | 4.0 | SafeTensors | weights (decode + prefill), KV cache |
| MXFP4 | 4.5 | GGUF | weights (decode + prefill attention) |

GGUF is mmap'd and uploaded as-is; `*.K` quants store block scales in the format the dp4a
kernels expect. NVFP4 prequant arrives packed with FP8 E4M3 micro-scales (per-16) and an FP32
tensor scale; imp registers it directly into the NVFP4 decode cache and CUTLASS GEMM path, no re-quantization.

## NVFP4 prequant (SafeTensors)

Calibrated per-tensor scales via AWQ or SmoothQuant. Compatible producers:

| Tool | Status |
|---|---|
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (Modelopt) | Primary path. Coherent on Qwen3-Coder-30B, Mistral-3.2, Qwen3.6, Gemma-4 (PR #88 lit up the CUTLASS NVFP4xNVFP4 prefill cache); usage recipe: [`internals/QUANT_PIPELINE.md`](internals/QUANT_PIPELINE.md). |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | Loads; several models degenerate past ~30 tokens. See [roadmap](roadmap.md). Prefer Modelopt. |
| `imp-quantize` (in-tree) | **Experimental.** AWQ-calibrated with `--calib`, round-to-nearest without. Below a published export either way. |

**Loader enforcement** (`quantization_config`, compressed-tensors checkpoints only): a Linear
neither packed nor in the declared `ignore` list, or a packed Linear with no
`weight_global_scale` (would silently default to 1.0, scaling every weight by `absmax / 6`),
both refuse to load; Modelopt checkpoints are exempt (no `ignore` list to check against), full
mechanics: [`internals/QUANT_PIPELINE.md`](internals/QUANT_PIPELINE.md#loader-enforcement-nvfp4-checkpoints).
`lm_head` sits in `ignore` on most exports; imp re-quantizes it anyway (`gemm.nvfp4_lm_head`,
default `auto`), see below; `--set gemm.nvfp4_lm_head=false` serves it at checkpoint precision.

**W4A16 on disk, W4A4 from M >= 2.** These checkpoints declare `input_activations: null`
(W4A16), which imp does not read: at `n == 1` decode the activations stay FP16 (the NVFP4 GEMV
family), from M >= 2 they are quantized to NVFP4 too and the GEMM runs W4A4
(`gemm.nvfp4_smallm`, default on, for M <= 32, the CUTLASS NVFP4 x NVFP4 prefill GEMM above it,
and the batched LM head). Deliberate and measured (+16% at 32 streams, +36% at 8, #1766);
`--set gemm.nvfp4_smallm=false` falls back to FP16 activations.

## imp-quantize: converting a checkpoint yourself (EXPERIMENTAL)

> Pipeline verified end to end; `--calib` recovers a measurable part of the quantization loss,
> but the result still sits below a published Modelopt export. For evaluation/perf, not shipping.

Turns a dense BF16/FP16 or block-scaled FP8 SafeTensors checkpoint into NVFP4:
```bash
# one calibration pass over a corpus, then quantize using it
imp-cli --model ./Qwen3-1.7B --perplexity ./calib_corpus.txt --calibrate ./calib.bin
imp-quantize --model ./Qwen3-1.7B --out ./Qwen3-1.7B-nvfp4 --calib ./calib.bin
imp-cli --model ./Qwen3-1.7B-nvfp4 --prompt "Hello"
```

| Flag | Effect |
|---|---|
| `--model <dir>` / `--out <dir>` | source checkpoint / destination |
| `--calib <file>` | AWQ search using the calibration file; omit for round-to-nearest |
| `--calib-weight abs\|sq` | search error weight, default `abs`; `sq` (2nd moment) is 1.10% better, needs `IMPCAL02` calibration file |
| `--calib-groups <letters>` | restrict AWQ to a subset of `ABCD` (A=q/k/v, B=gate/up, C=o_proj, D=down_proj), production rule below |
| `--lm-head` | also quantize `lm_head` (free at runtime, see below) |
| `--keep-attn-gate` | keep the fused Q+gate `q_proj` (Qwen3.5/Qwen3-Next) at source precision |
| `--keep-gdn-proj [all\|in,gate,out]` | keep GDN `linear_attn` projections at source precision, bare flag = `all` |
| `--format modelopt\|vllm` | output tensor layout, see below |
| `--dry-run` | preview sizes without touching the GPU |

`--calibrate` forces `runtime.deterministic_gemm`: without it two runs of the identical command
differ on most recorded floats and the resulting checkpoints' PPL spreads by ~1.6%; calibrate on
different prose than you score on (`tools/analysis/fetch_calib_corpus.sh` fetches a corpus for
this, not `ppl_corpus_45k.txt`, imp's own scoring corpus). Block-scaled FP8 sources (DeepSeek-V3,
Qwen3.8's FP8 line) work: each weight, including ones kept at full precision, is widened with
its `weight_scale_inv` grid before quantizing (on Qwen3.8-27B that is the MTP draft head, honesty there costs 350 MiB of output).

**Output layout (`--format`)**: `modelopt` (default) writes `.weight` (U8 packed) +
`.weight_scale` (F8_E4M3) + `.weight_scale_2` (F32) declared in `hf_quant_config.json`, read by
imp only; `vllm` (compressed-tensors) writes `.weight_packed` + `.weight_scale` +
`.weight_global_scale` declared in `config.json`'s `quantization_config`, read by imp **and**
`vllm serve`. Three silent-when-wrong differences beyond the names: the tensor scale is stored
inverted (compressed-tensors reads `1 / weight_global_scale`, a divisor; Modelopt stores the
multiplier directly, one convention's number under the other's name scales every weight by
`absmax^2 / 36`); `input_activations` must stay `null`; everything at source precision must be listed in `ignore` (a missing module is one vLLM hunts absent scales for).

### What stays at source precision (Qwen3.8-27B, BF16 source 51.75 GiB)

| Component | Size | Cost | Control |
|---|---:|---|---|
| `embed_tokens` | 2 425 MiB | not possible, no NVFP4 lookup; +0.94% PPL if forced via exact round-trip (Qwen3-0.6B 29.4204 -> 29.6982) | - |
| Vision tower | 875 MiB | loads at source precision only | - |
| MTP draft head | 810 MiB | quantized: draft acceptance 81% -> 0 (#1428) | refused |
| `lm_head` | 2 425 MiB | nothing extra (see below) | `--lm-head` |

`lm_head` is the exception: a native BF16 head becomes an NVFP4 decode cache at load anyway
(`gemm.nvfp4_lm_head`, default `auto` -> on; weights 17 920 -> 16 192 MiB, checkpoint
19.15 -> 17.44 GiB, perplexity 4.6158 either way, greedy output byte-identical over four
prompts). Trade against a real BF16 head (`--set gemm.nvfp4_lm_head=off`, tracked in #982),
Qwen3.8-27B, 248 320-token vocabulary:

[PROV: commit=bca9e9e date=2026-08-16 hw=RTX5090 model=Qwen3.8-27B quant=NVFP4 cuda=13.3
       path=nvfp4-decode-cache n=3-alternating-pairs
       cmd=`imp-cli --prompt ... --max-tokens 128 --temperature 0 --set gemm.nvfp4_lm_head=off|on`;
       ITL from `tools/agent_bench.py --concurrency 1,4,16`, one run per arm]

| `gemm.nvfp4_lm_head` | perplexity | decode, 128 tokens greedy | ITL @4 | ITL @16 |
|---|---:|---:|---:|---:|
| `off` (BF16 head) | **4.5707** | 78.56 tok/s | 53.1ms | 206.2ms |
| `on` (default here) | 4.6158 (+0.99%) | **86.70 tok/s (+10.4%)** | **39.7ms** | **190.1ms** |

Not amortised away by concurrency (head read whole once per token, ~11% of the batch-1 step,
2.43 GiB of 17.9 GiB weights): shrinks +25% at 4 streams -> +8% at 16 but never inverts. vLLM's
`ParallelLMHead` accepts no scales (`no module or parameter named lm_head.weight_global_scale`);
Modelopt / llm-compressor put `lm_head` in `ignore` for W4A4, no other engine has this option.

### Roles that must stay full precision

| Role | Why | Control |
|---|---|---|
| MLA latent projections (`kv_a_proj`, `kv_b_proj`) | runtime slices + reshapes both | always refused |
| MoE router (`.gate.weight`) | FP4 changes the top-k pick | always refused |
| fused Q+gate `q_proj` (Qwen3.5 / Qwen3-Next) | sigmoid-fed gate half, E2M1-sensitive; can't split | `--keep-attn-gate`, but excluding measured ~1.5% *worse* PPL |
| GDN `linear_attn` in_proj_\* / out_proj | quantized by default | `--keep-gdn-proj [all\|in,gate,out]` |
| 3-D expert stacks (gpt-oss, Gemma-4) / fused `q+k+v_proj`, `gate+up_proj` | per-expert split, per-model_type descriptor / one merged linear needs one shared scale | automatic; unknown `model_type` refused |

Bisection evidence, the RMSNorm-offset root cause behind the gate row, MoE per-expert bisection:
[`archive/quantization_awq_findings.md`](archive/quantization_awq_findings.md).

### Quality, `--calib` vs round-to-nearest

`imp-cli --perplexity` over `tools/analysis/ppl_corpus_45k.txt` (13 537 tokens, **not**
`ppl_corpus.txt`, whose 199 tokens invert the model-size trend); reproducible with
`tools/analysis/awq_ppl_ab.sh`.

| Model | BF16 | NVFP4 RTN | NVFP4 `--calib` | gap to BF16 |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 24.08 | 29.42 | **27.60** | +22.2% -> **+14.6%** |
| Qwen3-1.7B (2 shards) | 17.22 | 20.39 | **18.71** | +18.4% -> **+8.7%** |

- Recovers about a quarter of the RTN gap on the 0.6B, nearly two fifths on the 1.7B; does not close it against BF16. **Hurts at 14B and wider** (24-27% worse than RTN, the wrong direction): attention vs FFN, not model size.
- **Production rule**: default `ABCD` on narrow-GQA models (`n_rep < 5`); `--calib-groups BD` on wide-GQA (`n_rep >= 5`), where it beats RTN (9.7922 vs 9.9252, Qwen3-14B) and full `ABCD` does not.
- Uncalibrated `imp-quantize` already beats a published Modelopt export on the one locally comparable model (9.9252 vs 10.0301, Qwen3-14B).
- Mechanism, refuted variants, the won't-fit calibration trick: [`archive/quantization_awq_findings.md`](archive/quantization_awq_findings.md).

## MXFP4 (GGUF)

Same FP4 E2M1 nibble layout as NVFP4, UE8M0 micro-scales (per 32 elements), no separate tensor
scale: the format Blackwell tensor cores expect natively, so MXFP4 prefill goes through CUTLASS
at full throughput; shipped inside GGUF under a proprietary tensor-type code (31), which
llama.cpp reads as the removed `Q4_0_4_4` (cross-tool PPL comparison needs a standard export).
Round-to-nearest MXFP4 is +5-15% perplexity vs Q8_0, worse than Q4_K_M (+2.2% on Qwen3-4B
wikitext-2); MR-GPTQ calibration would close the gap, on the [roadmap](roadmap.md).

## KV cache element type

Set via `--kv-fp8` / `--kv-int8` / `--kv-int4` / `--kv-nvfp4` / `--kv-mxfp4`, or in `imp.conf`:

```toml
[kv_cache]
dtype = "auto"  # auto (default) | fp16 | fp8 | int8 | int4 | nvfp4 | mxfp4
```

| Dtype | Default behaviour | Notes |
|---|---|---|
| `auto` | FP16, upgrades to FP8 E4M3 for models declaring `kv_cache_quant_algo=FP8` on an allowlisted arch family (Qwen3 dense + Qwen3 MoE) | ~768 MiB KV VRAM saved on a 3.9k-token context: Qwen3-14B PPL 13.95 -> 14.10 (+1.07%), Qwen3-30B-A3B ~16.20 -> ~15.99 (neutral), both coherent |
| `fp16` | forced FP16 (`dtype = "fp16"`), opts out of the `auto` FP8 upgrade | |
| `fp8` | forced FP8 E4M3 | default flipped to FP16 in PR #51 (FP8 silently broke Llama, Mistral, DeepSeek at first decode); verified coherent on Qwen3 dense, Qwen3.5/3.6 GDN, Llama-3.2 (warmup-calibration bug fixed in PR #89), Gemma-4 (dual-head_dim carve-out removed in PR #91); opt-in beyond the `auto` allowlist |
| `int8` / `int4` | forced INT8 (dp4a attention) / INT4 | `int4` is VRAM pressure only: coherent, ~22% decode regression at 20K context |
| `nvfp4` / `mxfp4` | forced FP4, E4M3 or UE8M0 micro-scales | chunked prefill via `paged_kv_gather_{nvfp4_to_fp16,mxfp4_kv_to_fp16}` |

The `auto` allowlist and per-family quality gate (`kv_fp8_hint_default_safe` /
`kv_fp8_no_hint_default_safe`, `src/model/model.cpp`) is what makes `fp8` safe to force elsewhere.

## Choosing a quant

Quick guidance, not a benchmark:

- **Q8_0**: cleanest baseline; use when quality matters and VRAM allows.
- **Q4_K_M**: most VRAM-efficient GGUF, sufficient for most chat; degenerates on long code-gen on
  Gemma-4 (use Q5_K_M or Q8_0 there). **Q6_K** is in between, good MoE pick on Qwen3-Coder-30B.
- **IQ4_NL / IQ4_XS**: load and run since #556 via the dequant path (FP16-cache decode like
  Q4_K, dequant->cuBLAS prefill), for community-quant compatibility; at equal VRAM prefer
  Q4_K_M (dedicated dp4a/MMVQ kernels). IQ1/IQ2/IQ3 families unsupported.
- **NVFP4** (SafeTensors prequant): highest decode throughput on prequant-aware models (per-model
  numbers: [`BENCHMARKS.md`](BENCHMARKS.md)); requires AWQ/SmoothQuant calibration, only Modelopt
  fully tested.
- **MXFP4**: GGUF-native FP4, smallest footprint (Qwen3-4B at 2.8 GB); quality lags Q4_K_M without MR-GPTQ calibration.
- Quantizing your own checkpoint: prefer a published Modelopt export; `imp-quantize --calib` is
  experimental (production rule above); the per-block micro-scale search is a closed dead end (refuted, [archive](archive/quantization_awq_findings.md)).
