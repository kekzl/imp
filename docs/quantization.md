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
| AWQ 4-bit GEMM | 4.0 on disk, 16 in VRAM | SafeTensors | dequantized to FP16 at load, then served as FP16 ([below](#awq-safetensors)) |
| GPTQ 4-bit | 4.0 on disk, 16 in VRAM | SafeTensors | dequantized to FP16 at load, then served as FP16 ([below](#gptq-safetensors)) |

GGUF is mmap'd and uploaded as-is; `*.K` quants store block scales in the format the dp4a
kernels expect. NVFP4 prequant arrives packed with FP8 E4M3 micro-scales (per-16) and an FP32
tensor scale; imp registers it directly into the NVFP4 decode cache and CUTLASS GEMM path, no re-quantization.

## AWQ SafeTensors

| `quantization_config` | Result |
|---|---|
| `quant_method: awq`, `bits: 4`, `zero_point: true`, `version: gemm` (or absent) | loads; every q/k/v/o/gate/up/down `qweight` dequantized to FP16 at upload, `w = (q - z) * s` in AutoAWQ's packing order |
| GEMV, Marlin, `bits != 4`, `zero_point: false` | refused at load, log names the detected variant |
| a `.qweight` outside those seven projections, or a shape that is not `[K, N/8]` / `[K/g, N/8]` / `[K/g, N]` | refused at load |

- VRAM holds FP16 weights: 4x the checkpoint's weight bytes. For 4-bit weights in VRAM use an NVFP4 export.
- Checked: CPU reference bit-exact vs hand-packed tensors (`tests/test_dequant_awq.cpp`), kernel vs reference (`tests/test_dequant_awq_gpu.cu`), host dequant byte-identical to an AutoAWQ-source dequant (`tools/analysis/awq_ref_ppl.py`) on all 168 / 252 projections of Qwen2.5-0.5B-Instruct-AWQ / Qwen3-4B-AWQ, PPL within 1 % of that reference and first token vs the original (`scripts/accept_2205.sh`).
- Int4 cost, Qwen2.5-0.5B-Instruct-AWQ vs BF16 original, 45k corpus, transformers reference (FP32, FP8 head): PPL 26.1835 vs 21.0931, +24.1 %.

## GPTQ SafeTensors

| Config (`quantize_config.json`, else `config.json` `quantization_config`) | Result |
|---|---|
| `bits: 4`, `checkpoint_format` / `format` absent or `gptq` (v1) | loads; `w = (q - ((z + 1) & 0xF)) * s` |
| `bits: 4`, `checkpoint_format: gptq_v2` | loads; `w = (q - z) * s` |
| `marlin`, `bitblas`, `is_marlin_format: true`, any other format, `bits != 4` | refused at load, log names the detected config |
| a `.qweight` outside q/k/v/o/gate/up/down, or shapes not qweight `[K/8, N]`, qzeros `[ceil(K/g), N/8]`, scales F16 `[ceil(K/g), N]`, g_idx `[K]` in range | refused at load |

- Layout as AutoGPTQ `qlinear_cuda_old.py`: qweight packed along K, qzeros packed along N; `group_size: -1` is one group.
- VRAM holds FP16 weights: 4x the checkpoint's weight bytes.
- Checked: CPU reference bit-exact vs hand-packed tensors (`tests/test_dequant_gptq.cpp`), kernel vs reference (`tests/test_dequant_gptq_gpu.cu`), PPL within 1 % of an independent AutoGPTQ-dequant transformers reference and first token vs the original (`scripts/accept_2249.sh`, `tools/analysis/gptq_ref_ppl.py`).
- Int4 cost, Qwen2.5-0.5B-Instruct-GPTQ-Int4 vs BF16 original, 45k corpus: PPL +17.2 % in imp, +16.9 % in the transformers reference.

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
default `auto` = per-row FP8), see below; `--set gemm.nvfp4_lm_head=false` serves it at checkpoint precision.

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
| `--calib-weight abs\|sq` | search error weight, default `abs`; `sq` (2nd moment, needs an `IMPCAL02` file) is 1.10% better on Qwen3-0.6B, 0.15 PPL worse on Qwen3-14B `BD` |
| `--calib-groups <letters>` | subset of `ABCDEG` (A=q/k/v, B=gate/up, C=o_proj, D=down_proj, E/G=GDN); default `ABCDEG` at `n_rep < 5`, `BDEG` at `n_rep >= 5` on dense models (GDN hybrids keep `ABCDEG`: Qwen3.8-27B ABCD 4.5986 vs BDEG 4.6136) |
| `--lm-head [fp8\|nvfp4\|source]` | how `lm_head` is written: `fp8` (default for `modelopt`) = the per-row FP8 head `auto` runs, served without load-time conversion; `nvfp4` (bare flag) = quantized like any Linear; `source` (default for `vllm`) = copied as is ([below](#what-stays-at-source-precision-qwen38-27b-bf16-source-5175-gib)) |
| `--kv-hint auto\|fp8\|none` | `kv_cache_quant_algo` in `hf_quant_config.json` (`modelopt` only): `auto` (default) = `FP8` for `model_type` `qwen3` / `qwen3_moe`, the families with a measured FP8 KV PPL ([KV cache](#kv-cache-element-type)), else `null` |
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

### Recipe and reproducibility

`producer.recipe` in `hf_quant_config.json` (`vllm`: `imp_recipe.json`) records `recipe_version` 1,
`imp_version`, `imp_tree` (git tree id of the build, `unknown` outside `make build`), `format`,
`lm_head`, `kv_hint`, `kv_cache_fp8`, the keep flags and `calibration` (`null`, or `--calib` file
`file_sha256`, `model_id`, `samples`, `entries`, resolved `groups`, `weight`, `n_rep`, `hybrid`); no
timestamps or output paths. Same binary + recipe + calibration file = byte-identical export: `make
test-quantize-repro` (GPU, Qwen3-0.6B, two `--calib` exports, 9 of 9 files sha256-equal).

### What stays at source precision (Qwen3.8-27B, BF16 source 51.75 GiB)

| Component | Size | Cost | Control |
|---|---:|---|---|
| `embed_tokens` | 2 425 MiB | not possible, no NVFP4 lookup; +0.94% PPL if forced via exact round-trip (Qwen3-0.6B 29.4204 -> 29.6982) | - |
| Vision tower | 875 MiB | loads at source precision only | - |
| MTP draft head | 810 MiB | quantized: draft acceptance 81% -> 0 (#1428) | refused |
| `lm_head` | 2 425 MiB | written as the per-row FP8 head by default: 1 213.4 MiB, nothing converted at load (below) | `--lm-head` |

`lm_head` is the exception: a native BF16 head is re-quantized at load anyway (default `auto` = FP8
since #2166; the table below is the older NVFP4 default, `gemm.nvfp4_lm_head=on`; weights 17 920 -> 16 192 MiB, checkpoint
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
| `on` (NVFP4, default until #2166) | 4.6158 (+0.99%) | **86.70 tok/s (+10.4%)** | **39.7ms** | **190.1ms** |

Not amortised away by concurrency (head read whole once per token, ~11% of the batch-1 step,
2.43 GiB of 17.9 GiB weights): shrinks +25% at 4 streams -> +8% at 16 but never inverts. vLLM's
`ParallelLMHead` accepts no scales (`no module or parameter named lm_head.weight_global_scale`);
Modelopt / llm-compressor put `lm_head` in `ignore` for W4A4, no other engine has this option.

`auto` serves the head as per-row FP8 E4M3 (#2156, default since #2166; `on` keeps NVFP4): 1.78x the
NVFP4 head bytes, FP16 activations on every row count (decode, batch, `--perplexity`), numbers from #2176:

| Model | PPL 45k NVFP4 | PPL 45k FP8 | tg128 |
|---|---:|---:|---:|
| Qwen3-8B-Q8_0 | 11.1108 | 10.7623 | -4.7 % |
| Qwen3-30B-A3B-NVFP4 | 11.8443 | 11.3476 | -3.0 % |
| Qwen3.8-Flash-Next-NVFP4 | 4.6493 | 4.4873 | host-expert bound |

`imp-quantize --lm-head fp8` (default for `--format modelopt`, #2479) writes this head on disk:

- Tensors: `lm_head.weight` F8_E4M3 `[V, D]` + `lm_head.weight_row_scale` F32 `[V]`, codes from the runtime's own kernel on the loader's FP16 bits.
- Load: served directly in every `gemm.nvfp4_lm_head` mode, log `checkpoint E4M3 per-row scales served directly, no load-time conversion`.
- `lm_head` stays in `exclude_modules`, the embedding at source precision; a checkpoint without an `lm_head.weight` tensor (tied) has no head to write.
- `--format vllm` keeps `source`: vLLM's `ParallelLMHead` takes no scales.

| Qwen3-14B export | head on disk | PPL 45k, KV FP16 |
|---|---:|---:|
| `--lm-head source` (runtime builds the FP8 head at load) | 1483.8 MiB BF16 | 9.7350 |
| `--lm-head fp8` (default) | 742.5 MiB | 9.7350 |

[PROV: commit=07d5fc99 date=2026-10-02 hw=RTX5090 model=Qwen3-14B quant=NVFP4-RTN cuda=13.4.1
       path=imp-quantize+imp-cli-perplexity n=1-per-arm-deterministic
       cmd=`imp-cli --perplexity ppl_45k.txt --set runtime.deterministic_gemm=true --set kv_cache.dtype=fp16`
       note=measured on the same diff before the rebase onto 49236dfb (MoE permute only)]

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

`imp-cli --perplexity` over `tools/analysis/ppl_corpus_45k.txt` (13 537 tokens, rebuilt by `tools/analysis/make_ppl_corpus.sh`, **not**
`ppl_corpus.txt`, whose 199 tokens invert the model-size trend); reproducible with
`tools/analysis/awq_ppl_ab.sh`.

| Model | BF16 | NVFP4 RTN | NVFP4 `--calib` | gap to BF16 |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 24.08 | 29.42 | **27.60** | +22.2% -> **+14.6%** |
| Qwen3-1.7B (2 shards) | 17.22 | 20.39 | **18.71** | +18.4% -> **+8.7%** |

- Recovers about a quarter of the RTN gap on the 0.6B, nearly two fifths on the 1.7B; does not close it against BF16. **Hurts at 14B and wider** (24-27% worse than RTN, the wrong direction): attention vs FFN, not model size.
- **Production rule, the code default since roadmap row 6 closed**: all groups on narrow GQA (`n_rep < 5`), attention groups A and C off on dense wide GQA (`n_rep >= 5`; GDN hybrids keep all, Qwen3.8-27B ABCD 4.5986 vs BDEG 4.6136, `docs/audit/AUDIT_qwen38_nvfp4.md`), `--calib-weight abs` on both. Qwen3-14B BF16 source, `ppl_corpus_45k.txt`, deterministic:

| arm | RTN | ABCD `abs` | ABCD `sq` | BD `abs` | BD `sq` |
|---|---:|---:|---:|---:|---:|
| PPL | 9.9849 | 12.2634 | 10.7965 | **9.9068** | 10.0563 |

[PROV: commit=c6135a26 date=2026-10-01 hw=RTX5090 model=Qwen3-14B quant=NVFP4 cuda=13.4.1
       path=imp-quantize+imp-cli-perplexity n=1-per-arm-deterministic
       cmd=`tools/analysis/awq_wide_gqa_ab.sh` harness_md5=31019ae9
       note=calibration stats from the RTN checkpoint (BF16 14B does not fit for --calibrate)]


- `sq` removes 64 % of the ABCD damage (12.2634 -> 10.7965) and still loses to RTN; on BD it costs 0.15.
- On a dense model the default `BDEG` folds the same sites as `BD` (E/G need GDN tensors; `WideGqaDefaultIsBdOnADenseLayer`).
- The threshold sits at the measured wide point (n_rep 5); n_rep 3-4 (Qwen3-8B, Qwen3-4B, Phi-4) is unmeasured and keeps all groups, as before.
- Uncalibrated `imp-quantize` already beats a published Modelopt export on the one locally comparable model (9.9252 vs 10.0301, Qwen3-14B).
- Mechanism, refuted variants, the won't-fit calibration trick: [`archive/quantization_awq_findings.md`](archive/quantization_awq_findings.md).

## MXFP4 (GGUF)

Same FP4 E2M1 nibble layout as NVFP4, UE8M0 micro-scales (per 32 elements), no separate tensor
scale: the format Blackwell tensor cores expect natively, so MXFP4 prefill goes through CUTLASS
at full throughput; shipped inside GGUF under a proprietary tensor-type code (31), which
llama.cpp reads as the removed `Q4_0_4_4` (cross-tool PPL comparison needs a standard export).
Round-to-nearest MXFP4 is +5-15% perplexity vs Q8_0, worse than Q4_K_M (+2.2% on Qwen3-4B
wikitext-2); MR-GPTQ calibration would close the gap: [roadmap](roadmap.md#later) row 101.

## KV cache element type

Set via `--kv-fp8` / `--kv-int8` / `--kv-int4` / `--kv-nvfp4` / `--kv-mxfp4`, or in `imp.conf`:

```toml
[kv_cache]
dtype = "auto"  # auto (default) | fp16 | fp8 | int8 | int4 | nvfp4 | mxfp4
```

| Dtype | Default behaviour | Notes |
|---|---|---|
| `auto` | FP16, upgrades to FP8 E4M3 for models declaring `kv_cache_quant_algo=FP8` on an allowlisted arch family (Qwen3 dense + Qwen3 MoE); hint-less checkpoints (every GGUF): Qwen3 MoE only, Qwen3 dense stays FP16 (#2208: FP8 KV flipped Qwen3-8B-Q8_0 greedy output) | ~768 MiB KV VRAM saved on a 3.9k-token context: Qwen3-14B PPL 13.95 -> 14.10 (+1.07%), Qwen3-30B-A3B ~16.20 -> ~15.99 (neutral), both coherent |
| `fp16` | forced FP16 (`dtype = "fp16"`), opts out of the `auto` FP8 upgrade | |
| `fp8` | forced FP8 E4M3 | default flipped to FP16 in PR #51 (FP8 silently broke Llama, Mistral, DeepSeek at first decode); verified coherent on Qwen3 dense, Qwen3.5/3.6 GDN, Llama-3.2 (warmup-calibration bug fixed in PR #89), Gemma-4 (dual-head_dim carve-out removed in PR #91); opt-in beyond the `auto` allowlist |
| `int8` / `int4` | forced INT8 (dp4a attention) / INT4 | `int4` is VRAM pressure only: coherent, ~22% decode regression at 20K context |
| `nvfp4` / `mxfp4` | forced FP4, E4M3 or UE8M0 micro-scales | chunked prefill via `paged_kv_gather_{nvfp4_to_fp16,mxfp4_kv_to_fp16}` |

`imp-quantize --kv-hint auto` (default, #2480) declares the hint for the two families measured above:

- `model_type` `qwen3` / `qwen3_moe`: `kv_cache_quant_algo: "FP8"`; every other family: `null`; `--kv-hint fp8|none` overrides.
- Qwen3-14B NVFP4 export, `ppl_45k.txt`, deterministic: KV FP16 9.7350, KV FP8 under `auto` 9.7642 (+0.30 %).

The `auto` allowlist and per-family quality gate (`kv_fp8_hint_default_safe` /
`kv_fp8_no_hint_default_safe`, `src/model/model.cpp`) is what makes `fp8` safe to force elsewhere.

## Choosing a quant

Quick guidance, not a benchmark:

- **Q8_0**: cleanest baseline; use when quality matters and VRAM allows.
- **Decode on dense Q8_0 / Q6_K / Q5 GGUF** runs an NVFP4 copy of the weights by default
  (`NVFP4 decode: auto -> mode 1`); prefill stays at source precision. `--no-nvfp4` or
  `diagnostics.no_nvfp4_decode_cache = true` decodes at source precision.

| model | decode-path PPL, source -> default | tg128 source -> default | llama.cpp tg128 |
|---|---|---|---:|
| Qwen3-8B Q8_0 | 10.7277 -> 11.3713 (+6.0 %) | 159.31 / 158.96 -> 292.78 / 295.59 | 160.1 |
| Qwen3-14B Q6_K | 9.2289 -> 9.4524 (+2.4 %) | 115.39 / 115.55 -> 169.53 / 168.81 | 114.8 |

[PROV: commit=38e435dd date=2026-09-24 hw=RTX5090 model=Qwen3-8B,Qwen3-14B quant=Q8_0,Q6_K cuda=13.4
       path=gguf-nvfp4-decode-cache n=2-alternating-runs
       cmd=`imp-cli --perplexity ppl_corpus_45k.txt --prefill-chunk-size 1 --set runtime.deterministic=true [--no-nvfp4]`;
       tg from `imp-cli --bench --bench-pp 512 --bench-reps 5 --max-tokens 128 --set speculative.ngram=false`;
       llama.cpp column: BENCHMARKS.md competitive table (2026-08-30)]

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
