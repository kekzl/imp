---
name: quant-formats
description: Use when working on imp's quantization formats, loaders, or dequant paths - GGUF Q4_0...Q8_0/Q*_K/IQ4/MXFP4, NVFP4 two-level scaling (Modelopt vs compressed-tensors layouts), FP8 E4M3, StorageTier, decode cache, KV-cache dtypes (auto/fp16/fp8/int8/int4/nvfp4/mxfp4), "which quant should I use", "which KV dtype", scale-factor layout, dequant kernel wiring, imp-quantize / AWQ, judging quant quality (PPL). Do NOT use for writing/optimizing GEMM/GEMV kernels (sm120-cuda-expert) or measuring quant perf (benchmark-cuda).
---

# Quantization Formats & Pipelines - imp

Sources of truth: `docs/quantization.md` (formats, choosing a quant), `docs/internals/QUANT_PIPELINE.md` (files, GEMM-dispatch registry, boundary rules). This skill carries the gotchas only.

## The two worlds

| | GGUF | SafeTensors NVFP4 (prequant) |
|---|---|---|
| Formats | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, IQ4_NL/XS, MXFP4 | per-tensor scale (FP32) + per-16 FP8-E4M3 micro-scales |
| Decode | dp4a GEMV per qtype + NVFP4 decode cache built at init | native NVFP4 GEMV / smallm v2 |
| Prefill | cuBLAS on dequanted source | CUTLASS NVFP4 GEMM (+ stream-K, #1841) |
| Priority | legacy; community MXFP4 quality bugs: do not sink time | the strategic path |

- GGUF -> NVFP4 decode cache at init; `nvfp4_beneficial` weights skip the FP16 cache; GDN/SSM projections excluded (quality lock) and served by `gemm.fp8_ssm_proj` (default on, native F16 and GGUF-Q8_0 since #962). `gemm.nvfp4_attn_proj` opt-in.
- gpt-oss native MXFP4 experts convert to NVFP4 at load.
- `wcache_.nvfp4` is NOT a GGUF predicate (native models fill it too); GGUF = registry scan on `dequant_gpu_supported(source)` (#1792 trap).

## StorageTier is the dispatch contract (`src/core/storage_tier.h`)

`Undefined` (FATAL if dispatched), `FP32`, `FP16`, `FP8` (E4M3 + per-tensor scale), `NVFP4` (two-level micro-scale, decode-GEMV), `CUTLASS_NVFP4` (block-scaled, CUTLASS sm_120 fast path), `MXFP4` (CUTLASS FMHA).

- ONE decision point since #621: `plan_storage()` in `src/runtime/storage_planner.h`; caches are RAII-owned by the executor. Extend the planner's rules, never add downstream overrides.
- `NVFP4` != `CUTLASS_NVFP4` at large M: plain NVFP4 falls to `gemm_nvfp4` (dequant -> cuBLAS). SfAtom scales are built at load (`src/exec/pre_dequant_*.cu`); `convert_scales_sfatom` is a load-time artifact.
- M<=32 since #1766: `gemm.nvfp4_smallm` (`_impl=2`, `_pair` #1788) runs `mma.sync.kind::mxf4nvf4` on the PLAIN packed layout (`src/quant/nvfp4_gemm_smallm_v2.cu`, zero extra VRAM; the CUTLASS SF atom static-asserts 128 rows, so no CTA_M<128 tile exists). +16.0% at 32 streams, +36.0% at 8. At M=1 the GEMV family stays (#1789).
- StoragePlanner prices zero-copy native-NVFP4 entries at incremental cost since #1795 (a ~15 GiB phantom demand printed "plan failed" before).

## MoE

Per-expert NVFP4 tensors are packed into one contiguous `[ne, N, K_packed]` buffer per layer/projection at load (`cache_moe_native_nvfp4`); that is what makes graph capture possible (per-step dequant + cuBLAS is 5-17x slower, no graphs). Grouped-GEMM prefill class is closed at ~60% of the weight floor (sm120-cuda-expert).

## KV cache dtypes

`kv_cache.dtype = auto | fp16 | fp8 | int8 | int4 | nvfp4 | mxfp4` (CLI `--kv-fp8` etc.).

- `auto` (default) has three upgrade arms in `src/runtime/engine_init_resolver.cpp` gated by `src/model/model_arch.h` lists (`kv_nvfp4_default_safe`, `kv_fp8_hint_default_safe`, `kv_fp8_no_hint_default_safe`); an arch missing from them silently gets FP16 KV. Legacy: `kv_cache.fp8_auto_legacy`; FP8 nondeterminism opt-in `allow_nondeterministic_fp8`.
- QWEN35 family (Qwen3.5/3.6/3.8 dense hybrids) resolves to NVFP4 since #1750: Qwen3.8-27B `max_model_len` 48512 -> 131072 for +0.29-0.35% PPL; FP8 86 848 -> NVFP4 126 432 tokens (+45.6%). MoE siblings keep the old default.
- NVFP4 KV decode was 13.5% slower than FP8 until #1817 (byte loads: 20 `LDG.E.U8` per iteration); now 2.3% faster at 77k (74.1 vs 72.4 tok/s). `PagedOracle.HD256_Sweep` covers the shipped 24q/4kv shape across all 7 dtypes.
- `--kv-fp8` / `IMP_KV_FP8=1` on that family DOUBLES KV bytes (correct when auto meant FP16); the pin logs its context cost since #1823 (`kv_pin_context_cost_factor`, `kv_dtype_is_explicit_pin` in `model.cpp`).
- Sparse decode attention (`attention.sparse_topk_tokens`) covers F16/FP8/NVFP4 KV since #1818; the NVFP4 branches had to consume the compacted block table (a dtype gate alone = silent dense no-op). Budget arithmetic in `src/exec/sparse_attn_geometry.h` uses the resolved block size (32 on `n_kv_heads <= 4`, #1819).
- `kv_cache.swa_snapshot_mb` below the snapshot size switches prefix caching OFF (#1092).
- GDN recurrent state: `gdn.state_bf16` (default on since #1778): BF16 storage, FP32 register arithmetic, +12.5% at 32 streams, +0.21% PPL, state pool 4848 -> 2544 MiB; resolver keeps FP32 under `gdn.ref_kernel` or unsupported HD/SS (WARN on stderr). FP16 state REFUTED (subnormal truncation ~6e-5).

## Judging quantization quality

- Corpus `tools/analysis/ppl_corpus_45k.txt` (13 537 tokens); the 199-token `tools/analysis/ppl_corpus.txt` inverts verdicts (+42%/+57% vs +25%/+19%; +1.0% vs -0.03% on FP8 SSM).
- `--set runtime.deterministic=true` both arms (implies `runtime.deterministic_gemm`; 0.35% run-to-run otherwise); `--set speculative.mtp_k=0` (`--perplexity` otherwise loads the MTP head, +0.79 GiB, floors the 35B KV pool).
- Qwen3.6-35B PPL moves +-0.2..0.5% between fp32-equivalent kernels (routing flips): >1% = broken, below = no verdict. Numerics judge: Qwen3.8-27B-NVFP4-vllm (deterministic, fused GDN 4.6283).
- 35B recipe when model + caches fill 30 GB: corpus in ~1k-token slices (3200 chars), `--max-seq-len 1280`, both arms per slice, token-weighted NLL; warm-cache mount (`/root/.cache/imp/warm`) makes runs ~7 s; do NOT mount the library-reserve measurement (4399 MiB whole-init worsens the plan).
- PPL runs prefill: decode-only sidecars (fp8_attn_proj, NVFP4 decode cache on GGUF) need greedy identity + long coherent generations. PPL alone misses degenerate low-PPL models: run `tools/analysis/degen_suite.py` on a server too (check-degeneration).
- Per-block attribution: `diagnostics.dump_hidden_dir` + `tools/analysis/layer_ab_diff.py` (added divergence per block).

## Gotchas

| Topic | Fact |
|---|---|
| Q8_0 blocks | 34 bytes, not 4-aligned: `memcpy()`, never `reinterpret_cast` |
| FP8 prefill | auto-disabled on sm_120 (cuBLAS `NOT_SUPPORTED` at non-aligned M); `attention.fp8_prefill=always` for experiments only. FP8 GDN-projection prefill REFUTED e2e (record `docs/plans/2026-08-31-fp8-ssm-prefill.md`): `out / row_scale` in FP16 overflows to inf on small-absmax rows; cuBLASLt `SCALE_OUTER_VEC_32F` applies `scale[n & ~1]` on sm_120/13.3 |
| Activation quantize | producer-fused since #1771/#1773 (`src/quant/nvfp4_pack.cuh`; skip = act-quant hint AND scratch tag); invariant: bit-identity vs `quantize_fp16_to_nvfp4_into` |
| MXFP4 GGUF | Qwen3.5-4B MXFP4 works (#935/#937); Qwen3.5-27B MXFP4 blocked (OOM on 32 GB) |
| MXFP4 on GDN hybrids | decode falls back MXFP4 -> FP16; the planner reserves that fallback (#935) or token-0 `!` garbage; keep the reserve |
| MoE expert leak | host-resident experts left unpromoted (`status 15` / garbage): check `src/model/weight_upload.cu` promotion first (#925) |
| VRAM ordering | weight caches (bounded by the model) are built BEFORE the KV pool takes the measured residual (`src/runtime/engine_kv_cache_init.cpp`, #926 corrected by #1106); a successful `cudaMalloc` at 0 MiB free proves nothing (WDDM spills, ~1530 vs ~237 GB/s, #1103) |
| KV block size | resolved in `init_resolve_kv_block_size_()` BEFORE `executor_->init()` (#1819); anything sized in `init_weights()` from a value `init_kv_cache()` resolves later is this trap |
| Dequant correctness | golden-locked: GGUF bit-exact vs spec; f16-class cross-path tolerance 1e-2 (measured ~4e-4); a move is a bug |
| Bad community quants | a degenerate model can be the file; verify with a llama.cpp control |
| Two NVFP4 layouts, RECIPROCAL tensor scale | Modelopt `.weight`/`.weight_scale_2` multiply (`hf_quant_config.json`); compressed-tensors `.weight_packed`/`.weight_global_scale` DIVIDE (`quantization_config` in `config.json`). Wrong way = every weight off by `absmax^2/36`, still loads (PPL 1.2e47 vs 31.05). `recipe.yaml` is not the declaration |
| Fused layers share one tensor scale | `q/k/v`, `gate/up`, GDN `in_proj_qkv`+`in_proj_z`, `in_proj_b`+`in_proj_a` (vLLM `packed_modules_mapping`); per-tensor scales dequantize siblings against the wrong scale (amax spread 3.7x). Refuted: `absmax/(6*448)` scaling (31.05, worse than `absmax/6`) |
| `imp-quantize` | BF16/FP16 and block-scaled FP8 (`weight_scale_inv`) SafeTensors -> NVFP4, EXPERIMENTAL (RTN ~+18-22% PPL vs BF16); `--calib <file>` from `imp-cli --calibrate` (AWQ; `--calib-groups BD` on wide-GQA models, full `ABCD` hurts at n_rep >= 5), `--keep-attn-gate`, `--dry-run` (size + VRAM budget via HTTP-range header read), `--format vllm` writes compressed-tensors (loads in vLLM 0.27.1; `kekzle/Qwen3.8-27B-NVFP4-vllm` runs in both). 2-D per-expert HF MoE supported (DeepSeek-V2-Lite); 3-D stacked experts, MLA latent projections, MoE router REFUSED. Sharded sources need a rebuilt `model.safetensors.index.json` or the resolver says "No .gguf file found". One-shot download + quantize: `scripts/stage-model.sh` |
| Micro-scale search | REFUTED (#1083): 30.10 -> 29.88 PPL for ~6x cost; the FP4 grid is the error, AWQ moves it |
| NVFP4 lm_head | `gemm.nvfp4_lm_head`, `_gdn` default on: +2.2% PPL for +8-16% decode, owner-accepted; `--lm-head` off = +0.99% PPL for -10.4% decode on the measured case |
