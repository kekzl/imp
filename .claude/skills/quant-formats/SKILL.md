---
name: quant-formats
description: Use when working on imp's quantization formats, loaders, or dequant paths — GGUF Q4_0…Q8_0/Q*_K/IQ4/MXFP4, NVFP4 two-level scaling, FP8 E4M3, StorageTier, decode cache, KV-cache dtypes, "which quant should I use", scale-factor layout, dequant kernel wiring. Do NOT use for writing/optimizing GEMM/GEMV kernels (sm120-cuda-expert) or measuring quant perf (benchmark-cuda).
---

# Quantization Formats & Pipelines — imp

**Sources of truth: `docs/quantization.md` (formats, choosing a quant) and `docs/internals/QUANT_PIPELINE.md` (files, GEMM-dispatch registry, boundary rules).** This skill carries only the agent-facing gotchas — read those docs for the full picture.

## The two worlds

| | GGUF | SafeTensors NVFP4 (prequant) |
|---|---|---|
| Formats | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, IQ4_NL/XS, MXFP4 | NVFP4: per-tensor AWQ scale (FP32) + per-16 FP8-E4M3 micro-scales |
| Decode | dp4a GEMV per qtype; **plus NVFP4 decode cache built at init** (bandwidth win) | native NVFP4 GEMV (prmt LUT) |
| Prefill | cuBLAS on dequanted source (full precision) | CUTLASS NVFP4 GEMM (sm_120 TC) |
| Priority | legacy/maintenance — esp. community MXFP4 quality bugs: don't sink time | **the strategic path** |

- GGUF→NVFP4 decode cache: weights converted at init; `nvfp4_beneficial` weights skip the FP16 cache. GDN/SSM projections are excluded (quality lock) and served by the `gemm.fp8_ssm_proj` decode sidecar instead (default on; covers native-F16 AND GGUF-Q8_0 sources since #962 — the old `nvfp4_ssm_proj` opt-in was removed 2026-07-11, bit-rotted). `gemm.nvfp4_attn_proj` stays opt-in.
- gpt-oss: native MXFP4 experts are converted to NVFP4 at load.

## StorageTier is the dispatch contract (`src/core/storage_tier.h`)

`Undefined` (FATAL if dispatched) · `FP32` · `FP16` · `FP8` (E4M3 + per-tensor scale) · `NVFP4` (two-level micro-scale, decode-GEMV path) · `CUTLASS_NVFP4` (block-scaled, CUTLASS sm_120 grouped-GEMM **fast path**) · `MXFP4` (CUTLASS FMHA path).

**Tier decisions have ONE source of truth since PR #621**: `plan_storage()` in `src/runtime/storage_planner.h` (StoragePlan + arch rules) decides every weight's tier at load; the caches it fills are RAII-owned by the executor. Don't add ad-hoc tier overrides downstream — extend the planner's rules.

**`NVFP4` ≠ `CUTLASS_NVFP4`**: a weight stuck on plain `NVFP4` falls through to the slow `gemm_nvfp4` dequant→cuBLAS fallback. For the fast path the scale factors need the CUTLASS SfAtom layout (set up in `src/exec/pre_dequant_*.cu`, Phase 3b). `convert_scales_sfatom` is a load-time artifact — not a runtime perf lever.

## MoE specifics

Per-expert NVFP4 tensors are copied into one contiguous `[ne, N, K_packed]` buffer per layer/projection at load (`cache_moe_native_nvfp4`) — this is what makes CUDA-Graph capture possible. Without it: per-step FP16 dequant + cuBLAS, 5–17× slower, no graphs.

## KV cache dtypes

`kv_cache.dtype = fp16 | fp8 | int8 | int4 | nvfp4` (CLI: `--kv-fp8` etc.). FP8 KV has a nondeterminism opt-in (`allow_nondeterministic_fp8`). Quant-KV accuracy envelopes are frozen in tests (TEST_AUDIT).

## Judging quantization quality (the measurement is easy to get wrong)

- **Use `tools/analysis/ppl_corpus_45k.txt` (13 536 tokens), never `ppl_corpus.txt` (199 tokens).** The short corpus does not just add noise, it *inverts conclusions*: the same imp-quantize pair reads +42%/+57% on it vs +25%/+19% on the real corpus, and appears to get worse with model size when it actually gets better. Add `--set runtime.deterministic_gemm=true` to both arms.
- **PPL alone is not enough** — it cannot see a degenerate-but-low-perplexity model. Run `tools/analysis/degen_suite.py` against a server on the quantized weights (41 checks: streaming, json_schema, tool calls, thinking). See `check-degeneration`.
- PPL measured via `--perplexity` runs the PREFILL path, so it does NOT see decode-only sidecars (fp8_attn_proj, NVFP4 decode cache on GGUF). Judge those by greedy-identity + coherent long generations instead.

## Gotchas

- **Q8_0 blocks are 34 bytes, NOT 4-aligned** — `memcpy()`, never `reinterpret_cast`.
- **FP8 prefill is disabled on sm_120** (cuBLAS `NOT_SUPPORTED` at non-aligned M) — don't build on it.
- **MXFP4 GGUF status (2026-07-09)**: Qwen3.5-4B MXFP4 **works** (server garbage fixed in PRs #935/#937); Qwen3.5-27B MXFP4 stays blocked (loads OOM on 32 GB, no GGUF source).
- **MXFP4-on-GDN-hybrids decode falls back MXFP4→FP16 — that fallback MUST be VRAM-budgeted** (PR #935): unbudgeted it silently failed to allocate and produced token-0 `!` garbage. The planner now reserves it at init and fails loud; don't remove the reserve.
- **MoE expert leak fingerprint** (PR #925): host-resident experts left unpromoted (raw INT8/FP4 handed to cuBLAS → `status 15` / garbage). For any MoE-expert weight bug, check `src/model/weight_upload.cu` promotion logic FIRST.
- **VRAM ordering** (PR #926, corrected by #1106): mandatory NVFP4 caches are reserved BEFORE workspaces/KV (`cudaMemGetInfo` lies after async frees — a balloon reservation holds the floor). Don't reorder allocations "for simplicity". **The reservation alone was not enough**: it was sized from an *estimate* of cache demand that ran ~1.6 GiB low, the caches took the difference back out of the reserve, and gpt-oss-20b-mxfp4 at server defaults ended at exactly 0 MiB free — where WSL2/WDDM spills into host memory and decode collapses (55 → 331–359 tok/s once fixed; that is #1103). The shipped rule is the *build* order: the weight caches, whose demand is bounded by the model, are built first, and the KV pool — the elastic tier — takes the **measured** residual rather than a predicted one (`src/runtime/engine_kv_cache_init.cpp`). Corollary: a successful allocation at 0 MiB free proves nothing, since WDDM oversubscribes into host memory and still returns `cudaSuccess`; bandwidth is the discriminator (~1530 GB/s resident vs ~237 GB/s spilled).
- **Dequant correctness is golden-locked**: GGUF dequant is bit-exact vs spec; f16-class cross-path tolerance is strict 1e-2 (measured ~4e-4). If your change moves these, it's a bug, not noise.
- Quantizing new checkpoints normally happens OUTSIDE imp (NVIDIA ModelOpt / llm-compressor). Bad community quants exist — a degenerate model can be the file, not the engine (verify with llama.cpp control where possible).
- **Two NVFP4 layouts exist and their tensor scale is RECIPROCAL.** Modelopt: `.weight` / `.weight_scale_2`, multiply, declared in `hf_quant_config.json`. compressed-tensors: `.weight_packed` / `.weight_global_scale`, **divide**, declared in `quantization_config` inside `config.json`. Reading one as the other scales every weight by `absmax²/36` and the checkpoint still loads and generates (measured PPL 1.2e47 vs 31.05). `recipe.yaml` is llm-compressor's record of the run, NOT the checkpoint's declaration — detecting the format from it alone missed every export published without one.
- **Fused layers share one tensor scale** (`q/k/v`, `gate/up`, GDN `in_proj_qkv`+`in_proj_z`, `in_proj_b`+`in_proj_a` — vLLM's `packed_modules_mapping`). A merged linear keeps ONE scale, so per-tensor scales leave the other members dequantized against the wrong one; amax spread inside a group reaches 3.7x. Also the better quantization (0.6B: 30.40 → 29.42). **Refuted, do not re-attempt:** scaling by `absmax/(6*448)` so micro-scales fill FP8's range, which is what published exports do — measured 31.05, i.e. worse than `absmax/6`.
- **`imp-quantize` (2026-07-26, #1081) converts dense BF16/FP16 SafeTensors → NVFP4 in-tree, and is EXPERIMENTAL**: uncalibrated round-to-nearest, +19–25% PPL vs BF16. Use it to get a model onto the NVFP4 path for evaluation or perf work, never to produce weights anyone relies on. `--format vllm` writes the compressed-tensors layout instead (verified loading in vLLM 0.27.1); the default stays `modelopt`. Dense only — 3-D MoE expert stacks are reported and left unquantized. Sharded sources need a REBUILT `model.safetensors.index.json` (one weight becomes three tensors); without it the resolver reports the misleading "No .gguf file found in directory".
- **REFUTED, do not re-attempt (#1083): searching micro-scale candidates instead of `absmax`.** Measured 30.10 → 29.88 PPL (0.7%) for ~6× quantization cost. The micro-block is only 16 values, where absmax is already near-optimal; the dominant error is the FP4 grid itself (8 magnitudes), which no scale improves. The open lever is *moving* the error — AWQ (protect high-activation channels) or GPTQ (compensate in not-yet-quantized columns), both needing infrastructure imp lacks.
- NVFP4 lm_head quantization (`gemm.nvfp4_lm_head`, `_gdn` default ON) trades +2.2% PPL for +8–16% decode — owner-accepted; don't "fix" the PPL delta by reverting silently.
