---
name: quant-formats
description: Use when working on imp's quantization formats, loaders, or dequant paths — GGUF Q4_0…Q8_0/Q*_K/IQ4/MXFP4, NVFP4 two-level scaling, FP8 E4M3, StorageTier, decode cache, KV-cache dtypes, "which quant should I use", scale-factor layout, dequant kernel wiring. Do NOT use for writing/optimizing GEMM/GEMV kernels (sm120-cuda-expert) or measuring quant perf (benchmark-cuda).
---

# Quantization Formats & Pipelines — imp

**Sources of truth: `docs/quantization.md` (formats, choosing a quant) and `docs/quant-pipeline.md` (files, GEMM-dispatch registry, boundary rules).** This skill carries only the agent-facing gotchas — read those docs for the full picture.

## The two worlds

| | GGUF | SafeTensors NVFP4 (prequant) |
|---|---|---|
| Formats | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, IQ4_NL/XS, MXFP4 | NVFP4: per-tensor AWQ scale (FP32) + per-16 FP8-E4M3 micro-scales |
| Decode | dp4a GEMV per qtype; **plus NVFP4 decode cache built at init** (bandwidth win) | native NVFP4 GEMV (prmt LUT) |
| Prefill | cuBLAS on dequanted source (full precision) | CUTLASS NVFP4 GEMM (sm_120 TC) |
| Priority | legacy/maintenance — esp. community MXFP4 quality bugs: don't sink time | **the strategic path** |

- GGUF→NVFP4 decode cache: weights converted at init; `nvfp4_beneficial` weights skip the FP16 cache. Opt-in extensions: `gemm.nvfp4_ssm_proj` (+53% GGUF hybrid), `gemm.nvfp4_attn_proj`.
- gpt-oss: native MXFP4 experts are converted to NVFP4 at load.

## StorageTier is the dispatch contract (`src/core/storage_tier.h`)

`Undefined` (FATAL if dispatched) · `FP32` · `FP16` · `FP8` (E4M3 + per-tensor scale) · `NVFP4` (two-level micro-scale, decode-GEMV path) · `CUTLASS_NVFP4` (block-scaled, CUTLASS sm_120 grouped-GEMM **fast path**) · `MXFP4` (CUTLASS FMHA path).

**Tier decisions have ONE source of truth since PR #621**: `plan_storage()` in `src/runtime/storage_planner.h` (StoragePlan + arch rules) decides every weight's tier at load; the caches it fills are RAII-owned by the executor. Don't add ad-hoc tier overrides downstream — extend the planner's rules.

**`NVFP4` ≠ `CUTLASS_NVFP4`**: a weight stuck on plain `NVFP4` falls through to the slow `gemm_nvfp4` dequant→cuBLAS fallback. For the fast path the scale factors need the CUTLASS SfAtom layout (set up in `src/exec/pre_dequant_*.cu`, Phase 3b). `convert_scales_sfatom` is a load-time artifact — not a runtime perf lever.

## MoE specifics

Per-expert NVFP4 tensors are copied into one contiguous `[ne, N, K_packed]` buffer per layer/projection at load (`cache_moe_native_nvfp4`) — this is what makes CUDA-Graph capture possible. Without it: per-step FP16 dequant + cuBLAS, 5–17× slower, no graphs.

## KV cache dtypes

`kv_cache.dtype = fp16 | fp8 | int8 | int4 | nvfp4` (CLI: `--kv-fp8` etc.). FP8 KV has a nondeterminism opt-in (`allow_nondeterministic_fp8`). Quant-KV accuracy envelopes are frozen in tests (TEST_AUDIT).

## Gotchas

- **Q8_0 blocks are 34 bytes, NOT 4-aligned** — `memcpy()`, never `reinterpret_cast`.
- **FP8 prefill is disabled on sm_120** (cuBLAS `NOT_SUPPORTED` at non-aligned M) — don't build on it.
- **MXFP4 GGUF status (2026-07-09)**: Qwen3.5-4B MXFP4 **works** (server garbage fixed in PRs #935/#937); Qwen3.5-27B MXFP4 stays blocked (loads OOM on 32 GB, no GGUF source).
- **MXFP4-on-GDN-hybrids decode falls back MXFP4→FP16 — that fallback MUST be VRAM-budgeted** (PR #935): unbudgeted it silently failed to allocate and produced token-0 `!` garbage. The planner now reserves it at init and fails loud; don't remove the reserve.
- **MoE expert leak fingerprint** (PR #925): host-resident experts left unpromoted (raw INT8/FP4 handed to cuBLAS → `status 15` / garbage). For any MoE-expert weight bug, check `src/exec/weight_upload.cu` promotion logic FIRST.
- **VRAM ordering** (PR #926): mandatory NVFP4 caches are reserved BEFORE workspaces/KV (`cudaMemGetInfo` lies after async frees — a balloon reservation holds the floor). Don't reorder allocations "for simplicity".
- **Dequant correctness is golden-locked**: GGUF dequant is bit-exact vs spec; f16-class cross-path tolerance is strict 1e-2 (measured ~4e-4). If your change moves these, it's a bug, not noise.
- Quantizing new checkpoints happens OUTSIDE imp (NVIDIA ModelOpt / llm-compressor); imp only loads. Bad community quants exist — a degenerate model can be the file, not the engine (verify with llama.cpp control where possible).
- NVFP4 lm_head quantization (`gemm.nvfp4_lm_head`, `_gdn` default ON) trades +2.2% PPL for +8–16% decode — owner-accepted; don't "fix" the PPL delta by reverting silently.
